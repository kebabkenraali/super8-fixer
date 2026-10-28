//! Repair pipeline: analysis -> classifier -> decisions -> seam refinement -> user edits -> report
//! -> rendering.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::analysis::{self, AH, Run};
use crate::assemble::{self, Fill, FrameJob, NEIGHBOURS, Size};
use crate::enhance::{self, Colour, EnhanceSettings, Enhancer, Levels};
use crate::model::{self, Classifier};
use crate::video::{Reader, VideoInfo, Writer, parse_range};

/// A frame inside a repaired run is left as it is when its calibrated classifier probability is
/// below this (0.5 is the run threshold).
const FRAME_NORMAL: f32 = 0.1;

#[derive(Clone, Debug)]
pub struct RepairSettings {
    /// confidence needed to repair a run (when the classifier does not decide)
    pub threshold: f32,
    pub use_model: bool,
    pub model_threshold: f32,
    /// time ranges "03:37-03:44": never repair / always repair
    pub skip: Vec<String>,
    pub force: Vec<String>,
    /// rebuild the frame from neighbours (otherwise only roll)
    pub assemble: bool,
    /// automatic seam refinement against neighbouring frames
    pub refine: bool,
    /// dust removal and colour restoration
    pub enhance: EnhanceSettings,
    pub crf: u32,
    pub preset: String,
    pub gpu: bool,
}

impl Default for RepairSettings {
    fn default() -> Self {
        RepairSettings {
            threshold: 0.45,
            use_model: true,
            model_threshold: 0.5,
            skip: vec![],
            force: vec![],
            assemble: true,
            refine: false,
            enhance: EnhanceSettings::default(),
            crf: 16,
            preset: "medium".into(),
            gpu: true,
        }
    }
}

/// The user's changes to one run. Saved next to the video (<video>.s8edits.json) so that they
/// survive a new analysis.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct RunEdit {
    pub start: usize,
    pub end: usize,
    /// None = automatic decision
    pub repair: Option<bool>,
    /// seam position correction for the whole run (video rows, + moves the picture down)
    pub offset: i64,
    /// additional correction for single frames (video rows)
    pub frame_offsets: BTreeMap<usize, i64>,
    /// where the missing part of the frame is taken from
    pub fill: Fill,
    /// alignment correction of the filled part (video pixels)
    pub nudge_y: i64,
    pub nudge_x: i64,
    /// ignore the automatic seam refinement in this run
    pub no_refine: bool,
    /// repair every frame of the run, also those the classifier judged normal
    pub no_gate: bool,
}

impl RunEdit {
    fn for_run(r: &Run) -> RunEdit {
        RunEdit { start: r.start, end: r.end, ..Default::default() }
    }
    pub fn is_default(&self) -> bool {
        let d = RunEdit { start: self.start, end: self.end, ..Default::default() };
        *self == d
    }
}

#[derive(Serialize, Deserialize, Default)]
struct EditFile {
    app: String,
    runs: Vec<RunEdit>,
}

/// Analysis result; runs and edits can be changed before rendering.
#[derive(Clone)]
pub struct Analysis {
    pub info: VideoInfo,
    pub path: Vec<i32>,
    pub runs: Vec<Run>,
    /// normal frame-line position (analysis rows) for every run
    pub reference: Vec<i32>,
    /// automatic seam refinement per frame (video rows)
    pub auto_adjust: Vec<i64>,
    pub edits: Vec<RunEdit>,
    /// repair decision of the analysis itself, before the user's changes
    pub auto_repair: Vec<bool>,
    /// the automatic seam refinement was run
    pub refined: bool,
    /// frames the classifier judged clearly normal: left as they are in runs ticked by the user
    pub keep: Vec<bool>,
}

impl Analysis {
    /// Tracked shift (video rows) of frame `k` in run `ri`, before refinement and edits.
    pub fn base_shift(&self, ri: usize, k: usize) -> i64 {
        let s = (self.reference[ri] - self.path[k]).rem_euclid(AH as i32);
        analysis::rows_to_px(s, self.info.h)
    }

    /// Final shift (video rows, 0..h) of frame `k` in run `ri`.
    pub fn shift(&self, ri: usize, k: usize) -> usize {
        let e = &self.edits[ri];
        let mut s = self.base_shift(ri, k) + e.offset + e.frame_offsets.get(&k).copied().unwrap_or(0);
        if !e.no_refine {
            s += self.auto_adjust[k];
        }
        s.rem_euclid(self.info.h as i64) as usize
    }

    /// Sets the frame offsets of frames `from..to` of run `ri` so that their final shift is `target`.
    pub fn set_shift(&mut self, ri: usize, from: usize, to: usize, target: i64) {
        let h = self.info.h as i64;
        for k in from..to {
            self.edits[ri].frame_offsets.remove(&k);
            let now = self.shift(ri, k) as i64;
            let mut d = (target - now).rem_euclid(h);
            if d > h / 2 {
                d -= h;
            }
            if d != 0 {
                self.edits[ri].frame_offsets.insert(k, d);
            }
        }
    }

    /// Shift that puts a frame line seen at `row` of frame `k` where the normal frames have it.
    pub fn shift_for_line(&self, ri: usize, row: usize) -> i64 {
        let h = self.info.h as i64;
        // the tracked frame line of a frame maps to the reference position: shift = reference - line
        (analysis::rows_to_px(self.reference[ri], self.info.h) - row as i64).rem_euclid(h)
    }

    /// What to do with every frame under the current decisions and edits.
    pub fn jobs(&self, assemble: bool) -> Vec<FrameJob> {
        let mut j = vec![FrameJob::default(); self.path.len()];
        for (ri, r) in self.runs.iter().enumerate() {
            if !r.repair {
                continue;
            }
            let e = &self.edits[ri];
            for k in r.start..r.end {
                // in a run the user ticked against the analysis, a frame the classifier judges clearly
                // normal is not cut (unless the user set its seam)
                if !self.auto_repair[ri] && self.keep[k] && !e.frame_offsets.contains_key(&k) && !e.no_gate {
                    continue;
                }
                j[k] = FrameJob {
                    shift: self.shift(ri, k),
                    fill: if assemble { e.fill } else { Fill::RollOnly },
                    nudge: (e.nudge_y, e.nudge_x),
                };
            }
        }
        j
    }

    pub fn repaired_count(&self) -> (usize, usize) {
        let r: Vec<&Run> = self.runs.iter().filter(|r| r.repair).collect();
        (r.len(), r.iter().map(|r| r.end - r.start).sum())
    }

    /// Sets the user's repair decision of run `ri`.
    pub fn set_repair(&mut self, ri: usize, repair: bool) {
        self.runs[ri].repair = repair;
        self.runs[ri].manual = repair != self.auto_repair[ri];
        self.edits[ri].repair = if repair == self.auto_repair[ri] { None } else { Some(repair) };
    }

    pub fn edit_file(&self) -> PathBuf {
        Path::new(&self.info.path).with_extension("s8edits.json")
    }

    /// Saves the non-default edits next to the video (removes the file when there are none).
    pub fn save_edits(&self) -> Result<()> {
        let runs: Vec<RunEdit> = self.edits.iter().filter(|e| !e.is_default()).cloned().collect();
        let p = self.edit_file();
        if runs.is_empty() {
            let _ = std::fs::remove_file(&p);
            return Ok(());
        }
        let f = EditFile { app: format!("Super8 Fixer {}", env!("CARGO_PKG_VERSION")), runs };
        std::fs::write(&p, serde_json::to_string_pretty(&f)?)?;
        Ok(())
    }

    /// Applies saved edits to the runs they overlap most (at least half of either run).
    fn load_edits(&mut self) {
        let Ok(t) = std::fs::read_to_string(self.edit_file()) else { return };
        let Ok(f) = serde_json::from_str::<EditFile>(&t) else {
            crate::log("The edit file could not be read; it is ignored.");
            return;
        };
        let mut used = 0;
        for saved in f.runs {
            let best = self
                .runs
                .iter()
                .enumerate()
                .map(|(i, r)| (i, saved.end.min(r.end).saturating_sub(saved.start.max(r.start))))
                .max_by_key(|&(_, o)| o);
            let Some((i, o)) = best else { continue };
            let r = &self.runs[i];
            if o * 2 < (r.end - r.start).min(saved.end - saved.start).max(1) {
                continue;
            }
            let mut e = saved.clone();
            e.start = r.start;
            e.end = r.end;
            e.frame_offsets.retain(|k, _| (r.start..r.end).contains(k));
            if let Some(rep) = e.repair {
                self.runs[i].repair = rep;
                self.runs[i].manual = rep != self.auto_repair[i];
            }
            self.edits[i] = e;
            used += 1;
        }
        if used > 0 {
            crate::log(&format!("Applied the saved edits of {used} runs: {}", self.edit_file().display()));
        }
    }
}

pub fn stem(video: &str) -> PathBuf {
    let p = Path::new(video);
    p.with_file_name(p.file_stem().unwrap_or_default())
}

pub fn with_suffix(stem: &Path, suffix: &str) -> String {
    format!("{}{suffix}", stem.display())
}

/// Reads downscaled frames (MH x MW RGB) at the given indices in one pass.
pub fn read_small(v: &VideoInfo, indices: &BTreeSet<usize>) -> Result<HashMap<usize, Vec<u8>>> {
    let mut out = HashMap::new();
    let Some(&last) = indices.iter().next_back() else {
        return Ok(out);
    };
    let filter = format!("scale={}:{}:flags=area", model::MW, model::MH);
    let mut r = Reader::new(&v.path, Some(&filter), "rgb24", model::MW * model::MH * 3, None, Some(last + 1))?;
    let mut i = 0;
    while let Some(b) = r.next_frame() {
        if indices.contains(&i) {
            out.insert(i, b);
        }
        i += 1;
        if i % 2000 == 0 {
            crate::check_stop()?;
            crate::progress(&format!("Frames for the classifier: {i} / {}", last + 1), i, last + 1);
        }
        if i > last {
            break;
        }
    }
    Ok(out)
}

pub fn analyze(video: &str, a: &RepairSettings) -> Result<Analysis> {
    let v = VideoInfo::read(video)?;
    crate::log(&format!(
        "Video: {}x{}, {:.3} fps, {:.1} min, audio: {}",
        v.w,
        v.h,
        v.fps,
        v.duration / 60.0,
        if v.audio { "yes" } else { "no" }
    ));
    let s = analysis::analyze(&v)?;
    crate::log("Tracking the frame line...");
    let path = analysis::track(&s);
    let ranges = |x: &[String]| -> Result<Vec<(usize, usize)>> { x.iter().map(|r| parse_range(r, v.fps)).collect() };
    let (skip, force) = (ranges(&a.skip)?, ranges(&a.force)?);
    let none = HashMap::new();
    let preliminary = analysis::decide(&s, &path, a.threshold, &skip, &force, &none, a.model_threshold);
    let mut model_p = HashMap::new();
    let mut frame_p: HashMap<usize, f32> = HashMap::new();
    if a.use_model {
        if !model::model_path(model::MODEL_FILE).is_file() && !model::model_path(model::PERSONAL_FILE).is_file() {
            crate::log(&format!("(Classifier models\\{} not found: using the confidence value only)", model::MODEL_FILE));
        } else {
            let samples = analysis::model_samples(&path, &preliminary.runs);
            let mut need: BTreeSet<usize> = samples.iter().flat_map(|(_, v)| v.iter().copied()).collect();
            // every frame of a run whose frame line is inside the picture is also judged on its own
            let zone = |p: i32| (analysis::MODEL_ZONE * AH as f64) as i32 <= p && p <= ((1.0 - analysis::MODEL_ZONE) * AH as f64) as i32;
            let single: Vec<usize> = preliminary.runs.iter().flat_map(|r| (r.start..r.end).filter(|&k| zone(path[k]))).collect();
            need.extend(single.iter().copied());
            if !need.is_empty() {
                let mut c = Classifier::load(a.gpu)?;
                let frames = read_small(&v, &need)?;
                let idx: Vec<usize> = single.iter().copied().filter(|k| frames.contains_key(k)).collect();
                for chunk in idx.chunks(512) {
                    let imgs: Vec<Vec<u8>> = chunk.iter().map(|k| frames[k].clone()).collect();
                    for (&k, p) in chunk.iter().zip(c.predict(&imgs)?) {
                        frame_p.insert(k, c.calibrate(p));
                    }
                }
                for (key, idx) in samples {
                    let k: Vec<Vec<u8>> = idx.iter().filter_map(|i| frames.get(i).cloned()).collect();
                    if k.is_empty() {
                        continue;
                    }
                    let p = c.predict(&k)?;
                    let mean = p.iter().map(|&x| x as f64).sum::<f64>() as f32 / p.len() as f32;
                    model_p.insert(key, c.calibrate(mean));
                }
            }
        }
    }
    let d = analysis::decide(&s, &path, a.threshold, &skip, &force, &model_p, a.model_threshold);
    let n = path.len();
    let auto_repair = d.runs.iter().map(|r| r.repair).collect();
    let edits = d.runs.iter().map(RunEdit::for_run).collect();
    let mut keep = vec![false; n];
    for (&k, &p) in &frame_p {
        keep[k] = p < FRAME_NORMAL;
    }
    let mut an =
        Analysis { info: v, path, runs: d.runs, reference: d.reference, auto_adjust: vec![0; n], edits, auto_repair, refined: a.refine, keep };
    if a.refine && !an.runs.is_empty() {
        crate::log("Refining the seam positions against neighbouring frames...");
        let jobs: Vec<(usize, usize, Vec<usize>)> = an
            .runs
            .iter()
            .enumerate()
            .map(|(ri, r)| (r.start, r.end, (r.start..r.end).map(|k| an.base_shift(ri, k).rem_euclid(an.info.h as i64) as usize).collect()))
            .collect();
        let adj = crate::refine::refine(&an.info, &jobs)?;
        let mut changed = 0;
        for (r, a) in an.runs.iter().zip(adj) {
            for (k, d) in (r.start..r.end).zip(a) {
                an.auto_adjust[k] = d;
                changed += usize::from(d != 0);
            }
        }
        crate::log(&format!("Seam refinement changed {changed} frames."));
        // development aid: SUPER8FIXER_DUMP=1 writes the tracked and refined shift of every run frame
        if std::env::var_os("SUPER8FIXER_DUMP").is_some() {
            let mut t = String::from("frame,base,adjust,keep\n");
            for (ri, r) in an.runs.iter().enumerate() {
                for k in r.start..r.end {
                    t += &format!("{k},{},{},{}\n", an.base_shift(ri, k).rem_euclid(an.info.h as i64), an.auto_adjust[k], u8::from(an.keep[k]));
                }
            }
            let _ = std::fs::write(with_suffix(&stem(video), "_shifts.csv"), t);
        }
    }
    an.load_edits();
    let (nr, nf) = an.repaired_count();
    let rep = with_suffix(&stem(video), "_repair_report.txt");
    analysis::report(Path::new(&rep), &an.runs, an.info.fps)?;
    crate::log(&format!("Found {} runs where the frame line is away from the edge; repairing {nr} runs ({nf} frames). Report: {rep}", an.runs.len()));
    Ok(an)
}

pub struct Preview {
    pub original: Vec<u8>,
    pub repaired: Vec<u8>,
    /// row of the frame line in the original, and of the seam in the repaired frame (None = not repaired)
    pub seam: Option<(usize, usize)>,
}

type Originals = std::collections::VecDeque<(usize, Vec<u8>)>;

/// Repairs and enhances the frames `start..start+count` (to the end of the video when `count` is
/// None) and calls `sink(original, result)` for every frame in order. `levels`: colour levels per
/// frame starting at frame `levels.1`, or one entry for all frames.
fn process(
    an: &Analysis,
    a: &RepairSettings,
    start: usize,
    count: Option<usize>,
    levels: (Vec<Levels>, usize),
    progress: Option<&str>,
    mut sink: impl FnMut(&[u8], Vec<u8>) -> Result<()>,
) -> Result<usize> {
    let v = &an.info;
    let jobs = an.jobs(a.assemble);
    let k = Size { w: v.w, h: v.h };
    let size = v.w * v.h * 3;
    let reach = if a.enhance.dust { enhance::REACH } else { 0 };
    // the dust removal needs `reach` repaired frames on both sides, the repair NEIGHBOURS more
    let lead = start.saturating_sub(reach);
    let end = count.map(|c| start + c);
    let read_from = lead.saturating_sub(NEIGHBOURS);
    let max_frames = end.map(|e| e + reach + NEIGHBOURS - read_from);
    let mut r = Reader::new(&v.path, None, "rgb24", size, (read_from > 0).then(|| read_from as f64 / v.fps), max_frames)?;
    let mut enh = a.enhance.active().then(|| Enhancer::new(a.enhance, v.w, v.h, levels.0, levels.1).starting_at(lead));
    let mut originals: Originals = Default::default();
    let mut done = 0usize;
    let mut emit = |outs: Vec<Vec<u8>>, originals: &mut Originals, done: &mut usize| -> Result<()> {
        for o in outs {
            let (i, orig) = originals.pop_front().unwrap();
            if i >= start && end.is_none_or(|e| i < e) {
                sink(&orig, o)?;
                *done += 1;
            }
        }
        Ok(())
    };
    const BATCH: usize = 24;
    // window[0] = frame `pos - NEIGHBOURS`
    let mut window: Vec<Option<Vec<u8>>> = vec![None; NEIGHBOURS - (lead - read_from)];
    let mut pos = lead;
    let mut ended = false;
    loop {
        crate::check_stop()?;
        while !ended && window.len() < NEIGHBOURS + BATCH + NEIGHBOURS {
            match r.next_frame() {
                Some(b) => window.push(Some(b)),
                None => ended = true,
            }
        }
        let available = window.len().saturating_sub(NEIGHBOURS);
        let n = if ended { available } else { available.saturating_sub(NEIGHBOURS).min(BATCH) };
        if n == 0 {
            break;
        }
        let repaired: Vec<Vec<u8>> = if jobs[pos.min(jobs.len())..(pos + n).min(jobs.len())].iter().all(|j| j.shift == 0) {
            (0..n).map(|d| window[NEIGHBOURS + d].clone().unwrap()).collect()
        } else {
            // at the end of the video there are no frames in the neighbour slots
            while window.len() < NEIGHBOURS + n + NEIGHBOURS {
                window.push(None);
            }
            assemble::repair_batch(&window, pos, n, k, &jobs)
        };
        for (d, f) in repaired.into_iter().enumerate() {
            originals.push_back((pos + d, window[NEIGHBOURS + d].clone().unwrap()));
            let outs = match &mut enh {
                Some(e) => e.push(f),
                None => vec![f],
            };
            emit(outs, &mut originals, &mut done)?;
        }
        pos += n;
        if let Some(p) = progress {
            let total = count.unwrap_or(v.n);
            crate::progress(&format!("{p}: {done} / {total} frames"), done, total.max(done));
        }
        if ended {
            break;
        }
        window.drain(..n);
    }
    if let Some(e) = &mut enh {
        let outs = e.finish();
        emit(outs, &mut originals, &mut done)?;
        if a.enhance.dust && progress.is_some() {
            crate::log(&format!("Dust and dirt: {} spots removed.", e.spots));
        }
    }
    Ok(done)
}

/// One repaired and enhanced frame for the preview, at full size.
pub fn preview(an: &Analysis, i: usize, a: &RepairSettings) -> Result<Preview> {
    let v = &an.info;
    let levels = if a.enhance.colour != Colour::Off { local_levels(v, i)? } else { vec![] };
    let mut res = None;
    process(an, a, i, Some(1), (levels, 0), None, |o, f| {
        res = Some((o.to_vec(), f));
        Ok(())
    })?;
    let (original, repaired) = res.ok_or_else(|| anyhow::anyhow!("frame {i} could not be read"))?;
    let s = an.jobs(a.assemble).get(i).map(|j| j.shift).unwrap_or(0);
    let seam = (s != 0).then(|| ((v.h - s) % v.h, s));
    Ok(Preview { original, repaired, seam })
}

/// Colour levels from a few frames around `i` (one entry, used for all frames).
fn local_levels(v: &VideoInfo, i: usize) -> Result<Vec<Levels>> {
    let a = i.saturating_sub(6);
    let mut r = Reader::new(&v.path, None, "rgb24", v.w * v.h * 3, Some(a as f64 / v.fps), Some(13))?;
    let mut f = Vec::new();
    while let Some(b) = r.next_frame() {
        f.push(b);
    }
    let refs: Vec<&[u8]> = f.iter().step_by(3).map(|x| &x[..]).collect();
    Ok(if refs.is_empty() { vec![] } else { vec![enhance::levels_of(&refs, v.w, v.h)] })
}

/// Before/after page with the middle frame of every repaired run.
pub fn before_after(an: &Analysis, path: &str, a: &RepairSettings) -> Result<()> {
    let v = &an.info;
    let rows: Vec<&Run> = an.runs.iter().filter(|r| r.repair).collect();
    if rows.is_empty() {
        return Ok(());
    }
    let (tw, th) = (360u32, (360 * v.h / v.w) as u32);
    let mut page = image::RgbImage::from_pixel(tw * 2 + 10, (th + 4) * rows.len() as u32, image::Rgb([40, 40, 40]));
    for (n, r) in rows.iter().enumerate() {
        crate::check_stop()?;
        let i = (r.start + r.end) / 2;
        let p = preview(an, i, a)?;
        for (x0, img) in [(0u32, p.original), (tw + 10, p.repaired)] {
            let im = image::RgbImage::from_raw(v.w as u32, v.h as u32, img).unwrap();
            let mut small = image::imageops::resize(&im, tw, th, image::imageops::FilterType::Triangle);
            crate::font::draw(&mut small, 4, 4, &format!("{} #{i}", crate::video::timestamp(i, v.fps)), [255, 255, 0]);
            image::imageops::replace(&mut page, &small, x0 as i64, (n as u32 * (th + 4)) as i64);
        }
        crate::progress(&format!("Before/after images: {} / {}", n + 1, rows.len()), n + 1, rows.len());
    }
    page.save(path)?;
    Ok(())
}

/// Renders the repaired (and enhanced) video.
pub fn render(an: &Analysis, a: &RepairSettings, out: &str) -> Result<usize> {
    let v = &an.info;
    let levels = if a.enhance.colour != Colour::Off { enhance::colour_levels(v, 0, None)? } else { vec![] };
    let mut w = Writer::new(out, v, a.crf, &a.preset, v.audio.then_some(v.path.as_str()))?;
    let n = process(an, a, 0, None, (levels, 0), Some("Rendering"), |_, f| w.write(&f))?;
    w.finish()?;
    crate::log(&format!("Done: {out} ({n} frames)"));
    Ok(n)
}

/// A side-by-side clip of `seconds` around frame `centre`: the original on the left, the result on
/// the right. Written next to the video as VIDEO_compare_MMmSSs.mp4; returns the path.
pub fn compare_clip(an: &Analysis, a: &RepairSettings, centre: usize, seconds: f64) -> Result<String> {
    let v = &an.info;
    let len = ((seconds * v.fps).round() as usize).clamp(1, v.n.max(1));
    let start = centre.saturating_sub(len / 2).min(v.n.saturating_sub(len));
    let secs = (centre as f64 / v.fps) as u64;
    let out = with_suffix(&stem(&v.path), &format!("_compare_{:02}m{:02}s.mp4", secs / 60, secs % 60));
    let levels = if a.enhance.colour != Colour::Off { enhance::colour_levels(v, start, Some(len))? } else { vec![] };
    let mut sv = v.clone();
    sv.w = v.w * 2;
    let mut w = Writer::new(&out, &sv, a.crf.max(18), "fast", None)?;
    let row = v.w * 3;
    let scale = (v.h / 180).max(2) as u32;
    let n = process(an, a, start, Some(len), (levels, start), Some("Side-by-side clip"), |o, f| {
        let mut both = vec![0u8; sv.w * v.h * 3];
        for y in 0..v.h {
            both[y * 2 * row..y * 2 * row + row].copy_from_slice(&o[y * row..(y + 1) * row]);
            both[y * 2 * row + row..(y + 1) * 2 * row].copy_from_slice(&f[y * row..(y + 1) * row]);
        }
        let mut img = image::RgbImage::from_raw(sv.w as u32, v.h as u32, both).unwrap();
        crate::font::label(&mut img, 8 * scale, 8 * scale, "ORIGINAL", scale);
        crate::font::label(&mut img, v.w as u32 + 8 * scale, 8 * scale, "REPAIRED", scale);
        w.write(img.as_raw())
    })?;
    w.finish()?;
    crate::log(&format!("Side-by-side clip: {out} ({n} frames)"));
    Ok(out)
}
