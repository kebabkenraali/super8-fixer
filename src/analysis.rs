//! Frame-line detection: per-row features for every frame, Viterbi tracking over time, and the
//! decision which runs are slips and how much each frame is shifted.

use std::path::Path;

use anyhow::Result;
use rayon::prelude::*;

use crate::video::{Reader, VideoInfo};

pub const AH: usize = 540; // analysis height (rows)
pub const AW: usize = 64; // analysis width
pub const EDGE: i32 = 20; // a frame line this close to the edge = normal frame
const MERGE_GAP: usize = 3; // a normal gap this short between two runs belongs to the run
pub const MODEL_ZONE: f64 = 0.05; // the classifier decides runs whose frame line is at 5-95 % height

/// Mean over the window [y-r, y+r] per column, edges replicated.
fn box_mean(a: &[f32], h: usize, w: usize, r: usize) -> Vec<f32> {
    let mut c = vec![0f64; (h + 2 * r + 2) * w];
    // cumulative sum of the edge-replicated array (r+1 rows above, r below)
    for i in 0..h + 2 * r + 1 {
        let y = (i as i64 - r as i64 - 1).clamp(0, h as i64 - 1) as usize;
        for x in 0..w {
            c[(i + 1) * w + x] = c[i * w + x] + a[y * w + x] as f64;
        }
    }
    let mut o = vec![0f32; h * w];
    let n = (2 * r + 1) as f64;
    for y in 0..h {
        for x in 0..w {
            o[y * w + x] = ((c[(y + 2 * r + 2) * w + x] - c[(y + 1) * w + x]) / n) as f32;
        }
    }
    o
}

/// f32 -> nearest float16 value (the reference implementation stored features as float16).
fn round_f16(x: f32) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let b = x.to_bits();
    let r = b + 0x0FFF + ((b >> 13) & 1);
    f32::from_bits(r & !0x1FFF)
}

/// Per-row likelihood that the row holds a frame line (0..~1.2). `img`: AH x AW gray.
pub fn frame_features(img: &[u8]) -> Vec<f32> {
    let (h, w) = (AH, AW);
    let i: Vec<f32> = img.iter().map(|&v| v as f32).collect();
    // a straight seam across the full width
    let mut g = vec![0f32; h * w];
    for y in 2..h - 2 {
        for x in 0..w {
            g[y * w + x] = (i[(y + 2) * w + x] - i[(y - 2) * w + x]).abs();
        }
    }
    let b12 = box_mean(&g, h, w, 12);
    let b3 = box_mean(&g, h, w, 3);
    let mut seam = vec![0f32; h];
    for y in 0..h {
        let mut n = 0;
        for x in 0..w {
            let level = (b12[y * w + x] * 25.0 - b3[y * w + x] * 7.0) / 18.0;
            if g[y * w + x] > 2.0 * level + 6.0 {
                n += 1;
            }
        }
        seam[y] = n as f32 / w as f32;
    }
    let seam: Vec<f32> = (0..h).map(|y| seam[y].max(seam[(y + h - 1) % h])).collect();
    // a dark horizontal band
    let r: Vec<f32> = (0..h).map(|y| i[y * w..(y + 1) * w].iter().map(|&v| v as f64).sum::<f64>() as f32 / w as f32).collect();
    let rb = box_mean(&r, h, 1, 3);
    let r6 = box_mean(&r, h, 1, 6);
    (0..h)
        .map(|y| {
            let above = r6[(y + h - 10) % h];
            let below = r6[(y + 10) % h];
            let dark = if y < 10 {
                (below - rb[y]) / (below + 20.0)
            } else if y >= h - 10 {
                (above - rb[y]) / (above + 20.0)
            } else {
                let refv = above.min(below);
                (refv - rb[y]) / (refv + 20.0)
            };
            round_f16(seam[y].max(dark.clamp(0.0, 1.0) * 1.2))
        })
        .collect()
}

/// Features for the whole video (N x AH). Cached in <video>.s8analysis.bin.
pub fn analyze(v: &VideoInfo) -> Result<Vec<Vec<f32>>> {
    let cache = Path::new(&v.path).with_extension("s8analysis.bin");
    if let Some(s) = read_cache(&cache, &v.path) {
        crate::log(&format!("Using the earlier analysis: {}", cache.display()));
        return Ok(s);
    }
    // Gray is computed here from the Y plane of the scaled picture: ffmpeg's own gray conversion
    // differs between versions (rounding and dither), the Y plane does not.
    let filter = format!("scale={AW}:{AH}:flags=area");
    let mut r = Reader::new(&v.path, Some(&filter), "yuv420p", AW * AH * 3 / 2, None, None)?;
    let range: Vec<u8> = (0..256)
        .map(|y| if v.full_range { y as u8 } else { ((y as f64 - 16.0) * 255.0 / 219.0).round().clamp(0.0, 255.0) as u8 })
        .collect();
    let mut s: Vec<Vec<f32>> = Vec::with_capacity(v.n + 16);
    loop {
        crate::check_stop()?;
        let mut batch = Vec::with_capacity(256);
        while batch.len() < 256 {
            match r.next_frame() {
                Some(b) => batch.push(b[..AW * AH].iter().map(|&y| range[y as usize]).collect::<Vec<u8>>()),
                None => break,
            }
        }
        if batch.is_empty() {
            break;
        }
        let last = batch.len() < 256;
        s.extend(batch.par_iter().map(|b| frame_features(b)).collect::<Vec<_>>());
        crate::progress(&format!("Analysis: {} / {} frames", s.len(), v.n), s.len(), v.n.max(s.len()));
        if last {
            break;
        }
    }
    let _ = write_cache(&cache, &s);
    Ok(s)
}

fn read_cache(p: &Path, video: &str) -> Option<Vec<Vec<f32>>> {
    let cm = std::fs::metadata(p).ok()?.modified().ok()?;
    if cm <= std::fs::metadata(video).ok()?.modified().ok()? {
        return None;
    }
    let b = std::fs::read(p).ok()?;
    if b.len() < 8 || &b[..4] != b"S8A1" {
        return None;
    }
    let n = u32::from_le_bytes(b[4..8].try_into().ok()?) as usize;
    if b.len() != 8 + n * AH * 4 {
        return None;
    }
    Some(
        b[8..]
            .chunks_exact(AH * 4)
            .map(|r| r.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect())
            .collect(),
    )
}

fn write_cache(p: &Path, s: &[Vec<f32>]) -> Result<()> {
    let mut b = Vec::with_capacity(8 + s.len() * AH * 4);
    b.extend_from_slice(b"S8A1");
    b.extend_from_slice(&(s.len() as u32).to_le_bytes());
    for r in s {
        for v in r {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(p, b)?;
    Ok(())
}

/// Viterbi: frame-line position over time. Moving a row at a time is cheap, a fast slide
/// (at most MAX_SLIDE rows per frame, when the film rolls) moderately priced, a jump anywhere expensive.
pub fn track(s: &[Vec<f32>]) -> Vec<i32> {
    const JUMP: f32 = 1.2;
    const STEP: f64 = 0.04;
    const MAX_STEP: usize = 6;
    const SLIDE: f32 = 0.35;
    const MAX_SLIDE: usize = 48;
    let n = s.len();
    if n == 0 {
        return vec![];
    }
    let h = AH;
    let mut back = vec![0i16; n * h];
    let mut cost: Vec<f32> = s[0].iter().map(|v| -v).collect();
    let mut best = vec![0f32; h];
    let mut arg = vec![0i16; h];
    for t in 1..n {
        for i in 0..h {
            best[i] = cost[i];
            arg[i] = i as i16;
        }
        for dd in 1..=MAX_STEP {
            let add = (STEP * dd as f64) as f32;
            for sv in [dd as i64, -(dd as i64)] {
                for i in 0..h {
                    let j = (i as i64 - sv).rem_euclid(h as i64) as usize;
                    let c = cost[j] + add;
                    if c < best[i] {
                        best[i] = c;
                        arg[i] = j as i16;
                    }
                }
            }
        }
        for i in 0..h {
            let mut mk = 0;
            let mut mv = f32::INFINITY;
            for k in 0..=2 * MAX_SLIDE {
                let v = cost[(i + h + k - MAX_SLIDE) % h];
                if v < mv {
                    mv = v;
                    mk = k;
                }
            }
            let c = mv + SLIDE;
            if c < best[i] {
                best[i] = c;
                arg[i] = ((i + h + mk - MAX_SLIDE) % h) as i16;
            }
        }
        let mut j = 0;
        for i in 1..h {
            if cost[i] < cost[j] {
                j = i;
            }
        }
        let c = cost[j] + JUMP;
        for i in 0..h {
            if c < best[i] {
                best[i] = c;
                arg[i] = j as i16;
            }
        }
        back[t * h..(t + 1) * h].copy_from_slice(&arg);
        for i in 0..h {
            cost[i] = best[i] - s[t][i];
        }
    }
    let mut path = vec![0i32; n];
    let mut j = 0;
    for i in 1..h {
        if cost[i] < cost[j] {
            j = i;
        }
    }
    path[n - 1] = j as i32;
    for t in (1..n).rev() {
        path[t - 1] = back[t * h + path[t] as usize] as i32;
    }
    path
}

#[derive(Clone, Debug)]
pub struct Run {
    pub start: usize,
    pub end: usize,
    /// confidence: median feature along the path
    pub confidence: f32,
    /// median frame-line row (analysis rows)
    pub position: i32,
    pub repair: bool,
    /// classifier probability, when the run was judged by the model
    pub p: Option<f32>,
    /// the user changed the repair decision
    pub manual: bool,
}

impl Run {
    pub fn action(&self) -> &'static str {
        if self.repair {
            "REPAIRED"
        } else if self.manual {
            "skipped (by user)"
        } else {
            "skipped (not a slip)"
        }
    }
}

pub fn overlaps(a: usize, b: usize, ranges: &[(usize, usize)]) -> bool {
    ranges.iter().any(|&(ra, rb)| a < rb && ra < b)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

pub struct Decision {
    pub runs: Vec<Run>,
    /// normal frame-line position (analysis rows) for every run
    pub reference: Vec<i32>,
}

/// Runs and shifts. `model_p`: classifier probability for the run (start, end); when present it
/// decides instead of the confidence value.
pub fn decide(
    s: &[Vec<f32>],
    path: &[i32],
    threshold: f32,
    skip: &[(usize, usize)],
    force: &[(usize, usize)],
    model_p: &std::collections::HashMap<(usize, usize), f32>,
    model_threshold: f32,
) -> Decision {
    let n = s.len();
    let h = AH as i32;
    let conf: Vec<f32> = (0..n).map(|t| s[t][path[t] as usize]).collect();
    let mut normal: Vec<bool> = path.iter().map(|&p| p <= EDGE || p >= h - EDGE).collect();
    // in a rolling slip the frame line passes the edge for a couple of frames: merge the runs
    let mut gap: Vec<usize> = Vec::new();
    for t in 0..n {
        if normal[t] {
            gap.push(t);
            continue;
        }
        if !gap.is_empty() && gap.len() <= MERGE_GAP && gap[0] > 0 && !normal[gap[0] - 1] {
            for &v in &gap {
                normal[v] = false;
            }
        }
        gap.clear();
    }
    let unwrapped: Vec<f64> = path.iter().map(|&p| if p <= EDGE { (p + h) as f64 } else { p as f64 }).collect();
    let mut runs = Vec::new();
    let mut reference = Vec::new();
    let mut t = 0;
    while t < n {
        if normal[t] {
            t += 1;
            continue;
        }
        let a = t;
        while t < n && !normal[t] {
            t += 1;
        }
        let b = t;
        let cf = median(&mut conf[a..b].iter().map(|&v| v as f64).collect::<Vec<_>>()) as f32;
        let mut near: Vec<f64> = (a.saturating_sub(75)..a).chain(b..(b + 75).min(n)).filter(|&k| normal[k]).map(|k| unwrapped[k]).collect();
        let refv = if near.is_empty() { h - 7 } else { median(&mut near).round_ties_even() as i32 };
        let p = model_p.get(&(a, b)).copied();
        let detected = match p {
            Some(p) => p >= model_threshold,
            None => cf >= threshold,
        };
        let manual = overlaps(a, b, skip);
        let repair = (detected || overlaps(a, b, force)) && !manual;
        let position = median(&mut path[a..b].iter().map(|&v| v as f64).collect::<Vec<_>>()) as i32;
        runs.push(Run { start: a, end: b, confidence: cf, position, repair, p, manual });
        reference.push(refv.rem_euclid(h));
    }
    Decision { runs, reference }
}

/// Frames shown to the classifier: at most 12 evenly spaced from every run whose frame line is
/// clearly inside the picture.
pub fn model_samples(path: &[i32], runs: &[Run]) -> Vec<((usize, usize), Vec<usize>)> {
    let h = AH as f64;
    let mut out = Vec::new();
    for r in runs {
        let pos = r.position as f64;
        if !(MODEL_ZONE * h <= pos && pos <= (1.0 - MODEL_ZONE) * h) {
            continue;
        }
        let inside: Vec<usize> = (r.start..r.end).filter(|&t| EDGE < path[t] && path[t] < AH as i32 - EDGE).collect();
        if inside.is_empty() {
            continue;
        }
        let m = inside.len().min(12);
        let picked: Vec<usize> = if m == 1 {
            vec![inside[0]]
        } else {
            let step = (inside.len() - 1) as f64 / (m - 1) as f64;
            (0..m).map(|k| if k == m - 1 { inside[inside.len() - 1] } else { inside[(k as f64 * step) as usize] }).collect()
        };
        out.push(((r.start, r.end), picked));
    }
    out
}

/// Analysis rows -> video rows.
pub fn rows_to_px(v: i32, h: usize) -> i64 {
    (v as f64 * h as f64 / AH as f64).round_ties_even() as i64
}

pub fn report(path: &Path, runs: &[Run], fps: f64) -> Result<()> {
    use crate::video::timestamp;
    let mut t = String::from("start     end       frames  confidence  model  frame line (%)  action\n");
    for r in runs {
        let mp = match r.p {
            Some(p) => format!("{p:5.2}"),
            None => "    -".into(),
        };
        t += &format!(
            "{}  {}  {:6}  {:10.2}  {}  {:14.0}  {}\n",
            timestamp(r.start, fps),
            timestamp(r.end, fps),
            r.end - r.start,
            r.confidence,
            mp,
            100.0 * r.position as f64 / AH as f64,
            r.action()
        );
    }
    std::fs::write(path, t)?;
    Ok(())
}
