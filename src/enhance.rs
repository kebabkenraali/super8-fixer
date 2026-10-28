//! Picture enhancement after the slip repair: dust and dirt removal and colour restoration.
//!
//! Dust, hairs and dirt on film are in one film frame only. A spot is removed where the frame is
//! clearly darker (or brighter) than both its previous and next film frame while those two agree
//! with each other; the neighbours are aligned to the frame first. Only small spots are touched, so
//! moving objects are left alone.
//!
//! Colour restoration stretches every colour channel between its darkest and brightest values,
//! which removes the colour cast of faded film (typically red or magenta) and restores contrast.
//! The levels are measured per frame and smoothed over each scene, so that the picture does not flicker.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use anyhow::Result;

use crate::video::{Reader, VideoInfo};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Colour {
    Off,
    /// cast removal and contrast, close to the original look
    #[default]
    Natural,
    /// stronger contrast and saturation
    Vivid,
}

impl Colour {
    pub const ALL: [Colour; 3] = [Colour::Off, Colour::Natural, Colour::Vivid];
    pub fn label(self) -> &'static str {
        match self {
            Colour::Off => "Off",
            Colour::Natural => "Natural",
            Colour::Vivid => "Vivid",
        }
    }
    pub fn parse(s: &str) -> Option<Colour> {
        Colour::ALL.into_iter().find(|c| c.label().eq_ignore_ascii_case(s))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EnhanceSettings {
    pub dust: bool,
    /// dust sensitivity, 0.5 (only clear spots) .. 2.0 (also faint ones)
    pub dust_strength: f32,
    pub colour: Colour,
}

impl Default for EnhanceSettings {
    fn default() -> Self {
        EnhanceSettings { dust: true, dust_strength: 1.0, colour: Colour::Natural }
    }
}

impl EnhanceSettings {
    pub fn active(&self) -> bool {
        self.dust || self.colour != Colour::Off
    }
}

// ---------------------------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------------------------

const SW: usize = 160; // size of the copy the colour levels are measured on
const SH: usize = 120;
const SCENE_CUT: f64 = 0.35; // colour-histogram difference that starts a new scene
const LEVEL_SMOOTHING: f64 = 12.0; // frames

/// Colour levels of one frame: per channel the 0.5 % and 99.5 % points.
#[derive(Clone, Copy, Debug, Default)]
pub struct Levels {
    lo: [f32; 3],
    hi: [f32; 3],
}

fn measure_levels(rgb: &[u8]) -> (Levels, [f64; 48]) {
    let mut hist = [[0u32; 256]; 3];
    for p in rgb.chunks_exact(3) {
        for c in 0..3 {
            hist[c][p[c] as usize] += 1;
        }
    }
    let n = (rgb.len() / 3) as u32;
    let point = |h: &[u32; 256], q: f64| -> f32 {
        let target = (q * n as f64) as u32;
        let mut acc = 0;
        for (v, &k) in h.iter().enumerate() {
            acc += k;
            if acc > target {
                return v as f32;
            }
        }
        255.0
    };
    let mut l = Levels::default();
    let mut coarse = [0f64; 48];
    for c in 0..3 {
        l.lo[c] = point(&hist[c], 0.005);
        l.hi[c] = point(&hist[c], 0.995);
        for (v, &k) in hist[c].iter().enumerate() {
            coarse[c * 16 + v / 16] += k as f64 / n as f64;
        }
    }
    (l, coarse)
}

/// Colour levels for frames start..start+count (to the end when None), smoothed within scenes.
pub fn colour_levels(v: &VideoInfo, start: usize, count: Option<usize>) -> Result<Vec<Levels>> {
    let filter = format!("scale={SW}:{SH}:flags=area");
    let mut r = Reader::new(&v.path, Some(&filter), "rgb24", SW * SH * 3, (start > 0).then(|| start as f64 / v.fps), count)?;
    let mut raw = Vec::with_capacity(v.n + 16);
    let mut cut = Vec::with_capacity(v.n + 16);
    let mut prev: Option<[f64; 48]> = None;
    while let Some(b) = r.next_frame() {
        let (l, h) = measure_levels(&b);
        let d = prev.map(|p| p.iter().zip(h.iter()).map(|(a, b)| (a - b).abs()).sum::<f64>() / 6.0).unwrap_or(1.0);
        cut.push(d > SCENE_CUT);
        raw.push(l);
        prev = Some(h);
        if raw.len() % 500 == 0 {
            crate::check_stop()?;
            let total = count.unwrap_or(v.n);
            crate::progress(&format!("Measuring colours: {} / {total} frames", raw.len()), raw.len(), total.max(raw.len()));
        }
    }
    Ok(smooth_levels(&raw, &cut))
}

/// Gaussian smoothing of the levels inside each scene.
fn smooth_levels(raw: &[Levels], cut: &[bool]) -> Vec<Levels> {
    let n = raw.len();
    let mut out = raw.to_vec();
    let r = (3.0 * LEVEL_SMOOTHING) as i64;
    let k: Vec<f64> = (-r..=r).map(|i| (-0.5 * (i as f64 / LEVEL_SMOOTHING).powi(2)).exp()).collect();
    let mut starts: Vec<usize> = (0..n).filter(|&t| t == 0 || cut[t]).collect();
    starts.push(n);
    for w in starts.windows(2) {
        let (a, b) = (w[0], w[1]);
        for t in a..b {
            let (mut lo, mut hi, mut ws) = ([0f64; 3], [0f64; 3], 0f64);
            for (j, kv) in k.iter().enumerate() {
                let q = t as i64 + j as i64 - r;
                if q < a as i64 || q >= b as i64 {
                    continue;
                }
                for c in 0..3 {
                    lo[c] += kv * raw[q as usize].lo[c] as f64;
                    hi[c] += kv * raw[q as usize].hi[c] as f64;
                }
                ws += kv;
            }
            for c in 0..3 {
                out[t].lo[c] = (lo[c] / ws) as f32;
                out[t].hi[c] = (hi[c] / ws) as f32;
            }
        }
    }
    out
}

/// Levels measured from a few frames (for the preview).
pub fn levels_of(frames: &[&[u8]], w: usize, h: usize) -> Levels {
    let mut sum = Levels::default();
    for f in frames {
        let im = image::RgbImage::from_raw(w as u32, h as u32, f.to_vec()).unwrap();
        let small = image::imageops::resize(&im, SW as u32, SH as u32, image::imageops::FilterType::Triangle);
        let (l, _) = measure_levels(small.as_raw());
        for c in 0..3 {
            sum.lo[c] += l.lo[c] / frames.len() as f32;
            sum.hi[c] += l.hi[c] / frames.len() as f32;
        }
    }
    sum
}

/// Applies the colour restoration in place.
pub fn apply_colour(frame: &mut [u8], w: usize, l: &Levels, mode: Colour) {
    let (target, blend, sat, curve) = match mode {
        Colour::Off => return,
        Colour::Natural => ((8.0, 247.0), 0.8f32, 1.12f32, 0.0f32),
        Colour::Vivid => ((2.0, 253.0), 1.0, 1.3, 0.25),
    };
    // per-channel lookup table: stretch, at most 2.5x gain (dark or flat scenes stay natural)
    let mut lut = [[0f32; 256]; 3];
    let mean_lo = (l.lo[0] + l.lo[1] + l.lo[2]) / 3.0;
    let mean_hi = (l.hi[0] + l.hi[1] + l.hi[2]) / 3.0;
    for c in 0..3 {
        // a nearly black or white channel keeps the range of the others (a cast is removed, not invented)
        let lo = l.lo[c].min(mean_lo + 40.0);
        let hi = l.hi[c].max(mean_hi - 40.0).max(lo + 255.0 / 2.5);
        for v in 0..256 {
            let x = ((v as f32 - lo) / (hi - lo)).clamp(0.0, 1.0);
            let x = x + curve * (x * x * (3.0 - 2.0 * x) - x); // gentle S-curve
            let s = target.0 + x * (target.1 - target.0);
            lut[c][v] = v as f32 + blend * (s - v as f32);
        }
    }
    frame.par_chunks_mut(w * 3).for_each(|row| {
        for p in row.chunks_exact_mut(3) {
            let r = lut[0][p[0] as usize];
            let g = lut[1][p[1] as usize];
            let b = lut[2][p[2] as usize];
            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            p[0] = (y + sat * (r - y)).round().clamp(0.0, 255.0) as u8;
            p[1] = (y + sat * (g - y)).round().clamp(0.0, 255.0) as u8;
            p[2] = (y + sat * (b - y)).round().clamp(0.0, 255.0) as u8;
        }
    });
}

// ---------------------------------------------------------------------------------------------
// Dust and dirt
// ---------------------------------------------------------------------------------------------

const MW: usize = 480; // alignment resolution
const DUST_DIFF: f32 = 40.0; // a spot differs this much from both neighbours (gray levels, strength 1)
const AGREE: f32 = 8.0; // ...while the neighbours differ at most this much from each other
const MAX_AREA: f64 = 0.0006; // largest spot as a share of the picture area
const MIN_AREA: f64 = 0.000015; // smallest spot (10 px in a 960 x 720 picture); smaller ones are grain
const MAX_SIDE: f64 = 0.05; // largest spot side as a share of the picture width
const GROW: usize = 2; // the mask is grown by this many pixels around a spot
const MIN_FILL: f32 = 0.35; // a spot fills at least this share of its bounding box
const MAX_RESIDUAL: f32 = 9.0; // a neighbour is used only if it lines up this well (mean gray difference)

fn gray_full(rgb: &[u8]) -> Vec<f32> {
    rgb.par_chunks(3 * 1024).flat_map_iter(|c| c.chunks_exact(3).map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0)).collect()
}

/// Small gray copy for alignment (area average).
fn small_gray(rgb: &[u8], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let mh = (h * MW / w).max(8);
    let mut g = vec![0f32; MW * mh];
    let (fx, fy) = (w as f32 / MW as f32, h as f32 / mh as f32);
    for y in 0..mh {
        for x in 0..MW {
            let (x0, y0) = ((x as f32 * fx) as usize, (y as f32 * fy) as usize);
            let x1 = (((x + 1) as f32 * fx) as usize).min(w).max(x0 + 1);
            let y1 = (((y + 1) as f32 * fy) as usize).min(h).max(y0 + 1);
            let mut s = 0f32;
            for yy in (y0..y1).step_by(2) {
                for xx in (x0..x1).step_by(2) {
                    let p = (yy * w + xx) * 3;
                    s += rgb[p] as f32 + rgb[p + 1] as f32 + rgb[p + 2] as f32;
                }
            }
            let n = (y1 - y0).div_ceil(2) * (x1 - x0).div_ceil(2);
            g[y * MW + x] = s / (3 * n) as f32;
        }
    }
    (g, MW, mh)
}

/// Integer translation (dx, dy) at full resolution so that `other(x - d)` matches `cur(x)`,
/// found coarse-to-fine on small copies; also returns the mean difference after alignment.
fn align(cur: &(Vec<f32>, usize, usize), other: &(Vec<f32>, usize, usize), scale: f32) -> ((i64, i64), f32) {
    let (a, w, h) = (&cur.0, cur.1, cur.2);
    let b = &other.0;
    let cost = |dx: i64, dy: i64, step: usize| -> f32 {
        let (mut e, mut n) = (0f32, 0usize);
        for y in (h / 10..h * 9 / 10).step_by(step) {
            let sy = y as i64 - dy;
            if sy < 0 || sy >= h as i64 {
                continue;
            }
            for x in (w / 10..w * 9 / 10).step_by(step) {
                let sx = x as i64 - dx;
                if sx < 0 || sx >= w as i64 {
                    continue;
                }
                e += (a[y * w + x] - b[sy as usize * w + sx as usize]).abs();
                n += 1;
            }
        }
        if n == 0 { f32::INFINITY } else { e / n as f32 }
    };
    let mut best = (cost(0, 0, 4), 0i64, 0i64);
    for dy in (-12..=12).step_by(2) {
        for dx in (-12..=12).step_by(2) {
            let e = cost(dx, dy, 4);
            if e < best.0 {
                best = (e, dx, dy);
            }
        }
    }
    let (cx, cy) = (best.1, best.2);
    let mut fine = (cost(cx, cy, 2), cx, cy);
    for dy in cy - 1..=cy + 1 {
        for dx in cx - 1..=cx + 1 {
            let e = cost(dx, dy, 2);
            if e < fine.0 {
                fine = (e, dx, dy);
            }
        }
    }
    (((fine.1 as f32 * scale).round() as i64, (fine.2 as f32 * scale).round() as i64), fine.0)
}

/// Removes dust and dirt from `cur` using the previous and next frames (already repaired),
/// `prevs` = [t-1, t-2], `nexts` = [t+1, t+2]. Returns the number of removed spots.
fn remove_dust(cur: &mut [u8], c: &Frame, prevs: &[Option<&Frame>], nexts: &[Option<&Frame>], w: usize, h: usize, strength: f32) -> usize {
    let scale = w as f32 / MW as f32;
    let gc = &c.gray;
    let (Some((p, dp)), Some((n, dn))) = (pick(c, prevs, w, h, scale), pick(c, nexts, w, h, scale)) else {
        return 0;
    };
    // pixel index of (x, y) in a neighbour shifted by d (edges repeat)
    let at = |x: usize, y: usize, d: (i64, i64)| -> usize {
        let sx = (x as i64 - d.0).clamp(0, w as i64 - 1) as usize;
        let sy = (y as i64 - d.1).clamp(0, h as i64 - 1) as usize;
        sy * w + sx
    };
    let t = DUST_DIFF / strength.clamp(0.3, 3.0);
    let border = (w / 100).max(4);
    let mut mask = vec![0u8; w * h];
    mask.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if y < border || y >= h - border {
            return;
        }
        for x in border..w - border {
            let i = y * w + x;
            let (c, a, b) = (gc[i], p.gray[at(x, y, dp)], n.gray[at(x, y, dn)]);
            if (a - b).abs() > AGREE {
                continue;
            }
            let (d1, d2) = (c - a, c - b);
            if (d1 < -t && d2 < -t) || (d1 > t && d2 > t) {
                row[x] = 1;
            }
        }
    });
    // connected spots; keep only small ones
    let max_area = (MAX_AREA * (w * h) as f64) as usize;
    let max_side = (MAX_SIDE * w as f64) as usize;
    let min_area = ((MIN_AREA * (w * h) as f64) as usize).max(4);
    let mut label = vec![false; w * h];
    let mut stack = Vec::new();
    let mut spots: Vec<Vec<usize>> = Vec::new();
    for start in 0..w * h {
        if mask[start] == 0 || label[start] {
            continue;
        }
        let mut px = Vec::new();
        let (mut x0, mut x1, mut y0, mut y1) = (w, 0, h, 0);
        stack.push(start);
        label[start] = true;
        while let Some(i) = stack.pop() {
            px.push(i);
            let (x, y) = (i % w, i / w);
            (x0, x1, y0, y1) = (x0.min(x), x1.max(x), y0.min(y), y1.max(y));
            let left = if x > 0 { i - 1 } else { usize::MAX };
            let right = if x + 1 < w { i + 1 } else { usize::MAX };
            for j in [left, right, i.wrapping_sub(w), i + w] {
                if j < w * h && mask[j] == 1 && !label[j] {
                    label[j] = true;
                    stack.push(j);
                }
            }
        }
        // a single pixel is noise, a large or long area is picture content (moving objects)
        if px.len() < min_area || px.len() > max_area || x1 - x0 > max_side || y1 - y0 > max_side {
            continue;
        }
        // dust is a compact spot; thin strokes are the edges of something that moved
        let (bw, bh) = (x1 - x0 + 1, y1 - y0 + 1);
        if (px.len() as f32) < MIN_FILL * (bw * bh) as f32 || bw.max(bh) > 3 * bw.min(bh) + 3 {
            continue;
        }
        // the spot differs from its own surroundings, and the neighbours' picture fits into them
        let ring = ring_median(gc, w, h, (x0, x1, y0, y1), &label);
        let spot = px.iter().map(|&i| gc[i]).sum::<f32>() / px.len() as f32;
        let filled = px.iter().map(|&i| (p.gray[at(i % w, i / w, dp)] + n.gray[at(i % w, i / w, dn)]) / 2.0).sum::<f32>() / px.len() as f32;
        let contrast = (spot - ring).abs();
        if contrast < 0.5 * t || (filled - ring).abs() > 0.5 * contrast {
            continue;
        }
        spots.push(px);
    }
    let mut fill = vec![false; w * h];
    for s in &spots {
        for &i in s {
            let (x, y) = (i % w, i / w);
            for yy in y.saturating_sub(GROW)..(y + GROW + 1).min(h) {
                for xx in x.saturating_sub(GROW)..(x + GROW + 1).min(w) {
                    fill[yy * w + xx] = true;
                }
            }
        }
    }
    cur.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = y * w + x;
            if fill[i] {
                let (ip, inx) = (at(x, y, dp), at(x, y, dn));
                for ch in 0..3 {
                    row[x * 3 + ch] = ((p.rgb[ip * 3 + ch] as u16 + n.rgb[inx * 3 + ch] as u16) / 2) as u8;
                }
            }
        }
    });
    spots.len()
}

/// Median gray of a ring 3..6 px around the box (x0, x1, y0, y1), outside the marked pixels.
fn ring_median(g: &[f32], w: usize, h: usize, (x0, x1, y0, y1): (usize, usize, usize, usize), marked: &[bool]) -> f32 {
    let (ox0, ox1) = (x0.saturating_sub(6), (x1 + 6).min(w - 1));
    let (oy0, oy1) = (y0.saturating_sub(6), (y1 + 6).min(h - 1));
    let mut v = Vec::new();
    for y in oy0..=oy1 {
        for x in ox0..=ox1 {
            let inner = x + 3 > x0 && x < x1 + 3 && y + 3 > y0 && y < y1 + 3;
            if !inner && !marked[y * w + x] {
                v.push(g[y * w + x]);
            }
        }
    }
    if v.is_empty() {
        return 0.0;
    }
    let k = v.len() / 2;
    *v.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap()).1
}

/// Refines a translation at full resolution within +-2 px; also returns the mean difference left.
fn refine_full(a: &[f32], b: &[f32], w: usize, h: usize, d: (i64, i64)) -> ((i64, i64), f32) {
    let cost = |dx: i64, dy: i64| -> f32 {
        let (mut e, mut n) = (0f32, 0usize);
        for y in (h / 8..h * 7 / 8).step_by(6) {
            let sy = (y as i64 - dy).clamp(0, h as i64 - 1) as usize;
            for x in (w / 8..w * 7 / 8).step_by(3) {
                let sx = (x as i64 - dx).clamp(0, w as i64 - 1) as usize;
                e += (a[y * w + x] - b[sy * w + sx]).abs();
                n += 1;
            }
        }
        e / n.max(1) as f32
    };
    let mut best = (cost(d.0, d.1), d);
    for dy in d.1 - 2..=d.1 + 2 {
        for dx in d.0 - 2..=d.0 + 2 {
            let e = cost(dx, dy);
            if e < best.0 {
                best = (e, (dx, dy));
            }
        }
    }
    (best.1, best.0)
}

/// The nearest neighbour in `list` that is a different film frame (18 -> 25 fps repeats frames),
/// with its translation relative to `c`.
fn pick<'a>(c: &Frame, list: &[Option<&'a Frame>], w: usize, h: usize, scale: f32) -> Option<(&'a Frame, (i64, i64))> {
    for f in list.iter().flatten() {
        let (d, e) = align(&c.small, &f.small, scale);
        if e > 1.2 {
            // a different film frame: use it only when it lines up well (no strong motion or blur)
            let (d, rest) = refine_full(&c.gray, &f.gray, w, h, d);
            return (rest < MAX_RESIDUAL).then_some((f, d));
        }
    }
    None
}

/// A frame with the gray copies the dust removal needs (computed once per frame).
struct Frame {
    rgb: Vec<u8>,
    gray: Vec<f32>,
    small: (Vec<f32>, usize, usize),
}

/// The enhancement stage of the rendering: frames go in in order, enhanced frames come out
/// REACH frames later (dust removal needs the neighbours on both sides).
pub struct Enhancer {
    s: EnhanceSettings,
    w: usize,
    h: usize,
    /// colour levels per frame from frame `level_start` on, or a single entry for every frame
    levels: Vec<Levels>,
    level_start: usize,
    /// frames first..first+len
    window: std::collections::VecDeque<Frame>,
    first: usize,
    /// index of the next frame to output
    emitted: usize,
    /// video index of the first pushed frame
    frame0: usize,
    pub spots: usize,
}

/// Frames needed before and after a frame for the dust removal.
pub const REACH: usize = 2;

impl Enhancer {
    /// `first_frame`: video index of the first frame that will be pushed.
    pub fn new(s: EnhanceSettings, w: usize, h: usize, levels: Vec<Levels>, level_start: usize) -> Enhancer {
        Enhancer { s, w, h, levels, level_start, window: Default::default(), first: 0, emitted: 0, spots: 0, frame0: 0 }
    }

    /// Video index of the first pushed frame (for the colour levels).
    pub fn starting_at(mut self, frame: usize) -> Enhancer {
        self.frame0 = frame;
        self
    }

    fn process(&mut self, e: usize) -> Vec<u8> {
        let k = e - self.first;
        let mut cur = self.window[k].rgb.clone();
        if self.s.dust {
            let prevs: Vec<Option<&Frame>> = (1..=REACH).map(|d| k.checked_sub(d).map(|j| &self.window[j])).collect();
            let nexts: Vec<Option<&Frame>> = (1..=REACH).map(|d| self.window.get(k + d)).collect();
            self.spots += remove_dust(&mut cur, &self.window[k], &prevs, &nexts, self.w, self.h, self.s.dust_strength);
        }
        if self.s.colour != Colour::Off {
            let l = if self.levels.len() == 1 { self.levels.first() } else { (self.frame0 + e).checked_sub(self.level_start).and_then(|i| self.levels.get(i)) };
            if let Some(l) = l {
                apply_colour(&mut cur, self.w, l, self.s.colour);
            }
        }
        cur
    }

    fn advance(&mut self) {
        self.emitted += 1;
        while self.first + REACH < self.emitted {
            self.window.pop_front();
            self.first += 1;
        }
    }

    /// Adds the next frame; returns the frames that are ready.
    pub fn push(&mut self, f: Vec<u8>) -> Vec<Vec<u8>> {
        let frame = if self.s.dust {
            let (gray, small) = rayon::join(|| gray_full(&f), || small_gray(&f, self.w, self.h));
            Frame { rgb: f, gray, small }
        } else {
            Frame { rgb: f, gray: vec![], small: (vec![], 0, 0) }
        };
        self.window.push_back(frame);
        let mut out = Vec::new();
        while self.emitted + REACH < self.first + self.window.len() {
            out.push(self.process(self.emitted));
            self.advance();
        }
        out
    }

    /// Outputs the remaining frames at the end of the video.
    pub fn finish(&mut self) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        while self.emitted < self.first + self.window.len() {
            out.push(self.process(self.emitted));
            self.advance();
        }
        out
    }
}
