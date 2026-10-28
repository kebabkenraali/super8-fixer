//! Picture stabilization: global motion between frames (block matching on an image pyramid, median
//! over blocks), scene cuts, 18 -> 25 fps duplicate frames, path smoothing over real film frames,
//! and shifting the picture.

use std::path::Path;

use anyhow::Result;
use rayon::prelude::*;

use crate::video::{Reader, VideoInfo, Writer};

pub const LW: usize = 480; // motion measurement resolution
pub const LH: usize = 360;
const MIN_SCENE: usize = 12; // minimum distance between two scene cuts (frames)
const BATCH: usize = 64; // frame pairs measured in parallel

#[derive(Clone, Debug)]
pub struct StabilizeSettings {
    /// smoothing width in film frames; larger = steadier
    pub smoothing: f64,
    /// largest correction as a percentage of the width
    pub max_shift: f64,
    /// picture difference that starts a new scene
    pub scene_threshold: f64,
    pub crf: u32,
    pub preset: String,
}

impl Default for StabilizeSettings {
    fn default() -> Self {
        StabilizeSettings { smoothing: 6.0, max_shift: 3.0, scene_threshold: 30.0, crf: 16, preset: "medium".into() }
    }
}

pub struct Measurement {
    /// content displacement from frame t-1 to frame t (LW scale)
    pub motion: Vec<[f64; 2]>,
    /// picture difference after compensating the motion
    pub error: Vec<f64>,
    /// colour histogram difference 0..1
    pub hist: Vec<f64>,
}

// ---------------------------------------------------------------------------------------------
// Motion estimation
// ---------------------------------------------------------------------------------------------

/// Gray image pyramid: levels[0] = LW x LH, each next level half the size.
struct Pyramid {
    levels: Vec<(Vec<f32>, usize, usize)>,
}

impl Pyramid {
    fn new(rgb: &[u8]) -> Pyramid {
        let g: Vec<f32> = rgb.chunks_exact(3).map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0).collect();
        let mut levels = vec![(g, LW, LH)];
        for _ in 0..2 {
            let (p, w, h) = levels.last().unwrap();
            let (w2, h2) = (w / 2, h / 2);
            let mut o = vec![0f32; w2 * h2];
            for y in 0..h2 {
                for x in 0..w2 {
                    let i = 2 * y * w + 2 * x;
                    o[y * w2 + x] = (p[i] + p[i + 1] + p[i + w] + p[i + w + 1]) / 4.0;
                }
            }
            levels.push((o, w2, h2));
        }
        Pyramid { levels }
    }
}

/// Mean absolute difference of the zero-mean block at (x0, y0, bw, bh) in `cur` and the same block
/// displaced by (dx, dy) in `prev` (cur(x) ~ prev(x - d)). None if the block leaves the picture.
fn block_cost(cur: &[f32], prev: &[f32], w: usize, h: usize, b: (usize, usize, usize, usize), d: (i32, i32), cur_mean: f32) -> Option<f32> {
    let (x0, y0, bw, bh) = b;
    let (sx, sy) = (x0 as i32 - d.0, y0 as i32 - d.1);
    if sx < 0 || sy < 0 || sx as usize + bw > w || sy as usize + bh > h {
        return None;
    }
    let (sx, sy) = (sx as usize, sy as usize);
    let mut prev_sum = 0f32;
    for y in 0..bh {
        prev_sum += prev[(sy + y) * w + sx..(sy + y) * w + sx + bw].iter().sum::<f32>();
    }
    let prev_mean = prev_sum / (bw * bh) as f32;
    let mut e = 0f32;
    for y in 0..bh {
        let c = &cur[(y0 + y) * w + x0..(y0 + y) * w + x0 + bw];
        let p = &prev[(sy + y) * w + sx..(sy + y) * w + sx + bw];
        for x in 0..bw {
            e += ((c[x] - cur_mean) - (p[x] - prev_mean)).abs();
        }
    }
    Some(e / (bw * bh) as f32)
}

fn block_mean_std(img: &[f32], w: usize, b: (usize, usize, usize, usize)) -> (f32, f32) {
    let (x0, y0, bw, bh) = b;
    let (mut s, mut s2) = (0f64, 0f64);
    for y in 0..bh {
        for &v in &img[(y0 + y) * w + x0..(y0 + y) * w + x0 + bw] {
            s += v as f64;
            s2 += (v as f64).powi(2);
        }
    }
    let n = (bw * bh) as f64;
    let m = s / n;
    (m as f32, (s2 / n - m * m).max(0.0).sqrt() as f32)
}

/// Sub-pixel offset of the minimum from three costs around it (parabola fit).
fn parabola(a: f32, b: f32, c: f32) -> f64 {
    let d = a - 2.0 * b + c;
    if d <= 1e-6 { 0.0 } else { (0.5 * (a - c) / d).clamp(-0.5, 0.5) as f64 }
}

/// Global content displacement from `prev` to `cur` (LW scale): median over a 4 x 3 grid of
/// blocks covering the central 80 % of the picture, each found coarse-to-fine on the pyramid.
fn global_motion(prev: &Pyramid, cur: &Pyramid) -> [f64; 2] {
    const GX: usize = 4;
    const GY: usize = 3;
    const COARSE: i32 = 12; // search range on the coarsest level (x4 on the full level)
    let (x0, x1, y0, y1) = (LW / 10, LW * 9 / 10, LH / 10, LH * 9 / 10);
    let (bw, bh) = ((x1 - x0) / GX, (y1 - y0) / GY);
    let mut vx = Vec::with_capacity(GX * GY);
    let mut vy = Vec::with_capacity(GX * GY);
    for gy in 0..GY {
        for gx in 0..GX {
            let full = (x0 + gx * bw, y0 + gy * bh, bw, bh);
            let (_, sd) = block_mean_std(&cur.levels[0].0, LW, full);
            if sd < 3.0 {
                continue; // no texture: nothing to match
            }
            let mut d = (0i32, 0i32);
            let mut best_cost = f32::INFINITY;
            for lv in (0..3).rev() {
                let (c, w, h) = (&cur.levels[lv].0, cur.levels[lv].1, cur.levels[lv].2);
                let p = &prev.levels[lv].0;
                let f = 1 << lv;
                let b = (full.0 / f, full.1 / f, full.2 / f, full.3 / f);
                let (cm, _) = block_mean_std(c, w, b);
                let r = if lv == 2 { COARSE } else { 2 };
                let centre = if lv == 2 { (0, 0) } else { (d.0 * 2, d.1 * 2) };
                let mut best = (f32::INFINITY, centre);
                for dy in -r..=r {
                    for dx in -r..=r {
                        let dd = (centre.0 + dx, centre.1 + dy);
                        if let Some(e) = block_cost(c, p, w, h, b, dd, cm) {
                            if e < best.0 {
                                best = (e, dd);
                            }
                        }
                    }
                }
                if !best.0.is_finite() {
                    break;
                }
                d = best.1;
                best_cost = best.0;
            }
            if !best_cost.is_finite() {
                continue;
            }
            let c = &cur.levels[0].0;
            let p = &prev.levels[0].0;
            let (cm, _) = block_mean_std(c, LW, full);
            let cost = |dd: (i32, i32)| block_cost(c, p, LW, LH, full, dd, cm).unwrap_or(best_cost);
            let fx = parabola(cost((d.0 - 1, d.1)), best_cost, cost((d.0 + 1, d.1)));
            let fy = parabola(cost((d.0, d.1 - 1)), best_cost, cost((d.0, d.1 + 1)));
            vx.push(d.0 as f64 + fx);
            vy.push(d.1 as f64 + fy);
        }
    }
    if vx.is_empty() {
        return [0.0, 0.0];
    }
    [median(&vx), median(&vy)]
}

/// Mean picture difference in the central region after shifting `prev` by the rounded motion.
fn compensated_error(prev: &Pyramid, cur: &Pyramid, m: [f64; 2]) -> f64 {
    let (a, b) = (&prev.levels[0].0, &cur.levels[0].0);
    let (y0, y1, x0, x1) = (LH / 10, LH * 9 / 10, LW / 10, LW * 9 / 10);
    let (sx, sy) = (m[0].round() as i64, m[1].round() as i64);
    let mut e = 0.0;
    for y in y0..y1 {
        let ly = (y as i64 - sy).rem_euclid(LH as i64) as usize;
        for x in x0..x1 {
            let lx = (x as i64 - sx).rem_euclid(LW as i64) as usize;
            e += (a[ly * LW + lx] - b[y * LW + x]).abs() as f64;
        }
    }
    e / ((y1 - y0) * (x1 - x0)) as f64
}

/// Motion and error for the whole video. Cached in <video>.s8motion.bin.
pub fn measure(v: &VideoInfo) -> Result<Measurement> {
    let cache = Path::new(&v.path).with_extension("s8motion.bin");
    if let Some(m) = read_cache(&cache, &v.path) {
        crate::log(&format!("Using the earlier motion measurement: {}", cache.display()));
        return Ok(m);
    }
    let size = LW * LH * 3;
    let filter = format!("scale={LW}:{LH}:flags=area");
    let mut r = Reader::new(&v.path, Some(&filter), "rgb24", size, None, None)?;
    let mut motion = vec![[0.0, 0.0]];
    let mut error = vec![0.0];
    let mut prev: Option<Pyramid> = None;
    loop {
        crate::check_stop()?;
        let mut batch: Vec<Pyramid> = Vec::with_capacity(BATCH);
        while batch.len() < BATCH {
            match r.next_frame() {
                Some(b) => batch.push(Pyramid::new(&b)),
                None => break,
            }
        }
        if batch.is_empty() {
            break;
        }
        let last = batch.len() < BATCH;
        let start = usize::from(prev.is_none());
        let results: Vec<([f64; 2], f64)> = (start..batch.len())
            .into_par_iter()
            .map(|i| {
                let p = if i == 0 { prev.as_ref().unwrap() } else { &batch[i - 1] };
                let m = global_motion(p, &batch[i]);
                (m, compensated_error(p, &batch[i], m))
            })
            .collect();
        for (m, e) in results {
            motion.push(m);
            error.push(e);
        }
        prev = batch.pop();
        crate::progress(&format!("Motion measurement: {} / {} frames", motion.len(), v.n), motion.len(), v.n.max(motion.len()));
        if last {
            break;
        }
    }
    drop(r);
    crate::log("Computing colour histograms for scene-cut detection...");
    let mut hist = histogram_diffs(v)?;
    hist.resize(motion.len(), 0.0);
    let m = Measurement { motion, error, hist };
    let _ = write_cache(&cache, &m);
    Ok(m)
}

/// Colour-histogram difference of consecutive frames 0..1 (a scene cut changes it, a pan does not).
fn histogram_diffs(v: &VideoInfo) -> Result<Vec<f64>> {
    let (w, h) = (160, 120);
    let mut r = Reader::new(&v.path, Some(&format!("scale={w}:{h}:flags=area")), "rgb24", w * h * 3, None, None)?;
    let mut diff = vec![0.0];
    let mut prev: Option<[f64; 48]> = None;
    while let Some(b) = r.next_frame() {
        crate::check_stop()?;
        let mut hi = [0f64; 48];
        for p in b.chunks_exact(3) {
            for c in 0..3 {
                hi[c * 16 + (p[c] / 16) as usize] += 1.0;
            }
        }
        let n = 3.0 * (w * h) as f64;
        hi.iter_mut().for_each(|v| *v /= n);
        if let Some(e) = prev {
            diff.push(hi.iter().zip(e.iter()).map(|(a, b)| (a - b).abs()).sum::<f64>() / 2.0);
        }
        prev = Some(hi);
    }
    Ok(diff)
}

fn read_cache(p: &Path, video: &str) -> Option<Measurement> {
    if std::fs::metadata(p).ok()?.modified().ok()? <= std::fs::metadata(video).ok()?.modified().ok()? {
        return None;
    }
    let b = std::fs::read(p).ok()?;
    if b.len() < 8 || &b[..4] != b"S8M1" {
        return None;
    }
    let n = u32::from_le_bytes(b[4..8].try_into().ok()?) as usize;
    if b.len() != 8 + n * 4 * 8 {
        return None;
    }
    let v: Vec<f64> = b[8..].chunks_exact(8).map(|x| f64::from_le_bytes(x.try_into().unwrap())).collect();
    Some(Measurement {
        motion: (0..n).map(|i| [v[i * 4], v[i * 4 + 1]]).collect(),
        error: (0..n).map(|i| v[i * 4 + 2]).collect(),
        hist: (0..n).map(|i| v[i * 4 + 3]).collect(),
    })
}

fn write_cache(p: &Path, m: &Measurement) -> Result<()> {
    let mut b = Vec::with_capacity(8 + m.motion.len() * 32);
    b.extend_from_slice(b"S8M1");
    b.extend_from_slice(&(m.motion.len() as u32).to_le_bytes());
    for i in 0..m.motion.len() {
        for v in [m.motion[i][0], m.motion[i][1], m.error[i], m.hist[i]] {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(p, b)?;
    Ok(())
}

fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

// ---------------------------------------------------------------------------------------------
// Scenes, duplicates and the smoothed path
// ---------------------------------------------------------------------------------------------

/// Frame t starts a new scene when the colour histogram changes and the pictures do not match even
/// after compensating the motion, or when the histogram changes a lot. Of a cluster the strongest is kept.
pub fn scene_cuts(m: &Measurement, threshold: f64) -> Vec<bool> {
    let n = m.motion.len();
    let base = if n > 1 { median(&m.error[1..]) } else { 1.0 };
    let limit = threshold.max(4.0 * base);
    let mut cand: Vec<usize> = (0..n).filter(|&t| (m.error[t] > limit && m.hist[t] > 0.2) || m.hist[t] > 0.4).collect();
    cand.sort_by(|&a, &b| m.hist[b].partial_cmp(&m.hist[a]).unwrap()); // stable sort
    let mut keep: Vec<usize> = Vec::new();
    for t in cand {
        if keep.iter().all(|&k| t.abs_diff(k) >= MIN_SCENE) {
            keep.push(t);
        }
    }
    let mut cut = vec![false; n];
    for t in keep {
        cut[t] = true;
    }
    if n > 0 {
        cut[0] = true;
    }
    cut
}

/// 18 fps film as 25 fps video: some video frames repeat the previous one.
pub fn duplicates(m: &Measurement) -> Vec<bool> {
    (0..m.motion.len()).map(|t| m.motion[t][0].abs().max(m.motion[t][1].abs()) < 0.3 && m.error[t] < 1.0).collect()
}

fn gauss(x: &[[f64; 2]], sigma: f64) -> Vec<[f64; 2]> {
    if sigma <= 0.0 || x.len() < 2 {
        return x.to_vec();
    }
    let r = (3.0 * sigma) as i64;
    let mut k: Vec<f64> = (-r..=r).map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp()).collect();
    let s: f64 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    let n = x.len() as i64;
    (0..n)
        .map(|i| {
            let mut o = [0.0; 2];
            for (j, kv) in k.iter().enumerate() {
                let q = (i + j as i64 - r).clamp(0, n - 1) as usize;
                o[0] += kv * x[q][0];
                o[1] += kv * x[q][1];
            }
            o
        })
        .collect()
}

/// Shift (LW scale) added to every frame. The path is smoothed over real film frames only; a
/// duplicate frame gets the same shift as the frame it repeats.
pub fn corrections(m: &Measurement, cut: &[bool], dup: &[bool], sigma: f64, limit: f64) -> Vec<[f64; 2]> {
    let n = m.motion.len();
    let mut corr = vec![[0.0; 2]; n];
    let mut starts: Vec<usize> = (0..n).filter(|&t| cut[t]).collect();
    starts.push(n);
    for w in starts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let mut r = vec![[0.0; 2]; b - a];
        for j in 1..b - a {
            r[j] = [r[j - 1][0] + m.motion[a + j][0], r[j - 1][1] + m.motion[a + j][1]];
        }
        let real: Vec<usize> = std::iter::once(0).chain((1..b - a).filter(|&j| !dup[a + j])).collect();
        let rr: Vec<[f64; 2]> = real.iter().map(|&j| r[j]).collect();
        let g = gauss(&rr, sigma);
        let mut ri = 0;
        for j in 0..b - a {
            while ri + 1 < real.len() && real[ri + 1] <= j {
                ri += 1;
            }
            corr[a + j] = [(g[ri][0] - rr[ri][0]).clamp(-limit, limit), (g[ri][1] - rr[ri][1]).clamp(-limit, limit)];
        }
    }
    corr
}

/// Remaining shake in real film frames: (median, 95th percentile) px on the LW scale.
pub fn shake(m: &Measurement, cut: &[bool], dup: &[bool], sigma: f64) -> (f64, f64) {
    let k = corrections(m, cut, dup, sigma, 1e9);
    let mut d: Vec<f64> = (0..k.len()).filter(|&t| !dup[t]).map(|t| (k[t][0].powi(2) + k[t][1].powi(2)).sqrt()).collect();
    if d.is_empty() {
        return (0.0, 0.0);
    }
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = 0.95 * (d.len() - 1) as f64;
    let (i, f) = (p.floor() as usize, p - p.floor());
    let p95 = if i + 1 < d.len() { d[i] * (1.0 - f) + d[i + 1] * f } else { d[i] };
    (median(&d), p95)
}

/// Shifts the picture by (dx, dy) bilinearly; the uncovered edges become black.
fn shift_frame(fr: &[u8], w: usize, h: usize, dx: f64, dy: f64) -> Vec<u8> {
    let mut out = vec![0u8; fr.len()];
    out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let sy = y as f64 - dy;
        let y0 = sy.floor();
        let fy = sy - y0;
        let y0 = y0 as i64;
        for x in 0..w {
            let sx = x as f64 - dx;
            let x0 = sx.floor();
            let fx = sx - x0;
            let x0 = x0 as i64;
            let mut acc = [0f64; 3];
            for (yy, wy) in [(y0, 1.0 - fy), (y0 + 1, fy)] {
                for (xx, wx) in [(x0, 1.0 - fx), (x0 + 1, fx)] {
                    if yy < 0 || yy >= h as i64 || xx < 0 || xx >= w as i64 {
                        continue; // outside = black
                    }
                    let p = (yy as usize * w + xx as usize) * 3;
                    for c in 0..3 {
                        acc[c] += wy * wx * fr[p + c] as f64;
                    }
                }
            }
            for c in 0..3 {
                row[x * 3 + c] = acc[c].round().clamp(0.0, 255.0) as u8;
            }
        }
    });
    out
}

pub struct StabilizeResult {
    pub scenes: usize,
    pub duplicate_share: f64,
    /// shake before stabilization: (median, 95 %) px at the video size
    pub before: (f64, f64),
}

/// Measures, reports and (when `out` is given) renders the stabilized video.
pub fn stabilize(v: &VideoInfo, a: &StabilizeSettings, out: Option<&str>) -> Result<StabilizeResult> {
    let m = measure(v)?;
    let cut = scene_cuts(&m, a.scene_threshold);
    let dup = duplicates(&m);
    let limit = a.max_shift / 100.0 * LW as f64;
    let corr = corrections(&m, &cut, &dup, a.smoothing, limit);
    let s = v.w as f64 / LW as f64;
    let (med, p95) = shake(&m, &cut, &dup, a.smoothing);
    let res = StabilizeResult {
        scenes: cut.iter().filter(|&&c| c).count(),
        duplicate_share: dup.iter().filter(|&&c| c).count() as f64 / dup.len().max(1) as f64,
        before: (med * s, p95 * s),
    };
    crate::log(&format!(
        "Scenes: {}, duplicate frames {:.0} %. Shake: median {:.1} px, below {:.1} px in 95 % of film frames ({}x{} picture).",
        res.scenes,
        100.0 * res.duplicate_share,
        res.before.0,
        res.before.1,
        v.w,
        v.h
    ));
    let Some(out) = out else {
        return Ok(res);
    };
    let size = v.w * v.h * 3;
    let mut r = Reader::new(&v.path, None, "rgb24", size, None, None)?;
    let mut w = Writer::new(out, v, a.crf, &a.preset, v.audio.then_some(v.path.as_str()))?;
    let mut i = 0;
    while let Some(fr) = r.next_frame() {
        crate::check_stop()?;
        let (dx, dy) = if i < corr.len() { (corr[i][0] * s, corr[i][1] * s) } else { (0.0, 0.0) };
        if dx.abs() < 0.05 && dy.abs() < 0.05 {
            w.write(&fr)?;
        } else {
            w.write(&shift_frame(&fr, v.w, v.h, dx, dy))?;
        }
        i += 1;
        if i % 50 == 0 {
            crate::progress(&format!("Rendering the stabilized video: {i} / {} frames", v.n), i, v.n.max(i));
        }
    }
    w.finish()?;
    crate::log(&format!("Done: {out} ({i} frames)"));
    Ok(res)
}
