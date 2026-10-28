//! Automatic seam refinement. The frame-line tracker can jump onto a strong horizontal edge in the
//! picture (a horizon, a rock edge) for some frames of a run, which shifts those repaired frames too
//! much or too little. A real slip changes slowly, so for every frame the shifts of its neighbours
//! in the run are tried as alternatives: each candidate is judged by how well the larger part of
//! the repaired frame matches the previous and the next frame (on a small gray copy). The tracked
//! shift is kept unless a neighbour's shift is clearly better. Only shifts that the tracker found
//! somewhere in the run are used, so camera movement cannot accumulate into the result.

use anyhow::Result;
use rayon::prelude::*;

use crate::video::{Reader, VideoInfo};

const SW: usize = 240; // width of the small copy
const MARGIN: usize = 3; // rows skipped next to the seam and the picture edges (small scale)
const ACCEPT: f32 = 0.8; // a candidate must lower the difference to this share
const MAX_DIFF: f32 = 25.0; // above this mean difference (scene cut, blur) nothing is changed
const MAX_DX: i64 = 8; // horizontal weave searched (small scale; 8 = 1/30 of the width)
const REACH: usize = 3; // neighbours on each side whose shifts are tried

struct Small {
    g: Vec<f32>,
    w: usize,
    h: usize,
}

impl Small {
    /// Value of the frame rolled down by `s` rows at (x, y).
    #[inline]
    fn rolled(&self, s: usize, x: usize, y: usize) -> f32 {
        self.g[((y + self.h - s) % self.h) * self.w + x]
    }
}

/// Rows of the larger part of a frame rolled down by `s` (the part that keeps the picture), with a margin.
fn big_part(s: usize, h: usize) -> (usize, usize) {
    if s == 0 {
        (MARGIN, h - MARGIN)
    } else if s < h - s {
        (s + MARGIN, h - MARGIN)
    } else {
        (MARGIN, s.saturating_sub(MARGIN))
    }
}

/// Mean difference of `cur` rolled by `s` against `other` rolled by `os`, over the rows both keep,
/// with a small search for vertical and horizontal weave between the film frames.
fn difference(cur: &Small, s: usize, other: &Small, os: usize) -> f32 {
    let (a0, a1) = big_part(s, cur.h);
    let (b0, b1) = big_part(os, cur.h);
    let (y0, y1) = (a0.max(b0) + 2, a1.min(b1).saturating_sub(2));
    if y1 <= y0 + 8 {
        return f32::INFINITY;
    }
    let (w, h) = (cur.w, cur.h);
    let m = MAX_DX as usize;
    let mut best = f32::INFINITY;
    for dy in -1i64..=1 {
        for dx in -MAX_DX..=MAX_DX {
            let mut e = 0f32;
            let mut n = 0usize;
            for y in (y0..y1).step_by(2) {
                let py = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                for x in (m..w - m).step_by(2) {
                    e += (cur.rolled(s, x, y) - other.rolled(os, (x as i64 + dx) as usize, py)).abs();
                    n += 1;
                }
            }
            best = best.min(e / n.max(1) as f32);
        }
    }
    best
}

/// `frames[0]` is the frame before the run (or the first run frame when the run starts the video),
/// then the run frames, then the frame after the run when there is one; `base[i]` is the tracked
/// shift of `frames[i]` in small rows (0 outside the run). Returns the change per run frame.
fn refine_run(frames: &[Small], base: &[usize], first: usize, n: usize) -> Vec<i64> {
    let h = frames[0].h;
    let mut fin: Vec<usize> = base.to_vec();
    let mut out = Vec::with_capacity(n);
    let near = |a: usize, b: usize| {
        let d = a.abs_diff(b);
        d.min(h - d) <= 1
    };
    for i in first..first + n {
        let s0 = base[i];
        let score = |s: usize, fin: &[usize]| -> f32 {
            let mut v = Vec::with_capacity(2);
            if i > 0 {
                v.push(difference(&frames[i], s, &frames[i - 1], fin[i - 1]));
            }
            if i + 1 < frames.len() {
                v.push(difference(&frames[i], s, &frames[i + 1], base[i + 1]));
            }
            let v: Vec<f32> = v.into_iter().filter(|e| e.is_finite()).collect();
            if v.is_empty() { f32::INFINITY } else { v.iter().sum::<f32>() / v.len() as f32 }
        };
        let mut cands: Vec<usize> = Vec::new();
        for j in i.saturating_sub(REACH).max(first)..(i + REACH + 1).min(first + n) {
            let c = if j < i { fin[j] } else { base[j] };
            if j != i && c != 0 && !near(c, s0) && !cands.iter().any(|&x| near(x, c)) {
                cands.push(c);
            }
        }
        let mut d = 0i64;
        if !cands.is_empty() {
            let e0 = score(s0, &fin);
            let (be, bc) = cands.iter().map(|&c| (score(c, &fin), c)).fold((f32::INFINITY, s0), |a, b| if b.0 < a.0 { b } else { a });
            if be < ACCEPT * e0 && be < MAX_DIFF {
                let mut dd = bc as i64 - s0 as i64;
                if dd > h as i64 / 2 {
                    dd -= h as i64;
                } else if dd < -(h as i64) / 2 {
                    dd += h as i64;
                }
                d = dd;
                fin[i] = bc;
            }
        }
        out.push(d);
    }
    out
}

/// Adjustment in video rows for every frame of every run (`runs`: (start, end, tracked shift in
/// video rows per frame)). Frames are read run by run with seeking.
pub fn refine(v: &VideoInfo, runs: &[(usize, usize, Vec<usize>)]) -> Result<Vec<Vec<i64>>> {
    let sh = ((v.h * SW) as f64 / v.w as f64 / 2.0).round() as usize * 2;
    let filter = format!("scale={SW}:{sh}:flags=area");
    let scale = sh as f64 / v.h as f64;
    let mut loaded: Vec<(Vec<Small>, Vec<usize>, usize)> = Vec::with_capacity(runs.len());
    for (i, (a, b, shifts)) in runs.iter().enumerate() {
        crate::check_stop()?;
        let start = a.saturating_sub(1);
        let mut r = Reader::new(&v.path, Some(&filter), "gray", SW * sh, Some(start as f64 / v.fps), Some(b + 1 - start))?;
        let mut frames = Vec::with_capacity(b + 1 - start);
        while let Some(f) = r.next_frame() {
            frames.push(Small { g: f.iter().map(|&x| x as f32).collect(), w: SW, h: sh });
        }
        let first = a - start;
        let mut base = vec![0usize; frames.len()];
        for (k, &s) in shifts.iter().enumerate() {
            if first + k < base.len() {
                base[first + k] = ((s as f64 * scale).round() as usize) % sh;
            }
        }
        loaded.push((frames, base, first));
        if i % 10 == 0 {
            crate::progress(&format!("Seam refinement: reading run {} / {}", i + 1, runs.len()), i + 1, runs.len());
        }
    }
    let out: Vec<Vec<i64>> = loaded
        .par_iter()
        .zip(runs.par_iter())
        .map(|((frames, base, first), (a, b, _))| {
            let n = b - a;
            if frames.len() < first + n || frames.len() < 2 {
                return vec![0; n];
            }
            refine_run(frames, base, *first, n).into_iter().map(|d| (d as f64 / scale).round() as i64).collect()
        })
        .collect();
    Ok(out)
}
