//! Rebuilding a slipped frame: roll, fetch the smaller part from a neighbouring frame, align it,
//! match its brightness, blend the seam and fill a leftover dark frame-line band.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const MAX_DY: i64 = 16; // vertical fine adjustment at the seam (px)
const MAX_DX: i64 = 24; // horizontal weave between film frames (px)
const FINE_DX: i64 = 6; // exact horizontal search around the coarse result (px)
const BAND: i64 = 96; // rows of the kept part used to measure the weave of an intact neighbour
pub const NEIGHBOURS: usize = 2; // the smaller part is fetched from at most this many video frames away
const BAND_RANGE: i64 = 28; // a dark frame-line band is searched this close to the seam (px)
const SEAM_MAX: f32 = 40.0; // largest seam blend width (px)

/// Where the missing part of a repaired frame comes from.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Fill {
    /// best match among the frame itself and its neighbours
    #[default]
    Auto,
    /// only earlier frames
    Previous,
    /// only later frames
    Next,
    /// no rebuilding: the frame is only rolled
    RollOnly,
}

impl Fill {
    pub const ALL: [Fill; 4] = [Fill::Auto, Fill::Previous, Fill::Next, Fill::RollOnly];
    pub fn label(self) -> &'static str {
        match self {
            Fill::Auto => "Automatic",
            Fill::Previous => "Previous frame",
            Fill::Next => "Next frame",
            Fill::RollOnly => "Roll only",
        }
    }
}

/// What to do with one frame: shift 0 = leave as is.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameJob {
    pub shift: usize,
    pub fill: Fill,
    /// manual alignment correction of the filled part (dy, dx) in pixels
    pub nudge: (i64, i64),
}

/// An RGB image, row by row.
#[derive(Clone, Copy)]
pub struct Size {
    pub w: usize,
    pub h: usize,
}

fn roll(fr: &[u8], k: Size, s: i64) -> Vec<u8> {
    let row = k.w * 3;
    let mut out = vec![0u8; fr.len()];
    for y in 0..k.h {
        let l = (y as i64 - s).rem_euclid(k.h as i64) as usize;
        out[y * row..(y + 1) * row].copy_from_slice(&fr[l * row..(l + 1) * row]);
    }
    out
}

/// Gray of every fourth column.
fn gray(img: &[u8], k: Size) -> (Vec<f32>, usize) {
    let wc = k.w.div_ceil(4);
    let mut g = vec![0f32; k.h * wc];
    for y in 0..k.h {
        for (j, x) in (0..k.w).step_by(4).enumerate() {
            let p = (y * k.w + x) * 3;
            g[y * wc + j] = (img[p] as f32 + img[p + 1] as f32 + img[p + 2] as f32) / 3.0;
        }
    }
    (g, wc)
}

/// Finds a full-width dark band (remains of a frame line) on rows y0..y1: (b0, b1) or None.
/// The band has to be darker than its surroundings in nearly every column block.
fn dark_band(img: &[u8], k: Size, y0: i64, y1: i64) -> Option<(i64, i64)> {
    let h = k.h as i64;
    let (y0, y1) = (y0.max(12), y1.min(h - 13));
    if y1 - y0 < 4 {
        return None;
    }
    let m = k.w / 10;
    let bw = (k.w - 2 * m) / 16;
    let blocks = |y: i64| -> [f32; 16] {
        let mut o = [0f32; 16];
        let r = &img[y as usize * k.w * 3..];
        for (b, ob) in o.iter_mut().enumerate() {
            let mut s = 0f64;
            for x in m + b * bw..m + (b + 1) * bw {
                s += (r[x * 3] as f64 + r[x * 3 + 1] as f64 + r[x * 3 + 2] as f64) / 3.0;
            }
            *ob = (s / bw as f64) as f32;
        }
        o
    };
    let around: Vec<[f32; 16]> = (y0 - 12..y0).chain(y1 + 1..y1 + 13).map(blocks).collect();
    let mut refv = [0f32; 16];
    for (b, r) in refv.iter_mut().enumerate() {
        let mut v: Vec<f32> = around.iter().map(|l| l[b]).collect();
        v.sort_by(|a, c| a.partial_cmp(c).unwrap());
        let n = v.len();
        *r = if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 };
    }
    if refv.iter().sum::<f32>() / 16.0 < 40.0 {
        return None; // in a dark picture the band does not show and does no harm
    }
    let dark: Vec<bool> = (y0..=y1)
        .map(|y| {
            let l = blocks(y);
            let n = (0..16).filter(|&b| l[b] < 0.75 * refv[b] - 6.0).count();
            n as f32 / 16.0 >= 0.8
        })
        .collect();
    let mut best: Option<(usize, usize)> = None;
    let mut start: Option<usize> = None;
    for (i, &t) in dark.iter().chain(std::iter::once(&false)).enumerate() {
        if t && start.is_none() {
            start = Some(i);
        } else if !t {
            if let Some(a) = start {
                if (2..=30).contains(&(i - a)) && best.is_none_or(|p| i - a > p.1 - p.0) {
                    best = Some((a, i));
                }
                start = None;
            }
        }
    }
    best.map(|(a, b)| (y0 + a as i64, y0 + b as i64 - 1))
}

/// Fills the band rows by vertical interpolation between the rows above and below.
fn fill_band(img: &mut [u8], k: Size, (b0, b1): (i64, i64)) {
    let row = k.w * 3;
    let a: Vec<f32> = img[(b0 - 2) as usize * row..(b0 - 1) as usize * row].iter().map(|&v| v as f32).collect();
    let b: Vec<f32> = img[(b1 + 2) as usize * row..(b1 + 3) as usize * row].iter().map(|&v| v as f32).collect();
    let n = (b1 - b0 + 5) as f32;
    for (j, y) in (b0 - 1..=b1 + 1).enumerate() {
        let w = (j + 1) as f32 / n;
        let r = &mut img[y as usize * row..(y as usize + 1) * row];
        for x in 0..row {
            r[x] = ((1.0 - w) * a[x] + w * b[x]) as u8;
        }
    }
}

/// Gray values of one full row.
fn gray_row(img: &[u8], k: Size, y: i64) -> Vec<f32> {
    img[y as usize * k.w * 3..(y as usize + 1) * k.w * 3].chunks_exact(3).map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0).collect()
}

/// Horizontal offset (px) of candidate row `y` that continues the rows `edge` (next to the seam,
/// then one further) best, searched at 1 px around the coarse result `dx0`.
fn fine_dx(cand: &[u8], cur: &[u8], k: Size, y: i64, edge: (i64, i64), dx0: i64) -> i64 {
    let row = gray_row(cand, k, y);
    let a1 = gray_row(cur, k, edge.0);
    let a2 = gray_row(cur, k, edge.1);
    let w = k.w as i64;
    let inner = MAX_DX as usize + FINE_DX as usize..k.w - MAX_DX as usize - FINE_DX as usize;
    let mut best = (f32::INFINITY, dx0);
    for dx in (dx0 - FINE_DX).max(-MAX_DX)..=(dx0 + FINE_DX).min(MAX_DX) {
        let mut e = 0f32;
        for x in inner.clone() {
            let r = row[(x as i64 - dx).rem_euclid(w) as usize];
            e += (r - a1[x]).abs() + (r - (2.0 * a1[x] - a2[x])).abs();
        }
        let e = e / (2 * inner.len()) as f32 + 0.01 * dx.abs() as f32;
        if e < best.0 {
            best = (e, dx);
        }
    }
    best.1
}

/// Horizontal offset (px) that aligns `rows + dy` of the candidate with `rows` of the current frame,
/// compared on horizontal gradients (insensitive to exposure differences between film frames).
fn band_dx(cand: &[u8], cur: &[u8], k: Size, rows: &[i64], dy: i64) -> i64 {
    let h = k.h as i64;
    let rows: Vec<i64> = rows.iter().copied().filter(|&y| y >= 0 && y < h && y + dy >= 0 && y + dy < h).step_by(2).collect();
    if rows.len() < 8 {
        return 0;
    }
    let grad = |r: Vec<f32>| -> Vec<f32> { r.windows(2).map(|p| p[1] - p[0]).collect() };
    let a: Vec<Vec<f32>> = rows.iter().map(|&y| grad(gray_row(cur, k, y))).collect();
    let b: Vec<Vec<f32>> = rows.iter().map(|&y| grad(gray_row(cand, k, y + dy))).collect();
    let n = k.w - 1;
    let inner = MAX_DX as usize + 1..n - MAX_DX as usize - 1;
    let mut best = (f32::INFINITY, 0);
    for dx in -MAX_DX..=MAX_DX {
        let mut e = 0f32;
        for (ra, rb) in a.iter().zip(&b) {
            for x in inner.clone() {
                e += (rb[(x as i64 - dx) as usize] - ra[x]).abs();
            }
        }
        if e < best.0 {
            best = (e, dx);
        }
    }
    best.1
}

struct Candidate<'a> {
    img: std::borrow::Cow<'a, [u8]>,
    /// shift of the candidate itself (0 = an intact frame)
    shift: i64,
    /// position relative to the frame being repaired (0 = the frame itself)
    offset: i64,
}

/// Rolls the frame down by `job.shift` rows and rebuilds it from one film frame: the larger part is
/// kept, the smaller one is taken from the video frame (itself or +-NEIGHBOURS; `neighbours` =
/// (frame, shift, offset)) that continues it best at the seam. Finally the brightness is matched,
/// the seam blended and a leftover frame-line band filled.
pub fn repair_frame(fr: &[u8], k: Size, job: FrameJob, neighbours: &[(Option<&[u8]>, usize, i64)]) -> Vec<u8> {
    let h = k.h as i64;
    let mut s = job.shift as i64;
    let mut out = roll(fr, k, s);
    if job.fill == Fill::RollOnly || !(12 < s && s < h - 12) {
        if let Some(r) = dark_band(&out, k, s - BAND_RANGE, s + BAND_RANGE) {
            fill_band(&mut out, k, r);
        }
        return out;
    }
    let allowed = |o: i64| match job.fill {
        Fill::Auto => true,
        Fill::Previous => o < 0,
        Fill::Next => o > 0,
        Fill::RollOnly => false,
    };
    let mut cand: Vec<Candidate> = Vec::new();
    if allowed(0) {
        cand.push(Candidate { img: std::borrow::Cow::Borrowed(&out[..]), shift: s, offset: 0 });
    }
    for &(n, ns, o) in neighbours {
        if let Some(n) = n {
            if allowed(o) {
                cand.push(Candidate { img: std::borrow::Cow::Owned(roll(n, k, ns as i64)), shift: ns as i64, offset: o });
            }
        }
    }
    if cand.is_empty() {
        // the chosen direction has no frames (start or end of the video): use the frame itself
        cand.push(Candidate { img: std::borrow::Cow::Borrowed(&out[..]), shift: s, offset: 0 });
    }
    let top_big = s >= h - s; // the top part (rows 0..s) is the larger one
    // a frame-line band at the edge of the larger part is not picture: the seam moves past it
    let s0 = s;
    match dark_band(&out, k, s - BAND_RANGE, s + BAND_RANGE) {
        Some((r0, _)) if top_big && r0 < s => s = r0,
        Some((_, r1)) if !top_big && r1 >= s => s = r1 + 1,
        _ => {}
    }
    let (g, wc) = gray(&out, k);
    let grow = |g: &[f32], y: i64| -> Vec<f32> { g[y as usize * wc..(y as usize + 1) * wc].to_vec() };
    let (a1, a2) = if top_big { (grow(&g, s - 1), grow(&g, s - 2)) } else { (grow(&g, s), grow(&g, s + 1)) };
    let predicted: Vec<f32> = a1.iter().zip(&a2).map(|(a, b)| 2.0 * a - b).collect();
    let m = MAX_DX / 4;
    let inner = m as usize..wc - m as usize; // compared without the edges that the horizontal shift wraps
    let ninner = inner.len() as f32;
    let mut best = (f32::INFINITY, 0usize, 0i64, 0i64);
    let maxdy = MAX_DY + (s0 - s).abs();
    for (ci, c) in cand.iter().enumerate() {
        let (gc, _) = gray(&c.img, k);
        for dy in -maxdy..=maxdy {
            if !(0 <= s - 1 + dy && s + dy < h) {
                continue;
            }
            // only rows on the correct side of the candidate's own seam (shift 0 = intact frame)
            if c.shift != 0 && ((top_big && s + dy < c.shift + 2) || (!top_big && s - 1 + dy > c.shift - 3)) {
                continue;
            }
            let y = if top_big { s + dy } else { s - 1 + dy };
            let row = &gc[y as usize * wc..(y as usize + 1) * wc];
            for dx in -m..=m {
                let (mut e1, mut e2) = (0f32, 0f32);
                for j in inner.clone() {
                    let r = row[(j as i64 - dx).rem_euclid(wc as i64) as usize];
                    e1 += (r - a1[j]).abs();
                    e2 += (r - predicted[j]).abs();
                }
                let e = (e1 / ninner + e2 / ninner) / 2.0
                    + (0.05 * ((dy.abs() + (4 * dx).abs()) as f64)) as f32
                    + if c.offset != 0 { 0.5 } else { 0.0 };
                if e < best.0 {
                    best = (e, ci, dy, 4 * dx);
                }
            }
        }
    }
    if !best.0.is_finite() {
        best = (SEAM_MAX, 0, 0, 0);
    } else {
        // the coarse search moved in steps of 4 px: find the exact horizontal offset at full
        // resolution, so that vertical structures (a tree trunk, a door frame) continue over the seam
        let c = &cand[best.1];
        if c.shift == 0 && c.offset != 0 {
            // an intact neighbour shows the same rows as the kept part: measure the horizontal weave
            // between the two film frames over a wide band of the kept part
            let band: Vec<i64> = if top_big { (s - BAND..s - 4).collect() } else { (s + 4..s + BAND).collect() };
            best.3 = band_dx(&c.img, &out, k, &band, best.2);
        } else {
            let y = if top_big { s + best.2 } else { s - 1 + best.2 };
            best.3 = fine_dx(&c.img, &out, k, y, if top_big { (s - 1, s - 2) } else { (s, s + 1) }, best.3);
        }
    }
    let (err, ci, dy, dx) = (best.0, best.1, best.2 + job.nudge.0, best.3 + job.nudge.1);
    let rows: Vec<i64> = if top_big { (s..h).collect() } else { (0..s).collect() };
    let src = &cand[ci].img;
    let (w, rl) = (k.w as i64, k.w * 3);
    let mut part = vec![0f32; rows.len() * rl];
    for (i, &r) in rows.iter().enumerate() {
        let ly = (r + dy).clamp(0, h - 1) as usize;
        for x in 0..k.w {
            let lx = (x as i64 - dx).rem_euclid(w) as usize;
            for c in 0..3 {
                part[i * rl + x * 3 + c] = src[ly * rl + lx * 3 + c] as f32;
            }
        }
    }
    // film frames differ in exposure: match the smaller part's brightness at the seam (per channel)
    let n = 12.min(s).min(h - s) as usize;
    let mean3 = |data: &[f32]| -> [f32; 3] {
        let mut m = [0f64; 3];
        for p in data.chunks_exact(3) {
            for c in 0..3 {
                m[c] += p[c] as f64;
            }
        }
        let cnt = (data.len() / 3) as f64;
        [(m[0] / cnt) as f32, (m[1] / cnt) as f32, (m[2] / cnt) as f32]
    };
    let out_f = |y0: usize, y1: usize| -> Vec<f32> { out[y0 * rl..y1 * rl].iter().map(|&v| v as f32).collect() };
    let (own, new) = if top_big {
        (mean3(&out_f(s as usize - n, s as usize)), mean3(&part[..n * rl]))
    } else {
        (mean3(&out_f(s as usize, s as usize + n)), mean3(&part[part.len() - n * rl..]))
    };
    let gain: Vec<f32> = (0..3).map(|c| ((own[c] + 4.0) / (new[c] + 4.0)).clamp(0.8, 1.25)).collect();
    let mut res = out.clone();
    for (i, &r) in rows.iter().enumerate() {
        let dst = &mut res[r as usize * rl..(r as usize + 1) * rl];
        for x in 0..rl {
            dst[x] = (part[i * rl + x] * gain[x % 3]).clamp(0.0, 255.0) as u8;
        }
    }
    // blend the seam: the worse the parts continue, the wider the blend
    let (g2, wc2) = gray(&res, k);
    let mut diffs: Vec<f32> = (0..k.h - 1)
        .map(|y| (0..wc2).map(|x| (g2[(y + 1) * wc2 + x] - g2[y * wc2 + x]).abs()).sum::<f32>() / wc2 as f32)
        .collect();
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let nd = diffs.len();
    let background = if nd % 2 == 1 { diffs[nd / 2] } else { (diffs[nd / 2 - 1] + diffs[nd / 2]) / 2.0 } + 1.0;
    let su = s as usize;
    let jump = (0..wc2).map(|x| (g2[su * wc2 + x] - g2[(su - 1) * wc2 + x]).abs()).sum::<f32>() / wc2 as f32 / background;
    let width = (err * 1.5).max(4.0 + 6.0 * (jump - 2.0)).clamp(4.0, SEAM_MAX) as usize;
    if top_big {
        let edge: Vec<f32> = res[(su - 1) * rl..su * rl].iter().map(|&v| v as f32).collect();
        for j in 0..width.min(k.h - su) {
            let wj = (j + 1) as f32 / (width + 1) as f32;
            let r = &mut res[(su + j) * rl..(su + j + 1) * rl];
            for x in 0..rl {
                r[x] = ((1.0 - wj) * edge[x] + wj * r[x] as f32) as u8;
            }
        }
    } else {
        let edge: Vec<f32> = res[su * rl..(su + 1) * rl].iter().map(|&v| v as f32).collect();
        for j in 0..width.min(su) {
            let wj = (j + 1) as f32 / (width + 1) as f32;
            let r = &mut res[(su - 1 - j) * rl..(su - j) * rl];
            for x in 0..rl {
                r[x] = ((1.0 - wj) * edge[x] + wj * r[x] as f32) as u8;
            }
        }
    }
    drop(cand);
    // if a band still came along with the candidate, fill it
    if let Some(r) = dark_band(&res, k, s - BAND_RANGE, s + BAND_RANGE) {
        fill_band(&mut res, k, r);
    }
    res
}

/// Repairs a batch of frames in parallel. `frames[i]` = video frame `start + i - NEIGHBOURS`
/// (None outside the video); returns the frames `start..start+count`.
pub fn repair_batch(frames: &[Option<Vec<u8>>], start: usize, count: usize, k: Size, jobs: &[FrameJob]) -> Vec<Vec<u8>> {
    let job = |j: i64| -> FrameJob { if j >= 0 && (j as usize) < jobs.len() { jobs[j as usize] } else { FrameJob::default() } };
    (0..count)
        .into_par_iter()
        .map(|d| {
            let i = start + d;
            let p = d + NEIGHBOURS;
            let cur = frames[p].as_ref().unwrap();
            let jb = job(i as i64);
            if jb.shift == 0 {
                return cur.clone();
            }
            let neighbours: Vec<(Option<&[u8]>, usize, i64)> = (-(NEIGHBOURS as i64)..=NEIGHBOURS as i64)
                .filter(|&e| e != 0)
                .map(|e| {
                    let q = p as i64 + e;
                    let f = if q >= 0 { frames.get(q as usize).and_then(|x| x.as_deref()) } else { None };
                    (f, job(i as i64 + e).shift, e)
                })
                .collect();
            repair_frame(cur, k, jb, &neighbours)
        })
        .collect()
}
