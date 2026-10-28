"""Film scanning error emulator: turns any pictures or videos into a "badly scanned small-gauge film".

Produces three files:
  <out>.mp4             the faulty video (25 fps): frame-line slips, weave, duplicate frames, film look
  <out>_reference.mp4   the same without slips (the ideal repair result)
  <out>_truth.npz       exact information for every video frame: frame-line row, slip, weave, duplicate, scene

This way the repair and stabilization accuracy can be measured and the slip classifier trained without
anybody's own home movies.

Model: the film is a strip where every film frame is PIC_H rows of picture + a dark frame line. The
scanner sees a window that is slightly taller than a film frame (the frame line shows at the edge and
a few rows of the next frame behind it). 18 fps film is scanned to 25 fps video, so some frames repeat.
In a slip the window slides along the strip: the frame line moves into the picture.

Every run (seed) draws its own film and scanner: frame-line thickness and tone, perforation hole,
window over-height, colours, grain, vignette, sharpness and compression. So the model learns the slip
itself and not the style of one emulator.

Usage:
    python emulate.py --images C:\\Windows\\Web\\Wallpaper -o test --duration 120
    python emulate.py --videos sources -o emu5 --duration 90 --seed 5 --no-reference
(--videos: a folder or files of film transfers without slips; when a repair report of Super8Fixer.exe
 (<video>_repair_report.txt, from `Super8Fixer.exe repair VIDEO --analyze-only --no-model`) is next to
 a video, its false alarms are used more often as hard examples)
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import find_ffmpeg, seconds, video_info  # noqa: E402

W, H = 1440, 1080          # video frame
OVER = 6                   # the scan window is this many rows taller than a film frame (drawn per seed; < 0 = shorter)
LINE = 14                  # dark frame-line thickness (drawn per seed)
P = H - OVER               # film frame height on the strip (picture + frame line)
PIC_H = P - LINE           # picture height inside a film frame
FILM_FPS, VIDEO_FPS = 18, 25


def draw_geometry(rng):
    global OVER, LINE, P, PIC_H
    OVER = int(rng.integers(-60, 30))       # negative: the scanner crops the frame line out of normal frames
    LINE = int(rng.integers(18, 46))
    P = H - OVER
    PIC_H = P - LINE


# ---------------------------------------------------------------- source: camera moves over still images

def image_scenes(images, duration, rng):
    """List of scenes: (image, start crop, end crop, film frames). Crop = (cx, cy, width) 0..1."""
    out, n = [], 0
    while n < duration * FILM_FPS:
        k = int(rng.uniform(2.5, 8) * FILM_FPS)
        z0, z1 = rng.uniform(0.5, 0.95, 2)
        a = (rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7), z0)
        b = (np.clip(a[0] + rng.normal(0, 0.15), 0.3, 0.7), np.clip(a[1] + rng.normal(0, 0.08), 0.3, 0.7), z1)
        out.append((images[rng.integers(len(images))], a, b, k))
        n += k
    return out


def image_frames(scenes, rng):
    """Generator: (film-frame picture PIC_H x W x 3, scene number). Includes hand-held shake."""
    from PIL import Image
    for sn, (path, a, b, k) in enumerate(scenes):
        img = np.asarray(Image.open(path).convert("RGB"))
        ih, iw = img.shape[:2]
        drift = np.cumsum(rng.normal(0, 0.002, (k, 2)), 0)            # slow hand-held drift
        for j in range(k):
            u = j / max(k - 1, 1)
            cx, cy, z = (1 - u) * np.array(a) + u * np.array(b)
            cx, cy = cx + drift[j, 0], cy + drift[j, 1]
            cw = z * iw
            ch = cw * PIC_H / W
            if ch > ih:
                ch = ih
                cw = ch * W / PIC_H
            x0 = int(np.clip(cx * iw - cw / 2, 0, iw - cw))
            y0 = int(np.clip(cy * ih - ch / 2, 0, ih - ch))
            piece = Image.fromarray(np.ascontiguousarray(img[y0:y0 + int(ch), x0:x0 + int(cw)]))
            yield np.asarray(piece.resize((W, PIC_H), Image.BOX)), sn


# ---------------------------------------------------------------- source: clips of real film transfers

def video_scenes(ff, videos, duration, rng, hard_share=0.4):
    """List of scenes: (video, start_s, film frames, fps). Every video frame is one film frame.
    When a repair report is next to a video, `hard_share` of the scenes are taken from the places it
    marked for repair: the source has no slips, so those are false alarms (horizons, edges) and hence
    good hard negative examples."""
    info, hard = [], []
    for v in videos:
        _, _, fps, secs, _ = video_info(ff, v)
        info.append((v, fps, secs))
        rep = os.path.splitext(v)[0] + "_repair_report.txt"
        if os.path.exists(rep):
            for row in open(rep, encoding="utf-8").read().splitlines()[1:]:
                o = row.split()
                if len(o) > 2 and "REPAIRED" in row:
                    hard.append((v, fps, seconds(o[0]), seconds(o[1])))
    out, n = [], 0
    while n < duration * FILM_FPS:
        k = int(rng.uniform(2.5, 8) * FILM_FPS)
        if hard and rng.random() < hard_share:
            v, fps, x, y = hard[rng.integers(len(hard))]
            a = max(0.0, rng.uniform(x, max(x, y - 1)) - rng.uniform(0, 1))
        else:
            v, fps, secs = info[rng.integers(len(info))]
            a = rng.uniform(5, max(6, secs - k / fps - 5))
        out.append((v, a, k, fps))
        n += k
    return out


def video_frames(ff, scenes, rng):
    """Generator like image_frames, but the frames come from videos (the edges are cropped away)."""
    size = W * PIC_H * 3
    for sn, (v, a, k, fps) in enumerate(scenes):
        r = rng.uniform(0.02, 0.08)                                     # crop: edges and possible borders away
        p = subprocess.Popen([ff, "-v", "error", "-ss", f"{a:.3f}", "-i", v, "-frames:v", str(k), "-vf",
                              f"crop=iw*{1 - 2 * r:.3f}:ih*{1 - 2 * r:.3f},scale={W}:{PIC_H}:flags=bicubic",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=size)
        last, levels = None, None
        low_t, high_t = rng.uniform(0, 25), rng.uniform(205, 255)       # scanner auto exposure: level stretch
        for _ in range(k):
            b = p.stdout.read(size)
            if len(b) == size:
                last = np.frombuffer(b, np.uint8).reshape(PIC_H, W, 3)
                if levels is None:
                    lo, hi = np.percentile(last[::8, ::8], [1, 99])
                    levels = (lo, max(hi - lo, 40.0))
                last = np.clip((last - levels[0]) / levels[1] * (high_t - low_t) + low_t, 0, 255).astype(np.uint8)
            if last is None:
                last = np.zeros((PIC_H, W, 3), np.uint8)
            yield last, sn
        p.kill()


# ---------------------------------------------------------------- film look

class Film:
    def __init__(s, rng, real_film=False):
        """real_film: the source already is film (own grain and colours), so less is added."""
        s.rng = rng
        yy, xx = np.mgrid[0:PIC_H, 0:W]
        r = np.hypot((xx - W / 2) / (W / 2), (yy - PIC_H / 2) / (PIC_H / 2))
        s.vignette = (1 - rng.uniform(0, 0.4) * np.clip(r - 0.4, 0, 1) ** 1.5)[..., None].astype(np.float32)
        s.tint = (np.array([1.05, 1.0, 0.85]) ** rng.uniform(-0.5, 2)).astype(np.float32)
        s.saturation = rng.uniform(0.85, 1.1) if real_film else rng.uniform(0.5, 0.9)
        s.grain = rng.uniform(0, 4) if real_film else rng.uniform(2, 10)
        s.gamma = rng.uniform(0.55, 1.0)                                   # scanner exposure: dark films get brighter
        s.dust = rng.uniform(0.5, 3)
        s.light = 1.0
        s.scratch = None
        # frame line: tone, possible colour cast and soft edges
        s.line_tone = rng.uniform(3, 45) * (1 + rng.normal(0, 0.15, 3)).clip(0.6, 1.4)
        s.line_softness = int(rng.integers(2, 11))
        # perforation hole
        s.hole = None
        if rng.random() < 0.35:
            width, height = int(rng.integers(15, 70)), int(rng.integers(30, 140))
            left = rng.random() < 0.75
            centre = PIC_H + LINE // 2 + int(rng.integers(-40, 41))
            s.hole = (left, width, centre - height // 2, centre + height // 2, rng.uniform(60, 255),
                      int(rng.integers(0, 12)))

    def frame(s, pic):
        """A film frame on the strip: picture + dark frame line + perforation hole, P x W x 3."""
        rng = s.rng
        x = 255 * (pic.astype(np.float32) / 255) ** s.gamma
        gray = x.mean(2, keepdims=True)
        x = gray + (x - gray) * s.saturation
        s.light = 0.8 * s.light + 0.2 * (1 + rng.normal(0, 0.06))          # flicker
        x = x * s.tint * s.vignette * s.light
        if s.grain:
            x += rng.normal(0, s.grain, x.shape[:2])[..., None]
        for _ in range(rng.poisson(s.dust)):                               # dust
            cy, cx, r = rng.integers(0, PIC_H), rng.integers(0, W), rng.integers(2, 7)
            x[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r] *= 0.25
        if s.scratch is None and rng.random() < 0.01:                       # a scratch lasts a while
            s.scratch = [rng.integers(50, W - 50), rng.integers(10, 60)]
        if s.scratch:
            x[:, s.scratch[0]:s.scratch[0] + 2] = x[:, s.scratch[0]:s.scratch[0] + 2] * 0.5 + 110
            s.scratch[0] += rng.integers(-1, 2)
            s.scratch[1] -= 1
            if s.scratch[1] <= 0:
                s.scratch = None
        fr = np.zeros((P, W, 3), np.float32)
        fr[:PIC_H] = x
        fr[PIC_H:] = s.line_tone * rng.uniform(0.85, 1.15)                  # dark frame line
        if s.line_softness:                                                 # the frame-line edges blend into the picture
            for bottom in (PIC_H, P):
                a, b = bottom - s.line_softness, bottom + s.line_softness
                top_row, bottom_row = fr[a % P].copy(), fr[(b - 1) % P].copy()
                for k, y in enumerate(range(a, b)):
                    u = (k + 0.5) / (b - a)
                    fr[y % P] = (1 - u) * top_row + u * bottom_row
        if s.hole:
            left, wid, y0, y1, brightness, rounding = s.hole
            for y in range(y0, y1):
                narrow = max(0, rounding - min(y - y0, y1 - 1 - y))
                if left:
                    fr[y % P, :max(wid - narrow, 1)] = brightness
                else:
                    fr[y % P, W - wid + narrow:] = brightness
        return np.clip(fr, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- slips and weave

def slips(n, rng):
    """Slip (rows) for every video frame + event type: 0 none, 1 slide, 2 roll, 3 small."""
    s = np.zeros(n, np.float32)
    kind = np.zeros(n, np.int8)
    t = int(rng.uniform(2, 6) * VIDEO_FPS)
    while t < n:
        k = rng.choice([1, 2, 3], p=[0.4, 0.3, 0.3])
        length = int(rng.uniform(1, 4) * VIDEO_FPS)
        b = min(n, t + length)
        if k == 1:
            D = rng.uniform(60, P - 60)
            rise = int(0.5 * VIDEO_FPS)
            s[t:b] = D * np.clip((np.arange(b - t) + 1) / rise, 0, 1)
        elif k == 2:
            v = rng.uniform(15, 120) * rng.choice([-1, 1])
            s[t:b] = (np.arange(b - t) * v) % P
        else:
            s[t:b] = rng.choice([rng.uniform(15, 50), P - rng.uniform(15, 50)])
        kind[t:b] = k
        t = b + int(rng.uniform(4, 12) * VIDEO_FPS)
    return s, kind


def weave(n_film, rng):
    """Side and vertical weave of the film frames in the gate (px), an AR process."""
    h = np.zeros((n_film, 2), np.float32)
    strength = rng.uniform(0.3, 1.5)
    for f in range(1, n_film):
        h[f] = 0.6 * h[f - 1] + rng.normal(0, [3.0 * strength, 2.0 * strength])
    return h


# ---------------------------------------------------------------- scanning

def window(strip, O, dx):
    """The scan window at strip position O (rows), side shift dx. strip(f) -> film frame f."""
    out = np.empty((H, W, 3), np.uint8)
    y = 0
    O = int(round(O))
    while y < H:
        pos = O + y
        f, r = int(pos // P), int(pos % P)
        k = min(P - r, H - y)
        out[y:y + k] = strip(f)[r:r + k]
        y += k
    if dx:
        out = np.roll(out, dx, axis=1)
        if dx > 0:
            out[:, :dx] = 0
        else:
            out[:, dx:] = 0
    return out


class Scanner:
    """Traces of the scanner and compression: softness and possibly a lower resolution."""
    def __init__(s, rng):
        s.blur = rng.choice([0, rng.uniform(0.5, 2.5)], p=[0.3, 0.7])
        s.scale = rng.choice([1.0, rng.uniform(0.35, 0.8)], p=[0.5, 0.5])

    def __call__(s, img):
        from PIL import Image, ImageFilter
        if s.scale == 1 and not s.blur:
            return img
        k = Image.fromarray(img)
        if s.scale < 1:
            k = k.resize((int(W * s.scale), int(H * s.scale)), Image.BOX).resize((W, H), Image.BICUBIC)
        if s.blur:
            k = k.filter(ImageFilter.GaussianBlur(s.blur))
        return np.asarray(k)


def main():
    ap = argparse.ArgumentParser(description="Emulate small-gauge film scanning errors into a test video.")
    ap.add_argument("--images", help="folder whose pictures (jpg/png) become camera moves")
    ap.add_argument("--videos", nargs="+", help="video files or a folder to take clips from")
    ap.add_argument("-o", "--out", default="emulated")
    ap.add_argument("--duration", type=float, default=120, help="seconds")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-reference", action="store_true", help="do not write _reference.mp4 (enough for training)")
    ap.add_argument("--ffmpeg")
    a = ap.parse_args()
    if not (a.images or a.videos):
        sys.exit("Give --images or --videos.")
    rng = np.random.default_rng(a.seed)
    ff = find_ffmpeg(a.ffmpeg)
    draw_geometry(rng)
    if a.videos:
        videos = []
        for v in a.videos:
            videos += sorted(glob.glob(os.path.join(v, "*.mp4"))) if os.path.isdir(v) else [v]
        scenes = video_scenes(ff, videos, a.duration, rng)
        source = video_frames(ff, scenes, rng)
        n_film = sum(s[2] for s in scenes)
    else:
        images = sorted(p for e in ("jpg", "jpeg", "png")
                        for p in glob.glob(os.path.join(a.images, "**", "*." + e), recursive=True))
        if not images:
            sys.exit("No pictures found in the folder: " + a.images)
        scenes = image_scenes(images, a.duration, rng)
        source = image_frames(scenes, rng)
        n_film = sum(s[3] for s in scenes)
    n = int(n_film * VIDEO_FPS / FILM_FPS)
    film_index = np.minimum((np.arange(n) * FILM_FPS) // VIDEO_FPS, n_film - 1)   # 18 -> 25 fps repeat
    s, kind = slips(n, rng)
    wv = weave(n_film, rng)
    phase = rng.uniform(-LINE - 10, max(OVER, 0) + 6)          # the normal frame-line position depends on the scanner setup

    film = Film(rng, real_film=bool(a.videos))
    scanner = Scanner(rng)
    crf = int(rng.integers(14, 29))
    ready, scene, next_f = {}, np.zeros(n_film, np.int32), [0]

    def strip(f):
        f = min(max(f, 0), n_film - 1)
        while next_f[0] <= f:                              # film frames are made in order
            pic, sn = next(source)
            ready[next_f[0]] = film.frame(pic)
            scene[next_f[0]] = sn
            next_f[0] += 1
        return ready[f]

    def encoder(name):
        return subprocess.Popen([ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                 "-r", str(VIDEO_FPS), "-i", "-", "-c:v", "libx264", "-crf", str(crf),
                                 "-preset", "veryfast", "-pix_fmt", "yuv420p", name], stdin=subprocess.PIPE)
    e1 = encoder(a.out + ".mp4")
    e2 = None if a.no_reference else encoder(a.out + "_reference.mp4")
    line_row = np.zeros(n, np.float32)
    for t in range(n):
        f = int(film_index[t])
        dx, dy = int(round(wv[f, 0])), wv[f, 1]
        O0 = f * P + dy + phase
        e1.stdin.write(scanner(window(strip, O0 + s[t], dx)).tobytes())
        if e2:
            e2.stdin.write(scanner(window(strip, O0, dx)).tobytes())
        line_row[t] = (PIC_H - s[t] - dy - phase) % P     # top row of the frame line in the video frame
        for old in [k for k in ready if k < f - 2]:       # memory
            del ready[old]
        if t % 250 == 0:
            print(f"\rEmulating: {t}/{n}", end="", flush=True)
    e1.stdin.close()
    e1.wait()
    if e2:
        e2.stdin.close()
        e2.wait()
    np.savez(a.out + "_truth.npz", slip=s, kind=kind, line_row=line_row,
             film_frame=film_index, duplicate=np.r_[False, film_index[1:] == film_index[:-1]],
             weave=wv[film_index], scene=scene[film_index], P=P, H=H, line=LINE)
    print(f"\nDone: {a.out}.mp4 ({n} frames, slip frames {(kind > 0).sum()}; "
          f"frame line {LINE} px, over-height {OVER} px, crf {crf})")


if __name__ == "__main__":
    main()
