"""Train the shareable slip classifier on emulated videos only (emulate.py).

The model never sees anybody's own home movies, so it can be distributed with the application.

Usage:
    python emulate.py --videos sources -o emu1 --seed 1 --no-reference      (etc. with different seeds)
    python train_emulated.py emu1 emu2 emu3 -o slip_classifier.pt [--steps 3000]
    python export_onnx.py slip_classifier.pt ../Super8Fixer/models
(emuN = the -o name of emulate.py; emuN.mp4 and emuN_truth.npz are needed)
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slip_model as SM  # noqa: E402
from common import find_ffmpeg, video_info  # noqa: E402

EDGE_PX = 50          # a slip counts as real when the frame line is at least this far from the edges (1080 picture)


def load_truth(name):
    """(picture height, frame-line row per frame, slip kind per frame)"""
    f = name + "_truth.npz"
    if not os.path.exists(f):
        sys.exit("Truth file not found: " + f)
    T = np.load(f)
    return int(T["H"]), T["line_row"], T["kind"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emulated", nargs="+")
    ap.add_argument("-o", "--out", default="slip_classifier.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--slip-share", type=float, default=1.0,
                    help="share of positives taken from emulated slips; the rest are rolled normal frames")
    ap.add_argument("--negatives", nargs="*", default=[],
                    help="original slip-free film transfers: their frames are used as negatives as they are")
    ap.add_argument("--negatives-per-video", type=int, default=400)
    ap.add_argument("--ffmpeg")
    a = ap.parse_args()
    ff = find_ffmpeg(a.ffmpeg)
    normals, lines, slips = [], [], []
    for name in a.emulated:
        H, rr, kind = load_truth(name)
        norm = np.flatnonzero(kind == 0)[::2]
        real = np.flatnonzero((kind > 0) & (rr > EDGE_PX) & (rr < H - EDGE_PX))
        frames = SM.read_frames(ff, name + ".mp4", np.r_[norm, real])
        normals += [frames[t] for t in norm]
        lines += list(np.minimum(np.round(rr[norm] * SM.MH / H), SM.MH).astype(int))  # MH: frame line hidden
        visible = [t for t in real if frames[t].std() > 8]      # in a black frame (leader) a slip does not show
        slips += [frames[t] for t in visible]
        print(f"{name}: {len(norm)} normal, {len(visible)} slip frames")
    raw = []
    videos = []
    for v in a.negatives:                 # files, folders or wildcards (cmd does not expand them)
        videos += sorted(glob.glob(os.path.join(v, "*.mp4"))) if os.path.isdir(v) else sorted(glob.glob(v)) or [v]
    for v in videos:
        _, _, fps, secs, _ = video_info(ff, v)
        n = int(fps * secs)
        idx = np.linspace(int(fps * 5), max(n - int(fps * 5), int(fps * 5) + 1), a.negatives_per_video).astype(int)
        frames = SM.read_frames(ff, v, idx)
        raw += [f for f in frames.values() if f.std() > 8]
        print(f"{os.path.basename(v)}: {len(frames)} original frames as negatives")
    negatives = np.stack(raw) if raw else np.zeros((0, SM.MH, SM.MW, 3), np.uint8)
    SM.train(np.stack(normals), negatives, np.array(lines), a.steps, a.out,
             edge=7, slips=np.stack(slips), slip_share=a.slip_share, step_share=0.25)
    print("Model saved:", a.out)


if __name__ == "__main__":
    main()
