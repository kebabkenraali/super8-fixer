"""Fine-tune the slip classifier with your own corrections, so that the application learns from them.

When you review a film in Super8Fixer.exe and untick runs that are not slips (or tick runs it missed),
the decisions are kept in <video>.s8edits.json and in <video>_repair_report.txt. This script turns
them into training examples:
  - runs you unticked ("skipped (by user)")          -> negatives (false alarms)
  - runs you ticked yourself and repaired runs        -> positives (real slips)
  - frames outside every run                          -> negatives (normal frames)
and continues training from the shipped model with a small learning rate.

The result is learned from YOUR film, so keep it for yourself: it is written as
models\\slip_classifier_personal.onnx, which Super8Fixer.exe uses instead of the shipped model when
it exists. Delete the file to go back to the shipped model. Nothing is sent anywhere.

Usage:
    python finetune_from_edits.py film1.mp4 [film2.mp4 ...] --app ..\\Super8Fixer [--steps 800]
(PyTorch is needed: pip install torch onnx pillow numpy)
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slip_model as SM  # noqa: E402
from common import find_ffmpeg, seconds  # noqa: E402


def parse_report(video, fps):
    rep = os.path.splitext(video)[0] + "_repair_report.txt"
    if not os.path.exists(rep):
        sys.exit(f"Repair report not found: {rep}\nAnalyze the video in Super8Fixer.exe first.")
    runs = []
    for row in open(rep, encoding="utf-8").read().splitlines()[1:]:
        o = row.split()
        if len(o) < 6:
            continue
        # columns: start end frames confidence model line% action...
        a, b = int(round(seconds(o[0]) * fps)), int(round(seconds(o[1]) * fps))
        line = float(o[5]) / 100.0
        action = " ".join(o[6:])
        runs.append((a, b, line, action))
    return runs


def fps_of(ff, video):
    from common import video_info
    return video_info(ff, video)[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--app", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Super8Fixer"),
                    help="the Super8Fixer folder (its models folder gets the personal model)")
    ap.add_argument("--base", help="PyTorch model to start from (default: the shipped slip_classifier.pt next to this script)")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--ffmpeg")
    a = ap.parse_args()
    ff = find_ffmpeg(a.ffmpeg)
    base = a.base or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "slip_classifier.pt")
    if not os.path.exists(base):
        sys.exit(f"Base model not found: {base}")
    rng = np.random.default_rng(0)
    normals, lines, false_alarms, slips = [], [], [], []
    for v in a.videos:
        fps = fps_of(ff, v)
        runs = parse_report(v, fps)
        pos, neg = [], []
        for s, e, line, action in runs:
            idx = np.linspace(s, e - 1, min(12, max(e - s, 1))).astype(int)
            if not (0.12 <= line <= 0.88):
                continue                      # near the edge the classifier is not used
            if action.startswith("REPAIRED"):
                pos += list(idx)
            elif "by user" in action:
                neg += list(idx)
        last = max([e for _, e, _, _ in runs] + [1])
        in_run = np.zeros(last + 1000, bool)
        for s, e, _, _ in runs:
            in_run[s:e] = True
        free = np.flatnonzero(~in_run)
        normal = rng.choice(free, min(len(free), 1500), replace=False) if len(free) else np.zeros(0, int)
        frames = SM.read_frames(ff, v, np.r_[normal, pos, neg].astype(int))
        normals += [frames[t] for t in normal if t in frames]
        lines += [SM.MH] * sum(1 for t in normal if t in frames)
        slips += [frames[t] for t in pos if t in frames]
        false_alarms += [frames[t] for t in neg if t in frames]
        print(f"{os.path.basename(v)}: {len(normal)} normal, {len(pos)} slip, {len(neg)} false-alarm frames")
    if not slips or not normals:
        sys.exit("Not enough examples.")
    with tempfile.TemporaryDirectory() as tmp:
        pt = os.path.join(tmp, "personal.pt")
        SM.train(np.stack(normals), np.stack(false_alarms) if false_alarms else np.zeros((0, SM.MH, SM.MW, 3), np.uint8),
                 np.array(lines), a.steps, pt, start_from=base, edge=7, slips=np.stack(slips), slip_share=0.7,
                 step_share=0.15, lr=3e-4)
        target = os.path.join(a.app, "models")
        # the same calibration as the base model
        meta = os.path.splitext(base)[0] + ".json"
        if os.path.exists(meta):
            shutil.copyfile(meta, os.path.splitext(pt)[0] + ".json")
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_onnx.py"),
                        pt, target, "--name", "slip_classifier_personal.onnx"], check=True)
    print("Super8Fixer.exe now uses models\\slip_classifier_personal.onnx (delete it to use the shipped model).")


if __name__ == "__main__":
    main()
