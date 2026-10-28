"""A small CNN classifier: does the frame have a frame line inside the picture (a slip)?

Positive examples are real (emulated) slips or normal frames rolled artificially (the same
phenomenon as a real slip); negatives are normal frames, frames with a partly visible frame line at
the edge, and artificial brightness steps without a frame line (horizons, edges).
"""
import os
import subprocess
import time

import numpy as np

MW, MH = 96, 144                      # model input size


def network():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            c = [3, 32, 48, 64, 96, 128]
            s.convs = nn.ModuleList([nn.Conv2d(c[k], c[k + 1], 3, padding=1) for k in range(5)])
            s.bns = nn.ModuleList([nn.BatchNorm2d(c[k + 1]) for k in range(5)])
            s.fc = nn.Linear(128, 1)

        def forward(s, x):                      # x: B,H,W,3 uint8
            x = x.permute(0, 3, 1, 2).float() / 255.0
            # the vertical difference emphasises horizontal seams
            x = torch.cat([x[:, :, :1] * 0, x[:, :, 1:] - x[:, :, :-1]], 2) * 4 + x
            for k, (cv, bn) in enumerate(zip(s.convs, s.bns)):
                x = F.relu(bn(cv(x)))
                x = F.max_pool2d(x, 2) if k < 4 else x
            return s.fc(x.amax((2, 3))).squeeze(1)
    return Net()


def read_frames(ff, src, indices):
    """Downscaled frames (MH x MW RGB) at the given indices, in one pass: {index: frame}."""
    need = np.unique(np.asarray(indices, dtype=np.int64))
    out = np.zeros((len(need), MH, MW, 3), np.uint8)
    p = subprocess.Popen([ff, "-v", "error", "-i", src, "-vf", f"scale={MW}:{MH}:flags=area", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=MW * MH * 3 * 256)
    size = MW * MH * 3
    i = k = 0
    last = need[-1] if len(need) else -1
    while k < len(need):
        b = p.stdout.read(size)
        if len(b) < size:
            break
        if i == need[k]:
            out[k] = np.frombuffer(b, np.uint8).reshape(MH, MW, 3)
            k += 1
        i += 1
        if i > last:
            break
    p.kill()
    return dict(zip(need.tolist(), out))


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(path):
    import torch
    net = network()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.to(device()).eval()


def predict(net, frames):
    """frames: N,MH,MW,3 uint8 -> slip probability"""
    import torch
    if len(frames) == 0:
        return np.zeros(0)
    out = []
    with torch.no_grad():
        for k in range(0, len(frames), 256):
            x = torch.from_numpy(np.ascontiguousarray(frames[k:k + 256])).to(device())
            out.append(torch.sigmoid(net(x)).cpu().numpy())
    return np.concatenate(out)


def _augment(x, rng):
    """exposure, contrast, gamma, saturation, sharpness, compression and noise variation, mirroring"""
    import io
    from PIL import Image, ImageFilter
    if rng.random() < 0.4:                                               # softness or low resolution
        if rng.random() < 0.5:
            x = np.asarray(Image.fromarray(x).filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2))))
        else:
            k = rng.uniform(0.4, 0.8)
            x = np.asarray(Image.fromarray(x).resize((int(MW * k), int(MH * k)), Image.BOX).resize((MW, MH), Image.BILINEAR))
    if rng.random() < 0.3:                                               # compression artefacts
        b = io.BytesIO()
        Image.fromarray(x).save(b, "JPEG", quality=int(rng.integers(30, 90)))
        x = np.asarray(Image.open(b).convert("RGB"))
    x = x.astype(np.float32)
    x = 255 * (x / 255) ** rng.uniform(0.6, 1.6)
    gray = x.mean(2, keepdims=True)
    x = gray + (x - gray) * rng.uniform(0.3, 1.4)
    x = (x - 128) * rng.uniform(0.7, 1.8) + 128 + rng.uniform(-30, 30)
    x *= rng.uniform(0.8, 1.2, size=3)
    x += rng.normal(0, rng.uniform(0, 6), size=x.shape)
    if rng.random() < 0.5:
        x = x[:, ::-1]
    return np.clip(x, 0, 255).astype(np.uint8)


def _roll_amount(line, rng, edge):
    """A random roll that takes the frame line (at row `line`) at least `edge` rows from the picture
    edges: a real slip can also be close to the edge."""
    new = rng.integers(edge, MH - edge)
    return int(new - line) % MH


def _edge_band(x, rng, edge):
    """A dark frame line at the top or bottom edge (at most edge-1 rows), soft boundary."""
    x = x.astype(np.float32).copy()
    k = int(rng.integers(1, max(edge, 2)))
    gain = np.linspace(rng.uniform(0.05, 0.5), 1, k + 1)[:-1, None, None]
    if rng.random() < 0.5:
        x[:k] *= gain
    else:
        x[-k:] *= gain[::-1]
    return x.astype(np.uint8)


def _step(x, rng):
    """A horizontal brightness step and side shift without a dark frame line (horizon, edge, cut)."""
    x = x.astype(np.float32).copy()
    r = int(rng.integers(12, MH - 12))
    k = int(rng.integers(0, 4))
    gain = np.ones(MH, np.float32)
    gain[r:] = rng.choice([rng.uniform(0.45, 0.8), rng.uniform(1.25, 1.8)])
    if k:
        gain[r:r + k] = np.linspace(1, gain[-1], k + 2)[1:-1]
    x *= gain[:, None, None]
    if rng.random() < 0.5:
        x[r:] = np.roll(x[r:], int(rng.integers(-12, 13)), axis=1)
    return np.clip(x, 0, 255).astype(np.uint8)


def train(normals, false_alarms, lines, steps, path, start_from=None, edge=7, slips=None, slip_share=0.5,
          step_share=0.0, lr=2e-3, log=print):
    """normals, false_alarms: N,MH,MW,3 uint8; lines: frame-line row of each normal frame (MH scale,
    MH = hidden). slips: real slip frames used as positives; slip_share is the share of positives taken
    from them (the rest are rolled normal frames). step_share: share of negatives that get a brightness
    step without a frame line. Saves the model to `path` (PyTorch state dict)."""
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng()
    net = network()
    if start_from:
        net.load_state_dict(torch.load(start_from, map_location="cpu"))
        log(f"Continuing from {start_from}")
    dev = device()
    net = net.to(dev).train()
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    t0 = time.time()
    for step in range(steps):
        xs, ys = [], []
        for _ in range(64):            # positives: a real slip, or a normal frame rolled
            if slips is not None and len(slips) and rng.random() < slip_share:
                xs.append(_augment(slips[rng.integers(len(slips))], rng))
            else:
                j = rng.integers(len(normals))
                xs.append(_augment(np.roll(normals[j], _roll_amount(lines[j], rng, edge), 0), rng))
            ys.append(1)
        for _ in range(64):
            src = false_alarms if len(false_alarms) and rng.random() < 0.3 else normals
            x = src[rng.integers(len(src))]
            if rng.random() < 0.3:                 # frame line partly visible at the edge: still a normal frame
                x = _edge_band(x, rng, edge)
            elif rng.random() < step_share:
                x = _step(x, rng)
            xs.append(_augment(x, rng))
            ys.append(0)
        x = torch.from_numpy(np.stack(xs)).to(dev)
        y = torch.tensor(ys, dtype=torch.float32, device=dev)
        loss = F.binary_cross_entropy_with_logits(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 250 == 0:
            log(f"\rTraining: {step + 1}/{steps}, loss {loss.item():.3f}, {time.time() - t0:.0f} s   ", end="")
    log("")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(net.cpu().state_dict(), path)
    return net.to(dev).eval()
