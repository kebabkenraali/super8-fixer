"""Export a trained slip classifier (PyTorch state dict) to ONNX for Super8Fixer.exe.

Usage: python export_onnx.py slip_classifier.pt [target folder] [--name slip_classifier.onnx]
Input "images": B,144,96,3 float 0..255 RGB -> output "p": slip probability B.

When <model>.json exists next to the .pt ({"operating_point": 0.15}), it is copied next to the ONNX
file: Super8Fixer.exe calibrates the mean probability of a run so that the operating point becomes 0.5.
"""
import argparse
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slip_model as SM  # noqa: E402


class WithSigmoid(torch.nn.Module):
    def __init__(s, net):
        super().__init__()
        s.net = net

    def forward(s, x):
        return torch.sigmoid(s.net(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("target", nargs="?", default="models")
    ap.add_argument("--name", default="slip_classifier.onnx")
    a = ap.parse_args()
    os.makedirs(a.target, exist_ok=True)
    net = SM.network()
    net.load_state_dict(torch.load(a.model, map_location="cpu"))
    net.eval()
    out = os.path.join(a.target, a.name)
    torch.onnx.export(WithSigmoid(net), torch.zeros(2, SM.MH, SM.MW, 3), out,
                      input_names=["images"], output_names=["p"], dynamic_axes={"images": {0: "b"}, "p": {0: "b"}},
                      opset_version=17, dynamo=False)
    meta = os.path.splitext(a.model)[0] + ".json"
    if os.path.exists(meta):
        shutil.copyfile(meta, os.path.splitext(out)[0] + ".json")
    print("Written:", out)


if __name__ == "__main__":
    main()
