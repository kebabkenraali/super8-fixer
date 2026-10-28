# Third-party notices

Super8 Fixer itself is licensed under the PolyForm Noncommercial License 1.0.0 ([LICENSE.md](LICENSE.md)).
It contains or uses the following third-party components, which keep their own licenses. Their
noncommercial restriction does not apply to them.

## Linked into Super8Fixer.exe

| component | license | notes |
|---|---|---|
| Rust crates (egui/eframe, image, rayon, serde, anyhow, rfd, ort, windows-sys and their dependencies, 223 in total) | MIT, Apache-2.0, BSD, Zlib, ISC, Unicode-3.0, BSL-1.0, CDLA-Permissive-2.0 | Full texts and copyright holders: [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt). Dual-licensed crates (e.g. `self_cell`: Apache-2.0 OR GPL-2.0) are used under the permissive option. |
| Fonts embedded by egui (Ubuntu Light, Hack, Noto Emoji, emoji-icon-font) | Ubuntu Font Licence 1.0, MIT/Bitstream Vera, SIL OFL 1.1, MIT | Texts included in THIRD_PARTY_LICENSES.txt. |
| ONNX Runtime 1.31 (Microsoft), statically linked | MIT | [licenses/ONNXRuntime-LICENSE.txt](licenses/ONNXRuntime-LICENSE.txt); its own third-party notices: [licenses/ONNXRuntime-ThirdPartyNotices.txt](licenses/ONNXRuntime-ThirdPartyNotices.txt) (from ONNX Runtime 1.30; the 1.31 file is at <https://github.com/microsoft/onnxruntime/blob/v1.31.0/ThirdPartyNotices.txt>). |

## Shipped next to the exe

| component | license | notes |
|---|---|---|
| DirectML.dll 1.15.4 (Microsoft DirectML) | Microsoft DirectML license: may be distributed in applications that run on Windows | [licenses/DirectML-LICENSE.txt](licenses/DirectML-LICENSE.txt), [licenses/DirectML-ThirdPartyNotices.txt](licenses/DirectML-ThirdPartyNotices.txt) |
| msvcp140.dll, msvcp140_1.dll, vcruntime140.dll, vcruntime140_1.dll | Microsoft Visual C++ Redistributable (distributable code under the Visual Studio license terms) | Can be removed when the Visual C++ Redistributable is installed. |
| models/slip_classifier.onnx | CC BY-NC 4.0 ([licenses/CC-BY-NC-4.0.txt](licenses/CC-BY-NC-4.0.txt)) | This project's own model, see [MODEL_CARD.md](MODEL_CARD.md). Trained on emulated data made from Prelinger Archives films (public domain and CC BY 3.0). |

## Not shipped: ffmpeg

Super8 Fixer runs `ffmpeg.exe` and `ffprobe.exe` as separate programs to read and write video. They
are **not** part of the release. `get-ffmpeg.cmd` downloads a build from
<https://github.com/BtbN/FFmpeg-Builds> to `ffmpeg\bin\` on the user's own computer. FFmpeg is licensed
under the LGPL/GPL (the builds with libx264 under the GPL); see <https://ffmpeg.org/legal.html>.

## Training data of the model

Amateur films and home movies of the Prelinger Archives at the Internet Archive: public-domain films
and home movies licensed under Creative Commons Attribution 3.0 (attribution: Prelinger Archives).
The full list is in [MODEL_CARD.md](MODEL_CARD.md). The films themselves are not distributed with
Super8 Fixer.
