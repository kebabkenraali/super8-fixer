# Training tools

Python tools for making test videos with ground truth and for training the slip classifier.
The application itself does not need Python.

Requirements: Python 3.10+, `pip install numpy pillow torch onnx` (a CUDA build of PyTorch makes
training much faster), and ffmpeg (on the PATH, or in `..\Super8Fixer\ffmpeg\bin\`).

## Emulated test and training videos

`emulate.py` turns film transfers *without* slips into "badly scanned film" with exact ground truth:

```
python emulate.py --videos sources -o test1 --duration 60 --seed 31
```

writes `test1.mp4` (with slips, weave, duplicate frames and a film look), `test1_reference.mp4` (the
same without slips: the ideal repair result) and `test1_truth.npz` (frame-line row, slip, weave,
duplicate and scene of every frame). `--images FOLDER` makes camera moves over still pictures instead.

When a source video has a repair report next to it, its false alarms are used more often, which
gives hard negative examples:

```
Super8Fixer.exe repair sources\film.mp4 --analyze-only --no-model
```

## Training the shareable model

```
for /L %s in (1,1,12) do python emulate.py --videos sources -o emu\e%s --duration 90 --seed %s --no-reference
python train_emulated.py emu\e1 emu\e2 ... emu\e12 --negatives sources -o slip_classifier.pt --steps 4000
python export_onnx.py slip_classifier.pt ..\Super8Fixer\models
```

`--negatives` adds original frames of slip-free transfers as negatives as they are; without them the
model tends to learn the look of the emulator and raises false alarms in busy scenes. Keep one or two
sources out of training to measure the false alarms on them.

The model outputs a per-frame probability; the application averages it over up to 12 frames of a
run. `slip_classifier.json` next to the model (`{"operating_point": 0.15}`) tells the application
which mean probability counts as 0.5; `export_onnx.py` copies it next to the ONNX file.

Use only sources that you may use (for example public-domain films); the model learns from them.
The shipped model and its sources are described in [../MODEL_CARD.md](../MODEL_CARD.md).

## Learning from your own corrections

After you have reviewed a film in Super8Fixer.exe (unticked false alarms, ticked missed slips):

```
python finetune_from_edits.py D:\films\film1.mp4 D:\films\film2.mp4 --app ..\Super8Fixer
```

This continues training from `../models/slip_classifier.pt` with your decisions and writes
`Super8Fixer\models\slip_classifier_personal.onnx`, which the application then uses instead of the
shipped model. The personal model has learned from your own film: keep it for yourself.
Delete the file to return to the shipped model.
