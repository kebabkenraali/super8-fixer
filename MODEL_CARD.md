# Model card: slip classifier

File: `models/slip_classifier.onnx` (about 0.8 MB). License: CC BY-NC 4.0.

## What it does

Given a downscaled video frame (144 x 96 RGB), the model outputs the probability that a film frame
line is visible inside the picture (a frame-line slip) rather than a straight horizontal edge that
belongs to the picture (a horizon, a window frame, the edge of a table or pavement). Super8 Fixer
uses it to decide runs whose frame line is between 12 % and 88 % of the picture height; runs near the
edges are decided by the tracker's confidence value.

Architecture: five 3x3 convolution layers (32-128 channels) with batch normalization and max
pooling, global max pooling, one linear output. The input is augmented with a vertical difference
image that emphasizes horizontal seams. Input `images`: B x 144 x 96 x 3 float 0..255 RGB;
output `p`: B probabilities.

## Training data

Only emulated data generated with `training/emulate.py` from these amateur films and home movies of
the Prelinger Archives (Rick Prelinger) at the Internet Archive. The films themselves are not
distributed; the model contains no images.

Public domain (Creative Commons Public Domain Dedication):

| identifier | title |
|---|---|
| Medicusc1939_4 | Amateur film: Medicus collection: New York World's Fair, 1939-40 (Reel 4, Part I) |
| Summersc1941 | Amateur film: Summers collection: San Francisco, California, 1941 (Part I) |
| TexasFar1952 | Amateur film: Texas Farm Family |
| GoldenGa1939 | Amateur film: Golden Gate Bridge |
| Paradeof1941 | Amateur film: Parade of Krewe of Rex, New Orleans, February 25, 1941 |
| Wathenco1939 | Amateur film: Wathen collection: New York World's Fair, 1939-40 (Part I) |
| Sagaofth1957 | Amateur film: "Saga of the Happy Wanderers" |
| amateur_northwest | Amateur Film: Northwest |

Creative Commons Attribution 3.0 (<https://creativecommons.org/licenses/by/3.0/>), by Prelinger
Archives, from <https://archive.org/details/prelingerhomemovies>; used unchanged as source material
for the emulated training videos:

| identifier | title |
|---|---|
| HomeMoviesCan11261DetroitFamilyEarly1960s | Home Movies: Can 11261: Detroit Family, Early 1960s |
| HomeMoviesCan11268DetroitMichiganFamilyPostalWorkersCa.1955 | Home Movies: Can 11268: Detroit, Michigan Family, Postal Workers, ca. 1955 |
| HomeMoviesCan11269DetroitFamilyScenesAndVacation | Home Movies: Can 11269: Detroit Family Scenes and Vacation |
| HomeMoviesCan11283NiagaraFallsHomeScenesBirthdayParty | Home Movies: Can 11283: Niagara Falls, Home Scenes, Birthday Party |
| HomeMoviesCan11302DetroitMichiganFamily | Home Movies: Can 11302: Detroit, Michigan Family |
| HomeMoviesCan11339ALittlePicnic | Home Movies: Can 11339: A Little Picnic |
| HomeMoviesCan11243DetroitMichiganFamilyAndTravelFilm | Home Movies: Can 11243: Detroit, Michigan Family and Travel Film |
| HomeMoviesCan11309VisitToDetroitMichiganAndMilwaukeeWisconsin | Home Movies: Can 11309: Visit to Detroit, Michigan and Milwaukee, Wisconsin, 1951 |

The emulator draws for every training video its own film and scanner: frame-line thickness, tone
and softness, perforation hole, scan window over-height (frame line visible or cropped), exposure,
grain, vignette, sharpness and compression, and inserts slips of three kinds (slides, rolls, small
constant offsets) and gate weave. 24 videos of 90 s each were used: seeds 1-12 from the public-domain
films and seeds 13-24 from the CC BY home movies. Places where the
tracker alone raised false alarms in the source films (horizons, sign edges) were sampled more often
as hard negative examples, and artificial brightness steps without a frame line were added as
negatives. Original, unprocessed frames of 14 of the 16 source films (400 per film) were used as
negatives too, so that the model does not learn the look of the emulator; Paradeof1941 and
HomeMoviesCan11283 were held out for evaluation. 4000 training steps of 128 examples.

`models/slip_classifier.json` sets the operating point 0.15: Super8Fixer.exe calibrates the mean
probability of a run so that 0.15 becomes 0.5, the default classifier threshold.

## Evaluation

On the two held-out films, which have no slips (every repaired run is a false alarm), with the
default settings:

| film | length | runs repaired | frames repaired |
|---|---|---|---|
| Paradeof1941 | 9.5 min | 6 | 210 |
| HomeMoviesCan11283 | 12.7 min | 5 | 395 |

In a private home-movie transfer with known real slips (not used for training) all 10 known slip
runs were detected and 6 of 8 known false-alarm runs rejected.

No private home movies were used for training, and the model contains no images.

## Limitations

- Trained on emulation. Real scanners and films differ; review the run table before rendering.
- A frame line hidden by the scanner crop and a picture edge that happens to line up with it can
  still be confused. Use the run editor, or fine-tune a personal model with
  `training/finetune_from_edits.py`.
