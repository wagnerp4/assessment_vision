# Screw and nut detector (learned)

**Status:** **Implemented.** First model trained on 2026-10-18 (YOLO11n, 416 px; see [First model](#first-model-2026-10-18)). Without `models/fastener_detector.onnx` the app falls back to the classical detector in `fasteners.py`.

## Why a learned detector

The screwing game needs, in every frame, where the screw tip and head are and where the nut is along the shaft. The classical detector (`FastenerDetector`) separates gray metal from a white sheet and works only when the parts lie on that sheet.

During play the parts are held in the hands. Fingers cover the head, highlights blow the nut face out to white, and the background is skin, the desk, or the mousepad. On the October 2026 calibration clips (`runs/calibration/gameplay_*`, `live_*`):

- The classical detector placed the screw on the keyboard or the pad edge in most in-hand frames.
- A hand-guided prototype (searching between the pinch grips) found the screw in about 5 of 8 sampled frames, but the nut and depth were unreliable.

Hand-tuned color and edge rules break with each change of light, so the screw and nut are learned instead.

## What the model detects

One object detector with four box classes ([`datasets/fasteners/data.yaml`](../../datasets/fasteners/data.yaml)):

| id | class | Box covers | Label when |
|---|---|---|---|
| 0 | `screw` | the visible screw, head to tip | any part of the shaft is visible |
| 1 | `screw_head` | the head cylinder only | at least half of the head is visible |
| 2 | `screw_tip` | the last ~1.5 shaft diameters of the threaded end | the end of the thread is visible |
| 3 | `nut` | the whole nut, on or off the screw | at least half of the nut is visible |

Boxes rather than keypoints keep every small detector family usable, including permissively licensed ones. `geometry()` in [`fastener_model.py`](../../src/assessment_app/fastener_model.py) turns the boxes into the existing `FastenerObservation`:

- **Axis:** the head center to the tip center.
- **Hidden head:** the axis points at the far corner of the `screw` box. The length comes from the nut: across its corners the nut is 0.30 of the head-to-tip length (`NUT_TO_LENGTH`, from `runs/calibration/gameplay_2026-10-17T21-38-04/measurements.json`).
- **Depth:** the nut center projected on the axis, 0 at the tip and 1 at the head.
- **On the screw:** the nut center lies within 0.6 nut widths of the axis, and its depth is between 0 and 1. A nut held beside the screw fails this test.
- **Nut edges and turns:** still classical, now searched only inside the nut box.

The game logic and summaries do not change.

### Labeling rules

- Label every visible instance. There is one screw and one nut per frame.
- Frames with no part visible stay in the set with an empty label file. They teach the model what is *not* a screw (keyboard keys, the pad edge, rings).
- Boxes are tight around visible pixels. Do not guess the hidden part behind fingers.
- Motion-blurred parts are labeled if a person can still tell what they are.
- `screw_tip` is the threaded end even when the nut covers part of the thread.

## Data collection protocol

Record with [`scripts/record_fastener_dataset.py`](../../scripts/record_fastener_dataset.py). It runs until **q**. Keys **1–5** mark the stage (setup, pickup, screw, unscrew, putdown), **0** clears it, and **space** pauses. It saves the raw clip without overlays to `datasets/fasteners/raw/<session>/video.mp4`, with `frames.csv` (time and stage of every frame) and `meta.json`.

```bash
uv run python scripts/record_fastener_dataset.py --lighting "ceiling + desk lamp" --background "white rim"
```

Aim for **4–6 sessions of 1–2 minutes**. Each session plays the whole game at least twice, and the sessions differ in conditions:

| Vary | Examples |
|---|---|
| Lighting | daylight, ceiling light only, desk lamp, dim |
| Background | white rim, black pad, wood |
| Camera | the play position; a little higher, lower, or further away |
| Pose of the parts | nut edge-on, nut face-on with the hole visible, screw pointing at the camera |
| Hands | relaxed grip, fingers covering the head, fast and slow turning |

Keep one complete session, with the conditions you expect in use, for **validation only**.

## Dataset build

```bash
uv run python scripts/build_fastener_dataset.py extract --val <session-name>
# label images/train and images/val, export YOLO txt into labels/train and labels/val
uv run python scripts/build_fastener_dataset.py check
```

- **extract** keeps a frame every 400 ms, skips near-duplicates and black warm-up frames, and writes `manifest.csv` (session, frame, time, stage, split). Whole sessions go to one split, so validation never contains frames from a training clip. Older clips can be added with `--video path/to/clip.mp4`.
- **check** validates every label line and prints instances per class and split. Run it before every training run.

At these settings a 1.5-minute session gives about 150–220 frames. **300–600 labeled frames** are a realistic first target.

**Labeling tool:** any tool that exports YOLO txt works, for example CVAT (export "YOLO 1.1") or Label Studio (export "YOLO").

**Labeling loop:**
1. Label about 150 frames by hand.
2. Train a first model.
3. Let it pre-label the remaining frames.
4. Correct the pre-labels, which is much faster than drawing from scratch.

## Model choice

The detector must stay usable on small laptops next to MediaPipe hand tracking. The measurements below are an **upper bound**: inference only, COCO-pretrained nano models exported to ONNX (opset 12), on a desktop **i7-13700K**, median of 30 runs. A 2-core laptop CPU will likely be 2–4× slower.

| Model | Input | OpenCV DNN, 2 threads | ONNX Runtime, 2 threads | ONNX Runtime, 4 threads |
|---|---|---|---|---|
| YOLO11n | 320 | 41 ms | **7.6 ms** | 5.0 ms |
| YOLO11n | 416 | 76 ms | **13.1 ms** | 8.1 ms |
| YOLO11n | 512 | 114 ms | – | – |
| YOLOv8n | 320 | 45 ms | – | – |
| YOLOv8n | 416 | 73 ms | – | – |

- **Runtime:** ONNX Runtime is about 6× faster than OpenCV DNN here. The backend uses ONNX Runtime when the optional `detector` extra is installed (`uv sync --extra detector`) and falls back to OpenCV DNN otherwise. Both return identical boxes on the calibration frames.
- **Starting point:** **YOLO11n at 416**. Then train the same data at **320** and keep 320 if `screw_tip` and `nut` recall hold up. At 416 a 1280×720 frame shrinks by 0.33, so the nut (about 60 px across) becomes about 20 px. At 320 it is about 15 px, which is close to the limit for nano models.
- **Licence:** Ultralytics YOLO (YOLOv8, YOLO11) is **AGPL-3.0**; distributing a product that includes it or its trained weights requires AGPL compliance or an Ultralytics enterprise licence. This repository is MIT. If that matters, train the same four-class dataset with a permissively licensed detector, e.g. **YOLOX-Nano** or **NanoDet-Plus** (both Apache-2.0, around 1M parameters). The dataset format, the class list, and `geometry()` stay the same; only `decode()` changes, because their output layouts differ. **Decision pending.**

## Training

```bash
uv sync --extra train            # ultralytics + PyTorch (CPU is enough for a few hundred images)
uv run python scripts/train_fastener_detector.py train --model yolo11n.pt --imgsz 416
uv run python scripts/train_fastener_detector.py export --weights runs/fastener_train/yolo11n_416/weights/best.pt --imgsz 416
uv run python scripts/train_fastener_detector.py bench --threads 2
```

[`train_fastener_detector.py`](../../scripts/train_fastener_detector.py) starts from COCO-pretrained weights and adds augmentation for this task:

- Stronger brightness and saturation changes, since lighting is the main thing that varies.
- ±15° rotation, scale changes, mosaic, and left-right flips, since the parts look the same mirrored.

It stops early after 30 epochs without improvement. A few hundred images train in well under an hour on a GPU and in a few hours on a CPU.

**export** writes `models/fastener_detector.onnx` and a sidecar `fastener_detector.json` (classes, input size, nut ratio). It refuses weights whose class list differs from `CLASSES`. Weights and data are not committed (see `.gitignore`).

## Runtime and tracking

`create_fastener_backend()` returns `YoloFastenerBackend` when `models/fastener_detector.onnx` exists, and the classical `FastenerBackend` otherwise. The model loads on the first screwing frame.

`FastenerSmoother` keeps readings steady:

- Points are averaged over frames.
- A part that disappears is held for 6 frames (about 0.3–0.5 s) before it is reported lost.
- "On the screw" is a majority vote of the last 5 frames where the nut was seen.
- Depth is the median of the last 5 depths, so one wrong frame cannot move the game.

**Planned speed-ups, if a laptop is too slow:**
- Run the detector on every second frame and let the smoother carry the gap.
- Crop around the hands during stages 2–4, where the parts are always between the pinch grips. A 320 crop then shows the parts 2–3× larger than the full frame does.

## Evaluation

On the held-out session:

| Level | Metric | First target |
|---|---|---|
| Detection | mAP@0.5 per class; recall of `screw_tip` and `nut` | recall ≥ 0.9 for `screw` and `nut`, ≥ 0.8 for `screw_tip` |
| Geometry | depth error against hand-measured depth on 20 frames | median ≤ 0.05 |
| Geometry | nut on/off accuracy per frame | ≥ 95 % |
| Game | stages reached in order on a full replay | all five, no false completions |
| Runtime | full loop (hands + detector) on a 2–4 core laptop | ≥ 12 fps |

The `stage` column in `manifest.csv` (from the 1–5 keys) gives approximate ground truth for the game-level check.

## First model (2026-10-18)

**Data.** Four sessions recorded with `record_fastener_dataset.py`:

| Session | Conditions | Split |
|---|---|---|
| `2026-10-18T15-05-52` | normal light, close-ups, moving camera | train |
| `2026-10-18T15-13-29` | dark room, desk lamp | train |
| `2026-10-18T15-18-38` | wooden cutting board | train |
| `2026-10-18T15-21-57` | normal setup, fixed camera, two games | val |

- **Labels.** 151 train frames (spread over sessions and stages) and 151 val frames (every second extracted frame).
- **How they were made.** 37 frames were drawn by hand. A seed model pre-labeled the rest, and every pre-label was reviewed and corrected.
- **Unlabeled frames.** The remaining extracted frames are in `datasets/fasteners/unlabeled/`. They are kept out of `images/`, because an image without a label file trains as "nothing here".

**Training.** GPU (RTX 3090 Ti, CUDA PyTorch from the `cu130` index), 150 epochs, about 3 minutes per run.

| Model | Input | mAP@0.5 | Recall: screw / head / tip / nut |
|---|---|---|---|
| **YOLO11n** | **416** | **0.964** | 1.00 / 0.99 / 0.84 / 0.99 |
| YOLO11n | 320 | 0.861 | 1.00 / 1.00 / 0.41 / 0.91 |

At 320 the tip is too small, so **416 is the exported model**.

**Validation session, frame by frame** (no smoothing; the reference is `geometry()` on the hand labels):
- Nut on/off agrees on 98.6 % of frames.
- Depth error: median 0.011, p90 0.05.

**CPU speed** (ONNX Runtime, i7-13700K): 15.5 ms with 2 threads, 10.5 ms with 4.

**v2 (same day).** A fifth session, `2026-10-18T17-22-00`, was recorded at the play distance used in the live check, where the parts are smaller than in the close-up sessions. 106 of its frames were labeled, so train now has 257 frames.

- Check: the later half of that session (44 frames) was held out and a model was trained on the rest. On those held-out frames, screw_tip recall rose from 0.51 (v1) to 0.63, and mAP@0.5 from 0.893 to 0.927.
- On the original validation session v2 is level with v1: mAP@0.5 0.965, tip precision 0.89 and recall 0.77.
- v2 is the exported model. v1 is kept as `models/fastener_detector.v1.onnx`.

**Next.**
- The tip is still the weakest class at the play distance. Label more tip frames, or crop around the hands so the parts appear larger (see "Planned speed-ups").
- Label more of `unlabeled/`, especially frames with the tip close to the nut.
- Add a session on a different desk.
- Check the full loop with hand tracking on a laptop.

## Depth scale and game rules (2026-10-18)

- **Depth is the share of the thread.** The game shows thread progress: 0 % when the nut sits on the tip, 100 % at the end of the thread (`thread_progress()` in `screwing.py`, nut-center positions 0.065 to 0.565 of the tip-to-head axis). Before this, the plain shaft near the head counted as part of the length, so a 50 % target really meant about 60 % of the screwable part.
- **The targets are unchanged:** 25 / 40 / 50 / 75 % of the thread. All four can now be reached.
- **Pickup** needs both parts lifted from where they lay during setup, held for 1 s. Before this, hands reaching for the parts were enough to pass.
- **Unscrew** needs the nut seen clear of the screw for 1 s: its center at least one nut width past the tip, or beside the shaft. The plain on/off reading flickers at the tip.
- **No spinning.** If the nut travels 10 % of the thread while no thumb tip or index tip of either hand is on it, the game logs `nut_spun` and shows a warning. The target only counts while a fingertip is on the nut.
  - Limits: the camera cannot feel contact. "On the nut" means a fingertip within about one nut width of its center in the image.
  - A spin is only judged while a hand is tracked. Either hand counts, because the left and right labels sometimes swap.
  - A spin is caught by the distance the nut travels along the thread, not by its rotation speed. At 13–15 fps a flicked nut turns too fast to follow by its edges.

## Open decisions

- **Licence route:** Ultralytics (AGPL-3.0) or a permissively licensed detector (Apache-2.0).
