# Backend adapters

## Current backend

`HSVCubeBackend` adapts `CubeDetector` and its tracker to `ModelBackend`. Its capability is cube detection; it loads no separate weights, resets sequence-local tracking state, and returns `CubeDetections`.

`MediaPipeHandBackend` is the default optional CPU backend for `--pose`. It uses `models/hand_landmarker.task`, detects up to two hands independently of body pose, and returns pixel-coordinate `PoseObservation` values. Video mode tracks between detections; sequence resets recreate the tracker. The app selects the playing hand downstream. The model's left/right labels are predictions, not persistent identities.

`MediaPipeHolisticBackend` remains available with `--pose-backend holistic` for body and hand landmarks from `models/holistic_landmarker.task`. Both adapters use the optional `pose` extra and keep model-specific data out of task code. Holistic can fail to detect hands when the body is outside the frame; prefer the hand backend for tabletop views.

## Planned learned backends

A future PyTorch or Hugging Face adapter should implement the shared backend lifecycle and translate library outputs to contract types. Model loading/inference, preprocessing/postprocessing, optional dependencies, and device-specific code stay inside the adapter. The task should not branch on which library produced a result.

For each adapter document: supported capability and model/version; weights source/license; preprocessing and coordinate mapping; CPU/GPU selection; memory and latency; batch/stream behavior; reset/close semantics; known failure modes; and exact evaluation dataset/protocol. Avoid loading heavyweight dependencies at package import time. No PyTorch or Hugging Face model has been selected yet.
