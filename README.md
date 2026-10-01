# Assessment Vision

Assessment Vision detects colored cubes, tracks them across video frames, and evaluates simple stacking tasks. Its inference flow separates video datasets, model backends, and stateful tasks so additional computer-vision models can be integrated without coupling the application to a specific model library.

## Demo

The short clip below is derived from the existing `dev_frames/stack1.mp4` recording. The full development recordings remain local and are excluded from Git.

[Play the sample cube-detection video](docs/assets/demo.mp4)

Only publish video clips that are cleared for repository use. Replace this preview with a newly recorded demo when one is ready.

## Install and run

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dev]"
python -m assessment_app --mode video --video docs/assets/demo.mp4 --difficulty easy
```

To save an annotated video, add `--output assessment_output.mp4`. For a live camera, run `python -m assessment_app --mode camera --difficulty easy`; press `s` to start, `r` to reset, `p` to pause or resume, and `q` to quit.

Run the tests with `python -m pytest`.

## Integration contracts

- `Dataset` streams `DatasetSample` records and timestamped `Frame` objects. `VideoDataset` reads one video or a JSONL manifest. Manifest paths are relative to the manifest file; each non-comment line contains `sample_id`, `video`, and optional `annotations` and `metadata` fields.
- `ModelBackend` loads an implementation and returns normalized observations. The built-in `HSVCubeBackend` adapts the OpenCV detector. The core package has no PyTorch or Transformers dependency; future adapters can implement the same contract.
- `VisionTask` consumes ordered frame observations and maintains state for a sequence. `CubeStackingTask` adapts the current stacking and game logic.
- Shared observation types include cube detections, person and hand pose landmarks, scene-graph nodes and relations, and timed action events. Person identity is local to a sequence unless a future task explicitly defines otherwise.

For example, one JSONL entry can pair a video with a future annotation sidecar:

```json
{"sample_id":"session-001","video":"videos/session-001.mp4","annotations":"annotations/session-001.json","metadata":{"fps":30}}
```

The first person-focused extension is human and hand pose estimation. A later interaction task can combine hand landmarks with cube tracks to record observable reach, pickup, placement, and stacking intervals. Video timing and finger positions alone do not measure grip force. The application reports observable events; it does not diagnose illness or make clinical assessments.

## Development

Install the optional development tools with `python -m pip install -e ".[dev]"`. The sample video is useful for a visual smoke test, but labeled and consented recordings are needed to evaluate pose, hand-contact, and behavioral-analysis performance.
