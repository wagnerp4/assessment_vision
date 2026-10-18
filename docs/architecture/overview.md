# Architecture overview

## Implemented flow

```text
VideoDataset / live camera
          │ timestamped Frame
          ▼
VisionPipeline ── one or more ModelBackend adapters
          │ normalized ModelObservation values
          ▼
VisionTask ── sequence-local state and task output
          │
          ├── game/session logic
          └── visualization or saved output
```

`VideoDataset` streams one video or samples listed in JSONL. `HSVCubeBackend` wraps the OpenCV cube detector and tracker. The optional `MediaPipeHolisticBackend` adds single-person body/hand landmarks. `VisionPipeline` loads/resets backends at sequence boundaries, combines their observations, and calls the task in frame order. `CubeStackingTask` uses cube observations with the stacking analyzer and game engine. The application currently exposes local video processing and a live camera mode; `Visualizer` renders cube/game state and, when enabled, pose landmarks.

## Design boundaries

- **Data ingestion** owns source metadata, frame ordering, timestamps, and decoding. It should not know which model or game consumes a frame.
- **Model backends** own framework/model execution and normalize results. Framework-specific objects should not leak into task contracts.
- **Perception observations** describe what a model inferred at a frame/time. They are not application decisions.
- **Tasks** combine observations over time to answer a user-facing question, such as whether cubes form a valid stack or when an interaction occurred.
- **Game logic and interfaces** present task state and accept user interaction; they should not duplicate inference.
- **Device/runtime support** chooses capture and inference resources while preserving the same frame/task semantics.

## Planned evolution

Keep the core usable without PyTorch or Transformers installed. Add optional backend dependencies separately. A future web/tablet client should consume session/task outputs through a stable boundary rather than importing model implementation details. Do not add scene graphs, segmentation, or additional interfaces until their consumer and evaluation benefit are clear; see the [roadmap](../roadmap.md).
