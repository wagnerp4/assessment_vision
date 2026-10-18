# Core contracts

The public, library-neutral types currently live in `assessment_app.contracts`.

| Contract | Responsibility | Current notes |
| --- | --- | --- |
| `Frame` | Image plus sequence ID, source frame index, and timestamp in milliseconds. | Image is an OpenCV BGR array; index/timestamp refer to the source sequence even when sampling is strided. |
| `DatasetSample` / `Dataset` | Identify a video sample and stream its frames. | Optional metadata and annotation-sidecar path; `VideoDataset` handles video and JSONL manifests. |
| `ModelBackend` | Load, predict, reset, and close a model implementation. | Exposes capabilities and returns normalized `ModelObservation` values. |
| `ModelObservation` | Typed frame-level model output. | Cube detections and pose observations have backends; scene-graph values do not yet. |
| `VisionTask` | Consume ordered frames and observations with sequence-local state. | Begins/ends per sample and returns task-specific output. |
| `PipelineFrame` | Bundle sample, frame, observations, and task output. | Shared pipeline result used by application code. |

## Lifecycle and invariants

The pipeline loads backends once, resets them at sequence starts, begins the task for each sample, processes frames in order, and ends the task at sequence completion. The optional pose backend tracks one person; its `person-0` ID is sequence-local. Backend cleanup should happen even if loading fails partway. A task must not assume every frame contains every observation type. Stable IDs are local to a sequence unless a future contract explicitly states otherwise.

## Adding a contract

Add or change a shared contract only when at least one concrete backend can produce it and a task or evaluation consumer needs it. Specify coordinate units/origin, timestamp basis, confidence semantics, identity lifetime, and missing/ambiguous output behavior. Keep framework tensors and model-library classes behind adapters. Add contract tests for empty outputs, sequence resets, and timestamp/frame alignment.

Future segmentation and interaction outputs are not yet part of `ModelObservation`; add them only with those semantics and consumers decided.
