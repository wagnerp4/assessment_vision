# Object detection and tracking

## Current implementation

`CubeDetector` uses configured color/shape cues to detect colored cubes in OpenCV images. Its tracker associates detections over frames and assigns sequence-local IDs. `HSVCubeBackend` adapts those results into `CubeDetections`. This is a task-specific baseline, not a general-purpose object-recognition model.

## Planned direction

Keep detection and tracking conceptually separable: a detector proposes objects per frame; a tracker associates them across frames. A learned detector may replace or complement the HSV detector behind a backend, while task code consumes normalized detections/tracks. Specify category, bounding-box coordinate system, confidence, source frame/time, and track-ID lifetime before broadening the contract.

## Evaluation

Use representative lighting, occlusion, motion, cube colors, and backgrounds. Report detection precision/recall at a declared overlap threshold, localization quality, tracking ID switches/fragmentation, and end-to-end pickup/stack event quality where relevant. Keep clips and expected annotations versioned and split by recording/subject rather than neighboring frames.
