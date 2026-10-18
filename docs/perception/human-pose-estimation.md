# Human pose estimation

**Status: implemented as optional backends.** `--pose` defaults to `MediaPipeHandBackend`, using `models/hand_landmarker.task`. It detects and tracks both 21-point hands independently of body detection in MediaPipe Tasks video mode. The application displays only the selected playing hand (right by default), near or approaching cubes. This supports overhead tabletop views with no torso visible. Handedness is a model prediction; it does not establish person identity. The legacy `MediaPipeHolisticBackend` remains available through `--pose-backend holistic`, using `models/holistic_landmarker.task`. Install the `pose` extra; use `--pose-model` to override the selected model's path.

## Intended use

Pose is currently an observation and visualization source for locating hands relative to tracked cubes. It does not yet estimate reach/contact/pickup events. The MediaPipe Holistic graph tracks one person across successive frames; its person ID is fixed to `person-0` and is only meaningful within that sequence. It is not multi-person tracking. Camera keypoints do not measure applied force.

## Integration requirements

- The backend maps normalized model x/y values to pixel coordinates in the original frame. Body names follow the MediaPipe 33-landmark ordering; hand names are wrist followed by thumb, index, middle, ring, and pinky joints/tips.
- Optional `Keypoint.confidence` is `None` when the MediaPipe landmark has no visibility/presence score. Where provided, the score is passed through and is not assumed calibrated.
- The initial backend uses CPU inference and resets its tracker at sequence boundaries. It currently exposes a single person only. The model also computes a dense face mesh internally; the application does not store/render that mesh (the standard 33-pose output includes sparse facial keypoints).
- Evaluate body and hand landmarks separately under expected camera distance, viewpoint, lighting, occlusion, and laptop resource limits before relying on the output for analysis.

Record consent and data-handling requirements for videos of people. Report uncertainty and failure cases; pose outputs are not clinical findings.
