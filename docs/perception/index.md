# Perception capabilities

Perception describes frame-level visual evidence. It is distinct from an application [task](../application/tasks-and-game-logic.md), which interprets observations across time to serve a user goal.

| Capability | Status | Documentation |
| --- | --- | --- |
| Colored-cube object detection | Implemented with a classical OpenCV/HSV approach. | [Detection and tracking](object-detection-and-tracking.md) |
| Cube tracking | Implemented with sequence-local tracks. | [Detection and tracking](object-detection-and-tracking.md) |
| Human and hand pose | Optional MediaPipe hand-landmark backend (default; works without the body in view) or Holistic (body and both hands). | [Human pose estimation](human-pose-estimation.md) |
| Scene graph generation | Observation types exist; producer and evaluation are planned. | [Scene graphs](scene-graphs.md) |
| Segmentation | Planned; no mask contract/backend currently exists. | [Segmentation](segmentation.md) |

New capabilities should report normalized observations with confidence and source timing. Prefer the simplest capability set that can be validated for the target task.
