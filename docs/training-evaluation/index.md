# Training and evaluation

The project has one training pipeline, for the [screw and nut detector](../models/fastener-detector.md) (dataset build, training, export, CPU benchmark); it has not yet been run on a labeled dataset. There is no formal benchmark suite. Existing unit tests cover contracts, pipeline behavior, video datasets, imports, and detector array-shape compatibility; the sample clip is a smoke test, not evidence of general accuracy.

Future model training should be optional and backend-specific. Evaluation should have both capability-level metrics and end-to-end task/event metrics. Keep splits independent by recording and subject where appropriate, publish the exact manifest/configuration, and document limitations and runtime cost. See [protocols](protocols.md) and the [roadmap](../roadmap.md).
