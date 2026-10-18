# Interaction measures

**Status: partly implemented.** The cube-stacking task already combines hand landmarks with cube tracks to detect pinch grips and cube moves (`grasp.py`). Per-move timing and finger features are designed in [session analysis](session-analysis.md); the broader event set below remains planned.

## Candidate observable outputs

- Reach onset/offset and hand-to-object proximity.
- Pickup, transport, release, placement, and stacking intervals with confidence.
- Elapsed time between event types and sequence-level counts/rates.
- Which visible hand/finger landmarks are near an object, when pose quality supports that observation.

These are estimates from video and should retain event timestamps, confidence, source IDs, and ambiguity. Occlusion, camera perspective, frame rate, and model errors can shift or obscure event boundaries.

## Interpretation limits

Timing and finger configuration do not directly measure grip force. Estimating force would require an independently validated measurement method (for example, instrumented objects) and calibration; video-only proxies must be named as proxies. Do not infer illness patterns, intent, or clinical status from isolated movement metrics. Any future clinical use needs an explicit governance/validation pathway outside this exploratory system.

Validate event boundaries and measures against independent, consented annotations, report inter-annotator variability and error by condition, and avoid presenting uncertain events as facts.
