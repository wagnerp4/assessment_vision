# Project roadmap

This roadmap is a planning baseline, not a promise that every candidate capability will ship. Implement features only after their data contract, evaluation method, runtime cost, and user-facing purpose are understood. Update this page when evidence or project priorities change.

## Stage 1 — dependable baseline (implemented / stabilize)

- Keep video and camera ingestion on the timestamped `Frame` and `Dataset` path.
- Preserve the library-neutral `ModelBackend` → normalized observation → `VisionTask` separation.
- Stabilize cube detection/tracking and stacking behavior with reproducible clips, tests, and documented limitations.
- Make camera selection and common OS/device failures understandable to contributors.

**Exit evidence:** the sample video runs reproducibly; detector/task behavior has regression tests; camera paths fail with actionable diagnostics; interfaces are documented.

## Stage 2 — data and evaluation foundation (planned)

- Define annotation conventions and versioned dataset manifests before collecting more training/evaluation data.
- Establish consent, privacy, access, retention, and de-identification practices for recordings of people.
- Build repeatable evaluation splits and metrics for each capability and the end-to-end interaction task.
- Keep evaluation data separate from model training/tuning data and report dataset limitations.

**Exit evidence:** an annotator can label a small consented sample consistently; evaluation can be rerun from a manifest and produces documented metrics.

## Stage 3 — hand/person interaction observations (in progress)

- Optional MediaPipe backends (hand model, Holistic) emit hand keypoints; the stacking task uses them for pinch-grip and cube-move detection and an optional finger-pair rule (implemented).
- Per-move timing, finger features, and session storage are implemented ([session analysis](analysis/session-analysis.md)); next is validating them against annotated recordings.
- Validate landmark quality and runtime on target cameras/laptops; retain the backend as a baseline, not a clinical-grade measurement.
- Add a task that associates hand landmarks with cube tracks and derives observable reach, pickup, placement, and stacking intervals.
- Validate event boundaries, missed/ambiguous interactions, and performance across subjects and recording conditions before behavioral summaries are exposed.

**Exit evidence:** landmark performance and runtime are characterized; event outputs are evaluated against independent annotations and include confidence/known failure cases. Timing and finger pose remain proxies for observable behavior, not direct force measurements or diagnoses.

## Stage 4 — broader scene understanding (planned)

- Evaluate scene graphs and segmentation only where they add measurable value over pose plus object tracks.
- Introduce new observation contracts with clear coordinate, identity, confidence, and timing semantics.
- Add model/task combinations incrementally; avoid introducing multiple heavyweight model stacks without runtime evidence.

**Exit evidence:** each capability has a consumer task, a labeled evaluation set, and demonstrated benefit sufficient to justify its dependency and resource costs.

## Stage 5 — user interfaces and deployment targets (planned)

- Improve the local camera/video overlay and session feedback first.
- Treat web and tablet interfaces as separate clients of a stable processing/session boundary, not as assumptions embedded in model code. A browser client on Fly.io, including its analysis panel, is implemented as a prototype.
- Test OS-specific camera access and define CPU, memory, GPU, and latency profiles for supported hardware tiers.

**Exit evidence:** supported device/OS combinations and resource expectations are documented, and interface behavior remains usable when inference is slow or unavailable.

## Cross-cutting gates

Every stage involving human recordings requires appropriate consent and data handling. Any health-related hypothesis remains research-only until a separately defined, independently validated, and appropriately governed clinical pathway exists. A feature that cannot be evaluated reliably stays exploratory or deferred.
