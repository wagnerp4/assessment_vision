# Assessment Vision documentation

This documentation is contributor-oriented. It describes what exists today, the boundaries between components, and the sequence for growing the project. Each page distinguishes implemented behavior from planned work; a documented idea is not necessarily an implemented feature.

## Project map

- [Roadmap](roadmap.md) — staged capabilities, dependencies, and readiness criteria.
- [Architecture](architecture/overview.md) — end-to-end data flow and component responsibilities.
- [Contracts](architecture/contracts.md) — datasets, frames, normalized observations, backends, tasks, and pipeline lifecycle.
- [Perception](perception/index.md) — visual capabilities that produce observations.
- [Models](models/index.md) — backend adapters and runtime/model-library boundaries.
- [Datasets](datasets/index.md) — video samples, metadata, annotations, and data stewardship.
- [Training and evaluation](training-evaluation/index.md) — model training and system validation.
- [Analysis](analysis/index.md) — observable interaction measures and their limits.
- [Application](application/tasks-and-game-logic.md) — stacking and future user-facing tasks; [interfaces](application/interfaces/index.md).
- [Devices](devices/index.md) — cameras, operating systems, and constrained hardware.

## Current status at a glance

The implemented path reads local video or camera frames, adapts the HSV cube detector/tracker to the `ModelBackend` contract, and feeds normalized cube observations into the stateful cube-stacking task.

- **Hand landmarks:** optional MediaPipe backends (a dedicated hand model, or Holistic) provide them.
- **Game logic:** the task uses them for pinch-grip and cube-move detection, an optional finger-pair rule, and two game modes: Simple and Random stacking.
- **Web client:** runs the same pipeline from a browser camera and is deployed on Fly.io.
- **Session analysis:** timings, move phases, and finger-pair comparisons, shown in a local summary window and a web analysis panel ([session analysis](analysis/session-analysis.md)).
- **Screw and nut detector:** dataset, training, and export tooling for a learned detector; no trained weights yet ([concept](models/fastener-detector.md)).
- **Not implemented:** annotation parsing, validation of the derived measures, scene-graph inference, tablet interfaces, and clinical validation.

## Documentation conventions

- Label capability descriptions as **Implemented**, **Planned**, or **Deferred**.
- Put shared component responsibilities in architecture pages; put domain-specific algorithms and validation in their domain pages.
- Link related pages with relative Markdown links. Keep this index as the canonical navigation entry point; no generated documentation site is required.
- Report observable model outputs and uncertainty. Do not present behavior features as diagnoses, and do not infer physical grip force from video timing or finger pose alone.
