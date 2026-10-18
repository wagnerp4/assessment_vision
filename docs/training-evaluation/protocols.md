# Training and evaluation protocols

## Training (planned)

Document training separately for each learned backend: dataset/schema version, subject-aware splits, preprocessing/augmentation, model and checkpoint provenance, hyperparameters, random seeds, hardware/software environment, license, and reproducible command. Keep training dependencies optional and never tune on held-out evaluation recordings.

## Evaluation (planned)

Use layered evaluation:

1. **Capability:** detection/localization, tracking continuity, keypoint quality, graph relations, or mask quality as appropriate.
2. **Event/task:** pickup/release timing, object-hand association, stacking correctness, false/missed events, and timing error against annotated intervals.
3. **Runtime:** startup, per-frame latency/throughput, memory, CPU/GPU use, and behavior under dropped/slow frames on declared hardware.

Report dataset size and coverage, confidence/threshold choices, uncertainty, failure cases, and variation across subjects, lighting, viewpoint, and occlusion. Use a fixed regression clip suite for changes, but do not confuse regressions on those clips with independent validation. Preserve machine-readable raw results and a human-readable summary. Set acceptance thresholds before a model comparison rather than selecting them after seeing results.
