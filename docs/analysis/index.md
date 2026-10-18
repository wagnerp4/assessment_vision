# Analysis

Analysis converts model observations and task events into interpretable, time-based summaries.

- **Implemented:** [session analysis](session-analysis.md). Both games log events on a real capture clock; `analysis.py` derives global, per-round, and per-step timings, move phases, finger features, and per-pair comparisons. The local app shows a summary window and saves runs to `runs/`. The web app shows an analysis panel with charts and progression across runs stored in the browser.
- **Planned:** validation of the derived measures against annotated recordings.

All reported features must distinguish direct observations from derived measures, expose uncertainty, and be validated against annotations. The scope is behavioral description and research exploration, not diagnosis or a claim of illness. See [interaction measures](interaction-measures.md) and [evaluation](../training-evaluation/protocols.md).
