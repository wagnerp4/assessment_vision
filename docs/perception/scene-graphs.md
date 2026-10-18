# Scene graph generation

**Status: planned.** `SceneNode`, `SceneEdge`, and `SceneGraphObservation` can represent entities and relations, but no current backend generates scene graphs.

## Candidate role

A graph could summarize relations such as `hand near cube`, `cube on cube`, or `person reaches toward object`. It should add value beyond explicit pose, detection, tracking, and task-state logic; do not add a graph model merely to duplicate signals already available to a task.

## Contract and evaluation questions

Before implementation, settle node identity lifetime, relation vocabulary, directionality, confidence calibration, timestamps, and how unknown/occluded relations are represented. Evaluate node/edge precision and recall and downstream event/task improvement against a simpler baseline. Include temporal consistency and subject/object association in error analysis.

Scene relations describe model predictions, not intent, illness, or causality. Keep these interpretations out of the observation contract.
