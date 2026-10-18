# Dataset formats and annotations

## Implemented manifest

`VideoDataset` accepts JSONL with one sample per nonblank, non-comment line. The current fields are `sample_id`, `video`, optional `annotations`, and optional object-valued `metadata`. Relative paths resolve beside the manifest. Video metadata (FPS, frame count, width, height) is populated when available. Annotation paths are references only; annotations are not currently parsed or validated by the pipeline.

Example:

```json
{"sample_id":"session-001","video":"videos/session-001.mp4","annotations":"annotations/session-001.json","metadata":{"fps":30}}
```

## Planned annotation conventions

Choose versioned schemas for frame-level objects/keypoints/masks and interval-level actions. Every label should identify its sample and source frame/time; document coordinate space, class/relation/action vocabulary, annotator uncertainty, and missing/occluded labels. Preserve source timing when frames are sampled or resized.

Store large or sensitive videos outside source control and keep manifests relocatable where possible. Record consent/provenance and dataset split metadata; never put identifying data in public examples without authorization. Annotation schema changes require migration notes and updated validation tests.
