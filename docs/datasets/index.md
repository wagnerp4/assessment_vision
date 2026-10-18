# Datasets

`VideoDataset` currently streams local video files, either as one sample or through a JSONL manifest. Each `DatasetSample` has an ID, video path, optional annotation-sidecar path, and metadata. Frames retain source frame indices/timestamps; optional stride and resize settings affect sampling and image size.

## Planned data foundation

Document annotation formats before collecting training/evaluation sets. Keep raw media, derived labels, and generated outputs distinguishable; make references reproducible without assuming that sensitive videos belong in Git. Define consent, access, retention, de-identification, and sharing rules before human-subject recordings are used.

See [formats and annotations](formats-and-annotations.md), [evaluation protocols](../training-evaluation/protocols.md), and [roadmap](../roadmap.md).
