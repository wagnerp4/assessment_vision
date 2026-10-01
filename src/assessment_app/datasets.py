"""Local video datasets with optional JSONL annotation references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import cv2

from .contracts import DatasetSample, Frame


class VideoDataset:
    """Read local videos from a JSONL manifest or as one standalone sample.

    Each manifest line contains ``sample_id``, ``video``, and optionally
    ``annotations`` and ``metadata``. Relative paths are resolved beside the
    manifest. Frames retain source frame indices and timestamps when striding.
    """

    def __init__(
        self,
        manifest_path: Path | str | None = None,
        *,
        sample: DatasetSample | None = None,
        frame_stride: int = 1,
        resize_height: int | None = None,
    ) -> None:
        if (manifest_path is None) == (sample is None):
            raise ValueError("Provide exactly one of manifest_path or sample.")
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1.")
        self.manifest_path = Path(manifest_path).resolve() if manifest_path else None
        self._sample = sample
        self.frame_stride = frame_stride
        self.resize_height = resize_height

    @classmethod
    def from_video(
        cls,
        video_path: Path | str,
        *,
        sample_id: str | None = None,
        annotation_path: Path | str | None = None,
        frame_stride: int = 1,
        resize_height: int | None = None,
    ) -> "VideoDataset":
        path = Path(video_path).resolve()
        annotation = Path(annotation_path).resolve() if annotation_path else None
        sample = DatasetSample(sample_id or path.stem, path, annotation)
        return cls(sample=sample, frame_stride=frame_stride, resize_height=resize_height)

    def iter_samples(self) -> Iterator[DatasetSample]:
        if self._sample is not None:
            yield self._with_video_metadata(self._sample)
            return

        assert self.manifest_path is not None
        with self.manifest_path.open("r", encoding="utf-8") as manifest:
            for line_number, line in enumerate(manifest, start=1):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                try:
                    entry = json.loads(line)
                    sample_id = str(entry["sample_id"])
                    video_path = self._resolve_path(entry["video"])
                    annotation_value = entry.get("annotations")
                    annotation_path = (
                        self._resolve_path(annotation_value) if annotation_value else None
                    )
                    metadata = entry.get("metadata") or {}
                    if not isinstance(metadata, dict):
                        raise ValueError("metadata must be a JSON object")
                except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid dataset manifest entry at line {line_number}: {exc}"
                    ) from exc
                sample = DatasetSample(sample_id, video_path, annotation_path, metadata)
                yield self._with_video_metadata(sample)

    def iter_frames(self, sample: DatasetSample) -> Iterator[Frame]:
        if not sample.video_path.is_file():
            raise FileNotFoundError(f"Video not found: {sample.video_path}")
        capture = cv2.VideoCapture(str(sample.video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open video: {sample.video_path}")

        fps = float(sample.metadata.get("fps", 0.0))
        frame_index = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if frame_index % self.frame_stride == 0:
                    timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                    if timestamp_ms <= 0 and frame_index > 0 and fps > 0:
                        timestamp_ms = frame_index * 1000.0 / fps
                    if self.resize_height and image.shape[0] > self.resize_height:
                        scale = self.resize_height / image.shape[0]
                        image = cv2.resize(
                            image,
                            (int(image.shape[1] * scale), self.resize_height),
                        )
                    yield Frame(sample.sample_id, frame_index, timestamp_ms, image)
                frame_index += 1
        finally:
            capture.release()

    def _resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            assert self.manifest_path is not None
            path = self.manifest_path.parent / path
        return path.resolve()

    @staticmethod
    def _with_video_metadata(sample: DatasetSample) -> DatasetSample:
        if not sample.video_path.is_file():
            raise FileNotFoundError(f"Video not found: {sample.video_path}")
        capture = cv2.VideoCapture(str(sample.video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open video: {sample.video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        metadata = dict(sample.metadata)
        metadata.setdefault("fps", fps)
        metadata.setdefault("frame_count", frame_count)
        metadata.setdefault("width", width)
        metadata.setdefault("height", height)
        return DatasetSample(sample.sample_id, sample.video_path, sample.annotation_path, metadata)
