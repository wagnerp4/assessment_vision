"""Tests for manifest parsing and timestamp-preserving video iteration."""

import json

import numpy as np

from assessment_app import VideoDataset


class FakeCapture:
    def __init__(self, path):
        self.index = 0
        self.frame_count = 5
        self.opened = True

    def isOpened(self):
        return self.opened

    def get(self, property_id):
        import cv2

        return {
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_COUNT: self.frame_count,
            cv2.CAP_PROP_FRAME_WIDTH: 8,
            cv2.CAP_PROP_FRAME_HEIGHT: 6,
            cv2.CAP_PROP_POS_MSEC: 0.0,
        }.get(property_id, 0.0)

    def read(self):
        if self.index >= self.frame_count:
            return False, None
        image = np.zeros((6, 8, 3), dtype=np.uint8)
        self.index += 1
        return True, image

    def release(self):
        self.opened = False


def test_manifest_resolves_video_and_annotation_paths(tmp_path, monkeypatch):
    import assessment_app.datasets as datasets

    monkeypatch.setattr(datasets.cv2, "VideoCapture", FakeCapture)
    video = tmp_path / "clips" / "demo.mp4"
    video.parent.mkdir()
    video.touch()
    annotation = tmp_path / "labels" / "demo.json"
    annotation.parent.mkdir()
    annotation.touch()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "subject-a-session-1",
                "video": "clips/demo.mp4",
                "annotations": "labels/demo.json",
                "metadata": {"source": "local"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sample = next(VideoDataset(manifest).iter_samples())

    assert sample.sample_id == "subject-a-session-1"
    assert sample.video_path == video.resolve()
    assert sample.annotation_path == annotation.resolve()
    assert sample.metadata["fps"] == 25.0
    assert sample.metadata["source"] == "local"


def test_video_dataset_preserves_source_frame_indices_and_timestamps(tmp_path, monkeypatch):
    import assessment_app.datasets as datasets

    monkeypatch.setattr(datasets.cv2, "VideoCapture", FakeCapture)
    video = tmp_path / "demo.mp4"
    video.touch()
    dataset = VideoDataset.from_video(video, frame_stride=2)
    sample = next(dataset.iter_samples())

    frames = list(dataset.iter_frames(sample))

    assert [frame.frame_index for frame in frames] == [0, 2, 4]
    assert [frame.timestamp_ms for frame in frames] == [0.0, 80.0, 160.0]
    assert all(frame.sequence_id == sample.sample_id for frame in frames)
