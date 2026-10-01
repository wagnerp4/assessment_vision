"""Tests for the dataset/backend/task integration contracts."""

from pathlib import Path

import numpy as np

from assessment_app.contracts import (
    DatasetSample,
    Frame,
    PoseObservation,
    SceneGraphObservation,
)
from assessment_app.pipeline import VisionPipeline


class FakeDataset:
    def __init__(self):
        self.sample = DatasetSample("sample-1", Path("sample.mp4"))

    def iter_samples(self):
        yield self.sample

    def iter_frames(self, sample):
        for index in (0, 2, 4):
            yield Frame(
                sample.sample_id, index, index * 40.0, np.zeros((4, 4, 3), np.uint8)
            )


class FakeBackend:
    capabilities = frozenset({"test"})

    def __init__(self, observations=()):
        self.loaded = False
        self.reset_count = 0
        self.closed = False
        self.observations = observations

    def load(self, device=None):
        self.loaded = True

    def predict(self, frame):
        return self.observations

    def reset(self):
        self.reset_count += 1

    def close(self):
        self.closed = True


class FakeTask:
    name = "frame_index"

    def __init__(self):
        self.started = []
        self.processed = []
        self.ended = 0

    def begin_sequence(self, sample):
        self.started.append(sample.sample_id)

    def process(self, frame, observations):
        self.processed.append(
            (frame.frame_index, frame.timestamp_ms, tuple(observations))
        )
        return frame.frame_index

    def end_sequence(self):
        self.ended += 1
        return ()


def test_pipeline_runs_timestamped_frames_and_sequence_hooks():
    backend = FakeBackend()
    task = FakeTask()
    pipeline = VisionPipeline(backend, task)

    results = list(pipeline.run_dataset(FakeDataset()))

    assert [item.frame.frame_index for item in results] == [0, 2, 4]
    assert [item.frame.timestamp_ms for item in results] == [0.0, 80.0, 160.0]
    assert [item.task_output for item in results] == [0, 2, 4]
    assert task.started == ["sample-1"]
    assert task.ended == 1
    assert backend.loaded
    assert backend.reset_count == 1

    pipeline.close()
    assert backend.closed


def test_pipeline_requires_sequence_before_processing():
    pipeline = VisionPipeline(FakeBackend(), FakeTask())
    frame = Frame("sample-1", 0, 0.0, np.zeros((2, 2, 3), np.uint8))

    try:
        pipeline.process_frame(frame)
    except RuntimeError as exc:
        assert "begin_sequence" in str(exc)
    else:
        raise AssertionError("processing without a sequence should fail")


def test_pipeline_combines_observations_from_multiple_backends():
    pose = PoseObservation("sample-1", 0, 0.0, ())
    graph = SceneGraphObservation("sample-1", 0, 0.0, (), ())
    pose_backend = FakeBackend((pose,))
    graph_backend = FakeBackend((graph,))
    pipeline = VisionPipeline([pose_backend, graph_backend], FakeTask())

    result = next(pipeline.run_dataset(FakeDataset()))

    assert result.observations == (pose, graph)
    assert pose_backend.loaded and graph_backend.loaded
    assert pose_backend.reset_count == 1 and graph_backend.reset_count == 1
    pipeline.close()
    assert pose_backend.closed and graph_backend.closed
