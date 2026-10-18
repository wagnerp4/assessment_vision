"""Connect a dataset, model backend, and stateful task."""

from __future__ import annotations

from collections.abc import Sequence as RuntimeSequence
from typing import Generic, Iterator, Sequence, TypeVar

from .contracts import (
    Dataset,
    DatasetSample,
    Frame,
    ModelBackend,
    ModelObservation,
    PipelineFrame,
    VisionTask,
)

TaskOutputT = TypeVar("TaskOutputT")


class VisionPipeline(Generic[TaskOutputT]):
    """Run inference and task logic in sequence order, one frame at a time."""

    def __init__(
        self,
        backend: ModelBackend | Sequence[ModelBackend],
        task: VisionTask[TaskOutputT],
    ) -> None:
        self.backends = tuple(backend) if isinstance(backend, RuntimeSequence) else (backend,)
        if not self.backends:
            raise ValueError("At least one model backend is required.")
        self.task = task
        self._loaded = False
        self._active_sample: DatasetSample | None = None

    def begin_sequence(self, sample: DatasetSample, device: str | None = None) -> None:
        if not self._loaded:
            loaded = []
            try:
                for backend in self.backends:
                    backend.load(device=device)
                    loaded.append(backend)
            except Exception:
                for backend in reversed(loaded):
                    backend.close()
                raise
            self._loaded = True
        for backend in self.backends:
            backend.reset()
        self.task.begin_sequence(sample)
        self._active_sample = sample

    def process_frame(self, frame: Frame) -> PipelineFrame[TaskOutputT]:
        if self._active_sample is None:
            raise RuntimeError("Call begin_sequence before processing frames.")
        observations: Sequence[ModelObservation] = tuple(
            observation for backend in self.backends for observation in backend.predict(frame)
        )
        output = self.task.process(frame, observations)
        return PipelineFrame(self._active_sample, frame, tuple(observations), output)

    def end_sequence(self):
        if self._active_sample is None:
            return ()
        events = tuple(self.task.end_sequence())
        self._active_sample = None
        return events

    def run_dataset(self, dataset: Dataset) -> Iterator[PipelineFrame[TaskOutputT]]:
        for sample in dataset.iter_samples():
            self.begin_sequence(sample)
            try:
                for frame in dataset.iter_frames(sample):
                    yield self.process_frame(frame)
            finally:
                self.end_sequence()

    def close(self) -> None:
        if self._active_sample is not None:
            self.end_sequence()
        if self._loaded:
            for backend in reversed(self.backends):
                backend.close()
            self._loaded = False
