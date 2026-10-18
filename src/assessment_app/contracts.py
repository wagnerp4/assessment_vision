"""Library-neutral contracts shared by datasets, model backends, and tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Iterator, Mapping, Protocol, Sequence, TypeVar

import numpy as np

if TYPE_CHECKING:
    from .cube_detector import Cube


@dataclass(frozen=True)
class Frame:
    """A timestamped BGR frame in its original video sequence."""

    sequence_id: str
    frame_index: int
    timestamp_ms: float
    image: np.ndarray


@dataclass(frozen=True)
class DatasetSample:
    """One video sample and its optional annotation sidecar."""

    sample_id: str
    video_path: Path
    annotation_path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Dataset(Protocol):
    """A collection of samples that can stream frames without loading videos at once."""

    def iter_samples(self) -> Iterator[DatasetSample]: ...

    def iter_frames(self, sample: DatasetSample) -> Iterator[Frame]: ...


@dataclass(frozen=True)
class CubeDetections:
    sequence_id: str
    frame_index: int
    timestamp_ms: float
    cubes: tuple[Cube, ...]


@dataclass(frozen=True)
class Keypoint:
    name: str
    x: float
    y: float
    confidence: float | None = None


@dataclass(frozen=True)
class PersonPose:
    person_id: str | None
    body_keypoints: Mapping[str, Keypoint] = field(default_factory=dict)
    left_hand_keypoints: Mapping[str, Keypoint] = field(default_factory=dict)
    right_hand_keypoints: Mapping[str, Keypoint] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass(frozen=True)
class PoseObservation:
    sequence_id: str
    frame_index: int
    timestamp_ms: float
    people: tuple[PersonPose, ...]


@dataclass(frozen=True)
class SceneNode:
    node_id: str
    category: str
    confidence: float
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneEdge:
    source_id: str
    target_id: str
    relation: str
    confidence: float


@dataclass(frozen=True)
class SceneGraphObservation:
    sequence_id: str
    frame_index: int
    timestamp_ms: float
    nodes: tuple[SceneNode, ...]
    edges: tuple[SceneEdge, ...]


@dataclass(frozen=True)
class FastenerObservation:
    """Screw and nut found in one frame. Depth is the nut center's fraction from tip (0) to head (1)."""

    sequence_id: str
    frame_index: int
    timestamp_ms: float
    screw_tip: tuple[float, float] | None = None
    screw_head: tuple[float, float] | None = None
    screw_polygon: tuple[tuple[int, int], ...] = ()
    nut_center: tuple[float, float] | None = None
    nut_polygon: tuple[tuple[int, int], ...] = ()
    nut_edges: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = ()
    nut_angle_deg: float | None = None
    depth: float | None = None
    nut_on_screw: bool = False


@dataclass(frozen=True)
class ActionEvent:
    """A timed, observable interaction; it does not imply a clinical interpretation."""

    sequence_id: str
    action: str
    start_ms: float
    end_ms: float
    confidence: float
    subject_id: str | None = None
    object_id: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)


ModelObservation = CubeDetections | PoseObservation | SceneGraphObservation | FastenerObservation
TaskOutputT = TypeVar("TaskOutputT")


class ModelBackend(Protocol):
    """Loads a model implementation and returns normalized observations."""

    @property
    def capabilities(self) -> frozenset[str]: ...

    def load(self, device: str | None = None) -> None: ...

    def predict(self, frame: Frame) -> Sequence[ModelObservation]: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


class VisionTask(Protocol[TaskOutputT]):
    """Consumes ordered frame observations and keeps sequence-local state."""

    @property
    def name(self) -> str: ...

    def begin_sequence(self, sample: DatasetSample) -> None: ...

    def process(self, frame: Frame, observations: Sequence[ModelObservation]) -> TaskOutputT: ...

    def end_sequence(self) -> Sequence[ActionEvent]: ...


@dataclass(frozen=True)
class PipelineFrame(Generic[TaskOutputT]):
    sample: DatasetSample
    frame: Frame
    observations: tuple[ModelObservation, ...]
    task_output: TaskOutputT
