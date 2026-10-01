"""Built-in model backend adapters."""

from __future__ import annotations

from collections.abc import Sequence

from .contracts import CubeDetections, Frame, ModelObservation
from .cube_detector import CubeDetector


class HSVCubeBackend:
    """Adapt the existing OpenCV HSV detector/tracker to the model contract."""

    capabilities = frozenset({"cube_detection"})

    def __init__(self, detector: CubeDetector | None = None) -> None:
        self.detector = detector or CubeDetector()

    def load(self, device: str | None = None) -> None:
        # The OpenCV implementation has no separate weights or device context.
        del device

    def predict(self, frame: Frame) -> Sequence[ModelObservation]:
        cubes = self.detector.detect_cubes(frame.image, frame.frame_index)
        tracked = self.detector.track_cubes(cubes)
        return (
            CubeDetections(
                frame.sequence_id,
                frame.frame_index,
                frame.timestamp_ms,
                tuple(tracked),
            ),
        )

    def reset(self) -> None:
        self.detector.tracker.tracks.clear()
        self.detector.tracker.next_id = 0
        self.detector.last_masks.clear()

    def close(self) -> None:
        self.reset()
