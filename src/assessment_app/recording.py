"""Real-time screen recording of the annotated view for demos."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def letterbox(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Fit `image` into (width, height) on a dark background without distorting it."""
    width, height = size
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    canvas = np.full((height, width, 3), 17, np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


class RealtimeRecorder:
    """Write frames to a constant-rate MP4 so playback matches wall-clock time.

    Processing runs slower than the video rate, so each frame is repeated for every output slot that
    elapsed since the previous one (by capture timestamp)."""

    def __init__(self, path: Path | str, fps: float = 30.0) -> None:
        self.path = Path(path)
        self.fps = fps
        self.frames_written = 0
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self._slot_ms: float | None = None

    def _open(self, frame: np.ndarray) -> None:
        self._size = (frame.shape[1], frame.shape[0])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self._size)
        if not self._writer.isOpened():
            raise OSError(f"Could not open video writer for {self.path}")

    def _put(self, frame: np.ndarray, count: int = 1) -> None:
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = letterbox(frame, self._size)
        for _ in range(count):
            self._writer.write(frame)
        self.frames_written += count

    def write(self, frame: np.ndarray, t_ms: float) -> None:
        if self._writer is None:
            self._open(frame)
            self._slot_ms = t_ms
            self._put(frame)
            return
        period = 1000.0 / self.fps
        slots = int((t_ms - self._slot_ms) // period)
        if slots > 0:
            self._put(frame, slots)
            self._slot_ms += slots * period

    def hold(self, image: np.ndarray, seconds: float) -> None:
        """Append a still image (e.g. the session summary) for `seconds`."""
        if self._writer is None:
            self._open(image)
        self._put(image, max(1, round(seconds * self.fps)))

    @property
    def duration_s(self) -> float:
        return self.frames_written / self.fps

    def close(self) -> Path | None:
        if self._writer is None:
            return None
        self._writer.release()
        self._writer = None
        return self.path


def draw_rec_badge(frame: np.ndarray, seconds: float) -> np.ndarray:
    """Recording indicator for the on-screen view only (not written to the file)."""
    output = frame.copy()
    minutes, rest = divmod(int(seconds), 60)
    x = output.shape[1] - 150
    cv2.circle(output, (x, 22), 8, (0, 0, 255), -1)
    cv2.putText(output, f"REC {minutes}:{rest:02d}", (x + 16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    return output
