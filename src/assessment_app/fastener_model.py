"""Learned screw and nut detector: an exported YOLO ONNX model.

It runs with ONNX Runtime when the optional `detector` extra is installed (about 6x faster on a
laptop CPU), otherwise with OpenCV DNN, which needs nothing beyond OpenCV.

The model finds four boxes: the whole screw, its head, its tip, and the nut. Geometry turns them
into the same FastenerObservation the classical detector produces, so the game does not change.
When the head is hidden in the hand, the screw length comes from the nut size instead
(the nut is NUT_TO_LENGTH of the screw length, measured in calibration).

Training and export live in scripts/train_fastener_detector.py; see docs/models/fastener-detector.md.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .contracts import FastenerObservation, Frame
from .fasteners import FastenerBackend, _nut_edges

CLASSES = ("screw", "screw_head", "screw_tip", "nut")
MODEL_PATH = Path("models") / "fastener_detector.onnx"
# Nut across corners / screw head-to-tip length, from runs/calibration/gameplay_*/measurements.json.
NUT_TO_LENGTH = 0.30
# A nut on the shaft sits on the axis; one held beside it does not.
NUT_OFF_AXIS = 0.6

Box = tuple[float, float, float, float]


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping aspect, pad to a square. Returns the padded image, scale, and (pad_x, pad_y)."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = round(width * scale), round(height * scale)
    pad_x, pad_y = (size - new_w) / 2, (size - new_h) / 2
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, size - new_h - top, left, size - new_w - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, scale, (float(left), float(top))


def decode(
    output: np.ndarray,
    scale: float,
    pad: tuple[float, float],
    *,
    conf: float = 0.35,
    iou: float = 0.5,
) -> list[tuple[int, float, Box]]:
    """Ultralytics YOLO head output (1, 4 + classes, anchors) to (class, score, xyxy) in image pixels."""
    rows = np.squeeze(output, 0).T
    scores = rows[:, 4:]
    classes = scores.argmax(axis=1)
    best = scores[np.arange(len(rows)), classes]
    keep = best >= conf
    rows, classes, best = rows[keep], classes[keep], best[keep]
    if not len(rows):
        return []
    cx = (rows[:, 0] - pad[0]) / scale
    cy = (rows[:, 1] - pad[1]) / scale
    w = rows[:, 2] / scale
    h = rows[:, 3] / scale
    boxes = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)
    found = []
    for cls in np.unique(classes):
        index = np.where(classes == cls)[0]
        kept = cv2.dnn.NMSBoxes(boxes[index].tolist(), best[index].tolist(), conf, iou)
        for k in np.array(kept).reshape(-1):
            x, y, bw, bh = boxes[index[k]]
            found.append((int(cls), float(best[index[k]]), (float(x), float(y), float(x + bw), float(y + bh))))
    return found


def best_per_class(detections: Sequence[tuple[int, float, Box]]) -> dict[str, Box]:
    """The game expects one screw and one nut, so keep the most confident box of each class."""
    chosen: dict[str, tuple[float, Box]] = {}
    for cls, score, box in detections:
        name = CLASSES[cls]
        if name not in chosen or score > chosen[name][0]:
            chosen[name] = (score, box)
    return {name: box for name, (_, box) in chosen.items()}


def _center(box: Box) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def _corners(box: Box) -> np.ndarray:
    return np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])


def _far_corner(box: Box, point: np.ndarray) -> np.ndarray:
    corners = _corners(box)
    return corners[int(np.argmax(np.linalg.norm(corners - point, axis=1)))]


def _polygon(box: Box | None) -> tuple[tuple[int, int], ...]:
    if box is None:
        return ()
    return tuple((int(round(x)), int(round(y))) for x, y in _corners(box))


def geometry(boxes: dict[str, Box], nut_ratio: float = NUT_TO_LENGTH) -> dict:
    """Screw axis, nut position and depth from per-class boxes. Depth is 0 at the tip, 1 at the head."""
    screw, head_box, tip_box, nut_box = (boxes.get(name) for name in CLASSES)
    nut = None if nut_box is None else _center(nut_box)
    nut_size = None if nut_box is None else float(max(nut_box[2] - nut_box[0], nut_box[3] - nut_box[1]))
    tip = None if tip_box is None else _center(tip_box)
    head = None if head_box is None else _center(head_box)
    head_estimated = False
    if tip is not None and head is None and screw is not None:
        # The head is in the hand: aim at the far end of the visible screw.
        head, head_estimated = _far_corner(screw, tip), True
    elif head is not None and tip is None and screw is not None:
        tip = _far_corner(screw, head)
    result = {
        "screw_tip": None,
        "screw_head": None,
        "screw_polygon": _polygon(screw),
        "nut_center": None if nut is None else (float(nut[0]), float(nut[1])),
        "nut_polygon": _polygon(nut_box),
        "nut_edges": (),
        "nut_angle": None,
        "depth": None,
        "nut_on_screw": False,
        "head_estimated": head_estimated,
    }
    if tip is None or head is None:
        return result
    axis = head - tip
    length = float(np.linalg.norm(axis))
    if length < 1.0:
        return result
    unit = axis / length
    if head_estimated and nut_size is not None:
        # The visible part ends at the fingers; the nut gives the true scale.
        length = max(length, nut_size / nut_ratio)
        head = tip + unit * length
    result["screw_tip"] = (float(tip[0]), float(tip[1]))
    result["screw_head"] = (float(head[0]), float(head[1]))
    if nut is None:
        return result
    offset = nut - tip
    along = float(offset @ unit)
    across = abs(float(offset[0] * unit[1] - offset[1] * unit[0]))
    depth = along / length
    on_screw = across <= NUT_OFF_AXIS * (nut_size or 0.0) and -0.02 <= depth <= 1.05
    if on_screw:
        result["depth"] = float(min(1.0, max(0.0, depth)))
        result["nut_on_screw"] = True
    return result


class FastenerSmoother:
    """Steady readings across frames: hold a lost part briefly, smooth points, median the depth."""

    POINTS = ("screw_tip", "screw_head", "nut_center")

    def __init__(self, hold_frames: int = 6, window: int = 5, alpha: float = 0.6) -> None:
        self.hold_frames = hold_frames
        self.window = window
        self.alpha = alpha
        self.reset()

    def reset(self) -> None:
        self._last: dict | None = None
        self._missing = {name: 0 for name in self.POINTS}
        self._depths: deque = deque(maxlen=self.window)
        self._on: deque = deque(maxlen=self.window)

    def update(self, found: dict) -> dict:
        smoothed = dict(found)
        last = self._last or {}
        for name in self.POINTS:
            point, previous = found.get(name), last.get(name)
            if point is not None:
                self._missing[name] = 0
                if previous is not None:
                    point = tuple(self.alpha * a + (1 - self.alpha) * b for a, b in zip(point, previous))
                smoothed[name] = point
            elif previous is not None and self._missing[name] < self.hold_frames:
                self._missing[name] += 1
                smoothed[name] = previous
        nut_seen = found.get("nut_center") is not None
        if nut_seen:
            self._on.append(bool(found.get("nut_on_screw")))
            if found.get("depth") is not None:
                self._depths.append(float(found["depth"]))
        on = sum(self._on) * 2 > len(self._on) if self._on else False
        if smoothed.get("nut_center") is None:
            # Past the hold the nut is gone; do not report a stale "on the screw".
            on = False
            self._on.clear()
        smoothed["nut_on_screw"] = on
        smoothed["depth"] = float(np.median(self._depths)) if on and self._depths else None
        if not on:
            self._depths.clear()
        self._last = smoothed
        return smoothed


def _read_metadata(model_path: Path) -> dict:
    sidecar = model_path.with_suffix(".json")
    return json.loads(sidecar.read_text()) if sidecar.is_file() else {}


class YoloFastenerBackend:
    """Run the exported fastener detector on CPU: ONNX Runtime if installed, else OpenCV DNN."""

    capabilities = frozenset({"fasteners"})

    def __init__(
        self,
        model_path: Path | str = MODEL_PATH,
        *,
        conf: float = 0.35,
        iou: float = 0.5,
        threads: int = 2,
    ) -> None:
        self.model_path = Path(model_path)
        meta = _read_metadata(self.model_path)
        self.input_size = int(meta.get("imgsz", 416))
        self.nut_ratio = float(meta.get("nut_ratio", NUT_TO_LENGTH))
        self.conf = conf
        self.iou = iou
        self.threads = threads
        self.smoother = FastenerSmoother()
        self.runtime: str | None = None
        self._net = None

    def load(self, device: str | None = None) -> None:
        del device
        try:
            import onnxruntime
        except ImportError:
            self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self.runtime = "opencv"
            return
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = self.threads
        self._net = onnxruntime.InferenceSession(
            str(self.model_path), options, providers=["CPUExecutionProvider"]
        )
        self.runtime = "onnxruntime"

    def _forward(self, blob: np.ndarray) -> np.ndarray:
        if self.runtime == "onnxruntime":
            name = self._net.get_inputs()[0].name
            return self._net.run(None, {name: blob})[0]
        self._net.setInput(blob)
        return self._net.forward()

    def detect(self, image: np.ndarray) -> dict:
        if self._net is None:
            # The app adds this backend when screwing is chosen, after the pipeline loaded the others.
            self.load()
        padded, scale, pad = letterbox(image, self.input_size)
        blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, swapRB=True)
        boxes = best_per_class(decode(self._forward(blob), scale, pad, conf=self.conf, iou=self.iou))
        found = self.smoother.update(geometry(boxes, self.nut_ratio))
        center, tip, head = found["nut_center"], found["screw_tip"], found["screw_head"]
        if center is not None and tip is not None and head is not None and found["nut_polygon"]:
            xs = [point[0] for point in found["nut_polygon"]]
            # _nut_edges searches a radius of 1.6 x shaft; the nut is about 2.2 shaft widths across.
            shaft = (max(xs) - min(xs)) / 2.2
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = _nut_edges(gray, center, shaft, np.array(head), np.array(tip))
            found["nut_edges"] = edges
            if edges:
                (x_a, y_a), (x_b, y_b) = edges[0]
                found["nut_angle"] = float(np.degrees(np.arctan2(y_b - y_a, x_b - x_a))) % 60.0
        return found

    def predict(self, frame: Frame) -> Sequence[FastenerObservation]:
        found = self.detect(frame.image)
        return (
            FastenerObservation(
                frame.sequence_id,
                frame.frame_index,
                frame.timestamp_ms,
                screw_tip=found["screw_tip"],
                screw_head=found["screw_head"],
                screw_polygon=found["screw_polygon"],
                nut_center=found["nut_center"],
                nut_polygon=found["nut_polygon"],
                nut_edges=found["nut_edges"],
                nut_angle_deg=found["nut_angle"],
                depth=found["depth"],
                nut_on_screw=found["nut_on_screw"],
            ),
        )

    def reset(self) -> None:
        self.smoother.reset()

    def close(self) -> None:
        self._net = None


def create_fastener_backend(model_path: Path | str = MODEL_PATH):
    """The learned detector when its weights are present, otherwise the classical one."""
    path = Path(model_path)
    return YoloFastenerBackend(path) if path.is_file() else FastenerBackend()
