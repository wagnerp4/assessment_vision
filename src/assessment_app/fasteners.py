"""Screw and nut detection, and a turn counter based on the nut's visible flats.

A camera sees only two or three faces of a hex nut at once, and those faces look alike, so the
counter does not try to name each side. It follows the orientation of the visible edges. A hex
repeats every 60 degrees: one flat passing the camera is one side-step, and six side-steps are
one full turn. The count pauses while the nut outline is hidden.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .contracts import FastenerObservation, Frame


class NutTurnTracker:
    """Accumulate nut rotation from a flat angle in degrees, modulo 60."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._previous: float | None = None
        self.unwrapped = 0.0

    def update(self, angle_deg: float | None) -> None:
        if angle_deg is None:
            return
        angle = float(angle_deg) % 60.0
        if self._previous is None:
            self._previous = angle
            return
        delta = (angle - self._previous + 30.0) % 60.0 - 30.0
        self.unwrapped += delta
        self._previous = angle

    @property
    def side_steps(self) -> int:
        return int(abs(self.unwrapped) // 60.0)

    @property
    def turns(self) -> float:
        return abs(self.unwrapped) / 360.0


def _polygon(contour) -> tuple[tuple[int, int], ...]:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    points = approx.reshape(-1, 2) if len(approx) >= 3 else contour.reshape(-1, 2)
    return tuple((int(x), int(y)) for x, y in points)


def _flat_angle(contour) -> float | None:
    """Orientation of the longest visible edge, folded into one hex flat (0–60 degrees)."""
    points = np.array(_polygon(contour), dtype=np.float64)
    if len(points) < 2:
        return None
    best_length = 0.0
    best_angle = None
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        edge = end - start
        length = float(np.hypot(edge[0], edge[1]))
        if length > best_length:
            best_length = length
            best_angle = float(np.degrees(np.arctan2(edge[1], edge[0]))) % 60.0
    return best_angle


def _axis(contour):
    """Long axis of a contour. Returns origin, unit direction, and per-point projection."""
    points = contour.reshape(-1, 2).astype(np.float64)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    direction = None
    best = -1.0
    for index in range(4):
        edge = box[(index + 1) % 4] - box[index]
        length = float(np.hypot(edge[0], edge[1]))
        if length > best:
            best = length
            direction = edge / length
    origin = points.mean(axis=0)
    projection = (points - origin) @ direction
    normal = np.array([-direction[1], direction[0]])
    perpendicular = (points - origin) @ normal
    return origin, direction, projection, perpendicular


def _nut_along_shaft(widths: np.ndarray, centers: np.ndarray, head_at_end: bool, shaft: float) -> float | None:
    """Position of a wide region that is not the screw head. The head is the wide run at one end."""
    if shaft <= 1:
        return None
    wide = widths > shaft * 1.35
    runs: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(wide):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(wide) - 1))
    head_index = len(widths) - 1 if head_at_end else 0
    nuts = [run for run in runs if not (run[0] <= head_index <= run[1])]
    if not nuts:
        return None
    run = max(nuts, key=lambda item: item[1] - item[0])
    return float(np.mean(centers[run[0] : run[1] + 1]))


def _clip_pale_ends(gray: np.ndarray, head: np.ndarray, tip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pull each end in to the last sharp edge.

    Thread ridges and the nut corner stay bright in the gradient when the room gets lighter or
    darker. The shadow past the tip is smooth, so it drops out without a fixed brightness cutoff.
    """
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    magnitude = cv2.magnitude(
        cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3),
    )
    samples = 100
    points = np.linspace(head, tip, samples)
    xs = np.clip(np.rint(points[:, 0]).astype(int), 0, gray.shape[1] - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(int), 0, gray.shape[0] - 1)
    values = magnitude[ys, xs]
    gray_values = gray[ys, xs]
    darker = gray_values[gray_values <= np.median(gray_values)]
    metal_level = float(np.median(darker)) if len(darker) else float(np.median(gray_values))
    limit = max(8.0, float(np.percentile(values, 60)) * 0.45)
    # Sharp and not pale: the shadow can have a soft edge, but it is lighter than the threads.
    sharp = (values >= limit) & (gray_values <= metal_level + 28)
    if int(sharp.sum()) < 8:
        return head, tip
    start = 0
    while start < samples - 1 and not sharp[start]:
        start += 1
    end = samples - 1
    while end > start and not sharp[end]:
        end -= 1
    if end - start < 8:
        return head, tip
    return points[start], points[end]


def _nut_edges(
    gray: np.ndarray,
    center: tuple[float, float] | None,
    shaft: float,
    head: np.ndarray,
    tip: np.ndarray,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """Up to three visible straight edges of the nut. A side view cannot show more than three."""
    if center is None:
        return ()
    radius = int(max(16, shaft * 1.6))
    cx, cy = int(round(center[0])), int(round(center[1]))
    y0, x0 = max(cy - radius, 0), max(cx - radius, 0)
    y1, x1 = min(cy + radius, gray.shape[0] - 1), min(cx + radius, gray.shape[1] - 1)
    crop = gray[y0:y1, x0:x1]
    if crop.shape[0] < 12 or crop.shape[1] < 12:
        return ()
    edges = cv2.Canny(crop, 35, 110)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=14,
        minLineLength=max(8, radius // 3), maxLineGap=5,
    )
    if lines is None:
        return ()
    ranked = []
    for x_a, y_a, x_b, y_b in np.asarray(lines).reshape(-1, 4):
        length = float(np.hypot(x_b - x_a, y_b - y_a))
        midpoint = np.array([(x_a + x_b) / 2, (y_a + y_b) / 2])
        offset = np.array([cx - x0, cy - y0]) - midpoint
        if float(np.hypot(offset[0], offset[1])) > radius * 0.75:
            continue
        segment = (int(x_a + x0), int(y_a + y0), int(x_b + x0), int(y_b + y0))
        axis = tip - head
        axis_length = float(np.hypot(axis[0], axis[1])) or 1.0
        edge = np.array([segment[2] - segment[0], segment[3] - segment[1]], dtype=float)
        edge_length = float(np.hypot(edge[0], edge[1])) or 1.0
        parallel = abs(float(np.dot(edge / edge_length, axis / axis_length)))
        midpoint = np.array([(segment[0] + segment[2]) / 2, (segment[1] + segment[3]) / 2])
        along = float(np.dot(midpoint - head, axis / axis_length))
        closest = head + axis / axis_length * along
        offset_from_axis = float(np.hypot(*(midpoint - closest)))
        # A line along the shaft center is the screw, not a nut flat. Outer flats sit off the axis.
        if parallel > 0.8 and offset_from_axis < max(10.0, shaft * 0.6):
            continue
        ranked.append((length, segment))
    ranked.sort(reverse=True)
    kept: list[tuple[float, tuple[int, int, int, int]]] = []
    for length, segment in ranked:
        angle = float(np.degrees(np.arctan2(segment[3] - segment[1], segment[2] - segment[0]))) % 180.0
        if any(min(abs(angle - previous), 180.0 - abs(angle - previous)) < 20.0 for previous, _ in kept):
            continue
        kept.append((angle, segment))
        if len(kept) == 3:
            break
    return tuple(
        ((segment[0], segment[1]), (segment[2], segment[3]))
        for _, segment in kept
    )


def _width_profile(points: np.ndarray, origin, direction, bins: int = 28):
    """Width of a filled metal region along its long axis. `points` are (x, y) pixels."""
    if len(points) < 4:
        return None
    projection = (points - origin) @ direction
    normal = np.array([-direction[1], direction[0]])
    perpendicular = (points - origin) @ normal
    start, stop = float(projection.min()), float(projection.max())
    if stop - start < 1:
        return None
    edges = np.linspace(start, stop, bins + 1)
    widths = np.zeros(bins)
    centers = np.zeros(bins)
    for index in range(bins):
        if index == bins - 1:
            selected = (projection >= edges[index]) & (projection <= edges[index + 1])
        else:
            selected = (projection >= edges[index]) & (projection < edges[index + 1])
        centers[index] = (edges[index] + edges[index + 1]) / 2
        if int(selected.sum()) >= 2:
            widths[index] = float(perpendicular[selected].max() - perpendicular[selected].min())
    return centers, widths, start, stop


class FastenerDetector:
    """Find one screw and one nut by separating gray metal from a light background, then by shape."""

    def __init__(self) -> None:
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.min_area = 80

    def metal_mask(self, image: np.ndarray) -> np.ndarray:
        """Metal on the bright sheet, only the layer just under the white surface.

        The shadow under this screw is as wide as the part is long, so a plain darkness mask
        swallows the nut. Keeping a short run downward from the sheet leaves the screw and the nut
        and drops the shadow. The black mousepad sits above the sheet and is ignored.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
        sheet = (gray >= 165) & (saturation < 50)
        if float(sheet.mean()) < 0.05:
            sheet = gray >= float(np.percentile(gray, 70))
        height = gray.shape[0]
        rows = np.arange(height, dtype=np.int32)[:, None]
        has_sheet = sheet.any(axis=0)
        first_sheet = np.where(has_sheet, sheet.argmax(axis=0), height)
        below = rows >= first_sheet
        dark = below & (gray < 160)
        has_dark = dark.any(axis=0)
        first_dark = np.where(has_dark, dark.argmax(axis=0), -1)
        distance = rows - first_dark
        metal = (
            has_dark
            & (first_dark >= 0)
            & (distance >= 0)
            & (distance < 48)
            & (gray >= 55)
            & (gray <= 165)
            & (saturation < 80)
        )
        mask = metal.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        return mask

    def detect(self, image: np.ndarray) -> dict:
        mask = self.metal_mask(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = mask.shape[:2]
        contours = [
            contour
            for contour in contours
            if cv2.contourArea(contour) >= self.min_area and not self._on_border(contour, width, height)
        ]
        empty = {
            "screw_tip": None,
            "screw_head": None,
            "screw_polygon": (),
            "nut_center": None,
            "nut_polygon": (),
            "nut_edges": (),
            "nut_angle": None,
            "depth": None,
            "nut_on_screw": False,
        }
        if not contours:
            return empty
        screw = max(contours, key=self._screw_score)
        if self._screw_score(screw) <= 0:
            return empty
        others = [contour for contour in contours if contour is not screw]
        origin, direction, _, _ = _axis(screw)
        component = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(component, [screw], -1, 255, thickness=-1)
        ys, xs = np.where(component > 0)
        profile = _width_profile(np.stack([xs, ys], axis=1).astype(np.float64), origin, direction)
        if profile is None:
            return empty
        centers, widths, start, stop = profile
        head_at_end = self._head_at_high_end(widths)
        tip_t = start if head_at_end else stop
        head_t = stop if head_at_end else start
        tip = origin + direction * tip_t
        head = origin + direction * head_t
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        head, tip = _clip_pale_ends(gray, head, tip)
        shaft = self._shaft_width(widths, head_at_end)
        nut_contour, nut_t, on_screw = self._find_nut(
            others, origin, direction, tip, head, shaft, centers, widths, head_at_end, screw
        )
        depth = None
        nut_center = None
        nut_polygon = ()
        nut_edges = ()
        nut_angle = None
        if nut_t is not None and abs(head_t - tip_t) > 1:
            depth = float((nut_t - tip_t) / (head_t - tip_t))
            nut_center = tuple(float(v) for v in (origin + direction * nut_t))
        if nut_contour is not None:
            moments = cv2.moments(nut_contour)
            if moments["m00"]:
                nut_center = (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))
            nut_polygon = _polygon(nut_contour)
            nut_angle = _flat_angle(nut_contour)
        if nut_center is not None:
            nut_edges = _nut_edges(gray, nut_center, shaft, head, tip)
            if nut_edges and nut_angle is None:
                edge = nut_edges[0]
                span = (edge[1][0] - edge[0][0], edge[1][1] - edge[0][1])
                nut_angle = float(np.degrees(np.arctan2(span[1], span[0]))) % 60.0
        if on_screw and nut_center is not None:
            axis = head - tip
            length2 = float(np.dot(axis, axis))
            if length2 > 1:
                depth = float(np.dot(np.array(nut_center) - tip, axis) / length2)
        if nut_center is not None and not on_screw:
            depth = None
        elif (
            on_screw
            and nut_contour is not None
            and nut_center is not None
            and depth is not None
            and not 0.0 <= depth <= 1.0
        ):
            # The nut sits past the shaft outline (a shadow split them). Stretch the axis to the nut.
            union = np.vstack([screw.reshape(-1, 2), nut_contour.reshape(-1, 2)]).astype(np.float64)
            projection = (union - origin) @ direction
            start, stop = float(projection.min()), float(projection.max())
            head_projection = float((head - origin) @ direction)
            if abs(stop - head_projection) <= abs(start - head_projection):
                head_t, tip_t = stop, start
            else:
                head_t, tip_t = start, stop
            tip = origin + direction * tip_t
            head = origin + direction * head_t
            nut_projection = float((np.array(nut_center) - origin) @ direction)
            depth = float((nut_projection - tip_t) / (head_t - tip_t)) if abs(head_t - tip_t) > 1 else None
        return {
            "screw_tip": (float(tip[0]), float(tip[1])),
            "screw_head": (float(head[0]), float(head[1])),
            "screw_polygon": _polygon(screw),
            "nut_center": nut_center,
            "nut_polygon": nut_polygon,
            "nut_edges": nut_edges,
            "nut_angle": nut_angle,
            "depth": depth,
            "nut_on_screw": bool(on_screw and nut_center is not None),
        }

    @staticmethod
    def _on_border(contour, width: int, height: int) -> bool:
        """A thin strip along the frame edge is the table border, not a part."""
        x, y, box_w, box_h = cv2.boundingRect(contour)
        touches = x <= 1 or y <= 1 or x + box_w >= width - 1 or y + box_h >= height - 1
        if not touches:
            return False
        thin = box_h <= 24 or box_w <= 24
        spans = box_w >= width * 0.75 or box_h >= height * 0.75
        return thin and spans

    @staticmethod
    def _gap(contour, screw) -> float:
        points = np.vstack([contour.reshape(-1, 2), screw.reshape(-1, 2)])
        width = int(points[:, 0].max()) + 2
        height = int(points[:, 1].max()) + 2
        mask = np.zeros((height, width), np.uint8)
        cv2.drawContours(mask, [screw], -1, 255, thickness=-1)
        distance = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 3)
        gaps = [
            float(distance[y, x])
            for x, y in contour.reshape(-1, 2)
            if 0 <= x < width and 0 <= y < height
        ]
        return min(gaps) if gaps else 1e9

    @staticmethod
    def _length(contour) -> float:
        (_, _), (width, height), _ = cv2.minAreaRect(contour)
        return float(max(width, height))

    @staticmethod
    def _screw_score(contour) -> float:
        (_, _), (width, height), _ = cv2.minAreaRect(contour)
        long = float(max(width, height))
        short = max(float(min(width, height)), 1.0)
        aspect = long / short
        # The screw is elongated. A pad edge is a hairline, much thinner than this part.
        if aspect < 2.0 or aspect > 14.0:
            return 0.0
        return float(cv2.contourArea(contour))

    @staticmethod
    def _head_at_high_end(widths: np.ndarray) -> bool:
        count = max(2, len(widths) // 8)
        low = float(np.mean(widths[:count]))
        high = float(np.mean(widths[-count:]))
        return high >= low

    @staticmethod
    def _shaft_width(widths: np.ndarray, head_at_end: bool) -> float:
        count = max(2, len(widths) // 6)
        body = widths[:-count] if head_at_end else widths[count:]
        positive = body[body > 1]
        if len(positive) == 0:
            positive = widths[widths > 1]
        return float(np.median(positive)) if len(positive) else 1.0

    def _find_nut(self, others, origin, direction, tip, head, shaft, centers, widths, head_at_end, screw):
        # A nut already on the shaft is a wide spot in the screw outline. Prefer that over a
        # distant blob, which on this desk is often the curved rim.
        bulge = _nut_along_shaft(widths, centers, head_at_end, shaft)
        if bulge is not None:
            return None, bulge, True
        axis = head - tip
        length = float(np.hypot(axis[0], axis[1])) or 1.0
        unit = axis / length
        best = None
        for contour in others:
            moments = cv2.moments(contour)
            if not moments["m00"]:
                continue
            (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
            short = max(min(rect_w, rect_h), 1.0)
            if max(rect_w, rect_h) / short > 2.8:
                continue
            if self._length(contour) >= length * 0.6:
                continue
            center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
            along = float((center - tip) @ unit)
            distance = float(np.linalg.norm((center - tip) - unit * along))
            fraction = along / length
            touching = self._gap(contour, screw) < 22
            on_shaft = touching or (-0.05 <= fraction <= 1.05 and distance <= max(shaft * 1.6, 18))
            nut_t = float((center - origin) @ direction)
            if best is None or cv2.contourArea(contour) > cv2.contourArea(best[0]):
                best = (contour, nut_t, on_shaft)
        if best is not None:
            return best
        return None, None, False


class FastenerBackend:
    """Classical screw/nut detector. It has no weights and ignores the device argument."""

    capabilities = frozenset({"fasteners"})

    def __init__(self) -> None:
        self.detector = FastenerDetector()

    def load(self, device: str | None = None) -> None:
        del device

    def predict(self, frame: Frame) -> Sequence[FastenerObservation]:
        found = self.detector.detect(frame.image)
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
        return None

    def close(self) -> None:
        return None
