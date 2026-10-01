"""Cube detection: per-color HSV segmentation + shape filtering + tracked label voting."""
import cv2
import numpy as np
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from .config import CubeConfig


@dataclass
class Cube:
    color: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]
    area: int
    confidence: float
    frame_id: int
    cube_id: Optional[int] = None

    def get_top_edge(self):
        x, y, w, h = self.bbox
        return (x + w // 2, y)

    def get_bottom_edge(self):
        x, y, w, h = self.bbox
        return (x + w // 2, y + h)

    def get_left_edge(self):
        return self.bbox[0]

    def get_right_edge(self):
        return self.bbox[0] + self.bbox[2]


# OpenCV HSV: H 0-179, S/V 0-255. Red wraps, so it gets two hue intervals.
COLOR_RULES = {
    'red':    {'h': [(0, 8), (160, 179)], 's_min': 70, 'v_min': 60},
    'yellow': {'h': [(14, 36)],           's_min': 55, 'v_min': 90},
    'green':  {'h': [(40, 85)],           's_min': 50, 'v_min': 35},
    'blue':   {'h': [(92, 128)],          's_min': 85, 'v_min': 18},
}


class CubeDetector:
    def __init__(self, config: CubeConfig = None):
        self.config = config or CubeConfig()
        self.rules = getattr(self.config, 'COLOR_RULES', COLOR_RULES)
        self.min_area = self.config.MIN_CONTOUR_AREA
        self.max_area = self.config.MAX_CONTOUR_AREA
        self.min_solidity = 0.80
        self.min_extent = 0.45  # rotated cubes have hexagonal silhouettes (~0.5-0.75)
        self.aspect_range = (0.5, 2.0)
        self.relative_area_range = (0.3, 4.0)  # top cube of a tower shows its top face too
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.tracker = CubeTracker()
        self.last_masks: Dict[str, np.ndarray] = {}
        from .stacking_analyzer import StackingAnalyzer  # lazy: stacking_analyzer imports Cube from here
        self._stack_check = StackingAnalyzer(self.config)

    def color_mask(self, hsv: np.ndarray, color: str) -> np.ndarray:
        rule = self.rules[color]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        hue_ok = np.zeros(h.shape, dtype=bool)
        for lo, hi in rule['h']:
            hue_ok |= (h >= lo) & (h <= hi)
        mask = (hue_ok & (s >= rule['s_min']) & (v >= rule['v_min'])).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        return mask

    def _shape_ok(self, c):
        area = cv2.contourArea(c)
        if not (self.min_area <= area <= self.max_area):
            return None
        x, y, w, h = cv2.boundingRect(c)
        if not (self.aspect_range[0] <= w / h <= self.aspect_range[1]):
            return None
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull_area if hull_area else 0
        extent = area / (w * h)
        if extent < self.min_extent:
            return None
        if solidity < self.min_solidity and not (solidity >= 0.6 and self._top_notch(c, y, h)):
            return None
        return (x, y, w, h), area, min(solidity, extent)

    @staticmethod
    def _top_notch(c, y, h) -> bool:
        """True if the blob's concavity is a notch cut into its top edge: the chevron a cube shows
        when the cube stacked on it hides its top face."""
        hull_idx = cv2.convexHull(c, returnPoints=False)
        if hull_idx is None or len(hull_idx) < 4:
            return False
        try:
            defects = cv2.convexityDefects(c, np.sort(hull_idx, axis=0))
        except cv2.error:
            return False
        if defects is None:
            return False
        defects = np.asarray(defects)
        if defects.size == 0 or defects.size % 4:
            return False
        # OpenCV releases differ: convexityDefects may return (N, 4) or (N, 1, 4).
        defects = defects.reshape(-1, 4)
        s, e, far, depth = max(defects, key=lambda d: d[3])
        far_y = c[far][0][1]
        ends_y = (c[s][0][1] + c[e][0][1]) / 2
        # The deepest pit lies below its two hull endpoints, which sit near the blob's top edge.
        return depth / 256.0 > 0.1 * h and far_y > ends_y and ends_y < y + 0.35 * h

    def _split_by_brightness(self, hsv, contour):
        """A cube fused with a same-hue region of different brightness (shadow, packaging): split by Otsu on V."""
        blob = np.zeros(hsv.shape[:2], np.uint8)
        cv2.drawContours(blob, [contour], -1, 255, -1)
        v = hsv[..., 2]
        vals = v[blob > 0]
        if vals.size < self.min_area or int(vals.max()) - int(vals.min()) < 30:
            return []
        thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours = []
        for part in ((v > thr) & (blob > 0), (v <= thr) & (blob > 0)):
            sub = cv2.morphologyEx(part.astype(np.uint8) * 255, cv2.MORPH_OPEN, self.kernel)
            found, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(found)
        return contours

    def _close_to_size(self, shape, contour):
        """Fill notches (e.g. a shaded cube face) with a closing kernel scaled to the blob's own size."""
        area = cv2.contourArea(contour)
        k = max(7, int(0.2 * np.sqrt(area)) | 1)
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0 = max(0, x - k), max(0, y - k)
        x1, y1 = min(shape[1], x + w + k), min(shape[0], y + h + k)
        blob = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.drawContours(blob, [contour - [x0, y0]], -1, 255, -1)
        blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        found, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [f + [x0, y0] for f in found]

    def _merge_faces(self, contours):
        """A cube's two visible faces often segment as separate blobs split by the dark edge between them:
        merge same-color blobs whose bounding boxes (nearly) touch into one convex outline."""
        parts = [c for c in contours if cv2.contourArea(c) >= 0.2 * self.min_area]
        boxes = [cv2.boundingRect(c) for c in parts]
        parent = list(range(len(parts)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                (x1, y1, w1, h1), (x2, y2, w2, h2) = boxes[i], boxes[j]
                tol = 0.15 * min(max(w1, h1), max(w2, h2))
                gap_x = max(x1, x2) - min(x1 + w1, x2 + w2)
                gap_y = max(y1, y2) - min(y1 + h1, y2 + h2)
                if gap_x <= tol and gap_y <= tol:
                    parent[find(i)] = find(j)
        groups: Dict[int, list] = {}
        for i in range(len(parts)):
            groups.setdefault(find(i), []).append(parts[i])
        out = []
        for g in groups.values():
            if len(g) > 1:
                hull = cv2.convexHull(np.vstack(g))
                if self._shape_ok(hull) is not None:
                    out.append(hull)
                    continue
            out.extend(g)
        return out

    def detect_cubes(self, frame: np.ndarray, frame_id: int) -> List[Cube]:
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        cubes = []
        self.last_masks = {}
        for color in self.rules:
            mask = self.color_mask(hsv, color)
            self.last_masks[color] = mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in self._merge_faces(contours):
                candidates = [c]
                if self._shape_ok(c) is None and cv2.contourArea(c) >= 0.5 * self.min_area:
                    candidates = self._close_to_size(hsv.shape, c)
                    if not any(self._shape_ok(cc) for cc in candidates):
                        candidates = self._split_by_brightness(hsv, c)
                for cc in candidates:
                    ok = self._shape_ok(cc)
                    if ok is None:
                        continue
                    (x, y, w, h), area, quality = ok
                    if x <= 1 or y <= 1 or x + w >= hsv.shape[1] - 1 or y + h >= hsv.shape[0] - 1:
                        quality *= 0.5  # cut off by the image border: partial or background object
                    cubes.append(Cube(color, (x, y, w, h), (x + w // 2, y + h // 2),
                                      int(area), float(quality), frame_id))
        # All cubes are the same physical size, so blobs far off the median size are clutter.
        if len(cubes) >= 3:
            median = float(np.median([c.area for c in cubes]))
            lo, hi = self.relative_area_range
            cubes = [c for c in cubes if lo * median <= c.area <= hi * median]
        return cubes + self._search_stack_neighbors(hsv, cubes, frame_id)

    def _search_stack_neighbors(self, hsv, cubes: List[Cube], frame_id: int) -> List[Cube]:
        """Shaded cubes (e.g. dark blue at the base of a tower) can fall below the global saturation
        threshold. Look directly below/above each found cube, inside its own column, with relaxed thresholds.
        Staying inside the column keeps the tower's sideways shadow out."""
        found = []
        H, W = hsv.shape[:2]

        def on_border(b):
            return b[0] <= 1 or b[1] <= 1 or b[0] + b[2] >= W - 1 or b[1] + b[3] >= H - 1

        for anchor in cubes:
            x, y, w, h = anchor.bbox
            if on_border(anchor.bbox):
                continue
            for y0, y1 in ((y + int(0.5 * h), y + int(1.9 * h)), (y - int(0.9 * h), y + int(0.5 * h))):
                x0, x1 = max(0, x + int(0.05 * w)), min(W, x + int(0.95 * w))
                y0, y1 = max(0, y0), min(H, y1)
                if x1 - x0 < 10 or y1 - y0 < 10:
                    continue
                roi = hsv[y0:y1, x0:x1]
                for color, rule in self.rules.items():
                    # Skip colors already present in this column (e.g. the edge of the real cube above).
                    if any(c.color == color and x - 0.5 * w <= c.center[0] <= x + 1.5 * w
                           and y - 2.5 * h <= c.center[1] <= y + 3 * h for c in cubes + found):
                        continue
                    relaxed = dict(rule, s_min=int(rule['s_min'] * 0.5))
                    h_ok = np.zeros(roi.shape[:2], bool)
                    for lo, hi in relaxed['h']:
                        h_ok |= (roi[..., 0] >= lo) & (roi[..., 0] <= hi)
                    m = (h_ok & (roi[..., 1] >= relaxed['s_min']) & (roi[..., 2] >= relaxed['v_min'])).astype(np.uint8) * 255
                    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)
                    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.kernel, iterations=2)
                    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not cs:
                        continue
                    c = max(cs, key=cv2.contourArea)
                    area = cv2.contourArea(c)
                    bx, by, bw, bh = cv2.boundingRect(c)
                    # Must be cube-sized and span most of the column's width.
                    if area < 0.35 * anchor.area or bw < 0.6 * (x1 - x0):
                        continue
                    bx, by = bx + x0, by + y0
                    cand = Cube(color, (bx, by, bw, bh), (bx + bw // 2, by + bh // 2), int(area), 0.6, frame_id)
                    if on_border(cand.bbox):
                        continue
                    lower, upper = (cand, anchor) if cand.center[1] > anchor.center[1] else (anchor, cand)
                    if not self._stack_check.analyze_alignment(lower, upper).is_aligned:
                        continue
                    found.append(cand)
        return found

    def track_cubes(self, cubes: List[Cube]) -> List[Cube]:
        return self.tracker.update(cubes)


class _Track:
    def __init__(self, tid: int, cube: Cube):
        self.id = tid
        self.cube = cube
        self.hits = 1
        self.misses = 0
        self.votes = deque([cube.color], maxlen=15)
        self.quality = deque([cube.confidence], maxlen=15)

    @property
    def label(self) -> str:
        return Counter(self.votes).most_common(1)[0][0]


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter) if inter else 0.0


class CubeTracker:
    """IoU + centroid Hungarian matching; tracks need `min_hits` to show and survive `max_misses` gaps."""

    def __init__(self, max_dist: float = 120, max_misses: int = 10, min_hits: int = 3,
                 max_coast: int = 4, smooth: float = 0.5):
        self.max_dist = max_dist
        self.max_coast = max_coast
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.smooth = smooth
        self.tracks: List[_Track] = []
        self.next_id = 0

    def update(self, cubes: List[Cube]) -> List[Cube]:
        matched_t, matched_c = set(), set()
        if self.tracks and cubes:
            cost = np.full((len(self.tracks), len(cubes)), 1e6)
            for i, t in enumerate(self.tracks):
                for j, c in enumerate(cubes):
                    d = np.hypot(t.cube.center[0] - c.center[0], t.cube.center[1] - c.center[1])
                    if d > self.max_dist:
                        continue
                    cost[i, j] = d * (1.0 - _iou(t.cube.bbox, c.bbox)) + (0 if t.label == c.color else 40)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] >= 1e6:
                    continue
                t, c = self.tracks[i], cubes[j]
                t.cube = self._smooth(t.cube, c)
                t.votes.append(c.color)
                t.quality.append(c.confidence)
                t.hits += 1
                t.misses = 0
                matched_t.add(i)
                matched_c.add(j)

        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        for j, c in enumerate(cubes):
            if j not in matched_c:
                self.tracks.append(_Track(self.next_id, c))
                self.next_id += 1

        # Exactly one physical cube per color: keep the most established track of each color.
        best: Dict[str, _Track] = {}
        for t in self.tracks:
            if t.hits < self.min_hits or t.misses > self.max_coast:
                continue
            cur = best.get(t.label)
            if cur is None or self._rank(t) > self._rank(cur):
                best[t.label] = t
        out = []
        for t in best.values():
            c = t.cube
            c.color = t.label
            c.cube_id = t.id
            out.append(c)
        return out

    def _rank(self, t: _Track):
        # Age is capped so a cube that was just put back down can beat a long-lived look-alike.
        return (min(t.hits, 2 * self.min_hits), -t.misses, float(np.mean(t.quality)))

    def _smooth(self, old: Cube, new: Cube) -> Cube:
        a = self.smooth
        bx = tuple(int(round(a * o + (1 - a) * n)) for o, n in zip(old.bbox, new.bbox))
        new.bbox = bx
        new.center = (bx[0] + bx[2] // 2, bx[1] + bx[3] // 2)
        return new
