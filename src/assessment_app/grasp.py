"""Pinch-grip detection, cube move segmentation, and the required-finger-pair rule."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import hypot
from typing import Mapping, Sequence

from .contracts import Keypoint
from .cube_detector import Cube

FINGERS = ("index", "middle", "ring", "pinky")
TIPS = {
    "thumb": "thumb_tip",
    "index": "index_finger_tip",
    "middle": "middle_finger_tip",
    "ring": "ring_finger_tip",
    "pinky": "pinky_tip",
}


def pair_label(finger: str | None) -> str:
    return f"thumb+{finger}" if finger else "no pinch"


@dataclass(frozen=True)
class Grip:
    color: str
    finger: str


@dataclass(frozen=True)
class CubeMove:
    """A completed move. Times are session-clock milliseconds; finger features are hand-size normalized."""

    color: str
    start_frame: int
    end_frame: int
    finger: str | None
    start: tuple[int, int]
    end: tuple[int, int]
    grip_onset_ms: float | None = None
    lift_ms: float | None = None
    settle_ms: float | None = None
    release_ms: float | None = None
    end_ms: float | None = None
    vote_share: float | None = None
    coverage: float | None = None
    aperture_onset: float | None = None
    aperture_mean: float | None = None
    path_length: float | None = None
    peak_speed: float | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["start"], data["end"] = list(self.start), list(self.end)
        return data


@dataclass(frozen=True)
class CubeRelease:
    color: str
    frame: int
    t_ms: float


def _bbox_distance(point: Keypoint, bbox) -> float:
    x, y, w, h = bbox
    dx = max(x - point.x, 0.0, point.x - (x + w))
    dy = max(y - point.y, 0.0, point.y - (y + h))
    return hypot(dx, dy)


def _edge_distance(point: Keypoint, bbox) -> float:
    """Distance to the bbox outline; pinching fingertips sit on a cube's sides, not over its middle."""
    x, y, w, h = bbox
    outside = _bbox_distance(point, bbox)
    if outside > 0:
        return outside
    return min(point.x - x, x + w - point.x, point.y - y, y + h - point.y)


def detect_grip(
    hand: Mapping[str, Keypoint],
    cubes: Sequence[Cube],
    contact_ratio: float = 0.35,
    min_contact_px: float = 12.0,
) -> Grip | None:
    """Thumb and at least one fingertip touching the same cube; the finger nearest the cube's outline wins."""
    thumb = hand.get(TIPS["thumb"])
    if thumb is None:
        return None
    best: tuple[float, Grip] | None = None
    for cube in cubes:
        margin = max(min_contact_px, contact_ratio * max(cube.bbox[2], cube.bbox[3]))
        if _bbox_distance(thumb, cube.bbox) > margin:
            continue
        touching = []
        for finger in FINGERS:
            tip = hand.get(TIPS[finger])
            if tip is not None and _bbox_distance(tip, cube.bbox) <= margin:
                touching.append((_edge_distance(tip, cube.bbox), finger))
        if not touching:
            continue
        distance, finger = min(touching)
        score = distance + _edge_distance(thumb, cube.bbox)
        if best is None or score < best[0]:
            best = (score, Grip(cube.color, finger))
    return best[1] if best else None


def hand_scale(hand: Mapping[str, Keypoint]) -> float | None:
    """Wrist to middle-finger knuckle: a camera-distance-independent unit for hand measures."""
    wrist, knuckle = hand.get("wrist"), hand.get("middle_finger_mcp")
    if wrist is None or knuckle is None:
        return None
    scale = hypot(wrist.x - knuckle.x, wrist.y - knuckle.y)
    return scale if scale > 1 else None


class _Window:
    """Everything observed about one cube from first grip (or lift) until it settles."""

    def __init__(self, frame: int) -> None:
        self.start_frame = frame
        self.grip_onset_ms: float | None = None
        self.lift_ms: float | None = None
        self.release_ms: float | None = None
        self.last_grip_frame: int | None = None
        self.votes: Counter = Counter()
        self.frames = 0
        self.hand_frames = 0
        self.apertures: list[float] = []
        self.path: list[tuple[float, float, float]] = []


class MoveTracker:
    """Turn per-color cube positions into completed moves, labelled by the majority pinch seen during the move.

    A move starts when a cube leaves its resting place by more than `move_ratio` of its size (or reappears
    elsewhere after being hidden in the hand) and ends once it stays still for `rest_frames` frames. After a
    move, the first frame with the hand visible but no longer gripping that cube is its release.
    """

    def __init__(
        self,
        move_ratio: float = 0.6,
        still_ratio: float = 0.15,
        rest_frames: int = 6,
        vote_ttl_frames: int = 30,
        release_timeout_ms: float = 3000.0,
    ) -> None:
        self.move_ratio = move_ratio
        self.still_ratio = still_ratio
        self.rest_frames = rest_frames
        self.vote_ttl_frames = vote_ttl_frames
        self.release_timeout_ms = release_timeout_ms
        self._rest: dict[str, tuple[int, int]] = {}
        self._last: dict[str, tuple[int, int]] = {}
        self._moving: set[str] = set()
        self._still: dict[str, int] = {}
        self._settle_ms: dict[str, float] = {}
        self._seen_ms: dict[str, float] = {}
        self._windows: dict[str, _Window] = {}
        self._awaiting_release: dict[str, float] = {}

    def update(
        self,
        cubes: Sequence[Cube],
        grip: Grip | None,
        frame_id: int,
        t_ms: float | None = None,
        hand: Mapping[str, Keypoint] | None = None,
    ) -> tuple[list[CubeMove], list[CubeRelease]]:
        t = frame_id * 1000.0 / 30.0 if t_ms is None else t_ms
        scale = hand_scale(hand) if hand else None
        releases = self._releases(grip, hand, frame_id, t)

        if grip is not None:
            window = self._windows.setdefault(grip.color, _Window(frame_id))
            if window.grip_onset_ms is None:
                window.grip_onset_ms = t
            window.votes[grip.finger] += 1
            window.last_grip_frame = frame_id
            if scale:
                thumb, tip = hand[TIPS["thumb"]], hand[TIPS[grip.finger]]
                window.apertures.append(hypot(thumb.x - tip.x, thumb.y - tip.y) / scale)

        moves = []
        for cube in cubes:
            moves.extend(self._track(cube, frame_id, t))

        for color, window in self._windows.items():
            window.frames += 1
            window.hand_frames += hand is not None
            if color in self._moving and self._still.get(color, 0) > 0:
                if grip is not None and grip.color == color:
                    window.release_ms = None
                elif hand is not None and window.release_ms is None:
                    window.release_ms = t
            if color in self._moving and scale:
                finger = window.votes.most_common(1)[0][0] if window.votes else "index"
                thumb, tip = hand[TIPS["thumb"]], hand[TIPS[finger]]
                window.path.append((t, (thumb.x + tip.x) / 2 / scale, (thumb.y + tip.y) / 2 / scale))
        return moves, releases

    def _releases(self, grip, hand, frame_id: int, t: float) -> list[CubeRelease]:
        releases = []
        for color, deadline in list(self._awaiting_release.items()):
            if hand is not None and (grip is None or grip.color != color):
                releases.append(CubeRelease(color, frame_id, t))
                del self._awaiting_release[color]
            elif t > deadline:
                del self._awaiting_release[color]
        return releases

    def _track(self, cube: Cube, frame_id: int, t: float) -> list[CubeMove]:
        color, center = cube.color, cube.center
        size = max(cube.bbox[2], cube.bbox[3], 1)
        if color not in self._rest:
            self._rest[color] = self._last[color] = center
            return []
        rest = self._rest[color]
        away = hypot(center[0] - rest[0], center[1] - rest[1]) > self.move_ratio * size
        moves = []
        if color not in self._moving:
            window = self._windows.get(color)
            if away:
                self._moving.add(color)
                self._still[color] = 0
                window = self._windows.setdefault(color, _Window(frame_id))
                window.lift_ms = t
                self._awaiting_release.pop(color, None)
            elif window is not None:
                seen = window.last_grip_frame if window.last_grip_frame is not None else window.start_frame
                if frame_id - seen > self.vote_ttl_frames:
                    del self._windows[color]
        if color in self._moving:
            last = self._last[color]
            still = hypot(center[0] - last[0], center[1] - last[1]) <= self.still_ratio * size
            if still:
                if self._still[color] == 0:
                    # The cube arrived in the previous observation; this one only confirms it stopped.
                    self._settle_ms[color] = self._seen_ms.get(color, t)
                self._still[color] += 1
            else:
                self._still[color] = 0
                window = self._windows.get(color)
                if window is not None:
                    window.release_ms = None
            if self._still[color] >= self.rest_frames:
                window = self._windows.pop(color, None) or _Window(frame_id)
                self._moving.discard(color)
                if away:
                    moves.append(self._finish(cube, window, rest, frame_id, t))
                    self._rest[color] = center
                    if window.release_ms is None:
                        self._awaiting_release[color] = t + self.release_timeout_ms
        self._last[color] = center
        self._seen_ms[color] = t
        return moves

    def _finish(self, cube: Cube, window: _Window, rest, frame_id: int, t: float) -> CubeMove:
        total = sum(window.votes.values())
        finger, share = None, None
        if total:
            finger, count = window.votes.most_common(1)[0]
            share = count / total
        path_length, peak_speed = None, None
        if len(window.path) >= 2:
            path_length, peak_speed = 0.0, 0.0
            for (t0, x0, y0), (t1, x1, y1) in zip(window.path, window.path[1:]):
                step = hypot(x1 - x0, y1 - y0)
                path_length += step
                if t1 > t0:
                    peak_speed = max(peak_speed, step / ((t1 - t0) / 1000.0))
        apertures = window.apertures
        return CubeMove(
            color=cube.color,
            start_frame=window.start_frame,
            end_frame=frame_id,
            finger=finger,
            start=rest,
            end=cube.center,
            grip_onset_ms=window.grip_onset_ms,
            lift_ms=window.lift_ms,
            settle_ms=self._settle_ms.get(cube.color),
            release_ms=window.release_ms,
            end_ms=t,
            vote_share=share,
            coverage=window.hand_frames / window.frames if window.frames else None,
            aperture_onset=apertures[0] if apertures else None,
            aperture_mean=sum(apertures) / len(apertures) if apertures else None,
            path_length=path_length,
            peak_speed=peak_speed,
        )


class FingerPairRule:
    """The operator-selected thumb+finger pair a cube must be moved with for the move to count."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.index = 0
        self.placement_ok: dict[str, bool] = {}
        self.last_move: dict[str, tuple[float, bool]] = {}

    @property
    def required(self) -> str:
        return FINGERS[self.index]

    @property
    def label(self) -> str:
        return pair_label(self.required)

    def cycle(self) -> str:
        self.index = (self.index + 1) % len(FINGERS)
        return self.label

    def record(self, move: CubeMove) -> bool:
        ok = not self.enabled or move.finger == self.required
        self.placement_ok[move.color] = ok
        end_ms = move.end_ms if move.end_ms is not None else move.end_frame * 1000.0 / 30.0
        self.last_move[move.color] = (end_ms, ok)
        return ok

    def placed_ok(self, color: str) -> bool:
        return not self.enabled or self.placement_ok.get(color, False)

    def stack_ok(self, stack: Sequence[Cube]) -> bool:
        """Every cube above the base must have been put there with the required pair."""
        return all(self.placed_ok(cube.color) for cube in stack[1:])

    def reset(self) -> None:
        self.placement_ok.clear()
        self.last_move.clear()
