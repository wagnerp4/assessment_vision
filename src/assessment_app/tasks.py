"""Application tasks built from normalized model observations."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from math import hypot
from typing import Sequence

from .config import CubeConfig
from .contracts import (
    ActionEvent,
    CubeDetections,
    DatasetSample,
    FastenerObservation,
    Frame,
    ModelObservation,
    PoseObservation,
)
from .cube_detector import Cube
from .fasteners import NutTurnTracker
from .game_logic import GameDifficulty, GameEngine, GameMode
from .grasp import TIPS, FingerPairRule, Grip, MoveTracker, detect_grip, pair_label
from .random_stacking import RandomStackingEngine
from .screwing import ScrewingEngine, ScrewObservation, thread_progress
from .stacking_analyzer import StackingAnalyzer

QUALITY_WINDOW_MS = 3000.0
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StackingFrameResult:
    cubes: tuple[Cube, ...]
    stacks: tuple[tuple[Cube, ...], ...]
    game_result: dict
    game_status: dict


class CubeStackingTask:
    """Apply stack geometry and the assessment-game rules to cube tracks and, if available, hand landmarks."""

    name = "cube_stacking"

    def __init__(
        self,
        config: CubeConfig | None = None,
        difficulty: GameDifficulty = GameDifficulty.HARD,
        *,
        game_mode: GameMode = GameMode.SIMPLE,
        playing_hand: str = "right",
        hand_backend: str | None = None,
        finger_rule: bool = False,
        rounds: int = 4,
        seed: int | None = None,
    ) -> None:
        self.config = config or CubeConfig()
        self.difficulty = difficulty
        self.game_mode = game_mode
        self.playing_hand = playing_hand
        self.hand_backend = hand_backend
        self.rounds = rounds
        self.seed = seed
        self.pair_rule = FingerPairRule(enabled=finger_rule and hand_backend is not None)
        self.analyzer = StackingAnalyzer(self.config)
        self.moves = MoveTracker()
        self.grip: Grip | None = None
        self.game_engine = self._make_engine()
        self._new_session()

    def _make_engine(self):
        if self.game_mode == GameMode.RANDOM:
            return RandomStackingEngine(self.config, self.pair_rule, self.rounds, self.seed)
        return GameEngine(self.config, self.pair_rule)

    def _new_session(self) -> None:
        self.session_id = uuid.uuid4().hex
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._rule_logged = False
        self._last_t = 0.0
        self._last_frame = 0
        self._quality = [None, 0, 0]  # window start ms, frames, frames with a hand

    def begin_sequence(self, sample: DatasetSample) -> None:
        self.analyzer = StackingAnalyzer(self.config)
        self.moves = MoveTracker()
        self.grip = None
        self.pair_rule.reset()
        if self.game_engine.get_game_status()["mode"] != self.game_mode.value:
            self.game_engine = self._make_engine()
        self.game_engine.start_game(self.difficulty)
        self._new_session()

    def set_finger_rule(self, enabled: bool) -> None:
        self.pair_rule.enabled = enabled and self.hand_backend is not None
        self._log_rule()

    def next_pair(self) -> None:
        if self.pair_rule.enabled:
            self.pair_rule.cycle()
            self._log_rule()

    def _log_rule(self) -> None:
        self.game_engine.log_event(
            "rule_changed",
            {
                "enabled": self.pair_rule.enabled,
                "required": self.pair_rule.label if self.pair_rule.enabled else None,
            },
            self._last_t,
            self._last_frame,
        )

    def _hand(self, observations: Sequence[ModelObservation]):
        for observation in observations:
            if isinstance(observation, PoseObservation):
                for person in observation.people:
                    hand = getattr(person, f"{self.playing_hand}_hand_keypoints")
                    if hand:
                        return hand
        return None

    def process(
        self, frame: Frame, observations: Sequence[ModelObservation]
    ) -> StackingFrameResult:
        t, index = frame.timestamp_ms, frame.frame_index
        cube_observation = next(
            (item for item in observations if isinstance(item, CubeDetections)), None
        )
        cubes = list(cube_observation.cubes) if cube_observation else []
        hand = self._hand(observations)
        self.grip = detect_grip(hand, cubes) if hand else None
        moves, releases = self.moves.update(cubes, self.grip, index, t, hand)
        counted = [self.pair_rule.record(move) for move in moves]
        stacks = self.analyzer.find_stacks(cubes)
        game_result = self.game_engine.process_frame(cubes, index, tuple(moves), t)

        engine = self.game_engine
        self._last_t, self._last_frame = t, index
        if not self._rule_logged:
            self._rule_logged = True
            self._log_rule()
        for move, ok in zip(moves, counted):
            engine.log_event(
                "cube_moved",
                {
                    **move.to_dict(),
                    "required": self.pair_rule.label if self.pair_rule.enabled else None,
                    "counted": ok,
                },
                t,
                index,
            )
        for release in releases:
            engine.log_event("cube_released", {"color": release.color}, release.t_ms, release.frame)
        self._sample_quality(t, index, hand is not None)

        status = engine.get_game_status()
        status["required_pair"] = self.pair_rule.label if self.pair_rule.enabled else None
        status["grip"] = (
            f"{pair_label(self.grip.finger)} on {self.grip.color}" if self.grip else None
        )
        return StackingFrameResult(
            tuple(cubes),
            tuple(tuple(stack) for stack in stacks),
            game_result,
            status,
        )

    def _sample_quality(self, t: float, index: int, hand_seen: bool) -> None:
        window = self._quality
        if window[0] is None:
            window[:] = [t, 0, 0]
        window[1] += 1
        window[2] += hand_seen
        elapsed = t - window[0]
        if elapsed >= QUALITY_WINDOW_MS:
            self.game_engine.log_event(
                "quality",
                {
                    "fps": round(window[1] * 1000.0 / elapsed, 2),
                    "hand_coverage": round(window[2] / window[1], 3)
                    if self.hand_backend
                    else None,
                },
                t,
                index,
            )
            window[:] = [t, 0, 0]

    def session_record(self, client: str) -> dict:
        """Metadata plus the ordered event log; see docs/analysis/session-analysis.md."""
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "client": client,
            "started_at": self.started_at,
            "game": {
                "mode": self.game_mode.value,
                "difficulty": self.difficulty.name,
                "rounds": self.rounds if self.game_mode == GameMode.RANDOM else None,
                "seed": self.seed,
            },
            "hands": {
                "enabled": self.hand_backend is not None,
                "backend": self.hand_backend,
                "playing_hand": self.playing_hand,
            },
            "events": list(self.game_engine.event_log),
        }

    def end_sequence(self) -> Sequence[ActionEvent]:
        return ()


def _distance_to_segment(point, start, end) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    abx, aby = bx - ax, by - ay
    length2 = abx * abx + aby * aby
    if length2 < 1:
        return hypot(px - ax, py - ay)
    along = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / length2))
    return hypot(px - (ax + along * abx), py - (ay + along * aby))


class ScrewingTask:
    """Watch one screw and one nut, count nut turns, and run the screwing game."""

    name = "screwing"

    def __init__(
        self,
        config: CubeConfig | None = None,
        difficulty: GameDifficulty = GameDifficulty.HARD,
        *,
        playing_hand: str = "right",
        hand_backend: str | None = None,
        finger_rule: bool = False,
    ) -> None:
        self.config = config or CubeConfig()
        self.difficulty = difficulty
        self.game_mode = GameMode.SCREWING
        self.playing_hand = playing_hand
        self.hand_backend = hand_backend
        self.rounds = None
        self.seed = None
        self.pair_rule = FingerPairRule(enabled=finger_rule and hand_backend is not None)
        self.analyzer = StackingAnalyzer(self.config)
        self.game_engine = ScrewingEngine()
        self.turns = NutTurnTracker()
        self._new_session()

    def _new_session(self) -> None:
        self.session_id = uuid.uuid4().hex
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._rule_logged = False
        self._last_t = 0.0
        self._last_frame = 0
        self._quality = [None, 0, 0]
        # Where the screw and nut lay during setup; pickup needs both moved away from there.
        self._rest: dict[str, tuple[float, float]] = {}

    def begin_sequence(self, sample: DatasetSample) -> None:
        del sample
        self.turns.reset()
        self.pair_rule.reset()
        self.game_engine.start_game(self.difficulty)
        self._new_session()

    def set_finger_rule(self, enabled: bool) -> None:
        self.pair_rule.enabled = enabled and self.hand_backend is not None

    def next_pair(self) -> None:
        if self.pair_rule.enabled:
            self.pair_rule.cycle()

    def _hands(self, observations: Sequence[ModelObservation]):
        for observation in observations:
            if isinstance(observation, PoseObservation) and observation.people:
                person = observation.people[0]
                return person.left_hand_keypoints, person.right_hand_keypoints
        return {}, {}

    def _near_screw(self, hand, tip, head) -> bool:
        if not hand or tip is None or head is None:
            return False
        length = hypot(head[0] - tip[0], head[1] - tip[1])
        margin = max(48.0, length * 0.22)
        return any(
            _distance_to_segment((point.x, point.y), tip, head) <= margin
            for name, point in hand.items()
            if name in TIPS.values()
        )

    def _near_nut(self, hand, center, polygon) -> bool:
        if not hand or center is None:
            return False
        if polygon:
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            size = max(max(xs) - min(xs), max(ys) - min(ys), 1)
        else:
            size = 36
        margin = max(40.0, size * 1.4)
        return any(
            hypot(point.x - center[0], point.y - center[1]) <= margin
            for name, point in hand.items()
            if name in TIPS.values()
        )

    @staticmethod
    def _nut_size(center, polygon) -> float:
        if not polygon:
            return 36.0
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        return max(max(xs) - min(xs), max(ys) - min(ys), 1.0)

    def _parts_lifted(self, tip, head, nut_center, nut_size) -> bool:
        """Remember the parts' resting places during setup; afterwards both must have moved off them."""
        screw_center = None if tip is None or head is None else ((tip[0] + head[0]) / 2, (tip[1] + head[1]) / 2)
        if self.game_engine.phase == "setup":
            if screw_center is not None:
                self._rest["screw"] = screw_center
            if nut_center is not None:
                self._rest["nut"] = nut_center
            return False
        if screw_center is None or nut_center is None or len(self._rest) < 2:
            return False
        length = hypot(head[0] - tip[0], head[1] - tip[1])
        screw_moved = hypot(screw_center[0] - self._rest["screw"][0], screw_center[1] - self._rest["screw"][1])
        nut_moved = hypot(nut_center[0] - self._rest["nut"][0], nut_center[1] - self._rest["nut"][1])
        return screw_moved >= max(40.0, 0.35 * length) and nut_moved >= max(30.0, 1.5 * nut_size)

    @staticmethod
    def _nut_clear(tip, head, nut_center, nut_size) -> bool:
        """Nut center at least one nut width past the tip, or one nut width beside the shaft."""
        if tip is None or head is None or nut_center is None:
            return False
        ax, ay = head[0] - tip[0], head[1] - tip[1]
        length = hypot(ax, ay)
        if length < 1:
            return False
        ux, uy = ax / length, ay / length
        dx, dy = nut_center[0] - tip[0], nut_center[1] - tip[1]
        along = dx * ux + dy * uy
        beside = abs(dx * uy - dy * ux)
        return along <= -nut_size or beside >= nut_size

    @staticmethod
    def _nut_gripped(hands, nut_center, nut_size) -> bool | None:
        """Thumb or index tip of either hand on the nut (left/right labels swap now and then).
        None when no hand or no nut is seen."""
        hands = [hand for hand in hands if hand]
        if not hands or nut_center is None:
            return None
        reach = max(25.0, 0.9 * nut_size)
        return any(
            hypot(hand[name].x - nut_center[0], hand[name].y - nut_center[1]) <= reach
            for hand in hands
            for name in (TIPS["thumb"], TIPS["index"])
            if name in hand
        )

    def process(self, frame: Frame, observations: Sequence[ModelObservation]) -> StackingFrameResult:
        t, index = frame.timestamp_ms, frame.frame_index
        found = next((item for item in observations if isinstance(item, FastenerObservation)), None)
        self.turns.update(None if found is None else found.nut_angle_deg)
        left, right = self._hands(observations)
        hands_seen = bool(left or right)
        screw_visible = found is not None and found.screw_tip is not None and found.screw_head is not None
        nut_visible = found is not None and found.nut_center is not None
        tip = None if found is None else found.screw_tip
        head = None if found is None else found.screw_head
        nut_center = None if found is None else found.nut_center
        nut_polygon = () if found is None else found.nut_polygon
        hands_on_parts = any(
            self._near_screw(hand, tip, head) or self._near_nut(hand, nut_center, nut_polygon)
            for hand in (left, right)
        )
        nut_size = self._nut_size(nut_center, nut_polygon)
        sample = ScrewObservation(
            screw_visible=screw_visible,
            nut_visible=nut_visible,
            depth=None if found is None else thread_progress(found.depth),
            nut_on_screw=bool(found and found.nut_on_screw),
            turns=self.turns.turns,
            side_steps=self.turns.side_steps,
            left_on_screw=self._near_screw(left, tip, head),
            right_on_nut=self._near_nut(right, nut_center, nut_polygon),
            hands_seen=hands_seen,
            hands_on_parts=hands_on_parts,
            parts_lifted=self._parts_lifted(tip, head, nut_center, nut_size),
            nut_clear=self._nut_clear(tip, head, nut_center, nut_size),
            nut_gripped=self._nut_gripped((left, right), nut_center, nut_size),
        )
        game_result = self.game_engine.process_frame(sample, index, t)
        self._last_t, self._last_frame = t, index
        self._sample_quality(t, index, hands_seen)
        status = self.game_engine.get_game_status()
        status["screw_tip"] = None if found is None else found.screw_tip
        status["screw_head"] = None if found is None else found.screw_head
        status["screw_polygon"] = [] if found is None else [list(point) for point in found.screw_polygon]
        status["nut_center"] = None if found is None else found.nut_center
        status["nut_polygon"] = [] if found is None else [list(point) for point in found.nut_polygon]
        status["nut_edges"] = [] if found is None else [
            [list(edge[0]), list(edge[1])] for edge in found.nut_edges
        ]
        return StackingFrameResult((), (), game_result, status)

    def _sample_quality(self, t: float, index: int, hand_seen: bool) -> None:
        window = self._quality
        if window[0] is None:
            window[:] = [t, 0, 0]
        window[1] += 1
        window[2] += int(hand_seen)
        elapsed = t - window[0]
        if elapsed >= QUALITY_WINDOW_MS:
            self.game_engine.log_event(
                "quality",
                {
                    "fps": round(window[1] * 1000.0 / elapsed, 2),
                    "hand_coverage": round(window[2] / window[1], 3) if self.hand_backend else None,
                },
                t,
                index,
            )
            window[:] = [t, 0, 0]

    def session_record(self, client: str) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "client": client,
            "started_at": self.started_at,
            "game": {
                "mode": GameMode.SCREWING.value,
                "difficulty": self.difficulty.name,
                "rounds": None,
                "seed": None,
                "target_depth": self.game_engine.target,
            },
            "hands": {
                "enabled": self.hand_backend is not None,
                "backend": self.hand_backend,
                "playing_hand": self.playing_hand,
            },
            "events": list(self.game_engine.event_log),
        }

    def end_sequence(self) -> Sequence[ActionEvent]:
        return ()
