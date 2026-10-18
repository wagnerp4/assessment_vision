"""Screwing: pick up the screw and nut, screw the nut to a depth set by the difficulty, unscrew it,
and put both parts back on the table.

Depth is measured along the thread only: 0 when the nut sits on the tip, 1 when it reaches the end
of the thread. The plain shaft near the head cannot take the nut, so it is not part of the scale."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .game_logic import EventClock, GameDifficulty, GameMode, GameState

TARGET_DEPTH = {
    GameDifficulty.EASY: 0.25,
    GameDifficulty.MEDIUM: 0.40,
    GameDifficulty.HARD: 0.50,
    GameDifficulty.EXPERT: 0.75,
}
HOLD_FRAMES = 5
# Stages where a hand reaching past the parts could pass the check also need this much time in a row.
HOLD_MS = 1000.0
ENGAGED_DEPTH = 0.02
# Nut-center positions as fractions of the tip-to-head axis (calibrated screw, 2026-10-17): the
# thread covers 0.63 from the tip and the nut is 0.13 thick, so its center travels 0.065..0.565.
THREAD_START = 0.065
THREAD_END = 0.565
# Thread share the nut may travel while no fingertip is on it before it counts as spun.
SPIN_DEPTH = 0.10
SPIN_WARNING_MS = 2500.0
# Order of play. Each phase has its own timer; the summary reports all of them.
PHASES = ("setup", "pickup", "screw", "unscrew", "putdown")


@dataclass
class ScrewObservation:
    """What the engine needs from one frame. Geometry stays in the detector."""

    screw_visible: bool
    nut_visible: bool
    depth: float | None
    nut_on_screw: bool
    turns: float
    side_steps: int
    left_on_screw: bool
    right_on_nut: bool
    hands_seen: bool
    hands_on_parts: bool = False
    # Both parts moved away from where they lay during setup.
    parts_lifted: bool = False
    # The nut is well clear of the screw: past the tip or beside the shaft.
    nut_clear: bool = False
    # A thumb or index tip touches the nut; None when no hand or the nut is not seen.
    nut_gripped: bool | None = None


def thread_progress(axis_depth: float | None) -> float | None:
    """Tip-to-head axis fraction of the nut center -> share of the thread screwed (0..1)."""
    if axis_depth is None:
        return None
    return min(1.0, max(0.0, (axis_depth - THREAD_START) / (THREAD_END - THREAD_START)))


def axis_depth(progress: float) -> float:
    """Inverse of thread_progress, for drawing marks on the screw axis."""
    return THREAD_START + progress * (THREAD_END - THREAD_START)


def _turns_text(turns: float) -> str:
    rounded = round(turns)
    number = f"{rounded:.0f}" if abs(turns - rounded) < 0.05 else f"{turns:.1f}"
    return f"{number} turn" if abs(turns - 1) < 0.05 else f"{number} turns"


def _ms(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


class ScrewingEngine(EventClock):
    """Phases: setup, pickup, screw, unscrew, putdown. Difficulty is the target share of the thread.

    A phase advances only after its condition holds for HOLD_FRAMES frames in a row, so one
    misread frame cannot skip a stage."""

    def __init__(self) -> None:
        self.reset_game()

    def reset_game(self) -> None:
        self.state = GameState.IDLE
        self.difficulty: GameDifficulty | None = None
        self.target = TARGET_DEPTH[GameDifficulty.HARD]
        self.phase = "setup"
        self.feedback = ""
        self.depth: float | None = None
        self.turns = 0.0
        self.side_steps = 0
        self.phase_times: Dict[str, float] = {}
        self.score = 0.0
        self._phase_started: float | None = None
        self._hold = 0
        self._hold_since: float | None = None
        self._hold_ms = 0.0
        self._reached_depth: float | None = None
        self.spins = 0
        self._free_depth: float | None = None
        self._spin_warning_until = -1.0
        self._reset_clock()

    @property
    def pickup_ms(self) -> float | None:
        return self.phase_times.get("pickup")

    @property
    def screw_ms(self) -> float | None:
        return self.phase_times.get("screw")

    @property
    def unscrew_ms(self) -> float | None:
        return self.phase_times.get("unscrew")

    @property
    def putdown_ms(self) -> float | None:
        return self.phase_times.get("putdown")

    def start_game(self, difficulty: GameDifficulty) -> None:
        self.reset_game()
        self.difficulty = difficulty
        self.target = TARGET_DEPTH.get(difficulty, 0.50)
        self.state = GameState.DETECTING

    def process_frame(self, sample: ScrewObservation, frame_id: int, t_ms: Optional[float] = None) -> Dict:
        self._tick(frame_id, t_ms)
        if self._phase_started is None:
            self._phase_started = self.now_ms
        if self.state != GameState.COMPLETED:
            self.turns = sample.turns
            self.side_steps = sample.side_steps
            self.depth = sample.depth if sample.nut_on_screw else None
            self.phase_times[self.phase] = self.phase_ms()
            if self.phase in {"screw", "unscrew"}:
                self._track_spin(sample)
            getattr(self, f"_{self.phase}")(sample)
            if self.phase in {"screw", "unscrew"} and self.now_ms < self._spin_warning_until:
                self.feedback = "Keep your fingers on the nut — turn it, don't spin it"
            if self.state == GameState.COMPLETED or self.phase in {"unscrew", "putdown"}:
                self.score = 100.0
            else:
                self.score = 0.0 if self.depth is None else min(1.0, self.depth / self.target) * 100.0
        return {
            "frame_id": frame_id,
            "state": self.state,
            "new_achievements": [],
            "game_status": "completed" if self.state == GameState.COMPLETED else "ongoing",
        }

    def _held(self, condition: bool, min_ms: float = 0.0) -> bool:
        """Count frames in a row where the phase condition holds, and for how long."""
        if condition:
            self._hold += 1
            if self._hold_since is None:
                self._hold_since = self.now_ms
        else:
            self._hold, self._hold_since = 0, None
        self._hold_ms = min_ms
        held_for = 0.0 if self._hold_since is None else self.now_ms - self._hold_since
        return self._hold >= HOLD_FRAMES and held_for >= min_ms

    def _track_spin(self, sample: ScrewObservation) -> None:
        """A nut that travels along the thread with no fingertip on it was spun, not turned."""
        if not sample.nut_on_screw or sample.depth is None or sample.nut_gripped is None:
            return
        if sample.nut_gripped:
            self._free_depth = None
            return
        if self._free_depth is None:
            self._free_depth = sample.depth
        elif abs(sample.depth - self._free_depth) >= SPIN_DEPTH:
            self.spins += 1
            self.log_event(
                "nut_spun",
                {"phase": self.phase, "from": round(self._free_depth, 3), "to": round(sample.depth, 3)},
            )
            self._free_depth = sample.depth
            self._spin_warning_until = self.now_ms + SPIN_WARNING_MS

    def _advance(self, phase: str, data: Dict | None = None) -> None:
        self.phase_times[self.phase] = self.phase_ms()
        self.log_event(
            "phase_completed",
            {"phase": self.phase, "ms": _ms(self.phase_times[self.phase]), **(data or {})},
        )
        self.phase = phase
        self._phase_started = self.now_ms
        self._hold, self._hold_since, self._hold_ms = 0, None, 0.0
        self._free_depth = None

    def _setup(self, sample: ScrewObservation) -> None:
        ready = sample.screw_visible and sample.nut_visible and not sample.nut_on_screw and sample.hands_seen
        if self._held(ready):
            self._advance("pickup")
            self.feedback = "Pick up the screw with your left hand and the nut with your right hand"
        elif not sample.screw_visible:
            self.feedback = "Place the screw on the table"
        elif not sample.nut_visible:
            self.feedback = "Place the nut on the table"
        elif sample.nut_on_screw:
            self.feedback = "Take the nut off the screw and lay both on the table"
        elif not sample.hands_seen:
            self.feedback = "Show your hands to the camera"
        else:
            self.feedback = "Ready — hold still"

    def _pickup(self, sample: ScrewObservation) -> None:
        holding = sample.hands_seen and sample.left_on_screw and sample.right_on_nut
        # Hands near the parts are not enough: reaching for them passes that. The parts must be lifted.
        if self._held(holding and sample.parts_lifted, HOLD_MS):
            self._advance("screw")
            self.feedback = f"Screw the nut on to {int(round(self.target * 100))}% of the thread"
        elif not sample.hands_seen:
            self.feedback = "Show your hands to the camera"
        elif not sample.left_on_screw:
            self.feedback = "Pick up the screw with your left hand"
        elif not sample.right_on_nut:
            self.feedback = "Pick up the nut with your right hand"
        elif not sample.parts_lifted:
            self.feedback = "Lift the screw and the nut off the table"
        else:
            self.feedback = "Good — hold the screw and the nut"

    def _screw(self, sample: ScrewObservation) -> None:
        goal = int(round(self.target * 100))
        depth = sample.depth or 0.0
        # Spinning the nut up to the mark does not count: a fingertip must be on it when it gets there.
        gripped = sample.nut_gripped is not False
        if self._held(sample.nut_on_screw and depth >= self.target and gripped):
            self._reached_depth = depth
            self.log_event(
                "depth_reached",
                {
                    "depth": round(depth, 3),
                    "target": self.target,
                    "turns": round(self.turns, 2),
                    "side_steps": self.side_steps,
                    "screw_ms": _ms(self.phase_ms()),
                },
            )
            self._advance("unscrew", {"depth": round(depth, 3)})
            self.feedback = "Target reached — hold the screw and unscrew the nut"
        elif not sample.screw_visible:
            self.feedback = "Keep the screw in view"
        elif not sample.nut_on_screw or depth < ENGAGED_DEPTH:
            self.feedback = f"Put the nut on the tip of the screw and screw it on to {goal}%"
        else:
            self.feedback = f"Screw on — {int(round(depth * 100))}% of {goal}%, {_turns_text(self.turns)}"

    def _unscrew(self, sample: ScrewObservation) -> None:
        # A hidden nut is not an unscrewed nut, and "not on the screw" flickers at the tip: the nut
        # must be seen well clear of the screw for HOLD_MS.
        if self._held(sample.nut_visible and sample.nut_clear, HOLD_MS):
            self._advance("putdown")
            self.feedback = "Put the screw and the nut back on the table"
        elif sample.nut_on_screw and sample.depth is not None:
            self.feedback = f"Unscrew the nut — {int(round(sample.depth * 100))}% still on the screw"
        else:
            self.feedback = "Unscrew the nut until it comes off"

    def _putdown(self, sample: ScrewObservation) -> None:
        on_table = sample.screw_visible and sample.nut_visible and not sample.nut_on_screw
        if self._held(on_table and not sample.hands_on_parts):
            self._advance("done")
            self._finish()
        elif not on_table:
            self.feedback = "Put the screw and the nut back on the table"
        else:
            self.feedback = "Take your hands away from the parts"

    def _finish(self) -> None:
        self.state = GameState.COMPLETED
        self.end_ms = self.now_ms
        self.feedback = "Finished"
        self.score = 100.0
        self.log_event(
            "game_completed",
            {
                "setup_ms": _ms(self.phase_times.get("setup")),
                "pickup_ms": _ms(self.pickup_ms),
                "screw_ms": _ms(self.screw_ms),
                "unscrew_ms": _ms(self.unscrew_ms),
                "putdown_ms": _ms(self.putdown_ms),
                "spins": self.spins,
                "turns": round(self.turns, 2),
                "side_steps": self.side_steps,
                "depth": round(self._reached_depth if self._reached_depth is not None else self.target, 3),
                "target": self.target,
                "final_score": 100.0,
            },
        )

    def phase_ms(self) -> float:
        if self._phase_started is None:
            return 0.0
        return max(0.0, self.now_ms - self._phase_started)

    def _hold_progress(self) -> float:
        frames = self._hold / HOLD_FRAMES
        if self._hold_ms and self._hold_since is not None:
            frames = min(frames, (self.now_ms - self._hold_since) / self._hold_ms)
        return min(1.0, frames)

    def get_game_status(self) -> Dict:
        depth = self.depth
        step = PHASES.index(self.phase) + 1 if self.phase in PHASES else len(PHASES)
        return {
            "mode": GameMode.SCREWING.value,
            "state": self.state.value,
            "phase": self.phase,
            "step": step,
            "steps": len(PHASES),
            "goal": f"Screw the nut to {int(round(self.target * 100))}% of the thread",
            "feedback": self.feedback,
            "depth": None if depth is None else round(depth, 3),
            "target_depth": self.target,
            "turns": round(self.turns, 2),
            "side_steps": self.side_steps,
            "spins": self.spins,
            "phase_ms": _ms(self.phase_ms()),
            "pickup_ms": _ms(self.pickup_ms),
            "screw_ms": _ms(self.screw_ms),
            "unscrew_ms": _ms(self.unscrew_ms),
            "putdown_ms": _ms(self.putdown_ms),
            "hold_progress": self._hold_progress(),
            "score": self.score,
            "stacks_completed": self.side_steps,
            "cubes_detected": int(self.depth is not None) + int(self.phase != "setup"),
            "elapsed_ms": self.elapsed_ms,
            "level": self.difficulty.name if self.difficulty else None,
        }
