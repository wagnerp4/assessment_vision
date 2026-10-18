"""Random stacking: build a random color order bottom-to-top, take it apart top-first, repeat."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from .config import CubeConfig
from .cube_detector import Cube
from .game_logic import EventClock, GameDifficulty, GameMode, GameState
from .grasp import CubeMove, FingerPairRule
from .stacking_analyzer import StackingAnalyzer

CUBES_PER_DIFFICULTY = {
    GameDifficulty.EASY: 2,
    GameDifficulty.MEDIUM: 3,
    GameDifficulty.HARD: 4,
    GameDifficulty.EXPERT: 4,
}
ROUND_POINTS = 100


class RandomStackingEngine(EventClock):
    """Each round samples an order of `n` colors. STACK phase: the tower must equal that order (bottom->top).
    UNSTACK phase: remove the top cube one at a time until only the base is left.

    A step is accepted after its tower state has been seen for `step_frames` consecutive frames; events record
    both when the state was first reached (`achieved_ms`) and when it was accepted (`confirmed_ms`). With the
    finger-pair rule enabled, every placement or removal must have been made with the required pair."""

    def __init__(
        self,
        config: CubeConfig | None = None,
        pair_rule: FingerPairRule | None = None,
        rounds: int = 4,
        seed: int | None = None,
        step_frames: int = 10,
    ) -> None:
        self.config = config or CubeConfig()
        self.pair_rule = pair_rule or FingerPairRule()
        self.rounds = rounds
        self.seed = seed
        self.step_frames = step_frames
        self.analyzer = StackingAnalyzer(self.config)
        self.colors = tuple(self.config.CUBE_COLORS)
        self.reset_game()

    def reset_game(self) -> None:
        self.state = GameState.IDLE
        self.difficulty: GameDifficulty | None = None
        self._reset_clock()
        self.target: Tuple[str, ...] = ()
        self.phase = "stack"
        self.round_index = 0
        self.rounds_completed = 0
        self.rejected_moves = 0
        self.height = 0
        self.score = 0.0
        self.feedback = ""
        self.cubes_detected = 0
        self._confirmed = 0
        self._streak_key = None
        self._streak_since = 0.0
        self._streak_frames = 0
        self._wrong_logged = False
        self._step_start_ms = 0.0
        self._rng = random.Random(self.seed)

    def start_game(self, difficulty: GameDifficulty) -> None:
        self.reset_game()
        self.difficulty = difficulty
        self.state = GameState.DETECTING

    def _cube_count(self) -> int:
        return min(CUBES_PER_DIFFICULTY.get(self.difficulty, 4), len(self.colors))

    def _start_round(self) -> None:
        self.target = tuple(self._rng.sample(self.colors, self._cube_count()))
        self.phase = "stack"
        self.height = 1
        self._confirmed = 1
        self._streak_key = None
        self._step_start_ms = self.now_ms
        self.log_event("round_started", {"round": self.round_index + 1, "target": list(self.target)})

    def _tower(self, cubes: Sequence[Cube], stacks: Sequence[Sequence[Cube]]) -> List[Cube] | None:
        """The stack standing on the round's base cube, or the base alone; None if the base is not visible."""
        base = self.target[0]
        for stack in stacks:
            if stack[0].color == base:
                return list(stack)
        alone = [cube for cube in cubes if cube.color == base]
        return alone[:1] or None

    def _streak(self, key) -> bool:
        """Count consecutive frames with the same tower state; True once it has been stable long enough."""
        if key != self._streak_key:
            self._streak_key = key
            self._streak_since = self.now_ms
            self._streak_frames = 0
            self._wrong_logged = False
        self._streak_frames += 1
        return self._streak_frames >= self.step_frames

    def _step(self, index: int, color: str) -> None:
        self.log_event(
            "step_completed",
            {
                "round": self.round_index + 1,
                "phase": self.phase,
                "index": index,
                "color": color,
                "achieved_ms": round(self._streak_since, 1),
                "confirmed_ms": round(self.now_ms, 1),
            },
        )

    def process_frame(
        self,
        cubes: List[Cube],
        frame_id: int,
        moves: Tuple[CubeMove, ...] = (),
        t_ms: float | None = None,
    ) -> Dict:
        self._tick(frame_id, t_ms)
        for move in moves:
            if self.pair_rule.enabled and not self.pair_rule.placement_ok.get(move.color, True):
                self.rejected_moves += 1
        self.cubes_detected = len(cubes)
        stacks = self.analyzer.find_stacks(cubes)
        result = {
            "frame_id": frame_id,
            "state": self.state,
            "cubes_detected": len(cubes),
            "stacks_found": len(stacks),
            "stack_details": [],
            "new_achievements": [],
            "game_status": "completed" if self.state == GameState.COMPLETED else "ongoing",
        }
        if self.state != GameState.DETECTING:
            return result
        if not self.target:
            self._start_round()

        tower = self._tower(cubes, stacks)
        if tower is None:
            self._streak_key = None
            self.feedback = f"Base cube {self.target[0].upper()} not visible"
            return result
        colors = tuple(cube.color for cube in tower)
        if self.phase == "stack":
            self._stack_step(colors, result)
        else:
            self._unstack_step(colors, result)
        return result

    def _stack_step(self, colors, result) -> None:
        placed = 1
        while (
            placed < len(colors)
            and placed < len(self.target)
            and colors[placed] == self.target[placed]
            and self.pair_rule.placed_ok(colors[placed])
        ):
            placed += 1
        self.height = placed
        extra = placed < len(colors)
        wrong = extra and (placed >= len(self.target) or colors[placed] != self.target[placed])
        if wrong:
            self.feedback = f"Wrong cube: take {colors[placed].upper()} off"
        elif extra and self.pair_rule.placement_ok.get(colors[placed]) is False:
            self.feedback = f"Re-place {colors[placed].upper()} using {self.pair_rule.label}"
        else:
            self.feedback = ""

        if not self._streak((placed, colors[placed] if extra else None)):
            return
        if wrong and not self._wrong_logged:
            self._wrong_logged = True
            self.log_event(
                "wrong_cube", {"round": self.round_index + 1, "color": colors[placed]}
            )
        if placed > self._confirmed:
            for index in range(self._confirmed, placed):
                self._step(index, self.target[index])
            self._confirmed = placed
        elif placed < self._confirmed:
            self._confirmed = placed
        if placed == len(self.target) and not extra:
            self.phase = "unstack"
            self._step_start_ms = self.now_ms
            self._streak_key = None
            result["new_achievements"].append(f"Round {self.round_index + 1}: stacked")
            self.log_event(
                "stack_completed",
                {
                    "round": self.round_index + 1,
                    "achieved_ms": round(self._streak_since, 1),
                    "confirmed_ms": round(self.now_ms, 1),
                },
            )

    def _unstack_step(self, colors, result) -> None:
        expected = self.target[: self.height - 1]
        top = self.target[self.height - 1]
        last_move = self.pair_rule.last_move.get(top)
        moved_now = last_move is not None and last_move[0] >= self._step_start_ms
        removed_ok = not self.pair_rule.enabled or (moved_now and last_move[1])
        if colors == expected and moved_now and not removed_ok:
            self.feedback = f"Put {top.upper()} back and remove it using {self.pair_rule.label}"
        elif colors not in (self.target[: self.height], expected):
            self.feedback = f"Remove only {top.upper()} (top cube)"
        else:
            self.feedback = ""
        if not (colors == expected and removed_ok):
            self._streak_key = None
            return
        if not self._streak(("removed", self.height)):
            return
        self._step(self.height - 1, top)
        achieved = self._streak_since
        self.height -= 1
        self._step_start_ms = self.now_ms
        self._streak_key = None
        if self.height > 1:
            return
        self.rounds_completed += 1
        result["new_achievements"].append(f"Round {self.round_index + 1} complete")
        self.log_event(
            "round_completed",
            {
                "round": self.round_index + 1,
                "achieved_ms": round(achieved, 1),
                "confirmed_ms": round(self.now_ms, 1),
            },
        )
        self.round_index += 1
        if self.round_index >= self.rounds:
            self.state = GameState.COMPLETED
            self.end_ms = achieved
            result["game_status"] = "completed"
            self.score = self.rounds_completed * ROUND_POINTS
            self.log_event(
                "game_completed",
                {
                    "final_score": self.score,
                    "total_ms": round(self.elapsed_ms, 1),
                    "rounds": self.rounds_completed,
                    "rejected_moves": self.rejected_moves,
                },
            )
        else:
            self._start_round()

    def goal_text(self) -> str:
        if self.state == GameState.COMPLETED:
            return f"All {self.rounds} rounds done"
        if not self.target:
            return "Random stacking"
        order = "-".join(color[0].upper() for color in self.target)
        prefix = f"Round {self.round_index + 1}/{self.rounds}"
        if self.phase == "stack":
            return f"{prefix} STACK {order} (bottom->top)"
        return f"{prefix} UNSTACK: remove {self.target[self.height - 1].upper()}"

    def get_game_status(self) -> Dict:
        pending = self._streak_key is not None and (
            self.phase == "unstack" or self.height > self._confirmed
            or (self.height == len(self.target) and self._streak_key[1] is None)
        )
        return {
            "mode": GameMode.RANDOM.value,
            "state": self.state.value,
            "level": self.difficulty.name if self.difficulty else None,
            "goal": self.goal_text(),
            "hold_progress": min(1.0, self._streak_frames / self.step_frames) if pending else 0.0,
            "score": self.score,
            "stacks_completed": self.rounds_completed,
            "time_elapsed_frames": round(self.elapsed_ms * 30 / 1000),
            "elapsed_ms": self.elapsed_ms,
            "cubes_detected": self.cubes_detected,
            "sequence_completions": {},
            "target": list(self.target),
            "phase": self.phase,
            "round": min(self.round_index + 1, self.rounds),
            "rounds": self.rounds,
            "height": self.height,
            "rejected_moves": self.rejected_moves,
            "feedback": self.feedback,
        }
