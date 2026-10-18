from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .config import CubeConfig
from .cube_detector import Cube
from .grasp import CubeMove, FingerPairRule
from .stacking_analyzer import StackingAnalyzer


class GameState(Enum):
    """Possible game states"""

    IDLE = "idle"
    DETECTING = "detecting"
    STACKING = "stacking"
    COMPLETED = "completed"
    ERROR = "error"


class GameMode(Enum):
    SIMPLE = "simple"  # build a stack that meets the difficulty's goal
    RANDOM = "random"  # stack a random color order, unstack it in reverse, repeat for several rounds
    SCREWING = "screwing"  # screw a nut on to a depth set by the difficulty, then unscrew it


class GameDifficulty(Enum):
    """Game difficulty levels"""

    EASY = 1  # Any order of 2 cubes
    MEDIUM = 2  # Any order of 3 cubes
    HARD = 3  # Specific sequence of 4 cubes
    EXPERT = 4  # Multiple sequences in order


@dataclass
class GameLevel:
    """Represents a game level/challenge"""

    difficulty: GameDifficulty
    target_sequence: Tuple[str, ...]  # e.g., ('blue', 'red', 'green', 'yellow')
    min_duration_frames: int = 30  # How long stack must be stable
    description: str = ""


@dataclass
class GameScore:
    """Tracks player performance"""

    total_cubes_detected: int = 0
    valid_stacks_completed: int = 0
    sequence_completions: Dict[str, int] = field(default_factory=dict)
    time_elapsed_frames: int = 0
    accuracy_score: float = 0.0


class EventClock:
    """Session clock and ordered event log shared by the game engines.

    Events are `{"type", "t_ms", "frame", "data"}` on the session clock (capture timestamps), so durations do
    not depend on the processing frame rate."""

    def _reset_clock(self) -> None:
        self.event_log: List[Dict] = []
        self.now_ms = 0.0
        self.frame = 0
        self.start_ms: Optional[float] = None
        self.end_ms: Optional[float] = None

    def _tick(self, frame_id: int, t_ms: Optional[float]) -> None:
        self.frame = frame_id
        self.now_ms = frame_id * 1000.0 / 30.0 if t_ms is None else float(t_ms)
        if self.start_ms is None:
            self.start_ms = self.now_ms
            self.log_event("game_started", {})

    def log_event(
        self,
        event_type: str,
        data: Any,
        t_ms: Optional[float] = None,
        frame: Optional[int] = None,
    ) -> None:
        self.event_log.append(
            {
                "type": event_type,
                "t_ms": round(self.now_ms if t_ms is None else t_ms, 1),
                "frame": self.frame if frame is None else frame,
                "data": data,
            }
        )

    @property
    def elapsed_ms(self) -> float:
        if self.start_ms is None:
            return 0.0
        return (self.end_ms if self.end_ms is not None else self.now_ms) - self.start_ms


class GameEngine(EventClock):
    """Levels: EASY/MEDIUM = any 2/3-cube stack, HARD = exact order, EXPERT = every valid order in turn.
    A stack only counts after being held still for the level's `min_duration_frames`."""

    def __init__(self, config: CubeConfig = None, pair_rule: FingerPairRule | None = None):
        self.config = config or CubeConfig()
        self.pair_rule = pair_rule or FingerPairRule()
        self.state = GameState.IDLE
        self.analyzer = StackingAnalyzer(config)
        self.current_level: Optional[GameLevel] = None
        self.score = GameScore()
        self.previous_stacks = []
        self._hold: Dict[Tuple[int, ...], int] = {}
        self._hold_since: Dict[Tuple[int, ...], float] = {}
        self._counted: set = set()
        self._targets: List[Tuple[str, ...]] = []
        self._hold_progress = 0.0
        self._reset_clock()

    def start_game(self, difficulty: GameDifficulty):
        self.state = GameState.DETECTING
        self.score = GameScore()
        self._reset_clock()
        self._hold, self._hold_since, self._counted, self._hold_progress = {}, {}, set(), 0.0
        order = tuple(self.config.VALID_STACK_SEQUENCES[0])
        if difficulty == GameDifficulty.EASY:
            self.current_level = GameLevel(difficulty, (), 20, "Stack any 2 cubes")
            self._targets = []
        elif difficulty == GameDifficulty.MEDIUM:
            self.current_level = GameLevel(difficulty, (), 30, "Stack any 3 cubes")
            self._targets = []
        elif difficulty == GameDifficulty.HARD:
            self.current_level = GameLevel(
                difficulty, order, 40, "Stack " + "-".join(c[0].upper() for c in order)
            )
            self._targets = [order]
        else:
            seqs = [tuple(q) for q in self.config.VALID_STACK_SEQUENCES]
            self.current_level = GameLevel(
                difficulty, seqs[0], 50, "Build each valid order in turn"
            )
            self._targets = seqs

    def _required_height(self) -> int:
        return {GameDifficulty.EASY: 2, GameDifficulty.MEDIUM: 3}.get(
            self.current_level.difficulty, 4
        )

    def goal_text(self) -> str:
        if not self.current_level:
            return ""
        if self._targets:
            return "Goal (bottom->top): " + "-".join(c[0].upper() for c in self._targets[0])
        return self.current_level.description

    def _stack_meets_goal(self, colors: Tuple[str, ...]) -> bool:
        if self._targets:
            return colors == self._targets[0]
        return len(colors) >= self._required_height()

    def process_frame(
        self,
        cubes: List[Cube],
        frame_id: int,
        moves: Tuple[CubeMove, ...] = (),
        t_ms: Optional[float] = None,
    ) -> Dict:
        self._tick(frame_id, t_ms)
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
        self.score.total_cubes_detected = len(cubes)
        if self.state != GameState.DETECTING or not self.current_level:
            return result
        self.score.time_elapsed_frames = round(self.elapsed_ms * 30 / 1000)

        seen, best, achieved = set(), 0.0, None
        for stack in stacks:
            colors = tuple(c.color for c in stack)
            key = tuple(c.cube_id for c in stack)
            grip_ok = self.pair_rule.stack_ok(stack)
            meets = self._stack_meets_goal(colors) and grip_ok
            result["stack_details"].append(
                {
                    "height": len(stack),
                    "colors": list(colors),
                    "is_valid": meets,
                    "grip_ok": grip_ok,
                    "sequence": " -> ".join(colors),
                }
            )
            if not meets:
                continue
            seen.add(key)
            self._hold[key] = self._hold.get(key, 0) + 1
            self._hold_since.setdefault(key, self.now_ms)
            best = max(best, self._hold[key] / self.current_level.min_duration_frames)
            if (
                self._hold[key] >= self.current_level.min_duration_frames
                and key not in self._counted
            ):
                self._counted.add(key)
                self.score.valid_stacks_completed += 1
                seq = "_".join(colors)
                self.score.sequence_completions[seq] = (
                    self.score.sequence_completions.get(seq, 0) + 1
                )
                result["new_achievements"].append(f"Stack held: {' -> '.join(colors)}")
                achieved = self._hold_since[key]
                self.log_event(
                    "stack_confirmed",
                    {
                        "colors": list(colors),
                        "achieved_ms": round(achieved, 1),
                        "confirmed_ms": round(self.now_ms, 1),
                    },
                )
                if self._targets:
                    self._targets.pop(0)
        # Holding must be continuous: a stack that falls apart starts over.
        self._hold = {k: v for k, v in self._hold.items() if k in seen}
        self._hold_since = {k: v for k, v in self._hold_since.items() if k in seen}
        self._hold_progress = min(1.0, best)

        if self._level_done():
            self.state = GameState.COMPLETED
            self.end_ms = achieved if achieved is not None else self.now_ms
            result["game_status"] = "completed"
            self._calculate_final_score()
        self.previous_stacks = stacks
        return result

    def _level_done(self) -> bool:
        if self.current_level.difficulty in (GameDifficulty.HARD, GameDifficulty.EXPERT):
            return not self._targets
        return self.score.valid_stacks_completed >= 1

    def _calculate_final_score(self):
        time_bonus = max(0, 500 - self.elapsed_ms * 30 / 1000)
        self.score.accuracy_score = self.score.valid_stacks_completed * 100 + time_bonus
        self.log_event(
            "game_completed",
            {
                "final_score": self.score.accuracy_score,
                "total_ms": round(self.elapsed_ms, 1),
                "stacks_completed": self.score.valid_stacks_completed,
            },
        )

    def get_game_status(self) -> Dict:
        return {
            "mode": GameMode.SIMPLE.value,
            "state": self.state.value,
            "level": self.current_level.difficulty.name if self.current_level else None,
            "goal": self.goal_text(),
            "hold_progress": self._hold_progress,
            "score": self.score.accuracy_score,
            "stacks_completed": self.score.valid_stacks_completed,
            "time_elapsed_frames": self.score.time_elapsed_frames,
            "elapsed_ms": self.elapsed_ms,
            "cubes_detected": self.score.total_cubes_detected,
            "sequence_completions": self.score.sequence_completions,
        }

    def reset_game(self):
        self.state = GameState.IDLE
        self.score = GameScore()
        self._reset_clock()
        self.current_level = None
        self._hold, self._hold_since, self._counted = {}, {}, set()
        self._targets, self._hold_progress = [], 0.0
        self.analyzer = StackingAnalyzer(self.config)
