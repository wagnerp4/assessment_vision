import numpy as np
from typing import Any, List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from .cube_detector import Cube
from .stacking_analyzer import StackingEvent, StackingAnalyzer
from .config import CubeConfig

class GameState(Enum):
    """Possible game states"""
    IDLE = "idle"
    DETECTING = "detecting"
    STACKING = "stacking"
    COMPLETED = "completed"
    ERROR = "error"


class GameDifficulty(Enum):
    """Game difficulty levels"""
    EASY = 1        # Any order of 2 cubes
    MEDIUM = 2      # Any order of 3 cubes
    HARD = 3        # Specific sequence of 4 cubes
    EXPERT = 4      # Multiple sequences in order


@dataclass
class GameLevel:
    """Represents a game level/challenge"""
    difficulty: GameDifficulty
    target_sequence: Tuple[str, ...]  # e.g., ('blue', 'red', 'green', 'yellow')
    min_duration_frames: int = 30     # How long stack must be stable
    description: str = ""


@dataclass
class GameScore:
    """Tracks player performance"""
    total_cubes_detected: int = 0
    valid_stacks_completed: int = 0
    sequence_completions: Dict[str, int] = field(default_factory=dict)
    time_elapsed_frames: int = 0
    accuracy_score: float = 0.0


class GameEngine:
    """Levels: EASY/MEDIUM = any 2/3-cube stack, HARD = exact order, EXPERT = every valid order in turn.
    A stack only counts after being held still for the level's `min_duration_frames`."""

    def __init__(self, config: CubeConfig = None):
        self.config = config or CubeConfig()
        self.state = GameState.IDLE
        self.analyzer = StackingAnalyzer(config)
        self.current_level: Optional[GameLevel] = None
        self.score = GameScore()
        self.game_start_frame: Optional[int] = None
        self.event_log: List[Dict] = []
        self.previous_stacks = []
        self._hold: Dict[Tuple[int, ...], int] = {}
        self._counted: set = set()
        self._targets: List[Tuple[str, ...]] = []
        self._hold_progress = 0.0

    def start_game(self, difficulty: GameDifficulty):
        self.state = GameState.DETECTING
        self.score = GameScore()
        self.game_start_frame = None
        self.event_log = []
        self._hold, self._counted, self._hold_progress = {}, set(), 0.0
        order = tuple(self.config.VALID_STACK_SEQUENCES[0])
        if difficulty == GameDifficulty.EASY:
            self.current_level = GameLevel(difficulty, (), 20, "Stack any 2 cubes")
            self._targets = []
        elif difficulty == GameDifficulty.MEDIUM:
            self.current_level = GameLevel(difficulty, (), 30, "Stack any 3 cubes")
            self._targets = []
        elif difficulty == GameDifficulty.HARD:
            self.current_level = GameLevel(difficulty, order, 40, "Stack " + "-".join(c[0].upper() for c in order))
            self._targets = [order]
        else:
            seqs = [tuple(q) for q in self.config.VALID_STACK_SEQUENCES]
            self.current_level = GameLevel(difficulty, seqs[0], 50, "Build each valid order in turn")
            self._targets = seqs

    def _required_height(self) -> int:
        return {GameDifficulty.EASY: 2, GameDifficulty.MEDIUM: 3}.get(self.current_level.difficulty, 4)

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

    def process_frame(self, cubes: List[Cube], frame_id: int) -> Dict:
        if self.game_start_frame is None:
            self.game_start_frame = frame_id
        stacks = self.analyzer.find_stacks(cubes)
        result = {
            'frame_id': frame_id, 'state': self.state, 'cubes_detected': len(cubes),
            'stacks_found': len(stacks), 'stack_details': [], 'new_achievements': [],
            'game_status': 'completed' if self.state == GameState.COMPLETED else 'ongoing',
        }
        self.score.total_cubes_detected = len(cubes)
        if self.state != GameState.DETECTING or not self.current_level:
            return result
        self.score.time_elapsed_frames = frame_id - self.game_start_frame

        seen, best = set(), 0.0
        for stack in stacks:
            colors = tuple(c.color for c in stack)
            key = tuple(c.cube_id for c in stack)
            meets = self._stack_meets_goal(colors)
            result['stack_details'].append({'height': len(stack), 'colors': list(colors), 'is_valid': meets,
                                            'sequence': ' -> '.join(colors)})
            if not meets:
                continue
            seen.add(key)
            self._hold[key] = self._hold.get(key, 0) + 1
            best = max(best, self._hold[key] / self.current_level.min_duration_frames)
            if self._hold[key] >= self.current_level.min_duration_frames and key not in self._counted:
                self._counted.add(key)
                self.score.valid_stacks_completed += 1
                seq = '_'.join(colors)
                self.score.sequence_completions[seq] = self.score.sequence_completions.get(seq, 0) + 1
                result['new_achievements'].append(f"Stack held: {' -> '.join(colors)}")
                self._log_event('stack_confirmed', {'colors': list(colors), 'frame': frame_id})
                if self._targets:
                    self._targets.pop(0)
        # Holding must be continuous: a stack that falls apart starts over.
        self._hold = {k: v for k, v in self._hold.items() if k in seen}
        self._hold_progress = min(1.0, best)

        if self._level_done():
            self.state = GameState.COMPLETED
            result['game_status'] = 'completed'
            self._calculate_final_score()
        self.previous_stacks = stacks
        return result

    def _level_done(self) -> bool:
        if self.current_level.difficulty in (GameDifficulty.HARD, GameDifficulty.EXPERT):
            return not self._targets
        return self.score.valid_stacks_completed >= 1

    def _calculate_final_score(self):
        time_bonus = max(0, 500 - self.score.time_elapsed_frames)
        self.score.accuracy_score = self.score.valid_stacks_completed * 100 + time_bonus
        self._log_event('game_completed', {'final_score': self.score.accuracy_score,
                                           'time_frames': self.score.time_elapsed_frames,
                                           'stacks_completed': self.score.valid_stacks_completed})

    def _log_event(self, event_type: str, data: Any):
        self.event_log.append({'timestamp': datetime.now().isoformat(), 'event_type': event_type, 'data': data})

    def get_game_status(self) -> Dict:
        return {
            'state': self.state.value,
            'level': self.current_level.difficulty.name if self.current_level else None,
            'goal': self.goal_text(),
            'hold_progress': self._hold_progress,
            'score': self.score.accuracy_score,
            'stacks_completed': self.score.valid_stacks_completed,
            'time_elapsed_frames': self.score.time_elapsed_frames,
            'cubes_detected': self.score.total_cubes_detected,
            'sequence_completions': self.score.sequence_completions,
        }

    def reset_game(self):
        self.state = GameState.IDLE
        self.score = GameScore()
        self.game_start_frame = None
        self.event_log = []
        self.current_level = None
        self._hold, self._counted, self._targets, self._hold_progress = {}, set(), [], 0.0
        self.analyzer = StackingAnalyzer(self.config)
