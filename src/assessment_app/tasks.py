"""Application tasks built from normalized model observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import CubeConfig
from .contracts import (
    ActionEvent,
    CubeDetections,
    DatasetSample,
    Frame,
    ModelObservation,
)
from .cube_detector import Cube
from .game_logic import GameDifficulty, GameEngine
from .stacking_analyzer import StackingAnalyzer


@dataclass(frozen=True)
class StackingFrameResult:
    cubes: tuple[Cube, ...]
    stacks: tuple[tuple[Cube, ...], ...]
    game_result: dict
    game_status: dict


class CubeStackingTask:
    """Apply stack geometry and the existing assessment-game rules to cube tracks."""

    name = "cube_stacking"

    def __init__(
        self,
        config: CubeConfig | None = None,
        difficulty: GameDifficulty = GameDifficulty.HARD,
    ) -> None:
        self.config = config or CubeConfig()
        self.difficulty = difficulty
        self.analyzer = StackingAnalyzer(self.config)
        self.game_engine = GameEngine(self.config)

    def begin_sequence(self, sample: DatasetSample) -> None:
        self.analyzer = StackingAnalyzer(self.config)
        self.game_engine.start_game(self.difficulty)

    def process(
        self, frame: Frame, observations: Sequence[ModelObservation]
    ) -> StackingFrameResult:
        cube_observation = next(
            (item for item in observations if isinstance(item, CubeDetections)), None
        )
        cubes = list(cube_observation.cubes) if cube_observation else []
        stacks = self.analyzer.find_stacks(cubes)
        game_result = self.game_engine.process_frame(cubes, frame.frame_index)
        return StackingFrameResult(
            tuple(cubes),
            tuple(tuple(stack) for stack in stacks),
            game_result,
            self.game_engine.get_game_status(),
        )

    def end_sequence(self) -> Sequence[ActionEvent]:
        return ()
