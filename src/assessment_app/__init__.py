"""Assessment app: real-time colored cube detection and stacking validation."""

__version__ = "0.1.0"
__author__ = "Assessment App Contributors"

from .app import AssessmentApp
from .config import CubeConfig
from .cube_detector import CubeDetector, Cube
from .game_logic import GameEngine, GameDifficulty
from .contracts import (
    ActionEvent,
    CubeDetections,
    Dataset,
    DatasetSample,
    Frame,
    Keypoint,
    ModelBackend,
    PersonPose,
    PoseObservation,
    SceneGraphObservation,
    VisionTask,
)
from .datasets import VideoDataset
from .model_backends import HSVCubeBackend
from .pipeline import VisionPipeline
from .tasks import CubeStackingTask

__all__ = [
    "AssessmentApp",
    "CubeConfig",
    "CubeDetector",
    "Cube",
    "GameEngine",
    "GameDifficulty",
    "ActionEvent",
    "CubeDetections",
    "Dataset",
    "DatasetSample",
    "Frame",
    "Keypoint",
    "ModelBackend",
    "PersonPose",
    "PoseObservation",
    "SceneGraphObservation",
    "VisionTask",
    "VideoDataset",
    "HSVCubeBackend",
    "VisionPipeline",
    "CubeStackingTask",
]
