"""Assessment app: real-time colored cube detection and stacking validation."""

__version__ = "0.1.0"
__author__ = "Assessment App Contributors"

from .app import AssessmentApp
from .config import CubeConfig
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
from .cube_detector import Cube, CubeDetector
from .datasets import VideoDataset
from .game_logic import GameDifficulty, GameEngine
from .model_backends import HSVCubeBackend, MediaPipeHandBackend, MediaPipeHolisticBackend
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
    "MediaPipeHolisticBackend",
    "MediaPipeHandBackend",
    "VisionPipeline",
    "CubeStackingTask",
]
