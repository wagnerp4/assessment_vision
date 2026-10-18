"""Test that all modules import correctly."""


def test_import_assessment_app():
    """Test that the main app can be imported."""
    from assessment_app import AssessmentApp

    assert AssessmentApp is not None


def test_import_config():
    """Test that config can be imported."""
    from assessment_app import CubeConfig

    assert CubeConfig is not None


def test_import_detector():
    """Test that detector can be imported."""
    from assessment_app import Cube, CubeDetector

    assert CubeDetector is not None
    assert Cube is not None
    assert CubeDetector() is not None


def test_import_game():
    """Test that game logic can be imported."""
    from assessment_app import GameDifficulty, GameEngine

    assert GameEngine is not None
    assert GameDifficulty is not None


def test_import_pipeline_contracts():
    from assessment_app import (
        CubeStackingTask,
        Dataset,
        Frame,
        HSVCubeBackend,
        MediaPipeHolisticBackend,
        ModelBackend,
        VideoDataset,
        VisionPipeline,
        VisionTask,
    )

    assert all(
        (
            CubeStackingTask,
            Dataset,
            Frame,
            HSVCubeBackend,
            MediaPipeHolisticBackend,
            ModelBackend,
            VideoDataset,
            VisionPipeline,
            VisionTask,
        )
    )
