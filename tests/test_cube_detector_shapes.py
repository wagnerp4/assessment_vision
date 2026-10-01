"""Regression tests for OpenCV convexity-defect return shapes."""

import cv2
import numpy as np
import pytest

from assessment_app.cube_detector import CubeDetector


@pytest.mark.parametrize("shape", [(1, 4), (1, 1, 4)])
def test_top_notch_accepts_convexity_defect_shape_variants(monkeypatch, shape):
    contour = np.array(
        [[[0, 0]], [[10, 0]], [[10, 10]], [[5, 4]], [[0, 10]]], dtype=np.int32
    )
    hull_indices = np.array([[0], [1], [2], [4]], dtype=np.int32)
    defects = np.array([[0, 1, 3, 1024]], dtype=np.int32).reshape(shape)
    monkeypatch.setattr(cv2, "convexHull", lambda contour, returnPoints: hull_indices)
    monkeypatch.setattr(cv2, "convexityDefects", lambda contour, hull: defects)

    assert CubeDetector._top_notch(contour, y=0, h=10)
