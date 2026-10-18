import importlib.util
from pathlib import Path

import numpy as np
import pytest

from assessment_app.fastener_model import (
    CLASSES,
    FastenerSmoother,
    YoloFastenerBackend,
    best_per_class,
    create_fastener_backend,
    decode,
    geometry,
    letterbox,
)
from assessment_app.fasteners import FastenerBackend

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_letterbox_and_decode_map_boxes_back_to_image_pixels():
    image = np.zeros((720, 1280, 3), np.uint8)
    padded, scale, pad = letterbox(image, 416)
    assert padded.shape == (416, 416, 3)
    assert scale == pytest.approx(416 / 1280)
    assert pad == (0.0, pytest.approx(91.0, abs=1))

    # One nut at image (640, 360) size 64, plus a weaker duplicate that NMS removes.
    anchors = np.zeros((4 + len(CLASSES), 3), np.float32)
    for column, score in ((0, 0.9), (1, 0.6)):
        anchors[:4, column] = [640 * scale + pad[0], 360 * scale + pad[1], 64 * scale, 64 * scale]
        anchors[4 + 3, column] = score
    anchors[4 + 0, 2] = 0.1  # below the confidence cut
    found = decode(anchors[None], scale, pad)
    assert len(found) == 1
    cls, score, box = found[0]
    assert CLASSES[cls] == "nut" and score == pytest.approx(0.9)
    assert box == pytest.approx((608, 328, 672, 392), abs=1)


def test_geometry_measures_depth_from_tip_and_tells_a_nut_beside_the_screw():
    boxes = {
        "screw": (100, 190, 400, 210),
        "screw_head": (100, 185, 140, 215),
        "screw_tip": (380, 192, 400, 208),
        "nut": (250, 175, 290, 225),
    }
    found = geometry(boxes)
    assert found["nut_on_screw"] is True
    # Tip center 390, head center 120: the nut at 270 is (390 - 270) / 270 of the way.
    assert found["depth"] == pytest.approx(120 / 270, abs=0.01)

    beside = dict(boxes, nut=(250, 300, 290, 350))
    found = geometry(beside)
    assert found["nut_on_screw"] is False and found["depth"] is None
    assert found["nut_center"] == (270.0, 325.0)


def test_geometry_uses_the_nut_as_scale_when_the_head_is_hidden():
    boxes = {
        "screw": (200, 190, 400, 210),  # only the shaft outside the hand is visible
        "screw_tip": (380, 192, 400, 208),
        "nut": (330, 170, 390, 230),  # 60 px across: screw is 60 / 0.30 = 200 px long
    }
    found = geometry(boxes)
    assert found["head_estimated"] is True
    assert found["nut_on_screw"] is True
    assert found["depth"] == pytest.approx(30 / 200, abs=0.03)
    assert np.hypot(found["screw_head"][0] - 390, found["screw_head"][1] - 200) == pytest.approx(200, abs=2)


def test_best_per_class_keeps_the_most_confident_box():
    picked = best_per_class([(3, 0.5, (0, 0, 1, 1)), (3, 0.8, (5, 5, 6, 6)), (0, 0.7, (1, 1, 2, 2))])
    assert picked == {"nut": (5, 5, 6, 6), "screw": (1, 1, 2, 2)}


def test_smoother_holds_lost_parts_and_takes_the_median_depth():
    smoother = FastenerSmoother(hold_frames=2, window=5)
    seen = {"screw_tip": (0.0, 0.0), "screw_head": (100.0, 0.0), "nut_center": (40.0, 0.0), "nut_on_screw": True}
    for depth in (0.40, 0.42, 0.90, 0.41):  # one wild frame
        result = smoother.update(dict(seen, depth=depth))
    assert result["depth"] == pytest.approx(0.415, abs=0.01)

    lost = {"screw_tip": None, "screw_head": None, "nut_center": None, "nut_on_screw": False, "depth": None}
    held = smoother.update(dict(lost))
    assert held["screw_tip"] is not None and held["nut_on_screw"] is True
    smoother.update(dict(lost))
    gone = smoother.update(dict(lost))
    assert gone["screw_tip"] is None


def test_app_uses_the_classical_detector_until_a_model_is_exported(tmp_path):
    assert isinstance(create_fastener_backend(tmp_path / "missing.onnx"), FastenerBackend)
    model = tmp_path / "fastener_detector.onnx"
    model.write_bytes(b"")
    (tmp_path / "fastener_detector.json").write_text('{"imgsz": 320, "nut_ratio": 0.3}')
    backend = create_fastener_backend(model)
    assert isinstance(backend, YoloFastenerBackend)
    assert backend.input_size == 320


def test_dataset_check_counts_instances_and_flags_bad_lines(tmp_path):
    build = _script("build_fastener_dataset")
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
    for stem in ("a", "b", "c"):
        (tmp_path / "images" / "train" / f"{stem}.jpg").write_bytes(b"")
    (tmp_path / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.4 0.1\n3 0.6 0.5 0.05 0.08\n")
    (tmp_path / "labels" / "train" / "b.txt").write_text("")  # negative frame
    (tmp_path / "labels" / "train" / "orphan.txt").write_text("7 0.5 0.5 1.2 0.1\n")
    counts, problems = build.check_labels(tmp_path)
    assert counts["train"]["images"] == 3
    assert counts["train"]["labeled"] == 2
    assert counts["train"]["unlabeled"] == 1
    assert counts["train"]["screw"] == 1 and counts["train"]["nut"] == 1
    assert any("orphan.txt has no image" in problem for problem in problems)
    assert any("orphan.txt:1" in problem for problem in problems)
