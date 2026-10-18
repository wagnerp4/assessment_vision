from pathlib import Path

import cv2
import numpy as np
import pytest

from assessment_app.analysis import format_summary, summarize
from assessment_app.app import AssessmentApp
from assessment_app.camera_menu import CameraMenu
from assessment_app.contracts import DatasetSample
from assessment_app.fastener_model import YoloFastenerBackend
from assessment_app.fasteners import FastenerBackend, FastenerDetector, NutTurnTracker
from assessment_app.game_logic import GameDifficulty, GameMode, GameState
from assessment_app.screwing import (
    HOLD_FRAMES,
    ScrewingEngine,
    ScrewObservation,
    axis_depth,
    thread_progress,
)


def _hexagon(center, radius, rotation_deg):
    points = []
    for index in range(6):
        angle = np.deg2rad(rotation_deg + index * 60)
        points.append([center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)])
    return np.array(points, np.int32)


def test_detector_finds_a_separate_nut_and_a_nut_midway_on_the_screw():
    separate = np.full((240, 400, 3), 255, np.uint8)
    cv2.rectangle(separate, (30, 40), (250, 58), (150, 150, 150), -1)
    cv2.rectangle(separate, (220, 28), (260, 70), (120, 120, 120), -1)
    cv2.fillPoly(separate, [_hexagon((330, 160), 22, 0)], (140, 140, 140))
    found = FastenerDetector().detect(separate)
    assert found["screw_tip"] is not None and found["screw_head"] is not None
    assert found["nut_center"] is not None
    assert found["nut_on_screw"] is False
    assert found["depth"] is None

    seated = np.full((240, 400, 3), 255, np.uint8)
    cv2.rectangle(seated, (40, 110), (280, 130), (150, 150, 150), -1)
    cv2.rectangle(seated, (250, 96), (292, 144), (120, 120, 120), -1)
    cv2.rectangle(seated, (145, 96), (185, 144), (140, 140, 140), -1)
    found = FastenerDetector().detect(seated)
    assert found["nut_on_screw"] is True
    assert found["depth"] == pytest.approx(0.5, abs=0.08)
    assert found["nut_center"][0] == pytest.approx(165, abs=20)


def test_turn_tracker_counts_one_revolution_and_ignores_hidden_frames():
    tracker = NutTurnTracker()
    for degrees in range(0, 361, 10):
        tracker.update(float(degrees))
    assert tracker.side_steps == 6
    assert tracker.turns == pytest.approx(1.0)

    tracker.update(None)
    tracker.update(None)
    assert tracker.side_steps == 6
    assert tracker.turns == pytest.approx(1.0)


def _frame(engine, t, frame_id, **kwargs):
    values = dict(
        screw_visible=True,
        nut_visible=True,
        depth=None,
        nut_on_screw=False,
        turns=0.0,
        side_steps=0,
        left_on_screw=True,
        right_on_nut=True,
        hands_seen=True,
    )
    values.update(kwargs)
    return engine.process_frame(ScrewObservation(**values), frame_id, t)


def _hold(engine, start_t, start_frame, step=100, **kwargs):
    """HOLD_FRAMES frames in a row; step=250 spans the 1 s that pickup and unscrew also need."""
    for index in range(HOLD_FRAMES):
        _frame(engine, start_t + index * step, start_frame + index, **kwargs)


def test_engine_runs_setup_pickup_screw_unscrew_putdown_in_order():
    engine = ScrewingEngine()
    engine.start_game(GameDifficulty.HARD)
    assert engine.target == 0.50

    _frame(engine, 0, 0, screw_visible=False, hands_seen=False, left_on_screw=False, right_on_nut=False)
    assert engine.phase == "setup"
    assert engine.feedback == "Place the screw on the table"
    _frame(engine, 100, 1, hands_seen=False, left_on_screw=False, right_on_nut=False)
    assert engine.feedback == "Show your hands to the camera"

    _hold(engine, 1000, 10, left_on_screw=False, right_on_nut=False)
    assert engine.phase == "pickup"
    assert "left hand" in engine.feedback

    _frame(engine, 2000, 20, left_on_screw=True, right_on_nut=False)
    assert engine.phase == "pickup"
    assert engine.feedback == "Pick up the nut with your right hand"
    # Hands reaching the parts on the table is not a pickup, however long it lasts.
    _hold(engine, 2100, 21, step=250)
    assert engine.phase == "pickup"
    assert engine.feedback == "Lift the screw and the nut off the table"
    # Lifted, but not yet for a full second.
    _hold(engine, 3500, 30, parts_lifted=True)
    assert engine.phase == "pickup"
    # The streak continues; it passes once it is a second old (3500 + 1000 ms).
    _hold(engine, 4000, 35, step=250, parts_lifted=True)
    assert engine.phase == "screw"
    assert engine.pickup_ms == pytest.approx(3100, abs=1)

    _frame(engine, 5000, 40)
    assert "tip" in engine.feedback
    _frame(engine, 5500, 41, depth=0.35, nut_on_screw=True, turns=2, side_steps=12)
    assert engine.feedback == "Screw on — 35% of 50%, 2 turns"

    # Reaching the mark with no fingertip on the nut does not count.
    _hold(engine, 6000, 45, depth=0.55, nut_on_screw=True, nut_gripped=False, turns=3, side_steps=18)
    assert engine.phase == "screw"
    _hold(engine, 7000, 50, depth=0.55, nut_on_screw=True, nut_gripped=True, turns=3, side_steps=18)
    assert engine.phase == "unscrew"
    assert "unscrew" in engine.feedback
    assert engine.screw_ms == pytest.approx(2900, abs=1)

    # A hidden nut does not count as unscrewed, nor does a nut that only reads "off" at the tip.
    _hold(engine, 8000, 60, nut_visible=False, nut_on_screw=False)
    assert engine.phase == "unscrew"
    _hold(engine, 8500, 65, step=250, nut_on_screw=False)
    assert engine.phase == "unscrew"
    _hold(engine, 10000, 70, step=250, nut_on_screw=False, nut_clear=True, turns=4, side_steps=24)
    assert engine.phase == "putdown"
    assert engine.unscrew_ms == pytest.approx(3600, abs=1)

    _hold(engine, 11500, 80, hands_on_parts=True, turns=4, side_steps=24)
    assert engine.phase == "putdown"
    assert engine.feedback == "Take your hands away from the parts"
    _hold(engine, 12600, 90, hands_on_parts=False, turns=4, side_steps=24)
    assert engine.state == GameState.COMPLETED
    assert engine.feedback == "Finished"
    assert engine.putdown_ms == pytest.approx(2000, abs=1)
    phases = [event["data"]["phase"] for event in engine.event_log if event["type"] == "phase_completed"]
    assert phases == ["setup", "pickup", "screw", "unscrew", "putdown"]

    record = {
        "events": engine.event_log,
        "game": {"mode": "screwing", "difficulty": "HARD"},
        "hands": {"enabled": True},
    }
    summary = summarize(record)
    assert summary["completed"] is True
    assert summary["turns"] == 4
    assert summary["side_steps"] == 24
    assert summary["target"] == 0.5
    assert summary["depth"] == pytest.approx(0.55, abs=0.01)
    assert summary["pickup_ms"] == pytest.approx(3100, abs=1)
    assert summary["putdown_ms"] == pytest.approx(2000, abs=1)
    assert summary["spins"] == 0
    text = format_summary(summary)
    assert "Pick up 3.1s" in text
    assert "screw on" in text
    assert "put down 2.0s" in text


def test_menu_and_session_can_select_screwing():
    menu = CameraMenu()
    menu.activate("game")
    menu.activate("game")
    assert menu.game == GameMode.SCREWING
    rendered = menu.render(np.zeros((480, 800, 3), np.uint8))
    assert rendered.shape == (480, 800, 3)
    assert any(button.id == "game" for button in menu.buttons)

    app = AssessmentApp()
    app.apply_session(
        game_mode=GameMode.SCREWING,
        difficulty=GameDifficulty.EASY,
        hands=False,
        pose_backend="hands",
        finger_rule=False,
        pair_index=0,
    )
    assert app.task.game_mode == GameMode.SCREWING
    assert app.task.difficulty == GameDifficulty.EASY
    app.pipeline.begin_sequence(DatasetSample("screwing", Path("camera:0")))
    assert app.task.game_engine.target == 0.25
    assert app.task.game_engine.phase == "setup"
    # The learned detector replaces the classical one once models/fastener_detector.onnx exists.
    assert isinstance(app.pipeline.backends[-1], (FastenerBackend, YoloFastenerBackend))
    app.pipeline.close()


def test_depth_counts_only_the_thread():
    # Nut center on the tip (half a nut thickness in) is 0 %, at the end of the thread 100 %.
    assert thread_progress(0.065) == pytest.approx(0.0)
    assert thread_progress(0.565) == pytest.approx(1.0)
    assert thread_progress(0.315) == pytest.approx(0.5)
    # The plain shaft near the head cannot be reached, so the scale stops at the thread end.
    assert thread_progress(0.80) == 1.0
    assert thread_progress(0.0) == 0.0
    assert thread_progress(None) is None
    assert axis_depth(thread_progress(0.4)) == pytest.approx(0.4)


def test_nut_moving_with_no_finger_on_it_counts_as_a_spin():
    engine = ScrewingEngine()
    engine.start_game(GameDifficulty.HARD)
    engine.phase = "screw"
    _frame(engine, 0, 0, depth=0.10, nut_on_screw=True, nut_gripped=True)
    # Released, then the nut travels along the thread by itself.
    _frame(engine, 100, 1, depth=0.10, nut_on_screw=True, nut_gripped=False)
    _frame(engine, 200, 2, depth=0.16, nut_on_screw=True, nut_gripped=False)
    assert engine.spins == 0
    _frame(engine, 300, 3, depth=0.25, nut_on_screw=True, nut_gripped=False)
    assert engine.spins == 1
    assert "don't spin" in engine.feedback
    assert [event["type"] for event in engine.event_log].count("nut_spun") == 1
    # Turning it with the fingers, regripping in between, is fine.
    for frame, depth in enumerate((0.27, 0.27, 0.30, 0.30, 0.33), 4):
        _frame(engine, 3000 + frame * 100, frame, depth=depth, nut_on_screw=True, nut_gripped=frame % 2 == 0)
    assert engine.spins == 1
    # When the hand is not seen there is no evidence either way.
    _frame(engine, 4000, 20, depth=0.45, nut_on_screw=True, nut_gripped=None)
    assert engine.spins == 1


def test_nut_counts_as_clear_only_well_past_the_tip_or_beside_the_shaft():
    from assessment_app.tasks import ScrewingTask

    tip, head, size = (300.0, 200.0), (100.0, 200.0), 40.0
    assert not ScrewingTask._nut_clear(tip, head, (290.0, 200.0), size)  # at the tip, still threaded
    assert not ScrewingTask._nut_clear(tip, head, (320.0, 200.0), size)  # just off: "off" flickers here
    assert ScrewingTask._nut_clear(tip, head, (345.0, 200.0), size)  # one nut width past the tip
    assert ScrewingTask._nut_clear(tip, head, (250.0, 250.0), size)  # held beside the shaft
    assert not ScrewingTask._nut_clear(None, head, (345.0, 200.0), size)
