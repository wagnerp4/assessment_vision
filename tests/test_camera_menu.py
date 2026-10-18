from pathlib import Path

import cv2
import numpy as np
import pytest

from assessment_app.app import AssessmentApp
from assessment_app.camera_menu import CameraMenu, key_name
from assessment_app.contracts import DatasetSample
from assessment_app.game_logic import GameDifficulty, GameMode, GameState
from assessment_app.model_backends import MediaPipeHandBackend, MediaPipeHolisticBackend


def _menu(**kwargs) -> CameraMenu:
    defaults = dict(
        game=GameMode.SIMPLE,
        difficulty=GameDifficulty.EASY,
        hands=True,
        pose_backend="hands",
        finger_rule=False,
        record=False,
    )
    defaults.update(kwargs)
    return CameraMenu(**defaults)


def _center(button):
    x1, y1, x2, y2 = button.rect
    return (x1 + x2) // 2, (y1 + y2) // 2


def test_menu_cycles_difficulty_and_game_and_remembers_finger_pairs_while_hands_are_off():
    menu = _menu(finger_rule=True)
    menu.activate("difficulty", 1)
    assert menu.difficulty == GameDifficulty.MEDIUM
    menu.activate("difficulty", -1)
    assert menu.difficulty == GameDifficulty.EASY
    menu.activate("difficulty", -1)
    assert menu.difficulty == GameDifficulty.EXPERT

    menu.activate("game")
    assert menu.game == GameMode.RANDOM
    menu.activate("hands")
    assert menu.hands is False
    assert menu.finger_rule is True
    assert menu.finger_pairs_active is False
    menu.activate("finger_rule")
    menu.activate("next_pair")
    assert menu.finger_rule is True
    assert menu.pair_index == 0
    menu.activate("hands")
    assert menu.finger_pairs_active is True
    menu.activate("next_pair", -1)
    assert menu.pair_index == len(("index", "middle", "ring", "pinky")) - 1
    menu.activate("pose_backend")
    assert menu.pose_backend == "holistic"
    menu.activate("record")
    assert menu.record is True


def test_menu_click_starts_and_disabled_controls_ignore_clicks():
    menu = _menu(hands=False, finger_rule=True)
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[-20:, :] = (0, 255, 0)
    rendered = menu.render(frame)
    assert rendered.shape == frame.shape
    green = np.full((20, 1280, 3), (0, 255, 0), np.uint8)
    assert np.array_equal(frame[-20:], green)
    assert np.array_equal(rendered[-20:], green)
    assert menu.handle_click(0, 0) is None

    start = next(button for button in menu.buttons if button.id == "start")
    assert menu.handle_click(*_center(start)) == "start"

    pairs = next(button for button in menu.buttons if button.id == "finger_rule")
    assert pairs.enabled is False
    assert menu.handle_click(*_center(pairs)) is None
    assert menu.finger_rule is True

    for button in menu.buttons:
        x1, y1, x2, y2 = button.rect
        assert 0 <= x1 < x2 <= 1280
        assert 0 <= y1 < y2 <= 720


def test_menu_keys_move_focus_skip_disabled_controls_and_start_or_quit():
    menu = _menu(hands=False)
    menu.focus = 3
    assert key_name(2621440) == "down"
    menu.handle_key(2621440)
    assert menu.focus == 6  # start, skipping finger pairs and next pair
    assert menu.handle_key(ord("s")) == "start"
    assert menu.handle_key(13) == "start"
    assert menu.handle_key(ord("q")) == "quit"
    assert menu.handle_key(27) == "quit"

    menu.focus = 1
    menu.handle_key(ord(" "))
    assert menu.difficulty == GameDifficulty.MEDIUM
    menu.handle_key(2555904)  # right
    assert menu.difficulty == GameDifficulty.HARD
    menu.handle_key(2424832)  # left
    assert menu.difficulty == GameDifficulty.MEDIUM


def test_apply_session_uses_menu_choices_and_defers_pose_load(monkeypatch):
    loaded = []
    monkeypatch.setattr(
        MediaPipeHandBackend, "load", lambda self, device=None: loaded.append("hands")
    )
    monkeypatch.setattr(
        MediaPipeHolisticBackend, "load", lambda self, device=None: loaded.append("holistic")
    )
    app = AssessmentApp(enable_pose=True, pose_model_path="missing-hand.task", finger_rule=True)
    same = app.pose_backend
    app.apply_session(
        game_mode=GameMode.RANDOM,
        difficulty=GameDifficulty.HARD,
        hands=True,
        pose_backend="hands",
        finger_rule=True,
        pair_index=2,
    )
    assert app.pose_backend is same
    assert loaded == []
    assert app.task.game_mode == GameMode.RANDOM
    assert app.task.difficulty == GameDifficulty.HARD
    assert app.task.pair_rule.enabled is True
    assert app.task.pair_rule.label == "thumb+ring"

    app.apply_session(
        game_mode=GameMode.SIMPLE,
        difficulty=GameDifficulty.EASY,
        hands=True,
        pose_backend="holistic",
        finger_rule=False,
        pair_index=0,
    )
    assert isinstance(app.pose_backend, MediaPipeHolisticBackend)
    assert app.pose_backend._holistic is None
    assert app.task.hand_backend == "holistic"
    assert loaded == []
    assert len(app.pipeline.backends) == 2

    app.apply_session(
        game_mode=GameMode.SIMPLE,
        difficulty=GameDifficulty.EASY,
        hands=False,
        pose_backend="holistic",
        finger_rule=True,
        pair_index=1,
    )
    assert app.pose_backend is None
    assert app.pipeline.backends == (app.backend,)
    assert app.task.pair_rule.enabled is False
    app.pipeline.close()


def test_apply_session_rolls_back_when_a_loaded_pipeline_cannot_open_the_pose_model(monkeypatch):
    def fail_load(self, device=None):
        raise FileNotFoundError("Hand model not found: missing")

    monkeypatch.setattr(MediaPipeHandBackend, "load", fail_load)
    app = AssessmentApp()
    app.pipeline._loaded = True
    with pytest.raises(FileNotFoundError):
        app.apply_session(
            game_mode=GameMode.SIMPLE,
            difficulty=GameDifficulty.EASY,
            hands=True,
            pose_backend="hands",
            finger_rule=False,
            pair_index=0,
        )
    assert app.pose_backend is None
    assert app.pipeline.backends == (app.backend,)
    app.pipeline.close()


def _silence_windows(monkeypatch, keys):
    sequence = iter(keys)

    def next_key(_delay=0):
        return next(sequence, -1)

    monkeypatch.setattr(cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "waitKey", next_key)
    monkeypatch.setattr(cv2, "waitKeyEx", next_key)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *_args, **_kwargs: None)


def test_live_game_returns_to_the_menu_after_the_summary(monkeypatch):
    app = AssessmentApp()
    app.cap = type("Cap", (), {"read": lambda self: (True, np.zeros((480, 640, 3), np.uint8))})()
    app.is_running = True
    menu = CameraMenu(game=GameMode.RANDOM, difficulty=GameDifficulty.MEDIUM, hands=False)
    saved = []

    def fake_save(show=False):
        saved.append(show)
        app._summary_image = np.zeros((30, 40, 3), np.uint8)
        app._saved_session = app.task.session_id
        return None

    monkeypatch.setattr(app, "save_run", fake_save)
    _silence_windows(monkeypatch, [ord("q")])
    clicks = []
    assert app._run_camera_menu(menu, "win", clicks) == "quit"

    original = app._process_frame

    def finish_on_first_frame(frame, **kwargs):
        result = original(frame, **kwargs)
        app.game_engine.state = GameState.COMPLETED
        app.game_engine.log_event("stack_confirmed", {})
        return result

    monkeypatch.setattr(app, "_process_frame", finish_on_first_frame)
    _silence_windows(monkeypatch, [-1])
    app.is_running = True
    app._play_camera_game(menu, "win", clicks, DatasetSample("test", Path("camera:0")))

    assert saved == [False]
    assert app.task.game_mode == GameMode.RANDOM
    assert app.task.difficulty == GameDifficulty.MEDIUM
    assert app.is_running is True
    app.pipeline.close()
