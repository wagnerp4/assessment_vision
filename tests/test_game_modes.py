from pathlib import Path

from assessment_app.analysis import format_summary, summarize, with_summary
from assessment_app.contracts import (
    CubeDetections,
    DatasetSample,
    Frame,
    Keypoint,
    PersonPose,
    PoseObservation,
)
from assessment_app.cube_detector import Cube
from assessment_app.game_logic import GameDifficulty, GameEngine, GameMode, GameState
from assessment_app.grasp import CubeMove, FingerPairRule, MoveTracker, detect_grip
from assessment_app.random_stacking import RandomStackingEngine
from assessment_app.tasks import CubeStackingTask

SIZE = 60
TIP_NAMES = {
    "thumb": "thumb_tip",
    "index": "index_finger_tip",
    "middle": "middle_finger_tip",
    "ring": "ring_finger_tip",
    "pinky": "pinky_tip",
}


def cube(color, x, y, frame=0, cube_id=None):
    return Cube(color, (x, y, SIZE, SIZE), (x + SIZE // 2, y + SIZE // 2), SIZE * SIZE, 0.9, frame, cube_id)


def hand(**tips):
    far = {name: Keypoint(name, 900.0, 900.0) for name in TIP_NAMES.values()}
    far.update({TIP_NAMES[k]: Keypoint(TIP_NAMES[k], float(x), float(y)) for k, (x, y) in tips.items()})
    return far


def pinch(x, y, finger="index"):
    """A hand pinching the cube at (x, y) with thumb and `finger`; hand scale (wrist->knuckle) is 70 px."""
    keypoints = hand(thumb=(x - 5, y + 30), **{finger: (x + SIZE + 5, y + 30)})
    keypoints["wrist"] = Keypoint("wrist", x + 30.0, y + 150.0)
    keypoints["middle_finger_mcp"] = Keypoint("middle_finger_mcp", x + 30.0, y + 80.0)
    return keypoints


def away_hand():
    keypoints = hand(thumb=(700, 600), index=(760, 600))
    keypoints["wrist"] = Keypoint("wrist", 730.0, 720.0)
    keypoints["middle_finger_mcp"] = Keypoint("middle_finger_mcp", 730.0, 650.0)
    return keypoints


def test_detect_grip_picks_the_finger_opposite_the_thumb():
    red = cube("red", 100, 100)
    assert detect_grip(hand(thumb=(95, 130), index=(165, 130)), [red]).finger == "index"
    assert detect_grip(hand(thumb=(95, 130), ring=(165, 130)), [red]).finger == "ring"
    assert detect_grip(hand(thumb=(300, 300), index=(165, 130)), [red]) is None
    assert detect_grip(hand(thumb=(95, 130)), [red]) is None


def test_move_tracker_records_phases_and_release():
    tracker = MoveTracker(rest_frames=3)
    timeline = (
        [((100, 100), pinch(100, 100, "middle"))] * 3
        + [((200, 100), pinch(200, 100, "middle"))]
        + [((300, 100), pinch(300, 100, "middle"))] * 2
        + [((300, 100), away_hand())] * 4
    )
    moves, releases = [], []
    for frame, ((x, y), keypoints) in enumerate(timeline):
        cubes = [cube("red", x, y)]
        found, released = tracker.update(cubes, detect_grip(keypoints, cubes), frame, frame * 100.0, keypoints)
        moves += found
        releases += released
    assert len(moves) == 1 and releases == []
    move = moves[0]
    assert move.finger == "middle" and move.vote_share == 1.0
    assert (move.grip_onset_ms, move.lift_ms, move.settle_ms, move.release_ms) == (0.0, 300.0, 400.0, 600.0)
    assert move.coverage == 1.0
    assert abs(move.aperture_onset - 70 / 70) < 1e-6
    assert move.path_length is not None and move.path_length > 0


def test_move_tracker_ignores_jitter():
    tracker = MoveTracker(rest_frames=3)
    moves = []
    for frame in range(20):
        moves += tracker.update([cube("red", 100 + frame % 3, 100)], None, frame)[0]
    assert moves == []


def test_pair_rule_cycles_and_checks_stacks():
    rule = FingerPairRule(enabled=True)
    assert [rule.label] + [rule.cycle() for _ in range(4)] == [
        "thumb+index", "thumb+middle", "thumb+ring", "thumb+pinky", "thumb+index",
    ]
    base, top = cube("blue", 100, 300), cube("red", 100, 240)
    assert not rule.stack_ok([base, top])
    rule.record(CubeMove("red", 0, 5, "middle", (0, 0), (1, 1)))
    assert not rule.stack_ok([base, top])
    rule.record(CubeMove("red", 6, 9, "index", (0, 0), (1, 1)))
    assert rule.stack_ok([base, top])


def _run(engine, rule, frames, cubes, start, move=None):
    """Feed `frames` identical frames; optionally register a move first, as the task would."""
    moves = ()
    if move is not None:
        moves = (move,)
        rule.record(move)
    for i in range(frames):
        engine.process_frame(cubes, start + i, moves if i == 0 else ())
    return start + frames


def _round_layout(target):
    base, top = target
    apart = [cube(base, 100, 300), cube(top, 400, 300)]
    stacked = [cube(base, 100, 300), cube(top, 100, 240)]
    return apart, stacked


def test_random_game_full_round_without_finger_rule():
    rule = FingerPairRule(enabled=False)
    engine = RandomStackingEngine(pair_rule=rule, rounds=1, seed=7, step_frames=5)
    engine.start_game(GameDifficulty.EASY)
    engine.process_frame([], 0)
    assert len(engine.target) == 2 and engine.phase == "stack"
    apart, stacked = _round_layout(engine.target)

    frame = _run(engine, rule, 5, apart, 1)
    assert engine.phase == "stack"
    frame = _run(engine, rule, 5, stacked, frame)
    assert engine.phase == "unstack"
    _run(engine, rule, 5, apart, frame)
    assert engine.state == GameState.COMPLETED
    assert engine.get_game_status()["score"] == 100
    events = [event["type"] for event in engine.event_log]
    assert events == [
        "game_started", "round_started", "step_completed", "stack_completed",
        "step_completed", "round_completed", "game_completed",
    ]
    stack_step = engine.event_log[2]["data"]
    assert stack_step["achieved_ms"] < stack_step["confirmed_ms"]


def test_random_game_rejects_moves_with_wrong_pair():
    rule = FingerPairRule(enabled=True)
    engine = RandomStackingEngine(pair_rule=rule, rounds=2, seed=3, step_frames=5)
    engine.start_game(GameDifficulty.EASY)
    engine.process_frame([], 0)
    base, top = engine.target
    apart, stacked = _round_layout(engine.target)

    frame = _run(engine, rule, 8, stacked, 1, CubeMove(top, 1, 1, "ring", (0, 0), (0, 0)))
    assert engine.phase == "stack"
    assert "Re-place" in engine.feedback

    frame = _run(engine, rule, 8, stacked, frame, CubeMove(top, frame, frame, "index", (0, 0), (0, 0)))
    assert engine.phase == "unstack"

    frame = _run(engine, rule, 8, apart, frame, CubeMove(top, frame, frame, "pinky", (0, 0), (0, 0)))
    assert engine.phase == "unstack" and "Put" in engine.feedback

    frame = _run(engine, rule, 3, stacked, frame, CubeMove(top, frame, frame, "pinky", (0, 0), (0, 0)))
    _run(engine, rule, 8, apart, frame, CubeMove(top, frame, frame, "index", (0, 0), (0, 0)))
    assert engine.rounds_completed == 1 and engine.round_index == 1 and engine.phase == "stack"
    assert engine.rejected_moves == 3


def test_simple_game_needs_required_pair_when_rule_enabled():
    rule = FingerPairRule(enabled=True)
    engine = GameEngine(pair_rule=rule)
    engine.start_game(GameDifficulty.EASY)
    stacked = [cube("blue", 100, 300), cube("red", 100, 240)]
    _run(engine, rule, 30, stacked, 0, CubeMove("red", 0, 0, "middle", (0, 0), (0, 0)))
    assert engine.score.valid_stacks_completed == 0
    _run(engine, rule, 30, stacked, 30, CubeMove("red", 30, 30, "index", (0, 0), (0, 0)))
    assert engine.score.valid_stacks_completed == 1


def test_engine_timer_uses_capture_timestamps():
    engine = GameEngine()
    engine.start_game(GameDifficulty.EASY)
    stacked = [cube("blue", 100, 300), cube("red", 100, 240)]
    for i in range(25):
        engine.process_frame(stacked, i, (), t_ms=1000.0 + i * 125.0)
    confirmed = [event for event in engine.event_log if event["type"] == "stack_confirmed"][0]
    assert confirmed["data"]["achieved_ms"] == 1000.0
    assert engine.elapsed_ms == 0.0  # the stack stood from the first frame, so it was achieved at once


def _frames(task, script, start_frame=0, dt=100.0):
    """Run (cube positions, hand) pairs through the task like the pipeline would."""
    frame = start_frame
    for positions, keypoints in script:
        t = frame * dt
        cubes = tuple(cube(color, x, y, frame) for color, (x, y) in positions.items())
        observations = [CubeDetections("s", frame, t, cubes)]
        if keypoints is not None:
            person = PersonPose(None, right_hand_keypoints=keypoints)
            observations.append(PoseObservation("s", frame, t, (person,)))
        task.process(Frame("s", frame, t, None), observations)
        frame += 1
    return frame


def _carry(color, start, end, others, finger="index"):
    """Pinch at start, lift, set down at end, hold, then take the hand away."""
    (x0, y0), (x1, y1) = start, end
    mid = ((x0 + x1) // 2, (y0 + y1) // 2)
    script = [({**others, color: start}, pinch(x0, y0, finger))] * 3
    script += [({**others, color: mid}, pinch(*mid, finger))]
    script += [({**others, color: end}, pinch(x1, y1, finger))] * 3
    script += [({**others, color: end}, away_hand())] * 14
    return script


def test_task_records_a_random_round_and_analysis_links_moves():
    task = CubeStackingTask(
        difficulty=GameDifficulty.EASY, game_mode=GameMode.RANDOM, hand_backend="hands", rounds=1, seed=5
    )
    task.begin_sequence(DatasetSample("s", Path("synthetic")))
    frame = _frames(task, [({}, None)])
    base, top = task.game_engine.target
    apart = {base: (100, 300), top: (400, 300)}
    frame = _frames(task, [(apart, away_hand())] * 5, frame)
    frame = _frames(task, _carry(top, (400, 300), (100, 240), {base: (100, 300)}), frame)
    assert task.game_engine.phase == "unstack"
    frame = _frames(task, _carry(top, (100, 240), (400, 300), {base: (100, 300)}, "middle"), frame)
    assert task.game_engine.state == GameState.COMPLETED

    record = with_summary(task.session_record("local"))
    summary = record["summary"]
    assert summary["completed"] and summary["total_ms"] > 0
    (round_info,) = summary["rounds"]
    assert round_info["stack_ms"] > 0 and round_info["unstack_ms"] > 0
    place, remove = summary["steps"]
    assert (place["phase"], remove["phase"]) == ("stack", "unstack")
    assert place["has_move"] and remove["has_move"]
    assert (place["pair"], remove["pair"]) == ("thumb+index", "thumb+middle")
    assert place["pair_source"] == "detected"
    assert place["transport_ms"] == 100.0 and place["dwell_ms"] == 300.0
    assert not place["low_confidence"]
    assert {pair["pair"] for pair in summary["pairs"]} == {"thumb+index", "thumb+middle"}
    assert summary["quality"]["steps_without_move"] == 0
    assert "Round 1" in format_summary(summary)


def test_required_pair_owns_moves_when_rule_is_on():
    task = CubeStackingTask(
        difficulty=GameDifficulty.EASY, game_mode=GameMode.RANDOM, hand_backend="hands", rounds=1, seed=5
    )
    task.begin_sequence(DatasetSample("s", Path("synthetic")))
    frame = _frames(task, [({}, None)])
    task.set_finger_rule(True)
    task.next_pair()
    base, top = task.game_engine.target
    frame = _frames(task, _carry(top, (400, 300), (100, 240), {base: (100, 300)}, "middle"), frame)
    summary = summarize(task.session_record("web"))
    assert summary["finger_rule_used"]
    assert summary["steps"][0]["pair"] == "thumb+middle"
    assert summary["steps"][0]["pair_source"] == "required"
    rule_events = [e["data"] for e in task.game_engine.event_log if e["type"] == "rule_changed"]
    assert rule_events[-1] == {"enabled": True, "required": "thumb+middle"}


def test_summary_figure_renders_for_both_modes():
    import pytest

    pytest.importorskip("matplotlib")
    from assessment_app.summary_plot import render_summary

    for mode in (GameMode.SIMPLE, GameMode.RANDOM):
        task = CubeStackingTask(difficulty=GameDifficulty.EASY, game_mode=mode, hand_backend="hands", rounds=1, seed=5)
        task.begin_sequence(DatasetSample("s", Path("synthetic")))
        frame = _frames(task, [({}, None)])
        if mode == GameMode.RANDOM:
            base, top = task.game_engine.target
        else:
            base, top = "blue", "red"
        _frames(task, _carry(top, (400, 300), (100, 240), {base: (100, 300)}), frame)
        record = with_summary(task.session_record("local"))
        image = render_summary(record, record["summary"])
        assert image.ndim == 3 and image.shape[2] == 3 and image.shape[0] > 500
