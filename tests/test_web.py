import json
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from assessment_app.web import GameSession, app  # noqa: E402

DEMO = Path(__file__).parent.parent / "docs" / "assets" / "demo.mp4"


def _jpeg_frame() -> bytes:
    frame = None
    if DEMO.exists():
        cap = cv2.VideoCapture(str(DEMO))
        ok, frame = cap.read()
        cap.release()
        frame = frame if ok else None
    if frame is None:
        frame = np.zeros((540, 960, 3), np.uint8)
    frame = cv2.resize(frame, (960, 540))
    return cv2.imencode(".jpg", frame)[1].tobytes()


_clock = [0.0]


def _packet() -> bytes:
    """A frame as the browser sends it: capture time (ms, float64 LE) followed by the JPEG."""
    _clock[0] += 100.0
    return struct.pack("<d", _clock[0]) + _jpeg_frame()


def test_index_and_health():
    client = TestClient(app)
    assert client.get("/healthz").text == "ok"
    assert "getUserMedia" in client.get("/").text


def test_websocket_round_trip():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "start", "difficulty": "medium"}))
        ws.send_bytes(_packet())
        status = json.loads(ws.receive_text())
        assert status["type"] == "status"
        assert status["status"]["level"] == "MEDIUM"
        annotated = cv2.imdecode(np.frombuffer(ws.receive_bytes(), np.uint8), cv2.IMREAD_COLOR)
        assert annotated.shape[:2] == (540, 960)

        ws.send_bytes(struct.pack("<d", 1.0) + b"not a jpeg")
        assert json.loads(ws.receive_text())["type"] == "error"


def test_timings_per_backend():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(_packet())
        status = json.loads(ws.receive_text())
        ws.receive_bytes()
    assert set(status["timings"]) == {"cubes"}
    assert status["server_ms"] >= status["timings"]["cubes"]


def test_unknown_pose_mode_is_rejected():
    client = TestClient(app)
    with client.websocket_connect("/ws?pose=bogus") as ws:
        assert "Unknown pose mode" in json.loads(ws.receive_text())["message"]


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "models" / "hand_landmarker.task").is_file(),
    reason="hand model not downloaded",
)
def test_hands_mode_reports_pose_timing():
    pytest.importorskip("mediapipe")
    client = TestClient(app)
    with client.websocket_connect("/ws?pose=hands") as ws:
        ws.send_bytes(_packet())
        status = json.loads(ws.receive_text())
        ws.receive_bytes()
    assert set(status["timings"]) == {"cubes", "hands"}


def _status(ws):
    ws.send_bytes(_packet())
    status = json.loads(ws.receive_text())["status"]
    ws.receive_bytes()
    return status


def test_random_game_and_finger_rule_without_hands():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "start", "difficulty": "hard", "game": "random"}))
        ws.send_text(json.dumps({"type": "finger_rule", "enabled": True}))
        status = _status(ws)
    assert status["mode"] == "random" and len(status["target"]) == 4
    assert status["required_pair"] is None


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "models" / "hand_landmarker.task").is_file(),
    reason="hand model not downloaded",
)
def test_finger_rule_toggle_and_next_pair_with_hands():
    pytest.importorskip("mediapipe")
    client = TestClient(app)
    with client.websocket_connect("/ws?pose=hands") as ws:
        assert _status(ws)["required_pair"] is None
        ws.send_text(json.dumps({"type": "finger_rule", "enabled": True}))
        assert _status(ws)["required_pair"] == "thumb+index"
        ws.send_text(json.dumps({"type": "next_pair"}))
        assert _status(ws)["required_pair"] == "thumb+middle"
        ws.send_text(json.dumps({"type": "finger_rule", "enabled": False}))
        assert _status(ws)["required_pair"] is None


def test_analysis_script_is_served():
    client = TestClient(app)
    response = client.get("/analysis.js")
    assert response.status_code == 200 and "Analysis" in response.text


def test_summary_is_sent_once_per_progress_event():
    session = GameSession()
    assert session.summary() is None
    session.app.game_engine.log_event(
        "stack_confirmed", {"colors": ["blue", "red"], "achieved_ms": 500.0, "confirmed_ms": 900.0}
    )
    record = session.summary()
    assert record["client"] == "web" and "events" not in record
    assert record["summary"]["stacks"][0]["colors"] == ["blue", "red"]
    assert session.summary() is None
    session.close()
