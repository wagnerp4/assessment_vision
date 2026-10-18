"""Browser-camera web server: the page streams webcam JPEGs over a WebSocket and gets annotated frames back."""

from __future__ import annotations

import asyncio
import json
import os
import struct
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse

from .analysis import summarize
from .app import AssessmentApp
from .contracts import DatasetSample
from .game_logic import GameDifficulty, GameMode

STATIC_DIR = Path(__file__).parent / "static"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
POSE_MODELS = {"hands": "hand_landmarker.task", "holistic": "holistic_landmarker.task"}
MAX_FRAME_BYTES = 2 * 1024 * 1024
HEADER = struct.Struct("<d")  # browser capture time (performance.now(), ms) before the JPEG bytes
SUMMARY_EVENTS = {
    "step_completed",
    "stack_completed",
    "round_completed",
    "stack_confirmed",
    "game_completed",
    "depth_reached",
}
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "4"))
JPEG_QUALITY = 75

app = FastAPI(title="Assessment Vision")
_active_sessions = 0


class _TimedBackend:
    """Delegates to a model backend and records how long its last predict call took."""

    def __init__(self, backend, label: str) -> None:
        self.backend = backend
        self.label = label
        self.last_ms = 0.0

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def predict(self, frame):
        started = time.perf_counter()
        try:
            return self.backend.predict(frame)
        finally:
            self.last_ms = (time.perf_counter() - started) * 1000


class GameSession:
    """One pipeline per browser connection so trackers and scores never leak between players."""

    def __init__(self, pose: str = "off") -> None:
        self.app = AssessmentApp(
            enable_pose=pose in POSE_MODELS,
            pose_backend=pose if pose in POSE_MODELS else "hands",
            pose_model_path=MODELS_DIR / POSE_MODELS[pose] if pose in POSE_MODELS else None,
        )
        self._wrap_backends()
        self.sample = DatasetSample("web-session", Path("browser-camera"))
        self.frame_index = 0
        self._summarized = 0
        self.start(GameDifficulty.EASY)

    def timings(self) -> dict[str, float]:
        return {backend.label: backend.last_ms for backend in self.timed}

    def start(self, difficulty: GameDifficulty, game: GameMode = GameMode.SIMPLE) -> None:
        rule = self.app.task.pair_rule
        self.app.apply_session(
            game_mode=game,
            difficulty=difficulty,
            hands=self.app.pose_backend is not None,
            pose_backend=self.app.pose_backend_name,
            finger_rule=rule.enabled,
            pair_index=rule.index,
        )
        self._wrap_backends()
        self.app.pipeline.begin_sequence(self.sample)
        self.frame_index = 0
        self._summarized = 0

    def _wrap_backends(self) -> None:
        raw = []
        for backend in self.app.pipeline.backends:
            raw.append(backend.backend if isinstance(backend, _TimedBackend) else backend)
        labels = []
        for backend in raw:
            if backend is self.app.backend:
                labels.append("cubes")
            elif backend is self.app.pose_backend:
                labels.append(self.app.pose_backend_name)
            else:
                labels.append("fasteners")
        self.timed = tuple(_TimedBackend(backend, label) for backend, label in zip(raw, labels))
        self.app.pipeline.backends = self.timed

    def set_finger_rule(self, enabled: bool) -> None:
        self.app.task.set_finger_rule(enabled)

    def next_pair(self) -> None:
        self.app.task.next_pair()

    def summary(self) -> dict | None:
        """Session metadata plus its analysis, once the game has made progress worth showing."""
        events = self.app.game_engine.event_log
        progress = [event for event in events if event["type"] in SUMMARY_EVENTS]
        if len(progress) == self._summarized:
            return None
        self._summarized = len(progress)
        record = self.app.task.session_record("web")
        summary = summarize(record)
        record.pop("events")
        return {**record, "summary": summary}

    def reset(self) -> None:
        self.app.game_engine.reset_game()

    def process(self, data: bytes) -> tuple[bytes, dict] | None:
        if len(data) <= HEADER.size:
            return None
        (captured_ms,) = HEADER.unpack_from(data)
        frame = cv2.imdecode(np.frombuffer(data[HEADER.size :], np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        in_h, in_w = frame.shape[:2]
        # Detection thresholds are calibrated in pixels at FRAME_RESIZE_HEIGHT.
        target_h = self.app.config.FRAME_RESIZE_HEIGHT
        if in_h != target_h:
            frame = cv2.resize(frame, (round(in_w * target_h / in_h), target_h))
        result = self.app._process_frame(
            frame, frame_index=self.frame_index, timestamp_ms=captured_ms
        )
        self.frame_index += 1
        vis = result["visualization"]
        if vis.shape[0] != in_h:
            vis = cv2.resize(vis, (in_w, in_h), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return None
        return encoded.tobytes(), result["game_status"]

    def close(self) -> None:
        self.app.pipeline.close()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/analysis.js")
async def analysis_js() -> FileResponse:
    return FileResponse(STATIC_DIR / "analysis.js", media_type="text/javascript")


@app.get("/healthz")
async def healthz() -> PlainTextResponse:
    return PlainTextResponse("ok")


async def _send_error(ws: WebSocket, message: str) -> None:
    await ws.send_text(json.dumps({"type": "error", "message": message}))


@app.websocket("/ws")
async def play(ws: WebSocket, pose: str = "off") -> None:
    global _active_sessions
    await ws.accept()
    if pose != "off" and pose not in POSE_MODELS:
        await _send_error(ws, f"Unknown pose mode: {pose}")
        await ws.close()
        return
    if _active_sessions >= MAX_SESSIONS:
        await _send_error(ws, "Server busy, try again later.")
        await ws.close()
        return
    _active_sessions += 1
    try:
        session = await asyncio.to_thread(GameSession, pose)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        _active_sessions -= 1
        await _send_error(ws, f"Could not load models: {exc}")
        await ws.close()
        return
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                data = message["bytes"]
                if len(data) > MAX_FRAME_BYTES:
                    await _send_error(ws, "Frame too large.")
                    continue
                started = time.perf_counter()
                processed = await asyncio.to_thread(session.process, data)
                server_ms = (time.perf_counter() - started) * 1000
                if processed is None:
                    await _send_error(ws, "Could not decode frame.")
                    continue
                jpeg, status = processed
                summary = session.summary()
                if summary is not None:
                    final = summary["summary"]["completed"]
                    message = {"type": "summary", "final": final, "record": summary}
                    await ws.send_text(json.dumps(message, default=str))
                payload = {
                    "type": "status",
                    "status": status,
                    "server_ms": server_ms,
                    "timings": session.timings(),
                }
                await ws.send_text(json.dumps(payload, default=str))
                await ws.send_bytes(jpeg)
            elif message.get("text") is not None:
                try:
                    command = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if command.get("type") == "start":
                    name = str(command.get("difficulty", "easy")).upper()
                    difficulty = GameDifficulty.__members__.get(name, GameDifficulty.EASY)
                    try:
                        game = GameMode(command.get("game", GameMode.SIMPLE.value))
                    except ValueError:
                        game = GameMode.SIMPLE
                    await asyncio.to_thread(session.start, difficulty, game)
                elif command.get("type") == "reset":
                    session.reset()
                elif command.get("type") == "finger_rule":
                    session.set_finger_rule(bool(command.get("enabled")))
                elif command.get("type") == "next_pair":
                    session.next_pair()
    except WebSocketDisconnect:
        pass
    finally:
        _active_sessions -= 1
        session.close()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))


if __name__ == "__main__":
    main()
