import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .analysis import format_summary, with_summary
from .camera_menu import CameraMenu
from .config import CubeConfig
from .contracts import DatasetSample, Frame, PoseObservation
from .cube_detector import CubeDetector
from .datasets import VideoDataset
from .fastener_model import create_fastener_backend
from .game_logic import GameDifficulty, GameMode, GameState
from .grasp import FINGERS
from .tasks import CubeStackingTask, ScrewingTask
from .hand_tracking import HandCubeEngagementTracker
from .model_backends import HSVCubeBackend, MediaPipeHandBackend, MediaPipeHolisticBackend
from .pipeline import VisionPipeline
from .recording import RealtimeRecorder, draw_rec_badge
from .visualizer import Visualizer

SUMMARY_WINDOW = "Session summary"
SUMMARY_HOLD_S = 5.0
CAMERA_WINDOW = "Assessment App - Cube Detection"


def _draw_menu_hint(frame: np.ndarray) -> None:
    """Tell the player how to leave the live game. Drawn on the window, after recording."""
    text = "Q: menu"
    (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x = frame.shape[1] - width - 16
    y = frame.shape[0] - 16
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _summary_with_caption(image: np.ndarray) -> np.ndarray:
    caption = "Press any key to return to the menu"
    bar = 36
    canvas = np.full((image.shape[0] + bar, image.shape[1], 3), 20, np.uint8)
    canvas[: image.shape[0]] = image
    cv2.putText(
        canvas, caption, (12, image.shape[0] + 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA,
    )
    return canvas


class AssessmentApp:
    """Main application for cube detection and stacking game"""

    def __init__(
        self,
        config: CubeConfig = None,
        *,
        enable_pose: bool = False,
        pose_model_path: str | Path | None = None,
        playing_hand: str = "right",
        pose_backend: str = "hands",
        game_mode: GameMode = GameMode.SIMPLE,
        rounds: int = 4,
        seed: int | None = None,
        finger_rule: bool = False,
    ):
        self.config = config or CubeConfig()
        self.detector = CubeDetector(self.config)
        self.backend = HSVCubeBackend(self.detector)
        backends = [self.backend]
        if pose_backend not in {"hands", "holistic"}:
            raise ValueError("pose_backend must be 'hands' or 'holistic'")
        self.pose_backend_name = pose_backend
        self._pose_models = {
            "hands": Path("models") / "hand_landmarker.task",
            "holistic": Path("models") / "holistic_landmarker.task",
        }
        if pose_model_path is not None:
            self._pose_models[pose_backend] = Path(pose_model_path)
        backend_type = MediaPipeHandBackend if pose_backend == "hands" else MediaPipeHolisticBackend
        self.pose_backend = (
            backend_type(model_path=self._pose_models[pose_backend]) if enable_pose else None
        )
        self.hand_tracker = HandCubeEngagementTracker(playing_hand)
        if self.pose_backend is not None:
            backends.append(self.pose_backend)
        self.task = CubeStackingTask(
            self.config,
            game_mode=game_mode,
            playing_hand=playing_hand,
            hand_backend=pose_backend if self.pose_backend is not None else None,
            finger_rule=finger_rule,
            rounds=rounds,
            seed=seed,
        )
        self._stacking_task = self.task
        self._screwing_task: ScrewingTask | None = None
        self.fastener_backend = create_fastener_backend()
        self._fasteners_enabled = False
        self.pipeline = VisionPipeline(backends, self.task)
        if game_mode == GameMode.SCREWING:
            # --game screwing outside the menu (e.g. video mode) needs the screwing task too.
            self._activate_task(game_mode)
        # Keep this alias for the existing debugging scripts and HUD code.
        self.analyzer = self.task.analyzer

        self.cap = None
        self.is_running = False
        self.output_video = None
        self.recorder: RealtimeRecorder | None = None
        self.frame_count = 0
        self._sequence_id = "assessment-session"
        self._saved_session: str | None = None
        self._summary_image: np.ndarray | None = None
        self.runs_dir = Path("runs")

    @property
    def game_engine(self):
        return self.task.game_engine

    def apply_session(
        self,
        *,
        game_mode: GameMode,
        difficulty: GameDifficulty,
        hands: bool,
        pose_backend: str,
        finger_rule: bool,
        pair_index: int,
    ) -> None:
        """Apply the camera menu to the next game. Pose models load on the first frame of that game."""
        self._activate_task(game_mode)
        self._configure_pose(hands, pose_backend)
        self.task.difficulty = difficulty
        if game_mode != GameMode.SCREWING:
            self.task.game_mode = game_mode
        self.task.pair_rule.index = pair_index % len(FINGERS)
        self.task.set_finger_rule(finger_rule and hands)
        self.analyzer = self.task.analyzer

    def _activate_task(self, game_mode: GameMode) -> None:
        if game_mode == GameMode.SCREWING:
            if self._screwing_task is None:
                self._screwing_task = ScrewingTask(
                    self.config,
                    playing_hand=self._stacking_task.playing_hand,
                    hand_backend=self.task.hand_backend,
                )
            self.task = self._screwing_task
            self._fasteners_enabled = True
        else:
            self.task = self._stacking_task
            self._fasteners_enabled = False
        self.pipeline.task = self.task
        self._sync_backends()

    def _sync_backends(self) -> None:
        backends = [self.backend]
        if self.pose_backend is not None:
            backends.append(self.pose_backend)
        if self._fasteners_enabled:
            backends.append(self.fastener_backend)
        self.pipeline.backends = tuple(backends)

    def _configure_pose(self, enabled: bool, backend_name: str) -> None:
        if backend_name not in {"hands", "holistic"}:
            raise ValueError("pose_backend must be 'hands' or 'holistic'")
        if enabled and self.pose_backend is not None and self.pose_backend_name == backend_name:
            self.task.hand_backend = backend_name
            self._sync_backends()
            return
        if self.pose_backend is not None:
            self.pose_backend.close()
            self.pose_backend = None
        self.pose_backend_name = backend_name
        self.task.hand_backend = None
        if not enabled:
            self._sync_backends()
            return
        backend_type = MediaPipeHandBackend if backend_name == "hands" else MediaPipeHolisticBackend
        pose = backend_type(model_path=self._pose_models[backend_name])
        try:
            if self.pipeline._loaded:
                pose.load()
        except Exception:
            pose.close()
            self._sync_backends()
            raise
        self.pose_backend = pose
        self.task.hand_backend = backend_name
        self._sync_backends()

    def initialize_camera(self, camera_id: int = 0, width: int = 1280, height: int = 720) -> bool:
        """Initialize camera capture"""
        backends = (
            (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
            if sys.platform == "win32"
            else (cv2.CAP_ANY,)
        )
        self.cap = None
        for backend in backends:
            capture = cv2.VideoCapture(camera_id, backend)
            if capture.isOpened():
                self.cap = capture
                break
            capture.release()

        if self.cap is None:
            print(f"Error: Could not open camera {camera_id}")
            kernel_release = Path("/proc/sys/kernel/osrelease")
            is_wsl = (
                sys.platform.startswith("linux")
                and kernel_release.exists()
                and "microsoft" in kernel_release.read_text().lower()
            )
            if is_wsl:
                devices = ", ".join(str(path) for path in sorted(Path("/dev").glob("video*")))
                if not devices:
                    print(
                        "No /dev/video* devices are visible in WSL. Attach a USB camera with "
                        "usbipd (WSL 2), then retry; see README.md, 'Camera access from WSL'."
                    )
                else:
                    print(
                        f"Visible camera devices: {devices}. Try --camera-id with the device index."
                    )
            return False

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        exposure = getattr(self.config, "CAMERA_EXPOSURE", None)
        if exposure is not None:
            # Auto-exposure brightens toward dark desk surroundings, clipping the table and washing out cube colors.
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            print(f"Exposure locked at {exposure}")

        if not any(self.cap.read()[0] for _ in range(10)):
            print(
                f"Error: camera {camera_id} opened but delivers no frames. It is probably in use by "
                "another app (a browser tab, OBS, Teams, ...). Close that app and retry, or pick "
                "another camera with --camera-id."
            )
            self.cap.release()
            return False

        print("Camera initialized successfully")
        print(
            f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x "
            f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        )
        print(f"FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}")

        return True

    def initialize_video_writer(
        self, output_path: str, fps: float = 30.0, frame_size: Optional[tuple] = None
    ):
        """Initialize video writer for recording results"""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if frame_size is None:
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            frame_width, frame_height = frame_size

        self.output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        if not self.output_video.isOpened():
            print(f"Warning: Could not initialize video writer for {output_path}")
            self.output_video = None
        else:
            print(f"Video output initialized: {output_path}")

    def run_live_mode(
        self,
        difficulty: GameDifficulty = GameDifficulty.HARD,
        record_output: bool = False,
        camera_id: int = 0,
    ):
        """
        Run the app in live camera mode.

        The window opens on a menu (game, difficulty, hand tracking, finger pairs, recording).
        Starting a game plays that session; when it finishes, the summary is shown and the menu returns.

        Menu:
            Click an option, or use the arrow keys
            's' or Enter - Start the selected game
            'q' - Quit
        During a game:
            's' - Restart this game
            'r' - Reset game
            'p' - Pause/Resume
            'g' - Toggle the finger-pair rule (when hand tracking is on)
            'f' - Next required finger pair (while the rule is on)
            'v' - Start/stop recording the annotated view to runs/
            'q' - Finish and return to the menu
        """
        if not self.initialize_camera(camera_id=camera_id):
            return

        menu = CameraMenu(
            game=self.task.game_mode,
            difficulty=difficulty,
            hands=self.pose_backend is not None,
            pose_backend=self.pose_backend_name,
            finger_rule=self.task.pair_rule.enabled,
            record=record_output,
            pair_index=self.task.pair_rule.index,
        )
        window = CAMERA_WINDOW
        cv2.namedWindow(window)
        clicks: list[tuple[int, int, int]] = []

        def on_mouse(event, x, y, _flags, _param):
            if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
                clicks.append((event, x, y))

        cv2.setMouseCallback(window, on_mouse)
        self.is_running = True
        sample = DatasetSample(self._sequence_id, Path(f"camera:{camera_id}"))

        print("\n=== Assessment App - Live Mode ===")
        print("Choose a game in the window, then start it.")
        print("  Click an option, or use the arrow keys")
        print("  's' or Enter - Start game")
        print("  'q' - Quit")
        print("During a game:")
        print("  's' - Restart    'r' - Reset    'p' - Pause")
        print("  'g' - Finger-pair rule    'f' - Next pair")
        print("  'v' - Record    'q' - Summary, then back to the menu")

        try:
            while self.is_running:
                action = self._run_camera_menu(menu, window, clicks)
                if action != "start":
                    break
                self._play_camera_game(menu, window, clicks, sample)
        finally:
            self._stop_recording()
            self.cleanup()
            print(f"\nSession ended. Processed {self.frame_count} frames")

    def _window_closed(self, window: str) -> bool:
        # 0 means the player closed the window. -1 means this backend does not report visibility.
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) == 0

    def _run_camera_menu(self, menu: CameraMenu, window: str, clicks: list[tuple[int, int, int]]) -> str:
        """Show the live menu until the player starts a game or quits."""
        clicks.clear()
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to read frame")
                self.is_running = False
                return "quit"
            cv2.imshow(window, menu.render(frame))
            if self._window_closed(window):
                self.is_running = False
                return "quit"
            pending = clicks[:]
            clicks.clear()
            for event, x, y in pending:
                direction = -1 if event == cv2.EVENT_RBUTTONDOWN else 1
                action = menu.handle_click(x, y, direction)
                if action == "start":
                    return "start"
            action = menu.handle_key(cv2.waitKeyEx(1))
            if action in {"start", "quit"}:
                return action
        return "quit"

    def _play_camera_game(
        self,
        menu: CameraMenu,
        window: str,
        clicks: list[tuple[int, int, int]],
        sample: DatasetSample,
    ) -> None:
        clicks.clear()
        if menu.game == GameMode.SCREWING and not menu.hands and self._pose_models["hands"].is_file():
            menu.hands = True
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to read frame")
            self.is_running = False
            return
        menu.message = "Loading models..."
        cv2.imshow(window, menu.render(frame))
        cv2.waitKey(1)
        try:
            self.apply_session(
                game_mode=menu.game,
                difficulty=menu.difficulty,
                hands=menu.hands,
                pose_backend=menu.pose_backend,
                finger_rule=menu.finger_rule,
                pair_index=menu.pair_index,
            )
            self.pipeline.begin_sequence(sample)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            menu.message = str(exc)
            print(f"Could not start game: {exc}")
            return
        menu.message = ""
        self.analyzer = self.task.analyzer
        if menu.record and self.recorder is None:
            self._start_recording()
        paused = False
        clock = time.perf_counter()
        print(f"\nGame started - {menu.game.value}, {menu.difficulty.name.title()}")

        while self.is_running:
            ret, frame = self.cap.read()
            captured_ms = (time.perf_counter() - clock) * 1000.0
            if not ret:
                print("Error: Failed to read frame")
                self.is_running = False
                break

            if not paused:
                result = self._process_frame(frame, timestamp_ms=captured_ms)
                frame = result["visualization"]

            if self.recorder:
                self.recorder.write(frame, captured_ms)
                frame = draw_rec_badge(frame, self.recorder.duration_s)
            _draw_menu_hint(frame)
            cv2.imshow(window, frame)
            self.frame_count += 1
            if self._window_closed(window):
                self.is_running = False
                break

            completed = not paused and self.game_engine.state == GameState.COMPLETED
            key = cv2.waitKeyEx(1)
            low = key & 0xFF
            if completed or low in (ord("q"), 27):
                self._finish_camera_game(menu, window, completed=completed or self._has_progress())
                return
            if low == ord("r"):
                self.game_engine.reset_game()
                print("\nGame reset")
            elif low == ord("s"):
                self.pipeline.begin_sequence(sample)
                self.analyzer = self.task.analyzer
                print(f"\nGame started - {menu.game.value}, {menu.difficulty.name.title()}")
            elif low == ord("p"):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif low == ord("g") and self.pose_backend is not None:
                rule = self.task.pair_rule
                self.task.set_finger_rule(not rule.enabled)
                print(f"\nFinger-pair rule {'on: ' + rule.label if rule.enabled else 'off'}")
            elif low == ord("f") and self.task.pair_rule.enabled:
                self.task.next_pair()
                print(f"\nRequired finger pair: {self.task.pair_rule.label}")
            elif low == ord("v"):
                if self.recorder:
                    self._stop_recording()
                else:
                    self._start_recording()

        if self.is_running is False and self._has_progress():
            self._finish_camera_game(menu, window, completed=True)

    def _finish_camera_game(self, menu: CameraMenu, window: str, *, completed: bool) -> None:
        """Save the session when it has progress, show the summary, then hand control back to the menu."""
        if menu.hands:
            menu.finger_rule = self.task.pair_rule.enabled
            menu.pair_index = self.task.pair_rule.index
        worth_showing = completed or self._has_progress()
        if worth_showing:
            self.save_run(show=False)
            if self.recorder and self._summary_image is not None:
                self.recorder.hold(_summary_with_caption(self._summary_image), SUMMARY_HOLD_S)
        self._stop_recording()
        if self.is_running and worth_showing:
            if self._summary_image is not None:
                cv2.imshow(SUMMARY_WINDOW, _summary_with_caption(self._summary_image))
                print("Press any key in the summary window to return to the menu.")
                cv2.waitKey(0)
                try:
                    cv2.destroyWindow(SUMMARY_WINDOW)
                except cv2.error:
                    pass
                cv2.waitKey(1)
            else:
                notice = np.zeros((160, 920, 3), np.uint8)
                cv2.putText(
                    notice, "Session saved. Press any key to return to the menu.",
                    (24, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA,
                )
                cv2.imshow(window, notice)
                print("Session saved. Press any key to return to the menu.")
                cv2.waitKey(0)
        if self.is_running:
            print("\nBack to the menu.")

    def run_video_file_mode(
        self,
        video_path: str,
        difficulty: GameDifficulty = GameDifficulty.HARD,
        output_path: Optional[str] = None,
    ):
        """Process a local video through the same backend/task pipeline as camera mode."""
        self.is_running = True
        self.task.difficulty = difficulty

        print("\n=== Assessment App - Video Mode ===")
        print(f"Processing: {video_path}")
        print(f"Difficulty: {difficulty.name}")

        dataset = VideoDataset.from_video(
            video_path,
            frame_stride=self.config.FRAME_SKIP_RATE,
            resize_height=self.config.FRAME_RESIZE_HEIGHT,
        )
        try:
            for item in self.pipeline.run_dataset(dataset):
                if not self.is_running:
                    break
                frame = item.frame.image
                task_result = item.task_output
                visualization = Visualizer.composite_visualization(
                    frame,
                    list(task_result.cubes),
                    [list(stack) for stack in task_result.stacks],
                    task_result.game_status,
                    task_result.game_result,
                    self.task.analyzer,
                    pose_observations=self.hand_tracker.filter(
                        tuple(
                            observation
                            for observation in item.observations
                            if isinstance(observation, PoseObservation)
                        ),
                        task_result.cubes,
                    ),
                )
                self.frame_count += 1

                if output_path and self.output_video is None:
                    fps = float(item.sample.metadata.get("fps", 30.0))
                    fps = fps / self.config.FRAME_SKIP_RATE
                    self.initialize_video_writer(
                        output_path, fps, (visualization.shape[1], visualization.shape[0])
                    )
                if self.output_video:
                    self.output_video.write(visualization)

                if item.frame.frame_index % 30 == 0:
                    status = task_result.game_status
                    print(
                        f"Frame {item.frame.frame_index}: "
                        f"Cubes={status['cubes_detected']}, "
                        f"Stacks={status['stacks_completed']}, "
                        f"Score={status['score']:.0f}"
                    )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"Error: {exc}")
            self.cleanup()
            return

        self.cleanup()
        print(f"\nProcessing complete. Processed frames: {self.frame_count}")
        if self._has_progress():
            self.save_run()

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_index: Optional[int] = None,
        timestamp_ms: Optional[float] = None,
    ) -> dict:
        """
        Process a single frame through the detection and game pipeline

        Returns dict with detection results and visualization
        """
        index = self.frame_count if frame_index is None else frame_index
        timestamp = timestamp_ms if timestamp_ms is not None else index * 1000.0 / 30.0
        pipeline_frame = Frame(self._sequence_id, index, timestamp, frame)
        processed = self.pipeline.process_frame(pipeline_frame)
        task_result = processed.task_output
        self.analyzer = self.task.analyzer
        visualization = Visualizer.composite_visualization(
            frame,
            list(task_result.cubes),
            [list(stack) for stack in task_result.stacks],
            task_result.game_status,
            task_result.game_result,
            self.analyzer,
            pose_observations=self.hand_tracker.filter(
                tuple(
                    observation
                    for observation in processed.observations
                    if isinstance(observation, PoseObservation)
                ),
                task_result.cubes,
            ),
        )

        return {
            "frame_id": index,
            "cubes": list(task_result.cubes),
            "stacks": [list(stack) for stack in task_result.stacks],
            "game_result": task_result.game_result,
            "game_status": task_result.game_status,
            "visualization": visualization,
        }

    def _start_recording(self) -> None:
        stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.recorder = RealtimeRecorder(self.runs_dir / f"{stamp}_recording.mp4")
        print(f"\nRecording to {self.recorder.path} (press 'v' to stop)")

    def _stop_recording(self) -> None:
        if self.recorder is None:
            return
        path, seconds = self.recorder.close(), self.recorder.duration_s
        self.recorder = None
        if path:
            print(f"\nRecording saved: {path} ({seconds:.0f} s)")

    def _has_progress(self) -> bool:
        progress = {"step_completed", "stack_confirmed", "round_completed"}
        return any(event["type"] in progress for event in self.game_engine.event_log)

    def save_run(self, show: bool = False) -> Path | None:
        """Write the session record (with summary) to runs/, render the summary figure, optionally show it."""
        if self._saved_session == self.task.session_id:
            return None
        self._saved_session = self.task.session_id
        self._summary_image = None
        record = with_summary(self.task.session_record("local"))
        self.runs_dir.mkdir(exist_ok=True)
        stamp = record["started_at"].replace(":", "-").replace("+00-00", "Z")
        base = self.runs_dir / f"{stamp}_{record['game']['mode']}_{record['game']['difficulty'].lower()}"
        base.with_suffix(".json").write_text(json.dumps(record, indent=1))
        print("\n=== Session summary ===\n" + format_summary(record["summary"]))
        print(f"Saved {base.with_suffix('.json')}")
        try:
            from .summary_plot import render_summary

            self._summary_image = render_summary(record, record["summary"])
        except ImportError:
            print("Install matplotlib (uv sync --extra analysis) for the summary window.")
            return base.with_suffix(".json")
        cv2.imwrite(str(base.with_suffix(".png")), self._summary_image)
        if show:
            cv2.imshow(SUMMARY_WINDOW, self._summary_image)
        return base.with_suffix(".json")

    def cleanup(self):
        """Clean up resources"""
        self.is_running = False

        if self.cap:
            self.cap.release()

        if self.output_video:
            self.output_video.release()
            self.output_video = None

        self.pipeline.close()

        cv2.destroyAllWindows()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Assessment App - Cube Detection Game")
    parser.add_argument(
        "--mode",
        choices=["camera", "video"],
        default="camera",
        help="Run mode: camera for live, video for file processing",
    )
    parser.add_argument("--video", type=str, help="Path to video file (for video mode)")
    parser.add_argument(
        "--camera-id", type=int, default=0, help="Camera index for live mode (default: 0)"
    )
    parser.add_argument(
        "--pose",
        action="store_true",
        help="Enable optional MediaPipe playing-hand landmark detection",
    )
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=None,
        help="Override the selected backend's model bundle in models/",
    )
    parser.add_argument(
        "--pose-backend", choices=("hands", "holistic"), default="hands",
        help="Landmark backend: hands for tabletop views (default), holistic for full-body views",
    )
    parser.add_argument(
        "--playing-hand",
        choices=("right", "left"),
        default="right",
        help="Hand to show and track near cubes when pose landmarks are enabled (default: right)",
    )
    parser.add_argument("--output", type=str, help="Output video file path")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "expert"],
        default="hard",
        help="Game difficulty level (random game: easy=2, medium=3, hard/expert=4 cubes)",
    )
    parser.add_argument(
        "--game",
        choices=[mode.value for mode in GameMode],
        default=GameMode.SIMPLE.value,
        help="simple: build one stack; random: stack a random order, unstack in reverse, repeat",
    )
    parser.add_argument(
        "--rounds", type=int, default=4, help="Rounds in the random game (default: 4)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for reproducible random-game orders"
    )
    parser.add_argument(
        "--finger-rule",
        action="store_true",
        help="Start with the finger-pair rule on (needs --pose; toggle live with 'g')",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Start the camera menu with recording on (recording begins when the game starts; toggle live with 'v')",
    )

    args = parser.parse_args()

    # Map difficulty
    difficulty_map = {
        "easy": GameDifficulty.EASY,
        "medium": GameDifficulty.MEDIUM,
        "hard": GameDifficulty.HARD,
        "expert": GameDifficulty.EXPERT,
    }
    difficulty = difficulty_map[args.difficulty]

    # Create app
    app = AssessmentApp(
        enable_pose=args.pose,
        pose_model_path=args.pose_model,
        playing_hand=args.playing_hand,
        pose_backend=args.pose_backend,
        game_mode=GameMode(args.game),
        rounds=args.rounds,
        seed=args.seed,
        finger_rule=args.finger_rule,
    )

    # Run appropriate mode
    if args.mode == "camera":
        app.run_live_mode(
            difficulty=difficulty, record_output=args.record, camera_id=args.camera_id
        )
    else:
        if not args.video:
            print("Error: --video argument required for video mode")
            sys.exit(1)
        app.run_video_file_mode(args.video, difficulty=difficulty, output_path=args.output)


if __name__ == "__main__":
    main()
