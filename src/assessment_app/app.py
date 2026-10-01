import cv2
import numpy as np
import sys
from typing import Optional
from .config import CubeConfig
from .cube_detector import CubeDetector
from .game_logic import GameDifficulty
from .visualizer import Visualizer
import json
from pathlib import Path

from .contracts import DatasetSample, Frame
from .datasets import VideoDataset
from .model_backends import HSVCubeBackend
from .pipeline import VisionPipeline
from .tasks import CubeStackingTask


class AssessmentApp:
    """Main application for cube detection and stacking game"""

    def __init__(self, config: CubeConfig = None):
        self.config = config or CubeConfig()
        self.detector = CubeDetector(self.config)
        self.backend = HSVCubeBackend(self.detector)
        self.task = CubeStackingTask(self.config)
        self.pipeline = VisionPipeline(self.backend, self.task)
        # Keep these aliases for the existing debugging scripts and HUD code.
        self.analyzer = self.task.analyzer
        self.game_engine = self.task.game_engine

        self.cap = None
        self.is_running = False
        self.output_video = None
        self.frame_count = 0
        self._sequence_id = "assessment-session"

    def initialize_camera(self, camera_id: int = 0, width: int = 1280,
                         height: int = 720) -> bool:
        """Initialize camera capture"""
        backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(camera_id, backend)

        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return False

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        exposure = getattr(self.config, 'CAMERA_EXPOSURE', None)
        if exposure is not None:
            # Auto-exposure brightens toward dark desk surroundings, clipping the table and washing out cube colors.
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            print(f"Exposure locked at {exposure}")

        print(f"Camera initialized successfully")
        print(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x "
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}")

        return True

    def initialize_video_writer(self, output_path: str, fps: float = 30.0,
                                frame_size: Optional[tuple] = None):
        """Initialize video writer for recording results"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if frame_size is None:
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            frame_width, frame_height = frame_size

        self.output_video = cv2.VideoWriter(
            output_path, fourcc, fps, (frame_width, frame_height)
        )

        if not self.output_video.isOpened():
            print(f"Warning: Could not initialize video writer for {output_path}")
            self.output_video = None
        else:
            print(f"Video output initialized: {output_path}")

    def run_live_mode(self, difficulty: GameDifficulty = GameDifficulty.HARD,
                     record_output: bool = False):
        """
        Run the app in live camera mode

        Controls:
            'q' - Quit
            'r' - Reset game
            's' - Start game
            'p' - Pause/Resume
        """
        if not self.initialize_camera():
            return

        if record_output:
            self.initialize_video_writer("assessment_output.mp4")

        self.is_running = True
        paused = False

        print("\n=== Assessment App - Live Mode ===")
        print(f"Controls:")
        print(f"  's' - Start game")
        print(f"  'r' - Reset game")
        print(f"  'p' - Pause/Resume")
        print(f"  'q' - Quit")
        print(f"\nStarting live capture from camera...")

        self.task.difficulty = difficulty
        sample = DatasetSample(self._sequence_id, Path("camera:0"))
        self.pipeline.begin_sequence(sample)
        self.analyzer = self.task.analyzer

        while self.is_running:
            ret, frame = self.cap.read()

            if not ret:
                print("Error: Failed to read frame")
                break

            if not paused:
                # Process frame
                result = self._process_frame(frame)
                frame = result['visualization']

            # Display
            cv2.imshow('Assessment App - Cube Detection', frame)

            # Write to output video
            if self.output_video:
                self.output_video.write(frame)

            self.frame_count += 1

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
            elif key == ord('r'):
                self.game_engine.reset_game()
                print("\nGame reset")
            elif key == ord('s'):
                self.pipeline.begin_sequence(sample)
                self.analyzer = self.task.analyzer
                print(f"\nGame started - Difficulty: {difficulty.name}")
            elif key == ord('p'):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")

        self.cleanup()
        print(f"\nSession ended. Processed {self.frame_count} frames")

    def run_video_file_mode(self, video_path: str,
                           difficulty: GameDifficulty = GameDifficulty.HARD,
                           output_path: Optional[str] = None):
        """Process a local video through the same backend/task pipeline as camera mode."""
        self.is_running = True
        self.task.difficulty = difficulty

        print(f"\n=== Assessment App - Video Mode ===")
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
                    print(f"Frame {item.frame.frame_index}: "
                          f"Cubes={status['cubes_detected']}, "
                          f"Stacks={status['stacks_completed']}, "
                          f"Score={status['score']:.0f}")
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"Error: {exc}")
            self.cleanup()
            return

        self.cleanup()
        print(f"\nProcessing complete. Processed frames: {self.frame_count}")
        self._save_results()

    def _process_frame(self, frame: np.ndarray, frame_index: Optional[int] = None,
                       timestamp_ms: Optional[float] = None) -> dict:
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
        )

        return {
            'frame_id': index,
            'cubes': list(task_result.cubes),
            'stacks': [list(stack) for stack in task_result.stacks],
            'game_result': task_result.game_result,
            'game_status': task_result.game_status,
            'visualization': visualization
        }

    def _save_results(self):
        """Save game results to file"""
        status = self.game_engine.get_game_status()
        results = {
            'total_frames_processed': self.frame_count,
            'game_status': status,
            'events': self.game_engine.event_log
        }

        output_file = Path("game_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_file}")

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

    parser = argparse.ArgumentParser(description='Assessment App - Cube Detection Game')
    parser.add_argument('--mode', choices=['camera', 'video'], default='camera',
                       help='Run mode: camera for live, video for file processing')
    parser.add_argument('--video', type=str, help='Path to video file (for video mode)')
    parser.add_argument('--output', type=str, help='Output video file path')
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard', 'expert'],
                       default='hard', help='Game difficulty level')
    parser.add_argument('--record', action='store_true',
                       help='Record output video in camera mode')

    args = parser.parse_args()

    # Map difficulty
    difficulty_map = {
        'easy': GameDifficulty.EASY,
        'medium': GameDifficulty.MEDIUM,
        'hard': GameDifficulty.HARD,
        'expert': GameDifficulty.EXPERT
    }
    difficulty = difficulty_map[args.difficulty]

    # Create app
    app = AssessmentApp()

    # Run appropriate mode
    if args.mode == 'camera':
        app.run_live_mode(difficulty=difficulty, record_output=args.record)
    else:
        if not args.video:
            print("Error: --video argument required for video mode")
            sys.exit(1)
        app.run_video_file_mode(args.video, difficulty=difficulty,
                               output_path=args.output)


if __name__ == '__main__':
    main()
