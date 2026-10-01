import cv2
import numpy as np
from typing import List, Dict, Tuple
from .cube_detector import Cube
from .stacking_analyzer import StackingAnalyzer

class Visualizer:
    """Visualizes cube detection and game status on video frames"""

    # Color mappings for visualization (BGR)
    CUBE_COLORS_VIZ = {
        'blue': (255, 100, 0),
        'green': (0, 200, 0),
        'red': (0, 0, 255),
        'yellow': (0, 255, 255),
    }

    STACK_OUTLINE_COLOR = (0, 255, 0)   # Green for valid stacks
    INVALID_STACK_COLOR = (0, 0, 255)   # Red for invalid
    TEXT_COLOR = (255, 255, 255)        # White

    @staticmethod
    def draw_cubes(frame: np.ndarray, cubes: List[Cube],
                   analyzer: StackingAnalyzer = None) -> np.ndarray:
        """Draw detected cubes on frame"""
        output = frame.copy()

        # Draw each cube
        for cube in cubes:
            x, y, w, h = cube.bbox
            color = Visualizer.CUBE_COLORS_VIZ.get(cube.color, (200, 200, 200))

            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

            # Draw center point
            cv2.circle(output, cube.center, 3, color, -1)

            # Draw label
            label = f"{cube.color[:3].upper()} (ID:{cube.cube_id})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            label_y = max(y - 5, label_size[1] + 5)
            cv2.rectangle(output, (x, label_y - label_size[1] - 2),
                         (x + label_size[0], label_y + 2), color, -1)
            cv2.putText(output, label, (x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        return output

    @staticmethod
    def draw_stacks(frame: np.ndarray, stacks: List[List[Cube]],
                   analyzer: StackingAnalyzer) -> np.ndarray:
        """Draw stacking information on frame"""
        output = frame.copy()

        for stack_idx, stack in enumerate(stacks):
            is_valid, seq_str = analyzer.validate_stack_sequence(stack)
            color = Visualizer.STACK_OUTLINE_COLOR if is_valid else Visualizer.INVALID_STACK_COLOR

            # Draw connections between stacked cubes
            for i in range(len(stack) - 1):
                cube1 = stack[i]
                cube2 = stack[i + 1]

                pt1 = (cube1.center[0], cube1.bbox[1])
                pt2 = (cube2.center[0], cube2.bbox[1] + cube2.bbox[3])

                cv2.line(output, pt1, pt2, color, 2)
                cv2.circle(output, pt1, 5, color, -1)

            # Draw stack label
            if stack:
                x, y = stack[0].bbox[0], stack[0].bbox[1]
                label = f"Stack {stack_idx}: {seq_str}"
                status = "✓" if is_valid else "✗"

                label_text = f"{status} {label}"
                label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                y_pos = max(y - 25, label_size[1] + 5)
                cv2.rectangle(output, (x, y_pos - label_size[1] - 2),
                             (x + label_size[0], y_pos + 2), color, -1)
                cv2.putText(output, label_text, (x, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return output

    @staticmethod
    def draw_game_status(frame: np.ndarray, game_status: Dict,
                        result: Dict = None) -> np.ndarray:
        """Draw game status information on frame"""
        output = frame.copy()
        h, w = output.shape[:2]

        # Draw background panel for status
        panel_height = 120
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)

        # Draw status text
        y_offset = 25
        text_size = 0.6

        # State
        state_text = f"State: {game_status['state'].upper()}"
        cv2.putText(output, state_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, text_size, Visualizer.TEXT_COLOR, 1)

        # Score
        score_text = f"Score: {game_status['score']:.0f}"
        cv2.putText(output, score_text, (10, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, text_size, Visualizer.TEXT_COLOR, 1)

        # Stacks completed
        stacks_text = f"Stacks: {game_status['stacks_completed']}"
        cv2.putText(output, stacks_text, (10, y_offset + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, text_size, Visualizer.TEXT_COLOR, 1)

        # Cubes detected
        cubes_text = f"Cubes: {game_status['cubes_detected']}"
        cv2.putText(output, cubes_text, (w//2, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, text_size, Visualizer.TEXT_COLOR, 1)

        # Time
        time_text = f"Time: {game_status['time_elapsed_frames']//30}s"
        cv2.putText(output, time_text, (w//2, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, text_size, Visualizer.TEXT_COLOR, 1)

        # Frame number
        if result:
            frame_text = f"Frame: {result['frame_id']}"
            cv2.putText(output, frame_text, (w//2, y_offset + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, text_size, Visualizer.TEXT_COLOR, 1)

        goal = game_status.get('goal')
        if goal:
            cv2.putText(output, goal, (10, y_offset + 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            progress = game_status.get('hold_progress', 0.0)
            x0, y0, bw = w // 2, y_offset + 65, 300
            cv2.rectangle(output, (x0, y0), (x0 + bw, y0 + 18), (200, 200, 200), 1)
            cv2.rectangle(output, (x0, y0), (x0 + int(bw * progress), y0 + 18), (0, 255, 0), -1)
            cv2.putText(output, "hold", (x0 + bw + 10, y0 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, Visualizer.TEXT_COLOR, 1)
        if game_status['state'] == 'completed':
            cv2.putText(output, "LEVEL COMPLETE - press 's' to play again", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        return output

    @staticmethod
    def draw_debug_info(frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw debug information on frame"""
        output = frame.copy()
        h, w = output.shape[:2]

        y_offset = h - len(result['new_achievements']) * 25 - 20

        for achievement in result['new_achievements']:
            cv2.putText(output, f"+ {achievement}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 25

        return output

    @staticmethod
    def composite_visualization(frame: np.ndarray, cubes: List[Cube],
                               stacks: List[List[Cube]], game_status: Dict,
                               game_result: Dict, analyzer: StackingAnalyzer) -> np.ndarray:
        """Create complete visualization with all information"""
        output = Visualizer.draw_cubes(frame, cubes)
        output = Visualizer.draw_stacks(output, stacks, analyzer)
        output = Visualizer.draw_game_status(output, game_status, game_result)
        output = Visualizer.draw_debug_info(output, game_result)
        return output
