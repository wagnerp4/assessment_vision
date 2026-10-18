from typing import Dict, List, Sequence

import cv2
import numpy as np

from .contracts import PoseObservation
from .cube_detector import Cube
from .screwing import axis_depth
from .stacking_analyzer import StackingAnalyzer


class Visualizer:
    """Visualizes cube detection and game status on video frames"""

    # Color mappings for visualization (BGR)
    CUBE_COLORS_VIZ = {
        "blue": (255, 100, 0),
        "green": (0, 200, 0),
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
    }

    STACK_OUTLINE_COLOR = (0, 255, 0)  # Green for valid stacks
    INVALID_STACK_COLOR = (0, 0, 255)  # Red for invalid
    TEXT_COLOR = (255, 255, 255)  # White
    BODY_CONNECTIONS = (
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
    )
    HAND_CONNECTIONS = (
        ("wrist", "thumb_cmc"),
        ("thumb_cmc", "thumb_mcp"),
        ("thumb_mcp", "thumb_ip"),
        ("thumb_ip", "thumb_tip"),
        ("wrist", "index_finger_mcp"),
        ("index_finger_mcp", "index_finger_pip"),
        ("index_finger_pip", "index_finger_dip"),
        ("index_finger_dip", "index_finger_tip"),
        ("index_finger_mcp", "middle_finger_mcp"),
        ("middle_finger_mcp", "middle_finger_pip"),
        ("middle_finger_pip", "middle_finger_dip"),
        ("middle_finger_dip", "middle_finger_tip"),
        ("middle_finger_mcp", "ring_finger_mcp"),
        ("ring_finger_mcp", "ring_finger_pip"),
        ("ring_finger_pip", "ring_finger_dip"),
        ("ring_finger_dip", "ring_finger_tip"),
        ("ring_finger_mcp", "pinky_mcp"),
        ("pinky_mcp", "pinky_pip"),
        ("pinky_pip", "pinky_dip"),
        ("pinky_dip", "pinky_tip"),
        ("pinky_mcp", "wrist"),
    )

    @staticmethod
    def draw_cubes(
        frame: np.ndarray, cubes: List[Cube], analyzer: StackingAnalyzer = None
    ) -> np.ndarray:
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
            cv2.rectangle(
                output,
                (x, label_y - label_size[1] - 2),
                (x + label_size[0], label_y + 2),
                color,
                -1,
            )
            cv2.putText(output, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        return output

    @staticmethod
    def _draw_keypoint_graph(frame, keypoints, connections, color, *, threshold=0.25):
        def visible(point):
            return point is not None and (point.confidence is None or point.confidence >= threshold)

        output = frame
        for start_name, end_name in connections:
            start = keypoints.get(start_name)
            end = keypoints.get(end_name)
            if visible(start) and visible(end):
                cv2.line(
                    output,
                    (round(start.x), round(start.y)),
                    (round(end.x), round(end.y)),
                    color,
                    2,
                    cv2.LINE_AA,
                )
        for point in keypoints.values():
            if visible(point):
                cv2.circle(output, (round(point.x), round(point.y)), 3, color, -1, cv2.LINE_AA)
        return output

    @staticmethod
    def draw_pose_observations(
        frame: np.ndarray, observations: Sequence[PoseObservation]
    ) -> np.ndarray:
        """Draw upper-body pose and finger landmarks without face annotations."""
        output = frame.copy()
        body_color = (255, 180, 0)
        left_hand_color = (0, 220, 255)
        right_hand_color = (255, 100, 220)
        for observation in observations:
            for person in observation.people:
                output = Visualizer._draw_keypoint_graph(
                    output, person.body_keypoints, Visualizer.BODY_CONNECTIONS, body_color
                )
                for label, keypoints, color in (
                    ("L", person.left_hand_keypoints, left_hand_color),
                    ("R", person.right_hand_keypoints, right_hand_color),
                ):
                    if not keypoints:
                        continue
                    output = Visualizer._draw_keypoint_graph(
                        output, keypoints, Visualizer.HAND_CONNECTIONS, color
                    )
                    wrist = keypoints.get("wrist")
                    if wrist is not None:
                        cv2.putText(
                            output,
                            f"{person.person_id or 'person'} {label}",
                            (round(wrist.x) + 5, round(wrist.y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1,
                            cv2.LINE_AA,
                        )
        return output

    @staticmethod
    def draw_stacks(
        frame: np.ndarray, stacks: List[List[Cube]], analyzer: StackingAnalyzer
    ) -> np.ndarray:
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
                cv2.rectangle(
                    output,
                    (x, y_pos - label_size[1] - 2),
                    (x + label_size[0], y_pos + 2),
                    color,
                    -1,
                )
                cv2.putText(
                    output, label_text, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
                )

        return output

    @staticmethod
    def draw_game_status(frame: np.ndarray, game_status: Dict, result: Dict = None) -> np.ndarray:
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
        cv2.putText(
            output,
            state_text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            Visualizer.TEXT_COLOR,
            1,
        )

        # Score
        score_text = f"Score: {game_status['score']:.0f}"
        cv2.putText(
            output,
            score_text,
            (10, y_offset + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            Visualizer.TEXT_COLOR,
            1,
        )

        # Stacks completed
        stacks_text = f"Stacks: {game_status['stacks_completed']}"
        cv2.putText(
            output,
            stacks_text,
            (10, y_offset + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            Visualizer.TEXT_COLOR,
            1,
        )

        # Cubes detected
        cubes_text = f"Cubes: {game_status['cubes_detected']}"
        cv2.putText(
            output,
            cubes_text,
            (w // 2, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            Visualizer.TEXT_COLOR,
            1,
        )

        # Time
        minutes, seconds = divmod(int(game_status.get("elapsed_ms", 0) // 1000), 60)
        time_text = f"Time: {minutes}:{seconds:02d}"
        cv2.putText(
            output,
            time_text,
            (w // 2, y_offset + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            Visualizer.TEXT_COLOR,
            1,
        )

        # Frame number
        if result:
            frame_text = f"Frame: {result['frame_id']}"
            cv2.putText(
                output,
                frame_text,
                (w // 2, y_offset + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_size,
                Visualizer.TEXT_COLOR,
                1,
            )

        goal = game_status.get("goal")
        if goal:
            cv2.putText(
                output, goal, (10, y_offset + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            progress = game_status.get("hold_progress", 0.0)
            x0, y0, bw = w // 2, y_offset + 65, 300
            cv2.rectangle(output, (x0, y0), (x0 + bw, y0 + 18), (200, 200, 200), 1)
            cv2.rectangle(output, (x0, y0), (x0 + int(bw * progress), y0 + 18), (0, 255, 0), -1)
            cv2.putText(
                output,
                "hold",
                (x0 + bw + 10, y0 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                Visualizer.TEXT_COLOR,
                1,
            )
        if game_status["state"] == "completed":
            cv2.putText(
                output,
                "LEVEL COMPLETE",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3,
            )

        return output

    @staticmethod
    def _label(frame: np.ndarray, text: str, origin, color, scale: float = 0.6) -> None:
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x, y = origin
        cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + base + 2), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

    @staticmethod
    def draw_assessment(frame: np.ndarray, game_status: Dict) -> np.ndarray:
        """Required finger pair, detected grip, feedback, and the random game's target tower."""
        output = frame.copy()
        h, w = output.shape[:2]
        y = 150
        if game_status.get("required_pair"):
            Visualizer._label(
                output, f"Use: {game_status['required_pair']}", (10, y), (0, 255, 255)
            )
            y += 30
            grip = game_status.get("grip") or "-"
            Visualizer._label(output, f"Grip: {grip}", (10, y), (255, 255, 255), 0.55)
            y += 30
        if game_status.get("feedback"):
            Visualizer._label(output, game_status["feedback"], (10, y), (0, 165, 255))

        target = game_status.get("target") or []
        if not target:
            return output
        size, gap = 36, 6
        x0 = w - size - 24
        bottom = 140 + len(target) * (size + gap)
        Visualizer._label(output, "Target", (x0 - 14, 140), (255, 255, 255), 0.5)
        height = game_status.get("height", 0)
        stacking = game_status.get("phase") == "stack"
        for i, color in enumerate(target):
            top = bottom - (i + 1) * (size + gap) + gap
            fill = Visualizer.CUBE_COLORS_VIZ.get(color, (128, 128, 128))
            present = i < height
            if not present and not stacking:
                fill = tuple(c // 3 for c in fill)
            cv2.rectangle(output, (x0, top), (x0 + size, top + size), fill, -1)
            if present and not stacking and i == height - 1:
                border = (0, 165, 255)
            elif present:
                border = (255, 255, 255)
            elif stacking and i == height:
                border = (0, 255, 255)
            else:
                border = (90, 90, 90)
            cv2.rectangle(output, (x0, top), (x0 + size, top + size), border, 3)
        return output

    @staticmethod
    def draw_debug_info(frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw debug information on frame"""
        output = frame.copy()
        h, w = output.shape[:2]

        y_offset = h - len(result["new_achievements"]) * 25 - 20

        for achievement in result["new_achievements"]:
            cv2.putText(
                output,
                f"+ {achievement}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            y_offset += 25

        return output

    @staticmethod
    def composite_visualization(
        frame: np.ndarray,
        cubes: List[Cube],
        stacks: List[List[Cube]],
        game_status: Dict,
        game_result: Dict,
        analyzer: StackingAnalyzer,
        pose_observations: Sequence[PoseObservation] = (),
    ) -> np.ndarray:
        """Create complete visualization with all information"""
        output = Visualizer.draw_cubes(frame, cubes)
        if pose_observations:
            output = Visualizer.draw_pose_observations(output, pose_observations)
        output = Visualizer.draw_stacks(output, stacks, analyzer)
        if game_status.get("mode") == "screwing":
            output = Visualizer.draw_screwing(output, game_status)
        else:
            output = Visualizer.draw_game_status(output, game_status, game_result)
            output = Visualizer.draw_assessment(output, game_status)
        output = Visualizer.draw_debug_info(output, game_result)
        return output

    @staticmethod
    def draw_screwing(frame: np.ndarray, game_status: Dict) -> np.ndarray:
        """Screw axis, nut outline, depth against the difficulty mark, phase timer, and hint."""
        output = frame.copy()
        height, width = output.shape[:2]
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (width, 118), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, output, 0.65, 0, output)
        phase = str(game_status.get("phase") or "")
        depth = game_status.get("depth")
        target = float(game_status.get("target_depth") or 0)
        depth_text = "—" if depth is None else f"{int(round(depth * 100))}%"
        step = game_status.get("step")
        steps = game_status.get("steps")
        stage = f"{step}/{steps} {phase}" if step and steps else phase
        cv2.putText(output, f"Screwing  {stage}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            output,
            f"Thread {depth_text} / {int(round(target * 100))}%    turns {game_status.get('turns', 0):.1f}    sides {game_status.get('side_steps', 0)}    spins {game_status.get('spins', 0)}",
            (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        phase_s = (game_status.get("phase_ms") or 0) / 1000
        cv2.putText(
            output, f"Timer {phase_s:.1f}s", (10, 88),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
        )
        if game_status.get("feedback"):
            Visualizer._label(output, game_status["feedback"], (10, 148), (0, 165, 255))
        tip, head = game_status.get("screw_tip"), game_status.get("screw_head")
        if tip and head:
            p1 = (int(tip[0]), int(tip[1]))
            p2 = (int(head[0]), int(head[1]))
            cv2.line(output, p1, p2, (180, 180, 180), 2, cv2.LINE_AA)
            cv2.circle(output, p1, 5, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.circle(output, p2, 6, (255, 180, 0), -1, cv2.LINE_AA)
            # Depth and target are shares of the thread; place them on the tip-to-head axis.
            if depth is not None:
                along = axis_depth(depth)
                mark = (int(p1[0] + (p2[0] - p1[0]) * along), int(p1[1] + (p2[1] - p1[1]) * along))
                cv2.circle(output, mark, 7, (0, 255, 0), 2, cv2.LINE_AA)
            along = axis_depth(target)
            goal = (int(p1[0] + (p2[0] - p1[0]) * along), int(p1[1] + (p2[1] - p1[1]) * along))
            cv2.drawMarker(output, goal, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
        polygon = game_status.get("nut_polygon") or []
        if len(polygon) >= 3:
            points = np.array(polygon, np.int32).reshape(-1, 1, 2)
            cv2.polylines(output, [points], True, (0, 255, 180), 2, cv2.LINE_AA)
        for edge in game_status.get("nut_edges") or []:
            if len(edge) != 2:
                continue
            start = (int(edge[0][0]), int(edge[0][1]))
            end = (int(edge[1][0]), int(edge[1][1]))
            cv2.line(output, start, end, (255, 0, 255), 2, cv2.LINE_AA)
        bar_x, bar_y, bar_h, bar_w = width - 36, 140, min(220, height - 160), 16
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
        if target:
            goal_y = bar_y + bar_h - int(bar_h * min(target, 1))
            cv2.line(output, (bar_x - 4, goal_y), (bar_x + bar_w + 4, goal_y), (0, 255, 255), 2)
        if depth is not None:
            filled = int(bar_h * max(0.0, min(depth, 1.0)))
            cv2.rectangle(output, (bar_x, bar_y + bar_h - filled), (bar_x + bar_w, bar_y + bar_h), (0, 200, 0), -1)
        if game_status.get("state") == "completed":
            cv2.putText(output, "LEVEL COMPLETE", (10, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        return output
