from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import CubeConfig
from .cube_detector import Cube


@dataclass
class AlignmentResult:
    """Result of alignment check between two cubes"""

    is_aligned: bool
    alignment_type: str  # 'perfect', 'good', 'poor', 'none'
    horizontal_distance: int  # pixels between edges
    vertical_distance: int  # pixels between edges
    confidence: float  # 0-1


@dataclass
class StackingEvent:
    """Represents a stacking event"""

    cube_ids: List[int]
    cube_colors: List[str]
    timestamp_frame: int
    stable_frames: int = 0
    is_confirmed: bool = False


class StackingAnalyzer:
    """Analyzes cube alignment and stacking patterns"""

    def __init__(self, config: CubeConfig = None):
        self.config = config or CubeConfig()
        self.stacking_history: Dict[str, StackingEvent] = {}  # seq_key -> event
        self.alignment_cache: Dict[Tuple[int, int], AlignmentResult] = {}

    def analyze_alignment(self, lower: Cube, upper: Cube) -> AlignmentResult:
        """How well `upper` rests on `lower` (image y grows downward)."""
        x1, y1, w1, h1 = lower.bbox
        x2, y2, w2, h2 = upper.bbox
        overlap_w = min(x1 + w1, x2 + w2) - max(x1, x2)
        # Positive = visible gap between the cubes; negative = the upper cube hides part of the lower one.
        vertical_gap = y1 - (y2 + h2)
        if overlap_w <= 0:
            return AlignmentResult(False, "none", int(overlap_w), int(vertical_gap), 0.0)

        overlap = overlap_w / max(1, min(w1, w2))
        ref_h = max(1, max(h1, h2))  # the lower cube may be partly hidden by the upper one
        touching = (
            -0.45 * ref_h
            <= vertical_gap
            <= max(self.config.VERTICAL_DISTANCE_THRESHOLD, 0.25 * ref_h)
        )
        if not touching or overlap < 0.5:
            return AlignmentResult(
                False,
                "poor" if touching else "none",
                int(overlap_w),
                int(vertical_gap),
                0.3 if touching else 0.0,
            )
        if overlap >= self.config.VERTICAL_OVERLAP_THRESHOLD:
            return AlignmentResult(True, "perfect", int(overlap_w), int(vertical_gap), 0.95)
        return AlignmentResult(True, "good", int(overlap_w), int(vertical_gap), 0.6)

    def find_stacks(self, cubes: List[Cube]) -> List[List[Cube]]:
        """Return stacks of 2+ cubes, each ordered bottom -> top."""
        if len(cubes) < 2:
            return []
        # For every cube pick the best-aligned support directly beneath it.
        support: Dict[int, int] = {}
        for i, up in enumerate(cubes):
            best, best_conf = None, 0.0
            for j, low in enumerate(cubes):
                if i == j or low.center[1] <= up.center[1]:
                    continue
                al = self.analyze_alignment(low, up)
                if al.is_aligned and al.confidence > best_conf:
                    best, best_conf = j, al.confidence
            if best is not None:
                support[i] = best
        # A support can hold only one cube: keep the upper cube closest to it horizontally.
        holder: Dict[int, int] = {}
        for up, low in support.items():
            cur = holder.get(low)
            dx = abs(cubes[up].center[0] - cubes[low].center[0])
            if cur is None or dx < abs(cubes[cur].center[0] - cubes[low].center[0]):
                holder[low] = up
        supported = set(holder.values())
        stacks = []
        for i in range(len(cubes)):
            if i in supported or i not in holder:
                continue
            chain, cur = [cubes[i]], i
            while cur in holder and len(chain) <= len(cubes):
                cur = holder[cur]
                chain.append(cubes[cur])
            stacks.append(chain)
        return stacks

    def validate_stack_sequence(self, stack: List[Cube]) -> Tuple[bool, str]:
        """
        Validate if stack matches a valid game sequence
        Stack is ordered bottom to top

        Returns (is_valid, sequence_string)
        """
        sequence = tuple(cube.color for cube in stack)
        sequence_str = " -> ".join(sequence)

        is_valid = sequence in self.config.VALID_STACK_SEQUENCES

        return is_valid, sequence_str

    def track_stacking_events(self, stacks: List[List[Cube]], frame_id: int) -> List[StackingEvent]:
        """
        Track stacking events over time and confirm stability
        Returns newly confirmed stacking events
        """
        confirmed_events = []

        # Create sequence keys for current stacks
        current_sequences = {}
        for stack in stacks:
            seq_key = "_".join(str(c.cube_id) for c in stack)
            is_valid, seq_str = self.validate_stack_sequence(stack)
            current_sequences[seq_key] = (stack, is_valid, seq_str)

        # Update or create stacking events
        for seq_key, (stack, is_valid, seq_str) in current_sequences.items():
            if seq_key not in self.stacking_history:
                self.stacking_history[seq_key] = StackingEvent(
                    cube_ids=[c.cube_id for c in stack],
                    cube_colors=[c.color for c in stack],
                    timestamp_frame=frame_id,
                )

            event = self.stacking_history[seq_key]
            event.stable_frames += 1

            # Confirm if stable long enough
            if event.stable_frames >= self.config.STACK_CONFIRMATION_FRAMES:
                if not event.is_confirmed:
                    event.is_confirmed = True
                    if is_valid:
                        confirmed_events.append(event)

        # Remove old stacking events
        to_remove = [
            key for key, event in self.stacking_history.items() if key not in current_sequences
        ]
        for key in to_remove:
            del self.stacking_history[key]

        return confirmed_events

    def get_cube_stack_info(self, cube: Cube, stacks: List[List[Cube]]) -> Optional[Dict]:
        """Get information about which stack a cube belongs to"""
        for stack in stacks:
            if cube.cube_id in [c.cube_id for c in stack]:
                position = next(i for i, c in enumerate(stack) if c.cube_id == cube.cube_id)
                is_valid, seq_str = self.validate_stack_sequence(stack)
                return {
                    "stack_position": position,
                    "stack_height": len(stack),
                    "is_valid_sequence": is_valid,
                    "sequence": seq_str,
                }
        return None
