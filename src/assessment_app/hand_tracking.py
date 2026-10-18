"""Select and gate the playing hand using its proximity/motion relative to cubes."""

from dataclasses import replace
from math import hypot
from typing import Sequence

from .contracts import PersonPose, PoseObservation
from .cube_detector import Cube


class HandCubeEngagementTracker:
    """Expose only the configured hand while it approaches or reaches a cube."""

    def __init__(self, hand: str = "right"):
        if hand not in {"left", "right"}:
            raise ValueError("hand must be 'left' or 'right'")
        self.hand = hand
        self._sequence_id: str | None = None
        self._previous_distance: float | None = None

    @staticmethod
    def _distance_to_cubes(person: PersonPose, cubes: Sequence[Cube], hand: str) -> float | None:
        keypoints = getattr(person, f"{hand}_hand_keypoints")
        if not keypoints or not cubes:
            return None

        distances = []
        for point in keypoints.values():
            for cube in cubes:
                x, y, width, height = cube.bbox
                dx = max(x - point.x, 0.0, point.x - (x + width))
                dy = max(y - point.y, 0.0, point.y - (y + height))
                distances.append(hypot(dx, dy))
        return min(distances)

    def filter(
        self, observations: Sequence[PoseObservation], cubes: Sequence[Cube]
    ) -> tuple[PoseObservation, ...]:
        """Show only the playing hand near/moving toward a cube."""
        if not observations:
            self._previous_distance = None
        filtered = []
        for observation in observations:
            if observation.sequence_id != self._sequence_id:
                self._sequence_id = observation.sequence_id
                self._previous_distance = None

            people = []
            closest_distance = None
            for person in observation.people:
                distance = self._distance_to_cubes(person, cubes, self.hand)
                if distance is not None:
                    closest_distance = distance if closest_distance is None else min(
                        closest_distance, distance
                    )
                hand_keypoints = getattr(person, f"{self.hand}_hand_keypoints")
                if distance is None or not hand_keypoints:
                    engaged = False
                else:
                    cube = min(
                        cubes,
                        key=lambda item: min(
                            hypot(
                                max(item.bbox[0] - point.x, 0.0, point.x - (item.bbox[0] + item.bbox[2])),
                                max(item.bbox[1] - point.y, 0.0, point.y - (item.bbox[1] + item.bbox[3])),
                            )
                            for point in hand_keypoints.values()
                        ),
                    )
                    reach_distance = max(28.0, 0.75 * hypot(cube.bbox[2], cube.bbox[3]))
                    approaching = (
                        self._previous_distance is not None
                        and self._previous_distance - distance >= 2.0
                    )
                    engaged = distance <= reach_distance or approaching

                people.append(
                    replace(
                        person,
                        body_keypoints={},
                        left_hand_keypoints=(
                            hand_keypoints if engaged and self.hand == "left" else {}
                        ),
                        right_hand_keypoints=(
                            hand_keypoints if engaged and self.hand == "right" else {}
                        ),
                    )
                )

            self._previous_distance = closest_distance
            filtered.append(replace(observation, people=tuple(people)))
        return tuple(filtered)

    def reset(self) -> None:
        self._sequence_id = None
        self._previous_distance = None
