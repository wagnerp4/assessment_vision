"""Built-in model backend adapters."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2

from .contracts import (
    CubeDetections,
    Frame,
    Keypoint,
    ModelObservation,
    PersonPose,
    PoseObservation,
)
from .cube_detector import CubeDetector


class HSVCubeBackend:
    """Adapt the existing OpenCV HSV detector/tracker to the model contract."""

    capabilities = frozenset({"cube_detection"})

    def __init__(self, detector: CubeDetector | None = None) -> None:
        self.detector = detector or CubeDetector()

    def load(self, device: str | None = None) -> None:
        # The OpenCV implementation has no separate weights or device context.
        del device

    def predict(self, frame: Frame) -> Sequence[ModelObservation]:
        cubes = self.detector.detect_cubes(frame.image, frame.frame_index)
        tracked = self.detector.track_cubes(cubes)
        return (
            CubeDetections(
                frame.sequence_id,
                frame.frame_index,
                frame.timestamp_ms,
                tuple(tracked),
            ),
        )

    def reset(self) -> None:
        self.detector.tracker.tracks.clear()
        self.detector.tracker.next_id = 0
        self.detector.last_masks.clear()

    def close(self) -> None:
        self.reset()


class MediaPipeHolisticBackend:
    """Optional single-person body and hand landmark backend.

    MediaPipe's Holistic solution tracks one person across successive frames and
    returns normalized body and hand landmarks. The backend converts x/y to
    source-frame pixel coordinates; unavailable point-level confidence remains
    ``None`` instead of being presented as a calibrated score.
    """

    capabilities = frozenset({"person_pose", "hand_landmarks"})

    BODY_LANDMARK_NAMES = (
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    )
    HAND_LANDMARK_NAMES = (
        "wrist",
        "thumb_cmc",
        "thumb_mcp",
        "thumb_ip",
        "thumb_tip",
        "index_finger_mcp",
        "index_finger_pip",
        "index_finger_dip",
        "index_finger_tip",
        "middle_finger_mcp",
        "middle_finger_pip",
        "middle_finger_dip",
        "middle_finger_tip",
        "ring_finger_mcp",
        "ring_finger_pip",
        "ring_finger_dip",
        "ring_finger_tip",
        "pinky_mcp",
        "pinky_pip",
        "pinky_dip",
        "pinky_tip",
    )

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        min_detection_confidence: float = 0.5,
        min_hand_confidence: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.min_detection_confidence = min_detection_confidence
        self.min_hand_confidence = min_hand_confidence
        self._mp = None
        self._holistic = None
        self._last_timestamp_ms = -1
        self._initial_reset_pending = False

    def load(self, device: str | None = None) -> None:
        if device not in (None, "cpu"):
            raise ValueError("MediaPipe Holistic currently supports CPU inference only.")
        if self.model_path is None:
            raise RuntimeError(
                "MediaPipe Holistic requires its model bundle. Download "
                "holistic_landmarker.task and pass --pose-model <path>. See the README pose setup."
            )
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Holistic model not found: {self.model_path}. Download it from "
                "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
                "holistic_landmarker/float16/1/holistic_landmarker.task"
            )
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "Person/hand landmarks require the optional pose extra. "
                "Install it with: uv sync --extra dev --extra pose "
                '(or pip install -e ".[pose]").'
            ) from exc

        if not hasattr(mp, "tasks") or not hasattr(mp.tasks.vision, "HolisticLandmarker"):
            raise RuntimeError(
                "The installed MediaPipe package does not include Holistic Landmarker Tasks; "
                "install this project's supported pose extra."
            )
        self._mp = mp
        self._create_tracker()
        self._initial_reset_pending = True

    def _create_tracker(self) -> None:
        vision = self._mp.tasks.vision
        options = vision.HolisticLandmarkerOptions(
            base_options=self._mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_hand_landmarks_confidence=self.min_hand_confidence,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
        )
        self._holistic = vision.HolisticLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    @staticmethod
    def _point_confidence(landmark) -> float | None:
        for attribute in ("visibility", "presence"):
            value = getattr(landmark, attribute, None)
            if value is not None and value > 0:
                return float(value)
        return None

    @classmethod
    def _keypoints(cls, landmarks, names, width: int, height: int) -> dict[str, Keypoint]:
        if not landmarks:
            return {}
        return {
            name: Keypoint(
                name=name,
                x=float(landmark.x) * width,
                y=float(landmark.y) * height,
                confidence=cls._point_confidence(landmark),
            )
            for name, landmark in zip(names, landmarks)
        }

    def predict(self, frame: Frame) -> Sequence[ModelObservation]:
        if self._holistic is None:
            raise RuntimeError("Call load before predicting pose landmarks.")
        height, width = frame.image.shape[:2]
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(round(frame.timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._holistic.detect_for_video(mp_image, timestamp_ms)
        body = self._keypoints(result.pose_landmarks, self.BODY_LANDMARK_NAMES, width, height)
        left_hand = self._keypoints(
            result.left_hand_landmarks, self.HAND_LANDMARK_NAMES, width, height
        )
        right_hand = self._keypoints(
            result.right_hand_landmarks, self.HAND_LANDMARK_NAMES, width, height
        )
        if not (body or left_hand or right_hand):
            return ()
        body_scores = [point.confidence for point in body.values() if point.confidence is not None]
        person = PersonPose(
            person_id="person-0",
            body_keypoints=body,
            left_hand_keypoints=left_hand,
            right_hand_keypoints=right_hand,
            confidence=sum(body_scores) / len(body_scores) if body_scores else 0.0,
        )
        return (
            PoseObservation(
                frame.sequence_id,
                frame.frame_index,
                frame.timestamp_ms,
                (person,),
            ),
        )

    def reset(self) -> None:
        if self._holistic is None:
            return
        if self._initial_reset_pending:
            # VisionPipeline resets immediately after the first load.
            self._initial_reset_pending = False
            return
        self._holistic.close()
        self._create_tracker()

    def close(self) -> None:
        if self._holistic is not None:
            self._holistic.close()
            self._holistic = None
        self._initial_reset_pending = False


class MediaPipeHandBackend:
    """Detect hands independently of body pose, including tabletop camera views.

    Both hands are detected so the playing side can be selected downstream.
    Handedness labels are model predictions, not persistent person identities.
    """

    capabilities = frozenset({"hand_landmarks"})
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )

    def __init__(self, model_path: Path | str = "models/hand_landmarker.task") -> None:
        self.model_path = Path(model_path).expanduser()
        self._mp = None
        self._tracker = None
        self._last_timestamp_ms = -1

    def load(self, device: str | None = None) -> None:
        if device not in (None, "cpu"):
            raise ValueError("MediaPipe Hand Landmarker currently supports CPU inference only.")
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Hand model not found: {self.model_path}. Download it from {self.MODEL_URL}"
            )
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "Hand landmarks require: uv sync --extra dev --extra pose"
            ) from exc
        self._mp = mp
        self._create_tracker()

    def _create_tracker(self) -> None:
        vision = self._mp.tasks.vision
        self._tracker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=self._mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._last_timestamp_ms = -1

    def predict(self, frame: Frame) -> Sequence[ModelObservation]:
        if self._tracker is None:
            raise RuntimeError("Call load before predicting hand landmarks.")
        height, width = frame.image.shape[:2]
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB),
        )
        timestamp_ms = max(round(frame.timestamp_ms), self._last_timestamp_ms + 1)
        result = self._tracker.detect_for_video(image, timestamp_ms)
        self._last_timestamp_ms = timestamp_ms
        hands = {}
        scores = {}
        for landmarks, categories in zip(result.hand_landmarks, result.handedness):
            if not categories:
                continue
            category = max(categories, key=lambda item: item.score)
            side = category.category_name.lower()
            if side not in {"left", "right"} or category.score <= scores.get(side, -1):
                continue
            hands[side] = MediaPipeHolisticBackend._keypoints(
                landmarks, MediaPipeHolisticBackend.HAND_LANDMARK_NAMES, width, height
            )
            scores[side] = category.score
        if not hands:
            return ()
        # One player is assumed by the app; no body/person detection is required.
        person = PersonPose(
            person_id=None,
            left_hand_keypoints=hands.get("left", {}),
            right_hand_keypoints=hands.get("right", {}),
        )
        return (PoseObservation(frame.sequence_id, frame.frame_index, frame.timestamp_ms, (person,)),)

    def reset(self) -> None:
        if self._tracker is not None and self._last_timestamp_ms >= 0:
            self._tracker.close()
            self._create_tracker()

    def close(self) -> None:
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None
        self._last_timestamp_ms = -1
