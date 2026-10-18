from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from assessment_app.app import AssessmentApp
from assessment_app.contracts import Frame, Keypoint, PersonPose, PoseObservation
from assessment_app.cube_detector import Cube
from assessment_app.hand_tracking import HandCubeEngagementTracker
from assessment_app.model_backends import MediaPipeHandBackend, MediaPipeHolisticBackend
from assessment_app.visualizer import Visualizer


def _landmarks(count, *, visibility=None):
    return [
        SimpleNamespace(x=0.25, y=0.5, visibility=visibility, presence=None) for _ in range(count)
    ]


class FakeHolistic:
    def __init__(self, result):
        self.result = result
        self.image = None
        self.timestamp_ms = None

    def detect_for_video(self, image, timestamp_ms):
        self.image = image
        self.timestamp_ms = timestamp_ms
        return self.result


def test_holistic_backend_maps_body_and_hands_to_source_pixels():
    result = SimpleNamespace(
        pose_landmarks=_landmarks(33, visibility=0.8),
        left_hand_landmarks=_landmarks(21),
        right_hand_landmarks=None,
    )
    holistic = FakeHolistic(result)
    backend = MediaPipeHolisticBackend()
    backend._holistic = holistic
    backend._mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB="srgb"),
        Image=lambda image_format, data: data,
    )
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[0, 0] = (10, 20, 30)

    frame = Frame("seq-a", 4, 133.3, image)
    (observation,) = backend.predict(frame)

    assert observation.sequence_id == "seq-a"
    assert observation.frame_index == 4
    assert observation.timestamp_ms == 133.3
    (person,) = observation.people
    assert person.person_id == "person-0"
    assert len(person.body_keypoints) == 33
    assert len(person.left_hand_keypoints) == 21
    assert person.right_hand_keypoints == {}
    assert person.body_keypoints["left_shoulder"].x == 25.0
    assert person.body_keypoints["left_shoulder"].y == 40.0
    assert person.body_keypoints["left_shoulder"].confidence == 0.8
    assert person.left_hand_keypoints["index_finger_tip"].confidence is None
    assert tuple(holistic.image[0, 0]) == (30, 20, 10)
    backend.predict(frame)
    assert holistic.timestamp_ms == 134


def test_holistic_backend_returns_no_people_when_pose_is_missing():
    backend = MediaPipeHolisticBackend()
    backend._mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB="srgb"),
        Image=lambda image_format, data: data,
    )
    backend._holistic = FakeHolistic(
        SimpleNamespace(pose_landmarks=None, left_hand_landmarks=None, right_hand_landmarks=None)
    )
    frame = Frame("seq-a", 0, 0.0, np.zeros((8, 8, 3), dtype=np.uint8))

    assert backend.predict(frame) == ()


def test_pose_backend_import_does_not_require_optional_mediapipe():
    backend = MediaPipeHolisticBackend()

    assert backend.capabilities == frozenset({"person_pose", "hand_landmarks"})
    assert backend._holistic is None


def test_assessment_app_can_enable_pose_backend_without_loading_it():
    app = AssessmentApp(enable_pose=True, pose_model_path="missing-on-purpose.task")

    assert len(app.pipeline.backends) == 2
    assert isinstance(app.pipeline.backends[1], MediaPipeHandBackend)
    app.pipeline.close()


def test_camera_initialization_falls_back_from_directshow_on_windows(monkeypatch):
    opened_backends = []

    class FakeCapture:
        def __init__(self, camera_id, backend):
            assert camera_id == 0
            opened_backends.append(backend)
            self.opened = backend == cv2.CAP_MSMF

        def isOpened(self):
            return self.opened

        def read(self):
            return True, None

        def release(self):
            pass

        def set(self, *_args):
            return True

        def get(self, property_id):
            if property_id == cv2.CAP_PROP_FRAME_WIDTH:
                return 1280
            if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return 720
            return 30

    monkeypatch.setattr("assessment_app.app.sys.platform", "win32")
    monkeypatch.setattr("assessment_app.app.cv2.VideoCapture", FakeCapture)
    app = AssessmentApp()

    assert app.initialize_camera()
    assert opened_backends == [cv2.CAP_DSHOW, cv2.CAP_MSMF]


def test_pose_backend_reports_missing_model_bundle(tmp_path):
    backend = MediaPipeHolisticBackend(tmp_path / "holistic_landmarker.task")

    with pytest.raises(FileNotFoundError, match="Download it from"):
        backend.load()


def test_hand_backend_detects_both_sides_without_body_and_survives_reset(monkeypatch):
    result = SimpleNamespace(
        hand_landmarks=[_landmarks(21), _landmarks(21)],
        handedness=[
            [SimpleNamespace(category_name="Left", score=0.95)],
            [SimpleNamespace(category_name="Right", score=0.99)],
        ],
    )
    tracker = FakeHolistic(result)
    tracker.close = lambda: None
    backend = MediaPipeHandBackend()
    backend._tracker = tracker
    backend._mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB="srgb"),
        Image=lambda image_format, data: data,
    )
    frame = Frame("seq", 0, 0, np.zeros((100, 200, 3), dtype=np.uint8))
    (observation,) = backend.predict(frame)
    person = observation.people[0]
    assert not person.body_keypoints
    assert len(person.left_hand_keypoints) == len(person.right_hand_keypoints) == 21
    assert person.right_hand_keypoints["wrist"].x == 50
    backend.predict(frame)
    assert tracker.timestamp_ms == 1
    recreated = []
    monkeypatch.setattr(backend, "_create_tracker", lambda: recreated.append(True))
    backend.reset()
    assert recreated == [True]
    backend.close()
    assert backend._tracker is None


def test_holistic_preserves_hands_when_body_landmarks_are_missing():
    backend = MediaPipeHolisticBackend()
    backend._mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB="srgb"),
        Image=lambda image_format, data: data,
    )
    backend._holistic = FakeHolistic(SimpleNamespace(
        pose_landmarks=None, left_hand_landmarks=None, right_hand_landmarks=_landmarks(21)
    ))
    (observation,) = backend.predict(Frame("seq", 0, 0, np.zeros((100, 100, 3), np.uint8)))
    assert len(observation.people[0].right_hand_keypoints) == 21


def test_visualizer_draws_pose_and_finger_landmarks():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    hand = {
        "wrist": Keypoint("wrist", 20, 20),
        "thumb_cmc": Keypoint("thumb_cmc", 30, 20),
        "thumb_mcp": Keypoint("thumb_mcp", 40, 20),
        "thumb_ip": Keypoint("thumb_ip", 50, 20),
        "thumb_tip": Keypoint("thumb_tip", 60, 20),
        "index_finger_mcp": Keypoint("index_finger_mcp", 30, 30),
        "index_finger_pip": Keypoint("index_finger_pip", 40, 30),
        "index_finger_dip": Keypoint("index_finger_dip", 50, 30),
        "index_finger_tip": Keypoint("index_finger_tip", 60, 30),
    }
    person = PersonPose("person-0", left_hand_keypoints=hand)
    observation = PoseObservation("seq-a", 0, 0.0, (person,))

    output = Visualizer.draw_pose_observations(image, (observation,))

    assert np.count_nonzero(output) > 0
    assert not np.count_nonzero(image)


def test_hand_tracker_shows_only_right_hand_when_approaching_cube():
    tracker = HandCubeEngagementTracker()
    cube = Cube("red", (50, 40, 20, 20), (60, 50), 400, 1.0, 0)

    def observation(frame, wrist_x):
        right = {"wrist": Keypoint("wrist", wrist_x, 50)}
        left = {"wrist": Keypoint("wrist", 60, 50)}
        person = PersonPose(
            "person-0",
            body_keypoints={"nose": Keypoint("nose", 2, 2)},
            left_hand_keypoints=left,
            right_hand_keypoints=right,
        )
        return PoseObservation("camera", frame, frame * 33.3, (person,))

    (far,) = tracker.filter((observation(0, 0),), (cube,))
    assert far.people[0].right_hand_keypoints == {}
    assert far.people[0].left_hand_keypoints == {}
    assert far.people[0].body_keypoints == {}

    (approaching,) = tracker.filter((observation(1, 15),), (cube,))
    assert approaching.people[0].right_hand_keypoints
    assert approaching.people[0].left_hand_keypoints == {}

    (not_approaching,) = tracker.filter((observation(2, 15),), (cube,))
    assert not_approaching.people[0].right_hand_keypoints == {}
