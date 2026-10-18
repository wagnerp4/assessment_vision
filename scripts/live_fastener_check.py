"""Live check of the retuned screw and nut detector.

The window shows the camera with the detection drawn on it. Press q to stop early.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assessment_app.contracts import Frame  # noqa: E402
from assessment_app.fastener_model import MODEL_PATH, YoloFastenerBackend  # noqa: E402
from assessment_app.fasteners import FastenerDetector  # noqa: E402
from assessment_app.visualizer import Visualizer  # noqa: E402

PHASES = (
    (3.0, "Get ready", "Camera still, above the white rim."),
    (45.0, "Try it", "Purple lines are nut edges, up to 3. q quits."),
)
UNTIL_Q = ((math.inf, "Live", "Yellow screw, green nut, purple nut edges. q quits."),)


def open_camera(camera_id: int = 0):
    backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY) if sys.platform == "win32" else (cv2.CAP_ANY,)
    for backend in backends:
        capture = cv2.VideoCapture(camera_id, backend)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            capture.set(cv2.CAP_PROP_FPS, 30)
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            capture.set(cv2.CAP_PROP_EXPOSURE, -6)
            if any(capture.read()[0] for _ in range(8)):
                return capture
        capture.release()
    return None


def phase_at(elapsed: float):
    cursor = 0.0
    for duration, title, detail in PHASES:
        if elapsed < cursor + duration:
            return title, detail, duration - (elapsed - cursor)
        cursor += duration
    return None, None, 0.0


def draw_hands(output: np.ndarray, hands: list) -> None:
    """Hands sorted left to right on screen; with this camera that is the patient's left, then right."""
    for side, keypoints in zip(("L", "R"), hands):
        color = (255, 120, 0) if side == "L" else (0, 120, 255)
        Visualizer._draw_keypoint_graph(output, keypoints, Visualizer.HAND_CONNECTIONS, color)
        top = min(keypoints.values(), key=lambda point: point.y)
        cv2.putText(output, side, (int(top.x), int(top.y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def detect_hands(backend, frame: np.ndarray, index: int, t_ms: float) -> list:
    if backend is None:
        return []
    observations = backend.predict(Frame("live", index, t_ms, frame))
    if not observations:
        return []
    person = observations[0].people[0]
    hands = [keypoints for keypoints in (person.left_hand_keypoints, person.right_hand_keypoints) if keypoints]
    return sorted(hands, key=lambda keypoints: sum(point.x for point in keypoints.values()) / len(keypoints))


def draw(frame: np.ndarray, found: dict, title: str, detail: str, remaining: float, hands: list = ()) -> np.ndarray:
    output = frame.copy()
    draw_hands(output, list(hands))
    tip, head = found["screw_tip"], found["screw_head"]
    if tip and head:
        p1 = (int(tip[0]), int(tip[1]))
        p2 = (int(head[0]), int(head[1]))
        cv2.line(output, p1, p2, (0, 220, 255), 3, cv2.LINE_AA)
        cv2.circle(output, p1, 8, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(output, p2, 10, (255, 180, 0), -1, cv2.LINE_AA)
        if found["depth"] is not None:
            depth = float(found["depth"])
            mark = (int(p1[0] + (p2[0] - p1[0]) * depth), int(p1[1] + (p2[1] - p1[1]) * depth))
            cv2.circle(output, mark, 8, (0, 255, 0), -1, cv2.LINE_AA)
    polygon = found["nut_polygon"]
    if len(polygon) >= 3:
        points = np.array(polygon, np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [points], True, (0, 255, 0), 3, cv2.LINE_AA)
    center = found["nut_center"]
    if center and found["depth"] is None:
        cv2.circle(output, (int(center[0]), int(center[1])), 8, (0, 255, 0), -1, cv2.LINE_AA)
    for edge in found.get("nut_edges") or []:
        cv2.line(output, edge[0], edge[1], (255, 0, 255), 3, cv2.LINE_AA)
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1], 96), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, output, 0.45, 0, output)
    depth = found["depth"]
    depth_text = "-" if depth is None else f"{depth:.2f}"
    cv2.putText(output, title, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, detail, (16, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    edges = len(found.get("nut_edges") or ())
    clock = "" if math.isinf(remaining) else f"  {remaining:0.0f}s"
    status = f"on={found['nut_on_screw']}  depth={depth_text}  edges={edges}  hands={len(hands)}{clock}"
    cv2.putText(output, status, (output.shape[1] - 520, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def main() -> int:
    global PHASES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until-q", action="store_true", help="Run until q is pressed instead of a timed check")
    parser.add_argument("--no-hands", action="store_true", help="Skip MediaPipe hand landmarks")
    args = parser.parse_args()
    if args.until_q:
        PHASES = UNTIL_Q
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out = ROOT / "runs" / "calibration" / f"live_{stamp}"
    out.mkdir(parents=True)
    capture = open_camera()
    if capture is None:
        print("CAMERA_FAILED")
        return 1
    # The learned detector once it is exported; same detect() result as the classical one.
    detector = YoloFastenerBackend(ROOT / MODEL_PATH) if (ROOT / MODEL_PATH).is_file() else FastenerDetector()
    print(f"DETECTOR {type(detector).__name__}", flush=True)
    hand_backend = None
    model = ROOT / "models" / "hand_landmarker.task"
    if not args.no_hands and model.is_file():
        from assessment_app.model_backends import MediaPipeHandBackend

        hand_backend = MediaPipeHandBackend(model)
        hand_backend.load("cpu")
    index = 0
    window = "Fastener live check"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print(f"LIVE {out}", flush=True)
    started = time.perf_counter()
    last_save = -1.0
    while True:
        ok, frame = capture.read()
        if not ok:
            print("FRAME_FAILED")
            break
        elapsed = time.perf_counter() - started
        title, detail, remaining = phase_at(elapsed)
        if title is None:
            break
        found = detector.detect(frame)
        hands = detect_hands(hand_backend, frame, index, elapsed * 1000.0)
        index += 1
        shown = draw(frame, found, title, detail, remaining, hands)
        cv2.imshow(window, shown)
        if elapsed - last_save >= 2.0:
            last_save = elapsed
            slug = "".join(ch if ch.isalnum() else "_" for ch in title).strip("_")
            name = f"{elapsed:05.1f}_{slug}.jpg"
            cv2.imwrite(str(out / name), shown)
            depth = found["depth"]
            depth_text = "-" if depth is None else f"{depth:.2f}"
            screw = found["screw_tip"] is not None
            nut = found["nut_center"] is not None
            print(
                f"t={elapsed:04.1f} {title} screw={screw} nut={nut} on={found['nut_on_screw']} depth={depth_text} hands={len(hands)}",
                flush=True,
            )
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("QUIT_EARLY")
            break
    capture.release()
    if hand_backend is not None:
        hand_backend.close()
    cv2.destroyAllWindows()
    print(f"SAVED {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
