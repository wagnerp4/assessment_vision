"""Record a short calibration clip of this screw and nut.

The window tells you what to show. Press q to stop early.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SCRIPTS = {
    "halfway": (
        (3.0, "Get ready", "Camera still. Hand away from the parts."),
        (10.0, "Halfway", "Nut about halfway on. Hold still."),
    ),
    # The five stages of the screwing game, at a calm patient's pace.
    "gameplay": (
        (4.0, "Get ready", "Camera fixed. Screw and nut apart on the table."),
        (5.0, "Setup", "Parts on the table, hands in view. Hold still."),
        (6.0, "Pickup", "Screw in LEFT hand, nut in RIGHT hand. Lift both."),
        (20.0, "Screw", "Put the nut on the tip and screw it about halfway."),
        (15.0, "Unscrew", "Hold the screw and unscrew the nut until it is off."),
        (6.0, "Putdown", "Lay screw and nut apart on the table. Hands away."),
    ),
}
PHASES = SCRIPTS["halfway"]


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


def stats(frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return {
        "gray_p10": float(np.percentile(gray, 10)),
        "gray_p50": float(np.percentile(gray, 50)),
        "gray_p90": float(np.percentile(gray, 90)),
        "sat_p50": float(np.percentile(hsv[:, :, 1], 50)),
        "sat_p90": float(np.percentile(hsv[:, :, 1], 90)),
    }


def draw(frame: np.ndarray, title: str, detail: str, remaining: float) -> np.ndarray:
    output = frame.copy()
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1], 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, output, 0.45, 0, output)
    cv2.putText(output, title, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, detail, (24, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        output,
        f"{remaining:0.0f}s   q quits",
        (output.shape[1] - 220, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    return output


def main() -> int:
    global PHASES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", choices=sorted(SCRIPTS), default="halfway", help="Which prompts to show")
    parser.add_argument("--every", type=float, default=1.0, help="Seconds between saved still frames")
    args = parser.parse_args()
    PHASES = SCRIPTS[args.script]
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out = ROOT / "runs" / "calibration" / (stamp if args.script == "halfway" else f"{args.script}_{stamp}")
    frames = out / "frames"
    frames.mkdir(parents=True)
    capture = open_camera()
    if capture is None:
        print("CAMERA_FAILED")
        return 1
    window = "Fastener calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out / "calibration.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height))
    print(f"RECORDING {out}", flush=True)
    started = time.perf_counter()
    total = sum(item[0] for item in PHASES)
    saved = []
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
        # The clip stays raw so detectors can be replayed on it; the prompts are only on screen.
        writer.write(frame)
        cv2.imshow(window, draw(frame, title, detail, remaining))
        if elapsed - last_save >= args.every:
            last_save = elapsed
            slug = "".join(ch if ch.isalnum() else "_" for ch in title).strip("_")
            name = f"{elapsed:05.1f}_{slug}.jpg"
            cv2.imwrite(str(frames / name), frame)
            saved.append({"t": round(elapsed, 2), "phase": title, "file": name, **stats(frame)})
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("QUIT_EARLY")
            break
        if elapsed > total + 0.2:
            break
    writer.release()
    capture.release()
    cv2.destroyAllWindows()
    (out / "stats.json").write_text(json.dumps(saved, indent=1))
    print(f"SAVED {out} frames={len(saved)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
