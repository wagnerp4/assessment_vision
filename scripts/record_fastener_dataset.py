"""Record a raw session for the screw and nut detector dataset.

Opens in standby (preview only); space starts recording and pauses it again. Runs until q.
Keys 1-5 mark the game stage you are acting out, and 0 clears it. The clip is saved without overlays, with one row per frame in frames.csv (time and stage),
so frames can be sampled and labeled later. See docs/models/fastener-detector.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGES = {ord("1"): "setup", ord("2"): "pickup", ord("3"): "screw", ord("4"): "unscrew", ord("5"): "putdown"}
# What one session should show at some point. Shown on screen as a reminder.
CHECKLIST = (
    "parts apart on the table",
    "screw in left hand, nut in right",
    "nut on tip, screwing to the end of the thread",
    "unscrewing until the nut comes off",
    "nut face-on (hole visible) and edge-on",
    "parts put back, hands away",
)


def open_camera(camera_id: int, width: int, height: int):
    backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY) if sys.platform == "win32" else (cv2.CAP_ANY,)
    for backend in backends:
        capture = cv2.VideoCapture(camera_id, backend)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, 30)
            if any(capture.read()[0] for _ in range(8)):
                return capture
        capture.release()
    return None


def draw(frame: np.ndarray, stage: str | None, paused: bool, elapsed: float, frames: int) -> np.ndarray:
    output = frame.copy()
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1], 64), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, output, 0.45, 0, output)
    if paused and not frames:
        cv2.putText(output, "READY - press SPACE to start recording", (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2, cv2.LINE_AA)
        return output
    state = "PAUSED" if paused else "REC"
    color = (0, 200, 255) if paused else (0, 0, 255)
    cv2.putText(output, state, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    text = f"stage: {stage or '-'}   {elapsed:5.1f}s   {frames} frames   1-5 stage  0 clear  space pause  q quit"
    cv2.putText(output, text, (130, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    for row, item in enumerate(CHECKLIST):
        cv2.putText(
            output, f"- {item}", (16, output.shape[0] - 16 - 24 * (len(CHECKLIST) - 1 - row)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "datasets" / "fasteners" / "raw")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--lighting", default="", help="Free text, e.g. daylight, desk lamp, ceiling only")
    parser.add_argument("--background", default="", help="Free text, e.g. white rim, black pad")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    session = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out = args.out / session
    capture = open_camera(args.camera, args.width, args.height)
    if capture is None:
        print("CAMERA_FAILED")
        return 1
    out.mkdir(parents=True)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # The container rate is nominal; frames.csv holds the real capture time of every frame.
    writer = cv2.VideoWriter(str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    window = "Fastener dataset recording"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print(f"RECORDING {out}", flush=True)

    stage: str | None = None
    # Standby until space, so the camera can be placed and the parts laid out first.
    paused = True
    written = 0
    started: float | None = None
    with (out / "frames.csv").open("w", newline="") as handle:
        rows = csv.writer(handle)
        rows.writerow(["frame", "t_ms", "stage"])
        while True:
            ok, frame = capture.read()
            if not ok:
                # A loose USB cable drops the webcam for a moment; reopen it instead of ending the session.
                print("FRAME_FAILED, reopening camera", flush=True)
                capture.release()
                capture = None
                for _ in range(10):
                    capture = open_camera(args.camera, args.width, args.height)
                    if capture is not None:
                        break
                    time.sleep(1.0)
                if capture is None:
                    print("CAMERA_LOST")
                    break
                continue
            elapsed = time.perf_counter() - started if started is not None else 0.0
            if not paused:
                writer.write(frame)
                rows.writerow([written, round(elapsed * 1000.0, 1), stage or ""])
                written += 1
            cv2.imshow(window, draw(frame, stage, paused, elapsed, written))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
                if started is None:
                    started = time.perf_counter()
            elif key == ord("0"):
                stage = None
            elif key in STAGES:
                stage = STAGES[key]

    writer.release()
    if capture is not None:
        capture.release()
    cv2.destroyAllWindows()
    meta = {
        "session": session,
        "camera": args.camera,
        "width": width,
        "height": height,
        "frames": written,
        "duration_s": round(time.perf_counter() - started, 1) if started is not None else 0.0,
        "lighting": args.lighting,
        "background": args.background,
        "notes": args.notes,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"SAVED {out} frames={written}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
