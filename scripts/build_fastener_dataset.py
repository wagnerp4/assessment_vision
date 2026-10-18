"""Build and check the screw and nut detector dataset (YOLO format).

extract  Sample frames from recorded sessions into images/{train,val}. Whole sessions go to one
         split, so validation frames never come from a clip the model trained on.
check    Validate labels/{train,val} against the images and print instances per class.

See docs/models/fastener-detector.md for the labeling guide.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "fasteners"
CLASSES = ("screw", "screw_head", "screw_tip", "nut")
SPLITS = ("train", "val")


def _sessions(raw: Path, extra: list[Path]) -> list[tuple[str, Path]]:
    found = [(path.parent.name, path) for path in sorted(raw.glob("*/video.mp4"))]
    found += [(path.parent.name, path) for path in extra]
    return found


def _stages(video: Path) -> dict[int, tuple[float, str]]:
    table = video.parent / "frames.csv"
    if not table.is_file():
        return {}
    with table.open(newline="") as handle:
        return {int(row["frame"]): (float(row["t_ms"]), row["stage"]) for row in csv.DictReader(handle)}


def _thumbnail(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def extract(args) -> int:
    sessions = _sessions(args.raw, args.video)
    if not sessions:
        print(f"No sessions in {args.raw} (expected <session>/video.mp4)")
        return 1
    names = [name for name, _ in sessions]
    val = set(args.val) if args.val else ({names[-1]} if len(names) > 1 else set())
    unknown = val - set(names)
    if unknown:
        print(f"Unknown --val sessions: {sorted(unknown)}")
        return 1
    rows = []
    for name, video in sessions:
        split = "val" if name in val else "train"
        folder = args.out / "images" / split
        folder.mkdir(parents=True, exist_ok=True)
        stages = _stages(video)
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        index, kept, last_t, last_thumb = 0, 0, -1e9, None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            t_ms, stage = stages.get(index, (index * 1000.0 / fps, ""))
            if t_ms - last_t >= args.every_ms and (args.max_per_session <= 0 or kept < args.max_per_session):
                thumb = _thumbnail(frame)
                # Near-duplicates (nothing moved) add labeling work but no information; the camera's
                # first frames are often black while it warms up.
                lit = float(thumb.mean()) >= 12.0
                if lit and (last_thumb is None or float(np.mean(np.abs(thumb - last_thumb))) >= args.min_change):
                    image = f"{name}_{index:06d}.jpg"
                    cv2.imwrite(str(folder / image), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    rows.append([f"images/{split}/{image}", name, index, round(t_ms, 1), stage, split])
                    last_t, last_thumb = t_ms, thumb
                    kept += 1
            index += 1
        capture.release()
        print(f"{name}: {kept} frames -> {split}")
    manifest = args.out / "manifest.csv"
    new = not manifest.is_file()
    with manifest.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(["image", "session", "frame", "t_ms", "stage", "split"])
        writer.writerows(rows)
    print(f"{len(rows)} frames written; manifest {manifest}")
    return 0


def check_labels(dataset: Path) -> tuple[dict, list[str]]:
    """Per-split counts and a list of problems. An empty label file is a valid negative frame."""
    counts: dict = {}
    problems: list[str] = []
    for split in SPLITS:
        images = {path.stem for path in (dataset / "images" / split).glob("*.jpg")}
        labels = {path.stem: path for path in (dataset / "labels" / split).glob("*.txt")}
        instances: Counter = Counter()
        for stem, path in sorted(labels.items()):
            if stem not in images:
                problems.append(f"{split}/{stem}.txt has no image")
            for number, line in enumerate(path.read_text().splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                try:
                    cls = int(parts[0])
                    box = [float(value) for value in parts[1:]]
                except ValueError:
                    problems.append(f"{split}/{stem}.txt:{number} is not numeric")
                    continue
                if not 0 <= cls < len(CLASSES) or len(box) != 4:
                    problems.append(f"{split}/{stem}.txt:{number} needs 'class cx cy w h' with class 0-{len(CLASSES) - 1}")
                    continue
                if not all(0.0 <= value <= 1.0 for value in box) or box[2] <= 0 or box[3] <= 0:
                    problems.append(f"{split}/{stem}.txt:{number} box is not normalized to 0-1")
                    continue
                instances[CLASSES[cls]] += 1
        counts[split] = {
            "images": len(images),
            "labeled": len(images & set(labels)),
            "unlabeled": len(images - set(labels)),
            **{name: instances[name] for name in CLASSES},
        }
    return counts, problems


def check(args) -> int:
    counts, problems = check_labels(args.out)
    header = ["split", "images", "labeled", "unlabeled", *CLASSES]
    print("  ".join(f"{item:>10}" for item in header))
    for split, row in counts.items():
        print("  ".join(f"{value:>10}" for value in [split, *row.values()]))
    for problem in problems[:50]:
        print("PROBLEM", problem)
    if len(problems) > 50:
        print(f"... {len(problems) - 50} more")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DATASET, help="Dataset root (holds images/, labels/)")
    commands = parser.add_subparsers(dest="command", required=True)
    pull = commands.add_parser("extract", help="Sample frames from recorded sessions")
    pull.add_argument("--raw", type=Path, default=DATASET / "raw")
    pull.add_argument("--video", type=Path, action="append", default=[], help="Extra video; its folder name is the session")
    pull.add_argument("--val", action="append", default=[], help="Session name for validation (repeatable)")
    pull.add_argument("--every-ms", type=float, default=400.0)
    pull.add_argument("--min-change", type=float, default=3.0, help="Mean gray change to keep a frame")
    pull.add_argument("--max-per-session", type=int, default=0)
    commands.add_parser("check", help="Validate labels and count instances")
    args = parser.parse_args(argv)
    return extract(args) if args.command == "extract" else check(args)


if __name__ == "__main__":
    sys.exit(main())
