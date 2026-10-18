"""Train, export, and benchmark the screw and nut detector.

train   Fine-tune a small YOLO model on datasets/fasteners (needs the optional `train` extra).
export  Write models/fastener_detector.onnx and its .json sidecar (classes, input size).
bench   Time the exported model on CPU the way the app runs it (ONNX Runtime, else OpenCV DNN).

Example:
    uv sync --extra train            # CUDA PyTorch from the cu130 index (see pyproject.toml)
    uv run python scripts/train_fastener_detector.py train --model yolo11n.pt --imgsz 416
    uv run python scripts/train_fastener_detector.py export --weights runs/fastener_train/yolo11n_416/weights/best.pt
    uv run python scripts/train_fastener_detector.py bench --threads 2

See docs/models/fastener-detector.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assessment_app.fastener_model import (  # noqa: E402
    CLASSES,
    MODEL_PATH,
    NUT_TO_LENGTH,
    YoloFastenerBackend,
)

DATA = ROOT / "datasets" / "fasteners" / "data.yaml"


def _yolo():
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Training needs the optional extra: uv sync --extra train") from error
    return YOLO


def _resolved_data(data: Path, out: Path) -> Path:
    """Ultralytics resolves a relative `path` against its own datasets folder; pin it to ours."""
    import yaml  # installed with the train extra

    config = yaml.safe_load(data.read_text())
    config["path"] = str(data.parent.resolve())
    out.mkdir(parents=True, exist_ok=True)
    resolved = out / "data.resolved.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=False))
    return resolved


def train(args) -> int:
    project = ROOT / "runs" / "fastener_train"
    name = args.name or f"{Path(args.model).stem}_{args.imgsz}"
    model = _yolo()(args.model)
    if not args.device:
        import torch

        args.device = "0" if torch.cuda.is_available() else "cpu"
    model.train(
        data=str(_resolved_data(args.data, project)),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(project),
        name=name,
        exist_ok=True,
        # Lighting is the main thing that changes between rooms; push brightness harder than default.
        hsv_v=0.5,
        hsv_s=0.5,
        degrees=15.0,
        scale=0.5,
        # Head, tip and nut look the same mirrored, so left-right flips are free extra data.
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
    )
    print(f"Best weights: {project / name / 'weights' / 'best.pt'}")
    return 0


def export(args) -> int:
    model = _yolo()(str(args.weights))
    names = [model.names[index] for index in sorted(model.names)]
    if tuple(names) != CLASSES:
        print(f"Model classes {names} do not match {list(CLASSES)}")
        return 1
    # opset 12 and a fixed input size keep the graph readable by OpenCV DNN.
    onnx = Path(model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True, dynamic=False))
    target = ROOT / MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(onnx, target)
    meta = {
        "classes": list(CLASSES),
        "imgsz": args.imgsz,
        "nut_ratio": args.nut_ratio,
        "source": str(args.weights),
    }
    target.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"Exported {target}")
    return 0


def bench(args) -> int:
    model = ROOT / MODEL_PATH
    if not model.is_file():
        print(f"No model at {model}; run export first")
        return 1
    cv2.setNumThreads(args.threads)
    images = sorted((args.data.parent / "images" / "val").glob("*.jpg"))[: args.frames]
    frames = [cv2.imread(str(path)) for path in images] or [np.full((720, 1280, 3), 128, np.uint8)]
    backend = YoloFastenerBackend(model, threads=args.threads)
    backend.load()
    backend.detect(frames[0])
    times = []
    for index in range(args.frames):
        started = time.perf_counter()
        backend.detect(frames[index % len(frames)])
        times.append((time.perf_counter() - started) * 1000.0)
    print(
        f"{model.name} {backend.runtime} imgsz={backend.input_size} threads={args.threads}: "
        f"median {np.median(times):.1f} ms, p90 {np.percentile(times, 90):.1f} ms, "
        f"{1000.0 / np.median(times):.0f} fps"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--model", default="yolo11n.pt", help="Pretrained start point, e.g. yolo11n.pt or yolov8n.pt")
    fit.add_argument("--imgsz", type=int, default=416)
    fit.add_argument("--epochs", type=int, default=150)
    fit.add_argument("--batch", type=int, default=16)
    fit.add_argument("--device", default="", help="cpu, 0 for the first GPU, or mps; default: GPU if available")
    fit.add_argument("--workers", type=int, default=2)
    fit.add_argument("--patience", type=int, default=30)
    fit.add_argument("--name", default="")
    out = commands.add_parser("export")
    out.add_argument("--weights", type=Path, required=True)
    out.add_argument("--imgsz", type=int, default=416)
    out.add_argument("--nut-ratio", type=float, default=NUT_TO_LENGTH)
    timing = commands.add_parser("bench")
    timing.add_argument("--frames", type=int, default=100)
    timing.add_argument("--threads", type=int, default=2, help="CPU threads; small laptops have 2-4")
    args = parser.parse_args(argv)
    return {"train": train, "export": export, "bench": bench}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
