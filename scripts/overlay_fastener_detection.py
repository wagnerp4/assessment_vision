"""Draw the current fastener detector on the saved calibration frames."""

from pathlib import Path

import cv2
import numpy as np

from assessment_app.fasteners import FastenerDetector

ROOT = Path(__file__).resolve().parents[1]
FRAMES = (
    ROOT / "runs/calibration/2026-10-17T18-18-24/frames/011.3_Screw_only.jpg",
    ROOT / "runs/calibration/2026-10-17T18-22-08/frames/006.1_Halfway.jpg",
)


def main() -> None:
    detector = FastenerDetector()
    out = ROOT / "runs/calibration/detector_now"
    out.mkdir(parents=True, exist_ok=True)
    for path in FRAMES:
        image = cv2.imread(str(path))
        found = detector.detect(image)
        vis = image.copy()
        tip, head = found["screw_tip"], found["screw_head"]
        if tip and head:
            p1 = (int(tip[0]), int(tip[1]))
            p2 = (int(head[0]), int(head[1]))
            cv2.line(vis, p1, p2, (0, 220, 255), 3, cv2.LINE_AA)
            cv2.circle(vis, p1, 8, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, p2, 10, (255, 180, 0), -1, cv2.LINE_AA)
        polygon = found["nut_polygon"]
        if len(polygon) >= 3:
            points = np.array(polygon, np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [points], True, (0, 255, 0), 3, cv2.LINE_AA)
        center = found["nut_center"]
        if center:
            cv2.circle(vis, (int(center[0]), int(center[1])), 8, (0, 255, 0), -1, cv2.LINE_AA)
        for edge in found.get("nut_edges") or []:
            cv2.line(vis, edge[0], edge[1], (255, 0, 255), 2, cv2.LINE_AA)
        depth = found["depth"]
        depth_text = "-" if depth is None else f"{depth:.2f}"
        label = f"on={found['nut_on_screw']}  depth={depth_text}  edges={len(found.get('nut_edges') or ())}"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 48), (0, 0, 0), -1)
        cv2.putText(vis, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        dest = out / f"{path.parent.parent.name}_{path.stem}.jpg"
        cv2.imwrite(str(dest), vis)
        print(dest)
        print(" ", label)


if __name__ == "__main__":
    main()
