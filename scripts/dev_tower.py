import cv2
from cube_detector import CubeDetector
from stacking_analyzer import StackingAnalyzer

for e in [-4, -6, -8]:
    f = cv2.imread(f"dev_frames/exp{e}.png")
    d = CubeDetector()
    a = StackingAnalyzer()
    for i in range(4):
        tr = d.track_cubes(d.detect_cubes(f, i))
    print(
        " tower exp",
        e,
        sorted((c.color, c.bbox) for c in tr),
        "| stacks",
        [[c.color[0] for c in s] for s in a.find_stacks(tr)],
    )
