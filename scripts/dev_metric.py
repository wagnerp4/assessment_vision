import cv2, sys, collections
from cube_detector import CubeDetector
def run(path, det=None):
    cap = cv2.VideoCapture(path); det = det or CubeDetector(); n = 0; exact = 0; per = collections.Counter(); fp = 0; ids = set(); flips = 0; last = {}
    while True:
        ok, f = cap.read()
        if not ok: break
        tr = det.track_cubes(det.detect_cubes(f, n)); cols = [c.color for c in tr]
        exact += sorted(cols) == ['blue','green','red','yellow']
        for k in set(cols): per[k] += 1
        fp += sum(max(0, cols.count(k) - 1) for k in set(cols))
        for c in tr:
            ids.add(c.cube_id)
            if c.cube_id in last and last[c.cube_id] != c.color: flips += 1
            last[c.cube_id] = c.color
        n += 1
    print(f"{path}: exact-4 {100*exact/n:.0f}% | seen b {100*per['blue']/n:.0f}% g {100*per['green']/n:.0f}% r {100*per['red']/n:.0f}% y {100*per['yellow']/n:.0f}% | dup-color {fp} | ids {len(ids)} | label flips {flips}")
if __name__ == '__main__':
    for p in sys.argv[1:]: run(p)
