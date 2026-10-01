import cv2, sys, glob, numpy as np
from cube_detector import CubeDetector
BGR = {'red': (0,0,255), 'yellow': (0,220,255), 'green': (0,200,0), 'blue': (255,80,0)}
files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else 'dev_frames/f*.png'))
det = CubeDetector()
for fi, path in enumerate(files):
    f = cv2.imread(path)
    raw = det.detect_cubes(f, fi)
    out = f.copy()
    for c in raw:
        x, y, w, h = c.bbox
        cv2.rectangle(out, (x, y), (x+w, y+h), BGR[c.color], 3)
        cv2.putText(out, f"{c.color} a={c.area} s={c.confidence:.2f}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BGR[c.color], 2)
    tiles = []
    for k in ['red','yellow','green','blue']:
        m = cv2.cvtColor(cv2.resize(det.last_masks[k], (640, 360)), cv2.COLOR_GRAY2BGR)
        cv2.putText(m, k, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, BGR[k], 2); tiles.append(m)
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite(path.replace('.png', '_det.jpg'), np.vstack([out, grid]))
    print(path, [(c.color, c.bbox) for c in raw])
