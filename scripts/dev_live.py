import cv2, sys, time, collections
from cube_detector import CubeDetector
N = int(sys.argv[1]) if len(sys.argv) > 1 else 90
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
for _ in range(15): cap.read()
det = CubeDetector(); ms = []; seen = collections.Counter(); raw_n = collections.Counter(); ids = set()
BGR = {'red': (0,0,255), 'yellow': (0,220,255), 'green': (0,200,0), 'blue': (255,80,0)}
for i in range(N):
    ok, f = cap.read()
    t = time.perf_counter(); raw = det.detect_cubes(f, i); tr = det.track_cubes(raw); ms.append((time.perf_counter()-t)*1000)
    raw_n[len(raw)] += 1
    for c in tr: seen[(c.cube_id, c.color)] += 1; ids.add(c.cube_id)
    if i in (N//2, N-1):
        for c in tr:
            x,y,w,h = c.bbox; cv2.rectangle(f,(x,y),(x+w,y+h),BGR[c.color],3)
            cv2.putText(f,f"{c.color} #{c.cube_id}",(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.8,BGR[c.color],2)
        cv2.imwrite(f"dev_frames/live_{i}.jpg", f)
ms.sort()
print(f"detect+track ms: median {ms[len(ms)//2]:.1f}  p95 {ms[int(len(ms)*.95)]:.1f}")
print("raw detections per frame:", dict(raw_n))
print("track (id,color) -> frames:", dict(seen))
print("distinct track ids:", sorted(ids))
