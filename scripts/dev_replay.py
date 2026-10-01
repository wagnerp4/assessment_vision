import cv2, sys, numpy as np
from cube_detector import CubeDetector
from stacking_analyzer import StackingAnalyzer
BGR = {'red': (0,0,255), 'yellow': (0,220,255), 'green': (0,200,0), 'blue': (255,80,0)}
path = sys.argv[1]; step = int(sys.argv[2]) if len(sys.argv) > 2 else 25
cap = cv2.VideoCapture(path); det = CubeDetector(); an = StackingAnalyzer()
thumbs = []; i = 0; prev = None
while True:
    ok, f = cap.read()
    if not ok: break
    raw = det.detect_cubes(f, i); tr = det.track_cubes(raw); stacks = an.find_stacks(tr)
    summ = (len(raw), len(tr), tuple(sorted(c.color[0] for c in tr)), tuple(''.join(c.color[0] for c in s) for s in stacks))
    if summ != prev:
        print(f"{i:4d} raw={summ[0]} tracked={summ[1]} colors={''.join(summ[2])} stacks={summ[3]} ids={sorted(c.cube_id for c in tr)}")
        prev = summ
    if i % step == 0:
        v = f.copy()
        for c in raw:
            x,y,w,h = c.bbox; cv2.rectangle(v,(x,y),(x+w,y+h),(255,255,255),1)
        for c in tr:
            x,y,w,h = c.bbox; cv2.rectangle(v,(x,y),(x+w,y+h),BGR[c.color],4)
            cv2.putText(v,f"{c.color[0]}{c.cube_id}",(x+4,y+30),cv2.FONT_HERSHEY_SIMPLEX,1,BGR[c.color],3)
        cv2.putText(v,f"#{i} stacks={['-'.join(c.color[0] for c in s) for s in stacks]}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,0),6)
        cv2.putText(v,f"#{i} stacks={['-'.join(c.color[0] for c in s) for s in stacks]}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)
        thumbs.append(cv2.resize(v,(426,240)))
    i += 1
while len(thumbs) % 5: thumbs.append(np.zeros_like(thumbs[0]))
rows = [np.hstack(thumbs[r:r+5]) for r in range(0, len(thumbs), 5)]
for k in range(0, len(rows), 4):
    cv2.imwrite(path.replace('.mp4', f'_sheet{k//4}.jpg'), np.vstack(rows[k:k+4]))
print("frames", i)
