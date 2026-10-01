import cv2, numpy as np
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("auto_exposure", cap.get(cv2.CAP_PROP_AUTO_EXPOSURE), "exposure", cap.get(cv2.CAP_PROP_EXPOSURE))
R = {'red': (685,295,755,325), 'yellow': (690,340,755,375), 'blue': (695,380,755,420), 'table': (850,450,1100,600)}
def measure(tag):
    for _ in range(20): cap.read()
    ok, f = cap.read(); hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    out = []
    for k, (x1,y1,x2,y2) in R.items():
        r = hsv[y1:y2, x1:x2].reshape(-1,3); out.append(f"{k} S{int(np.median(r[:,1]))}/V{int(np.median(r[:,2]))}")
    print(f"{tag:14s}", " | ".join(out)); return f
measure("auto")
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
for e in [-4, -5, -6, -7, -8]:
    cap.set(cv2.CAP_PROP_EXPOSURE, e)
    f = measure(f"exp {e} (got {cap.get(cv2.CAP_PROP_EXPOSURE)})")
    cv2.imwrite(f"dev_frames/exp{e}.png", f)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
