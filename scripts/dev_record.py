import cv2, sys, time
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 30
name = sys.argv[2] if len(sys.argv) > 2 else 'clip'
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
for _ in range(10): cap.read()
out = cv2.VideoWriter(f'dev_frames/{name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280, 720))
t0 = time.time(); n = 0
while time.time() - t0 < secs:
    ok, f = cap.read()
    if ok: out.write(f); n += 1
out.release(); print(f"recorded {n} frames in {time.time()-t0:.1f}s -> dev_frames/{name}.mp4")
