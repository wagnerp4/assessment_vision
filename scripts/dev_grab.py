import cv2, sys, time
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened(): print("CAMERA_BUSY_OR_MISSING"); sys.exit(1)
for _ in range(15): cap.read()  # let auto-exposure settle
for i in range(n):
    ok, f = cap.read()
    if not ok: print("READ_FAIL"); sys.exit(1)
    cv2.imwrite(f"dev_frames/f{i:02d}.png", f); time.sleep(0.2)
print("OK", f.shape)
