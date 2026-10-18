import collections
import sys
import time

import cv2
from app import AssessmentApp
from game_logic import GameDifficulty

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
a = AssessmentApp()
assert a.initialize_camera()
for _ in range(15):
    a.cap.read()
a.game_engine.start_game(GameDifficulty.HARD)
readings = collections.Counter()
t0 = time.time()
for i in range(N):
    ok, f = a.cap.read()
    r = a._process_frame(f)
    a.frame_count += 1
    readings[" | ".join("-".join(c.color[0] for c in s) for s in r["stacks"]) or "no stack"] += 1
    if i == N - 1:
        cv2.imwrite("dev_frames/live_app.jpg", r["visualization"])
a.cap.release()
print(f"{N} frames in {time.time() - t0:.1f}s")
print("stack readings:", dict(readings))
print(
    "game:",
    {
        k: v
        for k, v in a.game_engine.get_game_status().items()
        if k in ("state", "goal", "stacks_completed", "score")
    },
)
