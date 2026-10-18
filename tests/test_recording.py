import cv2
import numpy as np

from assessment_app.recording import RealtimeRecorder, letterbox


def _frames_in(path) -> int:
    cap = cv2.VideoCapture(str(path))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    return count


def test_recorder_repeats_frames_to_keep_real_time(tmp_path):
    recorder = RealtimeRecorder(tmp_path / "demo.mp4", fps=30)
    frame = np.zeros((120, 160, 3), np.uint8)
    for t_ms in (0.0, 100.0, 250.0, 260.0):
        recorder.write(frame, t_ms)
    # 1 initial frame, then 3 slots by 100 ms, 4 more by 250 ms, none for the 10 ms after that.
    assert recorder.frames_written == 8
    recorder.hold(np.full((300, 300, 3), 255, np.uint8), 1.0)
    assert recorder.frames_written == 38
    path = recorder.close()
    assert _frames_in(path) == 38


def test_letterbox_keeps_aspect_ratio():
    boxed = letterbox(np.full((100, 400, 3), 255, np.uint8), (200, 200))
    assert boxed.shape == (200, 200, 3)
    assert boxed[0, 100].tolist() == [17, 17, 17]
    assert boxed[100, 100].tolist() == [255, 255, 255]
