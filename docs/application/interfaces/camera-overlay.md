# Camera and image-stream overlay

**Status: implemented locally; planned for refinement.** The application captures camera or video frames, renders cube/stack/game information over the image, and handles keyboard commands. With the optional `--pose` flag, it also overlays the playing hand's landmarks (only while it approaches or touches a cube). Camera selection is an OpenCV integer index; availability and capture behavior vary by OS and driver.

Camera mode starts on a menu drawn over the live view. The choices match the browser client: simple or random stacking, difficulty (easy through expert), hand tracking, hand model (hand or holistic), finger pairs and the required pair, and recording. Launch flags set the initial selection. Click a control or use the arrow keys; `s` or Enter starts the game, and `q` quits. The camera stays visible below the menu so the table can be framed before the game starts.

During a game: `s` restarts, `r` resets, `p` pauses, `v` starts or stops recording, `q` leaves the game. With hand tracking on, `g` toggles the finger-pair rule and `f` cycles the required pair. Random stacking adds a target tower (top right), and the pair rule adds "Use:" and "Grip:" lines. A hint line explains wrong cubes and rejected moves. When a game completes, or when `q` is pressed with progress, the run is saved to `runs/` and a summary window opens; any key there returns to the menu. See [session analysis](../../analysis/session-analysis.md#4-presentation). The HUD timer uses the real capture clock.

**Recording:** `v` (or `--record` at launch) writes the annotated view to `runs/<time>_recording.mp4`. Frames are repeated as needed so playback runs at real speed even though processing is slower than 30 fps. When a game finishes during a recording, the summary figure is appended for 5 seconds. The on-screen REC badge is not part of the file.

Keep the frame visible while overlaying concise status, task progress, and confidence/quality warnings where available. Avoid blocking inference on optional display work. Specify what happens on capture failure, dropped frames, pause/reset, and end-of-sequence. Camera acquisition belongs to device/input handling; drawing belongs to presentation; task semantics remain in task/game logic.

For setup, see [camera devices](../../devices/cameras.md). The overlay should not imply that derived behavior features are medical conclusions.
