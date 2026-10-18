# Web interface

**Status: implemented (prototype), deployed at https://assessment-vision.fly.dev**, including the analysis panel.

## How it works

The browser asks for camera access (`getUserMedia`), and sends 540p JPEG frames over a WebSocket (`/ws`), with one frame in flight at a time. Each frame starts with an 8-byte little-endian float64 capture time (`performance.now()`), which the server uses as the session clock. The server (`assessment_app.web`, FastAPI) runs the same pipeline as the local app and returns the annotated frame and a status message. Each connection gets its own pipeline, so trackers, scores and rules are never shared between players. The page shows fps, round trip, and per-model server time.

| Control | Effect |
|---|---|
| Camera picker | Lists cameras and skips virtual ones (OBS, Snap Camera, …) by default |
| Game | Simple or Random stacking (restarts the game) |
| Difficulty, Start, Reset | As in the local app |
| Hand tracking | Reconnects with the hand landmark model (or holistic, which needs the body in view) |
| Finger pairs, Next pair | Toggle the finger-pair rule and cycle the required pair (needs hand tracking) |
| Record + source | Record the annotated view, or the whole tab including the analysis panel (the browser asks which tab to share). The video is downloaded when recording stops (MP4 where the browser supports it, otherwise WebM). Recording happens in the browser; nothing is uploaded |

Query parameter `?pose=off|hands|holistic` selects the models per connection. Text commands are `start` (difficulty, game), `reset`, `finger_rule`, and `next_pair`. Besides `status`, the server sends `summary` messages (session metadata plus analysis) whenever the game makes progress.

## Analysis panel

Below the video: stat cards, rounds chart, step time by tower position, move phases, finger-pair comparison, and a progression chart over runs stored in this browser (same game mode and difficulty). Charts are inline SVG drawn by `static/analysis.js`. See [session analysis](../../analysis/session-analysis.md).

## Deployment

`Dockerfile` and `fly.toml` define a single shared-CPU machine with 1 GB in Frankfurt that stops when idle. The image includes MediaPipe and both model files, and loads every mode during the build. Pushes to `main` deploy through `.github/workflows/fly-deploy.yml`. The workflow is: test locally first, then deploy to Fly.

## Performance

Measured with the demo video:

| Models | Server time per frame |
|---|---|
| Cubes only | about 70 ms |
| Cubes + hand model | about 85 ms |
| Cubes + holistic | about 115 ms |

That gives roughly 6–9 fps through the network while the shared CPU can still burst. Under sustained load, Fly throttles the shared CPU to well under 1 fps. A dedicated CPU or moving detection into the browser are the options for higher frame rates.

## Privacy

Frames are processed in memory and never stored. The server keeps no data after a connection closes. The analysis panel stores finished runs in the browser's localStorage, not on the server; see [session analysis](../../analysis/session-analysis.md#5-storage).
