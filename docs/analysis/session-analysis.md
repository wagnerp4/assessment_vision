# Session analysis

**Status: implemented** (local summary window and web analysis panel). Code: `analysis.py` (measures), `grasp.py` (move phases and finger features), `tasks.py` (session record), `summary_plot.py` (local figure), `static/analysis.js` (web panel). This page defines what is measured, how session data is recorded and stored, and how each client presents it.

Measures here describe observable behavior in the stacking games. They are not diagnoses and do not measure grip force; see [interaction measures](interaction-measures.md) for interpretation limits.

## 1. Prerequisite: a real clock

Timestamps used to be synthesized as `frame_index × 1000 / 30`, while real processing runs below 30 fps (local: roughly 15–20 fps with hand tracking; web: about 6–10 fps), which made every duration too short by a factor of about 1.5–5. Now:

- **Local:** stamp each frame with `time.perf_counter()` at capture, relative to session start, and pass it as `Frame.timestamp_ms`.
- **Web:** the browser stamps each frame with `performance.now()` at capture and sends it with the JPEG (an 8-byte float64 header before the image bytes). The server uses that value, so network delay and server throttling do not distort durations.
- **Video files:** keep using the container's frame timestamps.
- Game engines keep frame counts for stability windows, but every logged event and every timer uses milliseconds.

Timing resolution is limited by the frame interval: at 8 fps an event boundary is only known to within about ±125 ms. Each session therefore records its effective fps, and summaries show it next to the timings.

## 2. Event model

Analysis derives everything from one ordered event log per session. Events carry the session clock (`t_ms`) and the frame index. Two rules make timings accurate:

- **Achieved vs. confirmed.** The game accepts a step only after it has been stable for `step_frames` frames, and a move only after the cube has rested for `rest_frames` frames. Events record both the moment the state was first reached (`achieved_ms`) and when the game confirmed it (`confirmed_ms`). Durations use `achieved_ms`; otherwise every step would include the confirmation delay.
- **Moves are split into phases.** `CubeMove` grows from start/end frames to the full phase timeline below.

### Move phases

For each cube move, with hand tracking on:

| Point | Definition |
|---|---|
| `grip_onset_ms` | First frame where thumb and a fingertip touch the cube (see `detect_grip`) |
| `lift_ms` | First frame where the cube is away from its resting place |
| `settle_ms` | First still frame at the new place (start of the rest window) |
| `release_ms` | First frame after settling in which the hand is visible but no longer gripping the cube (tracked from the settle frame on; if the move is confirmed first, a later `cube_released` event supplies it) |

Without hand tracking only `lift_ms` and `settle_ms` exist; grip-dependent fields are `null`.

### Event types

| Event | Data |
|---|---|
| `session_started` | Session metadata (section 5) |
| `rule_changed` | Finger-pair rule on/off, required pair |
| `round_started` | Round, target order |
| `cube_moved` | Color, phase timestamps, detected pair, vote share of the detected pair, finger features, required pair, counted |
| `cube_released` | Color (release found after the move was confirmed) |
| `step_completed` | Round, phase (`stack`/`unstack`), tower index, color, achieved/confirmed |
| `wrong_cube` | Round, color (a wrong cube stood on the tower long enough to be confirmed) |
| `stack_confirmed` | Simple stacking: colors, achieved/confirmed |
| `stack_completed` / `round_completed` | Round, achieved/confirmed |
| `game_completed` | Final score, totals |
| `quality` | Every few seconds: effective fps and the share of frames in which the hand was detected |

`rule_changed` is logged whenever the operator presses `g`/`f` (or the web buttons). That lets analysis split a session into segments by required pair.

## 3. Measures

### Global timer (both games)

- **Simple stacking:** from game start (first processed frame after start) to the first counted stack (`achieved_ms`), and the hold duration.
- **Random stacking:** from round 1 `round_started` to the last `round_completed`. The HUD shows it live as `mm:ss`.

### Random stacking, per placement in one sequence

Each round has `n` cubes: `n − 1` placements onto the base, then `n − 1` removals. For every placement and removal:

| Measure | Definition | Needs hands |
|---|---|---|
| Step time | Previous step's `achieved_ms` (or round start) → this step's `achieved_ms` | no |
| Search / reaction | Previous step achieved → `grip_onset_ms` | yes |
| Grasp-to-lift | `grip_onset_ms` → `lift_ms` | yes |
| Transport | `lift_ms` → `settle_ms` | no |
| Release dwell | `settle_ms` → `release_ms`: how long the fingers rest on the cube after a correct placement | yes |

Step times are compared by **tower index** (first cube on the base, second, third), because higher placements need more precision.

### Random stacking, per sequence

For each of the 4 rounds:

- **Stack time:** `round_started` → `stack_completed`.
- **Unstack time:** `stack_completed` → `round_completed`.
- **Round time:** the sum of both.
- **Wrong cubes:** times a wrong color was placed.
- **Rejected moves:** moves with the wrong finger pair.

Across the 4 rounds, the summary shows mean and spread, plus the trend from round 1 to round 4 (learning or fatigue within a session).

### Finger keypoint features

From the playing hand's 21 landmarks, per move. Pixel distances are normalized by hand size (wrist → middle-finger knuckle), so they do not depend on camera distance:

- **Pinch aperture** (thumb tip to gripping fingertip) at grip onset, mean during transport, and at release.
- **Transport path:** path length and peak speed of the pinch midpoint, and smoothness (count of speed peaks).
- **Grip consistency:** the vote share of the detected pair during the move, plus extra fingertips touching the cube.
- **Hand coverage:** the share of the move's frames in which the hand was detected. Moves below 60% coverage are flagged as low-confidence and excluded from finger-based comparisons.

All of these are 2D image measures from one camera. MediaPipe's depth value is not reliable enough to use.

### Comparing finger pairs

Every move records the **required pair** (when the rule is on) and the **detected pair** (whenever hand tracking is on, even with the rule off). Comparisons:

- **Per pair within a game:** the session is split by `rule_changed` events. For each pair, the summary shows step time, transport time and release dwell (median and spread, with the number of moves), and the rejected-move rate.
- **No pair schedule.** The game never assigns pairs on its own. With the finger-pair rule off, each move belongs to the pair detected from the grip. With the rule on, the operator selects a pair (`g`/`f` locally, the buttons on the web) and moves belong to that pair; moves made with another pair are counted as rejected for it.
- **Within a session**, each pair has only about 6 moves per round, so the view shows the individual points and medians rather than significance tests. **Across sessions**, per-pair results are pooled for the progression view.
- Pairs are compared at the same tower index and difficulty, because those also change step time.

### Data quality, always shown

Effective fps, hand coverage, the share of moves without a detected grip, and the number of steps confirmed without a recorded move. Local and web sessions are labeled by client and never pooled, because their frame rates differ.

## 4. Presentation

### Local app: summary window

When a game completes (or on quit with at least one completed round), a second window "Session summary" opens:

- **Header:** game mode, difficulty, total time, rounds, score, fps, hand coverage.
- **Round bars:** horizontal bars per round split into stack and unstack, with the target colors as chips.
- **Placement timeline:** for each move, a segmented bar (search, grasp, transport, dwell), grouped by round.
- **Pair table:** shown if the finger-pair rule was used; medians per pair.

The figure is rendered with matplotlib (`uv sync --extra analysis`; also installed with the `pose` extra) and saved as a PNG next to the session file. The same numbers are always printed to the console. After quitting, the window stays open until a key is pressed.

### Web app: analysis panel below the video

The server computes the summary in Python, using the same code as the local app, and sends a `summary` message whenever a step, stack, round, or the game completes, so the panel fills in while you play. The page renders it under the status tiles:

1. **Stat cards:** total time, mean round time, best round, rejected moves, hand coverage, fps.
2. **Rounds chart:** stacked horizontal bars (stack, unstack) per round with target chips.
3. **Placement chart:** step time by tower index, with each placement as a dot and a median line. With hand tracking, a stacked bar shows the phases.
4. **Finger pairs:** strip plot of step time per pair, plus a small table (n, median transport, median dwell, rejected).
5. **Progression:** line chart of total time and median step time across stored runs of the same game mode and difficulty, with a rolling median. It can be filtered by finger pair.

Charts are hand-drawn SVG: a few small helpers, no build step, and no third-party script. The dark palette matches the page, with cube colors used only for cube-specific marks.

## 5. Storage

### Session record (shared format)

One JSON document per session, `schema_version: 1`:

```json
{
  "schema_version": 1,
  "session_id": "7d0c…",
  "client": "local",
  "started_at": "2026-10-02T08:15:00Z",
  "game": {"mode": "random", "difficulty": "HARD", "rounds": 4, "seed": 12},
  "hands": {"backend": "hands", "playing_hand": "right"},
  "capture": {"width": 1280, "height": 720, "effective_fps": 17.8},
  "events": [{"t_ms": 0, "frame": 0, "type": "session_started", "data": {}}],
  "summary": {"total_ms": 183400, "rounds": [], "placements": [], "pairs": [], "quality": {}}
}
```

`summary` is derived and can always be recomputed from `events`, so analysis changes don't invalidate old runs.

### Where it lives

- **Local:** `runs/<started_at>_<mode>_<difficulty>.json` plus the summary PNG. The `runs/` folder is git-ignored. This replaces the single `game_results.json`, which is overwritten on every run.
- **Web:** finished runs (metadata and summary, without the event log) are stored **in the browser's localStorage** (up to 200 runs), and the progression view reads runs of the same game mode and difficulty from there. The server keeps nothing after the connection closes; Fly machines lose their disk on stop and redeploy, and server-side storage would need accounts and a privacy policy. **Export runs** downloads them as JSON; **Clear history** deletes them.
- **Raw keypoints** (optional, local only): a per-frame track of the playing hand's landmarks in a compressed `.npz` next to the session file, at about 250 bytes per frame (roughly 2 MB for 5 minutes at 30 fps). Off by default; enabled with a flag. The web app stores only derived features, never per-frame landmarks.

No video or images are stored by either client.

## 6. Decisions

- **Finger pairs:** no enforced schedule. Analysis uses the detected pair; only when the finger-pair rule is on does a move belong to the selected pair.
- **Storage:** kept simple. JSON files locally, localStorage in the browser, no server-side storage.
- **Usable moves:** a move needs the hand in at least 60% of its frames (`MIN_HAND_COVERAGE` in `analysis.py`) for search, grasp, dwell, and finger features. This threshold still needs checking against labeled recordings (see [evaluation protocols](../training-evaluation/protocols.md)).

## 7. Not implemented yet

- Raw per-frame keypoint tracks (`.npz`, local only).
- Importing exported runs back into a browser.
- Validation of grip and phase detection against annotated video.
