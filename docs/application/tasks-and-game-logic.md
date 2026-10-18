# Application tasks and game logic

## Current task: cube stacking

**Implemented.** `CubeStackingTask` consumes cube detections and, when hand tracking is on, the playing hand's landmarks. Each frame it:

1. Detects a pinch grip.
2. Turns cube displacements into moves.
3. Applies the optional finger-pair rule.
4. Hands everything to the active game engine.

The task owns sequence-local state through the `VisionTask` lifecycle.

### Game modes

| Mode | Engine | Goal |
|---|---|---|
| Simple stacking (`--game simple`) | `GameEngine` | Easy/medium: any stack of 2/3 cubes. Hard: a fixed 4-cube order. Expert: each valid order in turn. A stack counts after being held still. |
| Random stacking (`--game random`) | `RandomStackingEngine` | Each round samples a random color order. Build it bottom to top, then remove cubes top first until only the base is left. Default 4 rounds (`--rounds`); `--seed` makes the orders reproducible. Difficulty sets the cube count: easy 2, medium 3, hard/expert 4. |

Random stacking shows the target as a colored tower on screen. Each step is accepted after it is stable for 10 frames. A hint line explains wrong cubes and rejected moves. Score is 100 points per completed round.

### Grips, moves, and the finger-pair rule

`grasp.py` provides three pieces:

- **`detect_grip`:** a grip means the thumb tip and another fingertip are within a contact margin of the same cube. The finger nearest the cube's outline is taken as the gripping finger. This is a 2D heuristic; curled fingers resting over a cube can be mistaken for the gripping finger.
- **`MoveTracker`:** a move starts when a cube leaves its resting place by more than 0.6 of its size (or reappears elsewhere after being hidden in the hand). It ends after 6 still frames. Its pair is the majority grip seen during the move.
- **`FingerPairRule`:** an optional rule requiring thumb+index, thumb+middle, thumb+ring or thumb+pinky. While it is on, a cube moved with another pair doesn't count. In Simple stacking, a stack counts only if every cube above the base was placed correctly. In Random stacking, the step must be redone. The rule is off by default:
  - **Local:** `--finger-rule` starts it on, `g` toggles it, and `f` cycles the pair.
  - **Web:** the "Finger pairs" and "Next pair" buttons.
  - It needs hand tracking.

Every move is logged with its phase times, detected pair, finger features, required pair, and whether it counted. These logs feed [session analysis](../analysis/session-analysis.md).

## Future tasks

Keep new workflows as separate tasks when they have distinct goals, state, outputs, or evaluation. Share perception backends and observation types where their meaning is genuinely the same; avoid mixing clinical interpretations into game rules.

## Game logic boundary

Game rules decide success, progress, scoring, and session transitions from task-relevant state. They should not load models or access camera APIs. Document difficulty-specific rules and edge cases alongside their tests (`tests/test_game_modes.py`).

The local camera/video presentation and the web client are described under [interfaces](interfaces/index.md).
