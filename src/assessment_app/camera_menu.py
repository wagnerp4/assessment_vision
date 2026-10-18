"""Startup menu for camera mode. The choices match the browser client's controls."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .game_logic import GameDifficulty, GameMode
from .grasp import FINGERS, pair_label

GAME_LABELS = {
    GameMode.SIMPLE: "Simple stacking",
    GameMode.RANDOM: "Random stacking",
    GameMode.SCREWING: "Screwing",
}
BACKEND_LABELS = {
    "hands": "Hand model",
    "holistic": "Holistic (needs body in view)",
}
GAME_BLURB = {
    GameMode.SIMPLE: "Build one stack that meets the difficulty goal.",
    GameMode.RANDOM: "Stack a random order, unstack it in reverse, and repeat.",
    GameMode.SCREWING: "Pick up screw and nut, screw to the mark, unscrew, put both down.",
}
DIFFICULTIES = (
    GameDifficulty.EASY,
    GameDifficulty.MEDIUM,
    GameDifficulty.HARD,
    GameDifficulty.EXPERT,
)
# waitKeyEx arrow codes for the Windows and GTK/Qt HighGUI backends.
ARROW_KEYS = {
    2424832: "left",
    2490368: "up",
    2555904: "right",
    2621440: "down",
    65361: "left",
    65362: "up",
    65363: "right",
    65364: "down",
    0xFF51: "left",
    0xFF52: "up",
    0xFF53: "right",
    0xFF54: "down",
}


@dataclass
class MenuButton:
    id: str
    rect: tuple[int, int, int, int]
    enabled: bool


def key_name(key: int) -> str | None:
    """Map a waitKeyEx code to a menu action name."""
    if key is None or key < 0:
        return None
    if key in ARROW_KEYS:
        return ARROW_KEYS[key]
    if key > 255 and (key & ~0xFF) != 0:
        return None
    low = key & 0xFF
    if low in (10, 13):
        return "enter"
    if low == 27:
        return "esc"
    if low == 32:
        return "space"
    if 33 <= low < 127:
        return chr(low).lower()
    return None


class CameraMenu:
    """Game, difficulty, hand tracking, finger pairs, and recording for one live session."""

    def __init__(
        self,
        *,
        game: GameMode = GameMode.SIMPLE,
        difficulty: GameDifficulty = GameDifficulty.EASY,
        hands: bool = False,
        pose_backend: str = "hands",
        finger_rule: bool = False,
        record: bool = False,
        pair_index: int = 0,
    ) -> None:
        if pose_backend not in BACKEND_LABELS:
            raise ValueError("pose_backend must be 'hands' or 'holistic'")
        self.game = game
        self.difficulty = difficulty
        self.hands = hands
        self.pose_backend = pose_backend
        self.finger_rule = finger_rule
        self.record = record
        self.pair_index = pair_index % len(FINGERS)
        self.message = ""
        self.focus = 0
        self.buttons: list[MenuButton] = []

    @property
    def finger_pairs_active(self) -> bool:
        return self.hands and self.finger_rule

    def _rows(self) -> list[dict]:
        return [
            {
                "id": "game",
                "kind": "cycle",
                "label": "Game",
                "text": GAME_LABELS[self.game],
                "enabled": True,
                "on": False,
            },
            {
                "id": "difficulty",
                "kind": "cycle",
                "label": "Difficulty",
                "text": self.difficulty.name.title(),
                "enabled": True,
                "on": False,
            },
            {
                "id": "hands",
                "kind": "toggle",
                "label": "Hand tracking",
                "text": "on" if self.hands else "off",
                "enabled": True,
                "on": self.hands,
            },
            {
                "id": "pose_backend",
                "kind": "cycle",
                "label": "Hand model",
                "text": BACKEND_LABELS[self.pose_backend],
                "enabled": True,
                "on": False,
            },
            {
                "id": "finger_rule",
                "kind": "toggle",
                "label": "Finger pairs",
                "text": "on" if self.finger_pairs_active else "off",
                "enabled": self.hands,
                "on": self.finger_pairs_active,
            },
            {
                "id": "next_pair",
                "kind": "cycle",
                "label": "Next pair",
                "text": pair_label(FINGERS[self.pair_index]),
                "enabled": self.finger_pairs_active,
                "on": False,
            },
            {
                "id": "start",
                "kind": "action",
                "label": "Start game",
                "text": "Start game",
                "enabled": True,
                "on": True,
            },
            {
                "id": "record",
                "kind": "toggle",
                "label": "Record",
                "text": f"Record: {'on' if self.record else 'off'}",
                "enabled": True,
                "on": self.record,
            },
        ]

    def activate(self, control_id: str, direction: int = 1) -> str | None:
        """Apply a click or key to one control. `direction` is -1 or +1 for cycling."""
        row = next(item for item in self._rows() if item["id"] == control_id)
        if not row["enabled"]:
            return None
        step = 1 if direction >= 0 else -1
        if control_id == "start":
            return "start"
        if control_id == "game":
            modes = list(GAME_LABELS)
            self.game = modes[(modes.index(self.game) + step) % len(modes)]
        elif control_id == "difficulty":
            index = DIFFICULTIES.index(self.difficulty)
            self.difficulty = DIFFICULTIES[(index + step) % len(DIFFICULTIES)]
        elif control_id == "hands":
            self.hands = not self.hands
        elif control_id == "pose_backend":
            self.pose_backend = "holistic" if self.pose_backend == "hands" else "hands"
        elif control_id == "finger_rule":
            self.finger_rule = not self.finger_rule
        elif control_id == "next_pair":
            self.pair_index = (self.pair_index + step) % len(FINGERS)
        elif control_id == "record":
            self.record = not self.record
        return None

    def handle_click(self, x: int, y: int, direction: int = 1) -> str | None:
        for button in self.buttons:
            x1, y1, x2, y2 = button.rect
            if x1 <= x < x2 and y1 <= y < y2:
                if button.enabled:
                    self.focus = next(
                        index for index, row in enumerate(self._rows()) if row["id"] == button.id
                    )
                return self.activate(button.id, direction)
        return None

    def handle_key(self, key: int) -> str | None:
        name = key_name(key)
        if name is None:
            return None
        if name in {"q", "esc"}:
            return "quit"
        if name in {"s", "enter"}:
            return "start"
        if name in {"up", "down"}:
            self._move_focus(-1 if name == "up" else 1)
            return None
        if name in {"left", "right", "space"}:
            self._clamp_focus()
            control_id = self._rows()[self.focus]["id"]
            direction = -1 if name == "left" else 1
            return self.activate(control_id, direction)
        return None

    def _move_focus(self, delta: int) -> None:
        enabled = [index for index, row in enumerate(self._rows()) if row["enabled"]]
        if not enabled:
            return
        if self.focus not in enabled:
            self.focus = enabled[0]
            return
        position = enabled.index(self.focus)
        self.focus = enabled[(position + delta) % len(enabled)]

    def _clamp_focus(self) -> None:
        rows = self._rows()
        if not 0 <= self.focus < len(rows) or not rows[self.focus]["enabled"]:
            self._move_focus(1)

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Draw the menu across the top of a copy of the camera frame and record click targets."""
        output = frame.copy()
        height, width = output.shape[:2]
        rows = self._rows()
        options = [row for row in rows if row["id"] not in {"start", "record"}]
        actions = [row for row in rows if row["id"] in {"start", "record"}]
        option_lines = (len(options) + 1) // 2
        line_count = option_lines + (1 if actions else 0)
        message_lines = _wrap(self.message, max(40, width // 14)) if self.message else []
        margin = 8
        panel_w = max(1, width - margin * 2)
        header, row_h, footer = 62, 42, 28 + 18 * len(message_lines)
        panel_h = header + line_count * row_h + footer
        if panel_h > height - margin:
            row_h = max(28, (height - margin - header - footer) // max(line_count, 1))
            panel_h = min(height - margin, header + line_count * row_h + footer)
        x0, y0 = margin, margin
        x1, y1 = x0 + panel_w, y0 + panel_h
        roi = output[y0:y1, x0:x1]
        panel = np.full_like(roi, (22, 22, 22))
        cv2.addWeighted(panel, 0.88, roi, 0.12, 0, roi)
        cv2.rectangle(output, (x0, y0), (x1 - 1, y1 - 1), (90, 90, 90), 1)

        cv2.putText(
            output, "Cube Stacking Assessment", (x0 + 16, y0 + 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA,
        )
        cv2.putText(
            output, GAME_BLURB[self.game], (x0 + 16, y0 + 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA,
        )

        self.buttons = []
        column_w = (panel_w - 36) // 2
        for index, row in enumerate(options):
            column, line = index % 2, index // 2
            left = x0 + 12 + column * (column_w + 12)
            top = y0 + header + line * row_h
            _draw_option(
                output, rows, row, self.focus,
                (left, top + 4, left + column_w, top + row_h - 6),
            )
            self.buttons.append(MenuButton(row["id"], (left, top + 4, left + column_w, top + row_h - 6), row["enabled"]))

        action_top = y0 + header + option_lines * row_h
        for slot, row in enumerate(actions):
            left = x0 + 12 + slot * (column_w + 12)
            rect = (left, action_top + 4, left + column_w, action_top + row_h - 6)
            _draw_button(
                output, rect, row["text"],
                enabled=row["enabled"], on=row["on"], focused=rows.index(row) == self.focus,
                accent="record" if row["id"] == "record" and row["on"] else "start" if row["id"] == "start" else "default",
                center=True,
            )
            self.buttons.append(MenuButton(row["id"], rect, row["enabled"]))

        hint_y = min(y1 - 8 - 18 * len(message_lines), y0 + header + line_count * row_h + 16)
        cv2.putText(
            output, "Arrows select    S or Enter starts    Q quits",
            (x0 + 16, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA,
        )
        for line_index, line in enumerate(message_lines):
            cv2.putText(
                output, line, (x0 + 16, hint_y + 20 + line_index * 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 140, 230), 1, cv2.LINE_AA,
            )
        return output


def _draw_option(image, rows, row, focus, rect) -> None:
    x1, y1, x2, y2 = rect
    color = (220, 220, 220) if row["enabled"] else (110, 110, 110)
    cv2.putText(
        image, row["label"], (x1 + 2, y1 + (y2 - y1) // 2 + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
    )
    value_w = max(90, int((x2 - x1) * 0.58))
    _draw_button(
        image, (x2 - value_w, y1, x2, y2), row["text"],
        enabled=row["enabled"], on=row["on"], focused=rows.index(row) == focus,
    )


def _draw_button(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    text: str,
    *,
    enabled: bool,
    on: bool,
    focused: bool,
    accent: str = "default",
    center: bool = False,
) -> None:
    x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1:
        return
    if not enabled:
        fill, color = (48, 48, 48), (120, 120, 120)
    elif accent == "record":
        fill, color = (36, 36, 150), (255, 255, 255)
    elif accent == "start":
        fill, color = (50, 130, 48), (255, 255, 255)
    elif on:
        fill, color = (48, 112, 36), (255, 255, 255)
    else:
        fill, color = (58, 58, 58), (238, 238, 238)
    cv2.rectangle(image, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (245, 245, 245) if focused else (110, 110, 110), 2 if focused else 1)
    scale = 0.55
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    while tw > max(12, x2 - x1 - 16) and scale > 0.35:
        scale = round(scale - 0.05, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    tx = x1 + max(8, (x2 - x1 - tw) // 2) if center else x1 + 12
    ty = y1 + (y2 - y1 + th) // 2
    cv2.putText(image, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = word if not current else f"{current} {word}"
        if len(trial) > width and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines[:3]
