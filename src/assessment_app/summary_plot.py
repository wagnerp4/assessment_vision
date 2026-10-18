"""Render a session summary figure (local summary window and saved PNG). Requires matplotlib."""

from __future__ import annotations

import numpy as np

CUBE_RGB = {"blue": "#2f6fdf", "green": "#2e9e4f", "red": "#d93a3a", "yellow": "#e6c229"}
PHASES = (
    ("search_ms", "search", "#9aa7b8"),
    ("grasp_ms", "grasp", "#6c8ebf"),
    ("transport_ms", "transport", "#3f6fb0"),
    ("dwell_ms", "dwell", "#e39b3b"),
)


def _sec(ms):
    return "-" if ms is None else f"{ms / 1000:.1f}s"


def _clock(ms):
    if ms is None:
        return "-"
    minutes, seconds = divmod(ms / 1000, 60)
    return f"{int(minutes)}:{seconds:04.1f}"


def _header(record: dict, summary: dict) -> str:
    game, quality = record["game"], summary["quality"]
    parts = [
        f"{'Random' if summary['mode'] == 'random' else 'Simple'} stacking",
        game["difficulty"],
        f"total {_clock(summary['total_ms'])}",
    ]
    if summary["mode"] == "random":
        done = sum(1 for item in summary["rounds"] if item["round_ms"] is not None)
        parts.append(f"{done}/{game['rounds']} rounds")
    if quality["fps"]:
        parts.append(f"{quality['fps']:.0f} fps")
    if quality["hand_coverage"] is not None:
        parts.append(f"hand seen {quality['hand_coverage'] * 100:.0f}%")
    return "   |   ".join(parts)


def _rounds(ax, rounds):
    labels = []
    for row, item in enumerate(rounds):
        stack, unstack = item["stack_ms"] or 0, item["unstack_ms"] or 0
        ax.barh(row, stack / 1000, color="#3f6fb0", label="stack" if row == 0 else None)
        ax.barh(row, unstack / 1000, left=stack / 1000, color="#e39b3b", label="unstack" if row == 0 else None)
        if item["round_ms"] is not None:
            ax.text((stack + unstack) / 1000, row, f"  {_sec(item['round_ms'])}", va="center", fontsize=9)
        labels.append(f"R{item['round']} " + "-".join(color[0].upper() for color in item["target"]))
    ax.set_yticks(range(len(rounds)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("seconds")
    longest = max(((item["stack_ms"] or 0) + (item["unstack_ms"] or 0) for item in rounds), default=0)
    ax.set_xlim(0, max(longest / 1000 * 1.18, 1))
    ax.set_title("Rounds: stack (blue) + unstack (orange)", loc="left", fontsize=11)


def _steps_by_index(ax, steps):
    columns = sorted({(step["phase"], step["index"]) for step in steps}, key=lambda key: (key[0] != "stack", key[1] if key[0] == "stack" else -key[1]))
    names = [("S" if phase == "stack" else "U") + str(index) for phase, index in columns]
    for x, key in enumerate(columns):
        values = [step["step_ms"] / 1000 for step in steps if (step["phase"], step["index"]) == key and step["step_ms"] is not None]
        colors = [CUBE_RGB.get(step["color"], "#888") for step in steps if (step["phase"], step["index"]) == key and step["step_ms"] is not None]
        ax.scatter([x] * len(values), values, c=colors, s=36, edgecolors="#333", linewidths=0.5, zorder=3)
        if values:
            ax.hlines(float(np.median(values)), x - 0.3, x + 0.3, color="#333", linewidth=2)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("seconds")
    ax.set_title("Step time by tower position (S = place, U = remove; line = median)", loc="left", fontsize=11)


def _phases(ax, steps):
    rows = [step for step in steps if any(step.get(key) is not None for key, _, _ in PHASES)]
    if not rows:
        ax.text(0.5, 0.5, "No hand-tracked steps", ha="center", va="center", transform=ax.transAxes, color="#777")
        ax.set_axis_off()
        return
    for x, step in enumerate(rows):
        bottom = 0.0
        for key, label, color in PHASES:
            value = (step.get(key) or 0) / 1000
            ax.bar(x, value, bottom=bottom, color=color, label=label if x == 0 else None)
            bottom += value
    ax.set_xticks(
        range(len(rows)),
        [f"{step.get('round', '')}{'S' if step.get('phase') == 'stack' else 'U'}{step.get('index', '')}" for step in rows],
        fontsize=7,
        rotation=90,
    )
    ax.set_ylabel("seconds")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    ax.set_title("Move phases per step (hand-tracked, >= 60% coverage)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")


def _pairs(ax, pairs):
    ax.set_axis_off()
    ax.set_title("Finger pairs", loc="left", fontsize=11)
    if not pairs:
        ax.text(0.0, 0.8, "No finger-pair data (hand tracking off or no pinch detected)", color="#777", transform=ax.transAxes)
        return
    cells = [
        [pair["pair"], pair["n"], _sec(pair["step_ms"]), _sec(pair["transport_ms"]), _sec(pair["dwell_ms"]), pair["rejected"]]
        for pair in pairs
    ]
    table = ax.table(
        cellText=cells,
        colLabels=["pair", "n", "step", "transport", "dwell", "rejected"],
        loc="upper left",
        cellLoc="center",
        colWidths=[0.26, 0.08, 0.16, 0.18, 0.16, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)


def render_summary(record: dict, summary: dict) -> np.ndarray:
    """Return the summary figure as a BGR image."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(13, 7.5), dpi=100)
    FigureCanvasAgg(fig)
    fig.suptitle(_header(record, summary), x=0.01, ha="left", fontsize=13, fontweight="bold")
    grid = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.22, left=0.07, right=0.98, top=0.9, bottom=0.1)
    if summary["mode"] == "screwing":
        ax = fig.add_subplot(grid[0, :])
        ax.set_axis_off()
        depth = summary.get("depth")
        target = summary.get("target")
        lines = [
            f"Pick up: {_sec(summary.get('pickup_ms'))}",
            f"Screw on: {_sec(summary.get('screw_ms'))}",
            f"Unscrew: {_sec(summary.get('unscrew_ms'))}",
            f"Put down: {_sec(summary.get('putdown_ms'))}",
            f"Depth: {'-' if depth is None else f'{depth:.0%}'}   target {'-' if target is None else f'{target:.0%}'}",
            f"Turns: {summary.get('turns')}    sides: {summary.get('side_steps')}",
        ]
        ax.text(0.0, 0.9, "\n".join(lines), va="top", fontsize=14, transform=ax.transAxes)
    elif summary["mode"] == "random":
        _rounds(fig.add_subplot(grid[0, 0]), summary["rounds"])
        _steps_by_index(fig.add_subplot(grid[0, 1]), summary["steps"])
    else:
        ax = fig.add_subplot(grid[0, :])
        ax.set_axis_off()
        lines = [f"Time to first counted stack: {_sec(summary['stacks'][0]['achieved_ms']) if summary['stacks'] else '-'}"]
        lines += [f"Stack {'-'.join(item['colors'])} at {_sec(item['achieved_ms'])} (held {_sec(item['hold_ms'])})" for item in summary["stacks"]]
        ax.text(0.0, 0.9, "\n".join(lines), va="top", fontsize=12, transform=ax.transAxes)
    _phases(fig.add_subplot(grid[1, 0]), summary["steps"])
    _pairs(fig.add_subplot(grid[1, 1]), summary["pairs"])
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[:, :, 2::-1])
