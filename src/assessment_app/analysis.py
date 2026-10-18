"""Session analysis: turn a session record (metadata + event log) into timing and finger-pair summaries.

All times are milliseconds on the session clock. See docs/analysis/session-analysis.md for definitions."""

from __future__ import annotations

from statistics import median
from typing import Iterable

from .grasp import FINGERS, pair_label

MIN_HAND_COVERAGE = 0.6
LINK_SLACK_MS = 1000.0


def _median(values: Iterable[float | None]) -> float | None:
    values = [value for value in values if value is not None]
    return round(median(values), 1) if values else None


def _diff(later: float | None, earlier: float | None) -> float | None:
    if later is None or earlier is None:
        return None
    return round(later - earlier, 1)


def _of_type(events, *types):
    return [event for event in events if event["type"] in types]


def _attach_releases(moves: list[dict], releases: list[dict]) -> None:
    """A release belongs to the latest move of that color that settled before it."""
    for release in releases:
        candidates = [
            move
            for move in moves
            if move.get("release_ms") is None
            and move["color"] == release["data"]["color"]
            and (move["settle_ms"] or move["end_ms"] or 0) <= release["t_ms"]
        ]
        if candidates:
            candidates[-1]["release_ms"] = release["t_ms"]


def _moves(events) -> list[dict]:
    moves = [dict(event["data"]) for event in _of_type(events, "cube_moved")]
    for move in moves:
        move.setdefault("release_ms", None)
    _attach_releases(moves, _of_type(events, "cube_released"))
    return moves


def _pair(move: dict | None) -> tuple[str | None, str | None]:
    """With the finger-pair rule on, a move belongs to the selected pair; otherwise to the detected one."""
    if move is None:
        return None, None
    if move.get("required"):
        return move["required"], "required"
    if move.get("finger"):
        return pair_label(move["finger"]), "detected"
    return None, None


def _confident(move: dict | None, hands: bool) -> bool:
    if not hands or move is None:
        return False
    return (move.get("coverage") or 0.0) >= MIN_HAND_COVERAGE


def _phases(move: dict | None, reference: float | None) -> dict:
    if move is None:
        return {"search_ms": None, "grasp_ms": None, "transport_ms": None, "dwell_ms": None}
    return {
        "search_ms": _diff(move.get("grip_onset_ms"), reference),
        "grasp_ms": _diff(move.get("lift_ms"), move.get("grip_onset_ms")),
        "transport_ms": _diff(move.get("settle_ms"), move.get("lift_ms")),
        "dwell_ms": _diff(move.get("release_ms"), move.get("settle_ms")),
    }


def _move_item(move: dict | None, hands: bool, reference: float | None) -> dict:
    pair, source = _pair(move)
    confident = _confident(move, hands)
    phases = _phases(move, reference)
    if not confident:
        phases["search_ms"] = phases["grasp_ms"] = phases["dwell_ms"] = None
    return {
        "has_move": move is not None,
        "pair": pair,
        "pair_source": source,
        "counted": None if move is None else move.get("counted"),
        "coverage": None if move is None else move.get("coverage"),
        "low_confidence": hands and not confident,
        "aperture": None if move is None or not confident else move.get("aperture_mean"),
        "path_length": None if move is None or not confident else move.get("path_length"),
        "peak_speed": None if move is None or not confident else move.get("peak_speed"),
        **phases,
    }


def _link(moves: list[dict], color: str, reference: float, achieved: float) -> dict | None:
    """The move that produced a step: the latest move of that color between the step's reference and its time."""
    candidates = [
        move
        for move in moves
        if move["color"] == color
        and (move.get("lift_ms") or move.get("end_ms") or 0) >= reference - LINK_SLACK_MS
        and (move.get("settle_ms") or move.get("end_ms") or 0) <= achieved + LINK_SLACK_MS
    ]
    return candidates[-1] if candidates else None


def _random(events, moves, hands) -> dict:
    rounds, steps = [], []
    current, reference = None, None
    for event in events:
        kind, data, t = event["type"], event["data"], event["t_ms"]
        if kind == "round_started":
            current = {
                "round": data["round"],
                "target": data["target"],
                "start_ms": t,
                "stack_ms": None,
                "unstack_ms": None,
                "round_ms": None,
                "wrong_cubes": 0,
                "rejected_moves": 0,
            }
            rounds.append(current)
            reference = t
        elif current is None:
            continue
        elif kind == "step_completed":
            achieved = data["achieved_ms"]
            move = _link(moves, data["color"], reference, achieved)
            steps.append(
                {
                    "round": data["round"],
                    "phase": data["phase"],
                    "index": data["index"],
                    "color": data["color"],
                    "step_ms": _diff(achieved, reference),
                    **_move_item(move, hands, reference),
                }
            )
            reference = achieved
        elif kind == "stack_completed":
            current["stack_ms"] = _diff(data["achieved_ms"], current["start_ms"])
            current["unstack_start_ms"] = data["confirmed_ms"]
            reference = data["confirmed_ms"]
        elif kind == "round_completed":
            current["round_ms"] = _diff(data["achieved_ms"], current["start_ms"])
            current["unstack_ms"] = _diff(data["achieved_ms"], current.get("unstack_start_ms"))
        elif kind == "wrong_cube":
            current["wrong_cubes"] += 1
        elif kind == "cube_moved" and data.get("counted") is False:
            current["rejected_moves"] += 1
    for item in rounds:
        item.pop("unstack_start_ms", None)
    completed = [item["round_ms"] for item in rounds if item["round_ms"] is not None]
    return {
        "rounds": rounds,
        "steps": steps,
        "round_ms_median": _median(completed),
        "best_round_ms": min(completed) if completed else None,
    }


def _simple(events, moves, hands, start) -> dict:
    stacks = [
        {
            "colors": event["data"]["colors"],
            "achieved_ms": _diff(event["data"]["achieved_ms"], start),
            "hold_ms": _diff(event["data"]["confirmed_ms"], event["data"]["achieved_ms"]),
        }
        for event in _of_type(events, "stack_confirmed")
    ]
    items = [
        {"color": move["color"], "step_ms": None, **_move_item(move, hands, None)}
        for move in moves
    ]
    return {"stacks": stacks, "steps": items}


def _pairs(steps: list[dict], moves: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for step in steps:
        if step["pair"]:
            groups.setdefault(step["pair"], []).append(step)
    rejected: dict[str, int] = {}
    for move in moves:
        if move.get("counted") is False and move.get("required"):
            rejected[move["required"]] = rejected.get(move["required"], 0) + 1
    result = []
    order = {pair_label(finger): i for i, finger in enumerate(FINGERS)}
    for pair in sorted(set(groups) | set(rejected), key=lambda name: order.get(name, len(order))):
        items = groups.get(pair, [])
        result.append(
            {
                "pair": pair,
                "n": len(items),
                "step_ms": _median(item["step_ms"] for item in items),
                "transport_ms": _median(item["transport_ms"] for item in items),
                "dwell_ms": _median(item["dwell_ms"] for item in items),
                "aperture": _median(item["aperture"] for item in items),
                "rejected": rejected.get(pair, 0),
                "step_values": [item["step_ms"] for item in items if item["step_ms"] is not None],
            }
        )
    return result


def _quality(events, moves, steps, hands) -> dict:
    samples = [event["data"] for event in _of_type(events, "quality")]
    coverage = [sample["hand_coverage"] for sample in samples if sample.get("hand_coverage") is not None]
    return {
        "fps": _median(sample["fps"] for sample in samples),
        "hand_coverage": round(sum(coverage) / len(coverage), 3) if coverage else None,
        "moves": len(moves),
        "moves_without_grip": sum(1 for move in moves if not move.get("finger")) if hands else None,
        "low_confidence_steps": sum(1 for step in steps if step.get("low_confidence")),
        "steps_without_move": sum(1 for step in steps if "phase" in step and not step["has_move"]),
    }


def _screwing(events) -> dict:
    finished = _of_type(events, "game_completed")
    reached = _of_type(events, "depth_reached")
    data = dict(finished[-1]["data"] if finished else reached[-1]["data"] if reached else {})
    return {
        "stacks": [],
        "steps": [],
        "setup_ms": data.get("setup_ms"),
        "pickup_ms": data.get("pickup_ms"),
        "screw_ms": data.get("screw_ms"),
        "unscrew_ms": data.get("unscrew_ms"),
        "putdown_ms": data.get("putdown_ms"),
        "turns": data.get("turns"),
        "side_steps": data.get("side_steps"),
        "depth": data.get("depth"),
        "target": data.get("target"),
        # Times the nut moved along the thread with no fingertip on it.
        "spins": data.get("spins", len(_of_type(events, "nut_spun"))),
    }


def summarize(record: dict) -> dict:
    events = record["events"]
    hands = bool(record.get("hands", {}).get("enabled"))
    mode = record["game"]["mode"]
    started = _of_type(events, "game_started")
    start = started[0]["t_ms"] if started else (events[0]["t_ms"] if events else 0.0)
    finished = _of_type(events, "game_completed")
    last_t = events[-1]["t_ms"] if events else start
    moves = _moves(events)

    if mode == "screwing":
        detail = _screwing(events)
        end = finished[-1]["t_ms"] if finished else None
    elif mode == "random":
        detail = _random(events, moves, hands)
        rounds_done = [item for item in detail["rounds"] if item["round_ms"] is not None]
        end = max((item["start_ms"] + item["round_ms"] for item in rounds_done), default=None)
    else:
        detail = _simple(events, moves, hands, start)
        end = start + detail["stacks"][-1]["achieved_ms"] if detail["stacks"] else None

    total = _diff(end, start) if finished else _diff(end if end is not None else last_t, start)
    rule_used = any(event["data"].get("enabled") for event in _of_type(events, "rule_changed"))
    return {
        "mode": mode,
        "completed": bool(finished),
        "total_ms": total,
        "score": finished[-1]["data"].get("final_score") if finished else None,
        "finger_rule_used": rule_used,
        **detail,
        "pairs": _pairs(detail["steps"], moves),
        "quality": _quality(events, moves, detail["steps"], hands),
    }


def with_summary(record: dict) -> dict:
    return {**record, "summary": summarize(record)}


def format_summary(summary: dict) -> str:
    """Plain-text summary for consoles and logs."""

    def sec(ms):
        return "-" if ms is None else f"{ms / 1000:.1f}s"

    lines = [
        f"Mode: {summary['mode']}  completed: {summary['completed']}  total: {sec(summary['total_ms'])}"
    ]
    if summary.get("mode") == "screwing":
        depth = summary.get("depth")
        target = summary.get("target")
        lines.append(
            f"  Pick up {sec(summary.get('pickup_ms'))}, screw on {sec(summary.get('screw_ms'))}, "
            f"unscrew {sec(summary.get('unscrew_ms'))}, put down {sec(summary.get('putdown_ms'))}, "
            f"depth {'-' if depth is None else f'{depth:.0%}'}, "
            f"target {'-' if target is None else f'{target:.0%}'}, "
            f"turns {summary.get('turns')}, sides {summary.get('side_steps')}, "
            f"spins {summary.get('spins') or 0}"
        )
    for item in summary.get("rounds", []):
        target = "-".join(color[0].upper() for color in item["target"])
        lines.append(
            f"  Round {item['round']} {target}: stack {sec(item['stack_ms'])}, "
            f"unstack {sec(item['unstack_ms'])}, total {sec(item['round_ms'])}, "
            f"wrong cubes {item['wrong_cubes']}, rejected {item['rejected_moves']}"
        )
    for item in summary.get("stacks", []):
        lines.append(f"  Stack {'-'.join(item['colors'])} at {sec(item['achieved_ms'])}")
    for pair in summary["pairs"]:
        lines.append(
            f"  {pair['pair']}: n={pair['n']}, step {sec(pair['step_ms'])}, "
            f"transport {sec(pair['transport_ms'])}, dwell {sec(pair['dwell_ms'])}, "
            f"rejected {pair['rejected']}"
        )
    quality = summary["quality"]
    lines.append(
        f"  Quality: fps {quality['fps']}, hand coverage {quality['hand_coverage']}, "
        f"moves {quality['moves']}, low-confidence steps {quality['low_confidence_steps']}"
    )
    return "\n".join(lines)
