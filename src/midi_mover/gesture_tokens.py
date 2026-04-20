"""Helpers for dynamic gameplay lane/token spaces (3/4/5 circles per hand)."""

from __future__ import annotations


SUPPORTED_CIRCLE_COUNTS: tuple[int, ...] = (3, 4, 5)


def normalize_circles_per_hand(value: object) -> int:
    """Return a validated circles-per-hand value."""
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"circles_per_hand must be an integer in {SUPPORTED_CIRCLE_COUNTS}, got {value!r}."
        ) from exc
    if normalized not in SUPPORTED_CIRCLE_COUNTS:
        raise ValueError(
            f"circles_per_hand must be one of {SUPPORTED_CIRCLE_COUNTS}, got {normalized}."
        )
    return normalized


def build_lane_keys(circles_per_hand: int) -> tuple[str, ...]:
    """Return lane labels as strings ('1'..'N')."""
    count = normalize_circles_per_hand(circles_per_hand)
    return tuple(str(lane) for lane in range(1, count + 1))


def build_gesture_tokens(circles_per_hand: int) -> tuple[str, ...]:
    """Return gesture tokens for both hands, e.g. ('L1'..'LN', 'R1'..'RN')."""
    lane_keys = build_lane_keys(circles_per_hand)
    return tuple([*(f"L{lane}" for lane in lane_keys), *(f"R{lane}" for lane in lane_keys)])
