"""Five-circle gameplay geometry anchored to the tracked head center."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EXPECTED_CIRCLE_ORDER: tuple[str, ...] = ("1", "2", "3", "4", "5")


@dataclass(frozen=True)
class CircleGeometry:
    """One numbered gameplay circle positioned in source-frame coordinates."""

    lane: int
    center_xy: tuple[float, float]
    radius: int
    offset_xy: tuple[float, float]


def compute_circle_geometries(
    *,
    head_center_xy: tuple[float, float] | None,
    circle_offsets: dict[str, list[float] | tuple[float, float]],
    circle_radius: int,
) -> tuple[CircleGeometry, ...]:
    """Compute the five fixed gameplay circles anchored to the tracked head center."""

    if head_center_xy is None:
        return ()

    normalized_offsets = _normalize_circle_offsets(circle_offsets)
    head_x, head_y = float(head_center_xy[0]), float(head_center_xy[1])
    radius = max(1, int(circle_radius))
    geometries: list[CircleGeometry] = []

    for lane_key in EXPECTED_CIRCLE_ORDER:
        offset_x, offset_y = normalized_offsets[lane_key]
        geometries.append(
            CircleGeometry(
                lane=int(lane_key),
                center_xy=(head_x + offset_x, head_y + offset_y),
                radius=radius,
                offset_xy=(offset_x, offset_y),
            )
        )

    return tuple(geometries)


def _normalize_circle_offsets(
    circle_offsets: dict[str, list[float] | tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    missing = [lane for lane in EXPECTED_CIRCLE_ORDER if lane not in circle_offsets]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing circle offsets for lane(s): {joined}")

    normalized: dict[str, tuple[float, float]] = {}
    for lane_key in EXPECTED_CIRCLE_ORDER:
        raw_offset: Any = circle_offsets[lane_key]
        if not isinstance(raw_offset, (list, tuple)) or len(raw_offset) != 2:
            raise ValueError(f"Circle offset for lane {lane_key} must be a 2-item list or tuple.")
        normalized[lane_key] = (float(raw_offset[0]), float(raw_offset[1]))
    return normalized
