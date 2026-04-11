"""Five-circle gameplay geometry anchored to the tracked head center."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from midi_mover.pose import GameplayKeypoints


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
    gameplay_keypoints: GameplayKeypoints | None,
    frame_width: int,
    frame_height: int,
    circle_offsets_percent: dict[str, list[float] | tuple[float, float]],
    circle_radius_percent: float,
    scale_width_multiplier: float = 12.0,
) -> tuple[CircleGeometry, ...]:
    """Compute the five fixed gameplay circles anchored to the tracked head center.

    When eye landmarks are available, circle spacing is scaled from the current
    eye distance so the overlay stays visually tied to the player's head even
    as they move closer to or farther from the camera. If eye landmarks are not
    available, fall back to the historic full-frame percentage behavior.
    """

    if head_center_xy is None:
        return ()

    normalized_offsets = _normalize_circle_offsets(circle_offsets_percent)
    head_x, head_y = float(head_center_xy[0]), float(head_center_xy[1])
    base_dimension = _resolve_circle_scale_basis(
        gameplay_keypoints=gameplay_keypoints,
        frame_width=frame_width,
        frame_height=frame_height,
        scale_width_multiplier=scale_width_multiplier,
    )
    pixels_per_percent = base_dimension / 100.0
    radius = max(1, int(round(float(circle_radius_percent) * pixels_per_percent)))
    geometries: list[CircleGeometry] = []

    for lane_key in EXPECTED_CIRCLE_ORDER:
        offset_x_percent, offset_y_percent = normalized_offsets[lane_key]
        offset_x = offset_x_percent * pixels_per_percent
        offset_y = offset_y_percent * pixels_per_percent
        geometries.append(
            CircleGeometry(
                lane=int(lane_key),
                center_xy=(head_x + offset_x, head_y + offset_y),
                radius=radius,
                offset_xy=(offset_x, offset_y),
            )
        )

    return tuple(geometries)


def _resolve_circle_scale_basis(
    *,
    gameplay_keypoints: GameplayKeypoints | None,
    frame_width: int,
    frame_height: int,
    scale_width_multiplier: float,
) -> float:
    if gameplay_keypoints is not None:
        left_eye = gameplay_keypoints.left_eye
        right_eye = gameplay_keypoints.right_eye
        if left_eye is not None and right_eye is not None:
            eye_distance = abs(float(right_eye.xy[0]) - float(left_eye.xy[0]))
            if eye_distance > 0.0:
                return max(1.0, eye_distance * max(1.0, float(scale_width_multiplier)))

    return float(min(max(1, int(frame_width)), max(1, int(frame_height))))


def _normalize_circle_offsets(
    circle_offsets_percent: dict[str, list[float] | tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    missing = [lane for lane in EXPECTED_CIRCLE_ORDER if lane not in circle_offsets_percent]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing circle offsets for lane(s): {joined}")

    normalized: dict[str, tuple[float, float]] = {}
    for lane_key in EXPECTED_CIRCLE_ORDER:
        raw_offset: Any = circle_offsets_percent[lane_key]
        if not isinstance(raw_offset, (list, tuple)) or len(raw_offset) != 2:
            raise ValueError(f"Circle offset for lane {lane_key} must be a 2-item list or tuple.")
        normalized[lane_key] = (float(raw_offset[0]), float(raw_offset[1]))
    return normalized
