"""Debug overlay helpers for drawing full stage-1 and stage-2 keypoint sets."""

from __future__ import annotations

from typing import Any


def draw_stage1_full_keypoints(
    surface: Any,
    pygame_module: Any,
    *,
    keypoints_xy: tuple[tuple[float, float], ...],
    keypoints_conf: tuple[float, ...],
    crop_origin: tuple[int, int],
    scale_x: float,
    scale_y: float,
    style: dict[str, Any] | None,
) -> None:
    """Draw full-body stage-1 keypoints and skeleton."""

    style_payload = style or {}
    keypoint_color = tuple(style_payload.get("keypoint_color", [147, 197, 253]))
    keypoint_radius = max(1, int(style_payload.get("keypoint_radius", 4)))
    keypoint_outline_color = tuple(style_payload.get("keypoint_outline_color", [30, 64, 175]))
    keypoint_outline_width = max(1, int(style_payload.get("keypoint_outline_width", 1)))
    skeleton_color = tuple(style_payload.get("skeleton_color", [96, 165, 250]))
    skeleton_width = max(1, int(style_payload.get("skeleton_width", 2)))

    _draw_keypoint_skeleton(
        surface,
        pygame_module,
        keypoints_xy=keypoints_xy,
        keypoints_conf=keypoints_conf,
        edges=_COCO_KEYPOINT_EDGES,
        crop_origin=crop_origin,
        scale_x=scale_x,
        scale_y=scale_y,
        color=skeleton_color,
        width=skeleton_width,
    )
    _draw_keypoints(
        surface,
        pygame_module,
        keypoints_xy=keypoints_xy,
        keypoints_conf=keypoints_conf,
        crop_origin=crop_origin,
        scale_x=scale_x,
        scale_y=scale_y,
        color=keypoint_color,
        radius=keypoint_radius,
        outline_color=keypoint_outline_color,
        outline_width=keypoint_outline_width,
    )


def draw_stage2_full_keypoints(
    surface: Any,
    pygame_module: Any,
    *,
    hand_keypoints_xy: tuple[tuple[tuple[float, float], ...], ...],
    hand_keypoints_conf: tuple[tuple[float, ...], ...],
    crop_origin: tuple[int, int],
    scale_x: float,
    scale_y: float,
    style: dict[str, Any] | None,
) -> None:
    """Draw full-hand stage-2 keypoints/skeleton for each detected hand."""

    style_payload = style or {}
    keypoint_color = tuple(style_payload.get("keypoint_color", [196, 181, 253]))
    keypoint_radius = max(1, int(style_payload.get("keypoint_radius", 3)))
    keypoint_outline_color = tuple(style_payload.get("keypoint_outline_color", [91, 33, 182]))
    keypoint_outline_width = max(1, int(style_payload.get("keypoint_outline_width", 1)))
    skeleton_color = tuple(style_payload.get("skeleton_color", [167, 139, 250]))
    skeleton_width = max(1, int(style_payload.get("skeleton_width", 2)))
    emphasize_fingertips = bool(style_payload.get("emphasize_fingertips", False))
    fingertip_color = tuple(style_payload.get("fingertip_color", [244, 114, 182]))
    fingertip_radius = max(1, int(style_payload.get("fingertip_radius", keypoint_radius + 2)))
    fingertip_outline_color = tuple(style_payload.get("fingertip_outline_color", [131, 24, 67]))
    fingertip_outline_width = max(1, int(style_payload.get("fingertip_outline_width", keypoint_outline_width)))

    for hand_index, keypoints_xy in enumerate(hand_keypoints_xy):
        keypoints_conf = hand_keypoints_conf[hand_index] if hand_index < len(hand_keypoints_conf) else ()
        _draw_keypoint_skeleton(
            surface,
            pygame_module,
            keypoints_xy=keypoints_xy,
            keypoints_conf=keypoints_conf,
            edges=_HAND_KEYPOINT_EDGES,
            crop_origin=crop_origin,
            scale_x=scale_x,
            scale_y=scale_y,
            color=skeleton_color,
            width=skeleton_width,
        )
        _draw_keypoints(
            surface,
            pygame_module,
            keypoints_xy=keypoints_xy,
            keypoints_conf=keypoints_conf,
            crop_origin=crop_origin,
            scale_x=scale_x,
            scale_y=scale_y,
            color=keypoint_color,
            radius=keypoint_radius,
            outline_color=keypoint_outline_color,
            outline_width=keypoint_outline_width,
        )

        if not emphasize_fingertips:
            continue

        for fingertip_index in _HAND_FINGERTIP_INDICES:
            if fingertip_index >= len(keypoints_xy):
                continue
            if fingertip_index < len(keypoints_conf) and float(keypoints_conf[fingertip_index]) <= 0.0:
                continue
            x, y = keypoints_xy[fingertip_index]
            screen_x = int((x - crop_origin[0]) * scale_x)
            screen_y = int((y - crop_origin[1]) * scale_y)
            scaled_radius = max(2, int(round(fingertip_radius * min(scale_x, scale_y))))
            scaled_outline = max(1, int(round(fingertip_outline_width * min(scale_x, scale_y))))
            pygame_module.draw.circle(surface, fingertip_color, (screen_x, screen_y), scaled_radius)
            pygame_module.draw.circle(
                surface,
                fingertip_outline_color,
                (screen_x, screen_y),
                scaled_radius,
                width=scaled_outline,
            )


def _draw_keypoint_skeleton(
    surface: Any,
    pygame_module: Any,
    *,
    keypoints_xy: tuple[tuple[float, float], ...],
    keypoints_conf: tuple[float, ...],
    edges: tuple[tuple[int, int], ...],
    crop_origin: tuple[int, int],
    scale_x: float,
    scale_y: float,
    color: tuple[int, int, int],
    width: int,
) -> None:
    scaled_width = max(1, int(round(width * min(scale_x, scale_y))))
    for start_index, end_index in edges:
        if start_index >= len(keypoints_xy) or end_index >= len(keypoints_xy):
            continue
        if start_index < len(keypoints_conf) and float(keypoints_conf[start_index]) <= 0.0:
            continue
        if end_index < len(keypoints_conf) and float(keypoints_conf[end_index]) <= 0.0:
            continue
        start_xy = keypoints_xy[start_index]
        end_xy = keypoints_xy[end_index]
        start = (
            int((start_xy[0] - crop_origin[0]) * scale_x),
            int((start_xy[1] - crop_origin[1]) * scale_y),
        )
        end = (
            int((end_xy[0] - crop_origin[0]) * scale_x),
            int((end_xy[1] - crop_origin[1]) * scale_y),
        )
        pygame_module.draw.line(surface, color, start, end, width=scaled_width)


def _draw_keypoints(
    surface: Any,
    pygame_module: Any,
    *,
    keypoints_xy: tuple[tuple[float, float], ...],
    keypoints_conf: tuple[float, ...],
    crop_origin: tuple[int, int],
    scale_x: float,
    scale_y: float,
    color: tuple[int, int, int],
    radius: int,
    outline_color: tuple[int, int, int],
    outline_width: int,
) -> None:
    scaled_radius = max(2, int(round(radius * min(scale_x, scale_y))))
    scaled_outline = max(1, int(round(outline_width * min(scale_x, scale_y))))
    for index, keypoint_xy in enumerate(keypoints_xy):
        if index < len(keypoints_conf) and float(keypoints_conf[index]) <= 0.0:
            continue
        screen_x = int((keypoint_xy[0] - crop_origin[0]) * scale_x)
        screen_y = int((keypoint_xy[1] - crop_origin[1]) * scale_y)
        pygame_module.draw.circle(surface, color, (screen_x, screen_y), scaled_radius)
        pygame_module.draw.circle(
            surface,
            outline_color,
            (screen_x, screen_y),
            scaled_radius,
            width=scaled_outline,
        )


_COCO_KEYPOINT_EDGES: tuple[tuple[int, int], ...] = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (1, 2),
    (3, 5),
    (4, 6),
)

_HAND_KEYPOINT_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

_HAND_FINGERTIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)
