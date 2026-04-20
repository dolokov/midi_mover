"""Utility helpers for liveview overlay rendering and interpolation math."""

from __future__ import annotations

from typing import Any


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def lerp(start: int | float, end: int | float, alpha: float) -> float:
    return float(start) + (float(end) - float(start)) * float(alpha)


def crop_center_distance(first: Any, second: Any) -> float:
    first_center_x = first.x + first.width / 2.0
    first_center_y = first.y + first.height / 2.0
    second_center_x = second.x + second.width / 2.0
    second_center_y = second.y + second.height / 2.0
    dx = second_center_x - first_center_x
    dy = second_center_y - first_center_y
    return (dx * dx + dy * dy) ** 0.5


def get_overlay_font(pygame_module: Any, label_font_size: int) -> Any:
    pygame_module.font.init()
    return pygame_module.font.Font(None, max(12, int(label_font_size)))


def blend_rgb(
    base: tuple[int, int, int],
    overlay: tuple[int, int, int],
    alpha: float,
) -> tuple[int, int, int]:
    t = clamp01(alpha)
    return lerp_color(base, overlay, t)


def lerp_color(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    clamped_t = clamp01(t)
    return (
        int(round(lerp(start[0], end[0], clamped_t))),
        int(round(lerp(start[1], end[1], clamped_t))),
        int(round(lerp(start[2], end[2], clamped_t))),
    )


def draw_wrist_markers(
    surface: Any,
    pygame_module: Any,
    *,
    gameplay_keypoints: Any,
    crop_origin: tuple[int, int],
    scale_x: float,
    scale_y: float,
    wrist_marker_radius: int,
    wrist_marker_outline_width: int,
) -> None:
    marker_specs = (
        (getattr(gameplay_keypoints, "left_wrist", None), "L", (248, 113, 113), (127, 29, 29)),
        (getattr(gameplay_keypoints, "right_wrist", None), "R", (52, 211, 153), (6, 78, 59)),
    )
    crop_x, crop_y = crop_origin
    font = get_overlay_font(pygame_module, max(14, wrist_marker_radius * 2))
    scaled_radius = max(4, int(round(wrist_marker_radius * min(scale_x, scale_y))))
    scaled_outline = max(1, int(round(wrist_marker_outline_width * min(scale_x, scale_y))))

    for sample, label, fill_color, outline_color in marker_specs:
        if sample is None:
            continue
        marker_x = int((sample.xy[0] - crop_x) * scale_x)
        marker_y = int((sample.xy[1] - crop_y) * scale_y)
        pygame_module.draw.circle(surface, fill_color, (marker_x, marker_y), scaled_radius)
        pygame_module.draw.circle(
            surface,
            outline_color,
            (marker_x, marker_y),
            scaled_radius,
            width=scaled_outline,
        )
        text_surface = font.render(label, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(marker_x, marker_y))
        surface.blit(text_surface, text_rect)


def draw_circle_label(
    surface: Any,
    pygame_module: Any,
    font: Any,
    *,
    label: str,
    center_xy: tuple[int, int],
    circle_radius: int,
    label_color: tuple[int, int, int],
) -> None:
    text_surface = font.render(label, True, label_color)
    text_rect = text_surface.get_rect(center=center_xy)
    padding = max(4, circle_radius // 6)
    background_rect = text_rect.inflate(padding * 2, padding)
    pygame_module.draw.rect(surface, (15, 23, 42), background_rect, border_radius=max(6, padding))
    surface.blit(text_surface, text_rect)


def draw_keypoint_overlay_legend(
    surface: Any,
    pygame_module: Any,
    *,
    show_stage1_full_keypoints: bool,
    show_stage2_hand_keypoints: bool,
    stage1_keypoints_conf: tuple[float, ...],
    stage2_hand_keypoints_conf: tuple[tuple[float, ...], ...],
    show_overlay_confidence_values: bool,
    style: dict[str, Any] | None,
) -> None:
    style_payload = style or {}
    font_size = int(style_payload.get("font_size", 18))
    text_color = tuple(style_payload.get("text_color", [226, 232, 240]))
    muted_text_color = tuple(style_payload.get("muted_text_color", [148, 163, 184]))
    background_color = tuple(style_payload.get("background_color", [2, 6, 23]))
    border_color = tuple(style_payload.get("border_color", [51, 65, 85]))
    border_width = max(0, int(style_payload.get("border_width", 1)))
    panel_padding = max(2, int(style_payload.get("panel_padding", 8)))
    line_spacing = max(0, int(style_payload.get("line_spacing", 4)))

    font = get_overlay_font(pygame_module, font_size)
    lines = [
        ("Keypoint overlays", text_color),
        (
            f"Stage-1 full body: {'ON' if show_stage1_full_keypoints else 'OFF'}",
            text_color if show_stage1_full_keypoints else muted_text_color,
        ),
        (
            f"Stage-2 hand: {'ON' if show_stage2_hand_keypoints else 'OFF'}",
            text_color if show_stage2_hand_keypoints else muted_text_color,
        ),
        (
            f"Confidence text: {'ON' if show_overlay_confidence_values else 'OFF'}",
            text_color if show_overlay_confidence_values else muted_text_color,
        ),
    ]

    if show_overlay_confidence_values:
        stage1_avg_conf = average_confidence(stage1_keypoints_conf)
        stage2_avg_conf = average_confidence(
            tuple(value for row in stage2_hand_keypoints_conf for value in row)
        )
        lines.append(
            (
                f"Stage-1 avg conf: {stage1_avg_conf:.2f}" if stage1_avg_conf is not None else "Stage-1 avg conf: n/a",
                muted_text_color,
            )
        )
        lines.append(
            (
                f"Stage-2 avg conf: {stage2_avg_conf:.2f}" if stage2_avg_conf is not None else "Stage-2 avg conf: n/a",
                muted_text_color,
            )
        )

    rendered = [font.render(text, True, color) for text, color in lines]
    max_width = max((item.get_width() for item in rendered), default=0)
    content_height = sum(item.get_height() for item in rendered)
    content_height += line_spacing * max(0, len(rendered) - 1)
    panel_rect = pygame_module.Rect(
        panel_padding,
        panel_padding,
        max_width + panel_padding * 2,
        content_height + panel_padding * 2,
    )
    pygame_module.draw.rect(surface, background_color, panel_rect, border_radius=8)
    if border_width > 0:
        pygame_module.draw.rect(
            surface,
            border_color,
            panel_rect,
            width=border_width,
            border_radius=8,
        )

    text_y = panel_rect.y + panel_padding
    for item in rendered:
        surface.blit(item, (panel_rect.x + panel_padding, text_y))
        text_y += item.get_height() + line_spacing


def average_confidence(values: tuple[float, ...]) -> float | None:
    filtered = [float(value) for value in values if float(value) > 0.0]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)
