"""Liveview crop extraction, scaling, and overlay rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from midi_mover.interaction_visuals import CircleVisualState, CircleVisualStyle
from midi_mover.keypoint_overlay import (
    draw_stage1_full_keypoints,
    draw_stage2_full_keypoints,
)
from midi_mover.pose import GameplayKeypoints, PrimaryPersonSelection


@dataclass(frozen=True)
class CropRect:
    """A pixel-aligned crop rectangle within a source frame."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class LiveviewLayout:
    """Computed crop and scaled output size for the left gameplay panel."""

    crop: CropRect
    panel_width: int
    panel_height: int
    scaled_width: int
    scaled_height: int
    blit_x: int
    blit_y: int
    visible_width: int
    source_offset_x: int


@dataclass(frozen=True)
class SmoothedCropState:
    """Persisted crop rectangle used to stabilize the liveview framing across frames."""

    crop: CropRect


class CropSmoother:
    """Smooth crop movement while keeping large player motion responsive."""

    def __init__(self, *, smoothing_factor: float, max_jump_ratio: float) -> None:
        self._smoothing_factor = _clamp01(smoothing_factor)
        self._max_jump_ratio = max(0.0, float(max_jump_ratio))
        self._state: SmoothedCropState | None = None

    @property
    def state(self) -> SmoothedCropState | None:
        return self._state

    def reset(self) -> None:
        self._state = None

    def smooth(
        self,
        *,
        target_crop: CropRect,
        frame_width: int,
        frame_height: int,
    ) -> CropRect:
        if self._state is None:
            self._state = SmoothedCropState(crop=target_crop)
            return target_crop

        previous = self._state.crop
        jump_threshold = max(frame_width, frame_height) * self._max_jump_ratio
        center_jump = _crop_center_distance(previous, target_crop)

        if center_jump > jump_threshold:
            smoothed = target_crop
        else:
            alpha = self._smoothing_factor
            smoothed = CropRect(
                x=int(round(_lerp(previous.x, target_crop.x, alpha))),
                y=int(round(_lerp(previous.y, target_crop.y, alpha))),
                width=max(1, int(round(_lerp(previous.width, target_crop.width, alpha)))),
                height=max(1, int(round(_lerp(previous.height, target_crop.height, alpha)))),
            )

        clamped = clamp_crop_rect(smoothed, frame_width=frame_width, frame_height=frame_height)
        self._state = SmoothedCropState(crop=clamped)
        return clamped


def compute_liveview_layout(
    *,
    frame_width: int,
    frame_height: int,
    selection: PrimaryPersonSelection | None,
    gameplay_keypoints: GameplayKeypoints | None,
    eye_center_x_pct: float,
    eye_center_y_pct: float,
    crop_width_eye_dist: float,
    target_panel_width: int,
    target_panel_height: int,
) -> LiveviewLayout:
    """Compute an eye-centered crop and full-height scaled size.

    ``eye_center_x_pct`` / ``eye_center_y_pct`` define where within the
    resulting crop the detected eye center should appear (0.0–1.0 fractions
    of crop width / height).  ``crop_width_eye_dist`` sets the total crop
    width as a multiple of the inter-eye distance; smaller → tighter crop
    (bigger head in frame), larger → wider crop (smaller head).  The crop
    height is derived from the panel aspect ratio so the image always fills
    the panel without clipping.

    When eye landmarks are unavailable, the full frame is used as a safe
    fallback.
    """
    target_aspect_ratio = max(1e-6, float(target_panel_width) / max(1, float(target_panel_height)))
    crop = compute_person_crop(
        frame_width=frame_width,
        frame_height=frame_height,
        selection=selection,
        gameplay_keypoints=gameplay_keypoints,
        eye_center_x_pct=eye_center_x_pct,
        eye_center_y_pct=eye_center_y_pct,
        crop_width_eye_dist=crop_width_eye_dist,
        target_aspect_ratio=target_aspect_ratio,
    )
    return compute_liveview_layout_for_crop(
        crop=crop,
        target_panel_width=target_panel_width,
        target_panel_height=target_panel_height,
    )


def compute_liveview_layout_for_crop(
    *,
    crop: CropRect,
    target_panel_width: int,
    target_panel_height: int,
) -> LiveviewLayout:
    """Compute scaled liveview placement for an already-selected crop rectangle."""

    scaled_width, scaled_height = scale_crop_to_height(
        crop_width=crop.width,
        crop_height=crop.height,
        target_height=target_panel_height,
    )
    panel_width = max(1, int(target_panel_width))
    panel_height = max(1, int(target_panel_height))

    if scaled_width <= panel_width:
        blit_x = (panel_width - scaled_width) // 2
        visible_width = scaled_width
        source_offset_x = 0
    else:
        blit_x = 0
        visible_width = panel_width
        source_offset_x = (scaled_width - panel_width) // 2

    return LiveviewLayout(
        crop=crop,
        panel_width=panel_width,
        panel_height=panel_height,
        scaled_width=scaled_width,
        scaled_height=scaled_height,
        blit_x=blit_x,
        blit_y=0,
        visible_width=visible_width,
        source_offset_x=source_offset_x,
    )


def compute_person_crop(
    *,
    frame_width: int,
    frame_height: int,
    selection: PrimaryPersonSelection | None,
    gameplay_keypoints: GameplayKeypoints | None,
    eye_center_x_pct: float,
    eye_center_y_pct: float,
    crop_width_eye_dist: float,
    target_aspect_ratio: float,
) -> CropRect:
    """Return a clamped crop rectangle driven by the detected eye landmarks.

    Crop dimensions are derived as follows:

    * ``crop_width_eye_dist`` — total crop width as a multiple of the
      inter-eye distance::

          crop_width = crop_width_eye_dist * eye_distance

      Smaller value → tighter crop (bigger head).  Larger → wider crop.

    * ``target_aspect_ratio`` (= panel_width / panel_height) — the crop
      height is derived from the width so the crop matches the panel aspect
      ratio exactly::

          crop_height = crop_width / target_aspect_ratio

    * The crop origin is placed so the eye centre lands at the requested
      anchor fractions within the crop::

          left = eye_center_x - eye_center_x_pct * crop_width
          top  = eye_center_y - eye_center_y_pct * crop_height

    When eye landmarks are unavailable the full camera frame is returned
    as a safe fallback.
    """

    safe_frame_width = max(1, int(frame_width))
    safe_frame_height = max(1, int(frame_height))

    if selection is None:
        return CropRect(x=0, y=0, width=safe_frame_width, height=safe_frame_height)

    eye_center = getattr(gameplay_keypoints, "head_center_xy", None)
    left_eye = getattr(gameplay_keypoints, "left_eye", None)
    right_eye = getattr(gameplay_keypoints, "right_eye", None)
    if eye_center is not None and left_eye is not None and right_eye is not None:
        eye_distance = max(1.0, abs(float(right_eye.xy[0]) - float(left_eye.xy[0])))

        safe_x_pct = max(0.01, min(0.99, float(eye_center_x_pct)))
        safe_y_pct = max(0.01, min(0.99, float(eye_center_y_pct)))
        safe_w_mult = max(0.1, float(crop_width_eye_dist))
        safe_ar = max(1e-6, float(target_aspect_ratio))

        # Crop width = multiplier × eye_distance.
        width = max(1, int(round(safe_w_mult * eye_distance)))

        # Crop height derived from width and panel aspect ratio.
        height = max(1, int(round(width / safe_ar)))

        # Crop origin so the eye centre lands at the anchor fractions.
        left = int(round(float(eye_center[0]) - safe_x_pct * width))
        top = int(round(float(eye_center[1]) - safe_y_pct * height))

        return clamp_crop_rect(
            CropRect(x=left, y=top, width=width, height=height),
            frame_width=safe_frame_width,
            frame_height=safe_frame_height,
        )

    return CropRect(x=0, y=0, width=safe_frame_width, height=safe_frame_height)


def clamp_crop_rect(crop: CropRect, *, frame_width: int, frame_height: int) -> CropRect:
    """Clamp a crop rectangle so it remains fully inside the source frame."""

    safe_frame_width = max(1, int(frame_width))
    safe_frame_height = max(1, int(frame_height))
    width = max(1, int(crop.width))
    height = max(1, int(crop.height))
    left = int(crop.x)
    top = int(crop.y)

    if width >= safe_frame_width:
        left = 0
        width = safe_frame_width
    else:
        left = max(0, min(left, safe_frame_width - width))

    if height >= safe_frame_height:
        top = 0
        height = safe_frame_height
    else:
        top = max(0, min(top, safe_frame_height - height))

    return CropRect(x=left, y=top, width=width, height=height)


def crop_camera_frame(frame: Any, crop: CropRect, pygame_module: Any) -> Any:
    """Extract a cropped CameraFrame-like payload with an updated render surface."""

    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("Liveview cropping requires numpy in the active environment.") from exc

    x1 = crop.x
    y1 = crop.y
    x2 = crop.x + crop.width
    y2 = crop.y + crop.height

    bgr_crop = frame.bgr_frame[y1:y2, x1:x2].copy()
    rgb_crop = frame.rgb_frame[y1:y2, x1:x2].copy()
    contiguous_rgb = np.ascontiguousarray(rgb_crop)
    render_surface = pygame_module.image.frombuffer(
        contiguous_rgb.tobytes(),
        (contiguous_rgb.shape[1], contiguous_rgb.shape[0]),
        "RGB",
    )

    return type(frame)(
        bgr_frame=bgr_crop,
        rgb_frame=contiguous_rgb,
        render_surface=render_surface,
        width=int(contiguous_rgb.shape[1]),
        height=int(contiguous_rgb.shape[0]),
        mirrored=frame.mirrored,
    )


def scale_crop_to_height(*, crop_width: int, crop_height: int, target_height: int) -> tuple[int, int]:
    """Scale a crop to a fixed height while preserving aspect ratio."""

    safe_crop_width = max(1, int(crop_width))
    safe_crop_height = max(1, int(crop_height))
    safe_target_height = max(1, int(target_height))
    scale = safe_target_height / safe_crop_height
    scaled_width = max(1, int(round(safe_crop_width * scale)))
    return scaled_width, safe_target_height


def draw_liveview_overlay(
    surface: Any,
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    selection: PrimaryPersonSelection | None,
    gameplay_keypoints: Any,
    circle_geometries: tuple[Any, ...],
    circle_visual_states: tuple[CircleVisualState, ...] = (),
    circle_visual_styles: dict[str, CircleVisualStyle] | None = None,
    pygame_module: Any,
    crop_origin: tuple[int, int] = (0, 0),
    circle_stroke_width: int = 2,
    label_font_size: int = 24,
    show_head_center_marker: bool = False,
    show_wrist_markers: bool = True,
    wrist_marker_radius: int = 10,
    wrist_marker_outline_width: int = 2,
    show_stage1_full_keypoints: bool = False,
    stage1_keypoints_xy: tuple[tuple[float, float], ...] = (),
    stage1_keypoints_conf: tuple[float, ...] = (),
    stage1_full_keypoints_style: dict[str, Any] | None = None,
    show_stage2_hand_keypoints: bool = False,
    stage2_hand_keypoints_xy: tuple[tuple[tuple[float, float], ...], ...] = (),
    stage2_hand_keypoints_conf: tuple[tuple[float, ...], ...] = (),
    stage2_hand_keypoints_style: dict[str, Any] | None = None,
    show_keypoint_overlay_legend: bool = False,
    show_overlay_confidence_values: bool = True,
    keypoint_overlay_legend_style: dict[str, Any] | None = None,
) -> None:
    """Draw the tracked person box, head center, circles, and lane labels on a liveview surface."""

    if selection is None:
        return

    x1, y1, x2, y2 = selection.candidate.bbox_xyxy
    crop_x, crop_y = crop_origin
    x1 -= crop_x
    x2 -= crop_x
    y1 -= crop_y
    y2 -= crop_y
    scale_x = target_width / max(source_width, 1)
    scale_y = target_height / max(source_height, 1)
    rect = pygame_module.Rect(
        int(x1 * scale_x),
        int(y1 * scale_y),
        max(1, int((x2 - x1) * scale_x)),
        max(1, int((y2 - y1) * scale_y)),
    )
    pygame_module.draw.rect(surface, (96, 165, 250), rect, width=3)

    head_center_xy = getattr(gameplay_keypoints, "head_center_xy", None)
    if show_head_center_marker and head_center_xy is not None:
        head_x = int((head_center_xy[0] - crop_x) * scale_x)
        head_y = int((head_center_xy[1] - crop_y) * scale_y)
        pygame_module.draw.circle(surface, (250, 204, 21), (head_x, head_y), 6)

    if show_wrist_markers:
        _draw_wrist_markers(
            surface,
            pygame_module,
            gameplay_keypoints=gameplay_keypoints,
            crop_origin=crop_origin,
            scale_x=scale_x,
            scale_y=scale_y,
            wrist_marker_radius=wrist_marker_radius,
            wrist_marker_outline_width=wrist_marker_outline_width,
        )

    if show_stage1_full_keypoints and stage1_keypoints_xy:
        draw_stage1_full_keypoints(
            surface,
            pygame_module,
            keypoints_xy=stage1_keypoints_xy,
            keypoints_conf=stage1_keypoints_conf,
            crop_origin=crop_origin,
            scale_x=scale_x,
            scale_y=scale_y,
            style=stage1_full_keypoints_style,
        )

    if show_stage2_hand_keypoints and stage2_hand_keypoints_xy:
        draw_stage2_full_keypoints(
            surface,
            pygame_module,
            hand_keypoints_xy=stage2_hand_keypoints_xy,
            hand_keypoints_conf=stage2_hand_keypoints_conf,
            crop_origin=crop_origin,
            scale_x=scale_x,
            scale_y=scale_y,
            style=stage2_hand_keypoints_style,
        )

    if not circle_geometries:
        return

    font = _get_overlay_font(pygame_module, label_font_size)
    scaled_stroke_width = max(1, int(round(circle_stroke_width * min(scale_x, scale_y))))
    # Key by (hand, lane) to support separate L and R circles per lane.
    circle_states_by_key = {(state.hand, state.lane): state for state in circle_visual_states}
    # Default styles differ per hand for visual distinction.
    default_style_L = CircleVisualStyle(
        outline_color=(96, 165, 250),   # blue — left hand
        fill_color=(0, 0, 0),
        label_color=(255, 255, 255),
    )
    default_style_R = CircleVisualStyle(
        outline_color=(52, 211, 153),   # green — right hand
        fill_color=(0, 0, 0),
        label_color=(255, 255, 255),
    )

    for circle in circle_geometries:
        circle_x = int((circle.center_xy[0] - crop_x) * scale_x)
        circle_y = int((circle.center_xy[1] - crop_y) * scale_y)
        circle_radius = max(1, int(circle.radius * min(scale_x, scale_y)))
        hand = getattr(circle, "hand", "")
        visual_state = circle_states_by_key.get((hand, circle.lane))
        state_name = visual_state.state_name if visual_state is not None else "idle"
        default_style = default_style_L if hand == "L" else default_style_R
        style = default_style
        if circle_visual_styles is not None:
            style = circle_visual_styles.get(state_name, default_style)
        fill_rect = pygame_module.Rect(
            circle_x - circle_radius,
            circle_y - circle_radius,
            circle_radius * 2,
            circle_radius * 2,
        )
        pygame_module.draw.ellipse(surface, style.fill_color, fill_rect)
        pygame_module.draw.circle(
            surface,
            style.outline_color,
            (circle_x, circle_y),
            circle_radius,
            width=scaled_stroke_width,
        )
        # Label shows the full token (e.g. "L3" or "R5") so both the hand
        # side and the lane number are immediately readable in the overlay.
        token_label = f"{hand}{circle.lane}" if hand else str(circle.lane)
        _draw_circle_label(
            surface,
            pygame_module,
            font,
            label=token_label,
            center_xy=(circle_x, circle_y),
            circle_radius=circle_radius,
            label_color=style.label_color,
        )

    if show_keypoint_overlay_legend:
        _draw_keypoint_overlay_legend(
            surface,
            pygame_module,
            show_stage1_full_keypoints=show_stage1_full_keypoints,
            show_stage2_hand_keypoints=show_stage2_hand_keypoints,
            stage1_keypoints_conf=stage1_keypoints_conf,
            stage2_hand_keypoints_conf=stage2_hand_keypoints_conf,
            show_overlay_confidence_values=show_overlay_confidence_values,
            style=keypoint_overlay_legend_style,
        )


def _get_overlay_font(pygame_module: Any, label_font_size: int) -> Any:
    pygame_module.font.init()
    return pygame_module.font.Font(None, max(12, int(label_font_size)))


def _draw_wrist_markers(
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
    font = _get_overlay_font(pygame_module, max(14, wrist_marker_radius * 2))
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


def _draw_circle_label(
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


def _draw_keypoint_overlay_legend(
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

    font = _get_overlay_font(pygame_module, font_size)
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
        stage1_avg_conf = _average_confidence(stage1_keypoints_conf)
        stage2_avg_conf = _average_confidence(
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


def _average_confidence(values: tuple[float, ...]) -> float | None:
    filtered = [float(value) for value in values if float(value) > 0.0]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _lerp(start: int | float, end: int | float, alpha: float) -> float:
    return float(start) + (float(end) - float(start)) * float(alpha)


def _crop_center_distance(first: CropRect, second: CropRect) -> float:
    first_center_x = first.x + first.width / 2.0
    first_center_y = first.y + first.height / 2.0
    second_center_x = second.x + second.width / 2.0
    second_center_y = second.y + second.height / 2.0
    dx = second_center_x - first_center_x
    dy = second_center_y - first_center_y
    return (dx * dx + dy * dy) ** 0.5


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))