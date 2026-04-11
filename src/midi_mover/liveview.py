"""Liveview crop extraction, scaling, and overlay rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from midi_mover.interaction_visuals import CircleVisualState, CircleVisualStyle
from midi_mover.pose import PrimaryPersonSelection


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
    crop_margin_ratio: float,
    target_panel_width: int,
    target_panel_height: int,
) -> LiveviewLayout:
    """Compute a person-centered crop and full-height scaled size.

    The crop is centered on the selected player's bounding box, expanded by the
    configured margin ratio, and clamped to the source frame bounds. When no
    player is currently selected, the full frame is used as a safe fallback.
    """

    crop = compute_person_crop(
        frame_width=frame_width,
        frame_height=frame_height,
        selection=selection,
        crop_margin_ratio=crop_margin_ratio,
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
    crop_margin_ratio: float,
) -> CropRect:
    """Return a clamped crop rectangle centered on the primary person."""

    safe_frame_width = max(1, int(frame_width))
    safe_frame_height = max(1, int(frame_height))

    if selection is None:
        return CropRect(x=0, y=0, width=safe_frame_width, height=safe_frame_height)

    x1, y1, x2, y2 = selection.candidate.bbox_xyxy
    bbox_width = max(1.0, float(x2) - float(x1))
    bbox_height = max(1.0, float(y2) - float(y1))
    margin_ratio = max(0.0, float(crop_margin_ratio))

    expanded_width = bbox_width * (1.0 + margin_ratio * 2.0)
    expanded_height = bbox_height * (1.0 + margin_ratio * 2.0)

    center_x = (float(x1) + float(x2)) / 2.0
    center_y = (float(y1) + float(y2)) / 2.0

    left = int(round(center_x - expanded_width / 2.0))
    top = int(round(center_y - expanded_height / 2.0))
    width = max(1, int(round(expanded_width)))
    height = max(1, int(round(expanded_height)))

    return clamp_crop_rect(
        CropRect(x=left, y=top, width=width, height=height),
        frame_width=safe_frame_width,
        frame_height=safe_frame_height,
    )


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

    if not circle_geometries:
        return

    font = _get_overlay_font(pygame_module, label_font_size)
    scaled_stroke_width = max(1, int(round(circle_stroke_width * min(scale_x, scale_y))))
    circle_states_by_lane = {state.lane: state for state in circle_visual_states}
    default_style = CircleVisualStyle(
        outline_color=(96, 165, 250),
        fill_color=(0, 0, 0),
        label_color=(255, 255, 255),
    )

    for circle in circle_geometries:
        circle_x = int((circle.center_xy[0] - crop_x) * scale_x)
        circle_y = int((circle.center_xy[1] - crop_y) * scale_y)
        circle_radius = max(1, int(circle.radius * min(scale_x, scale_y)))
        visual_state = circle_states_by_lane.get(circle.lane)
        state_name = visual_state.state_name if visual_state is not None else "idle"
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
        _draw_circle_label(
            surface,
            pygame_module,
            font,
            label=str(circle.lane),
            center_xy=(circle_x, circle_y),
            circle_radius=circle_radius,
            label_color=style.label_color,
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