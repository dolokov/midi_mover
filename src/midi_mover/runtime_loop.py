"""Persistent liveview runtime loop and per-frame rendering helpers."""

from __future__ import annotations

import logging
from typing import Any

from midi_mover.audio import GesturePlaybackController, LoadedGestureSounds
from midi_mover.camera import CameraFrameError, CameraFrameReader
from midi_mover.circles import compute_circle_geometries
from midi_mover.interaction_visuals import (
    CircleVisualStateTracker,
    CircleVisualStyle,
    HandCircleTransitionTracker,
    describe_transition_snapshot,
    detect_hand_circle_interactions,
)
from midi_mover.liveview import (
    CropSmoother,
    compute_liveview_layout,
    compute_liveview_layout_for_crop,
    crop_camera_frame,
    draw_liveview_overlay,
)
from midi_mover.pose import (
    GameplayKeypointTracker,
    PoseProcessingError,
    PrimaryPersonSelection,
    PrimaryPersonTracker,
    run_pose_inference,
)

LOGGER = logging.getLogger("midi_mover")


class LiveviewRuntimeError(RuntimeError):
    """Raised when the persistent liveview runtime cannot continue safely."""


def load_circle_visual_styles(config: Any) -> dict[str, CircleVisualStyle]:
    visuals = config.raw["liveview"]["circle_visuals"]
    return {
        state_name: CircleVisualStyle(
            outline_color=tuple(visuals[state_name]["outline_color"]),
            fill_color=tuple(visuals[state_name]["fill_color"]),
            label_color=tuple(visuals[state_name]["label_color"]),
        )
        for state_name in ("idle", "active_contact", "hit_flash", "miss_flash")
    }


def render_liveview_frame(
    *,
    window: Any,
    pygame_module: Any,
    frame_reader: CameraFrameReader,
    pose_model: Any,
    primary_person_tracker: PrimaryPersonTracker,
    gameplay_keypoint_tracker: GameplayKeypointTracker,
    interaction_transition_tracker: HandCircleTransitionTracker,
    crop_smoother: CropSmoother,
    circle_visual_tracker: CircleVisualStateTracker,
    config: Any,
    gesture_sounds: LoadedGestureSounds | None = None,
    gesture_playback_controller: GesturePlaybackController | None = None,
) -> None:
    if window is None or pygame_module is None:
        raise LiveviewRuntimeError("Liveview rendering failed: pygame window/module was not initialized.")

    padding_color = tuple(config.raw["liveview"]["padding_color"])
    circle_visual_styles = load_circle_visual_styles(config)
    left_panel_ratio = float(config.raw["liveview"]["left_panel_ratio"])
    left_panel_width = int(window.get_width() * left_panel_ratio)
    left_panel_height = window.get_height()

    try:
        frame = frame_reader.read(pygame_module)
    except CameraFrameError as exc:
        raise LiveviewRuntimeError(f"Liveview rendering failed while reading camera frame: {exc}") from exc

    try:
        pose_result = run_pose_inference(
            pose_model,
            frame.bgr_frame,
            conf=float(config.raw["pose"]["confidence_threshold"]),
            iou=float(config.raw["pose"]["iou_threshold"]),
        )
    except PoseProcessingError as exc:
        raise LiveviewRuntimeError(f"Liveview rendering failed during pose inference: {exc}") from exc

    selection = primary_person_tracker.select(pose_result)
    gameplay_keypoints = gameplay_keypoint_tracker.extract(pose_result, selection)
    circle_geometries = compute_circle_geometries(
        head_center_xy=getattr(gameplay_keypoints, "head_center_xy", None),
        gameplay_keypoints=gameplay_keypoints,
        frame_width=frame.width,
        frame_height=frame.height,
        circle_offsets_percent=config.raw["liveview"]["circle_offsets_percent"],
        circle_radius_percent=float(config.raw["liveview"]["circle_radius_percent"]),
        scale_width_multiplier=float(config.raw["liveview"]["eye_crop_width_multiplier"]),
    )
    interaction_snapshot = detect_hand_circle_interactions(
        circle_geometries=circle_geometries,
        gameplay_keypoints=gameplay_keypoints,
        swap_hands=frame.mirrored,
    )
    transition_snapshot = interaction_transition_tracker.update(interaction_snapshot)
    if gesture_sounds is not None and gesture_playback_controller is not None:
        gesture_playback_controller.update(
            pygame_module=pygame_module,
            transition_snapshot=transition_snapshot,
            gesture_sounds=gesture_sounds,
        )
    circle_visual_states = circle_visual_tracker.update(
        circle_geometries=circle_geometries,
        gameplay_keypoints=gameplay_keypoints,
        interaction_snapshot=interaction_snapshot,
    )
    target_layout = compute_liveview_layout(
        frame_width=frame.width,
        frame_height=frame.height,
        selection=selection,
        gameplay_keypoints=gameplay_keypoints,
        eye_target_x_ratio=float(config.raw["liveview"]["eye_target_x_ratio"]),
        eye_target_y_ratio=float(config.raw["liveview"]["eye_target_y_ratio"]),
        eye_crop_width_multiplier=float(config.raw["liveview"]["eye_crop_width_multiplier"]),
        eye_crop_above_multiplier=float(config.raw["liveview"]["eye_crop_above_multiplier"]),
        eye_crop_below_multiplier=float(config.raw["liveview"]["eye_crop_below_multiplier"]),
        target_panel_width=left_panel_width,
        target_panel_height=left_panel_height,
    )
    smoothed_crop = crop_smoother.smooth(
        target_crop=target_layout.crop,
        frame_width=frame.width,
        frame_height=frame.height,
    )
    cropped_frame = crop_camera_frame(frame, smoothed_crop, pygame_module)
    layout = compute_liveview_layout_for_crop(
        crop=smoothed_crop,
        target_panel_width=left_panel_width,
        target_panel_height=left_panel_height,
    )

    window.fill((0, 0, 0))
    left_panel_rect = pygame_module.Rect(0, 0, left_panel_width, left_panel_height)
    window.fill(padding_color, left_panel_rect)

    scaled_surface = pygame_module.transform.smoothscale(
        cropped_frame.render_surface,
        (layout.scaled_width, layout.scaled_height),
    )
    draw_liveview_overlay(
        scaled_surface,
        source_width=cropped_frame.width,
        source_height=cropped_frame.height,
        target_width=layout.scaled_width,
        target_height=layout.scaled_height,
        selection=selection,
        gameplay_keypoints=gameplay_keypoints,
        circle_geometries=circle_geometries,
        circle_visual_states=circle_visual_states,
        circle_visual_styles=circle_visual_styles,
        pygame_module=pygame_module,
        crop_origin=(layout.crop.x, layout.crop.y),
        circle_stroke_width=int(config.raw["liveview"]["circle_stroke_width"]),
        label_font_size=int(config.raw["liveview"]["label_font_size"]),
        show_head_center_marker=bool(config.raw["liveview"]["debug"]["show_head_center"]),
        show_wrist_markers=bool(config.raw["liveview"]["debug"]["show_wrist_markers"]),
        wrist_marker_radius=int(config.raw["liveview"]["wrist_marker_radius"]),
        wrist_marker_outline_width=int(config.raw["liveview"]["wrist_marker_outline_width"]),
    )

    if layout.visible_width < layout.scaled_width:
        visible_surface = scaled_surface.subsurface(
            pygame_module.Rect(
                layout.source_offset_x,
                0,
                layout.visible_width,
                layout.scaled_height,
            )
        )
    else:
        visible_surface = scaled_surface

    window.blit(visible_surface, (layout.blit_x, layout.blit_y))
    pygame_module.display.flip()
    _log_frame_summary(
        frame=frame,
        layout=layout,
        left_panel_width=left_panel_width,
        left_panel_height=left_panel_height,
        selection=selection,
        gameplay_keypoint_tracker=gameplay_keypoint_tracker,
        gameplay_keypoints=gameplay_keypoints,
        transition_snapshot=transition_snapshot,
    )


def run_persistent_liveview_loop(
    *,
    window: Any,
    pygame_module: Any,
    frame_reader: CameraFrameReader,
    pose_model: Any,
    primary_person_tracker: PrimaryPersonTracker,
    gameplay_keypoint_tracker: GameplayKeypointTracker,
    interaction_transition_tracker: HandCircleTransitionTracker,
    crop_smoother: CropSmoother,
    circle_visual_tracker: CircleVisualStateTracker,
    config: Any,
    gesture_sounds: LoadedGestureSounds | None = None,
    gesture_playback_controller: GesturePlaybackController | None = None,
) -> None:
    target_fps = max(1, int(config.raw["app"]["target_fps"]))
    clock = pygame_module.time.Clock()
    LOGGER.info(
        "Starting persistent liveview loop. Press ESC or close the window to exit. target_fps=%s",
        target_fps,
    )
    running = True
    while running:
        for event in pygame_module.event.get():
            if event.type == pygame_module.QUIT:
                LOGGER.info("Received pygame QUIT event. Exiting persistent liveview loop.")
                running = False
                break
            if event.type == pygame_module.KEYDOWN and event.key == pygame_module.K_ESCAPE:
                LOGGER.info("Received ESC key input. Exiting persistent liveview loop.")
                running = False
                break

        if not running:
            continue

        render_liveview_frame(
            window=window,
            pygame_module=pygame_module,
            frame_reader=frame_reader,
            pose_model=pose_model,
            primary_person_tracker=primary_person_tracker,
            gameplay_keypoint_tracker=gameplay_keypoint_tracker,
            interaction_transition_tracker=interaction_transition_tracker,
            crop_smoother=crop_smoother,
            circle_visual_tracker=circle_visual_tracker,
            config=config,
            gesture_sounds=gesture_sounds,
            gesture_playback_controller=gesture_playback_controller,
        )
        clock.tick(target_fps)


def _log_frame_summary(
    *,
    frame: Any,
    layout: Any,
    left_panel_width: int,
    left_panel_height: int,
    selection: PrimaryPersonSelection | None,
    gameplay_keypoint_tracker: GameplayKeypointTracker,
    gameplay_keypoints: Any,
    transition_snapshot: Any,
) -> None:
    LOGGER.info(
        "Prepared person-centered liveview crop: frame=%sx%s crop=(x=%s y=%s w=%s h=%s) scaled=%sx%s visible_width=%s blit=(%s,%s) source_offset_x=%s mirrored=%s left_panel=%sx%s primary_person=%s keypoints=%s transitions=%s.",
        frame.width,
        frame.height,
        layout.crop.x,
        layout.crop.y,
        layout.crop.width,
        layout.crop.height,
        layout.scaled_width,
        layout.scaled_height,
        layout.visible_width,
        layout.blit_x,
        layout.blit_y,
        layout.source_offset_x,
        frame.mirrored,
        left_panel_width,
        left_panel_height,
        _format_primary_person_log(selection),
        gameplay_keypoint_tracker.describe(gameplay_keypoints),
        describe_transition_snapshot(transition_snapshot),
    )


def _format_primary_person_log(selection: PrimaryPersonSelection | None) -> str:
    if selection is None:
        return "none"
    candidate = selection.candidate
    return (
        f"index={candidate.index} track_id={candidate.track_id} area={candidate.area:.1f} "
        f"confidence={candidate.confidence:.3f} reason={selection.reason}"
    )