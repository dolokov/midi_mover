"""Persistent liveview runtime loop and per-frame rendering helpers."""

from __future__ import annotations

from dataclasses import replace
import logging
import time
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
from midi_mover.judgment import GameplayScoreState, GameplayScoreTracker, HitWindowJudge
from midi_mover.liveview import (
    CropSmoother,
    compute_liveview_layout,
    compute_liveview_layout_for_crop,
    crop_camera_frame,
    draw_liveview_overlay,
)
from midi_mover.pose import (
    GameplayKeypointTracker,
    HandRoiInference,
    PoseProcessingError,
    PrimaryPersonSelection,
    PrimaryPersonTracker,
    run_pose_inference,
    run_stage2_hand_inference_on_person_roi,
)
from midi_mover.timeline import (
    compute_timeline_panel_layout,
    draw_timeline_panel_layout,
    draw_upcoming_timeline_notes,
)
from midi_mover.runtime_helpers import (
    compute_precue_hints,
    extract_stage1_keypoints_for_selection,
)

LOGGER = logging.getLogger("midi_mover")
_LAST_STAGE2_HAND_INFERENCE: HandRoiInference | None = None
_LAST_STAGE2_HAND_INFERENCE_AT: float | None = None
def _final_score_state_or_none(gameplay_score_tracker: GameplayScoreTracker | None) -> GameplayScoreState | None:
    if gameplay_score_tracker is None:
        return None
    return gameplay_score_tracker.state


class LiveviewRuntimeError(RuntimeError):
    """Raised when the persistent liveview runtime cannot continue safely."""
def resolve_frame_hand_swap(*, frame_mirrored: bool, config: Any) -> bool:
    """Resolve whether handedness should be swapped for this mirrored frame."""

    return bool(frame_mirrored) and bool(config.raw["gameplay"].get("swap_hands_when_mirrored", False))
def resolve_overlay_gameplay_keypoints(*, gameplay_keypoints: Any, swap_hands: bool) -> Any:
    """Return gameplay keypoints adjusted for overlay wrist labeling when swapping is active."""

    if gameplay_keypoints is None or not swap_hands:
        return gameplay_keypoints
    return replace(
        gameplay_keypoints,
        left_wrist=getattr(gameplay_keypoints, "right_wrist", None),
        right_wrist=getattr(gameplay_keypoints, "left_wrist", None),
    )
def load_circle_visual_styles(config: Any) -> dict[str, CircleVisualStyle]:
    visuals = config.raw["liveview"]["circle_visuals"]
    return {
        state_name: CircleVisualStyle(
            outline_color=tuple(visuals[state_name]["outline_color"]),
            fill_color=tuple(visuals[state_name]["fill_color"]),
            label_color=tuple(visuals[state_name]["label_color"]),
        )
        for state_name in ("idle", "precue", "active_contact", "hit_flash", "miss_flash")
    }
def render_liveview_frame(
    *,
    window: Any,
    pygame_module: Any,
    frame_reader: CameraFrameReader,
    pose_model: Any,
    hand_pose_model: Any,
    primary_person_tracker: PrimaryPersonTracker,
    gameplay_keypoint_tracker: GameplayKeypointTracker,
    interaction_transition_tracker: HandCircleTransitionTracker,
    crop_smoother: CropSmoother,
    circle_visual_tracker: CircleVisualStateTracker,
    config: Any,
    gesture_sounds: LoadedGestureSounds | None = None,
    gesture_playback_controller: GesturePlaybackController | None = None,
    normalized_target_notes: tuple[Any, ...] = (),
    song_started_monotonic: float | None = None,
    pre_song_lead_in_ms: float = 0.0,
    song_speed_multiplier: float = 1.0,
    hit_window_judge: HitWindowJudge | None = None,
    gameplay_score_tracker: GameplayScoreTracker | None = None,
) -> None:
    if window is None or pygame_module is None:
        raise LiveviewRuntimeError("Liveview rendering failed: pygame window/module was not initialized.")

    padding_color = tuple(config.raw["liveview"]["padding_color"])
    circle_visual_styles = load_circle_visual_styles(config)
    left_panel_ratio = float(config.raw["liveview"]["left_panel_ratio"])
    left_panel_width = int(window.get_width() * left_panel_ratio)
    left_panel_height = window.get_height()
    timeline_layout = compute_timeline_panel_layout(
        pygame_module=pygame_module,
        window_width=window.get_width(),
        window_height=window.get_height(),
        lane_count=int(config.raw["liveview"]["circles_per_hand"]),
        now_line_ratio=float(config.raw["gameplay"]["timeline_now_line_ratio"]),
    )

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
    stage1_keypoints_xy, stage1_keypoints_conf = extract_stage1_keypoints_for_selection(
        result=pose_result,
        selection=selection,
    )
    hand_inference = _run_hand_roi_inference(
        hand_pose_model=hand_pose_model,
        frame=frame,
        selection=selection,
        config=config,
    )
    gameplay_keypoints = gameplay_keypoint_tracker.extract(
        pose_result,
        selection,
        hand_inference=hand_inference,
    )
    circle_geometries = compute_circle_geometries(
        head_center_xy=getattr(gameplay_keypoints, "head_center_xy", None),
        gameplay_keypoints=gameplay_keypoints,
        frame_width=frame.width,
        frame_height=frame.height,
        circles_per_hand=int(config.raw["liveview"]["circles_per_hand"]),
        left_circle_offsets_percent_by_count=config.raw["liveview"]["left_circle_offsets_percent_by_count"],
        right_circle_offsets_percent_by_count=config.raw["liveview"]["right_circle_offsets_percent_by_count"],
        circle_radius_percent=float(config.raw["liveview"]["circle_radius_percent"]),
        scale_width_multiplier=float(config.raw["liveview"]["crop_width_eye_dist"]),
    )
    resolved_swap_hands = resolve_frame_hand_swap(
        frame_mirrored=bool(frame.mirrored),
        config=config,
    )
    interaction_snapshot = detect_hand_circle_interactions(
        circle_geometries=circle_geometries,
        gameplay_keypoints=gameplay_keypoints,
        swap_hands=resolved_swap_hands,
    )
    overlay_gameplay_keypoints = resolve_overlay_gameplay_keypoints(
        gameplay_keypoints=gameplay_keypoints,
        swap_hands=resolved_swap_hands,
    )
    LOGGER.debug(
        "Handedness resolution: mirrored=%s swap_hands_when_mirrored=%s resolved_swap=%s.",
        bool(frame.mirrored),
        bool(config.raw["gameplay"].get("swap_hands_when_mirrored", False)),
        resolved_swap_hands,
    )
    transition_snapshot = interaction_transition_tracker.update(interaction_snapshot)
    judged_hit_tokens: tuple[str, ...] = ()
    judged_miss_tokens: tuple[str, ...] = ()
    if hit_window_judge is not None:
        judged_hits = hit_window_judge.register_transition_snapshot(
            transition_snapshot=transition_snapshot,
            song_started_monotonic=song_started_monotonic,
            pre_song_lead_in_ms=pre_song_lead_in_ms,
            song_speed_multiplier=song_speed_multiplier,
        )
        if song_started_monotonic is not None:
            lead_in_seconds = max(0.0, float(pre_song_lead_in_ms) / 1000.0)
            safe_song_speed = max(1e-6, float(song_speed_multiplier))
            song_elapsed_seconds = (
                (time.monotonic() - float(song_started_monotonic) - lead_in_seconds)
                * safe_song_speed
            )
            hit_window_judge.mark_misses_for_song_elapsed_ms(song_elapsed_seconds * 1000.0)
        if gameplay_score_tracker is not None:
            gameplay_score_tracker.register_hits(judged_hits)
            gameplay_score_tracker.sync_total_misses(hit_window_judge.miss_count())
        judged_hit_tokens = tuple(
            sorted({str(hit.token).strip().upper() for hit in judged_hits if str(hit.token).strip()})
        )
        newly_missed_note_ids = hit_window_judge.consume_newly_missed_note_ids()
        if newly_missed_note_ids:
            missed_note_id_set = set(newly_missed_note_ids)
            judged_miss_tokens = tuple(
                sorted(
                    {
                        str(getattr(note, "token", "")).strip().upper()
                        for note in normalized_target_notes
                        if str(getattr(note, "target_note_id", "")) in missed_note_id_set
                        and str(getattr(note, "token", "")).strip()
                    }
                )
            )
        if judged_hits:
            LOGGER.info(
                "Judged %s hit(s) this frame: %s",
                len(judged_hits),
                ", ".join(
                    f"{hit.token}@{hit.note_timestamp_ms:.1f}ms Δ{hit.timing_error_ms:+.1f}ms"
                    for hit in judged_hits
                ),
            )
    if gesture_sounds is not None and gesture_playback_controller is not None:
        gesture_playback_controller.update(
            pygame_module=pygame_module,
            transition_snapshot=transition_snapshot,
            gesture_sounds=gesture_sounds,
        )
    precue_hints = compute_precue_hints(
        normalized_target_notes=normalized_target_notes,
        song_started_monotonic=song_started_monotonic,
        pre_song_lead_in_ms=pre_song_lead_in_ms,
        song_speed_multiplier=song_speed_multiplier,
        precue_ms=float(config.raw["gameplay"].get("precue_ms", 0.0)),
        hit_window_ms=float(config.raw["gameplay"]["hit_window_ms"]),
        hit_window_judge=hit_window_judge,
    )
    precue_tokens = tuple(hint.token for hint in precue_hints)
    circle_visual_states = circle_visual_tracker.update(
        circle_geometries=circle_geometries,
        gameplay_keypoints=gameplay_keypoints,
        interaction_snapshot=interaction_snapshot,
        hit_tokens=judged_hit_tokens,
        miss_tokens=judged_miss_tokens,
        precue_tokens=precue_tokens,
        precue_hints=precue_hints,
    )
    target_layout = compute_liveview_layout(
        frame_width=frame.width,
        frame_height=frame.height,
        selection=selection,
        gameplay_keypoints=gameplay_keypoints,
        eye_center_x_pct=float(config.raw["liveview"]["eye_center_x_pct"]),
        eye_center_y_pct=float(config.raw["liveview"]["eye_center_y_pct"]),
        crop_width_eye_dist=float(config.raw["liveview"]["crop_width_eye_dist"]),
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
    draw_timeline_panel_layout(
        surface=window,
        pygame_module=pygame_module,
        layout=timeline_layout,
        base_color=(14, 23, 40),
    )
    song_elapsed_seconds = 0.0
    if song_started_monotonic is not None:
        lead_in_seconds = max(0.0, float(pre_song_lead_in_ms) / 1000.0)
        safe_song_speed = max(1e-6, float(song_speed_multiplier))
        song_elapsed_seconds = (
            (time.monotonic() - float(song_started_monotonic) - lead_in_seconds)
            * safe_song_speed
        )
    judged_note_outcomes: dict[str, str] | None = None
    if hit_window_judge is not None:
        judged_note_outcomes = hit_window_judge.judged_note_outcomes
    gameplay_score_state = None
    if gameplay_score_tracker is not None:
        gameplay_score_state = gameplay_score_tracker.state
    draw_upcoming_timeline_notes(
        surface=window,
        pygame_module=pygame_module,
        layout=timeline_layout,
        normalized_target_notes=normalized_target_notes,
        song_elapsed_seconds=song_elapsed_seconds,
        lookahead_ms=float(config.raw["gameplay"]["lookahead_ms"]),
        note_history_ms=float(config.raw["gameplay"]["note_history_ms"]),
        judged_note_outcomes=judged_note_outcomes,
        gameplay_score_state=gameplay_score_state,
    )

    scaled_surface = pygame_module.transform.smoothscale(
        cropped_frame.render_surface,
        (layout.scaled_width, layout.scaled_height),
    )
    circle_visuals_cfg = config.raw["liveview"]["circle_visuals"]
    draw_liveview_overlay(
        scaled_surface,
        source_width=cropped_frame.width,
        source_height=cropped_frame.height,
        target_width=layout.scaled_width,
        target_height=layout.scaled_height,
        selection=selection,
        gameplay_keypoints=overlay_gameplay_keypoints,
        circle_geometries=circle_geometries,
        circle_visual_states=circle_visual_states,
        circle_visual_styles=circle_visual_styles,
        pygame_module=pygame_module,
        crop_origin=(layout.crop.x, layout.crop.y),
        circle_stroke_width=int(config.raw["liveview"]["circle_stroke_width"]),
        label_font_size=int(config.raw["liveview"]["label_font_size"]),
        contact_stroke_color=tuple(circle_visuals_cfg.get("contact_stroke_color", [0, 255, 255])),
        contact_stroke_width=int(circle_visuals_cfg.get("contact_stroke_width", 6)),
        precue_gradient_start_color=tuple(
            circle_visuals_cfg.get("precue_gradient_start_color", [59, 130, 246])
        ),
        precue_gradient_end_color=tuple(
            circle_visuals_cfg.get("precue_gradient_end_color", [250, 204, 21])
        ),
        precue_min_stroke_width=int(circle_visuals_cfg.get("precue_min_stroke_width", 2)),
        precue_max_stroke_width=int(circle_visuals_cfg.get("precue_max_stroke_width", 7)),
        precue_min_fill_alpha=float(circle_visuals_cfg.get("precue_min_fill_alpha", 0.20)),
        precue_max_fill_alpha=float(circle_visuals_cfg.get("precue_max_fill_alpha", 0.70)),
        precue_primary_outline_color=tuple(
            circle_visuals_cfg.get("precue_primary_outline_color", [255, 255, 255])
        ),
        precue_primary_outline_extra_width=int(
            circle_visuals_cfg.get("precue_primary_outline_extra_width", 2)
        ),
        show_head_center_marker=bool(config.raw["liveview"]["debug"]["show_head_center"]),
        show_wrist_markers=bool(config.raw["liveview"]["debug"]["show_wrist_markers"]),
        wrist_marker_radius=int(config.raw["liveview"]["wrist_marker_radius"]),
        wrist_marker_outline_width=int(config.raw["liveview"]["wrist_marker_outline_width"]),
        show_stage1_full_keypoints=bool(config.raw["liveview"]["debug"]["show_stage1_full_keypoints"]),
        stage1_keypoints_xy=stage1_keypoints_xy,
        stage1_keypoints_conf=stage1_keypoints_conf,
        stage1_full_keypoints_style=dict(
            config.raw["liveview"]["debug"]["stage1_full_keypoints_style"]
        ),
        show_stage2_hand_keypoints=bool(config.raw["liveview"]["debug"]["show_stage2_hand_keypoints"]),
        stage2_hand_keypoints_xy=(
            () if hand_inference is None else hand_inference.remapped_keypoints_xy
        ),
        stage2_hand_keypoints_conf=(
            () if hand_inference is None else hand_inference.remapped_keypoints_conf
        ),
        stage2_hand_keypoints_style=dict(
            config.raw["liveview"]["debug"]["stage2_hand_keypoints_style"]
        ),
        show_keypoint_overlay_legend=bool(
            config.raw["liveview"]["debug"]["show_keypoint_overlay_legend"]
        ),
        show_overlay_confidence_values=bool(
            config.raw["liveview"]["debug"]["show_overlay_confidence_values"]
        ),
        keypoint_overlay_legend_style=dict(
            config.raw["liveview"]["debug"]["keypoint_overlay_legend_style"]
        ),
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
        hand_inference=hand_inference,
        transition_snapshot=transition_snapshot,
    )
def run_persistent_liveview_loop(
    *,
    window: Any,
    pygame_module: Any,
    frame_reader: CameraFrameReader,
    pose_model: Any,
    hand_pose_model: Any,
    primary_person_tracker: PrimaryPersonTracker,
    gameplay_keypoint_tracker: GameplayKeypointTracker,
    interaction_transition_tracker: HandCircleTransitionTracker,
    crop_smoother: CropSmoother,
    circle_visual_tracker: CircleVisualStateTracker,
    config: Any,
    gesture_sounds: LoadedGestureSounds | None = None,
    gesture_playback_controller: GesturePlaybackController | None = None,
    normalized_target_notes: tuple[Any, ...] = (),
    song_started_monotonic: float | None = None,
    pre_song_lead_in_ms: float = 0.0,
    song_speed_multiplier: float = 1.0,
    song_audio_start: Any | None = None,
    song_audio_start_at_monotonic: float | None = None,
) -> tuple[str, GameplayScoreState | None]:
    target_fps = max(1, int(config.raw["app"]["target_fps"]))
    clock = pygame_module.time.Clock()
    LOGGER.info(
        "Starting persistent liveview loop. Press ESC or close the window to exit. "
        "target_fps=%s song_speed_multiplier=%.3f",
        target_fps,
        float(song_speed_multiplier),
    )
    running = True
    hit_window_judge: HitWindowJudge | None = None
    gameplay_score_tracker: GameplayScoreTracker | None = None
    if normalized_target_notes:
        hit_window_judge = HitWindowJudge(
            normalized_target_notes=normalized_target_notes,
            hit_window_ms=float(config.raw["gameplay"]["hit_window_ms"]),
        )
        score_values = config.raw["gameplay"]["score_values"]
        gameplay_score_tracker = GameplayScoreTracker(
            total_notes=len(normalized_target_notes),
            hit_score=int(score_values["hit"]),
            miss_score=int(score_values["miss"]),
            combo_bonus=int(score_values["combo_bonus"]),
        )
    song_complete_at_monotonic: float | None = None
    if song_started_monotonic is not None and normalized_target_notes:
        safe_song_speed = max(1e-6, float(song_speed_multiplier))
        last_target_end_seconds = max(
            float(getattr(note, "timestamp_seconds", 0.0)) + float(getattr(note, "duration_seconds", 0.0))
            for note in normalized_target_notes
        )
        history_seconds = max(0.0, float(config.raw["gameplay"].get("note_history_ms", 0.0)) / 1000.0)
        lead_in_seconds = max(0.0, float(pre_song_lead_in_ms) / 1000.0)
        song_complete_at_monotonic = (
            float(song_started_monotonic)
            + lead_in_seconds
            + ((last_target_end_seconds + history_seconds) / safe_song_speed)
        )
    while running:
        for event in pygame_module.event.get():
            if event.type == pygame_module.QUIT:
                LOGGER.info("Received pygame QUIT event. Exiting persistent liveview loop.")
                return "quit", _final_score_state_or_none(gameplay_score_tracker)
            if event.type == pygame_module.KEYDOWN and event.key == pygame_module.K_ESCAPE:
                LOGGER.info("Received ESC key input. Exiting persistent liveview loop.")
                return "quit", _final_score_state_or_none(gameplay_score_tracker)

        if not running:
            continue

        if song_audio_start is not None and (
            song_audio_start_at_monotonic is None or time.monotonic() >= float(song_audio_start_at_monotonic)
        ):
            try:
                song_audio_start()
            except Exception as exc:
                raise LiveviewRuntimeError(f"Failed to start scheduled song audio playback: {exc}") from exc
            song_audio_start = None

        render_liveview_frame(
            window=window,
            pygame_module=pygame_module,
            frame_reader=frame_reader,
            pose_model=pose_model,
            hand_pose_model=hand_pose_model,
            primary_person_tracker=primary_person_tracker,
            gameplay_keypoint_tracker=gameplay_keypoint_tracker,
            interaction_transition_tracker=interaction_transition_tracker,
            crop_smoother=crop_smoother,
            circle_visual_tracker=circle_visual_tracker,
            config=config,
            gesture_sounds=gesture_sounds,
            gesture_playback_controller=gesture_playback_controller,
            normalized_target_notes=normalized_target_notes,
            song_started_monotonic=song_started_monotonic,
            pre_song_lead_in_ms=pre_song_lead_in_ms,
            song_speed_multiplier=song_speed_multiplier,
            hit_window_judge=hit_window_judge,
            gameplay_score_tracker=gameplay_score_tracker,
        )
        if song_complete_at_monotonic is not None and time.monotonic() >= song_complete_at_monotonic:
            LOGGER.info("Song timeline completed. Exiting gameplay loop for post-song transition.")
            return "song_complete", _final_score_state_or_none(gameplay_score_tracker)
        clock.tick(target_fps)
    return "quit", _final_score_state_or_none(gameplay_score_tracker)
def _run_hand_roi_inference(
    *,
    hand_pose_model: Any,
    frame: Any,
    selection: PrimaryPersonSelection | None,
    config: Any,
) -> HandRoiInference | None:
    global _LAST_STAGE2_HAND_INFERENCE, _LAST_STAGE2_HAND_INFERENCE_AT

    pose_config = config.raw.get("pose", {})
    expansion_px = int(pose_config.get("stage2_roi_expansion_px", 30))
    stage2_conf = float(pose_config.get("stage2_confidence_threshold", pose_config["confidence_threshold"]))
    stage2_iou = float(pose_config.get("stage2_iou_threshold", pose_config["iou_threshold"]))
    fallback_mode = str(pose_config.get("stage2_missing_fallback_mode", "clear")).strip().lower()
    fallback_timeout_seconds = max(
        0.0,
        float(pose_config.get("stage2_missing_fallback_timeout_seconds", 0.0)),
    )

    inference = run_stage2_hand_inference_on_person_roi(
        hand_model=hand_pose_model,
        frame_bgr=frame.bgr_frame,
        selection=selection,
        expansion_px=expansion_px,
        conf=stage2_conf,
        iou=stage2_iou,
    )

    has_stage2_keypoints = (
        inference is not None
        and len(inference.remapped_keypoints_xy) > 0
    )
    now = time.monotonic()

    if has_stage2_keypoints:
        _LAST_STAGE2_HAND_INFERENCE = inference
        _LAST_STAGE2_HAND_INFERENCE_AT = now
        return inference

    if (
        fallback_mode == "reuse_last"
        and _LAST_STAGE2_HAND_INFERENCE is not None
        and _LAST_STAGE2_HAND_INFERENCE_AT is not None
        and (now - _LAST_STAGE2_HAND_INFERENCE_AT) <= fallback_timeout_seconds
    ):
        return _LAST_STAGE2_HAND_INFERENCE

    return inference


def _log_frame_summary(
    *,
    frame: Any,
    layout: Any,
    left_panel_width: int,
    left_panel_height: int,
    selection: PrimaryPersonSelection | None,
    gameplay_keypoint_tracker: GameplayKeypointTracker,
    gameplay_keypoints: Any,
    hand_inference: HandRoiInference | None,
    transition_snapshot: Any,
) -> None:
    hand_roi_summary = "none"
    if hand_inference is not None:
        x1, y1, x2, y2 = hand_inference.roi_xyxy
        hand_roi_summary = (
            f"roi=({x1},{y1},{x2},{y2}) "
            f"hands={len(hand_inference.remapped_keypoints_xy)}"
        )

    if 0: LOGGER.info(
        "Prepared person-centered liveview crop: frame=%sx%s crop=(x=%s y=%s w=%s h=%s) scaled=%sx%s visible_width=%s blit=(%s,%s) source_offset_x=%s mirrored=%s left_panel=%sx%s primary_person=%s keypoints=%s stage2_hand=%s transitions=%s.",
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
        hand_roi_summary,
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
