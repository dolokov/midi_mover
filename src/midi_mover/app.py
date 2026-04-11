"""Application bootstrap for the midi_mover startup skeleton."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from midi_mover.camera import CameraFrameError, CameraFrameReader
from midi_mover.cli import StartupOptions, parse_args
from midi_mover.circles import compute_circle_geometries
from midi_mover.config import AppConfig, ConfigError, load_config
from midi_mover.audio import (
    AudioStartupError,
    LoadedGestureSounds,
    build_gesture_sounds,
    initialize_audio_output,
    trigger_gesture_sounds,
)
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
from midi_mover.logging_utils import configure_logging
from midi_mover.pose import (
    GameplayKeypointTracker,
    PrimaryPersonSelection,
    PrimaryPersonTracker,
    PoseProcessingError,
    run_pose_inference,
)
LOGGER = logging.getLogger("midi_mover")


class StartupError(RuntimeError):
    """Raised when a required runtime subsystem fails to initialize."""

@dataclass
class StartupResources:
    window: Any | None = None
    camera: Any | None = None
    frame_reader: CameraFrameReader | None = None
    pose_model: Any | None = None
    primary_person_tracker: PrimaryPersonTracker | None = None
    gameplay_keypoint_tracker: GameplayKeypointTracker | None = None
    interaction_transition_tracker: HandCircleTransitionTracker | None = None
    crop_smoother: CropSmoother | None = None
    circle_visual_tracker: CircleVisualStateTracker | None = None
    pygame_module: Any | None = None
    mixer_initialized: bool = False
    gesture_sounds: LoadedGestureSounds | None = None

    def cleanup(self) -> None:
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:  # pragma: no cover - defensive cleanup
                LOGGER.exception("Failed to release camera cleanly during cleanup.")
            else:
                LOGGER.info("Released OpenCV camera resource.")
            finally:
                self.camera = None

        if self.pygame_module is not None:
            if self.mixer_initialized:
                try:
                    self.pygame_module.mixer.quit()
                except Exception:  # pragma: no cover - defensive cleanup
                    LOGGER.exception("Failed to quit pygame mixer cleanly during cleanup.")
                else:
                    LOGGER.info("Shut down pygame mixer.")
                finally:
                    self.mixer_initialized = False

            try:
                self.pygame_module.display.quit()
            except Exception:  # pragma: no cover - defensive cleanup
                LOGGER.exception("Failed to quit pygame display cleanly during cleanup.")
            else:
                LOGGER.info("Shut down pygame display.")

            try:
                self.pygame_module.quit()
            except Exception:  # pragma: no cover - defensive cleanup
                LOGGER.exception("Failed to quit pygame cleanly during cleanup.")
            else:
                LOGGER.info("Shut down pygame.")


def initialize_runtime(options: StartupOptions, config: AppConfig) -> StartupResources:
    resources = StartupResources()
    try:
        resources.pygame_module = _initialize_pygame_display(config, resources)
        _initialize_pygame_mixer(config, resources)
        resources.camera = _initialize_camera(options, config)
        resources.pose_model = _initialize_pose_model(config)
    except StartupError:
        resources.cleanup()
        raise
    except Exception as exc:  # pragma: no cover - defensive conversion
        resources.cleanup()
        raise StartupError(f"Unexpected startup failure: {exc}") from exc

    return resources


def _initialize_pygame_display(config: AppConfig, resources: StartupResources) -> Any:
    try:
        import pygame
    except ModuleNotFoundError as exc:
        raise StartupError(
            "pygame is required for the display subsystem but is not installed in the active environment. "
            "Activate conda env 'midi_mover' and install pygame."
        ) from exc

    try:
        pygame.init()
        window = pygame.display.set_mode(
            (
                int(config.raw["app"]["window_width"]),
                int(config.raw["app"]["window_height"]),
            )
        )
        pygame.display.set_caption(str(config.raw["app"]["name"]))
    except Exception as exc:
        raise StartupError(f"Failed to initialize pygame display: {exc}") from exc

    resources.window = window
    LOGGER.info("Initialized pygame display at %sx%s.", *window.get_size())
    return pygame


def _initialize_pygame_mixer(config: AppConfig, resources: StartupResources) -> None:
    pygame = resources.pygame_module
    if pygame is None:
        raise StartupError("Internal error: pygame display must initialize before the mixer.")

    try:
        initialize_audio_output(
            pygame_module=pygame,
            audio_config=config.raw["audio"],
        )
    except AudioStartupError as exc:
        raise StartupError(str(exc)) from exc

    resources.mixer_initialized = True
    resources.gesture_sounds = build_gesture_sounds(
        pygame_module=pygame,
        audio_config=config.raw["audio"],
    )


def _initialize_camera(options: StartupOptions, config: AppConfig) -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise StartupError(
            "opencv-python is required for camera startup but is not installed in the active environment. "
            "Activate conda env 'midi_mover' and install opencv-python."
        ) from exc

    camera = cv2.VideoCapture(options.camera_id)
    if not camera.isOpened():
        camera.release()
        raise StartupError(
            f"Failed to open camera id {options.camera_id}. Check that the camera exists and is not busy."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(config.raw["camera"]["width"]))
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config.raw["camera"]["height"]))
    LOGGER.info(
        "Initialized OpenCV camera id=%s requested_resolution=%sx%s.",
        options.camera_id,
        config.raw["camera"]["width"],
        config.raw["camera"]["height"],
    )
    return camera


def _initialize_frame_reader(resources: StartupResources, config: AppConfig) -> CameraFrameReader:
    if resources.camera is None:
        raise StartupError("Internal error: camera must initialize before the frame reader.")

    frame_reader = CameraFrameReader(
        camera=resources.camera,
        mirror=bool(config.raw["camera"]["mirror"]),
    )
    LOGGER.info("Initialized camera frame reader with mirror=%s.", frame_reader.mirror)
    return frame_reader


def _initialize_primary_person_tracker(config: AppConfig) -> PrimaryPersonTracker:
    tracker = PrimaryPersonTracker(
        confidence_threshold=float(config.raw["pose"]["confidence_threshold"]),
        lost_timeout_seconds=float(config.raw["pose"]["lost_person_timeout_seconds"]),
    )
    LOGGER.info(
        "Initialized primary person tracker with confidence_threshold=%s lost_timeout_seconds=%s.",
        config.raw["pose"]["confidence_threshold"],
        config.raw["pose"]["lost_person_timeout_seconds"],
    )
    return tracker


def _initialize_pose_model(config: AppConfig) -> Any:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise StartupError(
            "ultralytics is required for pose-model startup but is not installed in the active environment. "
            "Activate conda env 'midi_mover' and install ultralytics."
        ) from exc

    model_name = str(config.raw["pose"]["model_name"])
    device = str(config.raw["pose"]["device"])
    try:
        model = YOLO(model_name)
        if hasattr(model, "to"):
            model.to(device)
    except Exception as exc:
        raise StartupError(
            f"Failed to initialize Ultralytics pose model '{model_name}' on device '{device}': {exc}"
        ) from exc

    LOGGER.info("Initialized Ultralytics pose model '%s' on device '%s'.", model_name, device)
    return model


def _initialize_gameplay_keypoint_tracker(config: AppConfig) -> GameplayKeypointTracker:
    tracker = GameplayKeypointTracker(
        confidence_threshold=float(config.raw["pose"]["confidence_threshold"]),
        fallback_timeout_seconds=float(config.raw["pose"]["lost_person_timeout_seconds"]),
    )
    LOGGER.info(
        "Initialized gameplay keypoint tracker with confidence_threshold=%s fallback_timeout_seconds=%s.",
        config.raw["pose"]["confidence_threshold"],
        config.raw["pose"]["lost_person_timeout_seconds"],
    )
    return tracker


def _initialize_crop_smoother(config: AppConfig) -> CropSmoother:
    smoother = CropSmoother(
        smoothing_factor=float(config.raw["liveview"]["crop_smoothing_factor"]),
        max_jump_ratio=float(config.raw["liveview"]["crop_max_jump_ratio"]),
    )
    LOGGER.info(
        "Initialized crop smoother with smoothing_factor=%s max_jump_ratio=%s.",
        config.raw["liveview"]["crop_smoothing_factor"],
        config.raw["liveview"]["crop_max_jump_ratio"],
    )
    return smoother


def _initialize_interaction_transition_tracker(config: AppConfig) -> HandCircleTransitionTracker:
    tracker = HandCircleTransitionTracker(
        debounce_ms=int(config.raw["gameplay"]["debounce_ms"]),
    )
    LOGGER.info(
        "Initialized hand-circle transition tracker with debounce_ms=%s.",
        config.raw["gameplay"]["debounce_ms"],
    )
    return tracker


def _initialize_circle_visual_tracker(config: AppConfig) -> CircleVisualStateTracker:
    visuals = config.raw["liveview"]["circle_visuals"]
    tracker = CircleVisualStateTracker(
        hit_flash_duration_ms=int(visuals["hit_flash_duration_ms"]),
        miss_flash_duration_ms=int(visuals["miss_flash_duration_ms"]),
    )
    LOGGER.info(
        "Initialized circle visual tracker with hit_flash_duration_ms=%s miss_flash_duration_ms=%s.",
        visuals["hit_flash_duration_ms"],
        visuals["miss_flash_duration_ms"],
    )
    return tracker


def _load_circle_visual_styles(config: AppConfig) -> dict[str, CircleVisualStyle]:
    visuals = config.raw["liveview"]["circle_visuals"]
    return {
        state_name: CircleVisualStyle(
            outline_color=tuple(visuals[state_name]["outline_color"]),
            fill_color=tuple(visuals[state_name]["fill_color"]),
            label_color=tuple(visuals[state_name]["label_color"]),
        )
        for state_name in ("idle", "active_contact", "hit_flash", "miss_flash")
    }


def discover_midi_files(midi_dir: Path, config: AppConfig) -> list[Path]:
    supported_extensions = {
        str(extension).lower() for extension in config.raw["midi"]["supported_extensions"]
    }
    midi_files = sorted(
        path
        for path in midi_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported_extensions
    )
    if not midi_files:
        raise StartupError(
            f"No supported MIDI files were found in {midi_dir}. "
            f"Expected one of: {', '.join(sorted(supported_extensions))}."
        )

    LOGGER.info("Discovered %s MIDI file(s) in %s.", len(midi_files), midi_dir)
    return midi_files


def _render_liveview_preview(resources: StartupResources, config: AppConfig) -> None:
    if resources.window is None:
        raise StartupError("Liveview preview failed: pygame window was not initialized.")
    if resources.pygame_module is None:
        raise StartupError("Liveview preview failed: pygame module was not initialized.")
    if resources.frame_reader is None:
        raise StartupError("Liveview preview failed: camera frame reader was not initialized.")
    if resources.pose_model is None:
        raise StartupError("Liveview preview failed: pose model was not initialized.")
    if resources.primary_person_tracker is None:
        raise StartupError("Liveview preview failed: primary person tracker was not initialized.")
    if resources.gameplay_keypoint_tracker is None:
        raise StartupError("Liveview preview failed: gameplay keypoint tracker was not initialized.")
    if resources.crop_smoother is None:
        raise StartupError("Liveview preview failed: crop smoother was not initialized.")
    if resources.interaction_transition_tracker is None:
        raise StartupError("Liveview preview failed: interaction transition tracker was not initialized.")
    if resources.circle_visual_tracker is None:
        raise StartupError("Liveview preview failed: circle visual tracker was not initialized.")

    padding_color = tuple(config.raw["liveview"]["padding_color"])
    circle_visual_styles = _load_circle_visual_styles(config)
    left_panel_ratio = float(config.raw["liveview"]["left_panel_ratio"])
    crop_margin_ratio = float(config.raw["liveview"]["player_crop_margin"])
    left_panel_width = int(resources.window.get_width() * left_panel_ratio)
    left_panel_height = resources.window.get_height()

    try:
        frame = resources.frame_reader.read(resources.pygame_module)
    except CameraFrameError as exc:
        raise StartupError(f"Liveview preview failed while reading camera frame: {exc}") from exc

    try:
        pose_result = run_pose_inference(
            resources.pose_model,
            frame.bgr_frame,
            conf=float(config.raw["pose"]["confidence_threshold"]),
            iou=float(config.raw["pose"]["iou_threshold"]),
        )
    except PoseProcessingError as exc:
        raise StartupError(f"Liveview preview failed during pose inference: {exc}") from exc

    selection = resources.primary_person_tracker.select(pose_result)
    gameplay_keypoints = resources.gameplay_keypoint_tracker.extract(pose_result, selection)
    circle_geometries = compute_circle_geometries(
        head_center_xy=getattr(gameplay_keypoints, "head_center_xy", None),
        circle_offsets=config.raw["liveview"]["circle_offsets"],
        circle_radius=int(config.raw["liveview"]["circle_radius"]),
    )
    interaction_snapshot = detect_hand_circle_interactions(
        circle_geometries=circle_geometries,
        gameplay_keypoints=gameplay_keypoints,
    )
    transition_snapshot = resources.interaction_transition_tracker.update(interaction_snapshot)
    if resources.gesture_sounds is not None:
        trigger_gesture_sounds(
            pygame_module=resources.pygame_module,
            transition_snapshot=transition_snapshot,
            gesture_sounds=resources.gesture_sounds,
        )
    circle_visual_states = resources.circle_visual_tracker.update(
        circle_geometries=circle_geometries,
        gameplay_keypoints=gameplay_keypoints,
        interaction_snapshot=interaction_snapshot,
    )
    target_layout = compute_liveview_layout(
        frame_width=frame.width,
        frame_height=frame.height,
        selection=selection,
        crop_margin_ratio=crop_margin_ratio,
        target_panel_width=left_panel_width,
        target_panel_height=left_panel_height,
    )
    smoothed_crop = resources.crop_smoother.smooth(
        target_crop=target_layout.crop,
        frame_width=frame.width,
        frame_height=frame.height,
    )
    cropped_frame = crop_camera_frame(frame, smoothed_crop, resources.pygame_module)
    layout = compute_liveview_layout_for_crop(
        crop=smoothed_crop,
        target_panel_width=left_panel_width,
        target_panel_height=left_panel_height,
    )

    resources.window.fill((0, 0, 0))
    left_panel_rect = resources.pygame_module.Rect(0, 0, left_panel_width, left_panel_height)
    resources.window.fill(padding_color, left_panel_rect)

    scaled_surface = resources.pygame_module.transform.smoothscale(
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
        pygame_module=resources.pygame_module,
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
            resources.pygame_module.Rect(
                layout.source_offset_x,
                0,
                layout.visible_width,
                layout.scaled_height,
            )
        )
    else:
        visible_surface = scaled_surface

    resources.window.blit(visible_surface, (layout.blit_x, layout.blit_y))
    resources.pygame_module.display.flip()

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
        resources.gameplay_keypoint_tracker.describe(gameplay_keypoints),
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


def run_smoke_test(
    options: StartupOptions,
    config: AppConfig,
    resources: StartupResources,
) -> None:
    LOGGER.info("Running startup smoke test.")
    midi_files = discover_midi_files(options.midi_dir, config)

    if resources.window is None:
        raise StartupError("Smoke test failed: pygame window was not initialized.")
    if resources.camera is None:
        raise StartupError("Smoke test failed: camera was not initialized.")
    if resources.pose_model is None:
        raise StartupError("Smoke test failed: pose model was not initialized.")
    if resources.pygame_module is None or not resources.mixer_initialized:
        raise StartupError("Smoke test failed: pygame mixer was not initialized.")
    if resources.gesture_sounds is None:
        raise StartupError("Smoke test failed: gesture-to-sound mapping was not initialized.")
    if resources.frame_reader is None:
        raise StartupError("Smoke test failed: camera frame reader was not initialized.")
    if resources.primary_person_tracker is None:
        raise StartupError("Smoke test failed: primary person tracker was not initialized.")
    if resources.gameplay_keypoint_tracker is None:
        raise StartupError("Smoke test failed: gameplay keypoint tracker was not initialized.")
    if resources.crop_smoother is None:
        raise StartupError("Smoke test failed: crop smoother was not initialized.")
    if resources.interaction_transition_tracker is None:
        raise StartupError("Smoke test failed: interaction transition tracker was not initialized.")
    if resources.circle_visual_tracker is None:
        raise StartupError("Smoke test failed: circle visual tracker was not initialized.")

    pygame = resources.pygame_module
    pygame.event.pump()
    _render_liveview_preview(resources, config)

    LOGGER.info(
        "Smoke test touched subsystems successfully: window=%s mixer=%s frame_reader=%s pose_model=%s midi_files=%s.",
        resources.window.get_size(),
        resources.mixer_initialized,
        type(resources.frame_reader).__name__,
        type(resources.pose_model).__name__,
        len(midi_files),
    )
    LOGGER.info("Smoke test completed successfully and exited cleanly.")


def _log_startup(options: StartupOptions) -> None:
    LOGGER.info("Starting midi_mover startup skeleton")
    LOGGER.info("camera_id=%s", options.camera_id)
    LOGGER.info("midi_dir=%s", options.midi_dir)
    LOGGER.info("config_path=%s", options.config_path)
    LOGGER.info("smoke_test=%s", options.smoke_test)


def _log_config_summary(config: AppConfig) -> None:
    raw = config.raw
    LOGGER.info(
        "Loaded config: window=%sx%s fps=%s pose_model=%s",
        raw["app"]["window_width"],
        raw["app"]["window_height"],
        raw["app"]["target_fps"],
        raw["pose"]["model_name"],
    )
    LOGGER.info(
        "Liveview ratio=%s circle_radius=%s supported_midi_extensions=%s",
        raw["liveview"]["left_panel_ratio"],
        raw["liveview"]["circle_radius"],
        ", ".join(raw["midi"]["supported_extensions"]),
    )


def run(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    configure_logging(options.log_level)
    _log_startup(options)
    try:
        config = load_config(options.config_path)
    except ConfigError as exc:
        LOGGER.error("Config validation failed: %s", exc)
        return 2

    configured_level = str(config.raw["app"]["logging_level"]).upper()
    if configured_level != options.log_level.upper():
        configure_logging(configured_level)
        LOGGER.info(
            "Adjusted logging level from CLI value %s to config value %s.",
            options.log_level.upper(),
            configured_level,
        )
        _log_startup(options)

    _log_config_summary(config)
    resources: StartupResources | None = None
    try:
        resources = initialize_runtime(options, config)
        resources.frame_reader = _initialize_frame_reader(resources, config)
        resources.primary_person_tracker = _initialize_primary_person_tracker(config)
        resources.gameplay_keypoint_tracker = _initialize_gameplay_keypoint_tracker(config)
        resources.interaction_transition_tracker = _initialize_interaction_transition_tracker(config)
        resources.crop_smoother = _initialize_crop_smoother(config)
        resources.circle_visual_tracker = _initialize_circle_visual_tracker(config)
        if options.smoke_test:
            run_smoke_test(options, config, resources)
        else:
            _render_liveview_preview(resources, config)
    except StartupError as exc:
        LOGGER.error("Startup initialization failed: %s", exc)
        return 3
    finally:
        if resources is not None:
            resources.cleanup()

    if options.smoke_test:
        LOGGER.info("Subtask 1.1.4 complete: smoke-test path exercised startup subsystems and exited cleanly.")
    else:
        LOGGER.info("Subtask 1.1.3 complete: runtime subsystems initialized and cleaned up successfully.")
    return 0