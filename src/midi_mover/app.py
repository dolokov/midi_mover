"""Application bootstrap for the midi_mover startup skeleton."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any

from midi_mover.camera import CameraFrameReader
from midi_mover.cli import StartupOptions, parse_args
from midi_mover.config import AppConfig, ConfigError, load_config
from midi_mover.audio import AudioStartupError, GesturePlaybackController, LoadedGestureSounds, build_gesture_sounds, initialize_audio_output
from midi_mover.interaction_visuals import CircleVisualStateTracker, HandCircleTransitionTracker
from midi_mover.integration_checks import verify_fingertip_audio_integration
from midi_mover.liveview import CropSmoother
from midi_mover.logging_utils import configure_logging
from midi_mover.midi_files import (
    MidiDiscoveryError,
    MidiRandomSelector,
    discover_supported_midi_files,
    normalize_supported_extensions,
)
from midi_mover.midi_parser import MidiParseError, build_midi_debug_report
from midi_mover.pose import GameplayKeypointTracker, PrimaryPersonTracker
from midi_mover.song_intro import build_song_title, show_pre_song_title_screen
from midi_mover.song_targets import load_normalized_target_notes_from_midi
from midi_mover.runtime_loop import LiveviewRuntimeError, render_liveview_frame, run_persistent_liveview_loop
LOGGER = logging.getLogger("midi_mover")


class StartupError(RuntimeError):
    """Raised when a required runtime subsystem fails to initialize."""

@dataclass
class StartupResources:
    window: Any | None = None
    camera: Any | None = None
    frame_reader: CameraFrameReader | None = None
    pose_model: Any | None = None
    hand_pose_model: Any | None = None
    primary_person_tracker: PrimaryPersonTracker | None = None
    gameplay_keypoint_tracker: GameplayKeypointTracker | None = None
    interaction_transition_tracker: HandCircleTransitionTracker | None = None
    crop_smoother: CropSmoother | None = None
    circle_visual_tracker: CircleVisualStateTracker | None = None
    pygame_module: Any | None = None
    mixer_initialized: bool = False
    gesture_sounds: LoadedGestureSounds | None = None
    gesture_playback_controller: GesturePlaybackController | None = None

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
                if self.gesture_playback_controller is not None and self.gesture_sounds is not None:
                    try:
                        self.gesture_playback_controller.reset(
                            fade_ms=self.gesture_sounds.playback_config.release_fade_ms,
                        )
                    except Exception:  # pragma: no cover - defensive cleanup
                        LOGGER.exception("Failed to reset gesture playback controller during cleanup.")
                    finally:
                        self.gesture_playback_controller = None
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
        stage1_model_name, stage2_model_name = _resolve_model_names(options, config)
        device = str(config.raw["pose"]["device"])
        resources.pose_model = _initialize_ultralytics_model(
            model_name=stage1_model_name,
            device=device,
            model_role="stage-1 pose",
        )
        resources.hand_pose_model = _initialize_ultralytics_model(
            model_name=stage2_model_name,
            device=device,
            model_role="stage-2 hand",
        )
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
    resources.gesture_playback_controller = GesturePlaybackController()


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


def _resolve_model_names(options: StartupOptions, config: AppConfig) -> tuple[str, str]:
    stage1_model_name = options.stage1_pose_model or str(config.raw["pose"]["stage1_model_name"])
    stage2_model_name = options.stage2_hand_model or str(config.raw["pose"]["stage2_hand_model_name"])

    if not stage1_model_name.strip():
        raise StartupError(
            "Stage-1 pose model is empty. Pass --stage1-pose-model or set pose.stage1_model_name in YAML."
        )
    if not stage2_model_name.strip():
        raise StartupError(
            "Stage-2 hand model is empty. Pass --stage2-hand-model or set pose.stage2_hand_model_name in YAML."
        )

    return stage1_model_name.strip(), stage2_model_name.strip()


def _initialize_ultralytics_model(*, model_name: str, device: str, model_role: str) -> Any:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise StartupError(
            "ultralytics is required for pose-model startup but is not installed in the active environment. "
            "Activate conda env 'midi_mover' and install ultralytics."
        ) from exc

    try:
        model = YOLO(model_name)
        if hasattr(model, "to"):
            model.to(device)
    except Exception as exc:
        raise StartupError(
            f"Failed to initialize Ultralytics {model_role} model '{model_name}' on device '{device}': {exc}"
        ) from exc

    LOGGER.info(
        "Initialized Ultralytics %s model '%s' on device '%s'.",
        model_role,
        model_name,
        device,
    )
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


def discover_midi_files(midi_dir: Path, config: AppConfig) -> list[Path]:
    supported_extensions = normalize_supported_extensions(
        config.raw["midi"]["supported_extensions"],
    )
    try:
        midi_files = discover_supported_midi_files(
            midi_dir,
            supported_extensions=supported_extensions,
        )
    except MidiDiscoveryError as exc:
        raise StartupError(str(exc)) from exc

    LOGGER.info("Discovered %s MIDI file(s) in %s.", len(midi_files), midi_dir)
    return midi_files


def choose_random_midi_file(midi_files: list[Path], config: AppConfig) -> Path:
    random_seed = config.raw["app"]["random_seed"]
    selector = MidiRandomSelector(midi_files, random_seed=random_seed)
    selected = selector.choose()

    if selector.random_seed is None:
        LOGGER.info("Selected random MIDI file: %s", selected.name)
    else:
        LOGGER.info(
            "Selected random MIDI file using deterministic seed %s: %s",
            selector.random_seed,
            selected.name,
        )

    return selected


def _render_liveview_preview(resources: StartupResources, config: AppConfig) -> None:
    if resources.window is None:
        raise StartupError("Liveview preview failed: pygame window was not initialized.")
    if resources.pygame_module is None:
        raise StartupError("Liveview preview failed: pygame module was not initialized.")
    if resources.frame_reader is None:
        raise StartupError("Liveview preview failed: camera frame reader was not initialized.")
    if resources.pose_model is None:
        raise StartupError("Liveview preview failed: pose model was not initialized.")
    if resources.hand_pose_model is None:
        raise StartupError("Liveview preview failed: hand pose model was not initialized.")
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

    try:
        render_liveview_frame(
            window=resources.window,
            pygame_module=resources.pygame_module,
            frame_reader=resources.frame_reader,
            pose_model=resources.pose_model,
            hand_pose_model=resources.hand_pose_model,
            primary_person_tracker=resources.primary_person_tracker,
            gameplay_keypoint_tracker=resources.gameplay_keypoint_tracker,
            interaction_transition_tracker=resources.interaction_transition_tracker,
            crop_smoother=resources.crop_smoother,
            circle_visual_tracker=resources.circle_visual_tracker,
            config=config,
            gesture_sounds=resources.gesture_sounds,
            gesture_playback_controller=resources.gesture_playback_controller,
        )
    except LiveviewRuntimeError as exc:
        raise StartupError(str(exc)) from exc


def run_smoke_test(
    options: StartupOptions,
    config: AppConfig,
    resources: StartupResources,
) -> None:
    LOGGER.info("Running startup smoke test.")
    midi_files = discover_midi_files(options.midi_dir, config)
    selected_midi = choose_random_midi_file(midi_files, config)
    selected_song_title = build_song_title(selected_midi)

    if resources.window is None:
        raise StartupError("Smoke test failed: pygame window was not initialized.")
    if resources.camera is None:
        raise StartupError("Smoke test failed: camera was not initialized.")
    if resources.pose_model is None:
        raise StartupError("Smoke test failed: pose model was not initialized.")
    if resources.hand_pose_model is None:
        raise StartupError("Smoke test failed: hand pose model was not initialized.")
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
    verify_fingertip_audio_integration()
    LOGGER.info("Verified fingertip-driven interaction transitions and downstream audio hook integration.")

    LOGGER.info(
        "Smoke test touched subsystems successfully: window=%s mixer=%s frame_reader=%s pose_model=%s midi_files=%s.",
        resources.window.get_size(),
        resources.mixer_initialized,
        type(resources.frame_reader).__name__,
        type(resources.pose_model).__name__,
        len(midi_files),
    )
    LOGGER.info("Smoke test random MIDI selection result: %s", selected_midi.name)
    LOGGER.info("Smoke test extracted song title: %s", selected_song_title)
    LOGGER.info("Smoke test completed successfully and exited cleanly.")


def run_interactive_runtime(
    options: StartupOptions,
    config: AppConfig,
    resources: StartupResources,
) -> None:
    LOGGER.info("Running persistent interactive liveview runtime.")
    midi_files = discover_midi_files(options.midi_dir, config)
    selected_midi = choose_random_midi_file(midi_files, config)
    selected_song_title = build_song_title(selected_midi)
    if options.midi_inspect:
        try:
            LOGGER.info(
                "\n%s",
                build_midi_debug_report(
                    midi_path=selected_midi,
                    midi_config=config.raw["midi"],
                    max_notes=options.midi_inspect_max_notes,
                ),
            )
        except MidiParseError as exc:
            raise StartupError(f"MIDI inspection failed for '{selected_midi.name}': {exc}") from exc

    try:
        normalized_target_notes = load_normalized_target_notes_from_midi(
            midi_path=selected_midi,
            midi_config=config.raw["midi"],
        )
    except MidiParseError as exc:
        raise StartupError(
            f"Failed to load normalized target notes for timeline rendering from '{selected_midi.name}': {exc}"
        ) from exc

    required = {
        "window": resources.window,
        "pose_model": resources.pose_model,
        "hand_pose_model": resources.hand_pose_model,
        "pygame_module": resources.pygame_module,
        "frame_reader": resources.frame_reader,
        "primary_person_tracker": resources.primary_person_tracker,
        "gameplay_keypoint_tracker": resources.gameplay_keypoint_tracker,
        "interaction_transition_tracker": resources.interaction_transition_tracker,
        "crop_smoother": resources.crop_smoother,
        "circle_visual_tracker": resources.circle_visual_tracker,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise StartupError(f"Interactive runtime failed: missing initialized resources: {', '.join(missing)}.")

    show_title_screen = show_pre_song_title_screen(
        pygame_module=resources.pygame_module,
        window=resources.window,
        song_title=selected_song_title,
        duration_seconds=float(config.raw["app"].get("song_title_screen_duration_seconds", 2.0)),
    )
    if not show_title_screen:
        LOGGER.info("Pre-song title screen closed by user before gameplay runtime loop started.")
        return

    try:
        run_persistent_liveview_loop(
            window=resources.window,
            pygame_module=resources.pygame_module,
            frame_reader=resources.frame_reader,
            pose_model=resources.pose_model,
            hand_pose_model=resources.hand_pose_model,
            primary_person_tracker=resources.primary_person_tracker,
            gameplay_keypoint_tracker=resources.gameplay_keypoint_tracker,
            interaction_transition_tracker=resources.interaction_transition_tracker,
            crop_smoother=resources.crop_smoother,
            circle_visual_tracker=resources.circle_visual_tracker,
            config=config,
            gesture_sounds=resources.gesture_sounds,
            gesture_playback_controller=resources.gesture_playback_controller,
            normalized_target_notes=normalized_target_notes,
            song_started_monotonic=time.monotonic(),
        )
    except LiveviewRuntimeError as exc:
        raise StartupError(str(exc)) from exc


def _log_startup(options: StartupOptions) -> None:
    LOGGER.info("Starting midi_mover startup skeleton")
    LOGGER.info("camera_id=%s", options.camera_id)
    LOGGER.info("midi_dir=%s", options.midi_dir)
    LOGGER.info("config_path=%s", options.config_path)
    LOGGER.info("stage1_pose_model_override=%s", options.stage1_pose_model or "<config>")
    LOGGER.info("stage2_hand_model_override=%s", options.stage2_hand_model or "<config>")
    LOGGER.info("smoke_test=%s", options.smoke_test)
    LOGGER.info("midi_inspect=%s midi_inspect_max_notes=%s", options.midi_inspect, options.midi_inspect_max_notes)


def _log_config_summary(config: AppConfig) -> None:
    raw = config.raw
    LOGGER.info(
        "Loaded config: window=%sx%s fps=%s pose_model=%s",
        raw["app"]["window_width"],
        raw["app"]["window_height"],
        raw["app"]["target_fps"],
        raw["pose"]["stage1_model_name"],
    )
    LOGGER.info(
        "Configured stage-2 hand model=%s",
        raw["pose"]["stage2_hand_model_name"],
    )
    LOGGER.info(
        "Liveview ratio=%s circle_radius_percent=%s supported_midi_extensions=%s",
        raw["liveview"]["left_panel_ratio"],
        raw["liveview"]["circle_radius_percent"],
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
            run_interactive_runtime(options, config, resources)
    except StartupError as exc:
        LOGGER.error("Startup initialization failed: %s", exc)
        return 3
    finally:
        if resources is not None:
            resources.cleanup()

    if options.smoke_test:
        LOGGER.info("Subtask 1.1.4 complete: smoke-test path exercised startup subsystems and exited cleanly.")
    else:
        LOGGER.info("Subtask 1.2.6 complete: persistent liveview runtime loop ran until explicit exit and cleaned up successfully.")
    return 0