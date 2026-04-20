"""Smoke-test helpers extracted from app.py to keep startup module compact."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from midi_mover.audio_integration_checks import verify_fluidsynth_profile_effect_settings
from midi_mover.highscore_integration_checks import (
    verify_headshot_crop_margin_behavior,
    verify_highscore_headshot_persistence_behavior,
)
from midi_mover.integration_checks import (
    verify_fingertip_audio_integration,
    verify_gameplay_score_tracking_behavior,
    verify_hit_window_judgment_behavior,
    verify_judgment_to_liveview_flash_behavior,
)
from midi_mover.runtime_loop import LiveviewRuntimeError, render_liveview_frame


def render_liveview_preview(*, resources: Any, config: Any) -> None:
    if resources.window is None:
        raise RuntimeError("Liveview preview failed: pygame window was not initialized.")
    if resources.pygame_module is None:
        raise RuntimeError("Liveview preview failed: pygame module was not initialized.")
    if resources.frame_reader is None:
        raise RuntimeError("Liveview preview failed: camera frame reader was not initialized.")
    if resources.pose_model is None:
        raise RuntimeError("Liveview preview failed: pose model was not initialized.")
    if resources.hand_pose_model is None:
        raise RuntimeError("Liveview preview failed: hand pose model was not initialized.")
    if resources.primary_person_tracker is None:
        raise RuntimeError("Liveview preview failed: primary person tracker was not initialized.")
    if resources.gameplay_keypoint_tracker is None:
        raise RuntimeError("Liveview preview failed: gameplay keypoint tracker was not initialized.")
    if resources.crop_smoother is None:
        raise RuntimeError("Liveview preview failed: crop smoother was not initialized.")
    if resources.interaction_transition_tracker is None:
        raise RuntimeError("Liveview preview failed: interaction transition tracker was not initialized.")
    if resources.circle_visual_tracker is None:
        raise RuntimeError("Liveview preview failed: circle visual tracker was not initialized.")
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
        raise RuntimeError(str(exc)) from exc


def run_startup_smoke_test(
    *,
    resources: Any,
    config: Any,
    midi_dir: Path,
    discover_midi_files: Any,
    choose_random_midi_file: Any,
    build_song_title: Any,
    logger: logging.Logger,
) -> None:
    logger.info("Running startup smoke test.")
    midi_files = discover_midi_files(midi_dir, config)
    selected_midi = choose_random_midi_file(midi_files, config)
    selected_song_title = build_song_title(selected_midi)
    if resources.window is None:
        raise RuntimeError("Smoke test failed: pygame window was not initialized.")
    if resources.camera is None:
        raise RuntimeError("Smoke test failed: camera was not initialized.")
    if resources.pose_model is None:
        raise RuntimeError("Smoke test failed: pose model was not initialized.")
    if resources.hand_pose_model is None:
        raise RuntimeError("Smoke test failed: hand pose model was not initialized.")
    if resources.pygame_module is None or not resources.mixer_initialized:
        raise RuntimeError("Smoke test failed: pygame mixer was not initialized.")
    if resources.gesture_sounds is None:
        raise RuntimeError("Smoke test failed: gesture-to-sound mapping was not initialized.")
    if resources.frame_reader is None:
        raise RuntimeError("Smoke test failed: camera frame reader was not initialized.")
    if resources.primary_person_tracker is None:
        raise RuntimeError("Smoke test failed: primary person tracker was not initialized.")
    if resources.gameplay_keypoint_tracker is None:
        raise RuntimeError("Smoke test failed: gameplay keypoint tracker was not initialized.")
    if resources.crop_smoother is None:
        raise RuntimeError("Smoke test failed: crop smoother was not initialized.")
    if resources.interaction_transition_tracker is None:
        raise RuntimeError("Smoke test failed: interaction transition tracker was not initialized.")
    if resources.circle_visual_tracker is None:
        raise RuntimeError("Smoke test failed: circle visual tracker was not initialized.")
    pygame = resources.pygame_module
    pygame.event.pump()
    render_liveview_preview(resources=resources, config=config)
    verify_fingertip_audio_integration()
    logger.info("Verified fingertip-driven interaction transitions and downstream audio hook integration.")
    verify_hit_window_judgment_behavior()
    logger.info("Verified hit-window hand/lane judgment integration behavior.")
    verify_judgment_to_liveview_flash_behavior()
    logger.info("Verified judgment-to-liveview circle flash integration behavior.")
    verify_gameplay_score_tracking_behavior()
    logger.info("Verified gameplay score tracking behavior for score/combo/hit/miss metrics.")
    verify_fluidsynth_profile_effect_settings()
    logger.info("Verified FluidSynth profile-based reverb/chorus settings across two instrument profiles.")
    verify_headshot_crop_margin_behavior()
    logger.info("Verified highscore headshot crop margin behavior for countdown-complete capture framing.")
    verify_highscore_headshot_persistence_behavior()
    logger.info("Verified highscore headshot persistence writes image path into stored leaderboard row.")
    logger.info(
        "Smoke test touched subsystems successfully: window=%s mixer=%s frame_reader=%s pose_model=%s midi_files=%s.",
        resources.window.get_size(),
        resources.mixer_initialized,
        type(resources.frame_reader).__name__,
        type(resources.pose_model).__name__,
        len(midi_files),
    )
    logger.info("Smoke test random MIDI selection result: %s", selected_midi.name)
    logger.info("Smoke test extracted song title: %s", selected_song_title)
    logger.info("Smoke test completed successfully and exited cleanly.")
