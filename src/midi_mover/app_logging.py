"""Logging helpers for application startup and configuration summaries."""

from __future__ import annotations

import logging

from midi_mover.cli import StartupOptions
from midi_mover.config import AppConfig


LOGGER = logging.getLogger("midi_mover")


def log_startup(options: StartupOptions) -> None:
    """Emit startup arguments in a stable, readable format."""
    LOGGER.info("Starting midi_mover startup skeleton")
    LOGGER.info("camera_id=%s", options.camera_id)
    LOGGER.info("midi_dir=%s", options.midi_dir)
    LOGGER.info("config_path=%s", options.config_path)
    LOGGER.info("stage1_pose_model_override=%s", options.stage1_pose_model or "<config>")
    LOGGER.info("stage2_hand_model_override=%s", options.stage2_hand_model or "<config>")
    LOGGER.info("smoke_test=%s", options.smoke_test)
    LOGGER.info(
        "midi_inspect=%s midi_inspect_max_notes=%s",
        options.midi_inspect,
        options.midi_inspect_max_notes,
    )


def log_config_summary(config: AppConfig) -> None:
    """Log high-value runtime configuration values used during startup."""
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
