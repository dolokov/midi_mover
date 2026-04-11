"""CLI parsing and startup validation for midi_mover."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StartupOptions:
    camera_id: int
    midi_dir: Path
    config_path: Path
    log_level: str
    smoke_test: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="midi_mover",
        description=(
            "Launch the midi_mover startup skeleton with explicit CLI inputs "
            "for camera, MIDI folder, and YAML config."
        ),
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="OpenCV camera index to open. Must be a non-negative integer. Default: 0.",
    )
    parser.add_argument(
        "--midi-dir",
        type=Path,
        default=Path("midi"),
        help="Directory containing target MIDI files. Default: ./midi",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to the YAML configuration file. Default: ./config/default.yaml",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Startup logging verbosity. Default: INFO.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run the required startup smoke-test path: initialize the main subsystems, "
            "perform lightweight validation, and exit cleanly without entering gameplay."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> StartupOptions:
    parser = build_parser()
    namespace = parser.parse_args(argv)

    camera_id = namespace.camera_id
    if camera_id < 0:
        parser.error(
            f"--camera-id must be a non-negative integer, but got {camera_id}. "
            "Try values like 0 or 1 depending on your connected webcam."
        )

    midi_dir = namespace.midi_dir.expanduser().resolve()
    if not midi_dir.exists():
        parser.error(
            f"MIDI directory does not exist: {midi_dir}. "
            "Create it or pass a valid folder with --midi-dir."
        )
    if not midi_dir.is_dir():
        parser.error(f"--midi-dir must point to a directory, but got: {midi_dir}.")

    config_path = namespace.config.expanduser().resolve()
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        parser.error(
            f"--config must point to a YAML file ending in .yaml or .yml, but got: {config_path}."
        )
    if not config_path.exists():
        parser.error(
            f"Config file does not exist: {config_path}. "
            "Create it or pass a valid YAML file with --config."
        )
    if not config_path.is_file():
        parser.error(f"--config must point to a file, but got: {config_path}.")

    return StartupOptions(
        camera_id=camera_id,
        midi_dir=midi_dir,
        config_path=config_path,
        log_level=namespace.log_level,
        smoke_test=bool(namespace.smoke_test),
    )