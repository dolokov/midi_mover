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
    stage1_pose_model: str | None
    stage2_hand_model: str | None
    log_level: str
    smoke_test: bool
    midi_inspect: bool
    midi_inspect_max_notes: int
    record: bool


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
        default=Path("~/data/midi_mover/midis"),
        help="Directory containing target MIDI files. Default: ~/data/midi_mover/midis",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to the YAML configuration file. Default: ./config/default.yaml",
    )
    parser.add_argument(
        "--stage1-pose-model",
        type=str,
        default=None,
        help=(
            "Optional override for the stage-1 Ultralytics pose model path/name. "
            "If omitted, the value from YAML config pose.stage1_model_name is used."
        ),
    )
    parser.add_argument(
        "--stage2-hand-model",
        type=str,
        default=None,
        help=(
            "Optional override for the stage-2 Ultralytics hand-keypoint model path/name. "
            "If omitted, the value from YAML config pose.stage2_hand_model_name is used."
        ),
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
    parser.add_argument(
        "--midi-inspect",
        action="store_true",
        help=(
            "Print a development inspection report of parsed MIDI note timing and "
            "gesture-token conversion for the selected song before gameplay."
        ),
    )
    parser.add_argument(
        "--midi-inspect-max-notes",
        type=int,
        default=20,
        help="Maximum number of converted notes to print in --midi-inspect mode. Default: 20.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help=(
            "Record the game window with audio output and save an MP4 to "
            "~/data/midi_mover/recordings when gameplay exits (including Ctrl+C)."
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

    stage1_pose_model = namespace.stage1_pose_model
    if stage1_pose_model is not None and not stage1_pose_model.strip():
        parser.error(
            "--stage1-pose-model cannot be empty. Provide a valid model path/name "
            "or omit the flag to use pose.stage1_model_name from YAML."
        )

    stage2_hand_model = namespace.stage2_hand_model
    if stage2_hand_model is not None and not stage2_hand_model.strip():
        parser.error(
            "--stage2-hand-model cannot be empty. Provide a valid model path/name "
            "or omit the flag to use pose.stage2_hand_model_name from YAML."
        )

    midi_inspect_max_notes = int(namespace.midi_inspect_max_notes)
    if midi_inspect_max_notes <= 0:
        parser.error(
            f"--midi-inspect-max-notes must be a positive integer, got {midi_inspect_max_notes}."
        )

    return StartupOptions(
        camera_id=camera_id,
        midi_dir=midi_dir,
        config_path=config_path,
        stage1_pose_model=stage1_pose_model.strip() if stage1_pose_model is not None else None,
        stage2_hand_model=stage2_hand_model.strip() if stage2_hand_model is not None else None,
        log_level=namespace.log_level,
        smoke_test=bool(namespace.smoke_test),
        midi_inspect=bool(namespace.midi_inspect),
        midi_inspect_max_notes=midi_inspect_max_notes,
        record=bool(namespace.record),
    )