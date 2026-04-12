"""Helpers for discovering MIDI files from content folders."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Iterable, Sequence


class MidiDiscoveryError(ValueError):
    """Raised when MIDI folder discovery fails with a user-facing validation error."""


def normalize_supported_extensions(raw_extensions: Iterable[str]) -> tuple[str, ...]:
    """Normalize configured extension values into canonical lowercase suffixes."""
    normalized: list[str] = []
    for raw_extension in raw_extensions:
        extension = str(raw_extension).strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        normalized.append(extension)
    return tuple(dict.fromkeys(normalized))


def discover_supported_midi_files(
    midi_dir: Path,
    *,
    supported_extensions: Iterable[str],
) -> list[Path]:
    """Return sorted MIDI files in ``midi_dir`` filtered by configured extensions."""
    normalized_extensions = set(normalize_supported_extensions(supported_extensions))
    if not normalized_extensions:
        raise MidiDiscoveryError(
            "No MIDI supported extensions are configured. "
            "Set midi.supported_extensions in YAML (for example: .mid, .midi)."
        )

    if not midi_dir.exists():
        raise MidiDiscoveryError(
            f"MIDI folder does not exist: {midi_dir}. "
            "Create it or pass a valid folder with --midi-dir."
        )
    if not midi_dir.is_dir():
        raise MidiDiscoveryError(f"MIDI path is not a directory: {midi_dir}.")

    try:
        entries = sorted(midi_dir.iterdir(), key=lambda path: path.name.lower())
    except OSError as exc:
        raise MidiDiscoveryError(
            f"Cannot read MIDI folder {midi_dir}: {exc}. "
            "Check folder permissions and try again."
        ) from exc

    files = [path for path in entries if path.is_file()]
    if not files:
        raise MidiDiscoveryError(
            f"MIDI folder is empty: {midi_dir}. "
            f"Add at least one MIDI file with extension {', '.join(sorted(normalized_extensions))}."
        )

    supported_files: list[Path] = []
    unreadable_supported_files: list[Path] = []
    unsupported_files: list[Path] = []

    for path in files:
        if path.suffix.lower() not in normalized_extensions:
            unsupported_files.append(path)
            continue

        try:
            with path.open("rb") as handle:
                handle.read(1)
        except OSError:
            unreadable_supported_files.append(path)
            continue

        supported_files.append(path)

    if unreadable_supported_files:
        unreadable_names = ", ".join(path.name for path in unreadable_supported_files)
        raise MidiDiscoveryError(
            "Found unreadable MIDI file(s): "
            f"{unreadable_names}. Check file permissions and try again."
        )

    if not supported_files:
        detected = ", ".join(path.name for path in unsupported_files[:10])
        if len(unsupported_files) > 10:
            detected += ", ..."
        raise MidiDiscoveryError(
            f"No supported MIDI files found in {midi_dir}. "
            f"Supported extensions: {', '.join(sorted(normalized_extensions))}. "
            f"Detected files: {detected or '<none>'}."
        )

    return sorted(supported_files)


class MidiRandomSelector:
    """Pick random MIDI files with optional deterministic seeded behavior."""

    def __init__(self, midi_files: Sequence[Path], *, random_seed: int | None) -> None:
        if not midi_files:
            raise MidiDiscoveryError("Cannot select a MIDI file from an empty list.")

        self._midi_files = tuple(midi_files)
        self._random_seed = random_seed
        self._rng = random.Random(random_seed)

    @property
    def random_seed(self) -> int | None:
        return self._random_seed

    def choose(self) -> Path:
        return self._rng.choice(self._midi_files)


def extract_song_title_from_filename(midi_path: Path) -> str:
    """Build a readable song title from a MIDI filename stem."""
    stem = midi_path.stem.strip()
    if not stem:
        return "Untitled Song"

    normalized = re.sub(r"[_\-]+", " ", stem)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return "Untitled Song"

    return normalized.title()
