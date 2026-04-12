"""Helpers for loading normalized timeline target notes from MIDI files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from midi_mover.midi_parser import (
    NormalizedTargetNote,
    build_normalized_target_notes,
    filter_playable_midi_notes,
    map_playable_midi_notes_to_gesture_tokens,
    parse_midi_events_with_absolute_time,
)


def load_normalized_target_notes_from_midi(
    *,
    midi_path: Path,
    midi_config: dict[str, Any],
) -> tuple[NormalizedTargetNote, ...]:
    """Parse a MIDI file and return normalized target notes for timeline rendering."""

    timed_events = parse_midi_events_with_absolute_time(midi_path)
    playable_notes = filter_playable_midi_notes(
        timed_events=timed_events,
        midi_config=midi_config,
    )
    mapped_notes = map_playable_midi_notes_to_gesture_tokens(
        playable_notes=playable_notes,
        midi_config=midi_config,
    )
    return build_normalized_target_notes(mapped_notes=mapped_notes)
