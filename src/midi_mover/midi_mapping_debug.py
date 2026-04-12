"""Debug/introspection helpers for adaptive MIDI note-to-gesture mapping."""

from __future__ import annotations

from collections import deque
from random import Random
from typing import Any
import math


GESTURE_TOKENS: tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5", "R1", "R2", "R3", "R4", "R5")


def build_mapping_state_debug_section(
    *,
    playable_notes: tuple[Any, ...],
    mapped_notes: tuple[Any, ...],
    midi_config: dict[str, Any],
    max_rows: int,
) -> list[str]:
    adaptive = midi_config.get("adaptive_mapping") or {}
    enabled = bool(adaptive.get("enabled", False))
    trigger_note_count = int(adaptive.get("trigger_note_count", 10))
    rolling_size = max(1, int(adaptive.get("rolling_recent_note_buffer_size", 30)))
    mode = str(adaptive.get("mode", "ranked_low_high")).strip().lower()
    creative_seed = adaptive.get("creative_seed")
    creative_span = max(2, int(adaptive.get("creative_cycle_span", 8)))

    unique_notes = {int(note.note_number) for note in playable_notes}
    adaptive_active = enabled and len(unique_notes) > trigger_note_count
    if not adaptive_active:
        return [
            "",
            "Adaptive mapping introspection: disabled or not triggered for this song.",
        ]

    recent_note_numbers: deque[int] = deque(maxlen=rolling_size)
    token_count = len(GESTURE_TOKENS)
    rng = Random(int(creative_seed)) if creative_seed is not None else None

    lines = [
        "",
        "Adaptive mapping introspection (rolling buffer and assignment timeline):",
        (
            "idx | src_note | buffer | ranked_assignment | chosen_token"
        ),
        "----+----------+--------+-------------------+-------------",
    ]

    for idx, (playable, mapped) in enumerate(zip(playable_notes[:max_rows], mapped_notes[:max_rows])):
        note_number = int(playable.note_number)
        recent_note_numbers.append(note_number)
        ranked_recent_notes = sorted(set(recent_note_numbers))

        assignment_chunks: list[str] = []
        for rank, ranked_note in enumerate(ranked_recent_notes):
            if len(ranked_recent_notes) <= token_count:
                token_index = rank
            else:
                token_index = int(round((rank * (token_count - 1)) / (len(ranked_recent_notes) - 1)))

            if mode == "creative_wave":
                phase = (idx % creative_span) / float(creative_span)
                wave_offset = int(round(2.0 * math.sin(phase * 2.0 * math.pi)))
                seeded_offset = rng.choice((-1, 0, 1)) if rng is not None else 0
                token_index = max(0, min(token_count - 1, token_index + wave_offset + seeded_offset))

            assignment_chunks.append(f"{ranked_note}->{GESTURE_TOKENS[token_index]}")

        lines.append(
            " | ".join(
                (
                    f"{idx:>3}",
                    f"{note_number:>8}",
                    f"{list(recent_note_numbers)!s:>6}",
                    f"{', '.join(assignment_chunks)}",
                    str(mapped.gesture_token),
                )
            )
        )

    if len(playable_notes) > max_rows:
        lines.append(f"... truncated {len(playable_notes) - max_rows} additional adaptive mapping state row(s).")

    return lines