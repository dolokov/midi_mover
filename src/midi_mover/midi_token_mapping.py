"""Token-space helpers for MIDI note mapping under variable lane counts."""

from __future__ import annotations

import math
from random import Random


def build_adaptive_token_sequence(
    *,
    note_numbers: tuple[int, ...],
    allowed_tokens: tuple[str, ...],
    adaptive_mode_variant: str,
    creative_seed: int | None,
    creative_cycle_span: int,
) -> tuple[str, ...]:
    token_sequence: list[str] = []
    token_count = len(allowed_tokens)
    creative_rng = Random(creative_seed) if creative_seed is not None else None
    unique_note_numbers = sorted(set(note_numbers))
    if not unique_note_numbers:
        return tuple()
    if len(unique_note_numbers) <= 1:
        base_token_by_note = {unique_note_numbers[0]: 0}
    else:
        divisor = len(unique_note_numbers) - 1
        base_token_by_note = {
            note_number: int(round((rank * (token_count - 1)) / divisor))
            for rank, note_number in enumerate(unique_note_numbers)
        }

    for note_index, note_number in enumerate(note_numbers):
        token_index = base_token_by_note[note_number]
        if adaptive_mode_variant == "creative_wave":
            phase = (note_index % creative_cycle_span) / float(creative_cycle_span)
            wave_offset = int(round(2.0 * math.sin(phase * 2.0 * math.pi)))
            seeded_offset = creative_rng.choice((-1, 0, 1)) if creative_rng is not None else 0
            token_index = max(0, min(token_count - 1, token_index + wave_offset + seeded_offset))
        token_sequence.append(allowed_tokens[token_index])
    return tuple(token_sequence)


def parse_gesture_token(token: str, *, allowed_tokens: tuple[str, ...]) -> tuple[str, int]:
    normalized = str(token).strip().upper()
    if normalized not in allowed_tokens:
        raise ValueError(f"Invalid gesture token {token!r}. Expected one of {', '.join(allowed_tokens)}.")
    return normalized[0], int(normalized[1])
