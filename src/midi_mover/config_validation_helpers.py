"""Shared validation helpers for config.py to keep file size manageable."""

from __future__ import annotations

from typing import Any, Callable


def validate_and_normalize_circle_offsets_by_count(
    *,
    liveview_payload: dict[str, Any],
    key: str,
    normalize_mapping_key: Callable[[Any], str],
    require_vector2: Callable[[Any, str], None],
    error_type: type[Exception],
) -> None:
    raw_by_count: dict[str, Any] = liveview_payload[key]
    normalized_by_count: dict[str, Any] = {
        normalize_mapping_key(k): value for k, value in raw_by_count.items()
    }
    liveview_payload[key] = normalized_by_count
    full_key = f"liveview.{key}"
    if sorted(normalized_by_count.keys()) != ["3", "4", "5"]:
        raise error_type(f"{full_key} must define exactly the string keys '3', '4', and '5'.")

    for count_key, offsets in normalized_by_count.items():
        if not isinstance(offsets, dict):
            raise error_type(f"{full_key}.{count_key} must be a mapping of lane keys to [x, y] offsets.")
        normalized_offsets = {
            normalize_mapping_key(k): value for k, value in offsets.items()
        }
        normalized_by_count[count_key] = normalized_offsets
        expected_lanes = [str(index) for index in range(1, int(count_key) + 1)]
        if sorted(normalized_offsets.keys()) != expected_lanes:
            raise error_type(
                f"{full_key}.{count_key} must define exactly the lane keys {', '.join(expected_lanes)}."
            )
        for lane_key, offset in normalized_offsets.items():
            require_vector2(offset, f"{full_key}.{count_key}.{lane_key}")


def validate_fluidsynth_immediate_cue_token_notes(
    *,
    token_notes: dict[str, Any],
    required_tokens: tuple[str, ...],
    normalize_mapping_key: Callable[[Any], str],
    require_type: Callable[[Any, type, str], None],
    error_type: type[Exception],
) -> None:
    normalized = {normalize_mapping_key(key): value for key, value in token_notes.items()}
    token_notes.clear()
    token_notes.update(normalized)
    missing = [token for token in required_tokens if token not in normalized]
    if missing:
        token_text = ", ".join(required_tokens)
        raise error_type(
            "audio.immediate_cues.fluidsynth.token_notes must define at least the active gesture tokens "
            f"{token_text}. Missing: {', '.join(missing)}."
        )
    for token in required_tokens:
        note_number = normalized[token]
        require_type(note_number, int, f"audio.immediate_cues.fluidsynth.token_notes.{token}")
        if not 0 <= int(note_number) <= 127:
            raise error_type(
                f"Config key audio.immediate_cues.fluidsynth.token_notes.{token} must be between 0 and 127."
            )
