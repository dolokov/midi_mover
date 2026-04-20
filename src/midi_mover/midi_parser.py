"""MIDI parsing helpers for absolute-time event timelines."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from midi_mover.gesture_tokens import build_gesture_tokens, normalize_circles_per_hand
from midi_mover.midi_mapping_debug import build_mapping_state_debug_section
from midi_mover.midi_token_mapping import build_adaptive_token_sequence, parse_gesture_token
DEFAULT_TEMPO_US_PER_BEAT = 500_000


class MidiParseError(RuntimeError):
    pass

@dataclass(frozen=True)
class TimedMidiEvent:
    absolute_tick: int
    absolute_seconds: float
    message: Any
    track_index: int
    track_event_index: int


@dataclass(frozen=True)
class PlayableMidiNote:
    note_number: int
    channel: int
    track_index: int
    velocity: int
    start_tick: int
    end_tick: int
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)

    @property
    def duration_ms(self) -> float:
        return self.duration_seconds * 1000.0


@dataclass(frozen=True)
class TokenMappedMidiNote:
    playable_note: PlayableMidiNote
    gesture_token: str


@dataclass(frozen=True)
class NormalizedTargetNote:
    target_note_id: str
    token: str
    lane: int
    hand: str
    timestamp_seconds: float
    timestamp_ms: float
    duration_seconds: float
    duration_ms: float
    source_note_number: int
    source_channel: int
    source_track_index: int
    source_velocity: int
    source_start_tick: int
    source_end_tick: int

def parse_midi_events_with_absolute_time(midi_path: Path) -> tuple[TimedMidiEvent, ...]:
    try:
        import mido
    except ModuleNotFoundError as exc:
        raise MidiParseError(
            "mido is required for MIDI parsing but is not installed in the active environment. "
            "Activate conda env 'midi_mover' and install mido."
        ) from exc

    try:
        midi_file = mido.MidiFile(str(midi_path))
    except Exception as exc:
        raise MidiParseError(f"Failed to read MIDI file '{midi_path}': {exc}") from exc

    ticks_per_beat = int(midi_file.ticks_per_beat)
    if ticks_per_beat <= 0:
        raise MidiParseError(
            f"MIDI file '{midi_path}' has invalid ticks_per_beat={ticks_per_beat}."
        )

    try:
        merged_track = mido.merge_tracks(midi_file.tracks)
    except Exception as exc:
        raise MidiParseError(f"Failed to merge MIDI tracks for '{midi_path}': {exc}") from exc

    tick_to_seconds = _build_tick_to_seconds_converter(
        merged_track=merged_track,
        ticks_per_beat=ticks_per_beat,
        mido_module=mido,
    )

    timed_events: list[TimedMidiEvent] = []
    for track_index, track in enumerate(midi_file.tracks):
        absolute_tick = 0
        for track_event_index, message in enumerate(track):
            delta_ticks = int(message.time)
            if delta_ticks < 0:
                raise MidiParseError(
                    f"Encountered negative delta tick ({delta_ticks}) in '{midi_path}'."
                )

            absolute_tick += delta_ticks
            timed_events.append(
                TimedMidiEvent(
                    absolute_tick=absolute_tick,
                    absolute_seconds=tick_to_seconds(absolute_tick),
                    message=message,
                    track_index=track_index,
                    track_event_index=track_event_index,
                )
            )

    timed_events.sort(
        key=lambda event: (
            event.absolute_tick,
            event.track_index,
            event.track_event_index,
        )
    )

    return tuple(timed_events)


def filter_playable_midi_notes(
    *,
    timed_events: tuple[TimedMidiEvent, ...],
    midi_config: dict[str, Any],
) -> tuple[PlayableMidiNote, ...]:
    track_filter = _normalize_optional_int_filter(
        raw_values=midi_config.get("track_filter", []),
        key_name="midi.track_filter",
    )
    channel_filter = _normalize_optional_int_filter(
        raw_values=midi_config.get("channel_filter", []),
        key_name="midi.channel_filter",
    )
    note_filter = _normalize_note_filter(midi_config.get("note_mapping", {}))
    min_duration_ms = _normalize_min_duration_ms(midi_config.get("minimum_note_duration_ms", 0))

    active_notes: dict[tuple[int, int, int], list[TimedMidiEvent]] = {}
    playable_notes: list[PlayableMidiNote] = []

    for timed_event in timed_events:
        message = timed_event.message
        event_type = getattr(message, "type", None)
        if event_type not in {"note_on", "note_off"}:
            continue

        track_index = int(timed_event.track_index)
        if track_filter is not None and track_index not in track_filter:
            continue

        channel = int(getattr(message, "channel", -1))
        if channel_filter is not None and channel not in channel_filter:
            continue

        note_number = int(getattr(message, "note", -1))
        if note_number not in note_filter:
            continue

        velocity = int(getattr(message, "velocity", 0))
        key = (track_index, channel, note_number)
        is_note_on = event_type == "note_on" and velocity > 0
        if is_note_on:
            active_notes.setdefault(key, []).append(timed_event)
            continue

        start_candidates = active_notes.get(key)
        if not start_candidates:
            continue

        note_on_event = start_candidates.pop(0)
        if not start_candidates:
            active_notes.pop(key, None)

        note_duration_ms = (timed_event.absolute_seconds - note_on_event.absolute_seconds) * 1000.0
        if note_duration_ms < min_duration_ms:
            continue

        playable_notes.append(
            PlayableMidiNote(
                note_number=note_number,
                channel=channel,
                track_index=track_index,
                velocity=int(getattr(note_on_event.message, "velocity", 0)),
                start_tick=note_on_event.absolute_tick,
                end_tick=timed_event.absolute_tick,
                start_seconds=note_on_event.absolute_seconds,
                end_seconds=timed_event.absolute_seconds,
            )
        )

    playable_notes.sort(key=lambda note: (note.start_seconds, note.track_index, note.note_number))
    return tuple(playable_notes)


def map_playable_midi_notes_to_gesture_tokens(
    *,
    playable_notes: tuple[PlayableMidiNote, ...],
    midi_config: dict[str, Any],
    circles_per_hand: int = 5,
) -> tuple[TokenMappedMidiNote, ...]:
    allowed_tokens = build_gesture_tokens(circles_per_hand)
    normalized_circles_per_hand = normalize_circles_per_hand(circles_per_hand)
    note_mapping = _normalize_note_mapping_table(
        midi_config.get("note_mapping", {}),
        allowed_tokens=allowed_tokens,
        circles_per_hand=normalized_circles_per_hand,
    )
    adaptive_mode = _normalize_adaptive_mapping_config(midi_config.get("adaptive_mapping"))

    should_use_adaptive_mapping = False
    if adaptive_mode["enabled"]:
        unique_note_numbers = {note.note_number for note in playable_notes}
        should_use_adaptive_mapping = len(unique_note_numbers) > adaptive_mode["trigger_note_count"]

    if should_use_adaptive_mapping:
        adaptive_tokens = build_adaptive_token_sequence(
            note_numbers=tuple(note.note_number for note in playable_notes),
            allowed_tokens=allowed_tokens,
            adaptive_mode_variant=adaptive_mode["mode"],
            creative_seed=adaptive_mode["creative_seed"],
            creative_cycle_span=adaptive_mode["creative_cycle_span"],
        )
        return tuple(
            TokenMappedMidiNote(
                playable_note=playable_note,
                gesture_token=gesture_token,
            )
            for playable_note, gesture_token in zip(playable_notes, adaptive_tokens)
        )

    mapped_notes: list[TokenMappedMidiNote] = []
    for playable_note in playable_notes:
        gesture_token = note_mapping.get(playable_note.note_number)
        if gesture_token is None:
            # ``filter_playable_midi_notes`` should already remove unmapped notes.
            continue
        mapped_notes.append(
            TokenMappedMidiNote(
                playable_note=playable_note,
                gesture_token=gesture_token,
            )
        )

    return tuple(mapped_notes)


def build_normalized_target_notes(
    *,
    mapped_notes: tuple[TokenMappedMidiNote, ...],
    circles_per_hand: int = 5,
) -> tuple[NormalizedTargetNote, ...]:
    sorted_notes = sorted(
        mapped_notes,
        key=lambda mapped: (
            mapped.playable_note.start_seconds,
            mapped.playable_note.track_index,
            mapped.playable_note.note_number,
            mapped.playable_note.channel,
            mapped.playable_note.start_tick,
        ),
    )

    normalized: list[NormalizedTargetNote] = []
    allowed_tokens = build_gesture_tokens(circles_per_hand)
    for index, mapped_note in enumerate(sorted_notes):
        try:
            hand, lane = parse_gesture_token(mapped_note.gesture_token, allowed_tokens=allowed_tokens)
        except ValueError as exc:
            raise MidiParseError(str(exc)) from exc
        source = mapped_note.playable_note
        normalized.append(
            NormalizedTargetNote(
                target_note_id=(
                    f"tn_{index:06d}_{mapped_note.gesture_token}_{source.note_number}_"
                    f"{source.start_tick}_{source.track_index}_{source.channel}"
                ),
                token=mapped_note.gesture_token,
                lane=lane,
                hand=hand,
                timestamp_seconds=source.start_seconds,
                timestamp_ms=source.start_seconds * 1000.0,
                duration_seconds=source.duration_seconds,
                duration_ms=source.duration_ms,
                source_note_number=source.note_number,
                source_channel=source.channel,
                source_track_index=source.track_index,
                source_velocity=source.velocity,
                source_start_tick=source.start_tick,
                source_end_tick=source.end_tick,
            )
        )

    return tuple(normalized)


def build_midi_debug_report(
    *,
    midi_path: Path,
    midi_config: dict[str, Any],
    circles_per_hand: int = 5,
    max_notes: int = 20,
) -> str:
    if max_notes <= 0:
        raise MidiParseError("MIDI debug report max_notes must be greater than zero.")

    timed_events = parse_midi_events_with_absolute_time(midi_path)
    playable_notes = filter_playable_midi_notes(timed_events=timed_events, midi_config=midi_config)
    mapped_notes = map_playable_midi_notes_to_gesture_tokens(
        playable_notes=playable_notes,
        midi_config=midi_config,
        circles_per_hand=circles_per_hand,
    )
    normalized_notes = build_normalized_target_notes(
        mapped_notes=mapped_notes,
        circles_per_hand=circles_per_hand,
    )

    lines = [
        f"MIDI inspection report for: {midi_path}",
        (
            "Counts: "
            f"events={len(timed_events)} "
            f"playable_notes={len(playable_notes)} "
            f"mapped_notes={len(mapped_notes)} "
            f"normalized_notes={len(normalized_notes)}"
        ),
        (
            "Showing first "
            f"{min(max_notes, len(normalized_notes))} normalized notes "
            f"(max_notes={max_notes}):"
        ),
        "idx | time_ms | dur_ms | token | hand | lane | src_note | src_ch | src_track | target_id",
        "----+---------+--------+-------+------+------+----------+--------+-----------+----------------",
    ]

    preview_notes = normalized_notes[:max_notes]
    for index, note in enumerate(preview_notes):
        lines.append(
            " | ".join(
                (
                    f"{index:>3}",
                    f"{note.timestamp_ms:>7.1f}",
                    f"{note.duration_ms:>6.1f}",
                    f"{note.token:>5}",
                    f"{note.hand:>4}",
                    f"{note.lane:>4}",
                    f"{note.source_note_number:>8}",
                    f"{note.source_channel:>6}",
                    f"{note.source_track_index:>9}",
                    note.target_note_id,
                )
            )
        )

    if not preview_notes:
        lines.append("<no normalized target notes generated>")

    if len(normalized_notes) > max_notes:
        lines.append(f"... truncated {len(normalized_notes) - max_notes} additional note(s).")

    lines.extend(
        build_mapping_state_debug_section(
            playable_notes=playable_notes,
            mapped_notes=mapped_notes,
            midi_config=midi_config,
            circles_per_hand=circles_per_hand,
            max_rows=max_notes,
        )
    )

    return "\n".join(lines)


def _build_tick_to_seconds_converter(*, merged_track: Any, ticks_per_beat: int, mido_module: Any) -> Any:
    tempo_changes: list[tuple[int, int]] = [(0, DEFAULT_TEMPO_US_PER_BEAT)]
    absolute_tick = 0

    for message in merged_track:
        delta_ticks = int(message.time)
        if delta_ticks < 0:
            raise MidiParseError(f"Encountered negative delta tick ({delta_ticks}) while building tempo map.")
        absolute_tick += delta_ticks

        if getattr(message, "type", None) == "set_tempo":
            tempo_changes.append((absolute_tick, int(message.tempo)))

    normalized_tempo_changes: list[tuple[int, int]] = []
    for tick, tempo in sorted(tempo_changes, key=lambda item: item[0]):
        if normalized_tempo_changes and normalized_tempo_changes[-1][0] == tick:
            normalized_tempo_changes[-1] = (tick, tempo)
        else:
            normalized_tempo_changes.append((tick, tempo))

    tempo_ticks = [tick for tick, _ in normalized_tempo_changes]
    tempo_values = [tempo for _, tempo in normalized_tempo_changes]

    cumulative_seconds: list[float] = [0.0]
    for index in range(1, len(tempo_ticks)):
        previous_tick = tempo_ticks[index - 1]
        current_tick = tempo_ticks[index]
        delta_ticks = current_tick - previous_tick
        cumulative_seconds.append(
            cumulative_seconds[index - 1]
            + float(
                mido_module.tick2second(
                    delta_ticks,
                    ticks_per_beat=ticks_per_beat,
                    tempo=tempo_values[index - 1],
                )
            )
        )

    def tick_to_seconds(tick: int) -> float:
        if tick < 0:
            raise MidiParseError(f"Cannot convert negative absolute tick ({tick}) to seconds.")
        segment_index = bisect_right(tempo_ticks, tick) - 1
        if segment_index < 0:
            segment_index = 0
        segment_tick = tempo_ticks[segment_index]
        segment_tempo = tempo_values[segment_index]
        extra_ticks = tick - segment_tick
        return cumulative_seconds[segment_index] + float(
            mido_module.tick2second(
                extra_ticks,
                ticks_per_beat=ticks_per_beat,
                tempo=segment_tempo,
            )
        )

    return tick_to_seconds


def _normalize_optional_int_filter(*, raw_values: Any, key_name: str) -> set[int] | None:
    if not isinstance(raw_values, list):
        raise MidiParseError(f"{key_name} must be a list of integers.")
    if not raw_values:
        return None

    normalized: set[int] = set()
    for raw_value in raw_values:
        try:
            normalized.add(int(raw_value))
        except (TypeError, ValueError) as exc:
            raise MidiParseError(f"{key_name} contains a non-integer value: {raw_value!r}.") from exc
    return normalized


def _normalize_note_filter(note_mapping: Any) -> set[int]:
    # Filtering is based on configured note numbers only; token validation is
    # deferred to token-aware mapping steps.
    if not isinstance(note_mapping, dict):
        raise MidiParseError("midi.note_mapping must be a mapping of MIDI note numbers to gesture tokens.")
    normalized: set[int] = set()
    for raw_note_number in note_mapping.keys():
        try:
            normalized.add(int(raw_note_number))
        except (TypeError, ValueError) as exc:
            raise MidiParseError(
                "midi.note_mapping contains an invalid MIDI note number key: "
                f"{raw_note_number!r}."
            ) from exc
    return normalized


def _normalize_note_mapping_table(
    note_mapping: Any,
    *,
    allowed_tokens: tuple[str, ...],
    circles_per_hand: int,
) -> dict[int, str]:
    if not isinstance(note_mapping, dict):
        raise MidiParseError("midi.note_mapping must be a mapping of MIDI note numbers to gesture tokens.")

    normalized: dict[int, str] = {}
    for raw_note_number, raw_token in note_mapping.items():
        try:
            note_number = int(raw_note_number)
        except (TypeError, ValueError) as exc:
            raise MidiParseError(
                "midi.note_mapping contains an invalid MIDI note number key: "
                f"{raw_note_number!r}."
            ) from exc

        token = str(raw_token).strip().upper()
        try:
            token = _normalize_token_for_lane_count(
                token,
                circles_per_hand=circles_per_hand,
            )
        except ValueError as exc:
            raise MidiParseError(
                "midi.note_mapping contains an invalid gesture token value "
                f"for note {note_number}: {raw_token!r}. Expected one of {', '.join(allowed_tokens)}."
            ) from exc
        if token not in allowed_tokens:
            raise MidiParseError(
                "midi.note_mapping contains an invalid gesture token value "
                f"for note {note_number}: {raw_token!r}. Expected one of {', '.join(allowed_tokens)}."
            )
        normalized[note_number] = token

    return normalized


def _normalize_token_for_lane_count(token: str, *, circles_per_hand: int) -> str:
    """Clamp legacy lane tokens (e.g. L5) into active range for smaller lane counts.

    This allows one static config to be reused across 3/4/5-circle modes by
    remapping out-of-range lanes to the highest available lane on the same hand.
    """
    if len(token) < 2:
        raise ValueError("Gesture token is too short.")
    hand = token[0].upper()
    if hand not in {"L", "R"}:
        raise ValueError("Gesture token must start with 'L' or 'R'.")
    try:
        lane = int(token[1:])
    except ValueError as exc:
        raise ValueError("Gesture token lane must be an integer suffix.") from exc
    if lane < 1:
        raise ValueError("Gesture token lane must be >= 1.")
    clamped_lane = min(lane, int(circles_per_hand))
    return f"{hand}{clamped_lane}"


def _normalize_min_duration_ms(raw_value: Any) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise MidiParseError("midi.minimum_note_duration_ms must be numeric.") from exc
    if value < 0:
        raise MidiParseError("midi.minimum_note_duration_ms must be >= 0.")
    return value


def _normalize_adaptive_mapping_config(raw_config: Any) -> dict[str, Any]:
    if raw_config is None:
        return {
            "enabled": False,
            "trigger_note_count": 10,
            "rolling_recent_note_buffer_size": 30,
            "mode": "ranked_low_high",
            "creative_seed": None,
            "creative_cycle_span": 8,
        }
    if not isinstance(raw_config, dict):
        raise MidiParseError("midi.adaptive_mapping must be an object when provided.")

    enabled = raw_config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise MidiParseError("midi.adaptive_mapping.enabled must be a boolean.")

    trigger_note_count = raw_config.get("trigger_note_count", 10)
    try:
        trigger_note_count_int = int(trigger_note_count)
    except (TypeError, ValueError) as exc:
        raise MidiParseError("midi.adaptive_mapping.trigger_note_count must be an integer.") from exc

    if trigger_note_count_int < 1:
        raise MidiParseError("midi.adaptive_mapping.trigger_note_count must be >= 1.")

    rolling_recent_note_buffer_size = raw_config.get("rolling_recent_note_buffer_size", 30)
    try:
        rolling_recent_note_buffer_size_int = int(rolling_recent_note_buffer_size)
    except (TypeError, ValueError) as exc:
        raise MidiParseError(
            "midi.adaptive_mapping.rolling_recent_note_buffer_size must be an integer."
        ) from exc
    if rolling_recent_note_buffer_size_int < 1:
        raise MidiParseError(
            "midi.adaptive_mapping.rolling_recent_note_buffer_size must be >= 1."
        )

    mode = str(raw_config.get("mode", "ranked_low_high")).strip().lower()
    if mode not in {"ranked_low_high", "creative_wave"}:
        raise MidiParseError(
            "midi.adaptive_mapping.mode must be either 'ranked_low_high' or 'creative_wave'."
        )

    creative_seed = raw_config.get("creative_seed", None)
    if creative_seed is not None:
        try:
            creative_seed = int(creative_seed)
        except (TypeError, ValueError) as exc:
            raise MidiParseError("midi.adaptive_mapping.creative_seed must be null or an integer.") from exc

    creative_cycle_span = raw_config.get("creative_cycle_span", 8)
    try:
        creative_cycle_span_int = int(creative_cycle_span)
    except (TypeError, ValueError) as exc:
        raise MidiParseError("midi.adaptive_mapping.creative_cycle_span must be an integer.") from exc
    if creative_cycle_span_int < 2:
        raise MidiParseError("midi.adaptive_mapping.creative_cycle_span must be >= 2.")

    return {
        "enabled": enabled,
        "trigger_note_count": trigger_note_count_int,
        "rolling_recent_note_buffer_size": rolling_recent_note_buffer_size_int,
        "mode": mode,
        "creative_seed": creative_seed,
        "creative_cycle_span": creative_cycle_span_int,
    }


