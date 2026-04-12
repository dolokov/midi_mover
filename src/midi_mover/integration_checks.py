"""Runtime integration checks for fingertip-driven interaction and audio hooks."""

from __future__ import annotations

from dataclasses import dataclass

from midi_mover.audio import (
    AudioPlaybackConfig,
    GesturePlaybackController,
    GestureSoundMapping,
    GestureSoundSpec,
    LoadedGestureSounds,
)
from midi_mover.circles import CircleGeometry
from midi_mover.hand_keypoints import FingertipSample
from midi_mover.interaction_visuals import HandCircleTransitionTracker, detect_hand_circle_interactions
from midi_mover.midi_parser import (
    TimedMidiEvent,
    build_normalized_target_notes,
    filter_playable_midi_notes,
    map_playable_midi_notes_to_gesture_tokens,
)
from midi_mover.midi_mapping_debug import build_mapping_state_debug_section
from midi_mover.pose import GameplayKeypoints, KeypointSample


@dataclass
class _FakeSound:
    token: str
    volume: float = 0.0

    def set_volume(self, value: float) -> None:
        self.volume = float(value)


@dataclass
class _FakeChannel:
    played_tokens: list[str]
    busy: bool = False

    def get_busy(self) -> bool:
        return self.busy

    def play(self, sound: _FakeSound, loops: int = 0) -> None:
        self.busy = True
        self.played_tokens.append(sound.token)

    def fadeout(self, _: int) -> None:
        self.busy = False

    def stop(self) -> None:
        self.busy = False


@dataclass
class _FakeMixer:
    channels: list[_FakeChannel]

    def find_channel(self, force: bool = False) -> _FakeChannel | None:
        for channel in self.channels:
            if not channel.get_busy():
                return channel
        if force and self.channels:
            return self.channels[0]
        return None


@dataclass
class _FakePygame:
    mixer: _FakeMixer


@dataclass(frozen=True)
class _FakeMidiMessage:
    type: str
    time: int = 0
    channel: int = 0
    note: int = 0
    velocity: int = 0


def verify_fingertip_audio_integration() -> None:
    """Verify fingertip events still drive transition + audio hooks correctly."""

    circles = (
        CircleGeometry(lane=1, center_xy=(100.0, 100.0), radius=20, offset_xy=(0.0, 0.0)),
    )
    left_wrist = KeypointSample(
        name="left_wrist",
        xy=(90.0, 100.0),
        confidence=1.0,
        source="detected",
        observed_at_monotonic=1.0,
    )
    right_wrist = KeypointSample(
        name="right_wrist",
        xy=(200.0, 100.0),
        confidence=1.0,
        source="detected",
        observed_at_monotonic=1.0,
    )
    inside = GameplayKeypoints(
        person_index=0,
        captured_at_monotonic=1.0,
        head_center_xy=(100.0, 80.0),
        left_eye=None,
        right_eye=None,
        left_wrist=left_wrist,
        right_wrist=right_wrist,
        fingertip_samples=(
            FingertipSample(
                hand_index=0,
                fingertip_name="index_tip",
                keypoint_index=8,
                xy=(100.0, 100.0),
                confidence=1.0,
            ),
        ),
    )
    outside = GameplayKeypoints(
        person_index=0,
        captured_at_monotonic=1.1,
        head_center_xy=(100.0, 80.0),
        left_eye=None,
        right_eye=None,
        left_wrist=left_wrist,
        right_wrist=right_wrist,
        fingertip_samples=(
            FingertipSample(
                hand_index=0,
                fingertip_name="index_tip",
                keypoint_index=8,
                xy=(220.0, 220.0),
                confidence=1.0,
            ),
        ),
    )

    playback_config = AudioPlaybackConfig(
        note_duration_seconds=0.1,
        gesture_volume=0.8,
        max_concurrent_sounds=2,
        restart_busy_channel=True,
        sustain_while_inside=True,
        release_fade_ms=0,
    )
    mapping = GestureSoundMapping(
        specs={"L1": GestureSoundSpec(token="L1", frequency_hz=261.63, waveform="sine")}
    )
    sounds = {"L1": _FakeSound(token="L1")}
    gesture_sounds = LoadedGestureSounds(
        mapping=mapping,
        sounds=sounds,
        playback_config=playback_config,
    )
    pygame_module = _FakePygame(mixer=_FakeMixer(channels=[_FakeChannel(played_tokens=[])]))

    transition_tracker = HandCircleTransitionTracker(debounce_ms=0)
    playback_controller = GesturePlaybackController()

    inside_snapshot = detect_hand_circle_interactions(circle_geometries=circles, gameplay_keypoints=inside)
    enter_transitions = transition_tracker.update(inside_snapshot)
    started_tokens = playback_controller.update(
        pygame_module=pygame_module,
        transition_snapshot=enter_transitions,
        gesture_sounds=gesture_sounds,
    )
    if "L1" not in started_tokens:
        raise RuntimeError("Fingertip integration check failed: expected L1 playback on fingertip enter.")

    outside_snapshot = detect_hand_circle_interactions(circle_geometries=circles, gameplay_keypoints=outside)
    exit_transitions = transition_tracker.update(outside_snapshot)
    playback_controller.update(
        pygame_module=pygame_module,
        transition_snapshot=exit_transitions,
        gesture_sounds=gesture_sounds,
    )
    playback_controller.reset()


def verify_midi_note_filtering_rules() -> None:
    """Verify playable-note filtering honors YAML track/channel/note/duration rules."""

    timed_events = (
        TimedMidiEvent(
            absolute_tick=0,
            absolute_seconds=0.00,
            message=_FakeMidiMessage(type="note_on", channel=0, note=60, velocity=96),
            track_index=0,
            track_event_index=0,
        ),
        TimedMidiEvent(
            absolute_tick=12,
            absolute_seconds=0.05,
            message=_FakeMidiMessage(type="note_off", channel=0, note=60, velocity=0),
            track_index=0,
            track_event_index=1,
        ),
        TimedMidiEvent(
            absolute_tick=24,
            absolute_seconds=0.10,
            message=_FakeMidiMessage(type="note_on", channel=0, note=61, velocity=88),
            track_index=0,
            track_event_index=2,
        ),
        TimedMidiEvent(
            absolute_tick=60,
            absolute_seconds=0.22,
            message=_FakeMidiMessage(type="note_off", channel=0, note=61, velocity=0),
            track_index=0,
            track_event_index=3,
        ),
        TimedMidiEvent(
            absolute_tick=72,
            absolute_seconds=0.25,
            message=_FakeMidiMessage(type="note_on", channel=2, note=60, velocity=99),
            track_index=0,
            track_event_index=4,
        ),
        TimedMidiEvent(
            absolute_tick=90,
            absolute_seconds=0.35,
            message=_FakeMidiMessage(type="note_off", channel=2, note=60, velocity=0),
            track_index=0,
            track_event_index=5,
        ),
        TimedMidiEvent(
            absolute_tick=100,
            absolute_seconds=0.40,
            message=_FakeMidiMessage(type="note_on", channel=0, note=60, velocity=91),
            track_index=1,
            track_event_index=0,
        ),
        TimedMidiEvent(
            absolute_tick=130,
            absolute_seconds=0.55,
            message=_FakeMidiMessage(type="note_off", channel=0, note=60, velocity=0),
            track_index=1,
            track_event_index=1,
        ),
    )

    midi_config = {
        "note_mapping": {
            "60": "L1",
            "61": "L2",
        },
        "track_filter": [0],
        "channel_filter": [0],
        "minimum_note_duration_ms": 60,
    }

    playable_notes = filter_playable_midi_notes(
        timed_events=timed_events,
        midi_config=midi_config,
    )
    if len(playable_notes) != 1:
        raise RuntimeError(
            "MIDI filtering integration check failed: expected exactly one playable note "
            f"after track/channel/note/duration filtering, got {len(playable_notes)}."
        )

    filtered_note = playable_notes[0]
    if filtered_note.note_number != 61:
        raise RuntimeError(
            "MIDI filtering integration check failed: expected surviving note number 61, "
            f"got {filtered_note.note_number}."
        )
    if filtered_note.track_index != 0 or filtered_note.channel != 0:
        raise RuntimeError(
            "MIDI filtering integration check failed: surviving note has unexpected "
            f"track/channel ({filtered_note.track_index}, {filtered_note.channel})."
        )

    mapped_notes = map_playable_midi_notes_to_gesture_tokens(
        playable_notes=playable_notes,
        midi_config=midi_config,
    )
    if len(mapped_notes) != 1:
        raise RuntimeError(
            "MIDI mapping integration check failed: expected exactly one mapped note, "
            f"got {len(mapped_notes)}."
        )
    if mapped_notes[0].gesture_token != "L2":
        raise RuntimeError(
            "MIDI mapping integration check failed: expected note 61 to map to L2, "
            f"got {mapped_notes[0].gesture_token}."
        )

    normalized_target_notes = build_normalized_target_notes(mapped_notes=mapped_notes)
    if len(normalized_target_notes) != 1:
        raise RuntimeError(
            "Normalized target-note integration check failed: expected exactly one normalized target note, "
            f"got {len(normalized_target_notes)}."
        )

    target_note = normalized_target_notes[0]
    if target_note.token != "L2" or target_note.hand != "L" or target_note.lane != 2:
        raise RuntimeError(
            "Normalized target-note integration check failed: expected token/hand/lane "
            f"L2/L/2, got {target_note.token}/{target_note.hand}/{target_note.lane}."
        )
    if target_note.source_note_number != 61:
        raise RuntimeError(
            "Normalized target-note integration check failed: expected source note 61, "
            f"got {target_note.source_note_number}."
        )


def verify_midi_adaptive_mapping_for_dense_note_sets() -> None:
    """Verify dense-note adaptive mapping preserves deterministic low-to-high ranking."""

    timed_events: list[TimedMidiEvent] = []
    tick = 0
    event_index = 0
    for note_number in range(60, 72):
        timed_events.append(
            TimedMidiEvent(
                absolute_tick=tick,
                absolute_seconds=tick * 0.01,
                message=_FakeMidiMessage(type="note_on", channel=0, note=note_number, velocity=90),
                track_index=0,
                track_event_index=event_index,
            )
        )
        event_index += 1
        tick += 8
        timed_events.append(
            TimedMidiEvent(
                absolute_tick=tick,
                absolute_seconds=tick * 0.01,
                message=_FakeMidiMessage(type="note_off", channel=0, note=note_number, velocity=0),
                track_index=0,
                track_event_index=event_index,
            )
        )
        event_index += 1
        tick += 2

    midi_config = {
        "note_mapping": {str(note): "L1" for note in range(60, 72)},
        "track_filter": [],
        "channel_filter": [],
        "minimum_note_duration_ms": 1,
        "adaptive_mapping": {
            "enabled": True,
            "trigger_note_count": 10,
        },
    }

    playable_notes = filter_playable_midi_notes(
        timed_events=tuple(timed_events),
        midi_config=midi_config,
    )
    mapped_notes = map_playable_midi_notes_to_gesture_tokens(
        playable_notes=playable_notes,
        midi_config=midi_config,
    )
    normalized_target_notes = build_normalized_target_notes(mapped_notes=mapped_notes)

    if len(normalized_target_notes) != 12:
        raise RuntimeError(
            "Adaptive mapping integration check failed: expected 12 normalized notes, "
            f"got {len(normalized_target_notes)}."
        )

    expected_sequence = (
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R5",
        "R5",
    )
    actual_sequence = tuple(note.token for note in normalized_target_notes)
    if actual_sequence != expected_sequence:
        raise RuntimeError(
            "Adaptive mapping integration check failed: unexpected token sequence for dense-note remap. "
            f"expected={expected_sequence} actual={actual_sequence}"
        )


def verify_midi_adaptive_mapping_rolling_buffer() -> None:
    """Verify adaptive mapping recalculates low/high ranking from rolling-buffer context."""

    note_sequence = (60, 61, 62, 63, 60)
    timed_events: list[TimedMidiEvent] = []
    tick = 0
    event_index = 0
    for note_number in note_sequence:
        timed_events.append(
            TimedMidiEvent(
                absolute_tick=tick,
                absolute_seconds=tick * 0.01,
                message=_FakeMidiMessage(type="note_on", channel=0, note=note_number, velocity=90),
                track_index=0,
                track_event_index=event_index,
            )
        )
        event_index += 1
        tick += 8
        timed_events.append(
            TimedMidiEvent(
                absolute_tick=tick,
                absolute_seconds=tick * 0.01,
                message=_FakeMidiMessage(type="note_off", channel=0, note=note_number, velocity=0),
                track_index=0,
                track_event_index=event_index,
            )
        )
        event_index += 1
        tick += 2

    midi_config = {
        "note_mapping": {str(note): "L1" for note in range(60, 64)},
        "track_filter": [],
        "channel_filter": [],
        "minimum_note_duration_ms": 1,
        "adaptive_mapping": {
            "enabled": True,
            "trigger_note_count": 3,
            "rolling_recent_note_buffer_size": 3,
        },
    }

    playable_notes = filter_playable_midi_notes(
        timed_events=tuple(timed_events),
        midi_config=midi_config,
    )
    mapped_notes = map_playable_midi_notes_to_gesture_tokens(
        playable_notes=playable_notes,
        midi_config=midi_config,
    )

    actual_sequence = tuple(note.gesture_token for note in mapped_notes)
    expected_sequence = ("L1", "L2", "L3", "L3", "L1")
    if actual_sequence != expected_sequence:
        raise RuntimeError(
            "Adaptive mapping rolling-buffer integration check failed: expected remap after note eviction. "
            f"expected={expected_sequence} actual={actual_sequence}"
        )


def verify_midi_adaptive_mapping_creative_mode_seeded_determinism() -> None:
    """Verify creative adaptive mode is deterministic for a fixed seed and config."""

    note_sequence = (60, 64, 67, 62, 69, 61, 65, 68, 63, 66, 70, 71)
    timed_events: list[TimedMidiEvent] = []
    tick = 0
    event_index = 0
    for note_number in note_sequence:
        timed_events.append(
            TimedMidiEvent(
                absolute_tick=tick,
                absolute_seconds=tick * 0.01,
                message=_FakeMidiMessage(type="note_on", channel=0, note=note_number, velocity=90),
                track_index=0,
                track_event_index=event_index,
            )
        )
        event_index += 1
        tick += 8
        timed_events.append(
            TimedMidiEvent(
                absolute_tick=tick,
                absolute_seconds=tick * 0.01,
                message=_FakeMidiMessage(type="note_off", channel=0, note=note_number, velocity=0),
                track_index=0,
                track_event_index=event_index,
            )
        )
        event_index += 1
        tick += 2

    base_midi_config = {
        "note_mapping": {str(note): "L1" for note in range(60, 72)},
        "track_filter": [],
        "channel_filter": [],
        "minimum_note_duration_ms": 1,
        "adaptive_mapping": {
            "enabled": True,
            "trigger_note_count": 10,
            "rolling_recent_note_buffer_size": 6,
            "mode": "creative_wave",
            "creative_seed": 123,
            "creative_cycle_span": 8,
        },
    }

    playable_notes = filter_playable_midi_notes(
        timed_events=tuple(timed_events),
        midi_config=base_midi_config,
    )

    mapped_first = map_playable_midi_notes_to_gesture_tokens(
        playable_notes=playable_notes,
        midi_config=base_midi_config,
    )
    mapped_second = map_playable_midi_notes_to_gesture_tokens(
        playable_notes=playable_notes,
        midi_config=base_midi_config,
    )

    first_sequence = tuple(note.gesture_token for note in mapped_first)
    second_sequence = tuple(note.gesture_token for note in mapped_second)
    if first_sequence != second_sequence:
        raise RuntimeError(
            "Creative adaptive mapping integration check failed: token sequence was not deterministic "
            f"for fixed seed/config. first={first_sequence} second={second_sequence}"
        )

    changed_seed_config = {
        **base_midi_config,
        "adaptive_mapping": {
            **base_midi_config["adaptive_mapping"],
            "creative_seed": 999,
        },
    }
    mapped_third = map_playable_midi_notes_to_gesture_tokens(
        playable_notes=playable_notes,
        midi_config=changed_seed_config,
    )
    third_sequence = tuple(note.gesture_token for note in mapped_third)
    if third_sequence == first_sequence:
        raise RuntimeError(
            "Creative adaptive mapping integration check failed: expected a seed change to alter "
            f"the creative token sequence. sequence={first_sequence}"
        )


def verify_midi_adaptive_mapping_state_introspection_output() -> None:
    """Verify adaptive mapping debug output shows rolling buffer + assignment state."""

    note_sequence = (60, 62, 64, 65, 67)
    timed_events: list[TimedMidiEvent] = []
    tick = 0
    event_index = 0
    for note_number in note_sequence:
        timed_events.append(
            TimedMidiEvent(
                absolute_tick=tick,
                absolute_seconds=tick * 0.01,
                message=_FakeMidiMessage(type="note_on", channel=0, note=note_number, velocity=90),
                track_index=0,
                track_event_index=event_index,
            )
        )
        event_index += 1
        tick += 8
        timed_events.append(
            TimedMidiEvent(
                absolute_tick=tick,
                absolute_seconds=tick * 0.01,
                message=_FakeMidiMessage(type="note_off", channel=0, note=note_number, velocity=0),
                track_index=0,
                track_event_index=event_index,
            )
        )
        event_index += 1
        tick += 2

    midi_config = {
        "note_mapping": {str(note): "L1" for note in range(60, 68)},
        "track_filter": [],
        "channel_filter": [],
        "minimum_note_duration_ms": 1,
        "adaptive_mapping": {
            "enabled": True,
            "trigger_note_count": 3,
            "rolling_recent_note_buffer_size": 3,
            "mode": "ranked_low_high",
        },
    }

    playable_notes = filter_playable_midi_notes(timed_events=tuple(timed_events), midi_config=midi_config)
    mapped_notes = map_playable_midi_notes_to_gesture_tokens(
        playable_notes=playable_notes,
        midi_config=midi_config,
    )
    lines = build_mapping_state_debug_section(
        playable_notes=playable_notes,
        mapped_notes=mapped_notes,
        midi_config=midi_config,
        max_rows=10,
    )
    rendered = "\n".join(lines)
    if "rolling buffer" not in rendered.lower() or "ranked_assignment" not in rendered:
        raise RuntimeError(
            "Adaptive mapping introspection integration check failed: expected rolling-buffer and "
            f"assignment details in debug output. output={rendered!r}"
        )
