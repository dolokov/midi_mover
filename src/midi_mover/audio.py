"""Audio startup helpers, gesture-sound mapping, and pygame sound generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import logging
from array import array
from typing import Any

from midi_mover.audio_levels import resolve_unified_audio_levels
from midi_mover.interaction_visuals import InteractionTransitionSnapshot
from midi_mover.song_audio import resolve_shared_fluidsynth_cue_instrument


LOGGER = logging.getLogger("midi_mover")

GESTURE_TOKENS: tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5", "R1", "R2", "R3", "R4", "R5")


@dataclass(frozen=True)
class GestureSoundSpec:
    """Config-backed tone specification for one gameplay gesture token."""

    token: str
    frequency_hz: float
    waveform: str


@dataclass(frozen=True)
class GestureSoundMapping:
    """Resolved mapping between all gesture tokens and their sound specifications."""

    specs: dict[str, GestureSoundSpec]

    def validate_complete(self) -> None:
        missing = [token for token in GESTURE_TOKENS if token not in self.specs]
        if missing:
            joined = ", ".join(missing)
            raise AudioStartupError(f"Gesture sound mapping is incomplete. Missing tokens: {joined}.")


@dataclass(frozen=True)
class LoadedGestureSounds:
    """Generated pygame sounds for all gesture tokens."""

    mapping: GestureSoundMapping
    sounds: dict[str, Any]
    playback_config: "AudioPlaybackConfig"

    def sound_for(self, token: str) -> Any:
        try:
            return self.sounds[token]
        except KeyError as exc:
            raise AudioStartupError(f"No generated sound is available for gesture token '{token}'.") from exc


@dataclass(frozen=True)
class AudioStartupStatus:
    """Resolved mixer startup details for logging and smoke-test verification."""

    frequency: int
    sample_size: int
    channels: int
    buffer_size: int
    driver_name: str | None
    init_settings: tuple[int, int, int] | None


class AudioStartupError(RuntimeError):
    """Raised when pygame audio cannot be initialized or verified."""


@dataclass(frozen=True)
class AudioPlaybackConfig:
    """YAML-backed playback settings for gesture-triggered audio."""

    note_duration_seconds: float
    gesture_volume: float
    max_concurrent_sounds: int
    restart_busy_channel: bool
    sustain_while_inside: bool
    release_fade_ms: int


@dataclass
class ActiveGesturePlayback:
    """One currently looping gesture playback bound to a token."""

    token: str
    channel: Any


class GesturePlaybackController:
    """Manage gesture sound lifecycle across enter/stay/exit transitions."""

    def __init__(self) -> None:
        self._active: dict[str, ActiveGesturePlayback] = {}

    def reset(self, *, fade_ms: int = 0) -> None:
        for playback in self._active.values():
            try:
                if fade_ms > 0:
                    playback.channel.fadeout(fade_ms)
                else:
                    playback.channel.stop()
            except Exception:
                LOGGER.exception("Failed to stop active gesture playback for %s during reset.", playback.token)
        self._active.clear()

    def update(
        self,
        *,
        pygame_module: Any,
        transition_snapshot: InteractionTransitionSnapshot,
        gesture_sounds: LoadedGestureSounds,
    ) -> tuple[str, ...]:
        playback_config = gesture_sounds.playback_config
        if not playback_config.sustain_while_inside:
            return trigger_gesture_sounds(
                pygame_module=pygame_module,
                transition_snapshot=transition_snapshot,
                gesture_sounds=gesture_sounds,
            )

        started_tokens: list[str] = []
        active_now: set[str] = set()
        for hand_state in (transition_snapshot.left_hand, transition_snapshot.right_hand):
            for lane_state in hand_state.lane_states:
                if not lane_state.is_inside:
                    continue
                token = lane_state.token
                active_now.add(token)
                existing = self._active.get(token)
                if existing is not None and existing.channel.get_busy():
                    continue

                sound = gesture_sounds.sound_for(token)
                sound.set_volume(playback_config.gesture_volume)
                channel = pygame_module.mixer.find_channel(force=playback_config.restart_busy_channel)
                if channel is None:
                    LOGGER.debug(
                        "Skipped sustained gesture sound for %s because %s concurrent sounds are already active.",
                        token,
                        playback_config.max_concurrent_sounds,
                    )
                    continue
                channel.play(sound, loops=-1)
                self._active[token] = ActiveGesturePlayback(token=token, channel=channel)
                started_tokens.append(token)

        inactive_tokens = [token for token in self._active if token not in active_now]
        for token in inactive_tokens:
            playback = self._active.pop(token)
            if playback_config.release_fade_ms > 0:
                playback.channel.fadeout(playback_config.release_fade_ms)
            else:
                playback.channel.stop()

        if started_tokens:
            LOGGER.debug("Started sustained gesture sounds for active tokens: %s.", ", ".join(started_tokens))
        return tuple(started_tokens)


@dataclass(frozen=True)
class FluidSynthImmediateCueConfig:
    """Config for FluidSynth-based immediate hand-in-circle cues."""

    enabled: bool
    channel: int
    velocity: int
    sustain_while_inside: bool
    bank: int
    program: int
    token_notes: dict[str, int]


class FluidSynthGesturePlaybackController:
    """Drive immediate cue notes directly through FluidSynth while hands stay inside circles."""

    def __init__(
        self,
        *,
        synth: Any,
        cue_config: FluidSynthImmediateCueConfig,
        soundfont_id: int | None = None,
    ) -> None:
        self._synth = synth
        self._cue_config = cue_config
        self._soundfont_id = soundfont_id
        self._active_notes_by_token: dict[str, int] = {}
        self._apply_instrument_preset()

    @property
    def enabled(self) -> bool:
        return bool(self._cue_config.enabled)

    @classmethod
    def from_audio_config(
        cls,
        *,
        synth: Any,
        audio_config: dict[str, Any],
        soundfont_id: int | None = None,
    ) -> "FluidSynthGesturePlaybackController":
        cue_config = _load_fluidsynth_immediate_cue_config(audio_config)
        return cls(synth=synth, cue_config=cue_config, soundfont_id=soundfont_id)

    def reset(self, *, fade_ms: int = 0) -> None:  # noqa: ARG002 - parity with pygame controller API
        for token, note in list(self._active_notes_by_token.items()):
            self._note_off(note)
            self._active_notes_by_token.pop(token, None)

    def update(
        self,
        *,
        pygame_module: Any,
        transition_snapshot: InteractionTransitionSnapshot,
        gesture_sounds: LoadedGestureSounds,
    ) -> tuple[str, ...]:
        del pygame_module, gesture_sounds
        if not self._cue_config.enabled:
            return ()

        active_now: set[str] = set()
        started_tokens: list[str] = []
        if self._cue_config.sustain_while_inside:
            for hand_state in (transition_snapshot.left_hand, transition_snapshot.right_hand):
                for lane_state in hand_state.lane_states:
                    if lane_state.is_inside:
                        active_now.add(lane_state.token)
        else:
            for hand_state in (transition_snapshot.left_hand, transition_snapshot.right_hand):
                for lane_state in hand_state.lane_states:
                    if lane_state.state_name == "enter":
                        active_now.add(lane_state.token)

        for token in sorted(active_now):
            if token in self._active_notes_by_token:
                continue
            note = int(self._cue_config.token_notes[token])
            self._note_on(note)
            self._active_notes_by_token[token] = note
            started_tokens.append(token)

        for token in [token for token in self._active_notes_by_token if token not in active_now]:
            self._note_off(self._active_notes_by_token[token])
            self._active_notes_by_token.pop(token, None)

        return tuple(started_tokens)

    def _apply_instrument_preset(self) -> None:
        if not self._cue_config.enabled:
            return
        program_select = getattr(self._synth, "program_select", None)
        if callable(program_select) and self._soundfont_id is not None:
            try:
                result = int(
                    program_select(
                        int(self._cue_config.channel),
                        int(self._soundfont_id),
                        int(self._cue_config.bank),
                        int(self._cue_config.program),
                    )
                )
                if result != 0:
                    LOGGER.warning(
                        "FluidSynth immediate cue program_select returned non-zero result=%s for channel=%s bank=%s program=%s.",
                        result,
                        self._cue_config.channel,
                        self._cue_config.bank,
                        self._cue_config.program,
                    )
            except Exception:
                LOGGER.exception("Failed to apply FluidSynth immediate cue program_select preset.")
            return

        program_change = getattr(self._synth, "program_change", None)
        if callable(program_change):
            try:
                program_change(int(self._cue_config.channel), int(self._cue_config.program))
            except Exception:
                LOGGER.exception("Failed to apply FluidSynth immediate cue program_change preset.")

    def _note_on(self, note: int) -> None:
        noteon = getattr(self._synth, "noteon", None)
        if not callable(noteon):
            return
        try:
            noteon(int(self._cue_config.channel), int(note), int(self._cue_config.velocity))
        except Exception:
            LOGGER.exception("Failed to trigger FluidSynth immediate cue noteon for note=%s.", note)

    def _note_off(self, note: int) -> None:
        noteoff = getattr(self._synth, "noteoff", None)
        if not callable(noteoff):
            return
        try:
            noteoff(int(self._cue_config.channel), int(note))
        except Exception:
            LOGGER.exception("Failed to trigger FluidSynth immediate cue noteoff for note=%s.", note)


def initialize_audio_output(*, pygame_module: Any, audio_config: dict[str, Any]) -> AudioStartupStatus:
    """Initialize pygame mixer and verify that an audio device is actually usable."""

    if pygame_module is None:
        raise AudioStartupError("Internal error: pygame module is required before audio startup.")

    mixer_cfg = audio_config["mixer"]
    requested_frequency = int(mixer_cfg["frequency"])
    requested_size = int(mixer_cfg["size"])
    requested_channels = int(mixer_cfg["channels"])
    requested_buffer = int(mixer_cfg["buffer"])

    try:
        pygame_module.mixer.init(
            frequency=requested_frequency,
            size=requested_size,
            channels=requested_channels,
            buffer=requested_buffer,
        )
        pygame_module.mixer.set_num_channels(int(mixer_cfg["max_channels"]))
    except Exception as exc:
        raise AudioStartupError(f"Failed to initialize pygame mixer: {exc}") from exc

    try:
        status = verify_audio_output(
            pygame_module=pygame_module,
            requested_buffer=requested_buffer,
        )
    except AudioStartupError:
        try:
            pygame_module.mixer.quit()
        except Exception:
            LOGGER.exception("Failed to roll back pygame mixer after audio verification error.")
        raise

    LOGGER.info(
        "Initialized pygame mixer and verified audio output: frequency=%s size=%s channels=%s buffer=%s driver=%s init=%s.",
        status.frequency,
        status.sample_size,
        status.channels,
        status.buffer_size,
        status.driver_name or "unknown",
        status.init_settings,
    )
    return status


def load_gesture_sound_mapping(audio_config: dict[str, Any]) -> GestureSoundMapping:
    """Load and validate the explicit audio mapping for all gameplay gesture tokens."""

    raw_mapping = audio_config.get("gesture_sounds")
    if not isinstance(raw_mapping, dict):
        raise AudioStartupError("Audio config must define audio.gesture_sounds as a mapping.")

    specs: dict[str, GestureSoundSpec] = {}
    for token in GESTURE_TOKENS:
        raw_spec = raw_mapping.get(token)
        if not isinstance(raw_spec, dict):
            raise AudioStartupError(f"Audio gesture token '{token}' must map to an object spec.")
        try:
            frequency_hz = float(raw_spec["frequency_hz"])
            waveform = str(raw_spec["waveform"]).strip().lower()
        except KeyError as exc:
            raise AudioStartupError(
                f"Audio gesture token '{token}' is missing required key '{exc.args[0]}'."
            ) from exc
        if frequency_hz <= 0:
            raise AudioStartupError(
                f"Audio gesture token '{token}' must have a positive frequency_hz value."
            )
        if waveform not in {"sine"}:
            raise AudioStartupError(
                f"Audio gesture token '{token}' uses unsupported waveform '{waveform}'."
            )
        specs[token] = GestureSoundSpec(token=token, frequency_hz=frequency_hz, waveform=waveform)

    mapping = GestureSoundMapping(specs=specs)
    mapping.validate_complete()
    return mapping


def build_gesture_sounds(
    *,
    pygame_module: Any,
    audio_config: dict[str, Any],
    amplitude: float = 0.35,
) -> LoadedGestureSounds:
    """Generate a short pygame Sound for every gesture token from the YAML mapping."""

    mapping = load_gesture_sound_mapping(audio_config)
    playback_config = _load_playback_config(audio_config)
    mixer_state = pygame_module.mixer.get_init()
    if mixer_state is None:
        raise AudioStartupError("Cannot build gesture sounds before pygame mixer initialization.")

    sample_rate, _, channels = mixer_state
    loaded_sounds: dict[str, Any] = {}
    for token, spec in mapping.specs.items():
        loaded_sounds[token] = _build_tone_sound(
            pygame_module=pygame_module,
            sample_rate=int(sample_rate),
            channels=int(channels),
            frequency_hz=spec.frequency_hz,
            duration_seconds=playback_config.note_duration_seconds,
            amplitude=amplitude,
        )

    pygame_module.mixer.set_num_channels(playback_config.max_concurrent_sounds)
    LOGGER.info("Built gesture-sound mapping for tokens: %s.", ", ".join(sorted(loaded_sounds)))
    return LoadedGestureSounds(mapping=mapping, sounds=loaded_sounds, playback_config=playback_config)


def trigger_gesture_sounds(
    *,
    pygame_module: Any,
    transition_snapshot: InteractionTransitionSnapshot,
    gesture_sounds: LoadedGestureSounds,
) -> tuple[str, ...]:
    """Play gesture sounds for freshly entered hand-circle contacts only."""

    triggered_tokens: list[str] = []
    normalized_volume = gesture_sounds.playback_config.gesture_volume
    for hand_state in (transition_snapshot.left_hand, transition_snapshot.right_hand):
        for lane_state in hand_state.lane_states:
            if lane_state.state_name != "enter":
                continue
            sound = gesture_sounds.sound_for(lane_state.token)
            sound.set_volume(normalized_volume)
            channel = pygame_module.mixer.find_channel(
                force=gesture_sounds.playback_config.restart_busy_channel
            )
            if channel is None:
                LOGGER.debug(
                    "Skipped gesture sound for %s because %s concurrent sounds are already active.",
                    lane_state.token,
                    gesture_sounds.playback_config.max_concurrent_sounds,
                )
                continue
            channel.play(sound)
            triggered_tokens.append(lane_state.token)

    if triggered_tokens:
        LOGGER.debug("Triggered gesture sounds for enter transitions: %s.", ", ".join(triggered_tokens))
    return tuple(triggered_tokens)


def verify_audio_output(*, pygame_module: Any, requested_buffer: int) -> AudioStartupStatus:
    """Validate that pygame mixer reports an active and usable audio device."""

    if not pygame_module.mixer.get_init():
        raise AudioStartupError("pygame mixer did not remain initialized after startup.")

    init_settings = pygame_module.mixer.get_init()
    if init_settings is None:
        raise AudioStartupError("pygame mixer reported no active audio device.")

    try:
        channel = pygame_module.mixer.find_channel(force=True)
    except Exception as exc:
        raise AudioStartupError(f"pygame mixer could not reserve a playback channel: {exc}") from exc

    if channel is None:
        raise AudioStartupError("pygame mixer did not provide any playback channels for audio output.")

    driver_name = None
    try:
        driver_name = pygame_module.mixer.get_driver()
    except Exception:
        driver_name = None

    frequency, sample_size, channels = (int(value) for value in init_settings)
    return AudioStartupStatus(
        frequency=frequency,
        sample_size=sample_size,
        channels=channels,
        buffer_size=int(requested_buffer),
        driver_name=driver_name,
        init_settings=init_settings,
    )


def _load_playback_config(audio_config: dict[str, Any]) -> AudioPlaybackConfig:
    playback = audio_config.get("playback")
    if not isinstance(playback, dict):
        raise AudioStartupError("Audio config must define audio.playback as a mapping.")
    levels = resolve_unified_audio_levels(audio_config)
    return AudioPlaybackConfig(
        note_duration_seconds=max(0.01, float(playback["note_duration_seconds"])),
        gesture_volume=max(0.0, min(float(playback["gesture_volume"]), 1.0)) * float(levels.cue),
        max_concurrent_sounds=max(1, int(playback["max_concurrent_sounds"])),
        restart_busy_channel=bool(playback["restart_busy_channel"]),
        sustain_while_inside=bool(playback["sustain_while_inside"]),
        release_fade_ms=max(0, int(playback["release_fade_ms"])),
    )


def _build_tone_sound(
    *,
    pygame_module: Any,
    sample_rate: int,
    channels: int,
    frequency_hz: float,
    duration_seconds: float,
    amplitude: float,
) -> Any:
    sample_count = max(1, int(sample_rate * duration_seconds))
    peak = max(0, min(int(32767 * amplitude), 32767))
    mono_samples = array(
        "h",
        (
            int(peak * math.sin((2.0 * math.pi * frequency_hz * sample_index) / sample_rate))
            for sample_index in range(sample_count)
        ),
    )

    if channels <= 1:
        return pygame_module.mixer.Sound(buffer=mono_samples.tobytes())

    interleaved = array("h")
    for sample in mono_samples:
        for _ in range(channels):
            interleaved.append(sample)
    return pygame_module.mixer.Sound(buffer=interleaved.tobytes())


def _load_fluidsynth_immediate_cue_config(audio_config: dict[str, Any]) -> FluidSynthImmediateCueConfig:
    immediate_cues = audio_config.get("immediate_cues")
    immediate_cues_payload = immediate_cues if isinstance(immediate_cues, dict) else {}
    fluidsynth_payload = immediate_cues_payload.get("fluidsynth")
    fluidsynth_cfg = fluidsynth_payload if isinstance(fluidsynth_payload, dict) else {}

    default_token_notes: dict[str, int] = {
        "L1": 48,
        "L2": 50,
        "L3": 52,
        "L4": 55,
        "L5": 57,
        "R1": 60,
        "R2": 62,
        "R3": 64,
        "R4": 67,
        "R5": 69,
    }
    configured_token_notes = fluidsynth_cfg.get("token_notes")
    raw_token_notes = configured_token_notes if isinstance(configured_token_notes, dict) else {}
    token_notes = {**default_token_notes}
    for token, raw_note in raw_token_notes.items():
        token_text = str(token).strip().upper()
        if token_text not in token_notes:
            continue
        try:
            normalized_note = int(raw_note)
        except (TypeError, ValueError):
            continue
        token_notes[token_text] = max(0, min(127, normalized_note))

    levels = resolve_unified_audio_levels(audio_config)
    base_velocity = max(1, min(127, int(fluidsynth_cfg.get("velocity", 96))))
    scaled_velocity = max(1, min(127, int(round(base_velocity * float(levels.cue)))))
    enabled = bool(fluidsynth_cfg.get("enabled", True)) and float(levels.cue) > 0.0
    shared_bank, shared_program = resolve_shared_fluidsynth_cue_instrument(audio_config=audio_config)

    return FluidSynthImmediateCueConfig(
        enabled=enabled,
        channel=max(0, min(15, int(fluidsynth_cfg.get("channel", 15)))),
        velocity=scaled_velocity,
        sustain_while_inside=bool(fluidsynth_cfg.get("sustain_while_inside", True)),
        bank=int(shared_bank),
        program=int(shared_program),
        token_notes=token_notes,
    )