"""Audio startup helpers and FluidSynth immediate-cue playback."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from midi_mover.audio_levels import resolve_unified_audio_levels
from midi_mover.gesture_tokens import normalize_circles_per_hand
from midi_mover.interaction_visuals import InteractionTransitionSnapshot
from midi_mover.song_audio import resolve_shared_fluidsynth_cue_instrument


LOGGER = logging.getLogger("midi_mover")


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
        circles_per_hand: int = 5,
        soundfont_id: int | None = None,
    ) -> "FluidSynthGesturePlaybackController":
        cue_config = _load_fluidsynth_immediate_cue_config(
            audio_config,
            circles_per_hand=circles_per_hand,
        )
        return cls(synth=synth, cue_config=cue_config, soundfont_id=soundfont_id)

    def reset(self, *, fade_ms: int = 0) -> None:  # noqa: ARG002 - parity with pygame controller API
        for token, note in list(self._active_notes_by_token.items()):
            self._note_off(note)
            self._active_notes_by_token.pop(token, None)

    def update(
        self,
        *,
        transition_snapshot: InteractionTransitionSnapshot,
    ) -> tuple[str, ...]:
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


def _load_fluidsynth_immediate_cue_config(
    audio_config: dict[str, Any],
    *,
    circles_per_hand: int,
) -> FluidSynthImmediateCueConfig:
    immediate_cues = audio_config.get("immediate_cues")
    immediate_cues_payload = immediate_cues if isinstance(immediate_cues, dict) else {}
    fluidsynth_payload = immediate_cues_payload.get("fluidsynth")
    fluidsynth_cfg = fluidsynth_payload if isinstance(fluidsynth_payload, dict) else {}

    normalized_count = normalize_circles_per_hand(circles_per_hand)
    default_left_hand_notes = (48, 50, 52, 55, 57)
    default_right_hand_notes = (60, 62, 64, 67, 69)
    default_token_notes: dict[str, int] = {}
    for lane_index in range(1, normalized_count + 1):
        default_token_notes[f"L{lane_index}"] = default_left_hand_notes[lane_index - 1]
        default_token_notes[f"R{lane_index}"] = default_right_hand_notes[lane_index - 1]
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