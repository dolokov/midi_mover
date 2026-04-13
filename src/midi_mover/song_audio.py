"""Target-song audio backend abstraction and backend implementations."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Protocol

from midi_mover.audio_levels import resolve_unified_audio_levels
from midi_mover.song_audio_scheduler import (
    FluidSynthPlaybackSchedulerError,
    start_scheduled_fluidsynth_playback,
)


LOGGER = logging.getLogger("midi_mover")


class SongAudioBackendError(RuntimeError):
    """Raised when target-song audio backend setup or playback fails."""


class SongAudioBackend(Protocol):
    """Backend contract for target-song playback."""

    backend_name: str

    def start_song(self, midi_path: Path) -> None:
        """Begin target-song playback for the selected MIDI path."""

    def stop_song(self) -> None:
        """Stop active target-song playback if one is running."""

    def shutdown(self) -> None:
        """Release backend resources."""


@dataclass
class PygameSongAudioBackend:
    """Target-song backend implemented with pygame.mixer.music."""

    pygame_module: Any
    playback_gain: float
    backend_name: str = "pygame"

    def start_song(self, midi_path: Path) -> None:
        try:
            self.pygame_module.mixer.music.load(str(midi_path))
            self.pygame_module.mixer.music.set_volume(self.playback_gain)
            self.pygame_module.mixer.music.play()
        except Exception as exc:
            raise SongAudioBackendError(
                f"Failed to start target-song playback with pygame backend for '{midi_path}': {exc}"
            ) from exc

    def stop_song(self) -> None:
        try:
            self.pygame_module.mixer.music.stop()
        except Exception as exc:
            raise SongAudioBackendError(f"Failed to stop pygame target-song playback: {exc}") from exc

    def shutdown(self) -> None:
        self.stop_song()
        try:
            unload = getattr(self.pygame_module.mixer.music, "unload", None)
            if callable(unload):
                unload()
        except Exception as exc:
            raise SongAudioBackendError(f"Failed to unload pygame target-song playback: {exc}") from exc


@dataclass
class FluidSynthSongAudioBackend:
    """Target-song backend implemented with pyfluidsynth + SoundFont synthesis."""

    fluidsynth_module: Any
    synth: Any
    soundfont_id: int
    soundfont_path: Path
    song_volume: float
    cue_channel: int
    backend_name: str = "pyfluidsynth"
    _active_player: Any = None

    @classmethod
    def initialize_from_config(cls, *, audio_config: dict[str, Any]) -> "FluidSynthSongAudioBackend":
        backend_config = (
            audio_config.get("fluidsynth")
            if isinstance(audio_config.get("fluidsynth"), dict)
            else {}
        )
        levels = resolve_unified_audio_levels(audio_config)
        soundfont_path_text = str(backend_config.get("soundfont_path", "")).strip()
        if not soundfont_path_text:
            raise SongAudioBackendError(
                "audio.backend is 'pyfluidsynth' but audio.fluidsynth.soundfont_path is missing or empty. "
                "Set it to a readable .sf2 SoundFont path in config/default.yaml."
            )

        soundfont_path = Path(soundfont_path_text).expanduser().resolve()
        if not soundfont_path.exists() or not soundfont_path.is_file():
            raise SongAudioBackendError(
                "Failed to initialize pyfluidsynth backend: configured SoundFont file does not exist: "
                f"'{soundfont_path}'. Update audio.fluidsynth.soundfont_path to a valid .sf2 file."
            )

        try:
            import fluidsynth  # type: ignore
        except ModuleNotFoundError as exc:
            raise SongAudioBackendError(
                "audio.backend is 'pyfluidsynth' but the Python package 'pyfluidsynth' is not installed. "
                "Install it in conda env 'midi_mover' and ensure libfluidsynth is available on the system."
            ) from exc

        try:
            synth = fluidsynth.Synth()
        except Exception as exc:
            raise SongAudioBackendError(
                "Failed to create FluidSynth Synth instance. Verify that libfluidsynth is installed and loadable. "
                f"Underlying error: {exc}"
            ) from exc

        try:
            start = getattr(synth, "start", None)
            if callable(start):
                start()
            sfload = getattr(synth, "sfload", None)
            if not callable(sfload):
                raise SongAudioBackendError(
                    "pyfluidsynth backend is unavailable because Synth.sfload is missing in this binding. "
                    "Upgrade/reinstall pyfluidsynth with SoundFont support."
                )
            soundfont_id = int(sfload(str(soundfont_path)))
            if soundfont_id < 0:
                raise SongAudioBackendError(
                    "FluidSynth could not load the configured SoundFont file "
                    f"'{soundfont_path}'. Ensure it is a valid .sf2 file readable by the current user."
                )
            channel_presets = resolve_fluidsynth_channel_presets(audio_config=audio_config)
            _apply_fluidsynth_channel_presets(
                synth=synth,
                soundfont_id=soundfont_id,
                channel_presets=channel_presets,
            )
            effects = resolve_fluidsynth_effects_settings(audio_config=audio_config)
            _apply_fluidsynth_effects(synth=synth, effects=effects)
        except SongAudioBackendError:
            delete = getattr(synth, "delete", None)
            if callable(delete):
                delete()
            raise
        except Exception as exc:
            delete = getattr(synth, "delete", None)
            if callable(delete):
                delete()
            raise SongAudioBackendError(
                "Failed while starting FluidSynth or loading the configured SoundFont "
                f"'{soundfont_path}'. Underlying error: {exc}"
            ) from exc

        return cls(
            fluidsynth_module=fluidsynth,
            synth=synth,
            soundfont_id=soundfont_id,
            soundfont_path=soundfont_path,
            song_volume=float(levels.song),
            cue_channel=_resolve_fluidsynth_cue_channel(audio_config),
        )

    def start_song(self, midi_path: Path) -> None:
        self.stop_song()
        self._apply_song_channel_volume()
        if not midi_path.exists() or not midi_path.is_file():
            raise SongAudioBackendError(
                f"Cannot start pyfluidsynth song playback: MIDI file does not exist: '{midi_path}'."
            )

        player_cls = getattr(self.fluidsynth_module, "Player", None)
        if player_cls is not None:
            try:
                player = player_cls(self.synth)
                player.add(str(midi_path))
                player.play()
                self._active_player = player
                return
            except Exception as exc:
                raise SongAudioBackendError(
                    "Failed to start MIDI playback via pyfluidsynth Player for "
                    f"'{midi_path}'. Underlying error: {exc}"
                ) from exc

        midi_file_play = getattr(self.synth, "midi_file_play", None)
        if callable(midi_file_play):
            try:
                midi_file_play(str(midi_path))
                return
            except Exception as exc:
                raise SongAudioBackendError(
                    "Failed to start MIDI playback via Synth.midi_file_play for "
                    f"'{midi_path}'. Underlying error: {exc}"
                ) from exc

        try:
            self._active_player = start_scheduled_fluidsynth_playback(
                synth=self.synth,
                midi_path=midi_path,
            )
            LOGGER.info(
                "Started pyfluidsynth scheduled MIDI playback worker for '%s' "
                "because this binding exposes neither Player nor Synth.midi_file_play.",
                midi_path,
            )
            return
        except FluidSynthPlaybackSchedulerError as exc:
            raise SongAudioBackendError(
                "pyfluidsynth backend initialized successfully, but this binding exposes neither "
                "Player nor Synth.midi_file_play, and fallback scheduled playback failed: "
                f"{exc}"
            ) from exc

    def stop_song(self) -> None:
        if self._active_player is not None:
            stop = getattr(self._active_player, "stop", None)
            if callable(stop):
                stop()
            self._active_player = None

        all_sounds_off = getattr(self.synth, "all_sounds_off", None)
        if callable(all_sounds_off):
            for channel in range(16):
                all_sounds_off(channel)

    def shutdown(self) -> None:
        self.stop_song()
        delete = getattr(self.synth, "delete", None)
        if callable(delete):
            try:
                delete()
            except Exception as exc:
                raise SongAudioBackendError(f"Failed to shut down pyfluidsynth backend cleanly: {exc}") from exc

    def _apply_song_channel_volume(self) -> None:
        cc = getattr(self.synth, "cc", None)
        if not callable(cc):
            return
        song_cc_volume = max(0, min(127, int(round(float(self.song_volume) * 127.0))))
        for channel in range(16):
            if channel == int(self.cue_channel):
                continue
            try:
                cc(channel, 7, song_cc_volume)
            except Exception:
                LOGGER.debug(
                    "Failed to apply FluidSynth song CC volume for channel=%s volume=%s.",
                    channel,
                    song_cc_volume,
                    exc_info=True,
                )


def create_song_audio_backend(*, audio_config: dict[str, Any], pygame_module: Any) -> SongAudioBackend:
    """Create the configured target-song audio backend from YAML audio.backend."""

    backend = str(audio_config.get("backend", "pygame")).strip().lower()
    if backend == "pygame":
        return _build_pygame_song_backend(audio_config=audio_config, pygame_module=pygame_module)
    if backend in {"pyfluidsynth", "fluidsynth"}:
        return FluidSynthSongAudioBackend.initialize_from_config(audio_config=audio_config)
    raise SongAudioBackendError(
        f"Unsupported target-song audio backend '{backend}'. Configure audio.backend to a supported backend."
    )


def _build_pygame_song_backend(*, audio_config: dict[str, Any], pygame_module: Any) -> PygameSongAudioBackend:
    """Build pygame song backend and enforce immediate-cue routing requirements."""

    levels = resolve_unified_audio_levels(audio_config)
    playback_gain = float(levels.song)
    return PygameSongAudioBackend(pygame_module=pygame_module, playback_gain=playback_gain)


def _resolve_fluidsynth_cue_channel(audio_config: dict[str, Any]) -> int:
    immediate_cfg = audio_config.get("immediate_cues") if isinstance(audio_config.get("immediate_cues"), dict) else {}
    fluidsynth_cfg = (
        immediate_cfg.get("fluidsynth")
        if isinstance(immediate_cfg.get("fluidsynth"), dict)
        else {}
    )
    return max(0, min(15, int(fluidsynth_cfg.get("channel", 15))))


def resolve_shared_fluidsynth_cue_instrument(*, audio_config: dict[str, Any]) -> tuple[int, int]:
    """Resolve immediate-cue bank/program from the same instrument profile used for song playback."""

    channel_presets = resolve_fluidsynth_channel_presets(audio_config=audio_config)
    cue_channel = _resolve_fluidsynth_cue_channel(audio_config)
    if cue_channel in channel_presets:
        return channel_presets[cue_channel]
    if 0 in channel_presets:
        return channel_presets[0]
    return (0, 0)


def resolve_fluidsynth_channel_presets(*, audio_config: dict[str, Any]) -> dict[int, tuple[int, int]]:
    """Resolve configured FluidSynth channel presets as {channel: (bank, program)}."""

    fluidsynth_cfg = audio_config.get("fluidsynth") if isinstance(audio_config.get("fluidsynth"), dict) else {}
    selected_profile = fluidsynth_cfg.get("instrument_profile")
    raw_profiles = fluidsynth_cfg.get("instrument_profiles")
    profile_map = raw_profiles if isinstance(raw_profiles, dict) else {}
    profile_name = str(selected_profile).strip() if selected_profile is not None else ""
    raw_presets = fluidsynth_cfg.get("channel_instruments") if isinstance(fluidsynth_cfg.get("channel_instruments"), dict) else {}

    if profile_name:
        if profile_name not in profile_map or not isinstance(profile_map[profile_name], dict):
            raise SongAudioBackendError(
                "audio.fluidsynth.instrument_profile is set to "
                f"'{profile_name}', but no matching mapping exists in audio.fluidsynth.instrument_profiles."
            )
        profile_payload = profile_map[profile_name]
        if isinstance(profile_payload.get("channels"), dict):
            raw_presets = {**profile_payload["channels"], **raw_presets}
        else:
            raw_presets = {**profile_payload, **raw_presets}

    presets: dict[int, tuple[int, int]] = {}
    for raw_channel, raw_spec in raw_presets.items():
        try:
            channel = int(raw_channel)
        except (TypeError, ValueError) as exc:
            raise SongAudioBackendError(
                f"Invalid FluidSynth channel '{raw_channel}' in instrument mapping. Use integer channels 0..15."
            ) from exc
        if channel < 0 or channel > 15:
            raise SongAudioBackendError(
                f"Invalid FluidSynth channel '{channel}' in instrument mapping. Channel must be in 0..15."
            )
        if not isinstance(raw_spec, dict):
            raise SongAudioBackendError(
                f"Invalid FluidSynth channel mapping for channel {channel}. Expected an object with bank/program."
            )
        bank = int(raw_spec.get("bank", 0))
        program = int(raw_spec.get("program", -1))
        if bank < 0 or bank > 16383:
            raise SongAudioBackendError(
                f"Invalid FluidSynth bank '{bank}' for channel {channel}. Bank must be in 0..16383."
            )
        if program < 0 or program > 127:
            raise SongAudioBackendError(
                f"Invalid FluidSynth program '{program}' for channel {channel}. Program must be in 0..127."
            )
        presets[channel] = (bank, program)
    return presets


def resolve_fluidsynth_effects_settings(*, audio_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve effective FluidSynth reverb/chorus settings from base + profile overrides."""

    fluidsynth_cfg = audio_config.get("fluidsynth") if isinstance(audio_config.get("fluidsynth"), dict) else {}
    selected_profile = str(fluidsynth_cfg.get("instrument_profile") or "").strip()
    profiles = fluidsynth_cfg.get("instrument_profiles") if isinstance(fluidsynth_cfg.get("instrument_profiles"), dict) else {}

    base_reverb = fluidsynth_cfg.get("reverb") if isinstance(fluidsynth_cfg.get("reverb"), dict) else {}
    base_chorus = fluidsynth_cfg.get("chorus") if isinstance(fluidsynth_cfg.get("chorus"), dict) else {}

    reverb: dict[str, Any] = {
        "enabled": bool(base_reverb.get("enabled", True)),
        "room_size": float(base_reverb.get("room_size", 0.2)),
        "damping": float(base_reverb.get("damping", 0.0)),
        "width": float(base_reverb.get("width", 0.5)),
        "level": float(base_reverb.get("level", 0.9)),
    }
    chorus: dict[str, Any] = {
        "enabled": bool(base_chorus.get("enabled", False)),
        "voices": int(base_chorus.get("voices", 3)),
        "level": float(base_chorus.get("level", 1.5)),
        "speed": float(base_chorus.get("speed", 0.3)),
        "depth_ms": float(base_chorus.get("depth_ms", 8.0)),
        "waveform": str(base_chorus.get("waveform", "sine")).strip().lower() or "sine",
    }

    if selected_profile:
        profile_payload = profiles.get(selected_profile)
        if isinstance(profile_payload, dict):
            effects = profile_payload.get("effects") if isinstance(profile_payload.get("effects"), dict) else {}
            profile_reverb = effects.get("reverb") if isinstance(effects.get("reverb"), dict) else {}
            profile_chorus = effects.get("chorus") if isinstance(effects.get("chorus"), dict) else {}
            reverb.update(profile_reverb)
            chorus.update(profile_chorus)

    if chorus["waveform"] not in {"sine", "triangle"}:
        raise SongAudioBackendError(
            "Invalid audio.fluidsynth.chorus.waveform setting. Use 'sine' or 'triangle'."
        )

    return {"reverb": reverb, "chorus": chorus}


def _apply_fluidsynth_channel_presets(*, synth: Any, soundfont_id: int, channel_presets: dict[int, tuple[int, int]]) -> None:
    if not channel_presets:
        return
    program_select = getattr(synth, "program_select", None)
    if not callable(program_select):
        raise SongAudioBackendError(
            "Configured FluidSynth instrument presets require Synth.program_select, but this pyfluidsynth binding "
            "does not expose it. Remove audio.fluidsynth channel/profile overrides or update pyfluidsynth."
        )
    for channel, (bank, program) in sorted(channel_presets.items()):
        result = int(program_select(channel, soundfont_id, bank, program))
        if result != 0:
            raise SongAudioBackendError(
                "Failed to apply FluidSynth instrument preset for "
                f"channel={channel} bank={bank} program={program} (result={result})."
            )


def _apply_fluidsynth_effects(*, synth: Any, effects: dict[str, dict[str, Any]]) -> None:
    reverb = effects.get("reverb", {})
    chorus = effects.get("chorus", {})

    reverb_on = getattr(synth, "reverb_on", None)
    if callable(reverb_on):
        reverb_on(1 if bool(reverb.get("enabled", True)) else 0)
    if bool(reverb.get("enabled", True)):
        set_reverb = getattr(synth, "set_reverb", None)
        if not callable(set_reverb):
            raise SongAudioBackendError(
                "Configured FluidSynth reverb settings require Synth.set_reverb, but this binding does not expose it."
            )
        set_reverb(
            float(reverb.get("room_size", 0.2)),
            float(reverb.get("damping", 0.0)),
            float(reverb.get("width", 0.5)),
            float(reverb.get("level", 0.9)),
        )

    chorus_on = getattr(synth, "chorus_on", None)
    if callable(chorus_on):
        chorus_on(1 if bool(chorus.get("enabled", False)) else 0)
    if bool(chorus.get("enabled", False)):
        set_chorus = getattr(synth, "set_chorus", None)
        if not callable(set_chorus):
            raise SongAudioBackendError(
                "Configured FluidSynth chorus settings require Synth.set_chorus, but this binding does not expose it."
            )
        waveform = str(chorus.get("waveform", "sine")).strip().lower()
        waveform_type = 0 if waveform == "sine" else 1
        set_chorus(
            int(chorus.get("voices", 3)),
            float(chorus.get("level", 1.5)),
            float(chorus.get("speed", 0.3)),
            float(chorus.get("depth_ms", 8.0)),
            waveform_type,
        )
