"""Target-song audio backend abstraction and backend implementations."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Protocol


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
    backend_name: str = "pyfluidsynth"
    _active_player: Any = None

    @classmethod
    def initialize_from_config(cls, *, audio_config: dict[str, Any]) -> "FluidSynthSongAudioBackend":
        backend_config = (
            audio_config.get("fluidsynth")
            if isinstance(audio_config.get("fluidsynth"), dict)
            else {}
        )
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
        )

    def start_song(self, midi_path: Path) -> None:
        self.stop_song()
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

        raise SongAudioBackendError(
            "pyfluidsynth backend initialized successfully, but this binding exposes neither "
            "Player nor Synth.midi_file_play for MIDI playback. Use pygame backend or update pyfluidsynth."
        )

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


def create_song_audio_backend(*, audio_config: dict[str, Any], pygame_module: Any) -> SongAudioBackend:
    """Create the configured target-song audio backend from YAML audio.backend."""

    backend = str(audio_config.get("backend", "pygame")).strip().lower()
    if backend == "pygame":
        return _build_pygame_song_backend(audio_config=audio_config, pygame_module=pygame_module)
    if backend in {"pyfluidsynth", "fluidsynth"}:
        fallback_enabled = bool(audio_config.get("fallback_to_pygame_on_error", True))
        try:
            song_backend = FluidSynthSongAudioBackend.initialize_from_config(audio_config=audio_config)
        except SongAudioBackendError as exc:
            if not fallback_enabled:
                raise
            LOGGER.warning(
                "Requested target-song backend '%s' is unavailable (%s). Falling back to pygame backend.",
                backend,
                exc,
            )
            fallback_backend = _build_pygame_song_backend(
                audio_config=audio_config,
                pygame_module=pygame_module,
            )
            LOGGER.info(
                "Using fallback target-song backend '%s' after '%s' initialization failure.",
                fallback_backend.backend_name,
                backend,
            )
            return fallback_backend
        ensure_immediate_cue_routing(
            audio_config=audio_config,
            pygame_module=pygame_module,
            song_backend_name=song_backend.backend_name,
        )
        return song_backend
    raise SongAudioBackendError(
        f"Unsupported target-song audio backend '{backend}'. Configure audio.backend to a supported backend."
    )


def _build_pygame_song_backend(*, audio_config: dict[str, Any], pygame_module: Any) -> PygameSongAudioBackend:
    """Build pygame song backend and enforce immediate-cue routing requirements."""

    volumes = audio_config.get("volumes", {}) if isinstance(audio_config.get("volumes"), dict) else {}
    playback_gain = max(0.0, min(float(volumes.get("cue", 1.0)), 1.0))
    song_backend = PygameSongAudioBackend(pygame_module=pygame_module, playback_gain=playback_gain)
    ensure_immediate_cue_routing(
        audio_config=audio_config,
        pygame_module=pygame_module,
        song_backend_name=song_backend.backend_name,
    )
    return song_backend


def ensure_immediate_cue_routing(
    *,
    audio_config: dict[str, Any],
    pygame_module: Any,
    song_backend_name: str,
) -> str:
    """Keep low-latency immediate interaction cues routed through pygame mixer."""

    immediate_cfg = (
        audio_config.get("immediate_cues")
        if isinstance(audio_config.get("immediate_cues"), dict)
        else {}
    )
    requested_routing = str(immediate_cfg.get("routing", "mixed_pygame")).strip().lower()
    if requested_routing in {"mixed", "pygame"}:
        requested_routing = "mixed_pygame"

    if requested_routing not in {"mixed_pygame", "song_backend"}:
        raise SongAudioBackendError(
            "Invalid audio.immediate_cues.routing value. "
            "Use 'mixed_pygame' or 'song_backend'."
        )

    resolved_routing = requested_routing
    if requested_routing == "song_backend" and str(song_backend_name).strip().lower() not in {"pygame"}:
        resolved_routing = "mixed_pygame"
        LOGGER.warning(
            "audio.immediate_cues.routing='song_backend' cannot provide low-latency hit/miss/UI cues with "
            "song backend '%s'. Falling back to pygame mixed strategy for immediate cues.",
            song_backend_name,
        )

    mixer = getattr(pygame_module, "mixer", None)
    if mixer is None or not callable(getattr(mixer, "get_init", None)):
        raise SongAudioBackendError(
            "Immediate cue routing requires pygame.mixer to remain available for hit/miss/UI interactions."
        )
    if mixer.get_init() is None:
        raise SongAudioBackendError(
            "Immediate cue routing requires an initialized pygame mixer for low-latency interaction cues."
        )

    playback_cfg = audio_config.get("playback") if isinstance(audio_config.get("playback"), dict) else {}
    target_channels = max(1, int(playback_cfg.get("max_concurrent_sounds", 1)))
    get_num_channels = getattr(mixer, "get_num_channels", None)
    set_num_channels = getattr(mixer, "set_num_channels", None)
    if callable(get_num_channels) and callable(set_num_channels):
        current_channels = int(get_num_channels())
        if current_channels < target_channels:
            set_num_channels(target_channels)

    LOGGER.info(
        "Configured immediate interaction cue routing: requested=%s resolved=%s song_backend=%s.",
        requested_routing,
        resolved_routing,
        song_backend_name,
    )
    return resolved_routing


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
