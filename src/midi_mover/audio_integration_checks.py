"""Audio-specific integration checks used by startup smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

from midi_mover.song_audio import ensure_immediate_cue_routing, resolve_fluidsynth_effects_settings


def verify_fluidsynth_profile_effect_settings() -> None:
    """Verify profile-based FluidSynth reverb/chorus settings resolve distinctly."""

    audio_config = {
        "fluidsynth": {
            "instrument_profile": "concert_piano",
            "reverb": {"enabled": True, "room_size": 0.35, "damping": 0.2, "width": 0.6, "level": 0.75},
            "chorus": {"enabled": False, "voices": 3, "level": 1.2, "speed": 0.3, "depth_ms": 8.0, "waveform": "sine"},
            "instrument_profiles": {
                "concert_piano": {
                    "effects": {
                        "reverb": {"room_size": 0.32, "damping": 0.18, "width": 0.55, "level": 0.62},
                        "chorus": {"enabled": False},
                    }
                },
                "synth_lead": {
                    "effects": {
                        "reverb": {"room_size": 0.52, "damping": 0.34, "width": 0.82, "level": 0.88},
                        "chorus": {"enabled": True, "voices": 4, "level": 1.8, "speed": 0.55, "depth_ms": 11.0, "waveform": "triangle"},
                    }
                },
            },
        }
    }

    piano_settings = resolve_fluidsynth_effects_settings(audio_config=audio_config)
    audio_config["fluidsynth"]["instrument_profile"] = "synth_lead"
    lead_settings = resolve_fluidsynth_effects_settings(audio_config=audio_config)

    if piano_settings["chorus"]["enabled"]:
        raise RuntimeError("FluidSynth profile effects check failed: concert_piano chorus should be disabled.")
    if not lead_settings["chorus"]["enabled"]:
        raise RuntimeError("FluidSynth profile effects check failed: synth_lead chorus should be enabled.")
    if float(piano_settings["reverb"]["room_size"]) >= float(lead_settings["reverb"]["room_size"]):
        raise RuntimeError(
            "FluidSynth profile effects check failed: synth_lead should resolve stronger reverb room_size than concert_piano."
        )


@dataclass
class _FakeMixer:
    initialized: bool = True
    num_channels: int = 1

    def get_init(self) -> tuple[int, int, int] | None:
        if self.initialized:
            return (44100, -16, 2)
        return None

    def get_num_channels(self) -> int:
        return int(self.num_channels)

    def set_num_channels(self, count: int) -> None:
        self.num_channels = int(count)


@dataclass
class _FakePygame:
    mixer: _FakeMixer


def verify_immediate_cue_mixed_routing_behavior() -> None:
    """Verify immediate cues remain pygame-routed for low latency with FluidSynth song backend."""

    audio_config = {
        "playback": {"max_concurrent_sounds": 6},
        "immediate_cues": {"routing": "song_backend"},
    }
    pygame_module = _FakePygame(mixer=_FakeMixer(initialized=True, num_channels=2))
    resolved = ensure_immediate_cue_routing(
        audio_config=audio_config,
        pygame_module=pygame_module,
        song_backend_name="pyfluidsynth",
    )
    if resolved != "mixed_pygame":
        raise RuntimeError(
            "Immediate cue routing integration check failed: expected fallback to mixed_pygame for pyfluidsynth backend. "
            f"resolved={resolved}"
        )
    if pygame_module.mixer.num_channels < 6:
        raise RuntimeError(
            "Immediate cue routing integration check failed: expected pygame mixer channels to be provisioned "
            "for gesture cue concurrency."
        )
