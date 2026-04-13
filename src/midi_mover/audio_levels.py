"""Shared audio volume resolution for pygame and FluidSynth backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UnifiedAudioLevels:
    """Normalized per-domain audio levels with global master applied."""

    master: float
    song: float
    cue: float
    ui: float


def resolve_unified_audio_levels(audio_config: dict[str, Any]) -> UnifiedAudioLevels:
    """Resolve unified volume controls across all backends.

    Canonical keys:
    - audio.volumes.master
    - audio.volumes.song
    - audio.volumes.hit (immediate circle cues)
    - audio.volumes.ui

    Backward compatibility:
    - none (song and cue/hit domains are explicit).
    """

    volumes = audio_config.get("volumes") if isinstance(audio_config.get("volumes"), dict) else {}
    master = _clamp01(volumes.get("master", 1.0))
    song_raw = volumes.get("song", 1.0)
    cue_raw = volumes.get("hit", 1.0)
    ui_raw = volumes.get("ui", 1.0)
    return UnifiedAudioLevels(
        master=master,
        song=_clamp01(song_raw) * master,
        cue=_clamp01(cue_raw) * master,
        ui=_clamp01(ui_raw) * master,
    )


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 1.0
