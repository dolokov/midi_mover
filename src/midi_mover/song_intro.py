"""Song-title extraction and pre-song title-screen rendering."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from midi_mover.midi_files import extract_song_title_from_filename


def build_song_title(midi_path: Path) -> str:
    """Return a human-readable title derived from a MIDI filename."""
    return extract_song_title_from_filename(midi_path)


def show_pre_song_title_screen(
    *,
    pygame_module: Any,
    window: Any,
    song_title: str,
    duration_seconds: float = 2.0,
) -> bool:
    """Render a lightweight pre-song title card and return False if user quits."""
    duration_seconds = max(0.0, float(duration_seconds))
    title_font = pygame_module.font.SysFont(None, 64)
    subtitle_font = pygame_module.font.SysFont(None, 30)

    title_surface = title_font.render(song_title, True, (229, 231, 235))
    subtitle_surface = subtitle_font.render("Get ready...", True, (156, 163, 175))

    end_time = time.monotonic() + duration_seconds
    while time.monotonic() < end_time:
        for event in pygame_module.event.get():
            if event.type == pygame_module.QUIT:
                return False
            if event.type == pygame_module.KEYDOWN and event.key == pygame_module.K_ESCAPE:
                return False

        window.fill((8, 15, 31))
        title_rect = title_surface.get_rect(center=(window.get_width() // 2, window.get_height() // 2 - 20))
        subtitle_rect = subtitle_surface.get_rect(center=(window.get_width() // 2, window.get_height() // 2 + 30))
        window.blit(title_surface, title_rect)
        window.blit(subtitle_surface, subtitle_rect)
        pygame_module.display.flip()
        pygame_module.time.delay(16)

    return True
