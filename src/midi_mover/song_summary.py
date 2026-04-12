"""Song-complete summary model and rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any


@dataclass(frozen=True)
class SongCompleteSummary:
    """Final run metrics prepared for highscore handoff."""

    song_title: str
    score: int
    notes_hit: int
    notes_missed: int
    total_notes: int
    percentage_hit: float
    max_combo: int


def show_song_complete_summary_screen(
    *,
    pygame_module: Any,
    window: Any,
    summary: SongCompleteSummary,
    duration_seconds: float,
) -> bool:
    """Render a post-song summary card and return False if user exits."""

    safe_duration = max(0.0, float(duration_seconds))
    title_font = pygame_module.font.SysFont(None, 54, bold=True)
    line_font = pygame_module.font.SysFont(None, 34)
    hint_font = pygame_module.font.SysFont(None, 26)

    title_surface = title_font.render("Song Complete", True, (241, 245, 249))
    subtitle_surface = hint_font.render(summary.song_title, True, (186, 230, 253))
    metric_lines = (
        f"Score: {summary.score}",
        f"Notes hit: {summary.notes_hit} / {summary.total_notes}",
        f"Misses: {summary.notes_missed}",
        f"Hit rate: {summary.percentage_hit:.1f}%",
        f"Max combo: {summary.max_combo}",
    )
    metric_surfaces = [line_font.render(line, True, (226, 232, 240)) for line in metric_lines]
    footer_surface = hint_font.render("Preparing highscore handoff...", True, (148, 163, 184))

    end_time = time.monotonic() + safe_duration
    while time.monotonic() < end_time:
        for event in pygame_module.event.get():
            if event.type == pygame_module.QUIT:
                return False
            if event.type == pygame_module.KEYDOWN and event.key == pygame_module.K_ESCAPE:
                return False

        window.fill((6, 10, 24))
        panel_width = min(window.get_width() - 80, 760)
        panel_height = min(window.get_height() - 80, 460)
        panel = pygame_module.Rect(
            (window.get_width() - panel_width) // 2,
            (window.get_height() - panel_height) // 2,
            panel_width,
            panel_height,
        )
        pygame_module.draw.rect(window, (15, 23, 42), panel, border_radius=14)
        pygame_module.draw.rect(window, (51, 65, 85), panel, width=2, border_radius=14)

        cursor_y = panel.y + 24
        title_rect = title_surface.get_rect(centerx=panel.centerx, y=cursor_y)
        window.blit(title_surface, title_rect)
        cursor_y = title_rect.bottom + 8

        subtitle_rect = subtitle_surface.get_rect(centerx=panel.centerx, y=cursor_y)
        window.blit(subtitle_surface, subtitle_rect)
        cursor_y = subtitle_rect.bottom + 24

        for metric_surface in metric_surfaces:
            metric_rect = metric_surface.get_rect(x=panel.x + 32, y=cursor_y)
            window.blit(metric_surface, metric_rect)
            cursor_y = metric_rect.bottom + 10

        footer_rect = footer_surface.get_rect(centerx=panel.centerx, bottom=panel.bottom - 20)
        window.blit(footer_surface, footer_rect)
        pygame_module.display.flip()
        pygame_module.time.delay(16)

    return True