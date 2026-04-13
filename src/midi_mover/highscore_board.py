"""Leaderboard rendering helpers for post-song highscore display."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Sequence

from midi_mover.highscore_storage import HighscoreEntry


LOGGER = logging.getLogger("midi_mover")


def sort_highscore_entries(entries: Sequence[HighscoreEntry]) -> tuple[HighscoreEntry, ...]:
    """Return entries sorted by the leaderboard rank rule."""

    return tuple(
        sorted(
            entries,
            key=lambda entry: (
                -int(entry.score),
                -float(entry.percentage_hit),
                str(entry.recorded_at_utc),
                int(entry.row_id),
            ),
        )
    )


def show_highscore_board_screen(
    *,
    pygame_module: Any,
    window: Any,
    entries: Sequence[HighscoreEntry],
    thumbnail_width: int,
    thumbnail_height: int,
    duration_seconds: float,
    highlight_row_id: int | None = None,
    current_run_summary: Any | None = None,
) -> bool:
    """Render the leaderboard board screen for a short post-song duration."""

    safe_duration = max(0.0, float(duration_seconds))
    end_time = time.monotonic() + safe_duration
    while time.monotonic() < end_time:
        for event in pygame_module.event.get():
            if event.type == pygame_module.QUIT:
                return False
            if event.type == pygame_module.KEYDOWN and event.key == pygame_module.K_ESCAPE:
                return False

        draw_highscore_board_once(
            pygame_module=pygame_module,
            window=window,
            entries=entries,
            thumbnail_width=thumbnail_width,
            thumbnail_height=thumbnail_height,
            highlight_row_id=highlight_row_id,
            current_run_summary=current_run_summary,
        )
        pygame_module.display.flip()
        pygame_module.time.delay(16)
    return True


def draw_highscore_board_once(
    *,
    pygame_module: Any,
    window: Any,
    entries: Sequence[HighscoreEntry],
    thumbnail_width: int,
    thumbnail_height: int,
    highlight_row_id: int | None = None,
    current_run_summary: Any | None = None,
) -> None:
    """Draw one leaderboard frame containing sorted rows and image thumbnails."""

    bg_color = (6, 10, 24)
    panel_color = (15, 23, 42)
    border_color = (51, 65, 85)
    header_color = (191, 219, 254)
    row_color_a = (17, 24, 39)
    row_color_b = (12, 19, 33)
    text_color = (226, 232, 240)
    muted_text_color = (148, 163, 184)
    highlight_row_bg = (30, 64, 52)
    highlight_border_color = (52, 211, 153)
    highlight_text_color = (209, 250, 229)

    window.fill(bg_color)
    panel_width = min(window.get_width() - 60, 1120)
    panel_height = min(window.get_height() - 60, 620)
    panel = pygame_module.Rect(
        (window.get_width() - panel_width) // 2,
        (window.get_height() - panel_height) // 2,
        panel_width,
        panel_height,
    )
    pygame_module.draw.rect(window, panel_color, panel, border_radius=14)
    pygame_module.draw.rect(window, border_color, panel, width=2, border_radius=14)

    title_font = pygame_module.font.SysFont(None, 52, bold=True)
    header_font = pygame_module.font.SysFont(None, 28, bold=True)
    row_font = pygame_module.font.SysFont(None, 30)
    meta_font = pygame_module.font.SysFont(None, 24)

    title_surface = title_font.render("Leaderboard", True, (241, 245, 249))
    title_rect = title_surface.get_rect(x=panel.x + 24, y=panel.y + 16)
    window.blit(title_surface, title_rect)

    table_top = title_rect.bottom + 16
    rank_col_x = panel.x + 24
    thumb_col_x = rank_col_x + 70
    score_col_x = thumb_col_x + max(52, int(thumbnail_width)) + 20
    hit_col_x = score_col_x + 180
    song_col_x = hit_col_x + 170

    header_y = table_top
    for text, x in (
        ("#", rank_col_x),
        ("Photo", thumb_col_x),
        ("Score", score_col_x),
        ("Hit %", hit_col_x),
        ("Song", song_col_x),
    ):
        surface = header_font.render(text, True, header_color)
        window.blit(surface, surface.get_rect(x=x, y=header_y))

    show_current_run_strip = current_run_summary is not None and highlight_row_id is None
    current_run_strip_height = 84 if show_current_run_strip else 0

    rows_top = header_y + 34
    row_height = max(56, int(thumbnail_height) + 8)
    max_rows = max(0, (panel.bottom - rows_top - 14 - current_run_strip_height) // row_height)
    visible_entries = sort_highscore_entries(entries)[:max_rows]

    for index, entry in enumerate(visible_entries):
        row_y = rows_top + index * row_height
        row_rect = pygame_module.Rect(panel.x + 12, row_y, panel.width - 24, row_height - 4)
        is_highlighted_entry = highlight_row_id is not None and int(entry.row_id) == int(highlight_row_id)
        row_bg = highlight_row_bg if is_highlighted_entry else (row_color_a if index % 2 == 0 else row_color_b)
        pygame_module.draw.rect(window, row_bg, row_rect, border_radius=8)
        if is_highlighted_entry:
            pygame_module.draw.rect(window, highlight_border_color, row_rect, width=2, border_radius=8)

        active_text_color = highlight_text_color if is_highlighted_entry else text_color

        rank_surface = row_font.render(str(index + 1), True, active_text_color)
        window.blit(rank_surface, rank_surface.get_rect(x=rank_col_x, centery=row_rect.centery))

        thumbnail_surface = _load_thumbnail_surface(
            pygame_module=pygame_module,
            headshot_path=entry.headshot_path,
            target_width=max(32, int(thumbnail_width)),
            target_height=max(32, int(thumbnail_height)),
        )
        thumb_rect = thumbnail_surface.get_rect(x=thumb_col_x, centery=row_rect.centery)
        window.blit(thumbnail_surface, thumb_rect)

        score_surface = row_font.render(str(entry.score), True, active_text_color)
        window.blit(score_surface, score_surface.get_rect(x=score_col_x, centery=row_rect.centery - 10))

        hit_surface = row_font.render(f"{entry.percentage_hit:.1f}%", True, active_text_color)
        window.blit(hit_surface, hit_surface.get_rect(x=hit_col_x, centery=row_rect.centery - 10))

        song_surface = row_font.render(str(entry.song_title), True, active_text_color)
        window.blit(song_surface, song_surface.get_rect(x=song_col_x, centery=row_rect.centery - 10))

        if is_highlighted_entry:
            badge_surface = meta_font.render("NEW", True, highlight_border_color)
            window.blit(
                badge_surface,
                badge_surface.get_rect(
                    right=row_rect.right - 14,
                    centery=row_rect.centery - 10,
                ),
            )

        timestamp_surface = meta_font.render(str(entry.recorded_at_utc), True, muted_text_color)
        window.blit(timestamp_surface, timestamp_surface.get_rect(x=score_col_x, centery=row_rect.centery + 16))

    if not visible_entries:
        empty_surface = row_font.render("No highscore entries yet.", True, muted_text_color)
        window.blit(empty_surface, empty_surface.get_rect(center=panel.center))

    if show_current_run_strip:
        run_summary = current_run_summary
        summary_rect = pygame_module.Rect(
            panel.x + 16,
            panel.bottom - current_run_strip_height + 10,
            panel.width - 32,
            current_run_strip_height - 16,
        )
        pygame_module.draw.rect(window, (30, 41, 59), summary_rect, border_radius=10)
        pygame_module.draw.rect(window, (100, 116, 139), summary_rect, width=1, border_radius=10)

        summary_title = meta_font.render("Current run (not in leaderboard)", True, (191, 219, 254))
        window.blit(summary_title, summary_title.get_rect(x=summary_rect.x + 12, y=summary_rect.y + 8))

        score = int(getattr(run_summary, "score", 0))
        percentage_hit = float(getattr(run_summary, "percentage_hit", 0.0))
        notes_hit = int(getattr(run_summary, "notes_hit", 0))
        total_notes = int(getattr(run_summary, "total_notes", 0))
        notes_missed = int(getattr(run_summary, "notes_missed", 0))
        summary_line = (
            f"Score: {score}   Hit: {percentage_hit:.1f}%   "
            f"Notes: {notes_hit}/{total_notes}   Misses: {notes_missed}"
        )
        summary_surface = row_font.render(summary_line, True, (226, 232, 240))
        window.blit(
            summary_surface,
            summary_surface.get_rect(x=summary_rect.x + 12, y=summary_rect.y + 34),
        )


def _load_thumbnail_surface(
    *,
    pygame_module: Any,
    headshot_path: str | None,
    target_width: int,
    target_height: int,
) -> Any:
    if headshot_path:
        absolute_path = Path(headshot_path)
        if not absolute_path.is_absolute():
            absolute_path = Path(__file__).resolve().parents[2] / absolute_path
        if absolute_path.exists():
            try:
                image_surface = pygame_module.image.load(str(absolute_path))
                return pygame_module.transform.scale(
                    image_surface,
                    (int(target_width), int(target_height)),
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                LOGGER.warning("Failed to load headshot thumbnail '%s': %s", absolute_path, exc)

    placeholder = pygame_module.Surface((int(target_width), int(target_height)))
    placeholder.fill((30, 41, 59))
    pygame_module.draw.rect(
        placeholder,
        (71, 85, 105),
        pygame_module.Rect(0, 0, int(target_width), int(target_height)),
        width=2,
    )
    return placeholder
