"""Right-panel timeline layout primitives for five-lane gameplay rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TimelineFallingNote:
    """Renderable timeline note projected into panel-space coordinates."""

    target_note_id: str
    lane: int
    hand: str
    rect: Any


@dataclass(frozen=True)
class TimelineLaneLayout:
    """Geometry for one lane inside the right-side timeline panel."""

    lane: int
    rect: Any


@dataclass(frozen=True)
class TimelineHistoryNote:
    """Renderable judged timeline note that remains visible below the now line."""

    target_note_id: str
    lane: int
    hand: str
    rect: Any
    outcome: str


@dataclass(frozen=True)
class TimelinePanelLayout:
    """Computed right-panel geometry with five equal-width lanes."""

    panel_rect: Any
    lanes: tuple[TimelineLaneLayout, ...]
    now_line_y: int


@dataclass(frozen=True)
class TimelineScoreSnapshot:
    """Read-only gameplay metric snapshot shown on the timeline panel."""

    score: int
    combo: int
    max_combo: int
    total_hits: int
    total_misses: int
    total_notes: int
    percentage_hit: float


def compute_timeline_panel_layout(
    *,
    pygame_module: Any,
    window_width: int,
    window_height: int,
    lane_count: int = 5,
    now_line_ratio: float = 0.25,
) -> TimelinePanelLayout:
    """Compute fixed 40%-width right panel and split it into five lanes."""

    safe_width = max(1, int(window_width))
    safe_height = max(1, int(window_height))
    right_panel_width = max(1, int(round(safe_width * 0.4)))
    right_panel_x = safe_width - right_panel_width
    panel_rect = pygame_module.Rect(right_panel_x, 0, right_panel_width, safe_height)

    safe_lane_count = max(1, int(lane_count))
    base_lane_width, remainder = divmod(panel_rect.width, safe_lane_count)
    lane_width = max(1, base_lane_width)
    lane_rects: list[TimelineLaneLayout] = []
    cursor_x = panel_rect.x
    for lane_index in range(safe_lane_count):
        extra_width = 1 if lane_index < remainder else 0
        current_lane_width = lane_width + extra_width
        if lane_index == (safe_lane_count - 1):
            current_lane_width = panel_rect.right - cursor_x
        lane_rects.append(
            TimelineLaneLayout(
                lane=lane_index + 1,
                rect=pygame_module.Rect(cursor_x, panel_rect.y, current_lane_width, panel_rect.height),
            )
        )
        cursor_x += current_lane_width

    safe_now_line_ratio = max(0.0, min(1.0, float(now_line_ratio)))
    now_line_y = panel_rect.y + int(round(panel_rect.height * safe_now_line_ratio))
    return TimelinePanelLayout(
        panel_rect=panel_rect,
        lanes=tuple(lane_rects),
        now_line_y=now_line_y,
    )


def draw_timeline_panel_layout(
    *,
    surface: Any,
    pygame_module: Any,
    layout: TimelinePanelLayout,
    base_color: tuple[int, int, int],
) -> None:
    """Draw panel background, lane headers/separators, and the fixed now line."""

    surface.fill(base_color, layout.panel_rect)
    lane_even = _mix_color(base_color, (255, 255, 255), ratio=0.12)
    lane_odd = _mix_color(base_color, (255, 255, 255), ratio=0.05)
    separator_color = _mix_color(base_color, (255, 255, 255), ratio=0.30)
    header_text_color = _mix_color(base_color, (255, 255, 255), ratio=0.90)
    now_line_color = (255, 235, 120)
    now_line_shadow = _mix_color(base_color, now_line_color, ratio=0.35)

    for lane in layout.lanes:
        lane_color = lane_even if lane.lane % 2 == 0 else lane_odd
        surface.fill(lane_color, lane.rect)

    for lane in layout.lanes[:-1]:
        pygame_module.draw.line(
            surface,
            separator_color,
            (lane.rect.right, layout.panel_rect.top),
            (lane.rect.right, layout.panel_rect.bottom),
            2,
        )

    header_font_size = max(16, min(36, layout.panel_rect.width // 14))
    header_font = pygame_module.font.SysFont(None, header_font_size, bold=True)
    header_margin_top = max(8, layout.panel_rect.height // 120)
    for lane in layout.lanes:
        header_surface = header_font.render(str(lane.lane), True, header_text_color)
        header_rect = header_surface.get_rect()
        header_rect.centerx = lane.rect.centerx
        header_rect.top = layout.panel_rect.top + header_margin_top
        surface.blit(header_surface, header_rect)

    pygame_module.draw.line(
        surface,
        now_line_shadow,
        (layout.panel_rect.left, layout.now_line_y + 1),
        (layout.panel_rect.right, layout.now_line_y + 1),
        5,
    )
    pygame_module.draw.line(
        surface,
        now_line_color,
        (layout.panel_rect.left, layout.now_line_y),
        (layout.panel_rect.right, layout.now_line_y),
        3,
    )


def project_upcoming_timeline_notes(
    *,
    pygame_module: Any,
    layout: TimelinePanelLayout,
    normalized_target_notes: tuple[Any, ...],
    song_elapsed_seconds: float,
    lookahead_ms: float,
) -> tuple[TimelineFallingNote, ...]:
    """Project upcoming normalized target notes into timeline-lane rectangles."""

    safe_lookahead_ms = max(1.0, float(lookahead_ms))
    elapsed_ms = float(song_elapsed_seconds) * 1000.0
    projected: list[TimelineFallingNote] = []
    note_height = max(12, int(layout.panel_rect.height * 0.03))
    note_margin_x = max(3, int(layout.panel_rect.width * 0.004))

    for note in normalized_target_notes:
        lane = int(getattr(note, "lane", 0))
        if lane < 1 or lane > len(layout.lanes):
            continue

        delta_ms = float(getattr(note, "timestamp_ms", 0.0)) - elapsed_ms
        if delta_ms < 0.0 or delta_ms > safe_lookahead_ms:
            continue

        lane_layout = layout.lanes[lane - 1]
        progress_to_now = 1.0 - (delta_ms / safe_lookahead_ms)
        center_y = layout.panel_rect.top + int(round(progress_to_now * (layout.now_line_y - layout.panel_rect.top)))
        top = max(layout.panel_rect.top, center_y - (note_height // 2))
        if top + note_height > layout.now_line_y:
            top = layout.now_line_y - note_height
        note_rect = pygame_module.Rect(
            lane_layout.rect.x + note_margin_x,
            top,
            max(1, lane_layout.rect.width - (note_margin_x * 2)),
            note_height,
        )
        projected.append(
            TimelineFallingNote(
                target_note_id=str(getattr(note, "target_note_id", "")),
                lane=lane,
                hand=str(getattr(note, "hand", "?")).strip().upper()[:1] or "?",
                rect=note_rect,
            )
        )

    return tuple(projected)


def project_timeline_note_history(
    *,
    pygame_module: Any,
    layout: TimelinePanelLayout,
    normalized_target_notes: tuple[Any, ...],
    song_elapsed_seconds: float,
    note_history_ms: float,
    judged_note_outcomes: dict[str, str] | None = None,
) -> tuple[TimelineHistoryNote, ...]:
    """Project post-now judged notes into the visible history area below the now line."""

    safe_note_history_ms = max(1.0, float(note_history_ms))
    elapsed_ms = float(song_elapsed_seconds) * 1000.0
    note_height = max(12, int(layout.panel_rect.height * 0.03))
    note_margin_x = max(3, int(layout.panel_rect.width * 0.004))
    history_bottom = max(layout.now_line_y + note_height, layout.panel_rect.bottom - note_height)
    outcomes = judged_note_outcomes or {}
    projected: list[TimelineHistoryNote] = []

    for note in normalized_target_notes:
        lane = int(getattr(note, "lane", 0))
        if lane < 1 or lane > len(layout.lanes):
            continue

        note_id = str(getattr(note, "target_note_id", ""))
        age_ms = elapsed_ms - float(getattr(note, "timestamp_ms", 0.0))
        if age_ms < 0.0 or age_ms > safe_note_history_ms:
            continue

        progress_below_now = min(1.0, age_ms / safe_note_history_ms)
        y_span = max(1, history_bottom - layout.now_line_y)
        center_y = layout.now_line_y + int(round(progress_below_now * y_span))
        top = min(history_bottom, max(layout.now_line_y, center_y - (note_height // 2)))
        lane_layout = layout.lanes[lane - 1]
        note_rect = pygame_module.Rect(
            lane_layout.rect.x + note_margin_x,
            top,
            max(1, lane_layout.rect.width - (note_margin_x * 2)),
            note_height,
        )
        outcome = str(outcomes.get(note_id, "miss")).strip().lower()
        if outcome not in {"hit", "miss"}:
            outcome = "miss"
        projected.append(
            TimelineHistoryNote(
                target_note_id=note_id,
                lane=lane,
                hand=str(getattr(note, "hand", "?")).strip().upper()[:1] or "?",
                rect=note_rect,
                outcome=outcome,
            )
        )

    return tuple(projected)


def draw_upcoming_timeline_notes(
    *,
    surface: Any,
    pygame_module: Any,
    layout: TimelinePanelLayout,
    normalized_target_notes: tuple[Any, ...],
    song_elapsed_seconds: float,
    lookahead_ms: float,
    note_history_ms: float,
    judged_note_outcomes: dict[str, str] | None = None,
    gameplay_score_state: TimelineScoreSnapshot | Any | None = None,
) -> None:
    """Draw upcoming notes plus judged note history below the now line."""

    falling_notes = project_upcoming_timeline_notes(
        pygame_module=pygame_module,
        layout=layout,
        normalized_target_notes=normalized_target_notes,
        song_elapsed_seconds=song_elapsed_seconds,
        lookahead_ms=lookahead_ms,
    )
    history_notes = project_timeline_note_history(
        pygame_module=pygame_module,
        layout=layout,
        normalized_target_notes=normalized_target_notes,
        song_elapsed_seconds=song_elapsed_seconds,
        note_history_ms=note_history_ms,
        judged_note_outcomes=judged_note_outcomes,
    )
    fill_color = (96, 165, 250)
    outline_color = (191, 219, 254)
    hit_fill_color = (34, 197, 94)
    hit_outline_color = (134, 239, 172)
    miss_fill_color = (239, 68, 68)
    miss_outline_color = (254, 202, 202)
    hand_text_color = (240, 249, 255)
    hand_font_size = max(12, min(28, int(layout.panel_rect.height * 0.028)))
    hand_font = pygame_module.font.SysFont(None, hand_font_size, bold=True)
    for history_note in history_notes:
        history_fill = hit_fill_color if history_note.outcome == "hit" else miss_fill_color
        history_outline = hit_outline_color if history_note.outcome == "hit" else miss_outline_color
        pygame_module.draw.rect(surface, history_fill, history_note.rect, border_radius=5)
        pygame_module.draw.rect(surface, history_outline, history_note.rect, width=2, border_radius=5)
        hand_surface = hand_font.render(history_note.hand, True, hand_text_color)
        hand_rect = hand_surface.get_rect(center=history_note.rect.center)
        surface.blit(hand_surface, hand_rect)

    for falling_note in falling_notes:
        pygame_module.draw.rect(surface, fill_color, falling_note.rect, border_radius=5)
        pygame_module.draw.rect(surface, outline_color, falling_note.rect, width=2, border_radius=5)
        hand_surface = hand_font.render(falling_note.hand, True, hand_text_color)
        hand_rect = hand_surface.get_rect(center=falling_note.rect.center)
        surface.blit(hand_surface, hand_rect)

    if gameplay_score_state is not None:
        _draw_timeline_score_overlay(
            surface=surface,
            pygame_module=pygame_module,
            layout=layout,
            gameplay_score_state=gameplay_score_state,
        )


def _draw_timeline_score_overlay(
    *,
    surface: Any,
    pygame_module: Any,
    layout: TimelinePanelLayout,
    gameplay_score_state: TimelineScoreSnapshot | Any,
) -> None:
    panel_width = layout.panel_rect.width
    font_size = max(14, min(26, int(panel_width * 0.055)))
    font = pygame_module.font.SysFont(None, font_size, bold=True)
    text_color = (226, 232, 240)
    muted_color = (148, 163, 184)
    bg_color = (2, 6, 23)
    border_color = (51, 65, 85)
    margin = max(8, int(panel_width * 0.03))
    line_gap = max(4, int(font_size * 0.25))
    score = int(getattr(gameplay_score_state, "score", 0))
    combo = int(getattr(gameplay_score_state, "combo", 0))
    max_combo = int(getattr(gameplay_score_state, "max_combo", 0))
    total_hits = int(getattr(gameplay_score_state, "total_hits", 0))
    total_misses = int(getattr(gameplay_score_state, "total_misses", 0))
    total_notes = int(getattr(gameplay_score_state, "total_notes", 0))
    percentage_hit = float(getattr(gameplay_score_state, "percentage_hit", 0.0))
    lines = (
        (f"Score {score}", text_color),
        (f"Combo {combo}  Max {max_combo}", muted_color),
        (f"Hits {total_hits}  Misses {total_misses}  Notes {total_notes}", muted_color),
        (f"Hit {percentage_hit:.1f}%", text_color),
    )
    rendered = [font.render(text, True, color) for text, color in lines]
    content_width = max((line.get_width() for line in rendered), default=0)
    content_height = sum(line.get_height() for line in rendered) + line_gap * max(0, len(rendered) - 1)
    panel_rect = pygame_module.Rect(
        layout.panel_rect.x + margin,
        layout.panel_rect.y + margin,
        min(layout.panel_rect.width - (margin * 2), content_width + (margin * 2)),
        content_height + (margin * 2),
    )
    pygame_module.draw.rect(surface, bg_color, panel_rect, border_radius=8)
    pygame_module.draw.rect(surface, border_color, panel_rect, width=1, border_radius=8)
    cursor_y = panel_rect.y + margin
    for line_surface in rendered:
        surface.blit(line_surface, (panel_rect.x + margin, cursor_y))
        cursor_y += line_surface.get_height() + line_gap


def _mix_color(
    color_a: tuple[int, int, int],
    color_b: tuple[int, int, int],
    *,
    ratio: float,
) -> tuple[int, int, int]:
    clamped = max(0.0, min(1.0, float(ratio)))
    return (
        int(round(color_a[0] * (1.0 - clamped) + color_b[0] * clamped)),
        int(round(color_a[1] * (1.0 - clamped) + color_b[1] * clamped)),
        int(round(color_a[2] * (1.0 - clamped) + color_b[2] * clamped)),
    )
