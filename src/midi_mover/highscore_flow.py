"""Highscore decision flow helpers for post-song routing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from midi_mover.highscore_storage import (
    HighscoreStorage,
    HighscoreStorageError,
    fetch_configured_top_entries,
    qualifies_for_leaderboard,
)
from midi_mover.song_summary import SongCompleteSummary


@dataclass(frozen=True)
class PostSongRouteDecision:
    """Resolved post-song transition route for the state machine."""

    route: str
    qualifies_for_highscore: bool
    final_run_summary: SongCompleteSummary


def decide_post_song_route(
    *,
    final_run_summary: SongCompleteSummary,
    config_payload: Mapping[str, Any],
    storage: HighscoreStorage,
) -> PostSongRouteDecision:
    """Choose whether to continue to headshot countdown or leaderboard directly."""

    highscore_payload = config_payload.get("highscore")
    if not isinstance(highscore_payload, Mapping):
        raise HighscoreStorageError("Config payload is missing required mapping key 'highscore'.")

    leaderboard_limit = highscore_payload.get("list_size")
    if not isinstance(leaderboard_limit, int):
        raise HighscoreStorageError(
            "Config key highscore.list_size must be an integer for post-song routing."
        )

    existing_entries = fetch_configured_top_entries(storage, config_payload)
    now_utc = datetime.now(timezone.utc).isoformat()
    qualifies = qualifies_for_leaderboard(
        candidate_score=int(final_run_summary.score),
        candidate_percentage_hit=float(final_run_summary.percentage_hit),
        candidate_recorded_at_utc=now_utc,
        existing_entries=existing_entries,
        leaderboard_limit=leaderboard_limit,
    )
    route = "headshot_countdown" if qualifies else "leaderboard"
    return PostSongRouteDecision(
        route=route,
        qualifies_for_highscore=qualifies,
        final_run_summary=final_run_summary,
    )
