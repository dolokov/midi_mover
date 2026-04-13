"""Persistent highscore storage bootstrap utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence


LOGGER = logging.getLogger("midi_mover")

_HIGH_SCORE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS highscores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    score INTEGER NOT NULL,
    percentage_hit REAL NOT NULL,
    song_title TEXT NOT NULL,
    recorded_at_utc TEXT NOT NULL,
    headshot_path TEXT
);
"""


class HighscoreStorageError(RuntimeError):
    """Raised when highscore storage initialization fails."""


@dataclass(frozen=True)
class HighscoreStorage:
    """Filesystem and database handles for persistent highscore storage."""

    database_path: Path
    headshots_dir: Path

    def ensure_storage_paths(self) -> None:
        """Ensure database parent/headshot directories exist before use."""
        database_parent = self.database_path.parent
        try:
            database_parent.mkdir(parents=True, exist_ok=True)
            self.headshots_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HighscoreStorageError(
                "Failed to ensure highscore storage directories exist."
            ) from exc

    def connect(self) -> sqlite3.Connection:
        """Open a SQLite connection to the persistent highscore database file."""
        self.ensure_storage_paths()
        try:
            return sqlite3.connect(str(self.database_path))
        except sqlite3.Error as exc:
            raise HighscoreStorageError(
                f"Failed to open highscore database at {self.database_path}: {exc}"
            ) from exc

    def fetch_top_entries(self, limit: int) -> tuple["HighscoreEntry", ...]:
        """Return leaderboard entries sorted by rank with an explicit entry limit."""
        if limit <= 0:
            raise HighscoreStorageError(
                f"Highscore retrieval limit must be > 0, got {limit}."
            )

        connection = self.connect()
        try:
            cursor = connection.execute(
                """
                SELECT id, score, percentage_hit, song_title, recorded_at_utc, headshot_path
                FROM highscores
                ORDER BY score DESC, percentage_hit DESC, recorded_at_utc ASC, id ASC
                LIMIT ?
                """,
                (int(limit),),
            )
            rows = cursor.fetchall()
        except sqlite3.Error as exc:
            raise HighscoreStorageError(
                f"Failed to fetch top highscore entries from {self.database_path}: {exc}"
            ) from exc
        finally:
            connection.close()

        return tuple(
            HighscoreEntry(
                row_id=int(row[0]),
                score=int(row[1]),
                percentage_hit=float(row[2]),
                song_title=str(row[3]),
                recorded_at_utc=str(row[4]),
                headshot_path=None if row[5] is None else str(row[5]),
            )
            for row in rows
        )

    def insert_entry(
        self,
        *,
        score: int,
        percentage_hit: float,
        song_title: str,
        headshot_path: str | None,
        recorded_at_utc: str | None = None,
    ) -> "HighscoreEntry":
        """Insert a new persistent highscore row and return the stored entry."""
        normalized_song_title = str(song_title).strip()
        if not normalized_song_title:
            raise HighscoreStorageError("Highscore insert requires a non-empty song_title.")

        timestamp_utc = (
            str(recorded_at_utc)
            if recorded_at_utc is not None
            else datetime.now(timezone.utc).isoformat()
        )
        connection = self.connect()
        try:
            cursor = connection.execute(
                """
                INSERT INTO highscores (score, percentage_hit, song_title, recorded_at_utc, headshot_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    int(score),
                    float(percentage_hit),
                    normalized_song_title,
                    timestamp_utc,
                    None if headshot_path is None else str(headshot_path),
                ),
            )
            connection.commit()
            row_id = int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise HighscoreStorageError(
                f"Failed to insert highscore row into {self.database_path}: {exc}"
            ) from exc
        finally:
            connection.close()

        return HighscoreEntry(
            row_id=row_id,
            score=int(score),
            percentage_hit=float(percentage_hit),
            song_title=normalized_song_title,
            recorded_at_utc=timestamp_utc,
            headshot_path=None if headshot_path is None else str(headshot_path),
        )


@dataclass(frozen=True)
class HighscoreEntry:
    """Normalized leaderboard row returned from persistent storage."""

    row_id: int
    score: int
    percentage_hit: float
    song_title: str
    recorded_at_utc: str
    headshot_path: str | None


@dataclass(frozen=True)
class HighscoreRankCandidate:
    """Comparable highscore candidate used by qualification checks."""

    score: int
    percentage_hit: float
    recorded_at_utc: str
    row_id: int


def _leaderboard_rank_key(candidate: HighscoreRankCandidate) -> tuple[int, float, str, int]:
    """Return a deterministic ordering key used across leaderboard ranking checks.

    Tie-break rule (higher rank first) is intentionally explicit and stable:
    1) Higher ``score`` ranks ahead of lower score.
    2) If scores tie, higher ``percentage_hit`` ranks ahead.
    3) If still tied, earlier ``recorded_at_utc`` ranks ahead.
    4) If still tied, lower ``row_id`` ranks ahead.
    """

    return (
        -int(candidate.score),
        -float(candidate.percentage_hit),
        str(candidate.recorded_at_utc),
        int(candidate.row_id),
    )


def qualifies_for_leaderboard(
    *,
    candidate_score: int,
    candidate_percentage_hit: float,
    candidate_recorded_at_utc: str,
    existing_entries: Sequence[HighscoreEntry],
    leaderboard_limit: int,
) -> bool:
    """Return whether a run qualifies for the configured top-N leaderboard.

    The comparison uses the deterministic tie-break rule defined in
    ``_leaderboard_rank_key``.
    """

    if leaderboard_limit <= 0:
        raise HighscoreStorageError(
            f"Leaderboard limit must be > 0 for qualification checks, got {leaderboard_limit}."
        )

    if len(existing_entries) < leaderboard_limit:
        return True

    ranked_existing = sorted(
        (
            HighscoreRankCandidate(
                score=entry.score,
                percentage_hit=entry.percentage_hit,
                recorded_at_utc=entry.recorded_at_utc,
                row_id=entry.row_id,
            )
            for entry in existing_entries
        ),
        key=_leaderboard_rank_key,
    )
    cutoff_entry = ranked_existing[leaderboard_limit - 1]

    # New runs do not have a persisted row id yet, so we use a very large
    # sentinel row id. This guarantees deterministic behavior for perfect ties:
    # an existing row keeps its place and an equal new run does not displace it.
    candidate = HighscoreRankCandidate(
        score=int(candidate_score),
        percentage_hit=float(candidate_percentage_hit),
        recorded_at_utc=str(candidate_recorded_at_utc),
        row_id=2**63 - 1,
    )
    return _leaderboard_rank_key(candidate) < _leaderboard_rank_key(cutoff_entry)


def fetch_configured_top_entries(
    storage: HighscoreStorage,
    config_payload: Mapping[str, Any],
) -> tuple[HighscoreEntry, ...]:
    """Fetch leaderboard rows using YAML-controlled highscore.list_size."""
    highscore_payload = config_payload.get("highscore")
    if not isinstance(highscore_payload, Mapping):
        raise HighscoreStorageError(
            "Config payload is missing required mapping key 'highscore'."
        )

    configured_limit = highscore_payload.get("list_size")
    if not isinstance(configured_limit, int):
        raise HighscoreStorageError(
            "Config key highscore.list_size must be an integer for leaderboard retrieval."
        )

    return storage.fetch_top_entries(limit=configured_limit)


def initialize_highscore_storage(data_dir: Path) -> HighscoreStorage:
    """Initialize highscore storage rooted at the provided data directory."""
    highscores_dir = data_dir / "highscores"
    headshots_dir = data_dir / "headshots"
    database_path = highscores_dir / "highscores.db"

    storage = HighscoreStorage(database_path=database_path, headshots_dir=headshots_dir)
    storage.ensure_storage_paths()
    connection = storage.connect()
    try:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute(_HIGH_SCORE_SCHEMA_SQL)
        connection.commit()
    except sqlite3.Error as exc:
        raise HighscoreStorageError(
            f"Failed to initialize highscore database schema for {database_path}: {exc}"
        ) from exc
    finally:
        connection.close()

    LOGGER.info(
        "Initialized persistent highscore storage at db=%s headshots_dir=%s.",
        storage.database_path,
        storage.headshots_dir,
    )
    return storage


def initialize_default_highscore_storage() -> HighscoreStorage:
    """Initialize project-local highscore storage directories and database file."""
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    return initialize_highscore_storage(data_dir)
