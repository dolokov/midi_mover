"""Hit-window judgment helpers for matching interaction events to target notes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from midi_mover.interaction_visuals import InteractionTransitionSnapshot


@dataclass(frozen=True)
class JudgedHit:
    """A successful event-to-note match within the configured hit window."""

    target_note_id: str
    token: str
    hand: str
    lane: int
    note_timestamp_ms: float
    event_timestamp_ms: float
    timing_error_ms: float


@dataclass(frozen=True)
class GameplayScoreState:
    """Real-time gameplay score/accuracy counters derived from judgments."""

    score: int
    combo: int
    max_combo: int
    total_hits: int
    total_misses: int
    total_notes: int
    percentage_hit: float


class GameplayScoreTracker:
    """Track score, combo, and hit/miss metrics from judgment outcomes."""

    def __init__(
        self,
        *,
        total_notes: int,
        hit_score: int,
        miss_score: int,
        combo_bonus: int,
    ) -> None:
        self._total_notes = max(0, int(total_notes))
        self._hit_score = int(hit_score)
        self._miss_score = int(miss_score)
        self._combo_bonus = int(combo_bonus)
        self._score = 0
        self._combo = 0
        self._max_combo = 0
        self._total_hits = 0
        self._total_misses = 0
        self._synced_total_misses = 0

    @property
    def state(self) -> GameplayScoreState:
        percentage_hit = 0.0
        if self._total_notes > 0:
            percentage_hit = (float(self._total_hits) / float(self._total_notes)) * 100.0
        return GameplayScoreState(
            score=int(self._score),
            combo=int(self._combo),
            max_combo=int(self._max_combo),
            total_hits=int(self._total_hits),
            total_misses=int(self._total_misses),
            total_notes=int(self._total_notes),
            percentage_hit=percentage_hit,
        )

    def register_hits(self, hits: tuple[JudgedHit, ...]) -> None:
        for _ in hits:
            self._combo += 1
            self._max_combo = max(self._max_combo, self._combo)
            self._total_hits += 1
            combo_bonus_score = self._combo_bonus * max(0, self._combo - 1)
            self._score += self._hit_score + combo_bonus_score

    def register_misses(self, miss_count: int) -> None:
        for _ in range(max(0, int(miss_count))):
            self._combo = 0
            self._total_misses += 1
            self._score += self._miss_score

    def sync_total_misses(self, total_misses: int) -> None:
        """Synchronize tracker misses from a cumulative judge miss counter."""

        resolved_total = max(0, int(total_misses))
        if resolved_total < self._synced_total_misses:
            self._synced_total_misses = resolved_total
            return
        delta = resolved_total - self._synced_total_misses
        if delta > 0:
            self.register_misses(delta)
            self._synced_total_misses = resolved_total


class HitWindowJudge:
    """Match hand/lane enter events to unmatched target notes within a tolerance window."""

    def __init__(self, *, normalized_target_notes: tuple[Any, ...], hit_window_ms: float) -> None:
        self._notes = tuple(normalized_target_notes)
        self._hit_window_ms = max(0.0, float(hit_window_ms))
        self._matched_note_ids: set[str] = set()
        self._missed_note_ids: set[str] = set()
        self._newly_missed_note_ids: set[str] = set()
        self._hits: list[JudgedHit] = []

    @property
    def matched_note_ids(self) -> frozenset[str]:
        return frozenset(self._matched_note_ids)

    @property
    def hits(self) -> tuple[JudgedHit, ...]:
        return tuple(self._hits)

    @property
    def missed_note_ids(self) -> frozenset[str]:
        return frozenset(self._missed_note_ids)

    @property
    def judged_note_outcomes(self) -> dict[str, str]:
        outcomes: dict[str, str] = {}
        for note_id in self._missed_note_ids:
            outcomes[note_id] = "miss"
        for note_id in self._matched_note_ids:
            outcomes[note_id] = "hit"
        return outcomes

    def register_transition_snapshot(
        self,
        *,
        transition_snapshot: InteractionTransitionSnapshot,
        song_started_monotonic: float | None,
        pre_song_lead_in_ms: float,
    ) -> tuple[JudgedHit, ...]:
        """Consume enter transitions and return newly judged hits for this frame."""

        if song_started_monotonic is None:
            return ()

        lead_in_seconds = max(0.0, float(pre_song_lead_in_ms) / 1000.0)
        new_hits: list[JudgedHit] = []
        now_elapsed_ms: float | None = None

        for hand_snapshot in (transition_snapshot.left_hand, transition_snapshot.right_hand):
            for lane_state in hand_snapshot.lane_states:
                if lane_state.state_name != "enter":
                    continue
                event_monotonic = lane_state.transitioned_at_monotonic
                if event_monotonic is None:
                    continue

                event_timestamp_ms = (
                    (float(event_monotonic) - float(song_started_monotonic) - lead_in_seconds) * 1000.0
                )
                if now_elapsed_ms is None or event_timestamp_ms > now_elapsed_ms:
                    now_elapsed_ms = event_timestamp_ms
                note = self._find_best_note_match(
                    hand=str(lane_state.hand),
                    lane=int(lane_state.lane),
                    event_timestamp_ms=event_timestamp_ms,
                )
                if note is None:
                    continue

                note_id = str(getattr(note, "target_note_id", ""))
                note_timestamp_ms = float(getattr(note, "timestamp_ms", 0.0))
                hit = JudgedHit(
                    target_note_id=note_id,
                    token=f"{lane_state.hand}{lane_state.lane}",
                    hand=str(lane_state.hand),
                    lane=int(lane_state.lane),
                    note_timestamp_ms=note_timestamp_ms,
                    event_timestamp_ms=event_timestamp_ms,
                    timing_error_ms=event_timestamp_ms - note_timestamp_ms,
                )
                self._matched_note_ids.add(note_id)
                self._hits.append(hit)
                new_hits.append(hit)

        self._finalize_misses_until_elapsed_ms(now_elapsed_ms)
        return tuple(new_hits)

    def mark_misses_for_song_elapsed_ms(self, song_elapsed_ms: float) -> None:
        """Persist miss judgments for notes that are past the late hit boundary."""

        self._finalize_misses_until_elapsed_ms(float(song_elapsed_ms))

    def miss_count(self) -> int:
        """Return total number of notes judged as misses so far."""

        return len(self._missed_note_ids)

    def consume_newly_missed_note_ids(self) -> tuple[str, ...]:
        """Return and clear note IDs that were newly judged missed since last consume."""

        if not self._newly_missed_note_ids:
            return ()
        note_ids = tuple(sorted(self._newly_missed_note_ids))
        self._newly_missed_note_ids.clear()
        return note_ids

    def _finalize_misses_until_elapsed_ms(self, song_elapsed_ms: float | None) -> None:
        if song_elapsed_ms is None:
            return

        for note in self._notes:
            note_id = str(getattr(note, "target_note_id", ""))
            if not note_id:
                continue
            if note_id in self._matched_note_ids or note_id in self._missed_note_ids:
                continue
            note_timestamp_ms = float(getattr(note, "timestamp_ms", 0.0))
            if float(song_elapsed_ms) > (note_timestamp_ms + self._hit_window_ms):
                self._missed_note_ids.add(note_id)
                self._newly_missed_note_ids.add(note_id)

    def _find_best_note_match(self, *, hand: str, lane: int, event_timestamp_ms: float) -> Any | None:
        best_note: Any | None = None
        best_abs_error_ms: float | None = None

        for note in self._notes:
            note_id = str(getattr(note, "target_note_id", ""))
            if note_id in self._matched_note_ids:
                continue
            if str(getattr(note, "hand", "")).upper() != hand.upper():
                continue
            if int(getattr(note, "lane", 0)) != lane:
                continue

            note_timestamp_ms = float(getattr(note, "timestamp_ms", 0.0))
            abs_error_ms = abs(event_timestamp_ms - note_timestamp_ms)
            if abs_error_ms > self._hit_window_ms:
                continue

            if best_note is None or best_abs_error_ms is None:
                best_note = note
                best_abs_error_ms = abs_error_ms
                continue

            if abs_error_ms < best_abs_error_ms:
                best_note = note
                best_abs_error_ms = abs_error_ms
                continue

            if abs_error_ms == best_abs_error_ms:
                if float(getattr(note, "timestamp_ms", 0.0)) < float(getattr(best_note, "timestamp_ms", 0.0)):
                    best_note = note
                    best_abs_error_ms = abs_error_ms

        return best_note
