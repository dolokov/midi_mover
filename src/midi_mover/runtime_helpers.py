"""Runtime-loop helper functions extracted to keep modules focused and compact."""

from __future__ import annotations

import time
from typing import Any

from midi_mover.interaction_visuals import PrecueVisualHint
from midi_mover.judgment import HitWindowJudge
from midi_mover.pose import PrimaryPersonSelection


def extract_stage1_keypoints_for_selection(
    *,
    result: Any,
    selection: PrimaryPersonSelection | None,
) -> tuple[tuple[tuple[float, float], ...], tuple[float, ...]]:
    """Return stage-1 keypoints for the currently selected primary person."""
    if result is None or selection is None:
        return (), ()

    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return (), ()

    keypoints_xy = _to_rows(getattr(keypoints, "xy", None))
    keypoints_conf = _to_rows(getattr(keypoints, "conf", None))
    person_index = int(selection.candidate.index)
    if person_index < 0 or person_index >= len(keypoints_xy):
        return (), ()

    xy_row = keypoints_xy[person_index]
    conf_row = keypoints_conf[person_index] if person_index < len(keypoints_conf) else []
    normalized_xy: list[tuple[float, float]] = []
    for point in xy_row:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        normalized_xy.append((float(point[0]), float(point[1])))
    normalized_conf = tuple(float(value) for value in conf_row)
    return tuple(normalized_xy), normalized_conf


def compute_precue_hints(
    *,
    normalized_target_notes: tuple[Any, ...],
    song_started_monotonic: float | None,
    pre_song_lead_in_ms: float,
    song_speed_multiplier: float,
    precue_ms: float,
    hit_window_ms: float,
    hit_window_judge: HitWindowJudge | None,
) -> tuple[PrecueVisualHint, ...]:
    """Return continuous urgency hints for notes currently inside the pre-cue window."""
    if not normalized_target_notes or song_started_monotonic is None or precue_ms <= 0.0:
        return ()

    lead_in_seconds = max(0.0, float(pre_song_lead_in_ms) / 1000.0)
    safe_song_speed = max(1e-6, float(song_speed_multiplier))
    song_elapsed_ms = (
        (time.monotonic() - float(song_started_monotonic) - lead_in_seconds)
        * 1000.0
        * safe_song_speed
    )

    already_judged: frozenset[str] = frozenset()
    if hit_window_judge is not None:
        already_judged = hit_window_judge.matched_note_ids | hit_window_judge.missed_note_ids

    best_by_token: dict[str, tuple[float, float]] = {}
    primary_token = ""
    primary_time_to_note_ms = float("inf")
    for note in normalized_target_notes:
        note_id = str(getattr(note, "target_note_id", ""))
        if note_id and note_id in already_judged:
            continue
        note_ts_ms = float(getattr(note, "timestamp_ms", 0.0))
        if song_elapsed_ms < (note_ts_ms - precue_ms):
            continue
        if song_elapsed_ms > (note_ts_ms + hit_window_ms):
            continue

        token = str(getattr(note, "token", "")).strip().upper()
        if not token:
            continue

        time_to_note_ms = note_ts_ms - song_elapsed_ms
        urgency = 1.0 - (max(0.0, time_to_note_ms) / max(1.0, float(precue_ms)))
        urgency = max(0.0, min(1.0, urgency))

        previous = best_by_token.get(token)
        if previous is None or time_to_note_ms < previous[1]:
            best_by_token[token] = (urgency, time_to_note_ms)
        if time_to_note_ms < primary_time_to_note_ms:
            primary_time_to_note_ms = time_to_note_ms
            primary_token = token

    hints: list[PrecueVisualHint] = []
    for token, (urgency, time_to_note_ms) in best_by_token.items():
        hints.append(
            PrecueVisualHint(
                token=token,
                urgency=urgency,
                time_to_note_ms=time_to_note_ms,
                is_primary=(token == primary_token),
            )
        )
    hints.sort(key=lambda item: (item.time_to_note_ms, item.token))
    return tuple(hints)


def _to_rows(value: Any) -> list[list[Any]]:
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value is None or not isinstance(value, list):
        return []
    rows: list[list[Any]] = []
    for row in value:
        if isinstance(row, (list, tuple)):
            rows.append(list(row))
        else:
            rows.append([row])
    return rows
