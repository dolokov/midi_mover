"""Post-song highscore flow orchestration helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Mapping

import cv2

from midi_mover.camera import CameraFrameReader
from midi_mover.highscore_capture import capture_headshot_from_runtime
from midi_mover.highscore_board import show_highscore_board_screen
from midi_mover.highscore_countdown import show_highscore_countdown_screen
from midi_mover.highscore_flow import decide_post_song_route
from midi_mover.highscore_storage import HighscoreEntry, HighscoreStorage, HighscoreStorageError, fetch_configured_top_entries, initialize_default_highscore_storage
from midi_mover.pose import GameplayKeypointTracker, PrimaryPersonTracker
from midi_mover.song_summary import SongCompleteSummary


def persist_captured_highscore_entry(
    *,
    storage: HighscoreStorage,
    summary: SongCompleteSummary,
    image_rgb: Any,
) -> HighscoreEntry:
    """Persist a captured headshot image and insert the linked leaderboard row."""

    timestamp_fragment = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"headshot_{timestamp_fragment}.png"
    headshot_path = storage.headshots_dir / filename

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    write_ok = cv2.imwrite(str(headshot_path), image_bgr)
    if not write_ok:
        raise HighscoreStorageError(f"Failed to save headshot image to {headshot_path}.")

    try:
        relative_headshot_path = str(headshot_path.relative_to(Path(__file__).resolve().parents[2]))
    except ValueError:
        relative_headshot_path = str(headshot_path)

    return storage.insert_entry(
        score=int(summary.score),
        percentage_hit=float(summary.percentage_hit),
        song_title=str(summary.song_title),
        headshot_path=relative_headshot_path,
    )


def persist_highscore_entry_without_headshot(
    *,
    storage: HighscoreStorage,
    summary: SongCompleteSummary,
) -> HighscoreEntry:
    """Persist a qualifying leaderboard row without a headshot image path."""

    return storage.insert_entry(
        score=int(summary.score),
        percentage_hit=float(summary.percentage_hit),
        song_title=str(summary.song_title),
        headshot_path=None,
    )


def run_post_song_highscore_handoff(
    *,
    summary: SongCompleteSummary,
    config_payload: Mapping[str, Any],
    pygame_module: Any,
    window: Any,
    frame_reader: CameraFrameReader,
    pose_model: Any,
    primary_person_tracker: PrimaryPersonTracker,
    gameplay_keypoint_tracker: GameplayKeypointTracker,
    logger: logging.Logger,
) -> None:
    """Resolve post-song route and optionally render the highscore countdown screen."""

    try:
        highscore_storage = initialize_default_highscore_storage()
        route_decision = decide_post_song_route(
            final_run_summary=summary,
            config_payload=config_payload,
            storage=highscore_storage,
        )
    except HighscoreStorageError as exc:
        logger.warning(
            "Post-song highscore route decision failed (%s); defaulting to leaderboard transition.",
            exc,
        )
        return

    logger.info(
        "Post-song route decision resolved: qualifies_for_highscore=%s next_state=%s.",
        route_decision.qualifies_for_highscore,
        route_decision.route,
    )
    if route_decision.route != "headshot_countdown":
        _show_leaderboard_board(
            pygame_module=pygame_module,
            window=window,
            config_payload=config_payload,
            storage=highscore_storage,
            logger=logger,
            highlight_entry_row_id=None,
            current_run_summary=summary,
        )
        return

    show_countdown = show_highscore_countdown_screen(
        pygame_module=pygame_module,
        window=window,
        countdown_seconds=float(config_payload["highscore"].get("headshot_countdown_seconds", 3.0)),
    )
    if not show_countdown:
        logger.info("Highscore countdown screen closed by user.")
        return

    headshot_capture = capture_headshot_from_runtime(
        frame_reader=frame_reader,
        pygame_module=pygame_module,
        pose_model=pose_model,
        primary_person_tracker=primary_person_tracker,
        gameplay_keypoint_tracker=gameplay_keypoint_tracker,
        pose_confidence_threshold=float(config_payload["pose"].get("confidence_threshold", 0.5)),
        pose_iou_threshold=float(config_payload["pose"].get("iou_threshold", 0.45)),
        headshot_crop_margin=float(config_payload["highscore"].get("headshot_crop_margin", 0.35)),
    )
    if headshot_capture is None:
        logger.warning(
            "Highscore countdown completed, but fresh headshot capture failed. "
            "Persisting qualifying highscore entry without headshot image."
        )
        try:
            inserted_entry = persist_highscore_entry_without_headshot(
                storage=highscore_storage,
                summary=summary,
            )
        except HighscoreStorageError as exc:
            logger.warning(
                "Fallback highscore persistence without headshot also failed (%s).",
                exc,
            )
            return
        logger.info(
            "Persisted qualifying highscore entry id=%s without headshot_path due to capture failure: "
            "score=%s percentage_hit=%.1f.",
            inserted_entry.row_id,
            inserted_entry.score,
            inserted_entry.percentage_hit,
        )
        _show_leaderboard_board(
            pygame_module=pygame_module,
            window=window,
            config_payload=config_payload,
            storage=highscore_storage,
            logger=logger,
            highlight_entry_row_id=inserted_entry.row_id,
            current_run_summary=None,
        )
        return

    crop_x, crop_y, crop_w, crop_h = headshot_capture.crop_xywh
    logger.info(
        "Captured fresh highscore headshot frame after countdown completion: "
        "crop=(x=%s y=%s w=%s h=%s).",
        crop_x,
        crop_y,
        crop_w,
        crop_h,
    )

    try:
        inserted_entry = persist_captured_highscore_entry(
            storage=highscore_storage,
            summary=summary,
            image_rgb=headshot_capture.image_rgb,
        )
    except HighscoreStorageError as exc:
        logger.warning(
            "Highscore headshot capture succeeded, but persistence with headshot failed (%s). "
            "Falling back to persistence without headshot image.",
            exc,
        )
        try:
            inserted_entry = persist_highscore_entry_without_headshot(
                storage=highscore_storage,
                summary=summary,
            )
        except HighscoreStorageError as fallback_exc:
            logger.warning(
                "Fallback highscore persistence without headshot also failed (%s).",
                fallback_exc,
            )
            return
        logger.info(
            "Persisted qualifying highscore entry id=%s without headshot_path after image-write failure: "
            "score=%s percentage_hit=%.1f.",
            inserted_entry.row_id,
            inserted_entry.score,
            inserted_entry.percentage_hit,
        )
        _show_leaderboard_board(
            pygame_module=pygame_module,
            window=window,
            config_payload=config_payload,
            storage=highscore_storage,
            logger=logger,
            highlight_entry_row_id=inserted_entry.row_id,
            current_run_summary=None,
        )
        return

    logger.info(
        "Persisted qualifying highscore entry id=%s with headshot_path='%s' score=%s percentage_hit=%.1f.",
        inserted_entry.row_id,
        inserted_entry.headshot_path,
        inserted_entry.score,
        inserted_entry.percentage_hit,
    )
    _show_leaderboard_board(
        pygame_module=pygame_module,
        window=window,
        config_payload=config_payload,
        storage=highscore_storage,
        logger=logger,
        highlight_entry_row_id=inserted_entry.row_id,
        current_run_summary=None,
    )


def _show_leaderboard_board(
    *,
    pygame_module: Any,
    window: Any,
    config_payload: Mapping[str, Any],
    storage: HighscoreStorage,
    logger: logging.Logger,
    highlight_entry_row_id: int | None = None,
    current_run_summary: SongCompleteSummary | None = None,
) -> None:
    """Render leaderboard rows and thumbnails using persisted sorted entries."""

    try:
        entries = fetch_configured_top_entries(storage, config_payload)
    except HighscoreStorageError as exc:
        logger.warning("Skipping leaderboard screen due to retrieval failure: %s", exc)
        return

    app_payload = config_payload.get("app")
    highscore_payload = config_payload.get("highscore")
    if not isinstance(app_payload, Mapping) or not isinstance(highscore_payload, Mapping):
        logger.warning("Skipping leaderboard screen due to missing app/highscore config mappings.")
        return

    board_duration_seconds = float(
        highscore_payload.get(
            "leaderboard_screen_duration_seconds",
            app_payload.get("song_summary_screen_duration_seconds", 3.0),
        )
    )
    thumbnail_width = int(highscore_payload.get("thumbnail_width", 160))
    thumbnail_height = int(highscore_payload.get("thumbnail_height", 160))
    show_leaderboard = show_highscore_board_screen(
        pygame_module=pygame_module,
        window=window,
        entries=entries,
        thumbnail_width=thumbnail_width,
        thumbnail_height=thumbnail_height,
        duration_seconds=board_duration_seconds,
        highlight_row_id=highlight_entry_row_id,
        current_run_summary=current_run_summary,
    )
    if not show_leaderboard:
        logger.info("Leaderboard screen closed by user.")
