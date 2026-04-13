"""Highscore-flow integration checks used by startup smoke tests."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from midi_mover.highscore_capture import compute_headshot_crop_xywh
from midi_mover.highscore_handoff import (
    persist_captured_highscore_entry,
    persist_highscore_entry_without_headshot,
)
from midi_mover.highscore_storage import initialize_highscore_storage
from midi_mover.pose import GameplayKeypoints, KeypointSample
from midi_mover.song_summary import SongCompleteSummary


def verify_headshot_crop_margin_behavior() -> None:
    """Verify countdown-complete headshot crop uses tracked head center and YAML margin."""

    left_eye = KeypointSample(
        name="left_eye",
        xy=(140.0, 100.0),
        confidence=1.0,
        source="detected",
        observed_at_monotonic=1.0,
    )
    right_eye = KeypointSample(
        name="right_eye",
        xy=(180.0, 100.0),
        confidence=1.0,
        source="detected",
        observed_at_monotonic=1.0,
    )
    keypoints = GameplayKeypoints(
        person_index=0,
        captured_at_monotonic=1.0,
        head_center_xy=(160.0, 100.0),
        left_eye=left_eye,
        right_eye=right_eye,
        left_wrist=None,
        right_wrist=None,
        fingertip_samples=(),
    )

    tight_crop = compute_headshot_crop_xywh(
        frame_width=320,
        frame_height=240,
        gameplay_keypoints=keypoints,
        crop_margin=0.0,
    )
    loose_crop = compute_headshot_crop_xywh(
        frame_width=320,
        frame_height=240,
        gameplay_keypoints=keypoints,
        crop_margin=0.35,
    )

    tight_x, tight_y, tight_w, tight_h = tight_crop
    loose_x, loose_y, loose_w, loose_h = loose_crop

    if loose_w <= tight_w or loose_h <= tight_h:
        raise RuntimeError(
            "Headshot-crop integration check failed: positive headshot_crop_margin should expand crop size. "
            f"tight={tight_crop} loose={loose_crop}"
        )
    if tight_x < 0 or tight_y < 0 or (tight_x + tight_w) > 320 or (tight_y + tight_h) > 240:
        raise RuntimeError(
            "Headshot-crop integration check failed: tight crop must stay within frame bounds. "
            f"tight={tight_crop}"
        )
    if loose_x < 0 or loose_y < 0 or (loose_x + loose_w) > 320 or (loose_y + loose_h) > 240:
        raise RuntimeError(
            "Headshot-crop integration check failed: loose crop must stay within frame bounds. "
            f"loose={loose_crop}"
        )


def verify_highscore_headshot_persistence_behavior() -> None:
    """Verify captured and fallback highscore persistence behaviors."""

    summary = SongCompleteSummary(
        song_title="Integration Test Song",
        score=1234,
        notes_hit=12,
        notes_missed=3,
        total_notes=15,
        percentage_hit=80.0,
        max_combo=7,
    )
    image_rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    image_rgb[..., 0] = 255

    with TemporaryDirectory(prefix="midi_mover_headshot_check_") as temp_dir:
        storage = initialize_highscore_storage(Path(temp_dir) / "data")
        inserted = persist_captured_highscore_entry(
            storage=storage,
            summary=summary,
            image_rgb=image_rgb,
        )

        if inserted.headshot_path is None or not inserted.headshot_path.strip():
            raise RuntimeError(
                "Headshot persistence integration check failed: inserted leaderboard row did not include headshot path."
            )

        project_like_root = Path(temp_dir)
        persisted_headshot_path = project_like_root / inserted.headshot_path
        if not persisted_headshot_path.exists():
            raise RuntimeError(
                "Headshot persistence integration check failed: saved image file was not found at "
                f"{persisted_headshot_path}."
            )

        entries = storage.fetch_top_entries(limit=5)
        if len(entries) != 1:
            raise RuntimeError(
                "Headshot persistence integration check failed: expected one leaderboard row after insertion, "
                f"found {len(entries)}."
            )

        reloaded = entries[0]
        if reloaded.headshot_path != inserted.headshot_path:
            raise RuntimeError(
                "Headshot persistence integration check failed: persisted row headshot path mismatch. "
                f"inserted={inserted.headshot_path} reloaded={reloaded.headshot_path}"
            )

    verify_highscore_capture_failure_fallback_behavior()


def verify_highscore_capture_failure_fallback_behavior() -> None:
    """Verify qualifying entries persist safely when no headshot image is available."""

    summary = SongCompleteSummary(
        song_title="Fallback Capture Test Song",
        score=2222,
        notes_hit=20,
        notes_missed=5,
        total_notes=25,
        percentage_hit=80.0,
        max_combo=9,
    )

    with TemporaryDirectory(prefix="midi_mover_headshot_fallback_check_") as temp_dir:
        storage = initialize_highscore_storage(Path(temp_dir) / "data")
        inserted = persist_highscore_entry_without_headshot(
            storage=storage,
            summary=summary,
        )

        if inserted.headshot_path is not None:
            raise RuntimeError(
                "Headshot fallback integration check failed: fallback-inserted row should not have a headshot path. "
                f"got={inserted.headshot_path!r}"
            )

        entries = storage.fetch_top_entries(limit=5)
        if len(entries) != 1:
            raise RuntimeError(
                "Headshot fallback integration check failed: expected one leaderboard row after fallback insertion, "
                f"found {len(entries)}."
            )

        reloaded = entries[0]
        if reloaded.score != summary.score or abs(reloaded.percentage_hit - summary.percentage_hit) > 1e-9:
            raise RuntimeError(
                "Headshot fallback integration check failed: persisted fallback row score/percentage mismatch. "
                f"stored=(score={reloaded.score}, percentage_hit={reloaded.percentage_hit})"
            )
        if reloaded.headshot_path is not None:
            raise RuntimeError(
                "Headshot fallback integration check failed: persisted fallback row should keep headshot_path as None."
            )
