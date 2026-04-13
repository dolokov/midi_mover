"""Headshot capture helpers for qualifying highscore runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from midi_mover.camera import CameraFrameReader
from midi_mover.pose import (
    GameplayKeypointTracker,
    PoseProcessingError,
    PrimaryPersonTracker,
    run_pose_inference,
)


@dataclass(frozen=True)
class HeadshotCaptureResult:
    """Captured frame details for a new-highscore headshot."""

    crop_xywh: tuple[int, int, int, int]
    image_rgb: Any


def capture_headshot_from_runtime(
    *,
    frame_reader: CameraFrameReader,
    pygame_module: Any,
    pose_model: Any,
    primary_person_tracker: PrimaryPersonTracker,
    gameplay_keypoint_tracker: GameplayKeypointTracker,
    pose_confidence_threshold: float,
    pose_iou_threshold: float,
    headshot_crop_margin: float,
) -> HeadshotCaptureResult | None:
    """Capture a fresh frame and crop a headshot around the tracked head region."""

    frame = frame_reader.read(pygame_module)
    try:
        pose_result = run_pose_inference(
            pose_model,
            frame.bgr_frame,
            conf=float(pose_confidence_threshold),
            iou=float(pose_iou_threshold),
        )
    except PoseProcessingError:
        return None

    selection = primary_person_tracker.select(pose_result)
    gameplay_keypoints = gameplay_keypoint_tracker.extract(pose_result, selection)
    crop_x, crop_y, crop_w, crop_h = compute_headshot_crop_xywh(
        frame_width=frame.width,
        frame_height=frame.height,
        gameplay_keypoints=gameplay_keypoints,
        crop_margin=max(0.0, float(headshot_crop_margin)),
    )
    image_rgb = frame.rgb_frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
    if image_rgb is None or int(getattr(image_rgb, "size", 0)) <= 0:
        return None
    return HeadshotCaptureResult(crop_xywh=(crop_x, crop_y, crop_w, crop_h), image_rgb=image_rgb)


def compute_headshot_crop_xywh(
    *,
    frame_width: int,
    frame_height: int,
    gameplay_keypoints: Any,
    crop_margin: float,
) -> tuple[int, int, int, int]:
    """Compute a clamped headshot crop anchored to detected eye/head geometry."""

    safe_w = max(1, int(frame_width))
    safe_h = max(1, int(frame_height))
    center_x = safe_w / 2.0
    center_y = safe_h / 2.0
    eye_distance = max(1.0, min(safe_w, safe_h) * 0.15)

    if gameplay_keypoints is not None:
        head_center = getattr(gameplay_keypoints, "head_center_xy", None)
        left_eye = getattr(gameplay_keypoints, "left_eye", None)
        right_eye = getattr(gameplay_keypoints, "right_eye", None)
        if head_center is not None:
            center_x = float(head_center[0])
            center_y = float(head_center[1])
        if left_eye is not None and right_eye is not None:
            eye_distance = max(
                1.0,
                abs(float(right_eye.xy[0]) - float(left_eye.xy[0])),
            )

    margin = max(0.0, float(crop_margin))
    half_width = eye_distance * (1.25 + margin)
    above = eye_distance * (1.85 + margin)
    below = eye_distance * (2.45 + margin * 1.5)

    left = int(round(center_x - half_width))
    right = int(round(center_x + half_width))
    top = int(round(center_y - above))
    bottom = int(round(center_y + below))

    left = max(0, min(left, safe_w - 1))
    top = max(0, min(top, safe_h - 1))
    right = max(left + 1, min(right, safe_w))
    bottom = max(top + 1, min(bottom, safe_h))

    return left, top, right - left, bottom - top
