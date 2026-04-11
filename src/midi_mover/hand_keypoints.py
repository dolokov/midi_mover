"""Stage-2 hand-keypoint helpers for fingertip extraction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FingertipSample:
    """One confidence-gated stage-2 fingertip keypoint in full-frame coordinates."""

    hand_index: int
    fingertip_name: str
    keypoint_index: int
    xy: tuple[float, float]
    confidence: float


# Ultralytics hand-keypoint index mapping (21 points):
# 4=thumb_tip, 8=index_tip, 12=middle_tip, 16=ring_tip, 20=pinky_tip.
HAND_FINGERTIP_KEYPOINT_INDICES: dict[str, int] = {
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}


def extract_fingertip_samples(
    *,
    remapped_keypoints_xy: tuple[tuple[tuple[float, float], ...], ...],
    remapped_keypoints_conf: tuple[tuple[float, ...], ...],
    confidence_threshold: float,
) -> tuple[FingertipSample, ...]:
    """Extract confidence-gated fingertip points from stage-2 hand keypoints."""

    samples: list[FingertipSample] = []
    for hand_index, xy_row in enumerate(remapped_keypoints_xy):
        conf_row = remapped_keypoints_conf[hand_index] if hand_index < len(remapped_keypoints_conf) else ()
        for fingertip_name, keypoint_index in HAND_FINGERTIP_KEYPOINT_INDICES.items():
            if keypoint_index >= len(xy_row) or keypoint_index >= len(conf_row):
                continue
            confidence = float(conf_row[keypoint_index])
            if confidence < confidence_threshold:
                continue
            xy = xy_row[keypoint_index]
            samples.append(
                FingertipSample(
                    hand_index=hand_index,
                    fingertip_name=fingertip_name,
                    keypoint_index=keypoint_index,
                    xy=(float(xy[0]), float(xy[1])),
                    confidence=confidence,
                )
            )
    return tuple(samples)