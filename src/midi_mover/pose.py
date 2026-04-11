"""Pose inference and stable primary-person selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

from midi_mover.hand_keypoints import FingertipSample, extract_fingertip_samples


@dataclass(frozen=True)
class PoseCandidate:
    """One person candidate extracted from a pose inference result."""

    index: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    center_xy: tuple[float, float]
    area: float
    track_id: int | None
    confident_keypoint_count: int


@dataclass(frozen=True)
class PrimaryPersonSelection:
    """The currently selected primary person for gameplay framing."""

    candidate: PoseCandidate
    reason: str
    selected_at_monotonic: float


@dataclass(frozen=True)
class KeypointSample:
    """One extracted gameplay-relevant keypoint."""

    name: str
    xy: tuple[float, float]
    confidence: float
    source: str
    observed_at_monotonic: float


@dataclass(frozen=True)
class GameplayKeypoints:
    """Tracked eye and wrist keypoints for the selected primary person."""

    person_index: int
    captured_at_monotonic: float
    head_center_xy: tuple[float, float] | None
    left_eye: KeypointSample | None
    right_eye: KeypointSample | None
    left_wrist: KeypointSample | None
    right_wrist: KeypointSample | None
    fingertip_samples: tuple[FingertipSample, ...] = ()

    @property
    def all_visible(self) -> bool:
        return all(
            point is not None and point.source == "detected"
            for point in (self.left_eye, self.right_eye, self.left_wrist, self.right_wrist)
        )

    @property
    def missing_names(self) -> tuple[str, ...]:
        missing: list[str] = []
        for name, point in (
            ("left_eye", self.left_eye),
            ("right_eye", self.right_eye),
            ("left_wrist", self.left_wrist),
            ("right_wrist", self.right_wrist),
        ):
            if point is None:
                missing.append(name)
        return tuple(missing)

    def get(self, name: str) -> KeypointSample | None:
        return getattr(self, name)


def compute_head_center_xy(
    left_eye: KeypointSample | None,
    right_eye: KeypointSample | None,
) -> tuple[float, float] | None:
    """Compute the gameplay head-center coordinate from the two eye keypoints."""

    if left_eye is None or right_eye is None:
        return None

    return (
        (left_eye.xy[0] + right_eye.xy[0]) / 2.0,
        (left_eye.xy[1] + right_eye.xy[1]) / 2.0,
    )


class PoseProcessingError(RuntimeError):
    """Raised when pose results cannot be interpreted safely."""


@dataclass(frozen=True)
class HandRoiInference:
    """Stage-2 hand-model inference output mapped back to full-frame coordinates."""

    roi_xyxy: tuple[int, int, int, int]
    roi_shape_hw: tuple[int, int]
    hand_result: Any
    remapped_keypoints_xy: tuple[tuple[tuple[float, float], ...], ...]
    remapped_keypoints_conf: tuple[tuple[float, ...], ...]
    fingertip_samples: tuple[FingertipSample, ...]


COCO_KEYPOINT_INDICES: dict[str, int] = {
    "left_eye": 1,
    "right_eye": 2,
    "left_wrist": 9,
    "right_wrist": 10,
}

def run_pose_inference(model: Any, frame_bgr: Any, *, conf: float, iou: float) -> Any:
    """Run one Ultralytics pose inference pass and return the first result object."""

    try:
        results = model.predict(frame_bgr, conf=conf, iou=iou, verbose=False)
    except Exception as exc:  # pragma: no cover - depends on runtime model/backend
        raise PoseProcessingError(f"Pose inference failed: {exc}") from exc

    if not results:
        return None
    return results[0]


def run_stage2_hand_inference_on_person_roi(
    *,
    hand_model: Any,
    frame_bgr: Any,
    selection: PrimaryPersonSelection | None,
    expansion_px: int,
    conf: float,
    iou: float,
) -> HandRoiInference | None:
    """Run stage-2 hand inference on an expanded stage-1 selected person ROI."""

    if selection is None:
        return None

    frame_height = int(getattr(frame_bgr, "shape", [0, 0])[0])
    frame_width = int(getattr(frame_bgr, "shape", [0, 0])[1])
    if frame_width <= 0 or frame_height <= 0:
        return None

    roi = _compute_expanded_roi(
        bbox_xyxy=selection.candidate.bbox_xyxy,
        expansion_px=int(expansion_px),
        frame_width=frame_width,
        frame_height=frame_height,
    )
    x1, y1, x2, y2 = roi
    if x2 <= x1 or y2 <= y1:
        return None

    roi_frame_bgr = frame_bgr[y1:y2, x1:x2]
    if roi_frame_bgr is None or int(getattr(roi_frame_bgr, "size", 0)) <= 0:
        return None

    hand_result = run_pose_inference(hand_model, roi_frame_bgr, conf=conf, iou=iou)
    if hand_result is None:
        return HandRoiInference(
            roi_xyxy=roi,
            roi_shape_hw=(y2 - y1, x2 - x1),
            hand_result=None,
            remapped_keypoints_xy=(),
            remapped_keypoints_conf=(),
            fingertip_samples=(),
        )

    remapped_xy = _remap_keypoint_rows_to_full_frame(
        xy_rows=_to_rows(getattr(getattr(hand_result, "keypoints", None), "xy", None)),
        offset_xy=(x1, y1),
    )
    remapped_conf = tuple(
        tuple(float(value) for value in row)
        for row in _to_rows(getattr(getattr(hand_result, "keypoints", None), "conf", None))
    )
    fingertip_samples = extract_fingertip_samples(
        remapped_keypoints_xy=remapped_xy,
        remapped_keypoints_conf=remapped_conf,
        confidence_threshold=conf,
    )
    return HandRoiInference(
        roi_xyxy=roi,
        roi_shape_hw=(y2 - y1, x2 - x1),
        hand_result=hand_result,
        remapped_keypoints_xy=remapped_xy,
        remapped_keypoints_conf=remapped_conf,
        fingertip_samples=fingertip_samples,
    )


def extract_pose_candidates(result: Any, confidence_threshold: float) -> list[PoseCandidate]:
    """Extract confident person candidates from an Ultralytics pose result."""

    if result is None or getattr(result, "boxes", None) is None:
        return []

    boxes = result.boxes
    xyxy_rows = _to_rows(getattr(boxes, "xyxy", None))
    conf_rows = _to_flat_list(getattr(boxes, "conf", None))
    id_rows = _to_flat_list(getattr(boxes, "id", None)) if getattr(boxes, "id", None) is not None else []

    keypoint_rows = []
    keypoints = getattr(result, "keypoints", None)
    if keypoints is not None and getattr(keypoints, "conf", None) is not None:
        keypoint_rows = _to_rows(getattr(keypoints, "conf", None))

    candidates: list[PoseCandidate] = []
    for index, bbox in enumerate(xyxy_rows):
        if len(bbox) != 4:
            continue

        confidence = float(conf_rows[index]) if index < len(conf_rows) else 0.0
        if confidence < confidence_threshold:
            continue

        x1, y1, x2, y2 = (float(value) for value in bbox)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height
        if area <= 0.0:
            continue

        kp_confidences = keypoint_rows[index] if index < len(keypoint_rows) else []
        confident_keypoint_count = sum(1 for value in kp_confidences if float(value) >= confidence_threshold)
        track_id = None
        if index < len(id_rows) and id_rows[index] is not None:
            try:
                track_id = int(float(id_rows[index]))
            except (TypeError, ValueError):
                track_id = None

        candidates.append(
            PoseCandidate(
                index=index,
                confidence=confidence,
                bbox_xyxy=(x1, y1, x2, y2),
                center_xy=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                area=area,
                track_id=track_id,
                confident_keypoint_count=confident_keypoint_count,
            )
        )

    return candidates


class GameplayKeypointTracker:
    """Extract gameplay keypoints with confidence checks and short-lived fallback reuse."""

    def __init__(self, *, confidence_threshold: float, fallback_timeout_seconds: float) -> None:
        self._confidence_threshold = float(confidence_threshold)
        self._fallback_timeout_seconds = max(0.0, float(fallback_timeout_seconds))
        self._last_samples: dict[int, dict[str, KeypointSample]] = {}

    def reset(self) -> None:
        self._last_samples.clear()

    def extract(
        self,
        result: Any,
        selection: PrimaryPersonSelection | None,
        hand_inference: HandRoiInference | None = None,
        now_monotonic: float | None = None,
    ) -> GameplayKeypoints | None:
        if selection is None:
            return None

        timestamp = time.monotonic() if now_monotonic is None else float(now_monotonic)
        person_index = int(selection.candidate.index)
        xy_rows, conf_rows = _extract_keypoint_rows(result)

        person_xy = xy_rows[person_index] if person_index < len(xy_rows) else []
        person_conf = conf_rows[person_index] if person_index < len(conf_rows) else []
        cached_samples = self._last_samples.get(person_index, {})
        resolved_samples: dict[str, KeypointSample | None] = {}
        fresh_samples: dict[str, KeypointSample] = {}

        for name, index in COCO_KEYPOINT_INDICES.items():
            sample = _extract_keypoint_sample(
                name=name,
                keypoint_index=index,
                xy_row=person_xy,
                conf_row=person_conf,
                confidence_threshold=self._confidence_threshold,
                timestamp=timestamp,
            )
            if sample is not None:
                resolved_samples[name] = sample
                fresh_samples[name] = sample
                continue

            resolved_samples[name] = _reuse_fallback_sample(
                fallback=cached_samples.get(name),
                timestamp=timestamp,
                fallback_timeout_seconds=self._fallback_timeout_seconds,
            )

        persisted_samples = {
            name: sample
            for name, sample in {**cached_samples, **fresh_samples}.items()
            if (timestamp - sample.observed_at_monotonic) <= self._fallback_timeout_seconds
        }
        for name, sample in fresh_samples.items():
            persisted_samples[name] = KeypointSample(
                name=sample.name,
                xy=sample.xy,
                confidence=sample.confidence,
                source="detected",
                observed_at_monotonic=timestamp,
            )
        self._last_samples[person_index] = persisted_samples

        left_eye = resolved_samples.get("left_eye")
        right_eye = resolved_samples.get("right_eye")

        return GameplayKeypoints(
            person_index=person_index,
            captured_at_monotonic=timestamp,
            head_center_xy=compute_head_center_xy(left_eye, right_eye),
            left_eye=left_eye,
            right_eye=right_eye,
            left_wrist=resolved_samples.get("left_wrist"),
            right_wrist=resolved_samples.get("right_wrist"),
            fingertip_samples=()
            if hand_inference is None
            else tuple(hand_inference.fingertip_samples),
        )

    def describe(self, keypoints: GameplayKeypoints | None) -> str:
        if keypoints is None:
            return "none"

        parts: list[str] = []
        if keypoints.head_center_xy is None:
            parts.append("head_center=missing")
        else:
            parts.append(f"head_center=({keypoints.head_center_xy[0]:.1f},{keypoints.head_center_xy[1]:.1f})")
        for name in ("left_eye", "right_eye", "left_wrist", "right_wrist"):
            sample = keypoints.get(name)
            if sample is None:
                parts.append(f"{name}=missing")
                continue
            age_seconds = max(0.0, keypoints.captured_at_monotonic - sample.observed_at_monotonic)
            parts.append(
                f"{name}=({sample.xy[0]:.1f},{sample.xy[1]:.1f}) conf={sample.confidence:.2f} src={sample.source} age={age_seconds:.3f}s"
            )
        parts.append(f"fingertips={len(keypoints.fingertip_samples)}")
        return " ".join(parts)


class PrimaryPersonTracker:
    """Maintain a stable primary-person choice across frames."""

    def __init__(self, *, confidence_threshold: float, lost_timeout_seconds: float) -> None:
        self._confidence_threshold = float(confidence_threshold)
        self._lost_timeout_seconds = float(lost_timeout_seconds)
        self._last_selection: PrimaryPersonSelection | None = None

    @property
    def last_selection(self) -> PrimaryPersonSelection | None:
        return self._last_selection

    def select(self, result: Any, now_monotonic: float | None = None) -> PrimaryPersonSelection | None:
        timestamp = time.monotonic() if now_monotonic is None else float(now_monotonic)
        candidates = extract_pose_candidates(result, self._confidence_threshold)
        if not candidates:
            if self._last_selection is not None:
                age = timestamp - self._last_selection.selected_at_monotonic
                if age <= self._lost_timeout_seconds:
                    return self._last_selection
            self._last_selection = None
            return None

        selected, reason = self._choose_candidate(candidates, timestamp)
        selection = PrimaryPersonSelection(
            candidate=selected,
            reason=reason,
            selected_at_monotonic=timestamp,
        )
        self._last_selection = selection
        return selection

    def _choose_candidate(
        self,
        candidates: list[PoseCandidate],
        timestamp: float,
    ) -> tuple[PoseCandidate, str]:
        previous = self._last_selection
        if previous is not None and (timestamp - previous.selected_at_monotonic) <= self._lost_timeout_seconds:
            previous_center = previous.candidate.center_xy
            previous_diag = _bbox_diagonal(previous.candidate.bbox_xyxy)

            ranked = []
            for candidate in candidates:
                distance = math.dist(previous_center, candidate.center_xy)
                normalized_distance = distance / max(previous_diag, _bbox_diagonal(candidate.bbox_xyxy), 1.0)
                same_track = int(
                    previous.candidate.track_id is not None
                    and candidate.track_id is not None
                    and previous.candidate.track_id == candidate.track_id
                )
                plausibly_same = int(same_track or normalized_distance <= 0.85)
                ranked.append(
                    (
                        plausibly_same,
                        same_track,
                        -normalized_distance,
                        candidate.confident_keypoint_count,
                        candidate.area,
                        candidate.confidence,
                        candidate,
                    )
                )

            best = max(ranked, key=lambda item: item[:-1])
            if best[0]:
                reason = "matched previous target by track id" if best[1] else "matched previous target by proximity"
                return best[-1], reason

        fallback = max(
            candidates,
            key=lambda candidate: (
                candidate.area,
                candidate.confident_keypoint_count,
                candidate.confidence,
            ),
        )
        return fallback, "selected largest confident detection"


def _bbox_diagonal(bbox_xyxy: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    return math.hypot(x2 - x1, y2 - y1)


def _extract_keypoint_rows(result: Any) -> tuple[list[list[Any]], list[list[Any]]]:
    if result is None:
        return [], []

    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return [], []

    xy_rows = _to_rows(getattr(keypoints, "xy", None))
    conf_rows = _to_rows(getattr(keypoints, "conf", None))
    return xy_rows, conf_rows


def _compute_expanded_roi(
    *,
    bbox_xyxy: tuple[float, float, float, float],
    expansion_px: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    expand = max(0, int(expansion_px))
    roi_x1 = max(0, int(math.floor(x1)) - expand)
    roi_y1 = max(0, int(math.floor(y1)) - expand)
    roi_x2 = min(frame_width, int(math.ceil(x2)) + expand)
    roi_y2 = min(frame_height, int(math.ceil(y2)) + expand)
    return roi_x1, roi_y1, roi_x2, roi_y2


def _remap_keypoint_rows_to_full_frame(
    *,
    xy_rows: list[list[Any]],
    offset_xy: tuple[int, int],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    offset_x, offset_y = float(offset_xy[0]), float(offset_xy[1])
    remapped: list[tuple[tuple[float, float], ...]] = []
    for row in xy_rows:
        points: list[tuple[float, float]] = []
        for point in row:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            points.append((float(point[0]) + offset_x, float(point[1]) + offset_y))
        remapped.append(tuple(points))
    return tuple(remapped)


def _extract_keypoint_sample(
    *,
    name: str,
    keypoint_index: int,
    xy_row: list[Any],
    conf_row: list[Any],
    confidence_threshold: float,
    timestamp: float,
) -> KeypointSample | None:
    if keypoint_index >= len(xy_row) or keypoint_index >= len(conf_row):
        return None

    xy = xy_row[keypoint_index]
    if not isinstance(xy, (list, tuple)) or len(xy) < 2:
        return None

    confidence = float(conf_row[keypoint_index])
    if confidence < confidence_threshold:
        return None

    x = float(xy[0])
    y = float(xy[1])
    return KeypointSample(
        name=name,
        xy=(x, y),
        confidence=confidence,
        source="detected",
        observed_at_monotonic=timestamp,
    )


def _reuse_fallback_sample(
    *,
    fallback: KeypointSample | None,
    timestamp: float,
    fallback_timeout_seconds: float,
) -> KeypointSample | None:
    if fallback is None:
        return None

    age_seconds = timestamp - fallback.observed_at_monotonic
    if age_seconds > fallback_timeout_seconds:
        return None

    return KeypointSample(
        name=fallback.name,
        xy=fallback.xy,
        confidence=fallback.confidence,
        source="fallback",
        observed_at_monotonic=fallback.observed_at_monotonic,
    )


def _to_rows(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value is None:
        return []
    if isinstance(value, list):
        return [list(row) if isinstance(row, (list, tuple)) else [row] for row in value]
    return []


def _to_flat_list(value: Any) -> list[Any]:
    rows = _to_rows(value)
    flattened: list[Any] = []
    for row in rows:
        if len(row) == 1:
            flattened.append(row[0])
        else:
            flattened.extend(row)
    return flattened