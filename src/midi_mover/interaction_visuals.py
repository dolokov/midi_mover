"""Circle interaction visual-state helpers for the liveview overlay."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Literal

from midi_mover.circles import CircleGeometry
from midi_mover.hand_keypoints import FingertipSample
from midi_mover.pose import GameplayKeypoints, KeypointSample


CircleVisualStateName = Literal["idle", "precue", "active_contact", "hit_flash", "miss_flash"]
HandName = Literal["L", "R"]
TransitionStateName = Literal["idle", "enter", "stay", "exit"]


@dataclass(frozen=True)
class CircleVisualStyle:
    """Render style for one circle visual state."""

    outline_color: tuple[int, int, int]
    fill_color: tuple[int, int, int]
    label_color: tuple[int, int, int]


@dataclass(frozen=True)
class CircleVisualState:
    """Resolved state for one circle on the current frame."""

    lane: int
    hand: str  # "L" or "R"
    state_name: CircleVisualStateName
    active_hands: tuple[str, ...]


@dataclass(frozen=True)
class CircleContactState:
    """Per-frame hand contact summary for a circle."""

    lane: int
    hand: str  # "L" or "R"
    active_hands: tuple[str, ...]


@dataclass(frozen=True)
class HandCircleOccupancy:
    """One hand's overlap result for a single circle."""

    hand: HandName
    lane: int
    is_inside: bool

    @property
    def token(self) -> str:
        return f"{self.hand}{self.lane}"


@dataclass(frozen=True)
class HandInteractionState:
    """Per-hand occupancy across all five circles for the current frame."""

    hand: HandName
    wrist_xy: tuple[float, float] | None
    occupancies: tuple[HandCircleOccupancy, ...]

    @property
    def active_lanes(self) -> tuple[int, ...]:
        return tuple(occupancy.lane for occupancy in self.occupancies if occupancy.is_inside)

    @property
    def active_tokens(self) -> tuple[str, ...]:
        return tuple(occupancy.token for occupancy in self.occupancies if occupancy.is_inside)

    def is_inside_lane(self, lane: int) -> bool:
        return any(occupancy.lane == lane and occupancy.is_inside for occupancy in self.occupancies)


@dataclass(frozen=True)
class InteractionStateSnapshot:
    """Structured hand-in-circle interaction state for the current frame."""

    captured_at_monotonic: float | None
    left_hand: HandInteractionState
    right_hand: HandInteractionState
    active_tokens: tuple[str, ...]

    def hand_state(self, hand: HandName) -> HandInteractionState:
        return self.left_hand if hand == "L" else self.right_hand

    def active_hands_for_lane(self, lane: int) -> tuple[str, ...]:
        active_hands: list[str] = []
        if self.left_hand.is_inside_lane(lane):
            active_hands.append("L")
        if self.right_hand.is_inside_lane(lane):
            active_hands.append("R")
        return tuple(active_hands)


@dataclass(frozen=True)
class HandCircleTransitionState:
    """State transition for one hand-circle pair at the current frame."""

    hand: HandName
    lane: int
    token: str
    raw_is_inside: bool
    is_inside: bool
    state_name: TransitionStateName
    transitioned_at_monotonic: float | None
    entered_at_monotonic: float | None
    last_seen_inside_at_monotonic: float | None
    exited_at_monotonic: float | None
    debounce_until_monotonic: float | None


@dataclass(frozen=True)
class HandTransitionStateSnapshot:
    """Transition state for one hand across all five circles."""

    hand: HandName
    wrist_xy: tuple[float, float] | None
    lane_states: tuple[HandCircleTransitionState, ...]

    @property
    def active_lanes(self) -> tuple[int, ...]:
        return tuple(state.lane for state in self.lane_states if state.is_inside)

    @property
    def active_tokens(self) -> tuple[str, ...]:
        return tuple(state.token for state in self.lane_states if state.is_inside)


@dataclass(frozen=True)
class InteractionTransitionSnapshot:
    """State-transition-aware interaction snapshot for both hands."""

    captured_at_monotonic: float | None
    left_hand: HandTransitionStateSnapshot
    right_hand: HandTransitionStateSnapshot
    active_tokens: tuple[str, ...]

    def hand_state(self, hand: HandName) -> HandTransitionStateSnapshot:
        return self.left_hand if hand == "L" else self.right_hand


class HandCircleTransitionTracker:
    """Track enter/stay/exit transitions with timestamps per hand and lane."""

    def __init__(self, *, debounce_ms: int = 0) -> None:
        self._debounce_seconds = max(0.0, int(debounce_ms) / 1000.0)
        self._previous_states: dict[str, HandCircleTransitionState] = {}

    def reset(self) -> None:
        self._previous_states.clear()

    def update(self, snapshot: InteractionStateSnapshot) -> InteractionTransitionSnapshot:
        timestamp = snapshot.captured_at_monotonic
        left_hand = self._build_hand_snapshot(snapshot.left_hand, timestamp)
        right_hand = self._build_hand_snapshot(snapshot.right_hand, timestamp)
        return InteractionTransitionSnapshot(
            captured_at_monotonic=timestamp,
            left_hand=left_hand,
            right_hand=right_hand,
            active_tokens=left_hand.active_tokens + right_hand.active_tokens,
        )

    def _build_hand_snapshot(
        self,
        hand_state: HandInteractionState,
        timestamp: float | None,
    ) -> HandTransitionStateSnapshot:
        lane_states = tuple(
            self._resolve_transition_state(occupancy, timestamp)
            for occupancy in hand_state.occupancies
        )
        return HandTransitionStateSnapshot(
            hand=hand_state.hand,
            wrist_xy=hand_state.wrist_xy,
            lane_states=lane_states,
        )

    def _resolve_transition_state(
        self,
        occupancy: HandCircleOccupancy,
        timestamp: float | None,
    ) -> HandCircleTransitionState:
        previous = self._previous_states.get(occupancy.token)
        effective_is_inside = occupancy.is_inside

        if (
            not occupancy.is_inside
            and previous is not None
            and previous.is_inside
            and timestamp is not None
            and previous.last_seen_inside_at_monotonic is not None
            and (timestamp - previous.last_seen_inside_at_monotonic) < self._debounce_seconds
        ):
            effective_is_inside = True

        was_inside = previous.is_inside if previous is not None else False
        if effective_is_inside and not was_inside:
            state_name: TransitionStateName = "enter"
            transitioned_at = timestamp
            entered_at = timestamp
            last_seen_inside_at = timestamp
            exited_at = previous.exited_at_monotonic if previous is not None else None
        elif effective_is_inside and was_inside:
            state_name = "stay"
            transitioned_at = previous.transitioned_at_monotonic if previous is not None else timestamp
            entered_at = previous.entered_at_monotonic if previous is not None else timestamp
            last_seen_inside_at = timestamp
            exited_at = previous.exited_at_monotonic if previous is not None else None
        elif not effective_is_inside and was_inside:
            state_name = "exit"
            transitioned_at = timestamp
            entered_at = previous.entered_at_monotonic if previous is not None else None
            last_seen_inside_at = (
                previous.last_seen_inside_at_monotonic if previous is not None else None
            )
            exited_at = timestamp
        else:
            state_name = "idle"
            transitioned_at = previous.transitioned_at_monotonic if previous is not None else None
            entered_at = previous.entered_at_monotonic if previous is not None else None
            last_seen_inside_at = (
                previous.last_seen_inside_at_monotonic if previous is not None else None
            )
            exited_at = previous.exited_at_monotonic if previous is not None else None

        resolved = HandCircleTransitionState(
            hand=occupancy.hand,
            lane=occupancy.lane,
            token=occupancy.token,
            raw_is_inside=occupancy.is_inside,
            is_inside=effective_is_inside,
            state_name=state_name,
            transitioned_at_monotonic=transitioned_at,
            entered_at_monotonic=entered_at,
            last_seen_inside_at_monotonic=last_seen_inside_at,
            exited_at_monotonic=exited_at,
            debounce_until_monotonic=(
                None
                if timestamp is None or self._debounce_seconds <= 0.0
                else last_seen_inside_at + self._debounce_seconds
                if last_seen_inside_at is not None
                else None
            ),
        )
        self._previous_states[occupancy.token] = resolved
        return resolved


@dataclass(frozen=True)
class CircleFlashState:
    """Timed flash override for a circle."""

    lane: int
    hand: str  # "L" or "R"
    state_name: CircleVisualStateName
    expires_at_monotonic: float


class CircleVisualStateTracker:
    """Track active contact plus timed hit/miss flashes for each circle."""

    def __init__(self, *, hit_flash_duration_ms: int, miss_flash_duration_ms: int) -> None:
        self._hit_flash_duration_ms = max(0, int(hit_flash_duration_ms))
        self._miss_flash_duration_ms = max(0, int(miss_flash_duration_ms))
        self._previous_contacts: dict[tuple[str, int], tuple[str, ...]] = {}
        # Flash states keyed by (hand, lane) to support per-hand flashes.
        self._flash_states: dict[tuple[str, int], CircleFlashState] = {}

    def reset(self) -> None:
        self._previous_contacts.clear()
        self._flash_states.clear()

    def update(
        self,
        *,
        circle_geometries: tuple[CircleGeometry, ...],
        gameplay_keypoints: GameplayKeypoints | None,
        interaction_snapshot: InteractionStateSnapshot | None = None,
        hit_lanes: tuple[int, ...] = (),
        miss_lanes: tuple[int, ...] = (),
        now_monotonic: float | None = None,
        # Per-token hits/misses provide more granular flash control when
        # the caller knows which hand triggered each event.
        hit_tokens: tuple[str, ...] = (),
        miss_tokens: tuple[str, ...] = (),
        # Tokens whose circles should show the precue state (upcoming note).
        precue_tokens: tuple[str, ...] = (),
    ) -> tuple[CircleVisualState, ...]:
        timestamp = time.monotonic() if now_monotonic is None else float(now_monotonic)
        interactions = interaction_snapshot
        if interactions is None:
            interactions = detect_hand_circle_interactions(
                circle_geometries=circle_geometries,
                gameplay_keypoints=gameplay_keypoints,
                now_monotonic=timestamp,
            )
        contacts = interaction_snapshot_to_circle_contacts(interactions, circle_geometries)
        current_contacts = {(contact.hand, contact.lane): contact.active_hands for contact in contacts}

        # Resolve flash targets — prefer per-token (hand+lane) over lane-only.
        _apply_token_flashes(
            flash_states=self._flash_states,
            tokens=hit_tokens,
            state_name="hit_flash",
            duration_ms=self._hit_flash_duration_ms,
            timestamp=timestamp,
        )
        _apply_token_flashes(
            flash_states=self._flash_states,
            tokens=miss_tokens,
            state_name="miss_flash",
            duration_ms=self._miss_flash_duration_ms,
            timestamp=timestamp,
        )
        # Legacy lane-only flash support: flash both hands' circles for
        # that lane when no per-token info is available.
        if not hit_tokens:
            for lane in hit_lanes:
                for h in ("L", "R"):
                    self._flash_states[(h, int(lane))] = CircleFlashState(
                        lane=int(lane),
                        hand=h,
                        state_name="hit_flash",
                        expires_at_monotonic=timestamp + (self._hit_flash_duration_ms / 1000.0),
                    )
        if not miss_tokens:
            for lane in miss_lanes:
                for h in ("L", "R"):
                    self._flash_states[(h, int(lane))] = CircleFlashState(
                        lane=int(lane),
                        hand=h,
                        state_name="miss_flash",
                        expires_at_monotonic=timestamp + (self._miss_flash_duration_ms / 1000.0),
                    )

        self._flash_states = {
            key: flash
            for key, flash in self._flash_states.items()
            if flash.expires_at_monotonic > timestamp
        }
        self._previous_contacts = current_contacts

        # Normalise precue_tokens into a fast lookup set of (hand, lane) pairs.
        precue_set: set[tuple[str, int]] = set()
        for token in precue_tokens:
            token_s = str(token).strip()
            if len(token_s) < 2:
                continue
            hand_char = token_s[0].upper()
            lane_str = token_s[1:]
            if hand_char in ("L", "R") and lane_str.isdigit():
                precue_set.add((hand_char, int(lane_str)))

        resolved_states: list[CircleVisualState] = []
        for contact in contacts:
            flash = self._flash_states.get((contact.hand, contact.lane))
            if flash is not None:
                # Hit/miss flash has highest priority.
                state_name = flash.state_name
            elif contact.active_hands:
                # Hand is physically inside the circle.
                state_name = "active_contact"
            elif (contact.hand, contact.lane) in precue_set:
                # Upcoming note within the precue window.
                state_name = "precue"
            else:
                state_name = "idle"
            resolved_states.append(
                CircleVisualState(
                    lane=contact.lane,
                    hand=contact.hand,
                    state_name=state_name,
                    active_hands=contact.active_hands,
                )
            )
        return tuple(resolved_states)


def _apply_token_flashes(
    *,
    flash_states: dict[tuple[str, int], CircleFlashState],
    tokens: tuple[str, ...],
    state_name: CircleVisualStateName,
    duration_ms: int,
    timestamp: float,
) -> None:
    """Apply per-token flash entries to the flash state dict."""
    for token in tokens:
        token = str(token).strip()
        if len(token) < 2:
            continue
        hand_char = token[0].upper()
        lane_str = token[1:]
        if hand_char not in ("L", "R") or not lane_str.isdigit():
            continue
        lane = int(lane_str)
        flash_states[(hand_char, lane)] = CircleFlashState(
            lane=lane,
            hand=hand_char,
            state_name=state_name,
            expires_at_monotonic=timestamp + (duration_ms / 1000.0),
        )


def detect_hand_circle_interactions(
    *,
    circle_geometries: tuple[CircleGeometry, ...],
    gameplay_keypoints: GameplayKeypoints | None,
    now_monotonic: float | None = None,
    swap_hands: bool = False,
) -> InteractionStateSnapshot:
    """Build a structured per-hand occupancy snapshot for all gameplay circles.

    Interaction registration is fingertip-driven (stage-2 hand model): a hand is
    considered inside a lane if any confidence-gated fingertip assigned to that
    hand is inside the circle for that lane.

    Each hand only tests against circles tagged with its own hand identifier
    (``circle.hand == "L"`` for left, ``circle.hand == "R"`` for right).  When
    circles overlap, this guarantees that only the correct hand can trigger each
    circle regardless of spatial proximity.
    """

    timestamp = now_monotonic
    if timestamp is None and gameplay_keypoints is not None:
        timestamp = gameplay_keypoints.captured_at_monotonic
    left_wrist = getattr(gameplay_keypoints, "left_wrist", None)
    right_wrist = getattr(gameplay_keypoints, "right_wrist", None)
    fingertip_samples = tuple(getattr(gameplay_keypoints, "fingertip_samples", ()) or ())
    fingertips_by_hand = _resolve_hand_fingertips(
        fingertip_samples=fingertip_samples,
        left_wrist=left_wrist,
        right_wrist=right_wrist,
    )
    if swap_hands:
        fingertips_by_hand = {
            "L": fingertips_by_hand["R"],
            "R": fingertips_by_hand["L"],
        }

    # Each hand only tests against its own set of circles.
    left_circles = tuple(c for c in circle_geometries if c.hand == "L")
    right_circles = tuple(c for c in circle_geometries if c.hand == "R")
    # Fall back to all circles when hand tags are absent (legacy data).
    if not left_circles and not right_circles:
        left_circles = circle_geometries
        right_circles = circle_geometries

    left_hand = _build_hand_interaction_state("L", fingertips_by_hand["L"], left_circles)
    right_hand = _build_hand_interaction_state("R", fingertips_by_hand["R"], right_circles)
    active_tokens = left_hand.active_tokens + right_hand.active_tokens
    return InteractionStateSnapshot(
        captured_at_monotonic=timestamp,
        left_hand=left_hand,
        right_hand=right_hand,
        active_tokens=active_tokens,
    )


def interaction_snapshot_to_circle_contacts(
    snapshot: InteractionStateSnapshot,
    circle_geometries: tuple[CircleGeometry, ...],
) -> tuple[CircleContactState, ...]:
    """Convert structured hand occupancy into circle-centric contact summaries.

    Each circle carries a ``hand`` tag so the contact state correctly reflects
    whether the matching hand (L or R) is inside it.
    """

    contacts: list[CircleContactState] = []
    for circle in circle_geometries:
        hand_state = snapshot.hand_state(circle.hand)  # type: ignore[arg-type]
        is_inside = hand_state.is_inside_lane(circle.lane)
        contacts.append(
            CircleContactState(
                lane=circle.lane,
                hand=circle.hand,
                active_hands=(circle.hand,) if is_inside else (),
            )
        )
    return tuple(contacts)


def detect_circle_contacts(
    *,
    circle_geometries: tuple[CircleGeometry, ...],
    gameplay_keypoints: GameplayKeypoints | None,
) -> tuple[CircleContactState, ...]:
    """Determine which hands are currently inside each gameplay circle."""

    snapshot = detect_hand_circle_interactions(
        circle_geometries=circle_geometries,
        gameplay_keypoints=gameplay_keypoints,
    )
    return interaction_snapshot_to_circle_contacts(snapshot, circle_geometries)


def describe_transition_snapshot(snapshot: InteractionTransitionSnapshot) -> str:
    """Return a concise log string for hand-circle transition states."""

    parts: list[str] = []
    for hand_snapshot in (snapshot.left_hand, snapshot.right_hand):
        transitions = [
            f"{state.token}:{state.state_name}"
            for state in hand_snapshot.lane_states
            if state.state_name != "idle" or state.is_inside
        ]
        if not transitions:
            transitions.append(f"{hand_snapshot.hand}:idle")
        parts.append(" ".join(transitions))
    return " | ".join(parts)


def _build_hand_interaction_state(
    hand: HandName,
    fingertip_points: tuple[tuple[float, float], ...],
    circle_geometries: tuple[CircleGeometry, ...],
) -> HandInteractionState:
    """Test fingertip points against the circles assigned to this hand."""
    wrist_xy = _compute_centroid_xy(fingertip_points)
    occupancies = tuple(
        HandCircleOccupancy(
            hand=hand,
            lane=circle.lane,
            is_inside=any(point_in_circle(point_xy, circle) for point_xy in fingertip_points),
        )
        for circle in circle_geometries
    )
    return HandInteractionState(hand=hand, wrist_xy=wrist_xy, occupancies=occupancies)


def _resolve_hand_fingertips(
    *,
    fingertip_samples: tuple[FingertipSample, ...],
    left_wrist: KeypointSample | None,
    right_wrist: KeypointSample | None,
) -> dict[HandName, tuple[tuple[float, float], ...]]:
    grouped: dict[int, list[tuple[float, float]]] = {}
    for sample in fingertip_samples:
        grouped.setdefault(int(sample.hand_index), []).append((float(sample.xy[0]), float(sample.xy[1])))

    if not grouped:
        return {"L": (), "R": ()}

    grouped_centroids: list[tuple[int, tuple[tuple[float, float], ...], tuple[float, float]]] = []
    for hand_index, points in grouped.items():
        normalized_points = tuple(points)
        centroid = _compute_centroid_xy(normalized_points)
        if centroid is None:
            continue
        grouped_centroids.append((hand_index, normalized_points, centroid))

    if not grouped_centroids:
        return {"L": (), "R": ()}

    assigned: dict[HandName, list[tuple[float, float]]] = {"L": [], "R": []}
    if left_wrist is not None and right_wrist is not None:
        for _, points, centroid in grouped_centroids:
            left_distance = math.dist(centroid, left_wrist.xy)
            right_distance = math.dist(centroid, right_wrist.xy)
            target = "L" if left_distance <= right_distance else "R"
            assigned[target].extend(points)
    elif left_wrist is not None:
        for _, points, _ in grouped_centroids:
            assigned["L"].extend(points)
    elif right_wrist is not None:
        for _, points, _ in grouped_centroids:
            assigned["R"].extend(points)
    else:
        grouped_centroids.sort(key=lambda item: (item[2][0], item[0]))
        if len(grouped_centroids) == 1:
            assigned["L"].extend(grouped_centroids[0][1])
        else:
            left_anchor_x = grouped_centroids[0][2][0]
            right_anchor_x = grouped_centroids[-1][2][0]
            for _, points, centroid in grouped_centroids:
                if centroid[0] <= left_anchor_x:
                    assigned["L"].extend(points)
                    continue
                if centroid[0] >= right_anchor_x:
                    assigned["R"].extend(points)
                    continue
                to_left = abs(centroid[0] - left_anchor_x)
                to_right = abs(right_anchor_x - centroid[0])
                target = "L" if to_left <= to_right else "R"
                assigned[target].extend(points)

    return {
        "L": tuple(assigned["L"]),
        "R": tuple(assigned["R"]),
    }


def _compute_centroid_xy(points: tuple[tuple[float, float], ...]) -> tuple[float, float] | None:
    if not points:
        return None
    sum_x = 0.0
    sum_y = 0.0
    for x, y in points:
        sum_x += float(x)
        sum_y += float(y)
    count = float(len(points))
    return sum_x / count, sum_y / count


def point_in_circle(point_xy: tuple[float, float], circle: CircleGeometry) -> bool:
    """Return whether a source-frame point lies within a gameplay circle."""

    dx = float(point_xy[0]) - float(circle.center_xy[0])
    dy = float(point_xy[1]) - float(circle.center_xy[1])
    return (dx * dx + dy * dy) <= float(circle.radius * circle.radius)
