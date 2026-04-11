"""Circle interaction visual-state helpers for the liveview overlay."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

from midi_mover.circles import CircleGeometry
from midi_mover.pose import GameplayKeypoints, KeypointSample


CircleVisualStateName = Literal["idle", "active_contact", "hit_flash", "miss_flash"]
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
    state_name: CircleVisualStateName
    active_hands: tuple[str, ...]


@dataclass(frozen=True)
class CircleContactState:
    """Per-frame hand contact summary for a circle."""

    lane: int
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
    state_name: CircleVisualStateName
    expires_at_monotonic: float


class CircleVisualStateTracker:
    """Track active contact plus timed hit/miss flashes for each circle."""

    def __init__(self, *, hit_flash_duration_ms: int, miss_flash_duration_ms: int) -> None:
        self._hit_flash_duration_ms = max(0, int(hit_flash_duration_ms))
        self._miss_flash_duration_ms = max(0, int(miss_flash_duration_ms))
        self._previous_contacts: dict[int, tuple[str, ...]] = {}
        self._flash_states: dict[int, CircleFlashState] = {}

    def reset(self) -> None:
        self._previous_contacts.clear()
        self._flash_states.clear()

    def update(
        self,
        *,
        circle_geometries: tuple[CircleGeometry, ...],
        gameplay_keypoints: GameplayKeypoints | None,
        interaction_snapshot: InteractionStateSnapshot | None = None,
        now_monotonic: float | None = None,
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
        current_contacts = {contact.lane: contact.active_hands for contact in contacts}
        previous_contacts = dict(self._previous_contacts)

        for lane, hands in current_contacts.items():
            previous_hands = previous_contacts.get(lane, ())
            if hands and not previous_hands:
                self._flash_states[lane] = CircleFlashState(
                    lane=lane,
                    state_name="hit_flash",
                    expires_at_monotonic=timestamp + (self._hit_flash_duration_ms / 1000.0),
                )
            elif previous_hands and not hands:
                self._flash_states[lane] = CircleFlashState(
                    lane=lane,
                    state_name="miss_flash",
                    expires_at_monotonic=timestamp + (self._miss_flash_duration_ms / 1000.0),
                )

        self._flash_states = {
            lane: flash
            for lane, flash in self._flash_states.items()
            if flash.expires_at_monotonic > timestamp
        }
        self._previous_contacts = current_contacts

        resolved_states: list[CircleVisualState] = []
        for contact in contacts:
            flash = self._flash_states.get(contact.lane)
            if flash is not None:
                state_name = flash.state_name
            elif contact.active_hands:
                state_name = "active_contact"
            else:
                state_name = "idle"
            resolved_states.append(
                CircleVisualState(
                    lane=contact.lane,
                    state_name=state_name,
                    active_hands=contact.active_hands,
                )
            )
        return tuple(resolved_states)


def detect_hand_circle_interactions(
    *,
    circle_geometries: tuple[CircleGeometry, ...],
    gameplay_keypoints: GameplayKeypoints | None,
    now_monotonic: float | None = None,
) -> InteractionStateSnapshot:
    """Build a structured per-hand occupancy snapshot for all gameplay circles."""

    timestamp = now_monotonic
    if timestamp is None and gameplay_keypoints is not None:
        timestamp = gameplay_keypoints.captured_at_monotonic
    left_wrist = getattr(gameplay_keypoints, "left_wrist", None)
    right_wrist = getattr(gameplay_keypoints, "right_wrist", None)
    left_hand = _build_hand_interaction_state("L", left_wrist, circle_geometries)
    right_hand = _build_hand_interaction_state("R", right_wrist, circle_geometries)
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
    """Convert structured hand occupancy into circle-centric contact summaries."""

    return tuple(
        CircleContactState(lane=circle.lane, active_hands=snapshot.active_hands_for_lane(circle.lane))
        for circle in circle_geometries
    )


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
    wrist_sample: KeypointSample | None,
    circle_geometries: tuple[CircleGeometry, ...],
) -> HandInteractionState:
    wrist_xy = None if wrist_sample is None else wrist_sample.xy
    occupancies = tuple(
        HandCircleOccupancy(
            hand=hand,
            lane=circle.lane,
            is_inside=wrist_xy is not None and point_in_circle(wrist_xy, circle),
        )
        for circle in circle_geometries
    )
    return HandInteractionState(hand=hand, wrist_xy=wrist_xy, occupancies=occupancies)


def point_in_circle(point_xy: tuple[float, float], circle: CircleGeometry) -> bool:
    """Return whether a source-frame point lies within a gameplay circle."""

    dx = float(point_xy[0]) - float(circle.center_xy[0])
    dy = float(point_xy[1]) - float(circle.center_xy[1])
    return (dx * dx + dy * dy) <= float(circle.radius * circle.radius)
