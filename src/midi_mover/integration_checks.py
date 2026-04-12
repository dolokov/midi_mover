"""Runtime integration checks for fingertip-driven interaction and hand assignment."""

from __future__ import annotations

from dataclasses import dataclass

from midi_mover.audio import (
    AudioPlaybackConfig,
    GesturePlaybackController,
    GestureSoundMapping,
    GestureSoundSpec,
    LoadedGestureSounds,
)
from midi_mover.circles import CircleGeometry
from midi_mover.hand_keypoints import FingertipSample
from midi_mover.interaction_visuals import HandCircleTransitionTracker, detect_hand_circle_interactions
from midi_mover.pose import GameplayKeypoints, KeypointSample


@dataclass
class _FakeSound:
    token: str
    volume: float = 0.0

    def set_volume(self, value: float) -> None:
        self.volume = float(value)


@dataclass
class _FakeChannel:
    played_tokens: list[str]
    busy: bool = False

    def get_busy(self) -> bool:
        return self.busy

    def play(self, sound: _FakeSound, loops: int = 0) -> None:
        self.busy = True
        self.played_tokens.append(sound.token)

    def fadeout(self, _: int) -> None:
        self.busy = False

    def stop(self) -> None:
        self.busy = False


@dataclass
class _FakeMixer:
    channels: list[_FakeChannel]

    def find_channel(self, force: bool = False) -> _FakeChannel | None:
        for channel in self.channels:
            if not channel.get_busy():
                return channel
        if force and self.channels:
            return self.channels[0]
        return None


@dataclass
class _FakePygame:
    mixer: _FakeMixer


def verify_fingertip_audio_integration() -> None:
    """Verify fingertip events still drive transition + audio hooks correctly."""

    circles = (
        CircleGeometry(lane=1, center_xy=(100.0, 100.0), radius=20, offset_xy=(0.0, 0.0)),
    )
    left_wrist = KeypointSample(
        name="left_wrist",
        xy=(90.0, 100.0),
        confidence=1.0,
        source="detected",
        observed_at_monotonic=1.0,
    )
    right_wrist = KeypointSample(
        name="right_wrist",
        xy=(200.0, 100.0),
        confidence=1.0,
        source="detected",
        observed_at_monotonic=1.0,
    )
    inside = GameplayKeypoints(
        person_index=0,
        captured_at_monotonic=1.0,
        head_center_xy=(100.0, 80.0),
        left_eye=None,
        right_eye=None,
        left_wrist=left_wrist,
        right_wrist=right_wrist,
        fingertip_samples=(
            FingertipSample(
                hand_index=0,
                fingertip_name="index_tip",
                keypoint_index=8,
                xy=(100.0, 100.0),
                confidence=1.0,
            ),
        ),
    )
    outside = GameplayKeypoints(
        person_index=0,
        captured_at_monotonic=1.1,
        head_center_xy=(100.0, 80.0),
        left_eye=None,
        right_eye=None,
        left_wrist=left_wrist,
        right_wrist=right_wrist,
        fingertip_samples=(
            FingertipSample(
                hand_index=0,
                fingertip_name="index_tip",
                keypoint_index=8,
                xy=(220.0, 220.0),
                confidence=1.0,
            ),
        ),
    )

    playback_config = AudioPlaybackConfig(
        note_duration_seconds=0.1,
        gesture_volume=0.8,
        max_concurrent_sounds=2,
        restart_busy_channel=True,
        sustain_while_inside=True,
        release_fade_ms=0,
    )
    mapping = GestureSoundMapping(
        specs={"L1": GestureSoundSpec(token="L1", frequency_hz=261.63, waveform="sine")}
    )
    sounds = {"L1": _FakeSound(token="L1")}
    gesture_sounds = LoadedGestureSounds(
        mapping=mapping,
        sounds=sounds,
        playback_config=playback_config,
    )
    pygame_module = _FakePygame(mixer=_FakeMixer(channels=[_FakeChannel(played_tokens=[])]))

    transition_tracker = HandCircleTransitionTracker(debounce_ms=0)
    playback_controller = GesturePlaybackController()

    inside_snapshot = detect_hand_circle_interactions(circle_geometries=circles, gameplay_keypoints=inside)
    enter_transitions = transition_tracker.update(inside_snapshot)
    started_tokens = playback_controller.update(
        pygame_module=pygame_module,
        transition_snapshot=enter_transitions,
        gesture_sounds=gesture_sounds,
    )
    if "L1" not in started_tokens:
        raise RuntimeError("Fingertip integration check failed: expected L1 playback on fingertip enter.")

    outside_snapshot = detect_hand_circle_interactions(circle_geometries=circles, gameplay_keypoints=outside)
    exit_transitions = transition_tracker.update(outside_snapshot)
    playback_controller.update(
        pygame_module=pygame_module,
        transition_snapshot=exit_transitions,
        gesture_sounds=gesture_sounds,
    )
    playback_controller.reset()


def verify_mirrored_hand_assignment_behavior() -> None:
    """Verify mirrored handedness behavior is deterministic for swap on/off."""

    circles = (
        CircleGeometry(lane=1, center_xy=(100.0, 100.0), radius=20, offset_xy=(0.0, 0.0)),
    )
    left_wrist = KeypointSample(
        name="left_wrist",
        xy=(90.0, 100.0),
        confidence=1.0,
        source="detected",
        observed_at_monotonic=1.0,
    )
    right_wrist = KeypointSample(
        name="right_wrist",
        xy=(220.0, 100.0),
        confidence=1.0,
        source="detected",
        observed_at_monotonic=1.0,
    )
    left_inside = GameplayKeypoints(
        person_index=0,
        captured_at_monotonic=1.0,
        head_center_xy=(100.0, 80.0),
        left_eye=None,
        right_eye=None,
        left_wrist=left_wrist,
        right_wrist=right_wrist,
        fingertip_samples=(
            FingertipSample(
                hand_index=0,
                fingertip_name="index_tip",
                keypoint_index=8,
                xy=(100.0, 100.0),
                confidence=1.0,
            ),
        ),
    )

    no_swap_snapshot = detect_hand_circle_interactions(
        circle_geometries=circles,
        gameplay_keypoints=left_inside,
        swap_hands=False,
    )
    if "L1" not in no_swap_snapshot.active_tokens:
        raise RuntimeError(
            "Mirrored hand-assignment integration check failed: expected L1 without swap_hands. "
            f"active_tokens={no_swap_snapshot.active_tokens}"
        )

    swapped_snapshot = detect_hand_circle_interactions(
        circle_geometries=circles,
        gameplay_keypoints=left_inside,
        swap_hands=True,
    )
    if "R1" not in swapped_snapshot.active_tokens:
        raise RuntimeError(
            "Mirrored hand-assignment integration check failed: expected R1 with swap_hands=True. "
            f"active_tokens={swapped_snapshot.active_tokens}"
        )
