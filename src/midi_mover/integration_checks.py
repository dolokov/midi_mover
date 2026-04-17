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
from midi_mover.interaction_visuals import (
    CircleVisualStateTracker,
    HandCircleTransitionTracker,
    detect_hand_circle_interactions,
)
from midi_mover.judgment import GameplayScoreTracker, HitWindowJudge
from midi_mover.midi_parser import NormalizedTargetNote
from midi_mover.pose import GameplayKeypoints, KeypointSample
from midi_mover.runtime_loop import run_persistent_liveview_loop


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
        CircleGeometry(lane=1, hand="L", center_xy=(100.0, 100.0), radius=20, offset_xy=(0.0, 0.0)),
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
        CircleGeometry(lane=1, hand="L", center_xy=(100.0, 100.0), radius=20, offset_xy=(0.0, 0.0)),
        CircleGeometry(lane=1, hand="R", center_xy=(100.0, 100.0), radius=20, offset_xy=(0.0, 0.0)),
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


def verify_hit_window_judgment_behavior() -> None:
    """Verify hand/lane hit-window matching for enter transitions is deterministic."""

    target_notes = (
        NormalizedTargetNote(
            target_note_id="tn_1",
            token="L1",
            lane=1,
            hand="L",
            timestamp_seconds=1.0,
            timestamp_ms=1000.0,
            duration_seconds=0.2,
            duration_ms=200.0,
            source_note_number=60,
            source_channel=0,
            source_track_index=0,
            source_velocity=100,
            source_start_tick=480,
            source_end_tick=576,
        ),
        NormalizedTargetNote(
            target_note_id="tn_2",
            token="R2",
            lane=2,
            hand="R",
            timestamp_seconds=1.2,
            timestamp_ms=1200.0,
            duration_seconds=0.2,
            duration_ms=200.0,
            source_note_number=66,
            source_channel=0,
            source_track_index=0,
            source_velocity=100,
            source_start_tick=576,
            source_end_tick=672,
        ),
    )

    judge = HitWindowJudge(normalized_target_notes=target_notes, hit_window_ms=120.0)
    transition_tracker = HandCircleTransitionTracker(debounce_ms=0)
    circles = (
        CircleGeometry(lane=1, hand="L", center_xy=(100.0, 100.0), radius=20, offset_xy=(0.0, 0.0)),
        CircleGeometry(lane=2, hand="R", center_xy=(200.0, 100.0), radius=20, offset_xy=(0.0, 0.0)),
    )

    left_wrist = KeypointSample(
        name="left_wrist",
        xy=(100.0, 100.0),
        confidence=1.0,
        source="detected",
        observed_at_monotonic=101.0,
    )
    right_wrist = KeypointSample(
        name="right_wrist",
        xy=(200.0, 100.0),
        confidence=1.0,
        source="detected",
        observed_at_monotonic=101.0,
    )

    left_enter_keypoints = GameplayKeypoints(
        person_index=0,
        captured_at_monotonic=102.0,
        head_center_xy=(150.0, 80.0),
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
    left_enter = transition_tracker.update(
        detect_hand_circle_interactions(circle_geometries=circles, gameplay_keypoints=left_enter_keypoints)
    )
    left_hits = judge.register_transition_snapshot(
        transition_snapshot=left_enter,
        song_started_monotonic=100.0,
        pre_song_lead_in_ms=1000.0,
    )
    if len(left_hits) != 1 or left_hits[0].target_note_id != "tn_1":
        raise RuntimeError(
            "Hit-window judgment integration check failed: expected left-hand lane-1 enter to match tn_1. "
            f"hits={left_hits}"
        )

    right_enter_keypoints = GameplayKeypoints(
        person_index=0,
        captured_at_monotonic=102.2,
        head_center_xy=(150.0, 80.0),
        left_eye=None,
        right_eye=None,
        left_wrist=left_wrist,
        right_wrist=right_wrist,
        fingertip_samples=(
            FingertipSample(
                hand_index=1,
                fingertip_name="index_tip",
                keypoint_index=8,
                xy=(200.0, 100.0),
                confidence=1.0,
            ),
        ),
    )
    right_enter = transition_tracker.update(
        detect_hand_circle_interactions(circle_geometries=circles, gameplay_keypoints=right_enter_keypoints)
    )
    right_hits = judge.register_transition_snapshot(
        transition_snapshot=right_enter,
        song_started_monotonic=100.0,
        pre_song_lead_in_ms=1000.0,
    )
    if len(right_hits) != 1 or right_hits[0].target_note_id != "tn_2":
        raise RuntimeError(
            "Hit-window judgment integration check failed: expected right-hand lane-2 enter to match tn_2. "
            f"hits={right_hits}"
        )


def verify_gameplay_score_tracking_behavior() -> None:
    """Verify scoring, combo, hits/misses, and hit percentage calculations."""

    tracker = GameplayScoreTracker(
        total_notes=4,
        hit_score=100,
        miss_score=-25,
        combo_bonus=10,
    )

    tracker.register_hits(
        (
            type(
                "_DummyHit",
                (),
                {"target_note_id": "a", "token": "L1", "hand": "L", "lane": 1, "note_timestamp_ms": 0.0, "event_timestamp_ms": 0.0, "timing_error_ms": 0.0},
            )(),
            type(
                "_DummyHit",
                (),
                {"target_note_id": "b", "token": "R2", "hand": "R", "lane": 2, "note_timestamp_ms": 0.0, "event_timestamp_ms": 0.0, "timing_error_ms": 0.0},
            )(),
        )
    )
    tracker.sync_total_misses(1)
    tracker.register_hits(
        (
            type(
                "_DummyHit",
                (),
                {"target_note_id": "c", "token": "L3", "hand": "L", "lane": 3, "note_timestamp_ms": 0.0, "event_timestamp_ms": 0.0, "timing_error_ms": 0.0},
            )(),
        )
    )

    state = tracker.state
    if state.total_hits != 3:
        raise RuntimeError(
            "Gameplay score integration check failed: expected total_hits=3. "
            f"actual={state.total_hits}"
        )
    if state.total_misses != 1:
        raise RuntimeError(
            "Gameplay score integration check failed: expected total_misses=1. "
            f"actual={state.total_misses}"
        )
    if state.combo != 1 or state.max_combo != 2:
        raise RuntimeError(
            "Gameplay score integration check failed: expected combo=1 and max_combo=2 after miss reset. "
            f"combo={state.combo} max_combo={state.max_combo}"
        )
    if state.score != 285:
        raise RuntimeError(
            "Gameplay score integration check failed: expected score=285 from hit/miss/combo constants. "
            f"actual={state.score}"
        )
    if abs(state.percentage_hit - 75.0) > 1e-6:
        raise RuntimeError(
            "Gameplay score integration check failed: expected percentage_hit=75.0. "
            f"actual={state.percentage_hit}"
        )


def verify_judgment_to_liveview_flash_behavior() -> None:
    """Verify judged hit/miss outcomes drive the corresponding liveview circle flashes."""

    target_notes = (
        NormalizedTargetNote(
            target_note_id="tn_1",
            token="L1",
            lane=1,
            hand="L",
            timestamp_seconds=1.0,
            timestamp_ms=1000.0,
            duration_seconds=0.2,
            duration_ms=200.0,
            source_note_number=60,
            source_channel=0,
            source_track_index=0,
            source_velocity=100,
            source_start_tick=480,
            source_end_tick=576,
        ),
        NormalizedTargetNote(
            target_note_id="tn_2",
            token="R2",
            lane=2,
            hand="R",
            timestamp_seconds=1.2,
            timestamp_ms=1200.0,
            duration_seconds=0.2,
            duration_ms=200.0,
            source_note_number=66,
            source_channel=0,
            source_track_index=0,
            source_velocity=100,
            source_start_tick=576,
            source_end_tick=672,
        ),
    )
    circles = (
        CircleGeometry(lane=1, hand="L", center_xy=(100.0, 100.0), radius=20, offset_xy=(0.0, 0.0)),
        CircleGeometry(lane=1, hand="R", center_xy=(100.0, 100.0), radius=20, offset_xy=(0.0, 0.0)),
        CircleGeometry(lane=2, hand="L", center_xy=(200.0, 100.0), radius=20, offset_xy=(0.0, 0.0)),
        CircleGeometry(lane=2, hand="R", center_xy=(200.0, 100.0), radius=20, offset_xy=(0.0, 0.0)),
    )
    left_wrist = KeypointSample(
        name="left_wrist",
        xy=(100.0, 100.0),
        confidence=1.0,
        source="detected",
        observed_at_monotonic=101.0,
    )
    right_wrist = KeypointSample(
        name="right_wrist",
        xy=(200.0, 100.0),
        confidence=1.0,
        source="detected",
        observed_at_monotonic=101.0,
    )

    transition_tracker = HandCircleTransitionTracker(debounce_ms=0)
    judge = HitWindowJudge(normalized_target_notes=target_notes, hit_window_ms=120.0)
    circle_visual_tracker = CircleVisualStateTracker(hit_flash_duration_ms=200, miss_flash_duration_ms=200)

    left_enter_keypoints = GameplayKeypoints(
        person_index=0,
        captured_at_monotonic=102.0,
        head_center_xy=(150.0, 80.0),
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
    left_enter_snapshot = detect_hand_circle_interactions(
        circle_geometries=circles,
        gameplay_keypoints=left_enter_keypoints,
    )
    left_enter_transition = transition_tracker.update(left_enter_snapshot)
    judged_hits = judge.register_transition_snapshot(
        transition_snapshot=left_enter_transition,
        song_started_monotonic=100.0,
        pre_song_lead_in_ms=1000.0,
    )
    hit_tokens = tuple(
        sorted({str(hit.token).strip().upper() for hit in judged_hits if str(hit.token).strip()})
    )
    hit_states = circle_visual_tracker.update(
        circle_geometries=circles,
        gameplay_keypoints=left_enter_keypoints,
        interaction_snapshot=left_enter_snapshot,
        hit_tokens=hit_tokens,
        miss_tokens=(),
        now_monotonic=102.0,
    )
    lane1_left_state = next((state for state in hit_states if state.hand == "L" and state.lane == 1), None)
    lane1_right_state = next((state for state in hit_states if state.hand == "R" and state.lane == 1), None)
    if lane1_left_state is None or lane1_left_state.state_name != "hit_flash":
        raise RuntimeError(
            "Judgment-to-liveview integration check failed: expected L1 to flash hit on judged hit. "
            f"states={hit_states}"
        )
    if lane1_right_state is None or lane1_right_state.state_name == "hit_flash":
        raise RuntimeError(
            "Judgment-to-liveview integration check failed: R1 must not flash when only L1 is judged hit. "
            f"states={hit_states}"
        )

    judge.mark_misses_for_song_elapsed_ms(1400.0)
    missed_note_ids = set(judge.consume_newly_missed_note_ids())
    miss_tokens = tuple(
        sorted(
            {
                str(note.token).strip().upper()
                for note in target_notes
                if note.target_note_id in missed_note_ids
                and str(note.token).strip()
            }
        )
    )
    no_contact_keypoints = GameplayKeypoints(
        person_index=0,
        captured_at_monotonic=102.3,
        head_center_xy=(150.0, 80.0),
        left_eye=None,
        right_eye=None,
        left_wrist=left_wrist,
        right_wrist=right_wrist,
        fingertip_samples=(),
    )
    no_contact_snapshot = detect_hand_circle_interactions(
        circle_geometries=circles,
        gameplay_keypoints=no_contact_keypoints,
    )
    miss_states = circle_visual_tracker.update(
        circle_geometries=circles,
        gameplay_keypoints=no_contact_keypoints,
        interaction_snapshot=no_contact_snapshot,
        hit_tokens=(),
        miss_tokens=miss_tokens,
        now_monotonic=102.3,
    )
    lane2_right_state = next((state for state in miss_states if state.hand == "R" and state.lane == 2), None)
    lane2_left_state = next((state for state in miss_states if state.hand == "L" and state.lane == 2), None)
    if lane2_right_state is None or lane2_right_state.state_name != "miss_flash":
        raise RuntimeError(
            "Judgment-to-liveview integration check failed: expected R2 to flash miss on judged miss. "
            f"states={miss_states}"
        )
    if lane2_left_state is None or lane2_left_state.state_name == "miss_flash":
        raise RuntimeError(
            "Judgment-to-liveview integration check failed: L2 must not flash when only R2 is judged miss. "
            f"states={miss_states}"
        )


def verify_song_audio_start_timing_alignment() -> None:
    """Verify scheduled song audio start aligns with the configured gameplay lead-in clock."""

    import midi_mover.runtime_loop as runtime_loop

    original_render = runtime_loop.render_liveview_frame
    render_calls = {"count": 0}

    def _fake_render_liveview_frame(**_: object) -> None:
        render_calls["count"] += 1

    runtime_loop.render_liveview_frame = _fake_render_liveview_frame
    try:
        class _PygameStub:
            QUIT = 12
            KEYDOWN = 2
            K_ESCAPE = 27
            _event_calls = 0

            @dataclass
            class _LoopEvent:
                type: int = 0
                key: int = 0

            class _Event:
                @staticmethod
                def get() -> list["_PygameStub._LoopEvent"]:
                    _PygameStub._event_calls += 1
                    if _PygameStub._event_calls >= 2:
                        return [
                            _PygameStub._LoopEvent(
                                type=_PygameStub.KEYDOWN,
                                key=_PygameStub.K_ESCAPE,
                            )
                        ]
                    return [_PygameStub._LoopEvent(type=0)]

            class _Clock:
                def tick(self, _: int) -> None:
                    return None

            class _Time:
                @staticmethod
                def Clock() -> "_PygameStub._Clock":
                    return _PygameStub._Clock()

            event = _Event()
            time = _Time()

        class _ConfigStub:
            raw = {"app": {"target_fps": 30}, "gameplay": {"note_history_ms": 1500}}

        triggered = {"count": 0}
        run_persistent_liveview_loop(
            window=object(),
            pygame_module=_PygameStub(),
            frame_reader=object(),
            pose_model=object(),
            hand_pose_model=object(),
            primary_person_tracker=object(),
            gameplay_keypoint_tracker=object(),
            interaction_transition_tracker=object(),
            crop_smoother=object(),
            circle_visual_tracker=object(),
            config=_ConfigStub(),
            song_audio_start=lambda: triggered.__setitem__("count", triggered["count"] + 1),
            song_audio_start_at_monotonic=0.0,
        )
        if triggered["count"] != 1 or render_calls["count"] != 1:
            raise RuntimeError(
                "Song-audio start timing integration check failed: expected one scheduled audio start and one render pass. "
                f"audio_starts={triggered['count']} render_calls={render_calls['count']}"
            )
    finally:
        runtime_loop.render_liveview_frame = original_render




