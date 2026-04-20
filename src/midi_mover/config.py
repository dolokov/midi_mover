"""YAML config loading and validation for midi_mover."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from midi_mover.config_validation_helpers import (
    validate_and_normalize_circle_offsets_by_count as _validate_circle_offsets_by_count_helper,
    validate_fluidsynth_immediate_cue_token_notes,
)
from midi_mover.gesture_tokens import build_gesture_tokens, normalize_circles_per_hand
from midi_mover.simple_yaml import load_simple_yaml
try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without PyYAML
    yaml = None
class ConfigError(ValueError):
    """Raised when the YAML config is missing required data."""
@dataclass(frozen=True)
class AppConfig:
    raw: dict[str, Any]
REQUIRED_PATHS: tuple[tuple[str, ...], ...] = (
    ("app", "name"),
    ("app", "window_width"),
    ("app", "window_height"),
    ("app", "window_scale_factor"),
    ("app", "target_fps"),
    ("app", "song_title_screen_duration_seconds"),
    ("app", "song_summary_screen_duration_seconds"),
    ("app", "debug"),
    ("app", "logging_level"),
    ("app", "random_seed"),
    ("camera", "width"),
    ("camera", "height"),
    ("camera", "mirror"),
    ("pose", "stage1_model_name"),
    ("pose", "stage2_hand_model_name"),
    ("pose", "stage2_roi_expansion_px"),
    ("pose", "confidence_threshold"),
    ("pose", "iou_threshold"),
    ("pose", "stage2_confidence_threshold"),
    ("pose", "stage2_iou_threshold"),
    ("pose", "stage2_missing_fallback_mode"),
    ("pose", "stage2_missing_fallback_timeout_seconds"),
    ("pose", "device"),
    ("pose", "smoothing_factor"),
    ("pose", "lost_person_timeout_seconds"),
    ("liveview", "left_panel_ratio"),
    ("liveview", "crop_smoothing_factor"),
    ("liveview", "crop_max_jump_ratio"),
    ("liveview", "eye_center_x_pct"),
    ("liveview", "eye_center_y_pct"),
    ("liveview", "crop_width_eye_dist"),
    ("liveview", "circle_visuals", "idle", "outline_color"),
    ("liveview", "circle_visuals", "idle", "fill_color"),
    ("liveview", "circle_visuals", "idle", "label_color"),
    ("liveview", "circle_visuals", "active_contact", "outline_color"),
    ("liveview", "circle_visuals", "active_contact", "fill_color"),
    ("liveview", "circle_visuals", "active_contact", "label_color"),
    ("liveview", "circle_visuals", "hit_flash", "outline_color"),
    ("liveview", "circle_visuals", "hit_flash", "fill_color"),
    ("liveview", "circle_visuals", "hit_flash", "label_color"),
    ("liveview", "circle_visuals", "miss_flash", "outline_color"),
    ("liveview", "circle_visuals", "miss_flash", "fill_color"),
    ("liveview", "circle_visuals", "miss_flash", "label_color"),
    ("liveview", "circle_visuals", "hit_flash_duration_ms"),
    ("liveview", "circle_visuals", "miss_flash_duration_ms"),
    ("liveview", "debug", "show_head_center"),
    ("liveview", "debug", "show_wrist_markers"),
    ("liveview", "debug", "show_stage1_full_keypoints"),
    ("liveview", "debug", "show_stage2_hand_keypoints"),
    ("liveview", "debug", "show_keypoint_overlay_legend"),
    ("liveview", "debug", "show_overlay_confidence_values"),
    ("liveview", "debug", "keypoint_overlay_legend_style", "font_size"),
    ("liveview", "debug", "keypoint_overlay_legend_style", "text_color"),
    ("liveview", "debug", "keypoint_overlay_legend_style", "muted_text_color"),
    ("liveview", "debug", "keypoint_overlay_legend_style", "background_color"),
    ("liveview", "debug", "keypoint_overlay_legend_style", "border_color"),
    ("liveview", "debug", "keypoint_overlay_legend_style", "border_width"),
    ("liveview", "debug", "keypoint_overlay_legend_style", "panel_padding"),
    ("liveview", "debug", "keypoint_overlay_legend_style", "line_spacing"),
    ("liveview", "debug", "stage1_full_keypoints_style", "keypoint_color"),
    ("liveview", "debug", "stage1_full_keypoints_style", "keypoint_radius"),
    ("liveview", "debug", "stage1_full_keypoints_style", "keypoint_outline_color"),
    ("liveview", "debug", "stage1_full_keypoints_style", "keypoint_outline_width"),
    ("liveview", "debug", "stage1_full_keypoints_style", "skeleton_color"),
    ("liveview", "debug", "stage1_full_keypoints_style", "skeleton_width"),
    ("liveview", "debug", "stage2_hand_keypoints_style", "keypoint_color"),
    ("liveview", "debug", "stage2_hand_keypoints_style", "keypoint_radius"),
    ("liveview", "debug", "stage2_hand_keypoints_style", "keypoint_outline_color"),
    ("liveview", "debug", "stage2_hand_keypoints_style", "keypoint_outline_width"),
    ("liveview", "debug", "stage2_hand_keypoints_style", "skeleton_color"),
    ("liveview", "debug", "stage2_hand_keypoints_style", "skeleton_width"),
    ("liveview", "debug", "stage2_hand_keypoints_style", "emphasize_fingertips"),
    ("liveview", "debug", "stage2_hand_keypoints_style", "fingertip_color"),
    ("liveview", "debug", "stage2_hand_keypoints_style", "fingertip_radius"),
    ("liveview", "debug", "stage2_hand_keypoints_style", "fingertip_outline_color"),
    ("liveview", "debug", "stage2_hand_keypoints_style", "fingertip_outline_width"),
    ("liveview", "circles_per_hand"),
    ("liveview", "circle_radius_percent"),
    ("liveview", "left_circle_offsets_percent_by_count"),
    ("liveview", "right_circle_offsets_percent_by_count"),
    ("liveview", "circle_stroke_width"),
    ("liveview", "label_font_size"),
    ("liveview", "wrist_marker_radius"),
    ("liveview", "wrist_marker_outline_width"),
    ("liveview", "padding_color"),
    ("gameplay", "hit_window_ms"),
    ("gameplay", "song_speed_multiplier"),
    ("gameplay", "pre_song_lead_in_ms"),
    ("gameplay", "early_late_tolerance_ms"),
    ("gameplay", "debounce_ms"),
    ("gameplay", "swap_hands_when_mirrored"),
    ("gameplay", "note_history_ms"),
    ("gameplay", "lookahead_ms"),
    ("gameplay", "timeline_now_line_ratio"),
    ("gameplay", "score_values", "hit"),
    ("gameplay", "score_values", "miss"),
    ("gameplay", "score_values", "combo_bonus"),
    ("midi", "supported_extensions"),
    ("midi", "note_mapping"),
    ("midi", "track_filter"),
    ("midi", "channel_filter"),
    ("midi", "minimum_note_duration_ms"),
    ("midi", "ignore_meta_events"),
    ("audio", "mixer", "frequency"),
    ("audio", "mixer", "size"),
    ("audio", "mixer", "channels"),
    ("audio", "mixer", "buffer"),
    ("audio", "mixer", "max_channels"),
    ("audio", "backend"),
    ("audio", "fluidsynth", "soundfont_path"),
    ("audio", "fluidsynth", "sample_rate"),
    ("audio", "fluidsynth", "gain"),
    ("audio", "fluidsynth", "polyphony"),
    ("audio", "fluidsynth", "audio_driver"),
    ("audio", "fluidsynth", "audio_buffer_size"),
    ("audio", "volumes", "master"),
    ("audio", "volumes", "song"),
    ("audio", "volumes", "ui"),
    ("audio", "volumes", "hit"),
    ("audio", "immediate_cues", "fluidsynth", "enabled"),
    ("audio", "immediate_cues", "fluidsynth", "channel"),
    ("audio", "immediate_cues", "fluidsynth", "velocity"),
    ("audio", "immediate_cues", "fluidsynth", "sustain_while_inside"),
    ("audio", "immediate_cues", "fluidsynth", "token_notes"),
    ("highscore", "list_size"),
    ("highscore", "headshot_countdown_seconds"), ("highscore", "leaderboard_screen_duration_seconds"), ("highscore", "headshot_crop_margin"),
    ("highscore", "thumbnail_width"),
    ("highscore", "thumbnail_height"),
)
def load_config(config_path: Path) -> AppConfig:
    """Load and validate the application YAML config."""
    raw_text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        try:
            payload = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Failed to parse YAML config at {config_path}: {exc}"
            ) from exc
    else:
        try:
            payload = load_simple_yaml(raw_text, config_path)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    if payload is None:
        raise ConfigError(
            f"Config file {config_path} is empty. Add the required startup keys."
        )
    if not isinstance(payload, dict):
        raise ConfigError(
            f"Config file {config_path} must contain a top-level mapping/object."
        )
    _validate_required_paths(payload)
    _validate_value_types(payload)
    return AppConfig(raw=payload)
def _validate_required_paths(payload: dict[str, Any]) -> None:
    missing = []
    for path in REQUIRED_PATHS:
        if not _has_path(payload, path):
            missing.append(".".join(path))
    if missing:
        joined = "\n - ".join([""] + missing)
        raise ConfigError(
            "Config is missing required keys:" + joined + "\nAdd them to the YAML file and retry startup."
        )
def _has_path(payload: dict[str, Any], path: tuple[str, ...]) -> bool:
    current: Any = payload
    for segment in path:
        if not isinstance(current, dict) or segment not in current:
            return False
        current = current[segment]
    return True
def _validate_value_types(payload: dict[str, Any]) -> None:
    _require_type(payload["app"]["window_width"], int, "app.window_width")
    _require_type(payload["app"]["window_height"], int, "app.window_height")
    _require_numeric(payload["app"]["window_scale_factor"], "app.window_scale_factor")
    if float(payload["app"]["window_scale_factor"]) <= 0.0:
        raise ConfigError("Config key app.window_scale_factor must be > 0.")
    _require_type(payload["app"]["target_fps"], int, "app.target_fps")
    _require_numeric(
        payload["app"]["song_title_screen_duration_seconds"],
        "app.song_title_screen_duration_seconds",
    )
    _require_numeric(
        payload["app"]["song_summary_screen_duration_seconds"],
        "app.song_summary_screen_duration_seconds",
    )
    _require_type(payload["app"]["debug"], bool, "app.debug")
    _require_type(payload["app"]["logging_level"], str, "app.logging_level")
    _require_optional_type(payload["app"]["random_seed"], int, "app.random_seed")
    _require_type(payload["camera"]["width"], int, "camera.width")
    _require_type(payload["camera"]["height"], int, "camera.height")
    _require_type(payload["camera"]["mirror"], bool, "camera.mirror")
    _require_non_empty_string(
        payload["pose"]["stage1_model_name"],
        "pose.stage1_model_name",
    )
    _require_non_empty_string(
        payload["pose"]["stage2_hand_model_name"],
        "pose.stage2_hand_model_name",
    )
    _require_type(payload["pose"]["stage2_roi_expansion_px"], int, "pose.stage2_roi_expansion_px")
    _require_numeric(payload["pose"]["confidence_threshold"], "pose.confidence_threshold")
    _require_numeric(payload["pose"]["iou_threshold"], "pose.iou_threshold")
    _require_numeric(
        payload["pose"]["stage2_confidence_threshold"],
        "pose.stage2_confidence_threshold",
    )
    _require_numeric(payload["pose"]["stage2_iou_threshold"], "pose.stage2_iou_threshold")
    _require_non_empty_string(
        payload["pose"]["stage2_missing_fallback_mode"],
        "pose.stage2_missing_fallback_mode",
    )
    _require_numeric(
        payload["pose"]["stage2_missing_fallback_timeout_seconds"],
        "pose.stage2_missing_fallback_timeout_seconds",
    )
    _require_type(payload["pose"]["device"], str, "pose.device")
    _require_numeric(payload["pose"]["smoothing_factor"], "pose.smoothing_factor")
    _require_numeric(
        payload["pose"]["lost_person_timeout_seconds"],
        "pose.lost_person_timeout_seconds",
    )
    fallback_mode = str(payload["pose"]["stage2_missing_fallback_mode"]).strip().lower()
    if fallback_mode not in {"clear", "reuse_last"}:
        raise ConfigError(
            "Config key pose.stage2_missing_fallback_mode must be either 'clear' or 'reuse_last'."
        )
    payload["pose"]["stage2_missing_fallback_mode"] = fallback_mode
    _require_numeric(payload["liveview"]["left_panel_ratio"], "liveview.left_panel_ratio")
    _require_numeric(payload["liveview"]["crop_smoothing_factor"], "liveview.crop_smoothing_factor")
    _require_numeric(payload["liveview"]["crop_max_jump_ratio"], "liveview.crop_max_jump_ratio")
    _require_numeric(payload["liveview"]["eye_center_x_pct"], "liveview.eye_center_x_pct")
    _require_numeric(payload["liveview"]["eye_center_y_pct"], "liveview.eye_center_y_pct")
    _require_numeric(payload["liveview"]["crop_width_eye_dist"], "liveview.crop_width_eye_dist")
    _require_type(payload["liveview"]["circle_visuals"], dict, "liveview.circle_visuals")
    _require_type(payload["liveview"]["debug"], dict, "liveview.debug")
    _require_type(payload["liveview"]["debug"]["show_head_center"], bool, "liveview.debug.show_head_center")
    _require_type(payload["liveview"]["debug"]["show_wrist_markers"], bool, "liveview.debug.show_wrist_markers")
    _require_type(
        payload["liveview"]["debug"]["show_stage1_full_keypoints"],
        bool,
        "liveview.debug.show_stage1_full_keypoints",
    )
    _require_type(
        payload["liveview"]["debug"]["show_stage2_hand_keypoints"],
        bool,
        "liveview.debug.show_stage2_hand_keypoints",
    )
    _require_type(
        payload["liveview"]["debug"]["show_keypoint_overlay_legend"],
        bool,
        "liveview.debug.show_keypoint_overlay_legend",
    )
    _require_type(
        payload["liveview"]["debug"]["show_overlay_confidence_values"],
        bool,
        "liveview.debug.show_overlay_confidence_values",
    )
    _validate_overlay_legend_style(
        payload["liveview"]["debug"]["keypoint_overlay_legend_style"],
        "liveview.debug.keypoint_overlay_legend_style",
    )
    _validate_keypoint_style(
        payload["liveview"]["debug"]["stage1_full_keypoints_style"],
        "liveview.debug.stage1_full_keypoints_style",
    )
    _validate_keypoint_style(
        payload["liveview"]["debug"]["stage2_hand_keypoints_style"],
        "liveview.debug.stage2_hand_keypoints_style",
        include_fingertips=True,
    )
    _require_numeric(
        payload["liveview"]["circle_radius_percent"],
        "liveview.circle_radius_percent",
    )
    _require_type(payload["liveview"]["circles_per_hand"], int, "liveview.circles_per_hand")
    try:
        circles_per_hand = normalize_circles_per_hand(payload["liveview"]["circles_per_hand"])
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    payload["liveview"]["circles_per_hand"] = circles_per_hand
    _require_type(
        payload["liveview"]["left_circle_offsets_percent_by_count"],
        dict,
        "liveview.left_circle_offsets_percent_by_count",
    )
    _require_type(
        payload["liveview"]["right_circle_offsets_percent_by_count"],
        dict,
        "liveview.right_circle_offsets_percent_by_count",
    )
    _require_type(payload["liveview"]["circle_stroke_width"], int, "liveview.circle_stroke_width")
    _require_type(payload["liveview"]["label_font_size"], int, "liveview.label_font_size")
    _require_type(payload["liveview"]["wrist_marker_radius"], int, "liveview.wrist_marker_radius")
    _require_type(
        payload["liveview"]["wrist_marker_outline_width"],
        int,
        "liveview.wrist_marker_outline_width",
    )
    _require_color_triplet(payload["liveview"]["padding_color"], "liveview.padding_color")
    _validate_circle_visuals(payload["liveview"]["circle_visuals"])
    _require_type(payload["gameplay"]["hit_window_ms"], int, "gameplay.hit_window_ms")
    _require_numeric(payload["gameplay"]["song_speed_multiplier"], "gameplay.song_speed_multiplier")
    if float(payload["gameplay"]["song_speed_multiplier"]) <= 0.0:
        raise ConfigError("Config key gameplay.song_speed_multiplier must be > 0.")
    _require_type(payload["gameplay"]["pre_song_lead_in_ms"], int, "gameplay.pre_song_lead_in_ms")
    _require_type(
        payload["gameplay"]["early_late_tolerance_ms"],
        int,
        "gameplay.early_late_tolerance_ms",
    )
    _require_type(payload["gameplay"]["debounce_ms"], int, "gameplay.debounce_ms")
    _require_type(
        payload["gameplay"]["swap_hands_when_mirrored"],
        bool,
        "gameplay.swap_hands_when_mirrored",
    )
    _require_type(payload["gameplay"]["note_history_ms"], int, "gameplay.note_history_ms")
    _require_type(payload["gameplay"]["lookahead_ms"], int, "gameplay.lookahead_ms")
    _require_numeric(payload["gameplay"]["timeline_now_line_ratio"], "gameplay.timeline_now_line_ratio")
    _require_type(payload["gameplay"]["score_values"], dict, "gameplay.score_values")
    _require_type(payload["midi"]["supported_extensions"], list, "midi.supported_extensions")
    _require_type(payload["midi"]["note_mapping"], dict, "midi.note_mapping")
    _require_type(payload["midi"]["track_filter"], list, "midi.track_filter")
    _require_type(payload["midi"]["channel_filter"], list, "midi.channel_filter")
    _require_type(
        payload["midi"]["minimum_note_duration_ms"],
        int,
        "midi.minimum_note_duration_ms",
    )
    _require_type(payload["midi"]["ignore_meta_events"], bool, "midi.ignore_meta_events")
    _require_type(payload["audio"]["mixer"], dict, "audio.mixer")
    _require_non_empty_string(payload["audio"]["backend"], "audio.backend")
    _require_type(payload["audio"]["fluidsynth"], dict, "audio.fluidsynth")
    _require_type(
        payload["audio"]["fluidsynth"]["soundfont_path"],
        str,
        "audio.fluidsynth.soundfont_path",
    )
    _require_type(
        payload["audio"]["fluidsynth"]["sample_rate"],
        int,
        "audio.fluidsynth.sample_rate",
    )
    _require_numeric(payload["audio"]["fluidsynth"]["gain"], "audio.fluidsynth.gain")
    _require_type(
        payload["audio"]["fluidsynth"]["polyphony"],
        int,
        "audio.fluidsynth.polyphony",
    )
    _require_optional_type(
        payload["audio"]["fluidsynth"]["audio_driver"],
        str,
        "audio.fluidsynth.audio_driver",
    )
    _require_type(
        payload["audio"]["fluidsynth"]["audio_buffer_size"],
        int,
        "audio.fluidsynth.audio_buffer_size",
    )
    _require_optional_type(
        payload["audio"]["fluidsynth"].get("instrument_profile"),
        str,
        "audio.fluidsynth.instrument_profile",
    )
    _require_type(
        payload["audio"]["fluidsynth"].get("instrument_profiles", {}),
        dict,
        "audio.fluidsynth.instrument_profiles",
    )
    _require_type(
        payload["audio"]["fluidsynth"].get("channel_instruments", {}),
        dict,
        "audio.fluidsynth.channel_instruments",
    )
    _require_type(payload["audio"]["volumes"], dict, "audio.volumes")
    _require_numeric(payload["audio"]["volumes"]["master"], "audio.volumes.master")
    _require_numeric(payload["audio"]["volumes"]["song"], "audio.volumes.song")
    _require_numeric(payload["audio"]["volumes"]["ui"], "audio.volumes.ui")
    _require_numeric(payload["audio"]["volumes"]["hit"], "audio.volumes.hit")
    _require_type(payload["audio"]["mixer"]["max_channels"], int, "audio.mixer.max_channels")
    _require_type(payload["audio"]["immediate_cues"], dict, "audio.immediate_cues")
    _require_type(payload["audio"]["immediate_cues"]["fluidsynth"], dict, "audio.immediate_cues.fluidsynth")
    _require_type(
        payload["audio"]["immediate_cues"]["fluidsynth"]["enabled"],
        bool,
        "audio.immediate_cues.fluidsynth.enabled",
    )
    _require_type(
        payload["audio"]["immediate_cues"]["fluidsynth"]["channel"],
        int,
        "audio.immediate_cues.fluidsynth.channel",
    )
    _require_type(
        payload["audio"]["immediate_cues"]["fluidsynth"]["velocity"],
        int,
        "audio.immediate_cues.fluidsynth.velocity",
    )
    _require_type(
        payload["audio"]["immediate_cues"]["fluidsynth"]["sustain_while_inside"],
        bool,
        "audio.immediate_cues.fluidsynth.sustain_while_inside",
    )
    _require_type(
        payload["audio"]["immediate_cues"]["fluidsynth"]["token_notes"],
        dict,
        "audio.immediate_cues.fluidsynth.token_notes",
    )
    _validate_fluidsynth_immediate_cue_token_notes(
        payload["audio"]["immediate_cues"]["fluidsynth"]["token_notes"],
        circles_per_hand=circles_per_hand,
    )
    _require_type(payload["highscore"]["list_size"], int, "highscore.list_size")
    _require_numeric(payload["highscore"]["headshot_countdown_seconds"], "highscore.headshot_countdown_seconds")
    _require_numeric(payload["highscore"]["leaderboard_screen_duration_seconds"], "highscore.leaderboard_screen_duration_seconds")
    _require_numeric(payload["highscore"]["headshot_crop_margin"], "highscore.headshot_crop_margin")
    _require_type(payload["highscore"]["thumbnail_width"], int, "highscore.thumbnail_width")
    _require_type(payload["highscore"]["thumbnail_height"], int, "highscore.thumbnail_height")
    _validate_and_normalize_circle_offsets_by_count(
        payload["liveview"],
        "left_circle_offsets_percent_by_count",
    )
    _validate_and_normalize_circle_offsets_by_count(
        payload["liveview"],
        "right_circle_offsets_percent_by_count",
    )
def _validate_and_normalize_circle_offsets_by_count(liveview_payload: dict[str, Any], key: str) -> None:
    """Normalize and validate circle offsets grouped by circle count ('3', '4', '5')."""
    _validate_circle_offsets_by_count_helper(
        liveview_payload=liveview_payload,
        key=key,
        normalize_mapping_key=_normalize_mapping_key,
        require_vector2=_require_vector2,
        error_type=ConfigError,
    )
def _require_type(value: Any, expected_type: type, name: str) -> None:
    if not isinstance(value, expected_type):
        raise ConfigError(
            f"Config key {name} must be of type {expected_type.__name__}, got {type(value).__name__}."
        )
def _require_non_empty_string(value: Any, name: str) -> None:
    _require_type(value, str, name)
    if not value.strip():
        raise ConfigError(f"Config key {name} must be a non-empty string.")
def _require_optional_type(value: Any, expected_type: type, name: str) -> None:
    if value is not None and not isinstance(value, expected_type):
        raise ConfigError(
            f"Config key {name} must be null or {expected_type.__name__}, got {type(value).__name__}."
        )
def _require_numeric(value: Any, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"Config key {name} must be numeric, got {type(value).__name__}.")
def _require_vector2(value: Any, name: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"Config key {name} must be a 2-item list [x, y].")
    for index, item in enumerate(value):
        _require_numeric(item, f"{name}[{index}]")
def _require_color_triplet(value: Any, name: str) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise ConfigError(f"Config key {name} must be an RGB triplet like [0, 0, 0].")
    for index, item in enumerate(value):
        _require_type(item, int, f"{name}[{index}]")
        if not 0 <= item <= 255:
            raise ConfigError(f"Config key {name}[{index}] must be between 0 and 255.")
def _validate_keypoint_style(
    style_payload: Any,
    path: str,
    *,
    include_fingertips: bool = False,
) -> None:
    _require_type(style_payload, dict, path)
    _require_color_triplet(style_payload["keypoint_color"], f"{path}.keypoint_color")
    _require_type(style_payload["keypoint_radius"], int, f"{path}.keypoint_radius")
    _require_color_triplet(style_payload["keypoint_outline_color"], f"{path}.keypoint_outline_color")
    _require_type(style_payload["keypoint_outline_width"], int, f"{path}.keypoint_outline_width")
    _require_color_triplet(style_payload["skeleton_color"], f"{path}.skeleton_color")
    _require_type(style_payload["skeleton_width"], int, f"{path}.skeleton_width")
    if not include_fingertips:
        return
    _require_type(style_payload["emphasize_fingertips"], bool, f"{path}.emphasize_fingertips")
    _require_color_triplet(style_payload["fingertip_color"], f"{path}.fingertip_color")
    _require_type(style_payload["fingertip_radius"], int, f"{path}.fingertip_radius")
    _require_color_triplet(
        style_payload["fingertip_outline_color"],
        f"{path}.fingertip_outline_color",
    )
    _require_type(style_payload["fingertip_outline_width"], int, f"{path}.fingertip_outline_width")
def _validate_overlay_legend_style(style_payload: Any, path: str) -> None:
    _require_type(style_payload, dict, path)
    _require_type(style_payload["font_size"], int, f"{path}.font_size")
    _require_color_triplet(style_payload["text_color"], f"{path}.text_color")
    _require_color_triplet(style_payload["muted_text_color"], f"{path}.muted_text_color")
    _require_color_triplet(style_payload["background_color"], f"{path}.background_color")
    _require_color_triplet(style_payload["border_color"], f"{path}.border_color")
    _require_type(style_payload["border_width"], int, f"{path}.border_width")
    _require_type(style_payload["panel_padding"], int, f"{path}.panel_padding")
    _require_type(style_payload["line_spacing"], int, f"{path}.line_spacing")
def _validate_circle_visuals(circle_visuals: dict[str, Any]) -> None:
    state_names = ("idle", "active_contact", "hit_flash", "miss_flash")
    for state_name in state_names:
        if state_name not in circle_visuals:
            raise ConfigError(
                f"Config key liveview.circle_visuals must define state '{state_name}'."
            )
        state_payload = circle_visuals[state_name]
        _require_type(state_payload, dict, f"liveview.circle_visuals.{state_name}")
        _require_color_triplet(
            state_payload["outline_color"],
            f"liveview.circle_visuals.{state_name}.outline_color",
        )
        _require_color_triplet(
            state_payload["fill_color"],
            f"liveview.circle_visuals.{state_name}.fill_color",
        )
        _require_color_triplet(
            state_payload["label_color"],
            f"liveview.circle_visuals.{state_name}.label_color",
        )
    _require_type(
        circle_visuals["hit_flash_duration_ms"],
        int,
        "liveview.circle_visuals.hit_flash_duration_ms",
    )
    _require_type(
        circle_visuals["miss_flash_duration_ms"],
        int,
        "liveview.circle_visuals.miss_flash_duration_ms",
    )
    # Optional continuous precue styling controls.
    if "precue_gradient_start_color" in circle_visuals:
        _require_color_triplet(
            circle_visuals["precue_gradient_start_color"],
            "liveview.circle_visuals.precue_gradient_start_color",
        )
    if "precue_gradient_end_color" in circle_visuals:
        _require_color_triplet(
            circle_visuals["precue_gradient_end_color"],
            "liveview.circle_visuals.precue_gradient_end_color",
        )
    if "precue_min_stroke_width" in circle_visuals:
        _require_type(
            circle_visuals["precue_min_stroke_width"],
            int,
            "liveview.circle_visuals.precue_min_stroke_width",
        )
    if "precue_max_stroke_width" in circle_visuals:
        _require_type(
            circle_visuals["precue_max_stroke_width"],
            int,
            "liveview.circle_visuals.precue_max_stroke_width",
        )
    if "precue_min_fill_alpha" in circle_visuals:
        _require_numeric(
            circle_visuals["precue_min_fill_alpha"],
            "liveview.circle_visuals.precue_min_fill_alpha",
        )
    if "precue_max_fill_alpha" in circle_visuals:
        _require_numeric(
            circle_visuals["precue_max_fill_alpha"],
            "liveview.circle_visuals.precue_max_fill_alpha",
        )
    if "precue_primary_outline_color" in circle_visuals:
        _require_color_triplet(
            circle_visuals["precue_primary_outline_color"],
            "liveview.circle_visuals.precue_primary_outline_color",
        )
    if "precue_primary_outline_extra_width" in circle_visuals:
        _require_type(
            circle_visuals["precue_primary_outline_extra_width"],
            int,
            "liveview.circle_visuals.precue_primary_outline_extra_width",
        )
def _validate_fluidsynth_immediate_cue_token_notes(
    token_notes: dict[str, Any],
    *,
    circles_per_hand: int,
) -> None:
    validate_fluidsynth_immediate_cue_token_notes(
        token_notes=token_notes,
        required_tokens=build_gesture_tokens(circles_per_hand),
        normalize_mapping_key=_normalize_mapping_key,
        require_type=_require_type,
        error_type=ConfigError,
    )
def _normalize_mapping_key(key: Any) -> str:
    text = str(key)
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    return text
