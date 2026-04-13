"""Helpers for scheduling MIDI event playback through a FluidSynth synth instance."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time
from typing import Any


LOGGER = logging.getLogger("midi_mover")


class FluidSynthPlaybackSchedulerError(RuntimeError):
    """Raised when scheduled FluidSynth playback cannot be created or run."""


@dataclass
class ScheduledFluidSynthPlayback:
    """Background MIDI playback scheduler for a FluidSynth synth instance."""

    synth: Any
    worker_thread: threading.Thread
    stop_event: threading.Event

    def stop(self, *, join_timeout_seconds: float = 1.0) -> None:
        """Request playback stop and wait briefly for the worker thread to exit."""

        self.stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=max(0.0, float(join_timeout_seconds)))
        _all_sounds_off(self.synth)


def start_scheduled_fluidsynth_playback(*, synth: Any, midi_path: Path) -> ScheduledFluidSynthPlayback:
    """Start MIDI playback by scheduling parsed MIDI messages on a background thread."""

    try:
        import mido
    except ModuleNotFoundError as exc:
        raise FluidSynthPlaybackSchedulerError(
            "Scheduled FluidSynth playback requires 'mido' in the active environment."
        ) from exc

    if not midi_path.exists() or not midi_path.is_file():
        raise FluidSynthPlaybackSchedulerError(
            f"Cannot schedule FluidSynth MIDI playback: file does not exist: '{midi_path}'."
        )

    try:
        midi_file = mido.MidiFile(str(midi_path))
    except Exception as exc:
        raise FluidSynthPlaybackSchedulerError(
            f"Failed to parse MIDI file for scheduled FluidSynth playback '{midi_path}': {exc}"
        ) from exc

    stop_event = threading.Event()
    worker_thread = threading.Thread(
        target=_run_midi_schedule_worker,
        args=(synth, midi_file, stop_event),
        name="midi_mover_fluidsynth_scheduler",
        daemon=True,
    )
    worker_thread.start()
    return ScheduledFluidSynthPlayback(
        synth=synth,
        worker_thread=worker_thread,
        stop_event=stop_event,
    )


def _run_midi_schedule_worker(synth: Any, midi_file: Any, stop_event: threading.Event) -> None:
    playback_start = time.monotonic()
    elapsed_target_seconds = 0.0
    try:
        for message in midi_file:
            if stop_event.is_set():
                break
            delta_seconds = max(0.0, float(getattr(message, "time", 0.0)))
            elapsed_target_seconds += delta_seconds
            _sleep_until_elapsed_target(
                playback_start=playback_start,
                elapsed_target_seconds=elapsed_target_seconds,
                stop_event=stop_event,
            )
            if stop_event.is_set():
                break
            if bool(getattr(message, "is_meta", False)):
                continue
            _dispatch_message_to_synth(synth=synth, message=message)
    except Exception:
        LOGGER.exception("FluidSynth scheduled playback worker failed unexpectedly.")
    finally:
        _all_sounds_off(synth)


def _sleep_until_elapsed_target(
    *,
    playback_start: float,
    elapsed_target_seconds: float,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        remaining = (playback_start + elapsed_target_seconds) - time.monotonic()
        if remaining <= 0.0:
            return
        stop_event.wait(timeout=min(0.005, remaining))


def _dispatch_message_to_synth(*, synth: Any, message: Any) -> None:
    message_type = str(getattr(message, "type", "")).lower()
    channel = int(getattr(message, "channel", 0))

    if message_type == "note_on":
        note = int(getattr(message, "note", 0))
        velocity = int(getattr(message, "velocity", 0))
        if velocity <= 0:
            _call_if_exists(synth, "noteoff", channel, note)
            return
        _call_if_exists(synth, "noteon", channel, note, velocity)
        return

    if message_type == "note_off":
        note = int(getattr(message, "note", 0))
        _call_if_exists(synth, "noteoff", channel, note)
        return

    if message_type == "control_change":
        control = int(getattr(message, "control", 0))
        value = int(getattr(message, "value", 0))
        _call_if_exists(synth, "cc", channel, control, value)
        return

    if message_type == "program_change":
        program = int(getattr(message, "program", 0))
        _call_if_exists(synth, "program_change", channel, program)
        return

    if message_type == "pitchwheel":
        pitch = int(getattr(message, "pitch", 0))
        _call_if_exists(synth, "pitch_bend", channel, pitch)
        return

    if message_type == "aftertouch":
        pressure = int(getattr(message, "value", 0))
        _call_if_exists(synth, "channel_pressure", channel, pressure)
        return

    if message_type == "polytouch":
        note = int(getattr(message, "note", 0))
        value = int(getattr(message, "value", 0))
        _call_if_exists(synth, "key_pressure", channel, note, value)


def _call_if_exists(target: Any, method_name: str, *args: Any) -> None:
    method = getattr(target, method_name, None)
    if not callable(method):
        return
    method(*args)


def _all_sounds_off(synth: Any) -> None:
    all_sounds_off = getattr(synth, "all_sounds_off", None)
    if callable(all_sounds_off):
        for channel in range(16):
            all_sounds_off(channel)
