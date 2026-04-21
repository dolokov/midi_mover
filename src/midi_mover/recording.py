"""FFmpeg-based gameplay window recording helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import shlex
import signal
import shutil
import subprocess
import tempfile
import time
from typing import Any


class RecordingError(RuntimeError):
    """Raised when gameplay recording cannot be started or finalized."""


@dataclass
class ActiveRecording:
    """Handle for an active ffmpeg recording process."""

    process: subprocess.Popen[bytes]
    output_path: Path
    stderr_log_path: Path | None = None
    audio_process: subprocess.Popen[bytes] | None = None
    use_stdin_quit: bool = True

    def finalize(self, timeout_seconds: float = 8.0) -> Path:
        """Stop ffmpeg and return the saved output path."""

        if self.process.poll() is not None:
            if self.process.returncode != 0:
                raise RecordingError(
                    _build_ffmpeg_failure_message(
                        prefix=f"ffmpeg recording exited unexpectedly with code {self.process.returncode}.",
                        stderr_log_path=self.stderr_log_path,
                    )
                )
            return self.output_path

        try:
            if self.use_stdin_quit and self.process.stdin is not None:
                self.process.stdin.write(b"q\n")
                self.process.stdin.flush()
            else:
                self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=max(0.1, float(timeout_seconds)))
        except Exception:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except Exception:
                self.process.kill()
                self.process.wait(timeout=2.0)
        finally:
            if self.audio_process is not None:
                _stop_subprocess_quietly(self.audio_process)

        if self.process.returncode != 0:
            raise RecordingError(
                _build_ffmpeg_failure_message(
                    prefix=f"ffmpeg recording finalize failed with code {self.process.returncode}.",
                    stderr_log_path=self.stderr_log_path,
                )
            )
        return self.output_path


def start_window_recording(*, window_title: str, fps: int) -> ActiveRecording:
    """Start ffmpeg recording of an X11 window and output audio."""

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RecordingError("Cannot start recording: ffmpeg was not found on PATH.")

    display = str(os.environ.get("DISPLAY", ":0.0")).strip() or ":0.0"
    x, y, width, height = _resolve_window_geometry(window_title=window_title, display=display)
    output_path = _build_output_path()
    stderr_log_path = Path(
        tempfile.mkstemp(prefix="midi_mover_ffmpeg_recording_", suffix=".log")[1]
    )

    audio_source = _resolve_output_audio_source() or "default"
    pulse_supported = _ffmpeg_supports_input_format("pulse")

    command = _build_ffmpeg_command(
        ffmpeg_path=ffmpeg_path,
        display=display,
        x=x,
        y=y,
        width=width,
        height=height,
        fps=fps,
        output_path=output_path,
        audio_source=audio_source,
        pulse_supported=pulse_supported,
    )

    audio_process: subprocess.Popen[bytes] | None = None
    stdin_for_ffmpeg: Any = subprocess.PIPE
    use_stdin_quit = True

    if not pulse_supported:
        parec_path = shutil.which("parec")
        if parec_path is None:
            raise RecordingError(
                "ffmpeg does not support pulse input format and parec is unavailable. "
                "Install ffmpeg with pulse support or install pulseaudio-utils (parec)."
            )
        try:
            audio_process = subprocess.Popen(
                [
                    parec_path,
                    "--device",
                    audio_source,
                    "--format=s16le",
                    "--rate=48000",
                    "--channels=2",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise RecordingError(f"Failed to start parec fallback for audio capture: {exc}") from exc

        if audio_process.stdout is None:
            _stop_subprocess_quietly(audio_process)
            raise RecordingError("parec fallback did not expose stdout for ffmpeg audio pipe.")
        stdin_for_ffmpeg = audio_process.stdout
        use_stdin_quit = False

    try:
        stderr_log_file = stderr_log_path.open("wb")
        try:
            process = subprocess.Popen(
                command,
                stdin=stdin_for_ffmpeg,
                stdout=subprocess.DEVNULL,
                stderr=stderr_log_file,
            )
        finally:
            stderr_log_file.close()
    except OSError as exc:
        if audio_process is not None:
            _stop_subprocess_quietly(audio_process)
        raise RecordingError(f"Failed to launch ffmpeg recording process: {exc}") from exc

    if audio_process is not None and audio_process.stdout is not None:
        audio_process.stdout.close()

    time.sleep(0.25)
    if process.poll() is not None:
        if audio_process is not None:
            _stop_subprocess_quietly(audio_process)
        raise RecordingError(
            _build_ffmpeg_failure_message(
                prefix=f"ffmpeg recording process exited immediately with code {process.returncode}.",
                stderr_log_path=stderr_log_path,
            )
        )

    return ActiveRecording(
        process=process,
        output_path=output_path,
        stderr_log_path=stderr_log_path,
        audio_process=audio_process,
        use_stdin_quit=use_stdin_quit,
    )


def _build_ffmpeg_command(
    *,
    ffmpeg_path: str,
    display: str,
    x: int,
    y: int,
    width: int,
    height: int,
    fps: int,
    output_path: Path,
    audio_source: str,
    pulse_supported: bool,
) -> list[str]:
    command = [
        ffmpeg_path,
        "-y",
        "-f",
        "x11grab",
        "-framerate",
        str(max(1, int(fps))),
        "-video_size",
        f"{width}x{height}",
        "-i",
        f"{display}+{x},{y}",
    ]
    if pulse_supported:
        command.extend(["-f", "pulse", "-i", audio_source])
    else:
        command.extend(["-f", "s16le", "-ar", "48000", "-ac", "2", "-i", "pipe:0"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return command


def _build_ffmpeg_failure_message(*, prefix: str, stderr_log_path: Path | None) -> str:
    if stderr_log_path is None:
        return prefix
    details = _read_stderr_log_excerpt(stderr_log_path)
    if not details:
        return prefix
    return f"{prefix} ffmpeg output: {details}"


def _read_stderr_log_excerpt(path: Path, max_lines: int = 10) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    if not lines:
        return ""
    excerpt = lines[-max(1, int(max_lines)) :]
    return " | ".join(line.strip() for line in excerpt if line.strip())


def _ffmpeg_supports_input_format(format_name: str) -> bool:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return False
    result = subprocess.run(
        [ffmpeg_path, "-hide_banner", "-demuxers"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    pattern = re.compile(rf"\b{re.escape(format_name)}\b")
    return any(pattern.search(line) for line in result.stdout.splitlines())


def _stop_subprocess_quietly(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=1.5)
    except Exception:
        process.kill()
        process.wait(timeout=1.5)


def _build_output_path() -> Path:
    recordings_dir = Path("~/data/midi_mover/recordings").expanduser().resolve()
    recordings_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return recordings_dir / f"midi_mover_{timestamp}.mp4"


def _resolve_window_geometry(*, window_title: str, display: str) -> tuple[int, int, int, int]:
    for _ in range(20):
        geometry = _query_xwininfo_geometry(window_title=window_title, display=display)
        if geometry is not None:
            return geometry
        time.sleep(0.1)
    raise RecordingError(
        f"Unable to locate X11 window '{window_title}' for recording via xwininfo."
    )


def _query_xwininfo_geometry(*, window_title: str, display: str) -> tuple[int, int, int, int] | None:
    result = subprocess.run(
        ["xwininfo", "-display", display, "-name", window_title],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    output = result.stdout
    x = _extract_int(output, r"Absolute upper-left X:\s+(-?\d+)")
    y = _extract_int(output, r"Absolute upper-left Y:\s+(-?\d+)")
    width = _extract_int(output, r"Width:\s+(\d+)")
    height = _extract_int(output, r"Height:\s+(\d+)")
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def _extract_int(text: str, pattern: str) -> int:
    match = re.search(pattern, text)
    if match is None:
        raise RecordingError(f"Failed to parse xwininfo output for pattern: {shlex.quote(pattern)}")
    return int(match.group(1))


def _resolve_output_audio_source() -> str | None:
    sink = _run_text_command(["pactl", "get-default-sink"])
    if sink:
        sink_monitor = f"{sink}.monitor"
        sources = _run_text_command(["pactl", "list", "short", "sources"])
        if sink_monitor in sources:
            return sink_monitor
    return None


def _run_text_command(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()
