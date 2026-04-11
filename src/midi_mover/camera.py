"""Camera frame acquisition helpers for midi_mover."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CameraFrame:
    """One camera frame prepared for both CV and pygame consumers."""

    bgr_frame: Any
    rgb_frame: Any
    render_surface: Any
    width: int
    height: int
    mirrored: bool


class CameraFrameError(RuntimeError):
    """Raised when a camera frame cannot be read or converted."""


class CameraFrameReader:
    """Read OpenCV frames and convert them for pygame rendering."""

    def __init__(self, camera: Any, mirror: bool) -> None:
        self._camera = camera
        self._mirror = bool(mirror)

    @property
    def mirror(self) -> bool:
        return self._mirror

    def read(self, pygame_module: Any) -> CameraFrame:
        """Return the next prepared frame.

        The returned frame preserves a BGR copy for OpenCV-style processing,
        an RGB copy for general image processing, and a pygame surface for
        immediate rendering.
        """

        try:
            import cv2
            import numpy as np
        except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
            raise CameraFrameError(
                "Camera frame conversion requires opencv-python and numpy in the active environment."
            ) from exc

        frame_ok, bgr_frame = self._camera.read()
        if not frame_ok or bgr_frame is None:
            raise CameraFrameError("Camera read failed: OpenCV could not produce a frame.")

        if self._mirror:
            bgr_frame = cv2.flip(bgr_frame, 1)

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        contiguous_rgb = np.ascontiguousarray(rgb_frame)
        render_surface = pygame_module.image.frombuffer(
            contiguous_rgb.tobytes(),
            (contiguous_rgb.shape[1], contiguous_rgb.shape[0]),
            "RGB",
        )

        return CameraFrame(
            bgr_frame=bgr_frame,
            rgb_frame=contiguous_rgb,
            render_surface=render_surface,
            width=int(contiguous_rgb.shape[1]),
            height=int(contiguous_rgb.shape[0]),
            mirrored=self._mirror,
        )