"""OpenCV / UVC backend. Works with most USB webcams on Windows, macOS, Linux."""

from __future__ import annotations

import logging
import platform

import cv2

from wvft_align.cameras.base import Camera, CameraDevice, register

log = logging.getLogger(__name__)

# Typical OpenCV exposure range is log2(seconds). Map a 0–100 slider onto it.
_EXPOSURE_MIN = -13.0
_EXPOSURE_MAX = -1.0


@register
class OpenCVCamera(Camera):
    backend_id = "opencv"
    backend_label = "OpenCV"

    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None

    @classmethod
    def list_devices(cls) -> list[CameraDevice]:
        if platform.system() == "Windows":
            return _list_windows()
        return _list_opencv()

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def open(self, index: int = 0) -> None:
        self.close()
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}.")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap = cap
        _try_set(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        log.info(
            "Opened OpenCV camera %s (%dx%d)",
            index,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def grab(self):
        if not self.is_open():
            return None
        ok, bgr = self._cap.read()  # type: ignore[union-attr]
        if not ok:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def set_brightness(self, value: float) -> None:
        if not self.is_open():
            return
        _try_set(self._cap, cv2.CAP_PROP_BRIGHTNESS, value)  # type: ignore[arg-type]
        exposure = _EXPOSURE_MIN + (value / 100.0) * (_EXPOSURE_MAX - _EXPOSURE_MIN)
        _try_set(self._cap, cv2.CAP_PROP_EXPOSURE, exposure)  # type: ignore[arg-type]


def _try_set(cap: cv2.VideoCapture, prop: int, value: float) -> bool:
    try:
        return bool(cap.set(prop, value))
    except Exception:
        return False


def _list_windows() -> list[CameraDevice]:
    try:
        import duvc_ctl as duvc

        cameras = duvc.list_cameras()
        if not cameras:
            return []
        return [CameraDevice("opencv", i, str(name)) for i, name in enumerate(cameras)]
    except Exception:
        log.exception("duvc-ctl enumeration failed; falling back to OpenCV")
        return _list_opencv()


def _list_opencv(max_index: int = 6) -> list[CameraDevice]:
    found: list[CameraDevice] = []
    misses = 0
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        opened = cap.isOpened()
        cap.release()
        if opened:
            found.append(CameraDevice("opencv", index, f"Camera {index}"))
            misses = 0
        else:
            misses += 1
            if found and misses >= 2:
                break
    return found
