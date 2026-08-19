"""Camera plugin interface.

Subclass :class:`Camera`, set ``backend_id`` / ``backend_label``, implement
the abstract methods, and decorate the class with :func:`register`.
See ``wvft_align/cameras/opencv.py`` for a complete example.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from numpy.typing import NDArray

log = logging.getLogger(__name__)

_BACKENDS: dict[str, type[Camera]] = {}


@dataclass(frozen=True)
class CameraDevice:
    """One physical camera, as advertised by a backend."""

    backend: str
    index: int
    name: str

    @property
    def label(self) -> str:
        cls = _BACKENDS.get(self.backend)
        prefix = cls.backend_label if cls else self.backend
        return f"{prefix} · {self.name}"


class Camera(ABC):
    """Minimal camera contract used by the app.

    Frames must be RGB ``uint8`` arrays of shape ``(H, W, 3)`` (or grayscale
    ``(H, W)``). ``grab`` should be non-blocking if possible and return the
    latest frame, or ``None`` if nothing is available.
    """

    backend_id: str = ""
    backend_label: str = ""

    @classmethod
    @abstractmethod
    def list_devices(cls) -> list[CameraDevice]:
        """Return cameras this backend can currently see. Never raise."""

    @abstractmethod
    def open(self, index: int = 0) -> None:
        """Connect to the device at ``index`` from :meth:`list_devices`."""

    @abstractmethod
    def close(self) -> None:
        """Release hardware. Safe to call when already closed."""

    @abstractmethod
    def is_open(self) -> bool:
        """True while a device is connected and ready to grab."""

    @abstractmethod
    def grab(self) -> NDArray | None:
        """Return the latest RGB (or gray) frame, or ``None``."""

    def set_brightness(self, value: float) -> None:
        """Optional. ``value`` is 0 (dark) … 100 (bright). Default is a no-op."""


def register(cls: type[Camera]) -> type[Camera]:
    """Class decorator. Call this on every backend you want the app to see."""

    if not cls.backend_id:
        raise ValueError(f"{cls.__name__} must set backend_id")
    _BACKENDS[cls.backend_id] = cls
    log.debug("Registered camera backend %s", cls.backend_id)
    return cls


def list_devices() -> list[CameraDevice]:
    """Ask every registered backend for its cameras."""

    found: list[CameraDevice] = []
    for cls in _BACKENDS.values():
        try:
            found.extend(cls.list_devices())
        except Exception:
            log.exception("Failed to list cameras for backend %s", cls.backend_id)
    return found


def open_device(device: CameraDevice) -> Camera:
    """Construct the right backend and open ``device``."""

    try:
        cls = _BACKENDS[device.backend]
    except KeyError as exc:
        raise RuntimeError(f"Unknown camera backend {device.backend!r}") from exc
    camera = cls()
    camera.open(device.index)
    return camera
