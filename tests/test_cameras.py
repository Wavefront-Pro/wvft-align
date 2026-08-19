"""Registry tests plus the contract suite for built-in backends.

New plugin? Add a file like ``tests/test_my_camera.py``::

    import pytest
    from wvft_align.cameras.my_camera import MyCamera
    from camera_contract import CameraContract

    @pytest.mark.hardware          # skip unless: pytest -m hardware
    class TestMyCamera(CameraContract):
        camera_cls = MyCamera
"""

from __future__ import annotations

import numpy as np
import pytest
from camera_contract import CameraContract

from wvft_align.cameras.base import _BACKENDS, Camera, CameraDevice, open_device
from wvft_align.cameras.opencv import OpenCVCamera


def test_opencv_backend_is_registered() -> None:
    assert OpenCVCamera.backend_id in _BACKENDS


def test_unknown_backend_raises() -> None:
    try:
        open_device(CameraDevice("does-not-exist", 0, "x"))
    except RuntimeError as exc:
        assert "Unknown camera backend" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


class FakeCamera(Camera):
    """In-memory backend used to exercise the full contract in CI."""

    backend_id = "fake"
    backend_label = "Fake"

    def __init__(self) -> None:
        self._open = False

    @classmethod
    def list_devices(cls) -> list[CameraDevice]:
        return [CameraDevice("fake", 0, "Simulated")]

    def open(self, index: int = 0) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def is_open(self) -> bool:
        return self._open

    def grab(self):
        if not self._open:
            return None
        return np.zeros((16, 16, 3), dtype=np.uint8)


class TestFakeCamera(CameraContract):
    camera_cls = FakeCamera


@pytest.mark.hardware
class TestOpenCVCamera(CameraContract):
    """Live webcam. Run with: uv run pytest -m hardware"""

    camera_cls = OpenCVCamera


def test_register_decorator_rejects_empty_id() -> None:
    from wvft_align.cameras.base import register

    class NoId(Camera):
        backend_id = ""
        backend_label = "x"

        @classmethod
        def list_devices(cls):
            return []

        def open(self, index=0):
            pass

        def close(self):
            pass

        def is_open(self):
            return False

        def grab(self):
            return None

    try:
        register(NoId)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    finally:
        _BACKENDS.pop("", None)
