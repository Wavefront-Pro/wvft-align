"""Shared contract tests for every camera backend.

When you add a plugin, add a test class that points at it:

    # tests/test_my_camera.py
    import pytest
    from wvft_align.cameras.my_camera import MyCamera
    from camera_contract import CameraContract

    @pytest.mark.hardware          # skip unless: pytest -m hardware
    class TestMyCamera(CameraContract):
        camera_cls = MyCamera

Use ``@pytest.mark.hardware`` for backends that talk to real devices so
``uv run pytest`` stays fast. Hardware-only checks also skip when
``list_devices()`` is empty.

Required coverage (implemented here — do not drop these):

- ``backend_id`` and ``backend_label`` are non-empty
- ``list_devices()`` never raises and returns ``CameraDevice`` values
  whose ``backend`` matches ``backend_id``
- ``close()`` and ``grab()`` are safe before ``open()``
- ``open()`` / ``is_open()`` / ``close()`` form a working lifecycle
  (skipped if no device is present)
- ``close()`` is idempotent
- ``grab()`` returns ``None`` or a ``uint8`` array of shape ``(H, W)``
  or ``(H, W, 3)``
- ``set_brightness(0..100)`` does not raise

Add extra tests in your own file for vendor-specific behaviour
(triggers, bit depth, dropped frames, …).
"""

from __future__ import annotations

import numpy as np
import pytest

from wvft_align.cameras.base import Camera, CameraDevice


class CameraContract:
    """Mixin. Set ``camera_cls`` on a ``Test*`` subclass."""

    camera_cls: type[Camera]

    def _devices(self) -> list[CameraDevice]:
        return self.camera_cls.list_devices()

    def _require_device(self) -> CameraDevice:
        devices = self._devices()
        if not devices:
            pytest.skip(f"No {self.camera_cls.backend_id} cameras connected")
        return devices[0]

    def test_backend_identity(self) -> None:
        assert self.camera_cls.backend_id, "set backend_id"
        assert self.camera_cls.backend_label, "set backend_label"

    def test_list_devices_never_raises(self) -> None:
        devices = self.camera_cls.list_devices()
        assert isinstance(devices, list)
        for device in devices:
            assert isinstance(device, CameraDevice)
            assert device.backend == self.camera_cls.backend_id
            assert device.name

    def test_close_and_grab_before_open(self) -> None:
        camera = self.camera_cls()
        assert camera.is_open() is False
        camera.close()
        assert camera.grab() is None

    def test_open_close_lifecycle(self) -> None:
        device = self._require_device()
        camera = self.camera_cls()
        camera.open(device.index)
        try:
            assert camera.is_open()
        finally:
            camera.close()
        assert camera.is_open() is False
        camera.close()  # idempotent

    def test_grab_frame_shape(self) -> None:
        device = self._require_device()
        camera = self.camera_cls()
        camera.open(device.index)
        try:
            frame = camera.grab()
            if frame is None:
                pytest.skip("backend returned no frame")
            assert frame.dtype == np.uint8
            assert frame.ndim in (2, 3)
            if frame.ndim == 3:
                assert frame.shape[2] == 3
        finally:
            camera.close()

    def test_set_brightness_accepts_slider_range(self) -> None:
        device = self._require_device()
        camera = self.camera_cls()
        camera.open(device.index)
        try:
            camera.set_brightness(0)
            camera.set_brightness(50)
            camera.set_brightness(100)
        finally:
            camera.close()
