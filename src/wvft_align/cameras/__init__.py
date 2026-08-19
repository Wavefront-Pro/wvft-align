"""Camera backends.

Importing this package registers the built-in OpenCV backend. Add your own by
subclassing :class:`wvft_align.cameras.base.Camera` and decorating it with ``@register``.
"""

from wvft_align.cameras.base import (
    Camera,
    CameraDevice,
    list_devices,
    open_device,
    register,
)
from wvft_align.cameras.opencv import OpenCVCamera

__all__ = [
    "Camera",
    "CameraDevice",
    "OpenCVCamera",
    "list_devices",
    "open_device",
    "register",
]
