# Alignment Utility

A small desktop tool for lining up an interferometer. It shows a live camera
image next to a live FFT of the selected pupil so you can see fringe frequency
and orientation while you adjust the setup.

Works on Windows, macOS, and Linux with ordinary UVC / USB cameras via OpenCV.

## Features

- Live camera view
- Drag a circle on the camera image, or auto-fit one
- Live FFT of the cropped pupil, shown on the same page
- Brightness slider (hardware when the camera supports it, plus a software fallback)

## Run

This application **requires `uv`**, please install it [from here](https://docs.astral.sh/uv/getting-started/installation/).

On Linux, install Python's Tk support first:

```bash
# Debian / Ubuntu
sudo apt install python3-tk

# Fedora
sudo dnf install python3-tkinter

# Arch Linux
sudo pacman -S tk
```

Then run:

```bash
uvx --from git+https://github.com/Wavefront-Pro/wvft-align.git wvft-align
```

Or clone and run locally:

```bash
git clone https://github.com/Wavefront-Pro/wvft-align.git
cd wvft-alignment-utility
uv run wvft-align
```

To run the application from a desktop shortcut on Windows:

1. Right-click an empty space on your desktop, select New > Shortcut.
2. Paste the following command into the location box:
    `powershell.exe -ExecutionPolicy Bypass "uvx --from git+https://github.com/Wavefront-Pro/wvft-align.git wvft-align`
3. Click Next, add a descriptive name and click Finish.

## Usage

1. Pick a camera.
2. Drag on the camera image to set the analysis circle, or click
   **Auto-fit circle** once to detect the pupil.
3. Watch the FFT while you tip/tilt / change path length. A well-aligned
   interferogram shows a clean pair of sidebands.


## Adding a camera backend

Cameras are viewed as plugins. To add camera support, please open a pull request with the following:

Subclass `cameras.base.Camera`, give it a unique
`backend_id`, implement `list_devices` / `open` / `close` / `is_open` / `grab`,
and decorate the class with `@register`.

```python
from wvft_align.cameras.base import Camera, CameraDevice, register

@register
class MyCamera(Camera):
    backend_id = "my_camera"
    backend_label = "My Camera"

    @classmethod
    def list_devices(cls) -> list[CameraDevice]:
        return [CameraDevice("my_camera", 0, "Device 0")]

    def open(self, index: int = 0) -> None: ...
    def close(self) -> None: ...
    def is_open(self) -> bool: ...
    def grab(self): ...          # RGB uint8 array, or None
    def set_brightness(self, value: float) -> None: ...  # optional, 0–100
```

Import the class from `src/wvft_align/cameras/__init__.py` so it registers at startup.
The built-in OpenCV backend in `src/wvft_align/cameras/opencv.py` is a complete reference example.

### Required tests

Every backend must subclass the shared contract in `tests/camera_contract.py`.
That file is the checklist — do not skip tests from it.

```python
# tests/test_my_camera.py
import pytest
from wvft_align.cameras.my_camera import MyCamera
from camera_contract import CameraContract

@pytest.mark.hardware          # skip unless: pytest -m hardware
class TestMyCamera(CameraContract):
    camera_cls = MyCamera
```

What that covers:

| Test | What it checks |
|---|---|
| `test_backend_identity` | `backend_id` and `backend_label` are set |
| `test_list_devices_never_raises` | listing is safe and returns `CameraDevice`s |
| `test_close_and_grab_before_open` | unused instances are inert |
| `test_open_close_lifecycle` | open → `is_open` → close (and close again) |
| `test_grab_frame_shape` | `uint8` gray `(H, W)` or RGB `(H, W, 3)` |
| `test_set_brightness_accepts_slider_range` | 0 / 50 / 100 do not raise |

Mark real-device classes with `@pytest.mark.hardware` so a normal
`uv run pytest` stays fast (in-memory fake only). Hardware tests also
skip when `list_devices()` is empty. Add extra tests in the same file
for vendor-specific behaviour.

```bash
uv run pytest              # default: no live cameras
uv run pytest -m hardware  # open real devices, takes longer
```
