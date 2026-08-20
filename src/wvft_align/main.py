"""Alignment Utility — live camera + FFT for interferometer setup."""

from __future__ import annotations

import logging
import platform
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import sv_ttk
from numpy.typing import NDArray

from wvft_align.cameras import Camera, list_devices, open_device
from wvft_align.processing import (
    clamp_radius,
    compute_fft,
    crop_to_circle,
    fft_to_uint8,
    find_circle,
)
from wvft_align.ui import CameraDialog, ImageCanvas

log = logging.getLogger(__name__)

APP_VERSION = "20260819"
UI_INTERVAL_MS = 50
FFT_SIZE = 256
WORKER_INTERVAL_S = 0.05


class Worker:
    """Background thread for camera grab + FFT only."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._camera: Camera | None = None
        self._circle: tuple[int, int, int] | None = None
        self.frame: NDArray | None = None
        self.fft: NDArray | None = None
        self.timestamp = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.set_camera(None)

    def set_camera(self, camera: Camera | None) -> None:
        with self._lock:
            old = self._camera
            self._camera = camera
        if old is not None and old is not camera:
            old.close()

    def set_circle(self, circle: tuple[int, int, int] | None) -> None:
        with self._lock:
            self._circle = circle

    def _run(self) -> None:
        while not self._stop.is_set():
            tic = time.perf_counter()
            try:
                self._step()
            except Exception:
                log.exception("Worker step failed")
            sleep = max(0.0, WORKER_INTERVAL_S - (time.perf_counter() - tic))
            self._stop.wait(timeout=sleep)

    def _step(self) -> None:
        with self._lock:
            camera = self._camera
            circle = self._circle
        if camera is None or not camera.is_open():
            return

        frame = camera.grab()
        if frame is None:
            return

        preview = None
        if circle is not None:
            x, y, dia = circle
            crop = crop_to_circle(frame, x, y, dia, FFT_SIZE)
            preview = fft_to_uint8(compute_fft(crop, FFT_SIZE))

        with self._lock:
            self.frame = frame
            if preview is not None:
                self.fft = preview
            self.timestamp = time.perf_counter()


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title(f"Wavefront Alignment Utility v{APP_VERSION}")
        self.geometry("1180x700")
        self.minsize(1100, 640)

        if platform.system() == "Windows":
            import ctypes

            # Tell Windows that the app takes care of DPI scaling
            ctypes.windll.shcore.SetProcessDpiAwareness(2)

            icon = Path(__file__).with_name("resources") / "icon.ico"
            if icon.exists():
                self.iconbitmap(icon)

        sv_ttk.set_theme("light")

        self.worker = Worker()
        self.camera: Camera | None = None
        self.circle: tuple[int, int, int] | None = None  # sensor pixels
        self._gain = 1.0
        self._last_ts = 0.0

        self._build()
        self.worker.start()
        self.after(80, self._startup)
        self.protocol("WM_DELETE_WINDOW", self._quit)

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        page = ttk.Frame(self, padding=12)
        page.grid(row=0, column=0, sticky="nsew")
        page.columnconfigure(0, weight=1)
        page.columnconfigure(1, weight=1)
        page.rowconfigure(0, weight=1)

        self.live = ImageCanvas(page, size=540)
        self.live.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.live.on_circle_drawn = self._on_circle_drawn

        self.fft = ImageCanvas(page, size=540, interactive=False)
        self.fft.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        bar = ttk.Frame(page)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        bar.columnconfigure(4, weight=1)

        ttk.Button(bar, text="Auto-fit circle", command=self._auto_fit).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(bar, text="Change camera", command=self._pick_camera).grid(
            row=0, column=1, padx=(0, 16)
        )

        ttk.Label(bar, text="Brightness").grid(row=0, column=2, padx=(0, 6))
        self.brightness = tk.DoubleVar(value=50)
        ttk.Scale(
            bar,
            from_=0,
            to=100,
            variable=self.brightness,
            command=self._on_brightness,
            length=180,
        ).grid(row=0, column=3)

        self.status = ttk.Label(bar, text="Starting…", foreground="gray")
        self.status.grid(row=0, column=4, sticky="e")

        hint = ttk.Label(
            page,
            text="Drag on the camera image to set the analysis circle.  "
            "The FFT is computed from the cropped pupil.",
            foreground="gray",
        )
        hint.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _startup(self) -> None:
        self._pick_camera(first=True)
        self._tick()

    def _pick_camera(self, first: bool = False) -> None:
        self.status.configure(text="Scanning cameras…")
        self.update_idletasks()
        devices = list_devices()
        self.status.configure(text="Select camera…")
        dialog = CameraDialog(self, devices)
        self.wait_window(dialog)

        if dialog.cancelled or dialog.selected is None:
            if first:
                self.status.configure(text="No camera — pick a camera to start.")
            return

        try:
            self.status.configure(text="Connecting to camera…")
            self.update_idletasks()
            camera = open_device(dialog.selected)
        except Exception as exc:
            messagebox.showwarning("Camera", f"Could not open camera:\n{exc}")
            self.status.configure(text="Camera failed to open.")
            return

        self.camera = camera
        self.worker.set_camera(camera)
        self.status.configure(text=f"Live · {dialog.selected.label}")

    def _quit(self) -> None:
        self.worker.stop()
        self.destroy()

    def _on_circle_drawn(self, x: float, y: float, diameter: float) -> None:
        frame = self.worker.frame
        if frame is None:
            return
        h, w = frame.shape[:2]
        r = clamp_radius(x, y, diameter / 2, w, h)
        self.circle = (int(x), int(y), int(r * 2))
        self.live.set_circle(*self.circle[:2], r)
        self.worker.set_circle(self.circle)

    def _auto_fit(self) -> None:
        frame = self.worker.frame
        if frame is None:
            return
        cx, cy, dia = find_circle(frame)
        self.circle = (cx, cy, dia)
        self.live.set_circle(cx, cy, dia / 2)
        self.worker.set_circle(self.circle)

    def _on_brightness(self, _=None) -> None:
        value = self.brightness.get()
        if self.camera is not None:
            self.camera.set_brightness(value)
        self._gain = 0.5 + (value / 100.0) * 1.5

    def _tick(self) -> None:
        try:
            if not self.live.is_drawing and self.worker.timestamp != self._last_ts:
                self._last_ts = self.worker.timestamp
                if self.worker.frame is not None:
                    self.live.set_image(self.worker.frame, self._gain)
                    if self.circle is not None:
                        self.live.set_circle(
                            self.circle[0], self.circle[1], self.circle[2] / 2
                        )
                if self.worker.fft is not None:
                    self.fft.set_image(self.worker.fft)
        except Exception:
            log.exception("UI update failed")
        self.after(UI_INTERVAL_MS, self._tick)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    App().mainloop()


if __name__ == "__main__":
    main()
