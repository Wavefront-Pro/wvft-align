"""Tk widgets: zoomable camera canvas, FFT canvas, camera picker."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from numpy.typing import NDArray
from PIL import Image, ImageTk

from wvft_align.processing import apply_gain, scale_to_fit


class ImageCanvas(ttk.Frame):
    """Canvas that shows an image, scaled to fit, with an optional circle overlay.

    Circle coordinates are always in source (sensor) pixels.
    """

    def __init__(
        self, parent, size: int = 520, title: str = "", interactive: bool = True
    ) -> None:
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        if title:
            ttk.Label(self, text=title, font=("", 10, "bold")).grid(
                row=0, column=0, sticky="w", pady=(0, 4)
            )

        self.canvas_size = size
        self.scale = 1.0
        self._src_scale = 1.0
        self.image: Image.Image | None = None
        self.circle: tuple[float, float, float] | None = None
        self.on_circle_drawn = None
        self.is_drawing = False

        self._draw_start: tuple[float, float] | None = None
        self._photo: ImageTk.PhotoImage | None = None

        self.canvas = tk.Canvas(
            self,
            width=size,
            height=size,
            bg="#111",
            highlightthickness=0,
            cursor="crosshair" if interactive else "arrow",
        )
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._on_resize)
        if interactive:
            self.canvas.bind("<ButtonPress-1>", self._start_draw)
            self.canvas.bind("<B1-Motion>", self._drag_draw)
            self.canvas.bind("<ButtonRelease-1>", self._finish_draw)

    def set_image(self, frame: NDArray, gain: float = 1.0) -> None:
        fitted, self._src_scale = scale_to_fit(
            frame, self.canvas_size, self.canvas_size
        )
        if abs(gain - 1.0) >= 1e-3:
            fitted = apply_gain(fitted, gain)
        self.image = Image.fromarray(fitted)
        self._fit()
        self._redraw()

    def set_circle(self, x: float, y: float, radius: float) -> None:
        self.circle = (x, y, radius)
        self._draw_circle()

    def _on_resize(self, event) -> None:
        size = min(event.width, event.height)
        if size < 2 or size == self.canvas_size:
            return
        self.canvas_size = size
        if self.image is not None:
            self._fit()
            self._redraw()

    def _fit(self) -> None:
        if self.image is None:
            return
        w, h = self.image.size
        self.scale = min(self.canvas_size / max(w, 1), self.canvas_size / max(h, 1))

    def _redraw(self) -> None:
        if self.image is None:
            return
        w, h = self.image.size
        new_w = max(1, int(w * self.scale))
        new_h = max(1, int(h * self.scale))
        resized = self.image.resize((new_w, new_h), Image.Resampling.NEAREST)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("image")
        ox = (self.canvas_size - new_w) // 2
        oy = (self.canvas_size - new_h) // 2
        self.canvas.create_image(ox, oy, anchor="nw", image=self._photo, tags="image")
        self.canvas.tag_lower("image")
        self._draw_circle()

    def _image_origin(self) -> tuple[int, int]:
        if self.image is None:
            return 0, 0
        w, h = self.image.size
        ox = (self.canvas_size - int(w * self.scale)) // 2
        oy = (self.canvas_size - int(h * self.scale)) // 2
        return ox, oy

    def _canvas_to_src(self, cx: float, cy: float) -> tuple[float, float]:
        ox, oy = self._image_origin()
        s = self.scale * self._src_scale
        return (cx - ox) / s, (cy - oy) / s

    def _src_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        ox, oy = self._image_origin()
        s = self.scale * self._src_scale
        return ox + x * s, oy + y * s

    def _draw_circle(self, preview: tuple[float, float, float] | None = None) -> None:
        self.canvas.delete("circle")
        data = preview or self.circle
        if data is None:
            return
        x, y, r = data
        x1, y1 = self._src_to_canvas(x - r, y - r)
        x2, y2 = self._src_to_canvas(x + r, y + r)
        self.canvas.create_oval(
            x1, y1, x2, y2, outline="#ff4d27", width=2, tags="circle"
        )

    def _start_draw(self, event) -> None:
        self.is_drawing = True
        self._draw_start = (event.x, event.y)

    def _drag_draw(self, event) -> None:
        if self._draw_start is None:
            return
        sx, sy = self._draw_start
        cx, cy = (sx + event.x) / 2, (sy + event.y) / 2
        r = (((event.x - sx) ** 2 + (event.y - sy) ** 2) ** 0.5) / 2
        ix, iy = self._canvas_to_src(cx, cy)
        self._draw_circle((ix, iy, r / (self.scale * self._src_scale)))

    def _finish_draw(self, event) -> None:
        self.is_drawing = False
        if self._draw_start is None:
            return
        sx, sy = self._draw_start
        self._draw_start = None
        cx, cy = (sx + event.x) / 2, (sy + event.y) / 2
        r = (((event.x - sx) ** 2 + (event.y - sy) ** 2) ** 0.5) / 2
        if r < 5:
            self._draw_circle()
            return
        ix, iy = self._canvas_to_src(cx, cy)
        ir = r / (self.scale * self._src_scale)
        self.circle = (ix, iy, ir)
        self._draw_circle()
        if self.on_circle_drawn is not None:
            self.on_circle_drawn(ix, iy, ir * 2)


class CameraDialog(tk.Toplevel):
    def __init__(self, parent, devices) -> None:
        super().__init__(parent)
        self.title("Select camera")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.selected = None
        self.cancelled = False
        self._devices = list(devices)

        ttk.Label(self, text="Available cameras").pack(
            anchor="w", padx=16, pady=(16, 8)
        )

        self._var = tk.IntVar(value=0 if self._devices else -1)
        for i, device in enumerate(self._devices):
            ttk.Radiobutton(self, text=device.label, variable=self._var, value=i).pack(
                anchor="w", padx=24, pady=2
            )

        if not devices:
            ttk.Label(self, text="No cameras found.", foreground="gray").pack(
                padx=16, pady=8
            )

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", padx=16, pady=16)
        ttk.Button(buttons, text="Skip", command=self._skip).pack(side="right")
        connect = ttk.Button(buttons, text="Connect", command=self._connect)
        connect.pack(side="right", padx=(0, 8))
        if not devices:
            connect.state(["disabled"])

        self.protocol("WM_DELETE_WINDOW", self._skip)
        self.wait_visibility()
        self.geometry(f"+{parent.winfo_rootx() + 80}+{parent.winfo_rooty() + 80}")

    def _connect(self) -> None:
        index = self._var.get()
        if 0 <= index < len(self._devices):
            self.selected = self._devices[index]
        self.destroy()

    def _skip(self) -> None:
        self.cancelled = True
        self.destroy()
