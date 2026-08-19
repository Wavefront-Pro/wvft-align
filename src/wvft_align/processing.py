"""Image helpers: circle detection, crop, FFT preview, display scale."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image


def to_gray(frame: NDArray) -> NDArray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def apply_gain(frame: NDArray, gain: float) -> NDArray:
    if abs(gain - 1.0) < 1e-3:
        return frame
    out = np.clip(frame.astype(np.float32) * gain, 0, 255)
    return out.astype(np.uint8)


def clamp_radius(x: float, y: float, r: float, width: int, height: int) -> int:
    x = min(max(x, 0), width - 1)
    y = min(max(y, 0), height - 1)
    r = min(x, y, width - x, height - y, r)
    return max(1, int(r))


def scale_to_fit(frame: NDArray, max_w: int, max_h: int) -> tuple[NDArray, float]:
    """Resize so the frame fits in max_w × max_h. Returns (image, scale)."""

    h, w = frame.shape[:2]
    scale = min(1.0, max_w / max(w, 1), max_h / max(h, 1))
    if scale >= 1.0:
        return frame, 1.0
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA), scale


def find_circle(frame: NDArray) -> tuple[int, int, int]:
    """Return (cx, cy, diameter) of the brightest high-contrast disk."""

    gray = to_gray(frame)
    height, width = gray.shape[:2]
    fallback = (width // 2, height // 2, int(0.9 * min(width, height)))

    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    ksize = 7
    local_mean = cv2.blur(norm, (ksize, ksize))
    local_sq = cv2.blur(norm**2, (ksize, ksize))
    variance = local_sq - local_mean**2
    var8 = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(var8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    points = cv2.findNonZero(mask)
    if points is None:
        return fallback
    pts = np.ascontiguousarray(np.asarray(points).reshape(-1, 2)[::32], dtype=np.float32)
    if len(pts) < 3:
        return fallback

    try:
        (cx, cy), radius = cv2.minEnclosingCircle(pts)
    except cv2.error:
        return fallback

    radius = max(1.0, radius - (ksize - 1) / 2)
    cx, cy = int(cx), int(cy)
    radius = clamp_radius(cx, cy, radius, width, height)
    return cx, cy, int(radius * 2)


def crop_to_circle(frame: NDArray, x: int, y: int, diameter: int, size: int) -> NDArray:
    radius = max(1, diameter // 2)
    crop = frame[y - radius : y + radius, x - radius : x + radius]
    if crop.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    return np.asarray(Image.fromarray(crop).resize((size, size)), dtype=np.uint8)


def compute_fft(frame: NDArray, size: int = 256) -> NDArray:
    """Log-power spectrum of the (already cropped) pupil, center-cropped to size."""

    gray = to_gray(frame).astype(np.float64)
    if gray.size == 0:
        return np.zeros((size, size), dtype=np.float64)

    centered = np.where(gray == 0, 0.0, gray - np.nanmean(gray))
    n = max(size, max(centered.shape))
    n_pad = 1 if n <= 1 else 1 << (n - 1).bit_length()
    padded = np.zeros((n_pad, n_pad), dtype=np.float64)
    padded[: centered.shape[0], : centered.shape[1]] = centered
    spectrum = np.fft.fftshift(np.fft.fft2(padded))
    power = np.log(np.abs(spectrum) ** 2 + 1e-12)

    half = size // 2
    mid = power.shape[0] // 2
    return power[mid - half : mid + half, mid - half : mid + half]


def fft_to_uint8(power: NDArray) -> NDArray:
    finite = power[np.isfinite(power)]
    if finite.size == 0:
        return np.zeros(power.shape, dtype=np.uint8)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros(power.shape, dtype=np.uint8)
    norm = np.clip((power - lo) / (hi - lo), 0, 1)
    return (norm * 255).astype(np.uint8)
