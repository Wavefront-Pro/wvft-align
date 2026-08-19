import numpy as np

from wvft_align.processing import (
    apply_gain,
    clamp_radius,
    compute_fft,
    crop_to_circle,
    fft_to_uint8,
    find_circle,
    scale_to_fit,
    to_gray,
)


def test_to_gray_passthrough():
    gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
    assert to_gray(gray).shape == (4, 4)


def test_to_gray_rgb():
    rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
    assert to_gray(rgb).shape == (8, 8)


def test_apply_gain_identity():
    frame = np.full((4, 4), 100, dtype=np.uint8)
    np.testing.assert_array_equal(apply_gain(frame, 1.0), frame)


def test_apply_gain_clips():
    frame = np.full((4, 4), 200, dtype=np.uint8)
    out = apply_gain(frame, 2.0)
    assert out.max() == 255


def test_clamp_radius_stays_inside():
    assert clamp_radius(10, 10, 100, 50, 50) == 10
    assert clamp_radius(0, 0, 5, 50, 50) == 1


def test_find_circle_on_disk():
    yy, xx = np.ogrid[:200, :200]
    disk = (xx - 100) ** 2 + (yy - 100) ** 2 < 60**2
    # Fringe-like texture so local variance lives inside the pupil, not just the rim.
    fringes = (128 + 80 * np.sin(xx / 4)).astype(np.uint8)
    frame = np.where(disk, fringes, 0).astype(np.uint8)
    cx, cy, dia = find_circle(frame)
    assert abs(cx - 100) < 15
    assert abs(cy - 100) < 15
    assert 80 < dia < 160


def test_crop_and_fft_shapes():
    frame = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    crop = crop_to_circle(frame, 64, 64, 80, 64)
    assert crop.shape == (64, 64)
    power = compute_fft(crop, size=64)
    assert power.shape == (64, 64)
    preview = fft_to_uint8(power)
    assert preview.dtype == np.uint8
    assert preview.shape == (64, 64)


def test_scale_to_fit_downscales_large_frame():
    frame = np.zeros((1944, 2950, 3), dtype=np.uint8)
    out, scale = scale_to_fit(frame, 540, 540)
    assert out.shape[0] <= 540 and out.shape[1] <= 540
    assert 0 < scale < 1
    assert abs(out.shape[1] / 2950 - scale) < 1e-2


def test_scale_to_fit_leaves_small_frame():
    frame = np.zeros((480, 640), dtype=np.uint8)
    out, scale = scale_to_fit(frame, 800, 800)
    assert out.shape == (480, 640)
    assert scale == 1.0
