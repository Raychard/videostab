import numpy as np

from conftest import make_texture
from videostab.render import crop_and_resize, warp_frame


def test_constant_translation_matches_roll():
    img = make_texture(120, 160, seed=3)
    frame = np.stack([img] * 3, axis=-1)
    B = np.full((6, 8, 2), 0.0, np.float32)
    B[..., 0], B[..., 1] = 5.0, -3.0  # 内容右移 5, 上移 3
    out = warp_frame(frame, B)
    ref = np.roll(frame, ((-3), 5), axis=(0, 1))
    inner = (slice(10, 110), slice(10, 150))
    diff = np.abs(out[inner].astype(int) - ref[inner].astype(int))
    assert diff.mean() < 2.0


def test_warp_scale_applies_to_full_resolution():
    """代理系校正场按 scale 放大后作用于原分辨率."""
    img = make_texture(240, 320, seed=4)
    frame = np.stack([img] * 3, axis=-1)
    B = np.full((6, 8, 2), 0.0, np.float32)
    B[..., 0] = 2.0  # 代理系 2px, scale=2 -> 原图 4px
    out = warp_frame(frame, B, scale=2.0)
    ref = np.roll(frame, 4, axis=1)
    inner = (slice(20, 220), slice(20, 300))
    assert np.abs(out[inner].astype(int) - ref[inner].astype(int)).mean() < 2.0


def test_crop_and_resize_keeps_shape():
    frame = np.zeros((240, 320, 3), np.uint8)
    out = crop_and_resize(frame, 0.12)
    assert out.shape == frame.shape
    assert crop_and_resize(frame, 0.0).shape == frame.shape
