import numpy as np
import pytest
from PIL import Image

from ragregen import composite


def _mask(size=(64, 64), box=(16, 16, 48, 48)):
    m = Image.new("L", size, 0)
    m.paste(255, box)
    return m


def test_feather_alpha_produces_three_zones():
    a = composite.feather_alpha(_mask(), (64, 64), feather_px=4)
    assert a[0, 0] == 0.0, "outside must be exactly zero"
    assert a[32, 32] == pytest.approx(1.0), "interior must be exactly one"
    band = a[(a > 0.0) & (a < 1.0)]
    assert band.size > 0, "a feathered edge must exist between the two"


def test_feather_alpha_with_no_feather_is_hard():
    a = composite.feather_alpha(_mask(), (64, 64), feather_px=0)
    assert set(np.unique(a)) == {0.0, 1.0}


def test_composite_back_leaves_outside_pixels_bit_identical():
    original = Image.new("RGB", (64, 64), (10, 20, 30))
    raw = Image.new("RGB", (64, 64), (200, 0, 0))
    out, alpha = composite.composite_back(original, raw, _mask(), feather_px=0)

    o = np.asarray(original)
    got = np.asarray(out)
    outside = alpha == 0.0
    assert (got[outside] == o[outside]).all()
    assert tuple(got[32, 32]) == (200, 0, 0)


def test_composite_back_resizes_a_raw_of_a_different_size():
    """Kontext returns a preferred resolution, almost never the input's."""
    original = Image.new("RGB", (64, 64), (10, 20, 30))
    raw = Image.new("RGB", (128, 128), (200, 0, 0))
    out, _ = composite.composite_back(original, raw, _mask(), feather_px=0)
    assert out.size == (64, 64)


class _FakeMaskResult:
    """Stands in for mask.MaskResult; only `.mask` is read here."""

    def __init__(self, arr):
        self.mask = arr


def test_cutout_is_rgba_cropped_to_the_silhouette_box():
    ref = Image.new("RGB", (64, 64), (5, 6, 7))
    arr = np.zeros((64, 64), dtype=float)
    arr[16:48, 20:40] = 1.0
    out = composite.cutout_from(ref, _FakeMaskResult(arr))
    assert out.mode == "RGBA"
    assert out.size == (20, 32), "cropped to the silhouette bbox, not the frame"


def test_cutout_alpha_is_the_silhouette_not_a_rectangle():
    ref = Image.new("RGB", (32, 32), (5, 6, 7))
    arr = np.zeros((32, 32), dtype=float)
    arr[8:24, 8:24] = 1.0
    arr[8:12, 8:12] = 0.0            # bite a corner out of the square
    out = composite.cutout_from(ref, _FakeMaskResult(arr))
    alpha = np.asarray(out)[..., 3]
    assert alpha[0, 0] == 0, "the bitten corner must stay transparent"
    assert alpha[-1, -1] == 255


def test_cutout_of_an_empty_mask_is_none():
    ref = Image.new("RGB", (32, 32))
    assert composite.cutout_from(ref, _FakeMaskResult(np.zeros((32, 32)))) is None


def test_cutout_of_no_mask_result_is_none():
    assert composite.cutout_from(Image.new("RGB", (8, 8)), None) is None
