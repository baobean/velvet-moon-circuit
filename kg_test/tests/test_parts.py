import numpy as np
from PIL import Image
from graft import parts


def test_clamp_box_clips_to_image():
    assert parts.clamp_box((-5, -5, 50, 40), 30, 20) == (0, 0, 30, 20)


def test_mask_area_fraction():
    mask = np.zeros((20, 30), dtype=bool)
    mask[0:10, 0:10] = True                      # 100 true px
    frac = parts.mask_area_fraction(mask, (0, 0, 10, 10))  # box area 100
    assert abs(frac - 1.0) < 1e-9


def test_mask_bbox_tight_and_none():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:9, 3:7] = True
    assert parts.mask_bbox(mask) == (3, 5, 7, 9)
    assert parts.mask_bbox(np.zeros((5, 5), bool)) is None


def test_box_mask_image_changes_only_inside_box():
    img = Image.new("RGB", (40, 30), (200, 100, 50))
    masked, bmask = parts.box_mask_image(img, (10, 5, 25, 20))
    a, m = np.asarray(img), np.asarray(masked)
    assert bmask.shape == (30, 40) and bmask[5:20, 10:25].all()
    assert (a[0:5, :] == m[0:5, :]).all()
    assert not (a[5:20, 10:25] == m[5:20, 10:25]).all()


def test_crop_region_size():
    img = Image.new("RGB", (40, 30))
    assert parts.crop_region(img, (10, 5, 25, 20)).size == (15, 15)
