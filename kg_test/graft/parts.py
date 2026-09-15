"""Pure geometry helpers for the part-level inpaint restore. No GPU/models here
so every function is unit-testable; the worker calls GroundingDINO/SAM and hands
their raw box/mask into these."""
from __future__ import annotations
import numpy as np
from PIL import Image


def clamp_box(box, w, h):
    x0, y0, x1, y1 = box
    x0 = max(0, min(int(round(x0)), w)); x1 = max(0, min(int(round(x1)), w))
    y0 = max(0, min(int(round(y0)), h)); y1 = max(0, min(int(round(y1)), h))
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def mask_area_fraction(mask, box):
    x0, y0, x1, y1 = box
    box_area = max(1, (x1 - x0) * (y1 - y0))
    return float(np.asarray(mask, dtype=bool).sum()) / float(box_area)


def mask_bbox(mask):
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def box_mask_image(image, box, fill=(127, 127, 127)):
    w, h = image.size
    x0, y0, x1, y1 = clamp_box(box, w, h)
    px = np.asarray(image.convert("RGB")).copy()
    px[y0:y1, x0:x1] = np.array(fill, dtype=px.dtype)
    masked = Image.fromarray(px)
    bmask = np.zeros((h, w), dtype=bool)
    bmask[y0:y1, x0:x1] = True
    return masked, bmask


def crop_region(image, box):
    w, h = image.size
    return image.crop(clamp_box(box, w, h))
