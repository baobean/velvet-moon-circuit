"""Compose one reference canvas from several part crops (the mechanism core).

v1 = deterministic near-square grid, one part per cell (fit-inside, centred on
a neutral-grey cell so a cropped part is not read as an object edge). Relation-
driven placement is a Phase-1 ablation, deliberately NOT done here."""
from __future__ import annotations

import math

from PIL import Image

_GREY = (127, 127, 127)


def compose_reference(part_images, *, tile: int = 384, order=None):
    if not part_images:
        raise ValueError("compose_reference: no part images supplied")
    keys = list(order) if order is not None else sorted(part_images)
    n = len(keys)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    canvas = Image.new("RGB", (cols * tile, rows * tile), _GREY)
    for i, k in enumerate(keys):
        img = part_images[k].convert("RGB")
        w, h = img.size
        s = min(tile / w, tile / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        cell = img.resize((nw, nh), Image.LANCZOS)
        cx, cy = (i % cols) * tile, (i // cols) * tile
        canvas.paste(cell, (cx + (tile - nw) // 2, cy + (tile - nh) // 2))
    return canvas
