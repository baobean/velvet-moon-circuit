"""Gather a species' part-crop images from the MMKG store (no FAISS, no NN)."""
from __future__ import annotations

from PIL import Image


def select_part_images(store, global_id, *, parts=None):
    """One RGB image per part_type for a species. Empty dict if absent.

    First crop of each part_type wins (deterministic by record list order).
    `parts` optionally restricts to a subset of part_type names.
    """
    rec = store.get(global_id)
    if rec is None:
        return {}
    out: dict[str, Image.Image] = {}
    for crop in rec.get("part_crops", []):
        pt = crop["part_type"]
        if parts is not None and pt not in parts:
            continue
        if pt in out:
            continue
        out[pt] = Image.open(crop["image_path"]).convert("RGB")
    return out
