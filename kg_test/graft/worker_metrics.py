#!/usr/bin/env python
"""Subprocess worker: score one generated image against a concept's KG and
held-out real references, then exit. See refine.py's module docstring for
why every GPU-touching phase runs in its own subprocess.

Usage:
    python -m graft.worker_metrics <image_path> <kg_json> <species> \
        <cfg_yaml> <out_metrics_json> <heldout_ref>...

Writes {"dino", "siglip2", "clip_i", "attribute_accuracy", "clip_t"} to
<out_metrics_json> and exits 0. Exit 3 on OOM, 1 on any other error.
"""
from __future__ import annotations

import json
import sys

from graft import env

env.setup()


def main(argv: list[str]) -> int:
    if len(argv) < 6:
        print(
            "usage: worker_metrics.py <image_path> <kg_json> <species> "
            "<cfg_yaml> <out_metrics_json> <heldout_ref>...",
            file=sys.stderr,
        )
        return 2
    image_path, kg_json, species, cfg_yaml, out_json, *heldout_refs = argv

    from PIL import Image

    from graft.config import GraftConfig
    from graft.metrics import attribute_accuracy, clip_t, image_fidelity
    from graft.models import Models
    from graft.schema import ConceptKG

    cfg = GraftConfig.from_yaml(cfg_yaml)
    kg = ConceptKG.from_json(kg_json)
    models = Models(cfg)

    try:
        image = Image.open(image_path).convert("RGB")
        heldout_images = [Image.open(p).convert("RGB") for p in heldout_refs]
        row = {}
        row["dino"] = image_fidelity(image, heldout_images, models.dino)
        models.unload("dino")
        row["siglip2"] = image_fidelity(image, heldout_images, models.siglip)
        models.unload("siglip")
        row["clip_i"] = image_fidelity(image, heldout_images, models.clip)
        row["clip_t"] = clip_t(image, f"a photo of a {species}", models.clip)
        models.unload("clip")
        row["attribute_accuracy"] = attribute_accuracy(image, kg.attribute_texts(), models.vlm)
    except Exception as exc:  # noqa: BLE001 -- classify OOM for the caller
        if "out of memory" in str(exc).lower() or "OutOfMemoryError" in type(exc).__name__:
            print(f"OOM in worker_metrics: {exc}", file=sys.stderr)
            return 3
        raise

    with open(out_json, "w") as f:
        json.dump(row, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
