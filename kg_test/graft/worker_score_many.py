#!/usr/bin/env python
"""Phase 3 of the fast driver: score every generated image with ONE embedder
resident. Invoked once per embedder (dino|siglip|clip|vlm), so no process ever
churns two models -- the safety envelope the passing gpu tests validate.

Usage:
    python -m graft.worker_score_many <embedder> <cfg_yaml> <manifest_json> \
        <kg_dir> <heldout_json> <out_dir> <out_table_json>
"""
from __future__ import annotations

import json
import os
import sys

from graft import env

env.setup()

from graft.eval_common import id_to_filename

_EMBEDDERS = ("dino", "siglip", "clip", "vlm")


def score_plan(manifest, out_dir, exists_fn):
    plan = []
    for row in manifest:
        sp = row["species"]
        img_path = os.path.join(out_dir, sp, "gen_fast", id_to_filename(row["image_id"]))
        if exists_fn(img_path):
            plan.append({"image_id": row["image_id"], "species": sp, "img_path": img_path})
    return plan


def main(argv: list[str]) -> int:
    if len(argv) != 7:
        print("usage: worker_score_many.py <embedder> <cfg_yaml> <manifest_json> "
              "<kg_dir> <heldout_json> <out_dir> <out_table_json>", file=sys.stderr)
        return 2
    embedder, cfg_yaml, manifest_json, kg_dir, heldout_json, out_dir, out_table = argv

    if embedder not in _EMBEDDERS:
        print(f"unknown embedder '{embedder}' (expected one of {_EMBEDDERS})", file=sys.stderr)
        return 2

    from PIL import Image

    from graft.config import GraftConfig
    from graft.metrics import attribute_accuracy, clip_t, image_fidelity
    from graft.models import Models
    from graft.schema import ConceptKG

    cfg = GraftConfig.from_yaml(cfg_yaml)
    manifest = json.load(open(manifest_json))
    heldout = json.load(open(heldout_json))  # {species: [heldout paths]}
    plan = score_plan(manifest, out_dir, os.path.exists)
    species = sorted({p["species"] for p in plan})
    kg_by = {s: ConceptKG.from_json(os.path.join(kg_dir, s, "kg.json")) for s in species}
    heldout_imgs = {s: [Image.open(p).convert("RGB") for p in heldout[s]] for s in species}
    models = Models(cfg)

    table: dict = {}
    for item in plan:
        sp = item["species"]
        try:
            img = Image.open(item["img_path"]).convert("RGB")
            if embedder == "dino":
                table[item["image_id"]] = {"dino": image_fidelity(img, heldout_imgs[sp], models.dino)}
            elif embedder == "siglip":
                table[item["image_id"]] = {"siglip2": image_fidelity(img, heldout_imgs[sp], models.siglip)}
            elif embedder == "clip":
                table[item["image_id"]] = {
                    "clip_i": image_fidelity(img, heldout_imgs[sp], models.clip),
                    "clip_t": clip_t(img, f"a photo of a {sp}", models.clip),
                }
            elif embedder == "vlm":
                table[item["image_id"]] = {
                    "attribute_accuracy": attribute_accuracy(img, kg_by[sp].attribute_texts(), models.vlm)
                }
        except Exception as exc:  # noqa: BLE001 -- one bad image must not kill the phase
            sys.stderr.write(f"[score:{embedder}] {item['image_id']}: FAILED {type(exc).__name__}: {exc}\n")
            continue
    with open(out_table, "w") as f:
        json.dump(table, f, indent=2)
    print(f"[score:{embedder}] wrote {len(table)} rows -> {out_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
