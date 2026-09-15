#!/usr/bin/env python
"""Phase 2 of the fast driver: generate every manifest image with SDXL+IP-Adapter
resident (one load for the whole run). See spec 2026-09-03 section 3.2.

Usage:
    python -m graft.worker_generate_many <cfg_yaml> <manifest_json> <kg_dir> \
        <b2_attrs_json> <pool_json> <out_dir>
"""
from __future__ import annotations

import json
import os
import sys

from graft import env

env.setup()

from graft.eval_common import exemplar_for, id_to_filename, prompt_for

_NEG = "monochrome, lowres, bad anatomy, worst quality, low quality"


def generate_plan(manifest, kg_by_species, b2_attrs, species_pool, out_dir, exists_fn):
    """Pure: resolve prompt+exemplar+out_path for each seed-row not on disk.
    b0/b2 collapse across ip_scale to the same out file, so a duplicate out_path
    seen earlier in this plan is skipped too. ours/ours_notree/b1 all resolve
    their exemplar from the KG's stored ref_embeddings (see eval_common.exemplar_for)."""
    plan, planned_paths = [], set()
    for row in manifest:
        sp, method = row["species"], row["method"]
        out_path = os.path.join(out_dir, sp, "gen_fast", id_to_filename(row["image_id"]))
        if exists_fn(out_path) or out_path in planned_paths:
            continue
        kg = kg_by_species.get(sp)
        prompt = prompt_for(method, sp, kg, b2_attrs.get(sp, []),
                            neutralize=(row["name_mode"] == "neutral"))
        ip_path = exemplar_for(method, kg)
        planned_paths.add(out_path)
        plan.append({"image_id": row["image_id"], "prompt": prompt, "ip_path": ip_path,
                     "seed": row["seed"], "ip_scale": row["ip_scale"],
                     "out_path": out_path, "method": method})
    return plan


def main(argv: list[str]) -> int:
    if len(argv) != 6:
        print("usage: worker_generate_many.py <cfg_yaml> <manifest_json> <kg_dir> "
              "<b2_attrs_json> <pool_json> <out_dir>", file=sys.stderr)
        return 2
    cfg_yaml, manifest_json, kg_dir, b2_json, pool_json, out_dir = argv

    from PIL import Image

    from graft.config import GraftConfig
    from graft.models import Models
    from graft.schema import ConceptKG

    cfg = GraftConfig.from_yaml(cfg_yaml)
    manifest = json.load(open(manifest_json))
    b2_attrs = json.load(open(b2_json))
    pool = json.load(open(pool_json))
    species = sorted({r["species"] for r in manifest})
    kg_by_species = {
        s: ConceptKG.from_json(os.path.join(kg_dir, s, "kg.json"))
        for s in species
        if os.path.exists(os.path.join(kg_dir, s, "kg.json"))
    }
    models = Models(cfg)

    plan = generate_plan(manifest, kg_by_species, b2_attrs, pool, out_dir,
                         exists_fn=os.path.exists)
    for item in plan:
        os.makedirs(os.path.dirname(item["out_path"]), exist_ok=True)
        ip_image = Image.open(item["ip_path"]).convert("RGB") if item["ip_path"] else None
        if item["ip_scale"] is not None:
            models.generator.set_scale(item["ip_scale"])
        try:
            img = models.generator.generate(prompt=item["prompt"], ip_image=ip_image,
                                            seed=item["seed"], negative=_NEG, steps=50)
        except Exception as exc:  # noqa: BLE001 -- one bad cell must not kill the phase
            sys.stderr.write(f"[gen] {item['image_id']}: FAILED {type(exc).__name__}: {exc}\n")
            continue
        img.save(item["out_path"])
        print(f"[gen] {item['image_id']} -> {item['out_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
