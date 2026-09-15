"""Resident, single-process starvation runner -- the throughput path for the full
384-cell matrix. Loads SDXL+IP and the scorer embedders ONCE and keeps them resident
(the fast-eval-driver lesson: model reload, not compute, dominates), instead of the
per-cell subprocess in worker_starve_cell (~5.5 min/cell -> ~35 h for 384 cells).

Reuses graft.worker_starve_cell.build_cell_inputs verbatim, so the reviewed k_eff /
selection-only / part-restricted-pool / weighting logic lives in ONE place and this
runner is guaranteed to match the per-cell worker for any cell (same seed=0 gen).

Resume-safe: skips a cell whose row json already exists. Same row schema + filename as
the per-cell worker, so run_starvation.collect / starvation_analysis read it unchanged.
"""
from __future__ import annotations
import argparse
import json
import os

from PIL import Image

from graft.config import GraftConfig
from graft.dataset import load_species, split_refs
from graft.graph_schema import PartTypeGraph
from graft.schema import ConceptKG
from graft.starvation import iter_cells, CONDITIONS
from graft.starvation_analysis import recovery_curve, paired_delta
from graft.worker_starve_cell import build_cell_inputs
from graft.models import Models
from graft import metrics


def _cells():
    for (concept, level, draw) in iter_cells():
        for cond in CONDITIONS:
            yield (concept, level, draw, cond)


def main():
    ap = argparse.ArgumentParser()
    for f in ("graph", "kg-dir", "root", "out", "config"):
        ap.add_argument(f"--{f}", required=True)
    a = ap.parse_args()
    cfg = GraftConfig.from_yaml(a.config)
    graph = PartTypeGraph.from_json(a.graph)
    out = a.out
    os.makedirs(out, exist_ok=True)

    kg_cache, split_cache, held_cache = {}, {}, {}

    def kg_of(c):
        if c not in kg_cache:
            kg_cache[c] = ConceptKG.from_json(os.path.join(a.kg_dir, c, "kg.json"))
        return kg_cache[c]

    def split_of(c):
        if c not in split_cache:
            sp = load_species(a.root, c)
            split_cache[c] = split_refs(sp, k_build=cfg.k_build_refs, seed=0)
        return split_cache[c]

    def held_of(c):
        if c not in held_cache:
            _b, heldout = split_of(c)
            held_cache[c] = [Image.open(p).convert("RGB") for p in heldout]
        return held_cache[c]

    cells = list(_cells())
    img_dir = os.path.join(out, "_imgs")
    os.makedirs(img_dir, exist_ok=True)
    models = Models(cfg)

    # ---- Phase A: generate (only SDXL+IP resident; embedders NOT loaded) ----
    # Two-phase because SDXL+IP (~11.5GB) + dino+siglip+clip together exceed the 24GB
    # card (co-residency OOMs). Peak here is the generator alone.
    gen = models.generator
    for i, (concept, level, draw, cond) in enumerate(cells):
        png = os.path.join(img_dir, f"{concept}_{level}_{draw}_{cond}.png")
        if os.path.exists(png):
            continue
        build_refs, _heldout = split_of(concept)
        ref_imgs, weights, prompt = build_cell_inputs(
            graph, kg_of(concept), concept, level, draw, cond, build_refs)
        if ref_imgs:
            img = gen.generate_multi(prompt, ref_imgs, weights, seed=0)
        else:
            img = gen.generate(prompt, None, seed=0)
        img.save(png)
        import torch
        torch.cuda.empty_cache()                     # keep resident-loop VRAM flat
        print(f"[gen {i+1}/{len(cells)}] {concept} L{level} d{draw} {cond}", flush=True)
    models.unload("generator")                       # free SDXL+IP before loading scorers

    # ---- Phase B: score (only the embedders resident) ----
    done = 0
    for i, (concept, level, draw, cond) in enumerate(cells):
        rowp = os.path.join(out, f"{concept}_{level}_{draw}_{cond}.json")
        if os.path.exists(rowp):
            done += 1
            continue
        img = Image.open(os.path.join(img_dir, f"{concept}_{level}_{draw}_{cond}.png")).convert("RGB")
        hi = held_of(concept)
        row = {
            "concept": concept, "level": level, "draw": draw, "condition": cond,
            "dino": float(metrics.image_fidelity(img, hi, models.dino)),
            "siglip2": float(metrics.image_fidelity(img, hi, models.siglip)),
            "clip_i": float(metrics.image_fidelity(img, hi, models.clip)),
        }
        with open(rowp, "w") as f:
            json.dump(row, f)
        done += 1
        print(f"[score {done}/{len(cells)}] {concept} L{level} d{draw} {cond} dino={row['dino']:.3f}",
              flush=True)

    # ---- readout ----
    rows = []
    for (concept, level, draw, cond) in cells:
        rowp = os.path.join(out, f"{concept}_{level}_{draw}_{cond}.json")
        if os.path.exists(rowp):
            with open(rowp) as f:
                rows.append(json.load(f))
    with open(os.path.join(out, "recovery.json"), "w") as f:
        json.dump(recovery_curve(rows, "dino"), f, indent=2)
    with open(os.path.join(out, "hub_vs_rawnn.json"), "w") as f:
        json.dump(paired_delta(rows, "hub", "rawnn", level=1), f, indent=2)
    print(f"DONE {len(rows)} rows -> {out}/recovery.json, {out}/hub_vs_rawnn.json", flush=True)


if __name__ == "__main__":
    main()
