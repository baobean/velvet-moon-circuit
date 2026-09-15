"""Resident, single-process runner for the full ~1,440-cell restore matrix. Loads
SDXL-inpaint+IP once (Phase A), unloads, loads the embedders once (Phase B) -- the
two-phase split the Stage-A run needed to fit 24GB. Reuses build_restore_inputs so
the resident path matches the per-cell worker exactly. Resume-safe via existing PNG/row."""
from __future__ import annotations
import argparse, json, os
from PIL import Image
from graft.config import GraftConfig
from graft.graph_schema import PartTypeGraph
from graft import restore
from graft.worker_restore_cell import build_restore_inputs, restore_row, _score_restored, _inpaint
from graft.restore_analysis import recovery_curve, make_or_break, borrowing_helps, stratified_delta


def _own_count(graph, concept, part):
    return len(restore.own_part_crops(graph, concept, part))


def _targets(held_recs, concept, part):
    return [h for h in held_recs if h["concept"] == concept and h["part"] == part]


def iter_restore_cells(graph, held_recs, concepts, levels, n_draws, conditions, k):
    for c in concepts:
        parts_here = sorted({i.part for i in graph.part_instances if i.concept == c})
        for p in parts_here:
            k_eff = restore.compute_k_eff(graph, c, p, k)
            if k_eff == 0:
                continue
            n_tgt = len(_targets(held_recs, c, p))
            if n_tgt == 0:
                continue
            own_n = _own_count(graph, c, p)
            eff_levels = sorted({min(l, own_n) for l in levels})
            for lvl in eff_levels:
                for d in range(n_draws):
                    tgt = d % n_tgt
                    for cond in conditions:
                        yield (c, p, lvl, d, cond, tgt, k_eff)


def _tag(c, p, lvl, d, cond):
    return f"{c}_{p}_{lvl}_{d}_{cond}"


def main():
    ap = argparse.ArgumentParser()
    for f in ("graph", "heldout-dir", "out", "config"):
        ap.add_argument(f"--{f}", required=True)
    ap.add_argument("--concepts", default="", help="optional comma list to restrict (smoke)")
    a = ap.parse_args()
    cfg = GraftConfig.from_yaml(a.config)
    graph = PartTypeGraph.from_json(a.graph)
    concepts = sorted({i.concept for i in graph.part_instances})
    if a.concepts:
        want = {c.strip() for c in a.concepts.split(",")}
        concepts = [c for c in concepts if c in want]
    held_recs = []
    for c in concepts:
        hp = os.path.join(a.heldout_dir, f"{c}.json")
        if os.path.exists(hp):
            with open(hp) as f:
                held_recs.extend(json.load(f))
    conditions = ["isolated", "random", "rawnn", "hub"]
    cells = list(iter_restore_cells(graph, held_recs, concepts, list(cfg.restore_build_levels),
                                    cfg.restore_n_draws, conditions, cfg.restore_k))
    os.makedirs(a.out, exist_ok=True)
    img_dir = os.path.join(a.out, "_imgs"); os.makedirs(img_dir, exist_ok=True)
    from graft.models import Models
    models = Models(cfg)

    # ---- Phase A: inpaint (generator resident only) ----
    gen = models.generator
    for i, (c, p, lvl, d, cond, tgt, k_eff) in enumerate(cells):
        png = os.path.join(img_dir, _tag(c, p, lvl, d, cond) + ".png")
        if os.path.exists(png):
            continue
        inp = build_restore_inputs(graph, held_recs, c, p, tgt, lvl, cond, k_eff, d)
        img = _inpaint(gen, cfg, "a photo of a plant " + p, inp)
        img.save(png)
        import torch; torch.cuda.empty_cache()
        print(f"[inpaint {i+1}/{len(cells)}] {_tag(c,p,lvl,d,cond)}", flush=True)
    models.unload("generator")

    # ---- Phase B: score (embedders resident only) ----
    rows = []
    for i, (c, p, lvl, d, cond, tgt, k_eff) in enumerate(cells):
        rowp = os.path.join(a.out, _tag(c, p, lvl, d, cond) + ".json")
        if os.path.exists(rowp):
            with open(rowp) as f:
                rows.append(json.load(f))
            continue
        inp = build_restore_inputs(graph, held_recs, c, p, tgt, lvl, cond, k_eff, d)
        img = Image.open(os.path.join(img_dir, _tag(c, p, lvl, d, cond) + ".png")).convert("RGB")
        scores = _score_restored(img, inp["box"], inp["sam_mask"], inp["scoring_crops"], models)
        row = restore_row(c, p, lvl, cond, d, inp["k_eff"], scores)
        with open(rowp, "w") as f:
            json.dump(row, f)
        rows.append(row)
        print(f"[score {i+1}/{len(cells)}] {_tag(c,p,lvl,d,cond)} dino={scores['dino']:.3f}", flush=True)

    # ---- readout ----
    with open(os.path.join(a.out, "recovery.json"), "w") as f:
        json.dump(recovery_curve(rows, "dino"), f, indent=2)
    with open(os.path.join(a.out, "hub_vs_rawnn.json"), "w") as f:
        json.dump(make_or_break(rows, metric="dino"), f, indent=2)
    with open(os.path.join(a.out, "stratified.json"), "w") as f:
        json.dump({"hub_vs_rawnn": stratified_delta(rows, held_recs, "hub", "rawnn"),
                   "borrowing_helps": borrowing_helps(rows)}, f, indent=2)
    print(f"DONE {len(rows)} rows -> {a.out}/recovery.json,hub_vs_rawnn.json,stratified.json", flush=True)


if __name__ == "__main__":
    main()
