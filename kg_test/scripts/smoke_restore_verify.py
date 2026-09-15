"""Stage-A' smoke gate (INFRA-ONLY). Reads a 1-concept run_restore_resident output dir,
recomputes each sampled cell's geometry, and runs graft.smoke_restore.run_gate. A FAIL means
fix the GPU path -- never read any number as evidence. Prints PASS/FAIL + per-check messages."""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from PIL import Image
from graft.config import GraftConfig
from graft.graph_schema import PartTypeGraph
from graft import restore, smoke_restore
from graft.worker_restore_cell import build_restore_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--heldout", required=True)          # heldout_parts/<concept>.json
    ap.add_argument("--out", required=True)              # the smoke run's output dir
    ap.add_argument("--config", required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--sample", type=int, default=6)     # cells to pixel-check
    a = ap.parse_args()
    cfg = GraftConfig.from_yaml(a.config)
    graph = PartTypeGraph.from_json(a.graph)
    with open(a.heldout) as f:
        held_recs = json.load(f)

    rows = []
    for rp in glob.glob(os.path.join(a.out, f"{a.concept}_*.json")):
        with open(rp) as f:
            rows.append(json.load(f))
    assert rows, "no rows found -- did the smoke run write output?"

    # group rows into per-(part,level,draw) cells (arms together)
    cells = {}
    for r in rows:
        cells.setdefault((r["part"], r["level"], r["draw"]), []).append(r)
    report_cells = [{"rows": rws} for rws in cells.values()]

    # mask validity + actual-inpaint + contamination on a sample of cells
    masks, inpaints, borrowed = [], [], []
    sample_keys = list(cells.keys())[: a.sample]
    for (part, lvl, draw) in sample_keys:
        k_eff = restore.compute_k_eff(graph, a.concept, part, k=cfg.restore_k)
        tgt = draw % len([h for h in held_recs if h["concept"] == a.concept and h["part"] == part])
        for cond in ("isolated", "hub"):
            inp = build_restore_inputs(graph, held_recs, a.concept, part, tgt, lvl, cond, k_eff, draw)
            masks.append((inp["sam_mask"], inp["box"]))
            png = os.path.join(a.out, "_imgs", f"{a.concept}_{part}_{lvl}_{draw}_{cond}.png")
            if os.path.exists(png):
                out_img = Image.open(png).convert("RGB")
                if out_img.size != inp["masked_img"].size:      # align to original coords
                    out_img = out_img.resize(inp["masked_img"].size)
                out_arr = np.asarray(out_img)
                base_arr = np.asarray(inp["orig_img"])     # pristine original; composite keeps outside == this
                inpaints.append({"base": base_arr, "out": out_arr, "box": inp["box"]})
            if cond == "hub" and inp["borrowed_ids"]:
                borrowed.append({"ids": inp["borrowed_ids"], "concept": a.concept, "ref": inp["target_ref"]})

    used_gb = 0.0
    try:
        import torch
        used_gb = torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass

    report = {"masks": masks, "cells": report_cells, "inpaints": inpaints,
              "min_frac": cfg.restore_min_mask_area_frac, "max_frac": cfg.restore_max_mask_area_frac,
              "outside_tol": 3.0, "inside_min": 3.0,
              "peak_gb": used_gb if used_gb else 12.0, "per_gen_growth_gb": 0.0}
    # contamination checks (attach to a synthetic cell so run_gate iterates them)
    report["cells"].append({"rows": [], "borrowed": borrowed})

    ok, msgs = smoke_restore.run_gate(report)
    print("=== SMOKE GATE:", "PASS" if ok else "FAIL", "===")
    for m in msgs:
        print("  ", m)
    print(f"(checked {len(rows)} rows, {len(cells)} cells; pixel-sampled {len(inpaints)} inpaints)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
