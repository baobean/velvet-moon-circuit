#!/usr/bin/env python
"""Fast eval driver (spec 2026-09-03): four batched phases, ~14 model loads
instead of ~880. Slow run_eval.py stays the trusted reference; this reuses its
summarize + analysis.build_analysis so outputs match in shape.

Phases: (1) build KGs [worker_build_kg] + recall b2 [worker_recall_b2];
(2) generate all [worker_generate_many]; (3) score all, one embedder each
[worker_score_many]; (4) assemble (pure)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from graft import env


def assemble(manifest, tables, n_heldout_by_species, use_part_tree):
    from graft.analysis import build_analysis
    from graft.eval_common import rows_from_tables
    from graft.run_eval import summarize

    rows, missing = rows_from_tables(manifest, tables, n_heldout_by_species, use_part_tree)
    return {
        "rows": rows,
        "summary": summarize(rows),
        "expected_cells": len({(r["species"], r["method"], r["name_mode"], r["ip_scale"])
                               for r in manifest}),
        "missing_cells": missing,
        "analysis": build_analysis(rows),
    }


def _merge_tables(paths):
    merged: dict = {}
    for p in paths:
        if os.path.exists(p):
            for iid, m in json.load(open(p)).items():
                merged.setdefault(iid, {}).update(m)
    return merged


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--species", default=None, help="comma list; bypasses --select-by")
    ap.add_argument("--n-species", type=int, default=8)
    ap.add_argument("--select-by", choices=("alpha", "unique"), default="unique")
    ap.add_argument("--methods", default="ours,b0,b1,b2")
    ap.add_argument("--ip-scales", default="0.6")
    ap.add_argument("--name-modes", default="neutral")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    env.setup()
    from graft.config import GraftConfig
    from graft.dataset import list_species, load_species, split_refs
    from graft.eval_common import build_manifest, kg_is_usable
    from graft.pipeline import _build_kg_subprocess
    from graft.run_eval import _markdown_table, rank_species_by_unique_count

    methods = [m for m in args.methods.split(",") if m]
    name_modes = [m.strip() for m in args.name_modes.split(",") if m.strip()]
    ip_scales = [float(s) for s in args.ip_scales.split(",") if s.strip()]
    bad = set(name_modes) - {"neutral", "named"}
    if bad:
        ap.error(f"--name-modes: unknown {sorted(bad)}")
    if not ip_scales or not name_modes or not methods:
        ap.error("--methods/--ip-scales/--name-modes must be non-empty")

    cfg = GraftConfig.from_yaml(args.config) if os.path.exists(args.config) else GraftConfig()
    if args.species:
        names = [s.strip() for s in args.species.split(",") if s.strip()]
    elif args.select_by == "unique":
        names = rank_species_by_unique_count(args.root, limit=args.n_species)
    else:
        names = list_species(args.root)[: args.n_species]

    species_list = [load_species(args.root, n) for n in names]
    build_cfg = __import__("dataclasses").replace(cfg, neutralize_name=True)
    out_dir = args.out or os.path.join(cfg.outputs_dir, "eval_fast")
    os.makedirs(out_dir, exist_ok=True)
    work = os.path.join(out_dir, "_work")
    os.makedirs(work, exist_ok=True)

    # split + skip species that cannot form build+heldout
    n_heldout, pool, heldout, usable = {}, {}, {}, []
    for sp in species_list:
        b, h = split_refs(sp, k_build=cfg.k_build_refs, seed=0)
        if not b or not h:
            print(f"[fast] {sp.name}: no build+heldout split, skipping")
            continue
        usable.append(sp.name)
        n_heldout[sp.name], pool[sp.name], heldout[sp.name] = len(h), b, h

    # ---- Phase 1: build KGs (rebuild stale) + recall b2 ----
    cfg_yaml = os.path.join(work, "cfg.yaml"); cfg.to_yaml(cfg_yaml)
    kg_dir = cfg.outputs_dir
    for name in usable:
        kg_path = os.path.join(kg_dir, name, "kg.json")
        if not kg_is_usable(kg_path):
            print(f"[fast] {name}: building KG")
            _build_kg_subprocess(name, pool[name], build_cfg, os.path.join(kg_dir, name))
    b2_json = os.path.join(work, "b2_attrs.json")
    if "b2" in methods:
        subprocess.run([sys.executable, "-m", "graft.worker_recall_b2", cfg_yaml, b2_json, *usable], check=True)
    else:
        json.dump({}, open(b2_json, "w"))

    # ---- manifest + side inputs ----
    manifest = build_manifest(usable, methods, name_modes, ip_scales, args.seeds)
    man_json = os.path.join(work, "manifest.json"); json.dump(manifest, open(man_json, "w"))
    pool_json = os.path.join(work, "pool.json"); json.dump(pool, open(pool_json, "w"))
    heldout_json = os.path.join(work, "heldout.json"); json.dump(heldout, open(heldout_json, "w"))

    # ---- Phase 2: generate all (SDXL resident) ----
    subprocess.run([sys.executable, "-m", "graft.worker_generate_many",
                    cfg_yaml, man_json, kg_dir, b2_json, pool_json, kg_dir], check=True)

    # ---- Phase 3: score all (one embedder each) ----
    embedders = ["dino", "siglip", "clip", "vlm"]
    table_paths = []
    for emb in embedders:
        tp = os.path.join(work, f"table_{emb}.json")
        subprocess.run([sys.executable, "-m", "graft.worker_score_many",
                        emb, cfg_yaml, man_json, kg_dir, heldout_json, kg_dir, tp], check=True)
        table_paths.append(tp)

    # ---- Phase 4: assemble ----
    tables = _merge_tables(table_paths)
    result = assemble(manifest, tables, n_heldout, use_part_tree=False)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({k: result[k] for k in ("rows", "summary", "expected_cells", "missing_cells")}, f, indent=2)
    with open(os.path.join(out_dir, "analysis.json"), "w") as f:
        json.dump(result["analysis"], f, indent=2)
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(_markdown_table(result["summary"]) + "\n")
    if result["missing_cells"]:
        print(f"[fast] INCOMPLETE: {len(result['missing_cells'])}/{result['expected_cells']} cells missing",
              file=sys.stderr)
    print(_markdown_table(result["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
