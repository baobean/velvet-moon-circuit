#!/usr/bin/env python
"""Evaluation driver: for each species, build the MMKG on its build_refs,
run each requested method to one image, score every image against the
species' held-out real refs, and write a per-method summary table."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from graft import env

#: Methods swept over the full name_mode x ip_scale grid; the rest (b0, b2)
#: ignore ip_scale, so they run once per name_mode (spec section 10, stage 3).
SWEPT_METHODS = ("ours", "ours_notree", "b1")

#: Numeric row fields that tag a matrix cell rather than score it -- averaging
#: them across a method's rows would be meaningless, so summarize skips them.
NON_METRIC_KEYS = frozenset({"method", "ip_scale", "n_heldout"})


def summarize(rows: List[dict]) -> Dict[str, Dict[str, float]]:
    """Group rows by 'method' and mean every other numeric field, except the
    matrix tags in NON_METRIC_KEYS (they identify a cell, they are not scores)."""
    groups: Dict[str, List[dict]] = {}
    for row in rows:
        groups.setdefault(row["method"], []).append(row)

    result: Dict[str, Dict[str, float]] = {}
    for method, group_rows in groups.items():
        keys = set()
        for r in group_rows:
            for k, v in r.items():
                if k not in NON_METRIC_KEYS and isinstance(v, (int, float)):
                    keys.add(k)
        result[method] = {
            k: sum(r[k] for r in group_rows if k in r) / sum(1 for r in group_rows if k in r)
            for k in keys
        }
    return result


def _markdown_table(summary: Dict[str, Dict[str, float]]) -> str:
    methods = sorted(summary)
    if not methods:
        return "(no results)"
    metric_names = sorted({k for m in summary.values() for k in m})
    header = "| method | " + " | ".join(metric_names) + " |"
    sep = "|---" * (len(metric_names) + 1) + "|"
    lines = [header, sep]
    for method in methods:
        row = [f"{summary[method].get(k, float('nan')):.4f}" for k in metric_names]
        lines.append(f"| {method} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def rank_species_by_unique_count(root: str, limit: int = None) -> List[str]:
    """Species names under `root` ranked by deduped unique-image count, DESC.

    Replaces Phase 1's alphabetical selection (spec section 10): the validation
    subset should span a range of unique-image counts, not whichever species
    sort first. Species with fewer than 2 unique images cannot form a
    build+heldout split at all and are dropped.

    COST: this calls `load_species` on *every* species directory, and
    `load_species` content-hashes every file it finds. On this NAS that is the
    expensive path (see graft/dataset.py's module docstring: a full 132k-file
    scan stalled past 9 minutes, pure I/O wait). Call it once per run and cache
    the answer; do not call it in a loop.
    """
    from graft.dataset import list_species, load_species

    counts = []
    for name in list_species(root):
        n = len(load_species(root, name).images)
        if n >= 2:
            counts.append((n, name))
    # -n for DESC by count, name ASC as a deterministic tie-break.
    ranked = [name for _n, name in sorted(counts, key=lambda t: (-t[0], t[1]))]
    return ranked[:limit] if limit is not None else ranked


def _kg_is_usable(kg_path: str) -> bool:
    """True when `kg_path` holds a KG this sprint can actually generate from.

    A Phase-1 kg.json predates `ref_embeddings`, so reusing it sends
    `generate.select_exemplar` into `medoid_index([])`; the resulting worker
    crash is (correctly) treated as "no signal, try the next seed", exhausts
    the refine budget, and silently drops the GRAFT row from the readout.
    Treat such a KG as ABSENT so it is rebuilt. Torch-free by design -- this
    process never imports torch (see `evaluate`)."""
    from graft.schema import ConceptKG

    try:
        kg = ConceptKG.from_json(kg_path)
    except (OSError, ValueError, KeyError):
        return False
    return bool(kg.ref_embeddings) and len(kg.ref_embeddings) == len(kg.ref_paths)


def evaluate(
    species_list,
    methods: List[str],
    models,
    cfg,
    *,
    name_modes: List[str] = ("neutral",),
    ip_scales: List[float] = (0.6,),
    resume: bool = False,
) -> dict:
    """Pure orchestrator: every GPU-touching phase (build_kg, each method's
    generation, metrics scoring) runs in its own subprocess via
    worker_build_kg.py / worker_baseline.py / worker_metrics.py -- see
    refine.py's module docstring for why. This process never imports torch
    or touches the GPU itself, so a crash in any one (species, method) case
    can be logged and skipped without losing the rest of the run.
    `models` is accepted for call-site compatibility but unused.

    Runs the sprint matrix (spec section 10, stage 3): every
    `name_mode x ip_scale` cell for the methods in SWEPT_METHODS, and one cell
    per `name_mode` for the rest (b0/b2 ignore `ip_scale`, but the prompt-head
    neutralization still applies to them). The MMKG is built ONCE per species
    with a fixed `neutralize_name=True` cfg and reused by every cell: the graph
    is name-blind regardless of which arm consumes it, so the named arm differs
    from the neutral arm only at generation time -- otherwise a named cell
    would be scored against a differently-built graph and the name delta would
    confound the build.

    `resume=True` makes the run restartable at cell granularity (the plan's
    "staged/resumable" mandate): when a cell's persisted
    `eval_images/{method}_{name_mode}_ip{ip_scale}.png` already exists, its
    GENERATION subprocess is skipped and `worker_metrics` is re-run on that
    existing image to rebuild the row. Metrics are re-computed rather than
    cached in a sidecar: scoring is cheap next to a 50-step SDXL refine loop,
    and it keeps a single source of truth for every row.

    Returns `{"rows", "summary", "expected_cells", "missing_cells"}`, where
    `missing_cells` lists every intended cell that produced no row (an OOM or
    crash hole) as `{species, method, name_mode, ip_scale}` -- `ip_scale` is
    None for the unswept methods, which are intended once per name_mode. A
    species skipped outright (no build+heldout split, or a failed KG build) is
    logged and contributes no cells at all. Without this, holes silently shrink
    the paired `n` and the readout still looks complete."""
    del models
    import dataclasses
    import shutil
    import subprocess
    import sys
    import tempfile

    from graft.dataset import split_refs
    from graft.pipeline import _build_kg_subprocess

    build_cfg = dataclasses.replace(cfg, neutralize_name=True)

    rows = []
    # A cell key is (species, method, name_mode, ip_scale); ip_scale is None for
    # the unswept methods, which run once per name_mode whatever the sweep is.
    expected_keys = set()
    produced_keys = set()
    for sp in species_list:
        build_refs, heldout_refs = split_refs(sp, k_build=cfg.k_build_refs, seed=0)
        if not build_refs or not heldout_refs:
            print(f"[eval] {sp.name}: cannot form build+heldout split (n_unique={len(sp.images)}), skipping")
            continue

        outputs_dir = os.path.join(cfg.outputs_dir, sp.name)
        os.makedirs(outputs_dir, exist_ok=True)
        kg_path = os.path.join(outputs_dir, "kg.json")
        kg_ok = os.path.exists(kg_path) and _kg_is_usable(kg_path)
        if os.path.exists(kg_path) and not kg_ok:
            print(f"[eval] {sp.name}: kg.json predates ref_embeddings — rebuilding")
        if not kg_ok:
            try:
                _build_kg_subprocess(sp.name, build_refs, build_cfg, outputs_dir)
            except RuntimeError as exc:
                print(f"[eval] {sp.name}: build_kg failed, skipping species:\n{exc}", file=sys.stderr)
                continue

        for name_mode in name_modes:
            for method in methods:
                if method in SWEPT_METHODS:
                    expected_keys.update((sp.name, method, name_mode, ip) for ip in ip_scales)
                else:
                    expected_keys.add((sp.name, method, name_mode, None))

        images_dir = os.path.join(outputs_dir, "eval_images")
        for name_mode in name_modes:
            # b0/b2 ignore ip_scale: run them at the first ip_scale of this
            # name_mode only, and skip them on the remaining sweep points.
            unswept_done = set()
            for ip_scale in ip_scales:
                case_cfg = dataclasses.replace(
                    cfg,
                    neutralize_name=(name_mode == "neutral"),
                    ip_scale=ip_scale,
                )
                for method in methods:
                    swept = method in SWEPT_METHODS
                    if not swept and method in unswept_done:
                        continue

                    cell = f"{method}/{name_mode}/ip{ip_scale}"
                    persisted_path = os.path.join(
                        images_dir, f"{method}_{name_mode}_ip{ip_scale}.png"
                    )
                    with tempfile.TemporaryDirectory(prefix="graft_eval_") as workdir:
                        cfg_yaml_path = os.path.join(workdir, "cfg.yaml")
                        case_cfg.to_yaml(cfg_yaml_path)
                        kg_arg = kg_path if method in ("ours", "ours_notree") else "-"

                        resumed = resume and os.path.exists(persisted_path)
                        if resumed:
                            image_path = persisted_path
                            print(f"[eval] {sp.name}/{cell}: resuming from {persisted_path}")
                        else:
                            image_path = os.path.join(workdir, "image.png")
                            gen = subprocess.run(
                                [
                                    sys.executable, "-m", "graft.worker_baseline",
                                    method, sp.name, cfg_yaml_path, kg_arg, image_path,
                                    *build_refs,
                                ],
                                capture_output=True, text=True,
                            )
                            if gen.returncode != 0:
                                print(
                                    f"[eval] {sp.name}/{cell}: generation failed "
                                    f"(code={gen.returncode}), skipping row:\n{gen.stderr}",
                                    file=sys.stderr,
                                )
                                continue

                        metrics_json_path = os.path.join(workdir, "metrics.json")
                        met = subprocess.run(
                            [
                                sys.executable, "-m", "graft.worker_metrics",
                                image_path, kg_path, sp.name, cfg_yaml_path,
                                metrics_json_path, *heldout_refs,
                            ],
                            capture_output=True, text=True,
                        )
                        if met.returncode != 0:
                            print(
                                f"[eval] {sp.name}/{cell}: metrics failed "
                                f"(code={met.returncode}), skipping row:\n{met.stderr}",
                                file=sys.stderr,
                            )
                            continue

                        with open(metrics_json_path) as f:
                            metrics_row = json.load(f)
                        row = {
                            "method": method,
                            "species": sp.name,
                            "name_mode": name_mode,
                            "ip_scale": ip_scale,
                            "n_heldout": len(heldout_refs),
                            **metrics_row,
                        }
                        rows.append(row)
                        produced_keys.add(
                            (sp.name, method, name_mode, ip_scale if swept else None)
                        )
                        # Only now is this unswept method really done: marking it
                        # before the cell ran meant one transient failure lost it
                        # for every remaining ip point of this name_mode.
                        if not swept:
                            unswept_done.add(method)

                        if not resumed:
                            os.makedirs(images_dir, exist_ok=True)
                            shutil.copy(image_path, persisted_path)
                        print(f"[eval] {sp.name}/{cell}: {row}")

    missing_cells = [
        {"species": s, "method": m, "name_mode": nm, "ip_scale": ip}
        for s, m, nm, ip in sorted(expected_keys - produced_keys)
    ]
    if missing_cells:
        print(
            f"[eval] INCOMPLETE MATRIX: {len(missing_cells)}/{len(expected_keys)} "
            f"cells produced no row (see results.json['missing_cells'])",
            file=sys.stderr,
        )
    return {
        "rows": rows,
        "summary": summarize(rows),
        "expected_cells": len(expected_keys),
        "missing_cells": missing_cells,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Treevill species-parent directory")
    ap.add_argument("--n-species", type=int, default=12)
    ap.add_argument("--methods", default="ours,ours_notree,b0,b1,b2")
    ap.add_argument(
        "--select-by", choices=("alpha", "unique"), default="unique",
        help="species subset rule: 'unique' = top-N by deduped unique-image "
             "count (hashes every species dir); 'alpha' = first N by name",
    )
    ap.add_argument(
        "--species", default=None,
        help="comma-separated species names to run EXACTLY (bypasses "
             "--select-by/--n-species and its multi-minute NAS hash)",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="skip the generation subprocess for any cell whose eval_images "
             "PNG already exists and re-score that image instead",
    )
    ap.add_argument("--ip-scales", default="0.4,0.6,0.8",
                    help="comma-separated IP-Adapter scales to sweep")
    ap.add_argument("--name-modes", default="neutral,named",
                    help="comma-separated arms: neutral and/or named")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    # Validate the sweep BEFORE any model/env setup: an unvalidated arm name
    # used to fail open to NAMED, and an empty list voided the whole matrix
    # while still exiting 0 -- either way only discoverable after a GPU run.
    methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    ip_scales = [float(s) for s in args.ip_scales.split(",") if s.strip()]
    name_modes = [s.strip() for s in args.name_modes.split(",") if s.strip()]
    if not methods:
        ap.error("--methods must list at least one method")
    if not ip_scales:
        ap.error("--ip-scales must list at least one value")
    if not name_modes:
        ap.error("--name-modes must list at least one arm")
    bad = set(name_modes) - {"neutral", "named"}
    if bad:
        ap.error(f"--name-modes: unknown arm(s) {sorted(bad)}; expected neutral and/or named")

    env.setup()
    from graft.analysis import build_analysis
    from graft.config import GraftConfig
    from graft.dataset import list_species, load_species

    cfg = GraftConfig.from_yaml(args.config) if os.path.exists(args.config) else GraftConfig()
    if args.species:
        # Explicit subset: never rank -- rank_species_by_unique_count hashes
        # EVERY species dir on this NAS (multi-minute; see its docstring).
        names = [s.strip() for s in args.species.split(",") if s.strip()]
        if not names:
            ap.error("--species must list at least one species name")
    elif args.select_by == "unique":
        names = rank_species_by_unique_count(args.root, limit=args.n_species)
    else:
        # Hash only the species actually selected (see graft/dataset.py's module
        # docstring: hashing the full corpus up front is impractical on this NAS).
        names = list_species(args.root)[: args.n_species]
    species_list = [load_species(args.root, name) for name in names]

    # evaluate() never touches the GPU itself -- every phase runs in its own
    # subprocess (see its docstring) -- so this process never needs Models.
    result = evaluate(
        species_list, methods, None, cfg,
        name_modes=name_modes, ip_scales=ip_scales, resume=args.resume,
    )

    out_dir = args.out or os.path.join(cfg.outputs_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "analysis.json"), "w") as f:
        json.dump(build_analysis(result["rows"]), f, indent=2)
    table = _markdown_table(result["summary"])
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(table + "\n")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
