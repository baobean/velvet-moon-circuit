#!/usr/bin/env python
"""THE GATE (spec §10 step 0).

Drafts every case with FLUX.1-Kontext and writes labels.csv for hand-labelling.

If fewer than ~30% of drafts are labelled `fail`, FLUX already renders these
concepts and there is nothing for the pipeline to repair -- pick rarer concepts
before building anything else. The predecessor project assumed SDXL; FLUX is
much stronger on the long tail, so this must be measured, not assumed.

Usage: ./scripts/run.sh screen
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config, draft, env, trace, validate  # noqa: E402

FIELDNAMES = ["case_id", "prompt", "concept", "draft_path",
              "verdict", "verdict_identity", "notes"]


def print_gpu_info() -> None:
    gpu_id = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        print(f"[screen] GPU info unavailable: {exc}")
        return

    line = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    if not line:
        return
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 3:
        idx, name, mem = parts[0], parts[1], parts[2]
        print(f"[screen] GPU: {name} ({idx}) — {mem} MiB")
    else:
        print(f"[screen] GPU: {line}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET_PATH)
    ap.add_argument("--pipeline", type=Path, default=config.DEFAULT_PIPELINE_PATH)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="0 = all cases")
    args = ap.parse_args()
    print_gpu_info()

    ds = config.load_dataset(args.dataset)
    pipe_cfg = config.load_pipeline(args.pipeline)

    problems = validate.validate_dataset(ds) + validate.validate_disk()
    for p in problems:
        print(f"[{p.severity}] {p.code}: {p.message}")
    if any(p.severity == "error" for p in problems):
        print("\nFix the errors above before screening.")
        return 2

    cases = ds.cases[:args.limit] if args.limit else ds.cases
    if not cases:
        print(f"[screen] no cases to draft in {args.dataset}. "
              f"Add entries under `cases:` (or raise --limit).")
        return 2

    run = trace.open_run("screen", argv=sys.argv, args=vars(args))
    print(f"[screen] {len(cases)} cases -> {run.path}")

    pipe = draft.load_kontext_t2i(device=args.device)
    drafter = draft.Drafter(pipe, steps=pipe_cfg.steps, seed=pipe_cfg.seed,
                            device=args.device)

    # Written row-by-row rather than once at the end: a draft costs ~2 min, so
    # a crash 40 cases in must not discard 80 minutes of finished GPU work.
    labels = run.path / "labels.csv"
    n = 0
    with labels.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for i, case in enumerate(cases, 1):
            print(f"  [{i}/{len(cases)}] {case.id}: {case.prompt}", flush=True)
            image = drafter.draft(case.prompt)
            out = run.case_dir(case.id) / "draft.png"
            image.save(out)
            writer.writerow({"case_id": case.id, "prompt": case.prompt,
                             "concept": case.concept, "draft_path": str(out),
                             "verdict": "", "verdict_identity": "",
                             "notes": ""})
            fh.flush()
            n += 1

    run.finish("ok", {"n_cases": n})
    env.reclaim_gpu()

    print(f"\n[gate] Open {labels}")
    print("[gate] Fill `verdict` (any visible defect) and `verdict_identity` "
          "(is it the right thing?) with pass|fail for every row.")
    print("[gate] If fewer than ~30% are `fail`, pick rarer concepts and re-run.")
    return 0


if __name__ == "__main__":
    # Not SystemExit: unloading torch+faiss+PIL together segfaults in this
    # env, which scores a finished stage as a failure. See env.exit_now.
    env.exit_now(main())
