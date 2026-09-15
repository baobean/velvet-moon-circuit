#!/usr/bin/env python
"""Preflight the operator's configs. Runs before any GPU work.

Usage: ./scripts/run.sh validate
Exit codes: 0 = clean (warnings allowed), 2 = errors found.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config, validate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET_PATH)
    ap.add_argument("--db", type=Path, default=config.DEFAULT_DB_PATH)
    ap.add_argument("--pipeline", type=Path, default=config.DEFAULT_PIPELINE_PATH)
    args = ap.parse_args()

    try:
        ds = config.load_dataset(args.dataset)
        db = config.load_retrieval_db(args.db)
        pipe = config.load_pipeline(args.pipeline)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] config: {exc}")
        return 2

    problems = validate.validate_all(ds, db, pipe)
    errors = [p for p in problems if p.severity == "error"]
    warnings = [p for p in problems if p.severity == "warning"]

    for p in errors + warnings:
        print(f"[{p.severity}] {p.code}: {p.message}")

    print(f"\n[summary] {len(ds.cases)} cases, corpus at {db.images_root}")
    print(f"[summary] {len(errors)} error(s), {len(warnings)} warning(s)")
    if errors:
        print("[summary] Fix the errors above. Nothing will run until they are clear.")
        return 2
    print("[summary] OK to run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
