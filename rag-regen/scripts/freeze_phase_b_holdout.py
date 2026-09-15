#!/usr/bin/env python
"""Freeze the Phase B 24+24 holdout manifest (no GPU, seconds).

Joins hand identity labels, mechanical eligibility (queue + on-disk
artifacts, mirroring ``ragregen.eval_manifest``), the deterministically
retrieved top-1 reference per case (from a ``retrieval.json`` produced by
``retrieved-ref-score --retrieval-only``), and exact-SHA-256 contamination
against each case's ground-truth references. ``select_cohort`` then picks the
first 24 judgeable+eligible+uncontaminated cases per cohort in declared
dataset order, backfilling from whatever reserve cases follow in the file.

Refuses to proceed -- and never writes a manifest -- unless *both* cohorts
land on a complete 24-case selection; a short cohort makes Phase B
INCONCLUSIVE before any score is computed (design section 3).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config  # noqa: E402
from ragregen.eval_manifest import _eligibility, sha256_file  # noqa: E402
from ragregen.phase_b_manifest import build_holdout_manifest, select_cohort  # noqa: E402
from ragregen.retrieved_reference import contamination  # noqa: E402
from scripts.freeze_eval_manifest import (  # noqa: E402
    _identity_from_labels, _identity_from_reranker)


def _load_identity(source: Path, fmt: str) -> dict[str, str]:
    if fmt == "reranker_truth":
        return _identity_from_reranker(source)
    return _identity_from_labels(source)


def _candidates(dataset, identity_rows, queue_cases, candidate_run, retrieval):
    """Ordered ``select_cohort`` inputs, in the dataset's declared order."""
    rows = []
    for case in dataset.cases:
        cid = case.id
        if cid not in identity_rows:
            raise ValueError(f"{cid}: no identity label")
        truth = str(identity_rows[cid]).strip().upper()

        case_dir = candidate_run / cid
        eligible, _reason = _eligibility(queue_cases.get(cid, {}), case_dir)

        rcase = retrieval["cases"].get(cid)
        if rcase is None:
            # A mechanically-ineligible case (mask never grounded) is never
            # sent through retrieval -- there is nothing to score. Only a
            # case eligibility says SHOULD have a record is a real gap.
            if eligible:
                raise ValueError(f"{cid}: no retrieval record in --retrieval")
            contaminated = False
        else:
            top_path = Path(rcase["top"]["path"])
            contaminated = eligible and contamination(top_path, case.gt_refs)

        rows.append({"case_id": cid, "identity_truth": truth,
                     "eligible": eligible, "contaminated": contaminated})
    return rows


def _references(cases: list[str], retrieval: dict) -> dict[str, dict]:
    out = {}
    for cid in cases:
        top = retrieval["cases"][cid]["top"]
        out[cid] = {
            "path": top["path"], "score": top["score"], "rank": top["rank"],
            "image_sha256": retrieval["cases"][cid]["image_sha256"],
            "contaminated": retrieval["cases"][cid]["contaminated"],
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rare-dataset", type=Path, required=True)
    ap.add_argument("--control-dataset", type=Path, required=True)
    ap.add_argument("--identity-source", type=Path, required=True)
    ap.add_argument("--identity-format",
                    choices=["labels_csv", "reranker_truth"], required=True)
    ap.add_argument("--candidate-run", type=Path, required=True)
    ap.add_argument("--retrieval", type=Path, required=True,
                    help="retrieval.json from `retrieved-ref-score --retrieval-only`")
    ap.add_argument("--dino-reserves", type=Path, required=True,
                    help="JSON map: case_id -> [held-out DINO reference paths]")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    # Refuse overwrite first: never join an artifact if we cannot write.
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    rare_dataset = config.load_dataset(args.rare_dataset)
    control_dataset = config.load_dataset(args.control_dataset)
    identity_rows = _load_identity(args.identity_source, args.identity_format)
    retrieval = json.loads(args.retrieval.read_text())
    dino_reserves_all = json.loads(args.dino_reserves.read_text())

    queue_path = args.candidate_run / "queue.json"
    queue_cases = (json.loads(queue_path.read_text()).get("cases", {})
                   if queue_path.is_file() else {})

    rare_candidates = _candidates(rare_dataset, identity_rows, queue_cases,
                                  args.candidate_run, retrieval)
    control_candidates = _candidates(control_dataset, identity_rows, queue_cases,
                                     args.candidate_run, retrieval)

    rare_selection = select_cohort(rare_candidates, cohort="rare", needed=24)
    control_selection = select_cohort(control_candidates, cohort="control", needed=24)

    print(f"[rare] {len(rare_selection['cases'])}/24 "
          f"backfilled={rare_selection['backfilled']} "
          f"status={rare_selection['status']}")
    print(f"[control] {len(control_selection['cases'])}/24 "
          f"backfilled={control_selection['backfilled']} "
          f"status={control_selection['status']}")

    if rare_selection["status"] != "complete" or control_selection["status"] != "complete":
        print("INCONCLUSIVE: replacement reserve exhausted before reaching "
              "24 judgeable cases in at least one cohort; refusing to "
              "proceed (never shrinking a denominator).", file=sys.stderr)
        return 1

    chosen = rare_selection["cases"] + control_selection["cases"]
    references = _references(chosen, retrieval)
    dino_reserves = {cid: dino_reserves_all[cid] for cid in chosen}

    sources = {
        "rare_dataset": args.rare_dataset, "control_dataset": args.control_dataset,
        "identity_source": args.identity_source, "retrieval": args.retrieval,
        "dino_reserves": args.dino_reserves,
    }
    if queue_path.is_file():
        sources["queue"] = queue_path

    manifest = build_holdout_manifest(
        rare_selection, control_selection, dino_reserves=dino_reserves,
        references=references, sources=sources)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {args.out}: 24 rare + 24 control, "
          f"sha256={sha256_file(args.out)[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
