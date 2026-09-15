"""Freeze the immutable identity/eligibility manifest (CPU-only).

Refuses to overwrite an existing manifest *before* touching any model artifact,
so a stale run can never be clobbered and provenance is never half-written.
Identity truth is read from either the hand labels CSV or the reranker
artifact's frozen ``truth`` field; the latter is what keeps ``azawakh`` and
``bergamasco_shepherd`` as detector ``FAIL`` cases while remaining
selector-ineligible.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config, eval_manifest

_CSV_VERDICT = {"pass": "PASS", "fail": "FAIL",
                "exclude": "UNJUDGEABLE", "unjudgeable": "UNJUDGEABLE"}


def _identity_from_reranker(path: Path) -> dict[str, str]:
    cases = json.loads(path.read_text()).get("cases", {})
    return {cid: str(row["truth"]).strip().upper()
            for cid, row in cases.items()}


def _identity_from_labels(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            verdict = (row.get("verdict_identity") or row.get("verdict")
                       or "").strip().lower()
            if verdict not in _CSV_VERDICT:
                raise ValueError(
                    f"{row.get('case_id')!r}: unmapped verdict {verdict!r}")
            rows[row["case_id"]] = _CSV_VERDICT[verdict]
    return rows


def _git_status() -> str:
    try:
        out = subprocess.run(
            ["git", "status", "--short"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent, check=False)
        return out.stdout
    except OSError:
        return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--identity-source", type=Path, required=True)
    ap.add_argument("--identity-format",
                    choices=["labels_csv", "reranker_truth"], required=True)
    ap.add_argument("--screen-run", type=Path, required=True)
    ap.add_argument("--candidate-run", type=Path, required=True)
    ap.add_argument("--semantic", type=Path, required=True)
    ap.add_argument("--reranker", type=Path, required=True)
    ap.add_argument("--dino", type=Path, required=True)
    ap.add_argument("--protocol-status",
                    choices=["frozen_before_score", "reconstructed_after_score"],
                    required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    # Refuse overwrite first: never read model artifacts if we cannot write.
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    dataset = config.load_dataset(args.dataset)
    if args.identity_format == "reranker_truth":
        identity_rows = _identity_from_reranker(args.identity_source)
    else:
        identity_rows = _identity_from_labels(args.identity_source)

    source_paths = {
        "dataset": args.dataset,
        "identity_source": args.identity_source,
        "semantic": args.semantic,
        "reranker": args.reranker,
        "dino": args.dino,
        "queue": args.candidate_run / "queue.json",
    }
    manifest = eval_manifest.build_manifest(
        dataset, identity_rows, args.screen_run, args.candidate_run,
        protocol_status=args.protocol_status, source_paths=source_paths)

    manifest["provenance"] = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_status": _git_status(),
        "screen_run": str(args.screen_run),
        "candidate_run": str(args.candidate_run),
        "identity_format": args.identity_format,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {args.out}: {manifest['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
