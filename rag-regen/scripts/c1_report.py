#!/usr/bin/env python
"""Fuse the two stages, sweep tau, render the C1 report (spec §5).

No models: this reads the JSON the two GPU stages wrote. Re-running after a
label is revised costs seconds, which is the whole reason Stream A persists
raw similarities.

Usage: ./scripts/run.sh c1 --run outputs/screen_<ts>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import c1  # noqa: E402

COLUMNS = ("verdict_identity", "verdict")


def _confusion_dict(m) -> dict:
    return {"tp": m.tp, "fp": m.fp, "tn": m.tn, "fn": m.fn,
            "precision": m.precision, "recall": m.recall, "f1": m.f1,
            "balanced_accuracy": m.balanced_accuracy, "mcc": m.mcc,
            "accuracy": m.accuracy}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True,
                    help="a screen run directory holding labels.csv")
    ap.add_argument("--stream-a", type=Path, default=None,
                    help="alternate Stream A JSON")
    ap.add_argument("--stream-b", type=Path, default=None,
                    help="alternate Stream B JSON")
    ap.add_argument("--stem", default="c1",
                    help="output stem (default: c1)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="artifact directory (default: --run)")
    args = ap.parse_args(argv)

    labels = args.run / "labels.csv"
    a_path = args.stream_a or args.run / "stream_a.json"
    b_path = args.stream_b or args.run / "stream_b.json"

    for path, stage in ((labels, None), (a_path, "score-a"), (b_path, "score-b")):
        if not path.is_file():
            hint = f" Run `./scripts/run.sh {stage}` first." if stage else ""
            print(f"[error] missing {path.name} in {args.run}.{hint}")
            return 2

    rows = c1.load_labels(labels)
    stream_a = json.loads(a_path.read_text())
    stream_b = json.loads(b_path.read_text())
    degenerate = sum(1 for v in stream_b.values() if v.get("degenerate"))

    sections, artifact = [], {}
    for column in COLUMNS:
        kept, y_true, excluded = c1.ground_truth(rows, column)
        case_ids = [r.case_id for r in kept]
        sweep = c1.sweep_tau(stream_a, stream_b, case_ids, y_true)
        sections.append(c1.render_report(
            sweep=sweep, y_true=y_true, column=column, excluded=excluded,
            degenerate=degenerate, n_cases=len(case_ids)))

        fg_md, fg_art = c1.render_finegrained(stream_a, stream_b, case_ids,
                                              y_true, column=column)
        sections.append(fg_md)

        pr_md, pr_art = c1.render_prototype(stream_a, stream_b, case_ids,
                                            y_true, column=column)
        if pr_md:
            sections.append(pr_md)

        tau_star = c1.best_tau(sweep, "fused")
        at_star = next(r for r in sweep if r["tau"] == tau_star)
        artifact[column] = {
            "tau_star": tau_star,
            "n_fail": sum(y_true),
            "n_pass": len(y_true) - sum(y_true),
            "excluded": excluded,
            "degenerate": degenerate,
            "abstain_rate": at_star["abstain_rate"],
            "arms_at_tau_star": {k: _confusion_dict(v)
                                 for k, v in at_star["arms"].items()},
            "baselines": {k: _confusion_dict(v)
                          for k, v in c1.baseline_rows(y_true).items()},
            "finegrained": fg_art,
            "prototype": pr_art,
        }

    md = "\n\n---\n\n".join(sections)
    output_dir = args.output_dir or args.run
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{args.stem}.md"
    json_path = output_dir / f"{args.stem}.json"
    md_path.write_text(md)
    json_path.write_text(json.dumps(artifact, indent=2))
    print(md)
    print(f"\n[c1] wrote {md_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
