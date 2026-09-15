#!/usr/bin/env python
"""Apply the frozen hybrid policy to cached holdout artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import config, hybrid_verifier  # noqa: E402


def _load(path: Path):
    return json.loads(path.read_text())


def _labels(path: Path) -> dict[str, str]:
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            case_id = (row.get("case_id") or "").strip()
            value = (row.get("verdict_identity") or "").strip().upper()
            if not case_id:
                raise ValueError(f"{path}: label row is missing case_id")
            if value not in ("PASS", "FAIL", "EXCLUDE"):
                raise ValueError(
                    f"{path}: {case_id} verdict_identity={value!r}; "
                    "expected PASS, FAIL, or EXCLUDE")
            result[case_id] = value
    return result


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _materialize(result: dict, screen_run: Path, candidate_run: Path,
                 output_dir: Path) -> None:
    for case_id, decision in result["cases"].items():
        draft = screen_run / case_id / "draft.png"
        attempt_1 = candidate_run / case_id / "attempt_1.png"
        no_verifier = attempt_1 if attempt_1.is_file() else draft
        hybrid_label = decision["hybrid"]
        hybrid = (draft if hybrid_label == "draft" else
                  candidate_run / case_id / f"{hybrid_label}.png")
        for arm, source in (("draft", draft),
                            ("no_verifier", no_verifier),
                            ("hybrid", hybrid)):
            _copy(source, output_dir / "arms" / arm / f"{case_id}.png")


def _end_to_end_gates(result: dict, visual_review: dict | None) -> dict:
    rare = result["cohorts"].get("bridge")
    common = result["cohorts"].get("common")
    rare_positive = bool(rare and rare["mean_dino_delta"] > 0.0)
    common_interval = common and common.get("dino_delta_95ci")
    common_no_harm = (common_interval is not None
                      and common_interval[0] >= hybrid_verifier.HARM_MARGIN)
    selected_no_harm = (
        result["selector"]["harmful"] <= hybrid_verifier.MAX_HARMFUL)
    visual_passed = (None if visual_review is None else
                     bool(visual_review.get("passed")))
    return {
        "rare_positive_mean": rare_positive,
        "common_no_harm": common_no_harm,
        "common_interval_available": common_interval is not None,
        "selected_harmful_within_limit": selected_no_harm,
        "visual_review_passed": visual_passed,
        "passed": (rare_positive and common_no_harm and selected_no_harm
                   and visual_passed is True),
    }


def _status(result: dict) -> str:
    gates = result["gates"]
    if gates["detector"].get("reason"):
        return "INCONCLUSIVE"
    end = gates["end_to_end"]
    if (not end["common_interval_available"]
            or end["visual_review_passed"] is None):
        return "INCONCLUSIVE"
    return ("PASS" if gates["detector"]["passed"]
            and gates["selector"]["passed"] and end["passed"] else "FAIL")


def _render(result: dict) -> str:
    union = result["detector"]["union"]
    selector = result["selector"]
    lines = [
        "# Frozen hybrid verifier holdout", "",
        f"Decision: **{result['status']}**", "",
        "## Detector", "",
        "| TP | FP | TN | FN | recall | specificity | balanced accuracy | MCC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (f"| {union['tp']} | {union['fp']} | {union['tn']} | {union['fn']} "
         f"| {union['recall']:.3f} | {union['specificity']:.3f} "
         f"| {union['balanced_accuracy']:.3f} | {union['mcc']:+.3f} |"),
        "", "## Selector", "",
        "| comparisons | sign accuracy | exact DINO-best | mean DINO delta | harmful |",
        "|---:|---:|---:|---:|---:|",
        (f"| {selector['comparisons']} | {selector['sign_accuracy']:.3f} "
         f"| {selector['exact_dino_best']:.3f} "
         f"| {selector['mean_dino_delta']:+.3f} "
         f"| {selector['harmful']} |"),
        "", "## Cohorts", "",
        "| cohort | n | mean DINO delta | 95% CI | +/=/− |",
        "|---|---:|---:|---:|---:|",
    ]
    for cohort, row in result["cohorts"].items():
        interval = row["dino_delta_95ci"]
        rendered = ("—" if interval is None else
                    f"[{interval[0]:+.3f}, {interval[1]:+.3f}]")
        lines.append(
            f"| {cohort} | {row['n']} | {row['mean_dino_delta']:+.3f} "
            f"| {rendered} | {row['improved']}/{row['unchanged']}/"
            f"{row['worsened']} |")
    lines += ["", "## Per-case decisions", "",
              "| case | truth | semantic | reranker | routed | selected | DINO delta |",
              "|---|---|---|---|---|---|---:|"]
    for case_id, row in result["cases"].items():
        lines.append(
            f"| {case_id} | {row['truth']} | {row['semantic']} "
            f"| {row['reranker']} | {row['routed']} | {row['hybrid']} "
            f"| {row['dino_delta']:+.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--semantic", type=Path, required=True)
    ap.add_argument("--reranker", type=Path, required=True)
    ap.add_argument("--dino", type=Path, required=True)
    ap.add_argument("--screen-run", type=Path, required=True)
    ap.add_argument("--candidate-run", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--visual-review", type=Path, default=None)
    args = ap.parse_args(argv)

    dataset = config.load_dataset(args.dataset)
    labels = _labels(args.labels)
    dataset_ids = {case.id for case in dataset.cases}
    if set(labels) != dataset_ids:
        missing = sorted(dataset_ids - set(labels))
        extra = sorted(set(labels) - dataset_ids)
        raise ValueError(f"label/dataset mismatch: missing={missing}, extra={extra}")
    cohorts = {case.id: case.cohort for case in dataset.cases}
    result = hybrid_verifier.evaluate(
        labels, _load(args.semantic), _load(args.reranker),
        _load(args.dino), cohorts)
    visual = _load(args.visual_review) if args.visual_review else None
    result["gates"]["end_to_end"] = _end_to_end_gates(result, visual)
    result["visual_review"] = visual
    result["status"] = _status(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _materialize(result, args.screen_run, args.candidate_run, args.output_dir)
    (args.output_dir / "decisions.json").write_text(
        json.dumps(result["cases"], indent=2))
    (args.output_dir / "hybrid_validation.json").write_text(
        json.dumps(result, indent=2))
    (args.output_dir / "hybrid_validation.md").write_text(_render(result))
    with (args.output_dir / "per_case.csv").open("w", newline="") as handle:
        fields = ["case_id", "truth", "semantic", "reranker", "routed",
                  "hybrid", "dino_delta", "cohort"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case_id, row in result["cases"].items():
            writer.writerow({"case_id": case_id, **row})
    print(_render(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
