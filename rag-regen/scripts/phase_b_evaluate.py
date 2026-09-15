#!/usr/bin/env python
"""End-to-end evaluation CLI (GPU): aggregate, gate, and freeze readiness.

Design section 7-8: score every arm output that differs from its draft
(cropped held-out DINO, whole CLIP, whole SigLIP, preservation), aggregate
into the arm x cohort tables, apply the five e2e gates, and fold in the
Task-4 detector gate for the final conjunctive readiness plus the thin-PASS
guard. Only a non-thin PASS gets a frozen, hashed ``phase_b_result.json`` --
anything else stops at the report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen.phase_b_eval import aggregate, end_to_end_gates, readiness  # noqa: E402

SCOPE_STATEMENT = (
    "A PASS authorizes integration testing only -- wiring the frozen policy "
    "into the pipeline behind a flag and exercising it -- not an unqualified "
    "production claim and not the 92-case run."
)


def _default_score(arms_cases: dict, holdout: dict) -> list[dict]:
    """Real per-image scoring via the eval encoders.

    Best-effort GPU wiring, exercised by the operator's Runtime sequence
    (plan footer), never by these CPU tests -- every CLI test injects a fake
    ``score_fn`` instead. Needs the held-out DINO reserve crops from the
    holdout manifest (never the edit pool, per `ragregen.metrics.dino_identity`)
    and each arm output's own image to score prompt alignment / preservation.
    """
    from PIL import Image

    from ragregen import config, metrics

    pipe_cfg = config.load_pipeline()
    enc = metrics.EvalEncoders.load(pipe_cfg)
    try:
        rows = []
        for cid, case in arms_cases.items():
            cohort = case["cohort"]
            draft = Image.open(case["arm1"]["path"]).convert("RGB")
            draft_dino = metrics.dino_identity(draft, [
                Image.open(p) for p in holdout["dino_reserves"][cid]], enc.dino)
            for arm in ("arm1", "arm2", "arm3"):
                out = case[arm]
                image = Image.open(out["path"]).convert("RGB")
                out_dino = metrics.dino_identity(image, [
                    Image.open(p) for p in holdout["dino_reserves"][cid]], enc.dino)
                rows.append({
                    "case_id": cid, "cohort": cohort, "arm": arm,
                    "generation_failed": bool(out.get("generation_failed")),
                    "cropped_dino_delta": out_dino - draft_dino,
                    "whole_clip": metrics.prompt_alignment(image, "", enc.clip),
                    "whole_siglip": metrics.prompt_alignment(image, "", enc.siglip),
                    "preservation": 1.0 if arm == "arm1" else None,
                })
        return rows
    finally:
        enc.free()


def _render_md(report: dict) -> str:
    lines = [f"# Phase B evaluation -- {report['readiness']}", ""]
    if report.get("thin"):
        lines.append("**THIN PASS**: landed exactly on a predeclared discrete "
                     "boundary; requires one further untouched confirmation "
                     "holdout before the 92-case run.")
        lines.append("")
    lines.append("## Detector gate")
    for g in report["detector_gate"].get("gates", []):
        lines.append(f"- {g['name']}: {g['passed']}")
    lines.append("")
    lines.append("## End-to-end gates")
    for g in report["e2e_gates"].get("gates", []):
        lines.append(f"- {g['name']}: {g['passed']}")
    lines.append("")
    lines.append("## Arm x cohort")
    lines.append("| cohort | arm | n | mean cropped DINO delta | preservation min |")
    lines.append("| --- | --- | --- | --- | --- |")
    for cohort in ("rare", "control"):
        for arm in ("arm1", "arm2", "arm3"):
            stats = report["agg"][cohort][arm]
            lines.append(f"| {cohort} | {arm} | {stats['n']} | "
                         f"{stats['mean_cropped_dino_delta']} | "
                         f"{stats['preservation_min']} |")
    lines.append("")
    lines.append(SCOPE_STATEMENT)
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, agg: dict) -> None:
    with open(path, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["cohort", "arm", "n", "mean_cropped_dino_delta",
                   "whole_clip_mean", "whole_siglip_mean", "preservation_min",
                   "improved", "unchanged", "worsened"])
        for cohort in ("rare", "control"):
            for arm in ("arm1", "arm2", "arm3"):
                s = agg[cohort][arm]
                w.writerow([cohort, arm, s["n"], s["mean_cropped_dino_delta"],
                           s["whole_clip_mean"], s["whole_siglip_mean"],
                           s["preservation_min"], s["improved"], s["unchanged"],
                           s["worsened"]])


def main(argv=None, *, score_fn=_default_score) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=Path, required=True)
    ap.add_argument("--holdout-manifest", type=Path, required=True)
    ap.add_argument("--detector-report", type=Path, required=True)
    ap.add_argument("--visual-review", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty {args.out_dir}")

    arms = json.loads(args.arms.read_text())
    holdout = json.loads(args.holdout_manifest.read_text())
    detector_gate = json.loads(args.detector_report.read_text())
    visual_review = json.loads(args.visual_review.read_text())

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if arms.get("status") != "complete":
        report = {"readiness": "INCONCLUSIVE", "thin": False,
                  "reason": f"arms.json status={arms.get('status')!r}: "
                            f"{arms.get('reason', 'n/a')}"}
        (args.out_dir / "development.json").write_text(
            json.dumps(report, indent=2, sort_keys=True))
        (args.out_dir / "development.md").write_text(
            f"# Phase B evaluation -- INCONCLUSIVE\n\n{report['reason']}\n\n"
            f"{SCOPE_STATEMENT}\n")
        print(f"[phase-b-evaluate] INCONCLUSIVE: {report['reason']}")
        return 0

    metric_rows = score_fn(arms["cases"], holdout)
    agg = aggregate(metric_rows)
    e2e_gates = end_to_end_gates(agg, visual_review)
    overall = readiness(detector_gate, e2e_gates)

    report = dict(overall)
    report["agg"] = agg
    (args.out_dir / "development.json").write_text(
        json.dumps(report, indent=2, sort_keys=True))
    (args.out_dir / "development.md").write_text(_render_md(report))
    _write_csv(args.out_dir / "per_arm.csv", agg)

    if report["readiness"] == "PASS" and not report["thin"]:
        result_path = args.out_dir / "phase_b_result.json"
        payload = json.dumps(report, indent=2, sort_keys=True)
        result_path.write_text(payload)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        (args.out_dir / "phase_b_result.json.sha256").write_text(digest + "\n")
        print(f"[phase-b-evaluate] PASS (non-thin): froze phase_b_result.json "
              f"sha256={digest[:12]}")
    else:
        print(f"[phase-b-evaluate] readiness={report['readiness']} "
              f"thin={report['thin']} -- no result frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
