#!/usr/bin/env python
"""Detector-first scoring CLI (no GPU, seconds). Stops before generation.

Joins the frozen holdout manifest (Task 2) and frozen policy manifest
(Task 1) against the holdout's semantic verdicts (``stream_b.json``, from
``score-b``) and reranker scores (``retrieved_reranker.json``, from
``retrieved-ref-score``), strict-verifying that every scored draft is the one
the holdout froze -- then applies the frozen rule (``apply_frozen``) and the
discrete gate (``detector_gate``). Design section 5: run the detector first,
gate on it, and stop before generation on anything but a clean PASS.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen.detector_apply import (  # noqa: E402
    apply_frozen, degenerate_coverage, detector_gate)


def _rows(holdout: dict, retrieved: dict, stream_b: dict) -> list[dict]:
    rows = []
    for cohort, truth_fail in (("rare", True), ("control", False)):
        for cid in holdout["cohorts"][cohort]["cases"]:
            ref = holdout["references"].get(cid)
            if ref is None:
                raise ValueError(f"{cid}: no frozen reference in holdout manifest")

            rcase = retrieved["cases"].get(cid)
            if rcase is None:
                raise ValueError(f"{cid}: missing from --retrieved-reranker")
            if rcase["retrieval"]["image_sha256"] != ref["image_sha256"]:
                raise ValueError(
                    f"{cid}: retrieved reference {rcase['retrieval']['image_sha256']!r} "
                    f"does not match the frozen holdout reference {ref['image_sha256']!r}")

            bcase = stream_b.get(cid)
            if bcase is None:
                raise ValueError(f"{cid}: missing from --stream-b")

            scores = rcase["scores"]["draft"]
            rows.append({
                "case_id": cid, "truth_fail": truth_fail,
                "semantic_ok": bool(bcase["ok"]),
                "text_relevance": scores["text_relevance"],
                "retrieved_reference_relevance": scores["reference_relevance"],
                "cohort": cohort,
            })
    return rows


def _render_md(report: dict) -> str:
    lines = [f"# Phase B detector gate -- {report['readiness']}", ""]
    for name, k, n in (("rare recall", *report["recall_k"]),
                       ("control FP", *report["fp_k"])):
        lines.append(f"- {name}: {k}/{n}")
    lines.append("")
    lines.append("| gate | passed |")
    lines.append("| --- | --- |")
    for g in report["gates"]:
        lines.append(f"| {g['name']} | {g['passed']} |")
    lines.append("")
    lines.append(f"proceed_to_generation: {report.get('proceed_to_generation', False)}")
    return "\n".join(lines) + "\n"


def main(argv=None, *, apply_fn=apply_frozen, gate_fn=detector_gate) -> int:
    apply_fn = apply_fn or apply_frozen
    gate_fn = gate_fn or detector_gate

    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-manifest", type=Path, required=True)
    ap.add_argument("--policy-manifest", type=Path, required=True)
    ap.add_argument("--retrieved-reranker", type=Path, required=True)
    ap.add_argument("--stream-b", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty {args.out_dir}")

    holdout = json.loads(args.holdout_manifest.read_text())
    policy = json.loads(args.policy_manifest.read_text())["manifest"]
    retrieved = json.loads(args.retrieved_reranker.read_text())
    stream_b = json.loads(args.stream_b.read_text())

    if holdout["protocol_status"] != "frozen_before_score":
        raise ValueError("holdout manifest is not frozen_before_score")
    if policy["protocol_status"] != "frozen_before_score":
        raise ValueError("policy manifest is not frozen_before_score")
    # The scoring artifact records a short slug ("qwen3-vl-reranker-2b"); the
    # frozen policy records the full HF repo id ("Qwen/Qwen3-VL-Reranker-2B")
    # for genuine weight-file traceability. Same model, two specificities --
    # compare by normalising the repo id down to that same slug form.
    policy_slug = policy["reranker"]["model_id"].rsplit("/", 1)[-1].lower()
    if retrieved.get("model") != policy_slug:
        raise ValueError(
            f"reranker model mismatch: scored with {retrieved.get('model')!r}, "
            f"policy froze {policy['reranker']['model_id']!r} "
            f"(normalised {policy_slug!r})")
    if retrieved.get("prompt") != policy["reranker"]["prompt"]:
        raise ValueError("reranker prompt does not match the frozen policy")

    rare_ids = holdout["cohorts"]["rare"]["cases"]
    control_ids = holdout["cohorts"]["control"]["cases"]
    if len(rare_ids) != 24 or len(control_ids) != 24:
        raise ValueError("holdout manifest cohorts are not exactly 24 + 24")

    rows = _rows(holdout, retrieved, stream_b)

    n_contaminated = sum(1 for r in rare_ids + control_ids
                         if holdout["references"][r]["contaminated"])
    coverage_ok = (degenerate_coverage(rows) and n_contaminated == 0
                  and len(rows) == 48)

    applied = apply_fn(rows, policy["thresholds"])
    gate = gate_fn(applied, coverage_ok=coverage_ok)

    report = dict(gate)
    report["n_rows"] = len(rows)
    report["n_contaminated"] = n_contaminated
    report["proceed_to_generation"] = gate["readiness"] == "PASS"
    # Per-case routed flags, so phase-b-generate can build its three arms
    # without re-deriving the frozen rule from scratch.
    report["cases"] = applied.get("cases", {})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "detector.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (args.out_dir / "detector.md").write_text(_render_md(report))
    print(f"[phase-b-detector] readiness={report['readiness']} "
          f"recall={report['recall_k']} fp={report['fp_k']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
