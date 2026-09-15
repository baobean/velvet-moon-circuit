"""End-to-end evaluation, gates, thin-guard, and readiness. Pure.

``aggregate`` builds the full per-arm x per-cohort reporting tables (design
section 7): cropped held-out DINO, whole CLIP, whole SigLIP, preservation,
the paired bootstrap CI versus draft, and improved/unchanged/worsened
counts. ``end_to_end_gates`` then checks only the five predeclared gates
(section 7); ``readiness`` folds those together with the detector gate
(Task 3) and applies the thin-PASS guard (section 8) -- the trigger is the
discrete count boundary alone, never a confidence bound.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from ragregen.metrics import paired_bootstrap_ci

ARMS = ("arm1", "arm2", "arm3")
COHORTS = ("rare", "control")
HARMFUL_THRESHOLD = -0.02
NON_INFERIORITY_MARGIN = -0.02


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _min(values):
    values = [v for v in values if v is not None]
    return min(values) if values else None


def _arm_stats(rows: list[dict]) -> dict:
    deltas = [r["cropped_dino_delta"] for r in rows if r.get("cropped_dino_delta") is not None]
    ci = paired_bootstrap_ci(deltas) if deltas else None
    return {
        "n": len(rows),
        "mean_cropped_dino_delta": _mean(deltas) if deltas else None,
        "whole_clip_mean": _mean(r.get("whole_clip") for r in rows),
        "whole_siglip_mean": _mean(r.get("whole_siglip") for r in rows),
        "preservation_min": _min(r.get("preservation") for r in rows),
        "arm_minus_draft_ci": list(ci) if ci else None,
        "improved": sum(1 for d in deltas if d > 0),
        "unchanged": sum(1 for d in deltas if d == 0),
        "worsened": sum(1 for d in deltas if d < 0),
        "generation_failed_cases": [r["case_id"] for r in rows if r.get("generation_failed")],
    }


def aggregate(metric_rows: Sequence[Mapping]) -> dict:
    """One row per ``(case_id, arm)``; every arm x cohort table, plus the
    reduced fields ``end_to_end_gates`` consumes directly."""
    by_cohort_arm = {c: {a: [] for a in ARMS} for c in COHORTS}
    for r in metric_rows:
        by_cohort_arm[r["cohort"]][r["arm"]].append(r)

    tables = {c: {a: _arm_stats(rows) for a, rows in arms.items()}
             for c, arms in by_cohort_arm.items()}
    # Convenience top-level key the gate reads directly (section 7's
    # control-non-inferiority gate is scored on arm3 - draft only).
    tables["control"]["arm3_minus_draft_ci"] = tables["control"]["arm3"]["arm_minus_draft_ci"]

    harmful_count_48 = sum(
        1 for r in metric_rows
        if r["arm"] == "arm3" and r.get("cropped_dino_delta") is not None
        and r["cropped_dino_delta"] < HARMFUL_THRESHOLD)

    preservation_min = _min(
        r.get("preservation") for r in metric_rows if r["arm"] != "arm1")

    tables["harmful_count_48"] = harmful_count_48
    tables["preservation_min"] = preservation_min
    return tables


def end_to_end_gates(agg: Mapping, visual_review: Mapping) -> dict:
    """Design section 7, five conjunctive gates on arm3 vs arm1 (draft)."""
    rare_mean = agg["rare"]["arm3"]["mean_cropped_dino_delta"]
    harmful = agg["harmful_count_48"]
    ctrl_ci = agg["control"]["arm3_minus_draft_ci"]
    preservation_min = agg["preservation_min"]

    gates = [
        ("rare_mean_dino_positive", rare_mean is not None and rare_mean > 0),
        ("harmful_le_1", harmful <= 1),
        ("control_non_inferiority",
         ctrl_ci is not None and ctrl_ci[0] >= NON_INFERIORITY_MARGIN),
        ("preservation_exact_1", preservation_min is not None and preservation_min == 1.0),
        ("visual_review", bool(visual_review.get("pass"))),
    ]
    passed = all(p for _, p in gates)
    return {"readiness": "PASS" if passed else "FAIL",
           "gates": [{"name": n, "passed": p} for n, p in gates]}


def readiness(detector_gate: Mapping, e2e_gates: Mapping) -> dict:
    """Conjunctive over the detector gate (Task 3) and the e2e gates.

    ``thin`` fires on the discrete boundary alone -- recall exactly 18/24 or
    FP exactly 2/24 (section 8); no Wilson-bound condition enters this guard.
    """
    if detector_gate["readiness"] == "INCONCLUSIVE" or e2e_gates["readiness"] == "INCONCLUSIVE":
        overall = "INCONCLUSIVE"
    elif detector_gate["readiness"] == "PASS" and e2e_gates["readiness"] == "PASS":
        overall = "PASS"
    else:
        overall = "FAIL"

    recall_k = detector_gate.get("recall_k")
    fp_k = detector_gate.get("fp_k")
    thin = bool(tuple(recall_k) == (18, 24) if recall_k else False) or bool(
        fp_k is not None and fp_k[0] == 2)

    return {"readiness": overall, "thin": thin,
           "detector_gate": detector_gate, "e2e_gates": e2e_gates}
