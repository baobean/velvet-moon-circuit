"""Frozen-policy application and the detector gate. Pure, no fitting.

Applies the frozen routing rule (semantic_failure OR (text < 0.5 AND
retrieved_reference < 0.25048828125)) verbatim to the Phase B holdout rows --
there is no threshold search here, `apply_frozen` just replays
`ragregen.hybrid_verifier_v2.route_draft` at the two frozen thresholds. The
gate then checks the three discrete conditions design section 5 requires:
complete uncontaminated coverage, rare recall >= 18/24, control FP <= 2/24.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from ragregen import c1
from ragregen.hybrid_verifier_v2 import Policy, route_draft


def degenerate_coverage(rows: Sequence[Mapping]) -> bool:
    """False iff any row has a missing/non-finite detector score.

    Per the frozen degenerate-input rule
    (`ragregen.detector_policy.DEGENERATE_RULE`): a missing or non-finite
    ``text_relevance``/``retrieved_reference_relevance`` is a coverage
    failure, never silently defaulted.
    """
    for r in rows:
        for k in ("text_relevance", "retrieved_reference_relevance"):
            v = r.get(k)
            if v is None or not math.isfinite(float(v)):
                return False
    return True


def apply_frozen(rows: Sequence[Mapping], thresholds: Mapping) -> dict:
    """Apply the frozen rule to every row; split confusion by cohort."""
    p = Policy(thresholds["text"], thresholds["retrieved_reference"], 0.0, 0.0)
    cases: dict[str, dict] = {}
    by: dict[str, tuple[list[bool], list[bool]]] = {"rare": ([], []), "control": ([], [])}
    for r in rows:
        routed = route_draft(r["semantic_ok"], r["text_relevance"],
                             r["retrieved_reference_relevance"], p)
        cases[r["case_id"]] = {"routed": bool(routed), "truth_fail": r["truth_fail"],
                               "cohort": r["cohort"]}
        y_true, y_pred = by[r["cohort"]]
        y_true.append(bool(r["truth_fail"]))
        y_pred.append(bool(routed))
    confusion = {cohort: c1.confusion(*pair) for cohort, pair in by.items()}
    return {"cases": cases, "confusion": confusion}


def detector_gate(applied: dict, *, coverage_ok: bool) -> dict:
    """Design section 5: complete coverage AND recall>=18/24 AND fp<=2/24.

    `readiness` is INCONCLUSIVE iff coverage is not clean -- a missing or
    non-finite score is never silently defaulted into a PASS/FAIL verdict.
    """
    rare = applied["confusion"]["rare"]
    control = applied["confusion"]["control"]
    recall_k = (rare.tp, rare.tp + rare.fn)
    fp_k = (control.fp, control.tp + control.fp + control.tn + control.fn)
    recall_ok = rare.tp >= 18
    fp_ok = control.fp <= 2
    gates = [
        ("uncontaminated_coverage", coverage_ok),
        ("detector_recall_ge_18", recall_ok),
        ("detector_fp_le_2", fp_ok),
    ]
    if not coverage_ok:
        readiness = "INCONCLUSIVE"
    elif recall_ok and fp_ok:
        readiness = "PASS"
    else:
        readiness = "FAIL"
    return {
        "readiness": readiness,
        "gates": [{"name": name, "passed": passed} for name, passed in gates],
        "recall_k": recall_k,
        "fp_k": fp_k,
    }
