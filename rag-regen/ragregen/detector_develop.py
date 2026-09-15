"""Detector-only leave-one-concept-out evaluation.

Extracted from the full v2 developer so the retrieved-reference Phase A does not
re-run the already-rejected selector. Only the guarded-conjunction *routing*
detector is fit and graded here:

    route = semantic_failure OR (text < fitted_t AND reference < fitted_r)

Every held-out prediction comes from a fit that excluded the held-out case; the
detector threshold is re-fit inside each fold (never reused from the oracle
study). Contamination or incomplete coverage forces INCONCLUSIVE rather than a
FAIL, so a data problem is never mistaken for an oracle-dependence result.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from ragregen import c1
from ragregen import hybrid_verifier as v1
from ragregen import hybrid_verifier_v2 as v2

RARE_COHORT = "bridge"


def _loco(rows: Sequence[Mapping], ref_key: str):
    """Leave-one-concept-out routing over ``rows`` using ``ref_key`` as the
    reference signal. Returns (y_true, y_pred, folds)."""
    ordered = sorted(rows, key=lambda r: r["case_id"])
    y_true, y_pred, folds = [], [], []
    for held in ordered:
        cid = held["case_id"]
        train = [{"truth_fail": r["truth_fail"], "semantic_ok": r["semantic_ok"],
                  "text_relevance": r["text_relevance"],
                  "reference_relevance": r[ref_key], "case_id": r["case_id"]}
                 for r in ordered if r["case_id"] != cid]
        fit = v2.fit_detector(train)
        policy = v2.Policy(fit.text_threshold, fit.reference_threshold, 0.0, 0.0)
        pred = v2.route_draft(held["semantic_ok"], held["text_relevance"],
                              held[ref_key], policy)
        y_true.append(bool(held["truth_fail"]))
        y_pred.append(bool(pred))
        folds.append({
            "case_id": cid,
            "fit_case_ids": sorted(r["case_id"] for r in ordered
                                   if r["case_id"] != cid),
            "feasible": fit.feasible,
            "text_threshold": fit.text_threshold,
            "reference_threshold": fit.reference_threshold,
            "prediction": bool(pred),
            "truth": "FAIL" if held["truth_fail"] else "PASS",
        })
    return y_true, y_pred, folds


def _confusion(y_true, y_pred) -> dict:
    m = c1.confusion(y_true, y_pred)
    fpr = m.fp / (m.fp + m.tn) if (m.fp + m.tn) else 0.0
    lo, hi = c1.wilson_ci(m.tp, m.tp + m.fn)
    return {"tp": m.tp, "fp": m.fp, "tn": m.tn, "fn": m.fn,
            "recall": m.recall, "false_positive_rate": fpr,
            "specificity": m.specificity, "mcc": m.mcc,
            "recall_wilson95": [lo, hi]}


def _gate(name, passed, value, threshold, reason) -> dict:
    return {"name": name, "passed": passed, "value": value,
            "threshold": threshold, "reason": reason}


def develop(rows: Sequence[Mapping], *, contaminated: Sequence[str] = (),
            n_expected: int | None = None) -> dict:
    rows = list(rows)
    contaminated = list(contaminated)

    y_true, y_pred, folds = _loco(rows, "reference_relevance")
    retrieved = _confusion(y_true, y_pred)

    # Fixed-rule baselines over identical held-out rows.
    ordered = sorted(rows, key=lambda r: r["case_id"])
    truth = [bool(r["truth_fail"]) for r in ordered]
    sem = [not r["semantic_ok"] for r in ordered]
    refonly = [r["reference_relevance"] < v1.REFERENCE_THRESHOLD for r in ordered]
    baselines = {
        "semantic_only": _confusion(truth, sem),
        "retrieved_reference_only": _confusion(truth, refonly),
    }
    if all("oracle_reference_relevance" in r for r in ordered):
        ot, op, _ = _loco(rows, "oracle_reference_relevance")
        baselines["oracle_v2"] = _confusion(ot, op)

    coverage_ok = (not contaminated
                   and (n_expected is None or len(rows) == n_expected))
    all_feasible = all(f["feasible"] for f in folds)
    recall_ok = retrieved["recall"] >= v2.MIN_RECALL
    fp_ok = (retrieved["fp"] <= v2.MAX_HARMFUL
             or retrieved["false_positive_rate"] <= v2.MAX_FP_RATE)

    gates = [
        _gate("uncontaminated_coverage", coverage_ok,
              {"n": len(rows), "n_expected": n_expected,
               "contaminated": contaminated},
              {"contaminated": 0, "complete": True},
              "contaminated or incomplete retrieved-reference coverage"
              if not coverage_ok else "ok"),
        _gate("all_folds_feasible", all_feasible,
              sum(f["feasible"] for f in folds), len(folds),
              "some training fold had no feasible fit"
              if not all_feasible else "ok"),
        _gate("detector_recall", recall_ok, retrieved["recall"], v2.MIN_RECALL,
              "held-out FAIL recall below bar" if not recall_ok else "ok"),
        _gate("detector_false_positives", fp_ok,
              {"fp": retrieved["fp"],
               "fp_rate": retrieved["false_positive_rate"]},
              {"max_fp": 1, "max_fp_rate": v2.MAX_FP_RATE},
              "too many control false positives" if not fp_ok else "ok"),
    ]

    if not coverage_ok:
        readiness = "INCONCLUSIVE"
    elif not (all_feasible and recall_ok and fp_ok):
        readiness = "FAIL"
    else:
        readiness = "PASS"

    fail_n = sum(truth)
    return {
        "policy_version": "retrieved_detector",
        "folds": folds,
        "detector": {
            "denominator": {"n": len(rows), "fail": fail_n,
                            "pass": len(rows) - fail_n},
            "retrieved_guarded": retrieved,
            "baselines": baselines,
        },
        "gates": gates,
        "readiness": readiness,
        "contaminated": contaminated,
    }
