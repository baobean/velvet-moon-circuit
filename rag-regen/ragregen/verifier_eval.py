"""Auditable statistics for a hard-decision semantic verifier."""
from __future__ import annotations

from ragregen import c1


def evaluate(rows, stream_b: dict, column: str = "verdict_identity") -> dict:
    kept, y_true, excluded = c1.ground_truth(rows, column)
    missing = [r.case_id for r in kept if r.case_id not in stream_b]
    if missing:
        raise KeyError(f"missing semantic decisions for: {', '.join(missing)}")

    y_pred = [not bool(stream_b[r.case_id]["ok"]) for r in kept]
    cm = c1.confusion(y_true, y_pred)
    fail_n = sum(y_true)
    pass_n = len(y_true) - fail_n
    recall_ci = c1.wilson_ci(cm.tp, fail_n)
    specificity_ci = c1.wilson_ci(cm.tn, pass_n)
    cases = []
    for row, truth_fail, pred_fail in zip(kept, y_true, y_pred):
        rec = stream_b[row.case_id]
        cases.append({
            "case_id": row.case_id,
            "truth": "FAIL" if truth_fail else "PASS",
            "decision": "FAIL" if pred_fail else "PASS",
            "correct": truth_fail == pred_fail,
            "false_negative": truth_fail and not pred_fail,
            "false_positive": not truth_fail and pred_fail,
            "degenerate": bool(rec.get("degenerate")),
            "issues": rec.get("issues") or [],
            "raw": rec.get("raw", ""),
            "notes": row.notes,
        })

    return {
        "column": column, "n": len(kept), "excluded": excluded,
        "non_pass": {
            "n": fail_n, "detected": cm.tp, "missed": cm.fn,
            "recall": cm.recall, "false_negative_rate": 1.0 - cm.recall,
            "recall_95ci": list(recall_ci),
        },
        "pass": {
            "n": pass_n, "accepted": cm.tn, "rejected": cm.fp,
            "specificity": cm.specificity,
            "false_positive_rate": 1.0 - cm.specificity,
            "specificity_95ci": list(specificity_ci),
        },
        "overall": {
            "tp": cm.tp, "fp": cm.fp, "tn": cm.tn, "fn": cm.fn,
            "precision": cm.precision, "recall": cm.recall,
            "specificity": cm.specificity, "f1": cm.f1,
            "balanced_accuracy": cm.balanced_accuracy, "mcc": cm.mcc,
            "accuracy": cm.accuracy,
            "degenerate": sum(c["degenerate"] for c in cases),
        },
        "cases": cases,
    }


def render(result: dict) -> str:
    nonpass, passed, overall = (result["non_pass"], result["pass"],
                                result["overall"])
    lines = [
        f"# VLM verifier decisions — `{result['column']}`", "",
        f"{result['n']} labeled cases; {result['excluded']} excluded.", "",
        "| population | n | correct | incorrect | rate | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
        (f"| non-pass (recall) | {nonpass['n']} | {nonpass['detected']} | "
         f"{nonpass['missed']} | {nonpass['recall']:.3f} | "
         f"[{nonpass['recall_95ci'][0]:.3f}, "
         f"{nonpass['recall_95ci'][1]:.3f}] |"),
        (f"| pass (specificity) | {passed['n']} | {passed['accepted']} | "
         f"{passed['rejected']} | {passed['specificity']:.3f} | "
         f"[{passed['specificity_95ci'][0]:.3f}, "
         f"{passed['specificity_95ci'][1]:.3f}] |"),
        "", (f"Balanced accuracy **{overall['balanced_accuracy']:.3f}**; "
              f"MCC **{overall['mcc']:+.3f}**; degenerate replies "
              f"**{overall['degenerate']}**."), "",
        "## Per-case final decisions", "",
        "| case | truth | VLM | outcome |",
        "|---|---|---|---|",
    ]
    for case in result["cases"]:
        outcome = "correct" if case["correct"] else (
            "FALSE NEGATIVE" if case["false_negative"] else "FALSE POSITIVE")
        lines.append(
            f"| {case['case_id']} | {case['truth']} | {case['decision']} | "
            f"{outcome} |")
    return "\n".join(lines) + "\n"
