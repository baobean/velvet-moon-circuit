"""Pure evaluation for pointwise multimodal reranker smoke artifacts."""
from __future__ import annotations

from dataclasses import dataclass

from ragregen import verifier_eval


SIGNALS = ("text_relevance", "reference_relevance")


@dataclass(frozen=True)
class _Fit:
    threshold: float
    balanced_accuracy: float
    specificity: float


def _balanced(truth_fail: list[bool], predicted_fail: list[bool]):
    tp = sum(t and p for t, p in zip(truth_fail, predicted_fail))
    fn = sum(t and not p for t, p in zip(truth_fail, predicted_fail))
    fp = sum(not t and p for t, p in zip(truth_fail, predicted_fail))
    tn = sum(not t and not p for t, p in zip(truth_fail, predicted_fail))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return (recall + specificity) / 2, specificity


def _fit(rows: list[dict], signal: str) -> _Fit:
    values = sorted({float(row[signal]) for row in rows})
    if not values:
        raise ValueError("cannot fit a threshold without rows")
    eps = max(1e-9, (values[-1] - values[0]) * 1e-6)
    candidates = [values[0] - eps]
    candidates += [(a + b) / 2 for a, b in zip(values, values[1:])]
    candidates += [values[-1] + eps]
    truth = [row["truth"] == "FAIL" for row in rows]
    fits = []
    for threshold in candidates:
        prediction = [float(row[signal]) < threshold for row in rows]
        balanced, specificity = _balanced(truth, prediction)
        fits.append(_Fit(threshold, balanced, specificity))
    return max(fits, key=lambda fit: (fit.balanced_accuracy, fit.specificity,
                                      -fit.threshold))


def _report(rows: list[dict], signal: str, predictions: list[bool]) -> dict:
    stream = {
        row["case_id"]: {
            "ok": not fail,
            "degenerate": False,
            "raw": f"{signal}={row[signal]:.6f}",
        }
        for row, fail in zip(rows, predictions)
    }
    labels = [
        type("Label", (), {
            "case_id": row["case_id"],
            "verdict_identity": row["truth"],
            "notes": "",
        })()
        for row in rows
    ]
    return verifier_eval.evaluate(labels, stream, "verdict_identity")


def evaluate_absolute(artifact: dict) -> dict:
    rows = []
    for case_id, record in artifact.get("cases", {}).items():
        draft = record.get("scores", {}).get("draft")
        if not draft:
            continue
        rows.append({"case_id": case_id, "truth": record["truth"], **draft})
    if len(rows) < 2:
        raise ValueError("absolute evaluation requires at least two scored drafts")

    arms = {}
    for signal in SIGNALS:
        predictions, thresholds = [], []
        for index, row in enumerate(rows):
            fit = _fit(rows[:index] + rows[index + 1:], signal)
            thresholds.append(fit.threshold)
            predictions.append(float(row[signal]) < fit.threshold)
        arms[signal] = {
            "leave_one_out": _report(rows, signal, predictions),
            "leave_one_out_thresholds": thresholds,
        }
    primary = arms["reference_relevance"]["leave_one_out"]["overall"]
    gate = {
        "required": {"tp": 11, "max_fp": 1},
        "observed": {"tp": primary["tp"], "fp": primary["fp"]},
        "passed": primary["tp"] >= 11 and primary["fp"] <= 1,
    }
    return {"model": artifact.get("model", "unknown"), "n": len(rows),
            "rows": rows, "arms": arms, "primary_gate": gate}


def evaluate_pairwise(artifact: dict, dino_scores: dict,
                      harm_margin: float = -0.02) -> dict:
    arms = {}
    for signal in SIGNALS:
        comparisons, selections = [], []
        for case_id, dino in dino_scores.get("cases", {}).items():
            scores = artifact.get("cases", {}).get(case_id, {}).get("scores", {})
            if "draft" not in scores or "draft" not in dino:
                continue
            for label, score in scores.items():
                if label == "draft" or label not in dino:
                    continue
                predicted_better = score[signal] > scores["draft"][signal]
                actually_better = dino[label] > dino["draft"]
                comparisons.append(predicted_better == actually_better)
            available = {label: score[signal] for label, score in scores.items()
                         if label in dino}
            selected = max(available, key=available.get)
            oracle = max(available, key=lambda label: dino[label])
            delta = float(dino[selected] - dino["draft"])
            selections.append({"case_id": case_id, "selected": selected,
                               "oracle": oracle, "dino_delta": delta})
        n = len(comparisons)
        arms[signal] = {
            "comparisons": n,
            "sign_accuracy": sum(comparisons) / n if n else 0.0,
            "cases": len(selections),
            "exact_oracle": (sum(row["selected"] == row["oracle"]
                                 for row in selections) / len(selections)
                             if selections else 0.0),
            "mean_dino_delta": (sum(row["dino_delta"] for row in selections)
                                / len(selections) if selections else 0.0),
            "harmful": sum(row["dino_delta"] < harm_margin for row in selections),
            "selections": selections,
        }
    primary = arms["reference_relevance"]
    gate = {
        "required": {"min_sign_accuracy": 0.75, "max_harmful": 1},
        "observed": {"sign_accuracy": primary["sign_accuracy"],
                     "harmful": primary["harmful"]},
        "passed": (primary["sign_accuracy"] >= 0.75
                   and primary["harmful"] <= 1),
    }
    return {"model": artifact.get("model", "unknown"), "arms": arms,
            "primary_gate": gate}


def render_absolute(result: dict) -> str:
    lines = [f"# Reranker verifier smoke — {result['model']}", "",
             f"{result['n']} labeled drafts; decisions are leave-one-concept-out.", "",
             "| signal | TP | FP | TN | FN | recall | specificity | bal-acc | MCC |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for signal in SIGNALS:
        metric = result["arms"][signal]["leave_one_out"]["overall"]
        lines.append(
            f"| {signal} | {metric['tp']} | {metric['fp']} | {metric['tn']} "
            f"| {metric['fn']} | {metric['recall']:.3f} "
            f"| {metric['specificity']:.3f} "
            f"| {metric['balanced_accuracy']:.3f} | {metric['mcc']:+.3f} |")
    gate = result["primary_gate"]
    lines += ["", f"Primary promotion gate: **{'PASS' if gate['passed'] else 'FAIL'}** "
              f"— TP={gate['observed']['tp']}, FP={gate['observed']['fp']}; "
              f"requires TP≥{gate['required']['tp']}, "
              f"FP≤{gate['required']['max_fp']}."]
    return "\n".join(lines)


def render_pairwise(result: dict) -> str:
    lines = [f"# Pairwise reranker smoke — {result['model']}", "",
             "The edit reference is visible to the reference arm; held-out DINO "
             "references are used only as the answer key.", "",
             "| signal | comparisons | sign accuracy | cases | exact oracle | mean DINO delta | harmful |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for signal in SIGNALS:
        arm = result["arms"][signal]
        lines.append(
            f"| {signal} | {arm['comparisons']} | {arm['sign_accuracy']:.3f} "
            f"| {arm['cases']} | {arm['exact_oracle']:.3f} "
            f"| {arm['mean_dino_delta']:+.3f} | {arm['harmful']} |")
    gate = result["primary_gate"]
    lines += ["", f"Primary promotion gate: **{'PASS' if gate['passed'] else 'FAIL'}** "
              f"— sign accuracy {gate['observed']['sign_accuracy']:.3f}, "
              f"harmful selections {gate['observed']['harmful']}; requires "
              f"accuracy ≥{gate['required']['min_sign_accuracy']:.3f}, "
              f"harmful ≤{gate['required']['max_harmful']}."]
    return "\n".join(lines)
