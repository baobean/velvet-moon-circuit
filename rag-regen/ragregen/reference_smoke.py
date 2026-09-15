"""Pure scoring and cross-validation for reference-based verifier smokes.

The model worker persists normalized embeddings.  This module deliberately
loads no model: every threshold and decision can be audited or recomputed on
CPU without paying inference cost again.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ragregen import verifier_eval


FEATURES = ("positive_similarity", "prototype_margin", "text_margin")


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(a)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("embedding must have a finite non-zero norm")
    return a / norm


def case_features(artifact: dict) -> list[dict]:
    """Reduce raw embeddings to three predeclared continuous signals.

    ``prototype_margin`` is the primary signal: similarity to the case's
    positive reference centroid minus similarity to the nearest other case
    centroid.  Low values indicate a fine-identity failure.
    """
    cases = artifact.get("cases", {})
    if not cases:
        raise ValueError("embedding artifact has no cases")

    prototypes = {}
    for cid, rec in cases.items():
        refs = rec.get("reference_embeddings", [])
        if not refs:
            raise ValueError(f"{cid}: no reference embeddings")
        prototypes[cid] = _unit(np.mean([_unit(v) for v in refs], axis=0))

    rows = []
    for cid, rec in cases.items():
        image = _unit(rec["draft_embedding"])
        positive = float(image @ prototypes[cid])
        negatives = [float(image @ proto) for other, proto in prototypes.items()
                     if other != cid]
        if not negatives:
            raise ValueError("prototype margin requires at least two cases")
        fine = float(image @ _unit(rec["fine_text_embedding"]))
        coarse = float(image @ _unit(rec["coarse_text_embedding"]))
        rows.append({
            "case_id": cid,
            "truth": str(rec["truth"]).upper(),
            "positive_similarity": positive,
            "nearest_negative_similarity": max(negatives),
            "prototype_margin": positive - max(negatives),
            "fine_similarity": fine,
            "coarse_similarity": coarse,
            "text_margin": fine - coarse,
        })
    return rows


@dataclass(frozen=True)
class _ThresholdResult:
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


def fit_threshold(rows: list[dict], feature: str) -> _ThresholdResult:
    """Fit the rule ``score < threshold => FAIL`` with deterministic ties."""
    values = sorted({float(r[feature]) for r in rows})
    if not values:
        raise ValueError("cannot fit a threshold without rows")
    eps = max(1e-9, (values[-1] - values[0]) * 1e-6)
    candidates = [values[0] - eps]
    candidates += [(a + b) / 2 for a, b in zip(values, values[1:])]
    candidates += [values[-1] + eps]
    truth = [r["truth"] == "FAIL" for r in rows]
    fitted = []
    for threshold in candidates:
        pred = [float(r[feature]) < threshold for r in rows]
        balanced, specificity = _balanced(truth, pred)
        fitted.append(_ThresholdResult(threshold, balanced, specificity))
    # Prefer specificity, then the lower/more conservative threshold when
    # balanced accuracy ties. This prevents a smoke gate from winning by
    # needlessly routing good images into repair.
    return max(fitted, key=lambda x: (x.balanced_accuracy, x.specificity,
                                      -x.threshold))


def evaluate_feature(rows: list[dict], feature: str) -> dict:
    """In-sample diagnostic plus leave-one-concept-out hard decisions."""
    if feature not in FEATURES:
        raise ValueError(f"unknown feature {feature!r}")
    fitted = fit_threshold(rows, feature)
    in_sample = [float(r[feature]) < fitted.threshold for r in rows]

    loo, thresholds = [], []
    for i, row in enumerate(rows):
        train = rows[:i] + rows[i + 1:]
        fit = fit_threshold(train, feature)
        thresholds.append(fit.threshold)
        loo.append(float(row[feature]) < fit.threshold)

    def report(predictions):
        stream = {
            r["case_id"]: {"ok": not fail, "degenerate": False,
                            "raw": f"{feature}={r[feature]:.6f}"}
            for r, fail in zip(rows, predictions)
        }
        labels = [
            type("Label", (), {"case_id": r["case_id"],
                                "verdict_identity": r["truth"],
                                "notes": ""})()
            for r in rows
        ]
        return verifier_eval.evaluate(labels, stream, "verdict_identity")

    return {
        "feature": feature,
        "direction": "FAIL when score < threshold",
        "fitted_threshold": fitted.threshold,
        "in_sample": report(in_sample),
        "leave_one_out": report(loo),
        "leave_one_out_thresholds": thresholds,
    }


def evaluate(artifact: dict) -> dict:
    rows = case_features(artifact)
    arms = {feature: evaluate_feature(rows, feature) for feature in FEATURES}
    primary = arms["prototype_margin"]["leave_one_out"]["overall"]
    gate = {
        "required": {"tp": 11, "max_fp": 1},
        "observed": {"tp": primary["tp"], "fp": primary["fp"]},
        "passed": primary["tp"] >= 11 and primary["fp"] <= 1,
    }
    return {"model": artifact.get("model", "unknown"), "n": len(rows),
            "rows": rows, "arms": arms, "primary_gate": gate}


def render(result: dict) -> str:
    lines = [f"# Reference verifier smoke — {result['model']}", "",
             f"{result['n']} labeled drafts. Primary rule: candidate-to-own "
             "reference similarity minus nearest competing prototype; "
             "decisions are leave-one-concept-out.", "",
             "| signal | TP | FP | TN | FN | recall | specificity | bal-acc | MCC |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in FEATURES:
        m = result["arms"][name]["leave_one_out"]["overall"]
        lines.append(
            f"| {name} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} "
            f"| {m['recall']:.3f} | {m['specificity']:.3f} "
            f"| {m['balanced_accuracy']:.3f} | {m['mcc']:+.3f} |")
    gate = result["primary_gate"]
    lines += ["", f"Primary promotion gate: **{'PASS' if gate['passed'] else 'FAIL'}** "
              f"— observed TP={gate['observed']['tp']} and FP={gate['observed']['fp']}; "
              f"requires TP≥{gate['required']['tp']} and FP≤{gate['required']['max_fp']}.",
              "", "This oracle-reference smoke is a ceiling, not a deployable retrieval result."]
    return "\n".join(lines)


def _scores_by_case(artifact: dict, feature: str) -> dict[str, dict[str, float]]:
    """Feature values for draft and every cached candidate."""
    cases = artifact["cases"]
    prototypes = {
        cid: _unit(np.mean([_unit(v) for v in rec["reference_embeddings"]], axis=0))
        for cid, rec in cases.items()
    }
    out = {}
    for cid, rec in cases.items():
        images = {"draft": rec["draft_embedding"],
                  **rec.get("candidate_embeddings", {})}
        values = {}
        for label, vector in images.items():
            image = _unit(vector)
            positive = float(image @ prototypes[cid])
            nearest = max(float(image @ p) for other, p in prototypes.items()
                          if other != cid)
            fine = float(image @ _unit(rec["fine_text_embedding"]))
            coarse = float(image @ _unit(rec["coarse_text_embedding"]))
            values[label] = {
                "positive_similarity": positive,
                "prototype_margin": positive - nearest,
                "text_margin": fine - coarse,
            }[feature]
        out[cid] = values
    return out


def evaluate_pairwise(artifact: dict, dino_scores: dict,
                      harm_margin: float = -0.02) -> dict:
    """Does an embedding gain choose the held-out-DINO-better image?

    DINO scores are a separate artifact and use the reference reserved by
    ``metrics.split_refs``.  The verifier embeddings use only the edit pool.
    """
    arms = {}
    for feature in FEATURES:
        scores = _scores_by_case(artifact, feature)
        comparisons, selections = [], []
        for cid, dino in dino_scores.get("cases", {}).items():
            if cid not in scores or "draft" not in dino:
                continue
            available = {k: v for k, v in scores[cid].items() if k in dino}
            if len(available) <= 1:
                continue
            draft_feature = available["draft"]
            draft_dino = float(dino["draft"])
            for label, value in available.items():
                if label == "draft":
                    continue
                gain = float(value - draft_feature)
                delta = float(dino[label] - draft_dino)
                comparisons.append({"case_id": cid, "candidate": label,
                                    "feature_gain": gain, "dino_delta": delta,
                                    "correct_sign": (gain > 0) == (delta > 0)})

            best_candidate = max((k for k in available if k != "draft"),
                                 key=lambda k: available[k])
            selected = (best_candidate if available[best_candidate] > draft_feature
                        else "draft")
            oracle = max(available, key=lambda k: float(dino[k]))
            selected_delta = float(dino[selected] - draft_dino)
            selections.append({
                "case_id": cid, "selected": selected, "oracle": oracle,
                "selected_dino_delta": selected_delta,
                "exact_oracle": selected == oracle,
                "harmful": selected_delta < harm_margin,
            })

        correct = sum(r["correct_sign"] for r in comparisons)
        harmful = sum(r["harmful"] for r in selections)
        arms[feature] = {
            "n_comparisons": len(comparisons),
            "pairwise_sign_accuracy": (correct / len(comparisons)
                                       if comparisons else None),
            "n_cases": len(selections),
            "exact_oracle_rate": (sum(r["exact_oracle"] for r in selections)
                                  / len(selections) if selections else None),
            "mean_selected_dino_delta": (float(np.mean(
                [r["selected_dino_delta"] for r in selections]))
                if selections else None),
            "harmful_selections": harmful,
            "comparisons": comparisons, "selections": selections,
        }

    primary = arms["prototype_margin"]
    gate = {
        "required": {"pairwise_sign_accuracy": 0.75,
                     "max_harmful_selections": 1},
        "observed": {"pairwise_sign_accuracy": primary["pairwise_sign_accuracy"],
                     "harmful_selections": primary["harmful_selections"]},
        "passed": bool(primary["pairwise_sign_accuracy"] is not None
                       and primary["pairwise_sign_accuracy"] >= 0.75
                       and primary["harmful_selections"] <= 1),
    }
    return {"model": artifact.get("model", "unknown"), "arms": arms,
            "primary_gate": gate, "harm_margin": harm_margin}


def render_pairwise(result: dict) -> str:
    lines = [f"# Pairwise repair selector smoke — {result['model']}", "",
             "Verifier references exclude the held-out DINO reference. The "
             "selector keeps the highest-scoring repair only when it beats "
             "the draft.", "",
             "| signal | comparisons | sign accuracy | cases | exact oracle | mean DINO delta | harmful |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for feature in FEATURES:
        arm = result["arms"][feature]
        acc = arm["pairwise_sign_accuracy"]
        exact = arm["exact_oracle_rate"]
        delta = arm["mean_selected_dino_delta"]
        lines.append(
            f"| {feature} | {arm['n_comparisons']} | "
            f"{acc:.3f} | {arm['n_cases']} | {exact:.3f} | {delta:+.3f} "
            f"| {arm['harmful_selections']} |")
    gate = result["primary_gate"]
    lines += ["", f"Primary promotion gate: **{'PASS' if gate['passed'] else 'FAIL'}** "
              f"— sign accuracy {gate['observed']['pairwise_sign_accuracy']:.3f}, "
              f"harmful selections {gate['observed']['harmful_selections']}; "
              "requires accuracy ≥0.750 and harmful selections ≤1."]
    return "\n".join(lines)
