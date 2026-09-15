"""Leave-one-concept-out development evaluation of the v2 policy.

Each case is its own concept here, so leave-one-concept-out is leave-one-case-out
(design 5). For every case we fit detector thresholds and selector margins on all
*other* cases and apply the frozen fit to the held-out case without refitting;
only held-out predictions are aggregated. Fitting once on all rows and relabelling
it cross-validation is exactly the mistake this module refuses to make.

Detector metrics use every judgeable identity; selector and end-to-end metrics use
only ``selector_eligible`` cases, with the excluded denominator and its mechanical
reason reported explicitly. DINO is a fitting/evaluation signal only, never a live
input. Passing every gate is a readiness gate for a *new* holdout -- not evidence
of generalisation.
"""
from __future__ import annotations

from collections.abc import Mapping

from ragregen import c1, metrics
from ragregen import hybrid_verifier as v1
from ragregen import hybrid_verifier_v2 as v2

RARE_COHORT = "bridge"


def _detector_rows(manifest, semantic, reranker) -> dict[str, dict]:
    """One row per judgeable identity (PASS/FAIL), keyed by case id."""
    reranker_cases = reranker.get("cases", {})
    rows: dict[str, dict] = {}
    for cid, meta in manifest["cases"].items():
        truth = meta["identity_truth"]
        if truth not in ("PASS", "FAIL"):
            continue
        if cid not in semantic:
            raise ValueError(f"{cid}: missing semantic decision")
        if cid not in reranker_cases:
            raise ValueError(f"{cid}: missing reranker scores")
        scores = reranker_cases[cid].get("scores", {})
        if "draft" not in scores:
            raise ValueError(f"{cid}: missing reranker draft score")
        draft = scores["draft"]
        rows[cid] = {
            "case_id": cid,
            "truth_fail": truth == "FAIL",
            "semantic_ok": bool(semantic[cid].get("ok")),
            "text_relevance": v2._number(draft, "draft", "text_relevance"),
            "reference_relevance": v2._number(draft, "draft",
                                              "reference_relevance"),
            "cohort": meta["cohort"],
        }
    return rows


def _selector_rows(manifest, reranker, dino) -> dict[str, dict]:
    """One row per eligible case, carrying reranker scores and DINO answers."""
    reranker_cases = reranker.get("cases", {})
    dino_cases = dino.get("cases", {})
    rows: dict[str, dict] = {}
    for cid, meta in manifest["cases"].items():
        if not meta["selector_eligible"]:
            continue
        if cid not in reranker_cases:
            raise ValueError(f"{cid}: missing reranker scores")
        if cid not in dino_cases:
            raise ValueError(f"{cid}: eligible but missing DINO scores")
        rows[cid] = {
            "case_id": cid,
            "cohort": meta["cohort"],
            "scores": reranker_cases[cid].get("scores", {}),
            "dino": dino_cases[cid],
        }
    return rows


def _confusion_dict(y_true, y_pred) -> dict:
    m = c1.confusion(y_true, y_pred)
    fp_rate = m.fp / (m.fp + m.tn) if (m.fp + m.tn) else 0.0
    recall_lo, recall_hi = c1.wilson_ci(m.tp, m.tp + m.fn)
    return {
        "tp": m.tp, "fp": m.fp, "tn": m.tn, "fn": m.fn,
        "recall": m.recall, "false_positive_rate": fp_rate,
        "specificity": m.specificity, "mcc": m.mcc,
        "recall_wilson95": [recall_lo, recall_hi],
    }


def _gate(name, passed, value, threshold, reason) -> dict:
    return {"name": name, "passed": passed, "value": value,
            "threshold": threshold, "reason": reason}


def develop(manifest: Mapping, semantic: Mapping, reranker: Mapping,
            dino: Mapping) -> dict:
    det_rows = _detector_rows(manifest, semantic, reranker)
    sel_rows = _selector_rows(manifest, reranker, dino)
    case_ids = sorted(manifest["cases"])

    folds = []
    cases: dict[str, dict] = {}
    # Aggregated held-out predictions, in sorted case order.
    det_true: list[bool] = []
    det_pred: list[bool] = []
    sem_pred: list[bool] = []
    ref_pred: list[bool] = []
    union_pred: list[bool] = []
    sel_deltas: list[float] = []
    pair_correct: list[bool] = []
    exact_best: list[bool] = []
    cohort_deltas: dict[str, list[float]] = {}

    for cid in case_ids:
        fit_ids: set[str] = set()
        record: dict = {"detector_prediction": None, "selected": None}

        # --- detector fold ---
        if cid in det_rows:
            train = [r for c, r in sorted(det_rows.items()) if c != cid]
            fit_ids.update(r["case_id"] for r in train)
            det_fit = v2.fit_detector(train)
            row = det_rows[cid]
            policy = v2.Policy(det_fit.text_threshold,
                               det_fit.reference_threshold, 0.0, 0.0)
            pred = v2.route_draft(row["semantic_ok"], row["text_relevance"],
                                  row["reference_relevance"], policy)
            det_true.append(row["truth_fail"])
            det_pred.append(pred)
            # Fixed baselines over the same held-out row.
            sem = not row["semantic_ok"]
            ref = row["reference_relevance"] < v1.REFERENCE_THRESHOLD
            sem_pred.append(sem)
            ref_pred.append(ref)
            union_pred.append(sem or ref)
            record["detector_prediction"] = bool(pred)
            record["truth"] = "FAIL" if row["truth_fail"] else "PASS"
            record["detector_fit"] = {
                "text_threshold": det_fit.text_threshold,
                "reference_threshold": det_fit.reference_threshold,
                "feasible": det_fit.feasible}

        # --- selector fold ---
        if cid in sel_rows:
            train = [r for c, r in sorted(sel_rows.items()) if c != cid]
            fit_ids.update(r["case_id"] for r in train)
            sel_fit = v2.fit_selector(train)
            row = sel_rows[cid]
            policy = v2.Policy(0.0, 0.0, sel_fit.text_margin,
                               sel_fit.reference_margin)
            selected = v2.select_repair(row["scores"], policy)
            d = row["dino"]
            draft_dino = v2._finite(d["draft"], "dino draft")
            delta = v2._finite(d[selected], "dino") - draft_dino
            sel_deltas.append(delta)
            cohort_deltas.setdefault(row["cohort"], []).append(delta)
            for label in row["scores"]:
                if label == "draft":
                    continue
                predicted = v2._eligible(row["scores"], label, policy)
                actual = v2._finite(d[label], "dino") > draft_dino
                pair_correct.append(predicted == actual)
            available = [label for label in row["scores"] if label in d]
            oracle = max(available, key=lambda label: v2._finite(d[label], "dino"))
            exact_best.append(selected == oracle)
            record["selected"] = selected
            record["dino_delta"] = delta
            record["cohort"] = row["cohort"]
            record["selector_fit"] = {
                "text_margin": sel_fit.text_margin,
                "reference_margin": sel_fit.reference_margin,
                "feasible": sel_fit.feasible}

        cases[cid] = record
        folds.append({"case_id": cid, "fit_case_ids": sorted(fit_ids),
                      **record})

    # --- detector aggregation ---
    v2_conf = _confusion_dict(det_true, det_pred)
    baselines = {
        "semantic_only": _basic_confusion(det_true, sem_pred),
        "reference_only": _basic_confusion(det_true, ref_pred),
        "v1_union": _basic_confusion(det_true, union_pred),
    }
    fail_n = sum(det_true)
    pass_n = len(det_true) - fail_n
    detector = {
        "denominator": {"n": len(det_true), "fail": fail_n, "pass": pass_n},
        "v2_guarded": v2_conf,
        "baselines": baselines,
    }

    # --- selector aggregation ---
    excluded = {cid: meta["eligibility_reason"]
                for cid, meta in manifest["cases"].items()
                if not meta["selector_eligible"]}
    sign = sum(pair_correct) / len(pair_correct) if pair_correct else 0.0
    harmful = sum(1 for d in sel_deltas if d < v2.HARM_MARGIN)
    mean_delta = sum(sel_deltas) / len(sel_deltas) if sel_deltas else 0.0
    exact = sum(exact_best) / len(exact_best) if exact_best else 0.0
    cohorts = {}
    for cohort, deltas in sorted(cohort_deltas.items()):
        interval = metrics.paired_bootstrap_ci(deltas)
        cohorts[cohort] = {
            "n": len(deltas),
            "mean_dino_delta": sum(deltas) / len(deltas),
            "dino_delta_95ci": list(interval) if interval else None,
            "improved": sum(1 for d in deltas if d > 0),
            "unchanged": sum(1 for d in deltas if d == 0),
            "worsened": sum(1 for d in deltas if d < 0),
        }
    selector = {
        "denominator": {"n_eligible": len(sel_rows), "excluded": excluded},
        "comparisons": len(pair_correct),
        "sign_accuracy": sign,
        "harmful": harmful,
        "mean_dino_delta": mean_delta,
        "exact_best_rate": exact,
        "cohorts": cohorts,
    }

    gates = _readiness_gates(v2_conf, sign, harmful, cohorts)
    readiness = _readiness(gates)

    return {
        "protocol_status": manifest.get("protocol_status"),
        "dataset": manifest.get("dataset"),
        "folds": folds,
        "detector": detector,
        "selector": selector,
        "gates": gates,
        # The sixth spec-5 gate -- "no selected common case with material
        # identity/layout damage" -- cannot be judged from numbers. It is a
        # mandatory *manual* step that gates a numeric PASS before any Phase B
        # work, reported here rather than folded into the automated verdict.
        "manual_review": {
            "name": "no_visual_damage",
            "required": True,
            "resolved": False,
            "reason": "requires exhaustive manual visual review of selected "
                      "common-control outputs before promotion",
        },
        "readiness": readiness,
        "cases": cases,
    }


def _basic_confusion(y_true, y_pred) -> dict:
    m = c1.confusion(y_true, y_pred)
    fp_rate = m.fp / (m.fp + m.tn) if (m.fp + m.tn) else 0.0
    return {"tp": m.tp, "fp": m.fp, "tn": m.tn, "fn": m.fn,
            "recall": m.recall, "false_positive_rate": fp_rate, "mcc": m.mcc}


def _readiness_gates(v2_conf, sign, harmful, cohorts) -> list[dict]:
    rare = cohorts.get(RARE_COHORT)
    rare_mean = rare["mean_dino_delta"] if rare else None
    return [
        _gate("detector_recall", v2_conf["recall"] >= v2.MIN_RECALL,
              v2_conf["recall"], v2.MIN_RECALL,
              "held-out FAIL recall below bar"
              if v2_conf["recall"] < v2.MIN_RECALL else "ok"),
        _gate("detector_false_positives",
              v2_conf["fp"] <= v2.MAX_HARMFUL
              or v2_conf["false_positive_rate"] <= v2.MAX_FP_RATE,
              {"fp": v2_conf["fp"], "fp_rate": v2_conf["false_positive_rate"]},
              {"max_fp": 1, "max_fp_rate": v2.MAX_FP_RATE},
              "too many control false positives"
              if not (v2_conf["fp"] <= 1
                      or v2_conf["false_positive_rate"] <= v2.MAX_FP_RATE)
              else "ok"),
        _gate("selector_sign_accuracy", sign >= v2.MIN_SIGN_ACCURACY,
              sign, v2.MIN_SIGN_ACCURACY,
              "selector sign accuracy below bar"
              if sign < v2.MIN_SIGN_ACCURACY else "ok"),
        _gate("selector_harmful", harmful <= v2.MAX_HARMFUL,
              harmful, v2.MAX_HARMFUL,
              "too many harmful selections"
              if harmful > v2.MAX_HARMFUL else "ok"),
        _gate("rare_mean_dino_delta",
              None if rare_mean is None else rare_mean > 0.0,
              rare_mean, 0.0,
              "no rare cohort" if rare_mean is None
              else ("rare mean delta not positive"
                    if rare_mean <= 0.0 else "ok")),
    ]


def _readiness(gates) -> str:
    if any(g["passed"] is False for g in gates):
        return "FAIL"
    if any(g["passed"] is None for g in gates):
        return "INCONCLUSIVE"
    return "PASS"
