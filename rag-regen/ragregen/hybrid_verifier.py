"""Pure frozen policy and evaluation for the hybrid verifier holdout."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping

from ragregen import c1, metrics


REFERENCE_THRESHOLD = 0.30078125
TEXT_MARGIN = 0.02
HARM_MARGIN = -0.02
MIN_RECALL = 0.733
MAX_FALSE_POSITIVES = 1
MIN_SIGN_ACCURACY = 0.75
MAX_HARMFUL = 1

_ATTEMPT = re.compile(r"attempt_(\d+)$")


def _number(record: Mapping, label: str, signal: str) -> float:
    try:
        value = float(record[signal])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid {signal}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label}: invalid {signal}={value}")
    return value


def _attempt_number(label: str) -> int:
    match = _ATTEMPT.fullmatch(label)
    if not match:
        raise ValueError(f"invalid candidate label {label!r}")
    return int(match.group(1))


def route_draft(semantic_ok: bool, reference_relevance: float) -> bool:
    reference = float(reference_relevance)
    if not math.isfinite(reference):
        raise ValueError(f"invalid draft reference_relevance={reference}")
    return not bool(semantic_ok) or reference < REFERENCE_THRESHOLD


def _eligible(scores: Mapping[str, Mapping], label: str) -> bool:
    draft = scores["draft"]
    candidate = scores[label]
    text_gain = (_number(candidate, label, "text_relevance")
                 - _number(draft, "draft", "text_relevance"))
    reference_gain = (_number(candidate, label, "reference_relevance")
                      - _number(draft, "draft", "reference_relevance"))
    # Decimal fixture values such as 0.42 - 0.40 land one binary ulp below
    # 0.02. The policy boundary is inclusive, so tolerate only roundoff at
    # that boundary rather than accidentally making it strict.
    return text_gain + 1e-12 >= TEXT_MARGIN and reference_gain > 0.0


def select_repair(scores: Mapping[str, Mapping]) -> str:
    if "draft" not in scores:
        raise ValueError("scores are missing draft")
    _number(scores["draft"], "draft", "text_relevance")
    _number(scores["draft"], "draft", "reference_relevance")

    eligible = []
    for label, record in scores.items():
        if label == "draft":
            continue
        attempt = _attempt_number(label)
        # Validate both signals even when the first eligibility condition
        # fails, so a corrupt artifact cannot silently act like a rejection.
        text = _number(record, label, "text_relevance")
        reference = _number(record, label, "reference_relevance")
        if _eligible(scores, label):
            eligible.append((text, reference, -attempt, label))
    return max(eligible)[-1] if eligible else "draft"


def _confusion(truth_fail: list[bool], predicted_fail: list[bool]) -> dict:
    result = c1.confusion(truth_fail, predicted_fail)
    return {
        "tp": result.tp, "fp": result.fp,
        "tn": result.tn, "fn": result.fn,
        "recall": result.recall,
        "specificity": result.specificity,
        "balanced_accuracy": result.balanced_accuracy,
        "mcc": result.mcc,
    }


def evaluate(labels: Mapping[str, str], semantic: Mapping[str, Mapping],
             reranker: Mapping, dino: Mapping,
             cohorts: Mapping[str, str]) -> dict:
    truth_fail = []
    semantic_fail = []
    reranker_fail = []
    union_fail = []
    cases = {}
    pair_correct = []
    selected_deltas = []
    exact_best = []
    cohort_deltas: dict[str, list[float]] = {}

    reranker_cases = reranker.get("cases", {})
    dino_cases = dino.get("cases", {})
    for case_id, raw_truth in labels.items():
        truth = str(raw_truth).strip().upper()
        if truth == "EXCLUDE" or not truth:
            continue
        if truth not in ("PASS", "FAIL"):
            raise ValueError(f"{case_id}: invalid label {raw_truth!r}")
        if case_id not in semantic:
            raise ValueError(f"{case_id}: missing semantic decision")
        if case_id not in reranker_cases:
            raise ValueError(f"{case_id}: missing reranker scores")
        if case_id not in dino_cases:
            raise ValueError(f"{case_id}: missing DINO scores")
        if case_id not in cohorts:
            raise ValueError(f"{case_id}: missing cohort")

        scores = reranker_cases[case_id].get("scores", {})
        if "draft" not in scores:
            raise ValueError(f"{case_id}: missing reranker draft score")
        draft_reference = _number(
            scores["draft"], "draft", "reference_relevance")
        sem_fail = not bool(semantic[case_id].get("ok"))
        ref_fail = draft_reference < REFERENCE_THRESHOLD
        routed = sem_fail or ref_fail
        selected = select_repair(scores) if routed else "draft"

        answer = dino_cases[case_id]
        if "draft" not in answer or selected not in answer:
            raise ValueError(
                f"{case_id}: DINO scores do not contain draft and {selected}")
        delta = float(answer[selected]) - float(answer["draft"])
        if not math.isfinite(delta):
            raise ValueError(f"{case_id}: non-finite DINO delta")

        attempt_labels = [label for label in scores if label != "draft"]
        for label in attempt_labels:
            if label not in answer:
                raise ValueError(f"{case_id}: DINO score missing {label}")
            predicted_better = _eligible(scores, label)
            actually_better = float(answer[label]) > float(answer["draft"])
            pair_correct.append(predicted_better == actually_better)
        available = [label for label in scores if label in answer]
        oracle = max(available, key=lambda label: float(answer[label]))
        exact_best.append(selected == oracle)
        selected_deltas.append(delta)
        cohort_deltas.setdefault(cohorts[case_id], []).append(delta)

        is_fail = truth == "FAIL"
        truth_fail.append(is_fail)
        semantic_fail.append(sem_fail)
        reranker_fail.append(ref_fail)
        union_fail.append(routed)
        cases[case_id] = {
            "truth": truth,
            "semantic": "FAIL" if sem_fail else "PASS",
            "reranker": "FAIL" if ref_fail else "PASS",
            "routed": routed,
            "hybrid": selected,
            "dino_delta": delta,
            "cohort": cohorts[case_id],
        }

    detector = {
        "semantic": _confusion(truth_fail, semantic_fail),
        "reranker": _confusion(truth_fail, reranker_fail),
        "union": _confusion(truth_fail, union_fail),
    }
    selector = {
        "comparisons": len(pair_correct),
        "sign_accuracy": (sum(pair_correct) / len(pair_correct)
                          if pair_correct else 0.0),
        "cases": len(selected_deltas),
        "exact_dino_best": (sum(exact_best) / len(exact_best)
                            if exact_best else 0.0),
        "mean_dino_delta": (sum(selected_deltas) / len(selected_deltas)
                            if selected_deltas else 0.0),
        "harmful": sum(delta < HARM_MARGIN for delta in selected_deltas),
    }
    cohort_report = {}
    for cohort, deltas in cohort_deltas.items():
        interval = metrics.paired_bootstrap_ci(deltas)
        cohort_report[cohort] = {
            "n": len(deltas),
            "mean_dino_delta": sum(deltas) / len(deltas),
            "dino_delta_95ci": list(interval) if interval else None,
            "improved": sum(delta > 0 for delta in deltas),
            "unchanged": sum(delta == 0 for delta in deltas),
            "worsened": sum(delta < 0 for delta in deltas),
        }

    pass_n = sum(not value for value in truth_fail)
    fail_n = sum(truth_fail)
    union = detector["union"]
    detector_reason = None
    if not fail_n:
        detector_reason = "missing FAIL stratum"
    elif not pass_n:
        detector_reason = "missing PASS stratum"
    detector_passed = (detector_reason is None
                       and union["recall"] >= MIN_RECALL
                       and union["fp"] <= MAX_FALSE_POSITIVES)
    selector_passed = (bool(pair_correct)
                       and selector["sign_accuracy"] >= MIN_SIGN_ACCURACY
                       and selector["harmful"] <= MAX_HARMFUL)

    return {
        "constants": {
            "reference_threshold": REFERENCE_THRESHOLD,
            "text_margin": TEXT_MARGIN,
            "harm_margin": HARM_MARGIN,
        },
        "n": len(cases),
        "detector": detector,
        "selector": selector,
        "cohorts": cohort_report,
        "cases": cases,
        "gates": {
            "detector": {"passed": detector_passed,
                         "reason": detector_reason},
            "selector": {"passed": selector_passed},
        },
    }
