"""Pure v2 policy and deterministic leave-one-concept-out fitters.

Two changes from :mod:`ragregen.hybrid_verifier` (v1), and nothing else:

* Routing is a *guarded conjunction* (design 4.1). Semantic failure still routes
  on its own, but low reference relevance only routes when text relevance is
  also low -- the smallest change that removes v1's five reference-only false
  positives.
* Selection admits a nonzero, inclusive ``reference_margin`` (design 4.2).

Every decision is pure. The fitters are deterministic: sorted candidate grids,
order-independent metrics, and a total-order objective, so a shuffled input
yields an identical fit. DINO is used only to fit/evaluate the development
policy; it is never a live routing or selection input.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: Development readiness bars, held at v1's values so failure cannot hide behind
#: a moved target (design 5).
MIN_RECALL = 0.733
MAX_FP_RATE = 0.0834
MIN_SIGN_ACCURACY = 0.750
MAX_HARMFUL = 1
HARM_MARGIN = -0.02

#: Tolerate one binary ulp at an inclusive boundary (decimal fixtures such as
#: 0.42 - 0.40 land just below 0.02), without turning ``>=`` into ``>``.
_ULP = 1e-12

_ATTEMPT = re.compile(r"attempt_(\d+)$")


@dataclass(frozen=True)
class Policy:
    text_threshold: float
    reference_threshold: float
    text_margin: float
    reference_margin: float


@dataclass(frozen=True)
class DetectorFit:
    text_threshold: float
    reference_threshold: float
    feasible: bool
    recall: float
    false_positives: int
    false_positive_rate: float
    mcc: float
    routed: int


@dataclass(frozen=True)
class SelectorFit:
    text_margin: float
    reference_margin: float
    feasible: bool
    sign_accuracy: float
    harmful: int
    mean_dino_delta: float
    exact_best_rate: float


def _number(record: Mapping, label: str, signal: str) -> float:
    try:
        value = float(record[signal])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid {signal}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label}: invalid {signal}={value}")
    return value


def _finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label}={value} is not finite")
    return value


def _attempt_number(label: str) -> int:
    match = _ATTEMPT.fullmatch(label)
    if not match:
        raise ValueError(f"invalid candidate label {label!r}")
    return int(match.group(1))


# --- pure decisions -------------------------------------------------------

def route_draft(semantic_ok: bool, text_relevance: float,
                reference_relevance: float, policy: Policy) -> bool:
    """Route the draft to repair under the guarded conjunction."""
    text = _finite(text_relevance, "text_relevance")
    reference = _finite(reference_relevance, "reference_relevance")
    if not semantic_ok:
        return True
    return (text < policy.text_threshold
            and reference < policy.reference_threshold)


def _eligible(scores: Mapping[str, Mapping], label: str,
              policy: Policy) -> bool:
    draft = scores["draft"]
    candidate = scores[label]
    text_gain = (_number(candidate, label, "text_relevance")
                 - _number(draft, "draft", "text_relevance"))
    reference_gain = (_number(candidate, label, "reference_relevance")
                      - _number(draft, "draft", "reference_relevance"))
    return (text_gain + _ULP >= policy.text_margin
            and reference_gain + _ULP >= policy.reference_margin)


def select_repair(scores: Mapping[str, Mapping], policy: Policy) -> str:
    """Best eligible attempt by (text, reference, -attempt); else the draft."""
    if "draft" not in scores:
        raise ValueError("scores are missing draft")
    _number(scores["draft"], "draft", "text_relevance")
    _number(scores["draft"], "draft", "reference_relevance")

    eligible = []
    for label, record in scores.items():
        if label == "draft":
            continue
        attempt = _attempt_number(label)
        # Validate both signals even when this attempt is ineligible, so a
        # corrupt artifact cannot silently act like a rejection.
        text = _number(record, label, "text_relevance")
        reference = _number(record, label, "reference_relevance")
        if _eligible(scores, label, policy):
            eligible.append((text, reference, -attempt, label))
    return max(eligible)[-1] if eligible else "draft"


# --- deterministic fitters ------------------------------------------------

def _thresholds(values: Sequence[float]) -> list[float]:
    """Value midpoints plus one sentinel below and above the observed range.

    Midpoints separate every distinct routing decision; the sentinels give the
    two extremes (this signal never fires / always fires).
    """
    uniq = sorted(set(float(v) for v in values))
    if not uniq:
        return [0.0]
    mids = [(a + b) / 2 for a, b in zip(uniq, uniq[1:])]
    return sorted(set([uniq[0] - 1.0, *mids, uniq[-1] + 1.0]))


def fit_detector(rows: Sequence[Mapping]) -> DetectorFit:
    """Grid-search the guarded-conjunction thresholds on ``rows``.

    Each row needs ``truth_fail``, ``semantic_ok``, ``text_relevance``,
    ``reference_relevance``. Feasible fits meet recall and false-positive-rate
    bars; the objective then maximises MCC, minimises false positives,
    maximises recall, and finally routes the smallest population.
    """
    rows = list(rows)
    texts = [_finite(r["text_relevance"], "text_relevance") for r in rows]
    refs = [_finite(r["reference_relevance"], "reference_relevance")
            for r in rows]
    truth = [bool(r["truth_fail"]) for r in rows]
    semantic = [bool(r["semantic_ok"]) for r in rows]

    best: tuple | None = None
    best_fit: DetectorFit | None = None
    for tt in _thresholds(texts):
        for rt in _thresholds(refs):
            policy = Policy(tt, rt, 0.0, 0.0)
            predicted = [route_draft(s, t, r, policy)
                         for s, t, r in zip(semantic, texts, refs)]
            tp = sum(1 for a, p in zip(truth, predicted) if a and p)
            fp = sum(1 for a, p in zip(truth, predicted) if not a and p)
            tn = sum(1 for a, p in zip(truth, predicted) if not a and not p)
            fn = sum(1 for a, p in zip(truth, predicted) if a and not p)
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
            den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = (tp * tn - fp * fn) / den if den else 0.0
            routed = sum(predicted)
            feasible = recall >= MIN_RECALL and fp_rate <= MAX_FP_RATE
            key = (not feasible, -mcc, fp, -recall, routed, tt, rt)
            if best is None or key < best:
                best = key
                best_fit = DetectorFit(
                    text_threshold=tt, reference_threshold=rt,
                    feasible=feasible, recall=recall, false_positives=fp,
                    false_positive_rate=fp_rate, mcc=mcc, routed=routed)
    assert best_fit is not None
    return best_fit


def _selector_metrics(rows: Sequence[Mapping], policy: Policy):
    pair_correct: list[bool] = []
    selected_deltas: list[float] = []
    exact_best: list[bool] = []
    for row in rows:
        scores = row["scores"]
        dino = row["dino"]
        selected = select_repair(scores, policy)
        if "draft" not in dino or selected not in dino:
            raise ValueError(f"{row.get('case_id')}: DINO missing draft/selected")
        draft_dino = _finite(dino["draft"], "dino draft")
        selected_deltas.append(_finite(dino[selected], "dino") - draft_dino)
        for label in scores:
            if label == "draft":
                continue
            if label not in dino:
                raise ValueError(f"{row.get('case_id')}: DINO missing {label}")
            predicted = _eligible(scores, label, policy)
            actual = _finite(dino[label], "dino") > draft_dino
            pair_correct.append(predicted == actual)
        available = [label for label in scores if label in dino]
        oracle = max(available, key=lambda label: _finite(dino[label], "dino"))
        exact_best.append(selected == oracle)
    sign = sum(pair_correct) / len(pair_correct) if pair_correct else 0.0
    mean_delta = (sum(selected_deltas) / len(selected_deltas)
                  if selected_deltas else 0.0)
    harmful = sum(1 for d in selected_deltas if d < HARM_MARGIN)
    exact = sum(exact_best) / len(exact_best) if exact_best else 0.0
    return sign, harmful, mean_delta, exact


def _nonneg_deltas(rows: Sequence[Mapping], signal: str) -> set[float]:
    out = {0.0}
    for row in rows:
        scores = row["scores"]
        draft = _number(scores["draft"], "draft", signal)
        for label, record in scores.items():
            if label == "draft":
                continue
            gain = _number(record, label, signal) - draft
            if gain >= 0.0:
                out.add(round(gain, 12))
    return out


def fit_selector(rows: Sequence[Mapping]) -> SelectorFit:
    """Grid-search inclusive text/reference margins on eligible ``rows``.

    Feasible fits meet the sign-accuracy and harmful bars *and* earn a strictly
    positive mean selected DINO delta -- the last condition is what stops a
    no-op margin from winning on raw accuracy alone (design 4.2). The objective
    then maximises mean delta, exact-best rate, and the two margins.
    """
    rows = list(rows)
    text_margins = sorted(_nonneg_deltas(rows, "text_relevance"))
    ref_margins = sorted(_nonneg_deltas(rows, "reference_relevance"))

    best: tuple | None = None
    best_fit: SelectorFit | None = None
    for tm in text_margins:
        for rm in ref_margins:
            policy = Policy(0.0, 0.0, tm, rm)
            sign, harmful, mean_delta, exact = _selector_metrics(rows, policy)
            feasible = (sign >= MIN_SIGN_ACCURACY and harmful <= MAX_HARMFUL
                        and mean_delta > 0.0)
            key = (not feasible, -mean_delta, -exact, -tm, -rm, tm, rm)
            if best is None or key < best:
                best = key
                best_fit = SelectorFit(
                    text_margin=tm, reference_margin=rm, feasible=feasible,
                    sign_accuracy=sign, harmful=harmful,
                    mean_dino_delta=mean_delta, exact_best_rate=exact)
    assert best_fit is not None
    return best_fit
