"""Task 2: pure v2 policy decisions and deterministic fitters.

V2's routing family is the *guarded conjunction* (design 4.1): semantic failure
alone routes, but low reference relevance only routes when text relevance is
*also* low. That is the smallest change consistent with v1's failure, where low
reference relevance alone produced five false positives.
"""
import random

import pytest

from ragregen.hybrid_verifier_v2 import (
    Policy, route_draft, select_repair, fit_detector, fit_selector)


def _score(text, reference):
    return {"text_relevance": text, "reference_relevance": reference}


# --- pure decisions -------------------------------------------------------

def test_route_uses_guarded_conjunction_after_semantic_pass():
    p = Policy(0.50, 0.25, 0.02, 0.01)
    assert route_draft(False, 0.9, 0.9, p)
    assert route_draft(True, 0.49, 0.24, p)
    assert not route_draft(True, 0.49, 0.25, p)
    assert not route_draft(True, 0.50, 0.24, p)


def test_selector_uses_inclusive_margins_and_stable_ties():
    p = Policy(0.5, 0.25, 0.02, 0.01)
    scores = {"draft": _score(.40, .30),
              "attempt_2": _score(.42, .31),
              "attempt_1": _score(.42, .31)}
    assert select_repair(scores, p) == "attempt_1"


def test_selector_falls_back_to_draft_when_reference_margin_unmet():
    p = Policy(0.5, 0.25, 0.02, 0.02)
    scores = {"draft": _score(.40, .30),
              "attempt_1": _score(.60, .31)}  # ref gain 0.01 < 0.02
    assert select_repair(scores, p) == "draft"


def test_route_rejects_non_finite_draft_signals():
    p = Policy(0.5, 0.25, 0.02, 0.01)
    with pytest.raises(ValueError):
        route_draft(True, float("nan"), 0.2, p)


# --- deterministic fitters ------------------------------------------------

def _det_row(truth_fail, semantic_ok, text, reference):
    return {"truth_fail": truth_fail, "semantic_ok": semantic_ok,
            "text_relevance": text, "reference_relevance": reference,
            "cohort": "common" if not truth_fail else "bridge"}


def _detector_rows():
    """Reference-only routing gives a control FP; the conjunction avoids it.

    The lone PASS control has low reference relevance but *high* text relevance,
    so a reference-only rule misfires on it while the text-and-reference
    conjunction leaves it un-routed. Every FAIL is separable.
    """
    rows = []
    # 6 FAIL cases: low text AND low reference -> conjunction routes them.
    for i in range(6):
        rows.append(_det_row(True, True, 0.10 + 0.01 * i, 0.10 + 0.01 * i))
    # 5 clean PASS controls: high text, high reference.
    for i in range(5):
        rows.append(_det_row(False, True, 0.80, 0.80))
    # 1 trap PASS control: high text but low reference.
    rows.append(_det_row(False, True, 0.85, 0.05))
    return rows


def test_detector_fit_prefers_conjunction_over_reference_only():
    fit = fit_detector(_detector_rows())
    p = Policy(fit.text_threshold, fit.reference_threshold, 0.0, 0.0)
    routed = [route_draft(r["semantic_ok"], r["text_relevance"],
                          r["reference_relevance"], p)
              for r in _detector_rows()]
    # No PASS control is routed; every FAIL is.
    for r, is_routed in zip(_detector_rows(), routed):
        assert is_routed == r["truth_fail"]
    assert fit.feasible


def test_detector_fit_is_order_invariant():
    rows = _detector_rows()
    shuffled = rows[:]
    random.Random(1).shuffle(shuffled)
    a, b = fit_detector(rows), fit_detector(shuffled)
    assert (a.text_threshold, a.reference_threshold) == \
           (b.text_threshold, b.reference_threshold)


def _sel_row(cid, draft, attempts, dino):
    scores = {"draft": _score(*draft)}
    for label, ta in attempts.items():
        scores[label] = _score(*ta)
    return {"case_id": cid, "cohort": "bridge", "scores": scores, "dino": dino}


def _selector_rows():
    """A positive-gain margin must beat a no-op margin with high raw accuracy.

    Most attempts genuinely worsen DINO, so a margin high enough to select
    nothing scores well on sign accuracy but yields zero mean gain. A modest
    margin selects the one real improvement per case and earns positive gain.
    """
    rows = []
    for i in range(4):
        rows.append(_sel_row(
            f"c{i}",
            draft=(0.40, 0.30),
            attempts={"attempt_1": (0.60, 0.45),   # real improvement
                      "attempt_2": (0.41, 0.31)},  # marginal, worse DINO
            dino={"draft": 0.0, "attempt_1": 0.20, "attempt_2": -0.05}))
    return rows


def test_selector_fit_rejects_zero_gain_noop_margin():
    fit = fit_selector(_selector_rows())
    assert fit.feasible
    assert fit.mean_dino_delta > 0
    # The chosen margins actually select the improving attempt.
    p = Policy(0.5, 0.25, fit.text_margin, fit.reference_margin)
    for row in _selector_rows():
        assert select_repair(row["scores"], p) == "attempt_1"


def test_selector_fit_is_order_invariant():
    rows = _selector_rows()
    shuffled = rows[:]
    random.Random(2).shuffle(shuffled)
    a, b = fit_selector(rows), fit_selector(shuffled)
    assert (a.text_margin, a.reference_margin) == \
           (b.text_margin, b.reference_margin)
