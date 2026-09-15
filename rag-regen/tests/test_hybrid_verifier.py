import pytest

from ragregen.hybrid_verifier import (evaluate, route_draft,
                                      select_repair)


def _score(text, reference):
    return {"text_relevance": text, "reference_relevance": reference}


def test_route_is_a_union_and_threshold_equality_passes():
    assert route_draft(semantic_ok=False, reference_relevance=0.9)
    assert route_draft(semantic_ok=True, reference_relevance=0.3007)
    assert not route_draft(semantic_ok=True,
                           reference_relevance=0.30078125)


def test_selector_requires_both_improvements_and_inclusive_text_margin():
    scores = {
        "draft": _score(0.40, 0.30),
        "attempt_1": _score(0.42, 0.31),
        "attempt_2": _score(0.60, 0.29),
    }

    assert select_repair(scores) == "attempt_1"


def test_selector_falls_back_to_draft_when_no_attempt_is_eligible():
    scores = {
        "draft": _score(0.50, 0.50),
        "attempt_1": _score(0.519, 0.90),
        "attempt_2": _score(0.90, 0.50),
    }

    assert select_repair(scores) == "draft"


def test_selector_breaks_ties_by_reference_then_attempt_number():
    scores = {
        "draft": _score(0.40, 0.30),
        "attempt_2": _score(0.60, 0.50),
        "attempt_1": _score(0.60, 0.50),
        "attempt_3": _score(0.60, 0.55),
    }
    assert select_repair(scores) == "attempt_3"

    scores["attempt_3"] = _score(0.60, 0.50)
    assert select_repair(scores) == "attempt_1"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None])
def test_non_finite_or_missing_scores_fail_loudly(bad):
    scores = {"draft": _score(0.4, 0.3),
              "attempt_1": _score(bad, 0.5)}
    with pytest.raises(ValueError, match="attempt_1.*text_relevance"):
        select_repair(scores)


def test_evaluate_reports_detector_selector_and_cohort_outcomes():
    labels = {"rare_bad": "FAIL", "common_good": "PASS"}
    semantic = {
        "rare_bad": {"ok": True},
        "common_good": {"ok": True},
    }
    reranker = {"cases": {
        "rare_bad": {"scores": {
            "draft": _score(0.30, 0.20),
            "attempt_1": _score(0.50, 0.40),
        }},
        "common_good": {"scores": {
            "draft": _score(0.70, 0.60),
            "attempt_1": _score(0.60, 0.50),
        }},
    }}
    dino = {"cases": {
        "rare_bad": {"draft": 0.20, "attempt_1": 0.50},
        "common_good": {"draft": 0.70, "attempt_1": 0.60},
    }}

    result = evaluate(labels, semantic, reranker, dino,
                      {"rare_bad": "bridge", "common_good": "common"})

    assert result["detector"]["union"] == {
        "tp": 1, "fp": 0, "tn": 1, "fn": 0,
        "recall": 1.0, "specificity": 1.0,
        "balanced_accuracy": 1.0, "mcc": 1.0,
    }
    assert result["cases"]["rare_bad"]["hybrid"] == "attempt_1"
    assert result["cases"]["common_good"]["hybrid"] == "draft"
    assert result["selector"]["sign_accuracy"] == 1.0
    assert result["selector"]["harmful"] == 0
    assert result["cohorts"]["bridge"]["mean_dino_delta"] == pytest.approx(0.3)
    assert result["cohorts"]["common"]["mean_dino_delta"] == 0.0


def test_detector_is_inconclusive_without_both_label_classes():
    labels = {"only_failure": "FAIL"}
    semantic = {"only_failure": {"ok": False}}
    reranker = {"cases": {"only_failure": {
        "scores": {"draft": _score(0.2, 0.2)}}}}
    dino = {"cases": {"only_failure": {"draft": 0.1}}}

    result = evaluate(labels, semantic, reranker, dino,
                      {"only_failure": "bridge"})

    assert result["gates"]["detector"]["passed"] is False
    assert result["gates"]["detector"]["reason"] == "missing PASS stratum"
