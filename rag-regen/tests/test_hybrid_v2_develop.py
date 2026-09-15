"""Task 3: leave-one-concept-out development evaluator.

The two invariants under test are the ones the v1 study violated: every
held-out prediction must come from a fit that never saw the held-out case, and a
mask-failed case must stay in the detector population while being absent from
every selector comparison.
"""
from ragregen.hybrid_v2_develop import develop


def _score(text, reference):
    return {"text_relevance": text, "reference_relevance": reference}


def _fixture():
    """Two FAIL (one mask-failed), two PASS controls.

    ``bad_mask`` is a genuine identity FAIL whose mask never grounded: it belongs
    in the detector population but has no repair attempts and no DINO row.
    """
    manifest = {
        "protocol_status": "reconstructed_after_score",
        "counts": {"detector_rows": 4, "selector_eligible": 3, "excluded": 1},
        "cases": {
            "fail_ok": {"identity_truth": "FAIL", "selector_eligible": True,
                        "eligibility_reason": "eligible", "cohort": "bridge"},
            "bad_mask": {"identity_truth": "FAIL", "selector_eligible": False,
                         "eligibility_reason": "mask_not_grounded",
                         "cohort": "bridge"},
            "pass_a": {"identity_truth": "PASS", "selector_eligible": True,
                       "eligibility_reason": "eligible", "cohort": "common"},
            "pass_b": {"identity_truth": "PASS", "selector_eligible": True,
                       "eligibility_reason": "eligible", "cohort": "common"},
        },
    }
    semantic = {
        "fail_ok": {"ok": False},
        "bad_mask": {"ok": False},
        "pass_a": {"ok": True},
        "pass_b": {"ok": True},
    }
    reranker = {"cases": {
        "fail_ok": {"truth": "FAIL", "scores": {
            "draft": _score(0.10, 0.10),
            "attempt_1": _score(0.60, 0.45),
            "attempt_2": _score(0.11, 0.09)}},
        "bad_mask": {"truth": "FAIL", "scores": {"draft": _score(0.12, 0.11)}},
        "pass_a": {"truth": "PASS", "scores": {
            "draft": _score(0.80, 0.80),
            "attempt_1": _score(0.82, 0.79)}},
        "pass_b": {"truth": "PASS", "scores": {
            "draft": _score(0.85, 0.82),
            "attempt_1": _score(0.86, 0.83)}},
    }}
    dino = {"cases": {
        "fail_ok": {"draft": 0.0, "attempt_1": 0.20, "attempt_2": -0.05},
        "pass_a": {"draft": 0.0, "attempt_1": 0.03},
        "pass_b": {"draft": 0.0, "attempt_1": 0.02},
    }}
    return manifest, semantic, reranker, dino


def test_each_prediction_uses_fit_without_heldout_case():
    result = develop(*_fixture())
    assert len(result["folds"]) == len(_fixture()[0]["cases"])
    for fold in result["folds"]:
        assert fold["case_id"] not in fold["fit_case_ids"]


def test_mask_failed_case_stays_in_detector_not_selector():
    result = develop(*_fixture())

    detector_ids = {c for c, row in result["cases"].items()
                    if row.get("detector_prediction") is not None}
    assert "bad_mask" in detector_ids

    selector_ids = {c for c, row in result["cases"].items()
                    if row.get("selected") is not None}
    assert "bad_mask" not in selector_ids


def test_report_records_both_denominators_and_reasons():
    result = develop(*_fixture())
    assert result["detector"]["denominator"]["n"] == 4
    assert result["selector"]["denominator"]["n_eligible"] == 3
    excluded = result["selector"]["denominator"]["excluded"]
    assert excluded["bad_mask"] == "mask_not_grounded"


def test_readiness_and_gates_are_reported():
    result = develop(*_fixture())
    assert result["readiness"] in ("PASS", "FAIL", "INCONCLUSIVE")
    names = {g["name"] for g in result["gates"]}
    assert {"detector_recall", "detector_false_positives",
            "selector_sign_accuracy", "selector_harmful",
            "rare_mean_dino_delta"} <= names


def test_baselines_reported_over_identical_detector_rows():
    result = develop(*_fixture())
    baselines = result["detector"]["baselines"]
    assert {"semantic_only", "reference_only", "v1_union"} <= set(baselines)
    for row in baselines.values():
        assert row["tp"] + row["fp"] + row["tn"] + row["fn"] == 4
