"""v3 selector: semantic gate replaces the reference margin."""
from ragregen.hybrid_verifier_v3 import select_repair, fit_selector, develop


def _score(text, reference):
    return {"text_relevance": text, "reference_relevance": reference}


def test_semantic_veto_overrides_a_high_text_attempt():
    scores = {"draft": _score(0.40, 0.30),
              "attempt_1": _score(0.90, 0.80),   # great numbers...
              "attempt_2": _score(0.60, 0.50)}
    sem = {"attempt_1": False, "attempt_2": True}  # ...but VLM rejects attempt_1
    assert select_repair(scores, sem, text_margin=0.0) == "attempt_2"


def test_no_semantic_confirmation_keeps_draft():
    scores = {"draft": _score(0.40, 0.30), "attempt_1": _score(0.90, 0.80)}
    assert select_repair(scores, {"attempt_1": False}, 0.0) == "draft"


def test_text_floor_still_applies_under_semantic_ok():
    scores = {"draft": _score(0.40, 0.30), "attempt_1": _score(0.41, 0.90)}
    # semantic ok, but text gain 0.01 < 0.05 floor -> ineligible
    assert select_repair(scores, {"attempt_1": True}, text_margin=0.05) == "draft"


def _dev_fixture():
    manifest = {"protocol_status": "reconstructed_after_score",
                "dataset": "fix", "cases": {}}
    semantic, reranker, dino, cand = {}, {"cases": {}}, {"cases": {}}, {}
    for i in range(4):  # rare FAILs: attempt_1 truly improves, VLM confirms it
        cid = f"rare_{i}"
        manifest["cases"][cid] = {"identity_truth": "FAIL",
                                  "selector_eligible": True,
                                  "eligibility_reason": "eligible",
                                  "cohort": "bridge"}
        semantic[cid] = {"ok": False}
        reranker["cases"][cid] = {"truth": "FAIL", "scores": {
            "draft": _score(0.10, 0.10), "attempt_1": _score(0.60, 0.50),
            "attempt_2": _score(0.30, 0.20)}}
        dino["cases"][cid] = {"draft": 0.0, "attempt_1": 0.20, "attempt_2": -0.05}
        cand[cid] = {"attempt_1": {"ok": True}, "attempt_2": {"ok": False}}
    for i in range(4):  # controls: VLM says the untouched-looking repair is fine
        cid = f"ctrl_{i}"
        manifest["cases"][cid] = {"identity_truth": "PASS",
                                  "selector_eligible": True,
                                  "eligibility_reason": "eligible",
                                  "cohort": "common"}
        semantic[cid] = {"ok": True}
        reranker["cases"][cid] = {"truth": "PASS", "scores": {
            "draft": _score(0.80, 0.80), "attempt_1": _score(0.82, 0.81)}}
        dino["cases"][cid] = {"draft": 0.0, "attempt_1": 0.03}
        cand[cid] = {"attempt_1": {"ok": True}}
    return manifest, semantic, reranker, dino, cand


def test_develop_folds_agree_with_cases_when_v3_diverges_from_v2():
    """The per-fold table must report v3 selections, not stale v2 ones.

    ``diverge`` is a case where v2 would pick the high-margin attempt_1 but the
    VLM vetoes it, so v3 must pick attempt_2 instead. The folds table has to move
    with the cases table.
    """
    manifest, semantic, reranker, dino, cand = _dev_fixture()
    cid = "rare_0"
    reranker["cases"][cid]["scores"]["attempt_1"] = _score(0.90, 0.80)
    reranker["cases"][cid]["scores"]["attempt_2"] = _score(0.60, 0.50)
    dino["cases"][cid] = {"draft": 0.0, "attempt_1": 0.05, "attempt_2": 0.20}
    cand[cid] = {"attempt_1": {"ok": False}, "attempt_2": {"ok": True}}

    result = develop(manifest, semantic, reranker, dino, cand)
    folds = {f["case_id"]: f.get("selected") for f in result["folds"]}
    assert folds[cid] == result["cases"][cid]["selected"] == "attempt_2"
    for other in result["folds"]:
        assert other["selected"] == result["cases"][other["case_id"]].get("selected")


def test_develop_runs_and_reports_v3_selector():
    result = develop(*_dev_fixture())
    assert result["policy_version"] == "v3"
    assert result["selector"]["signal"].startswith("candidate_semantic")
    assert result["readiness"] in ("PASS", "FAIL", "INCONCLUSIVE")
    # A missing verdict for an eligible attempt must be caught, not silently ok.
    m, s, r, d, c = _dev_fixture()
    del c["rare_0"]["attempt_2"]
    import pytest
    with pytest.raises(ValueError):
        develop(m, s, r, d, c)
