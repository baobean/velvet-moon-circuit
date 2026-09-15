from ragregen import reranker_smoke


def _artifact():
    cases = {}
    for index, (truth, draft, repair) in enumerate([
        ("FAIL", 0.1, 0.8), ("FAIL", 0.2, 0.7),
        ("PASS", 0.8, 0.4), ("PASS", 0.9, 0.85),
    ]):
        cases[f"c{index}"] = {
            "truth": truth,
            "scores": {
                "draft": {"text_relevance": draft,
                          "reference_relevance": draft},
                "attempt_1": {"text_relevance": repair,
                              "reference_relevance": repair},
            },
        }
    return {"model": "fixture", "cases": cases}


def test_absolute_evaluation_is_leave_one_out_and_auditable():
    result = reranker_smoke.evaluate_absolute(_artifact())
    metric = result["arms"]["reference_relevance"]["leave_one_out"]["overall"]
    assert metric["tp"] == 2
    assert metric["fp"] == 0
    assert len(result["arms"]["reference_relevance"]["leave_one_out_thresholds"]) == 4


def test_pairwise_evaluation_uses_heldout_scores():
    dino = {"cases": {
        "c0": {"draft": 0.1, "attempt_1": 0.7},
        "c1": {"draft": 0.2, "attempt_1": 0.6},
        "c2": {"draft": 0.8, "attempt_1": 0.4},
        "c3": {"draft": 0.9, "attempt_1": 0.7},
    }}
    result = reranker_smoke.evaluate_pairwise(_artifact(), dino)
    primary = result["arms"]["reference_relevance"]
    assert primary["sign_accuracy"] == 1.0
    assert primary["harmful"] == 0
    assert primary["mean_dino_delta"] > 0


def test_renderers_include_gate():
    absolute = reranker_smoke.render_absolute(
        reranker_smoke.evaluate_absolute(_artifact()))
    dino = {"cases": {cid: {"draft": 0.2, "attempt_1": 0.8}
                      for cid in _artifact()["cases"]}}
    pairwise = reranker_smoke.render_pairwise(
        reranker_smoke.evaluate_pairwise(_artifact(), dino))
    assert "promotion gate" in absolute
    assert "promotion gate" in pairwise
