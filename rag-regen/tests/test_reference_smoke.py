import numpy as np

from ragregen import reference_smoke


def _artifact():
    # A and B pass near their own prototypes. C and D fail near the competing
    # prototype. Text embeddings are included to exercise every signal.
    return {"model": "fake", "cases": {
        "a": {"truth": "PASS", "draft_embedding": [1, 0, 0, 0],
              "reference_embeddings": [[1, 0, 0, 0]],
              "fine_text_embedding": [1, 0, 0, 0], "coarse_text_embedding": [0, 1, 0, 0]},
        "b": {"truth": "PASS", "draft_embedding": [0, 1, 0, 0],
              "reference_embeddings": [[0, 1, 0, 0]],
              "fine_text_embedding": [0, 1, 0, 0], "coarse_text_embedding": [1, 0, 0, 0]},
        "c": {"truth": "FAIL", "draft_embedding": [1, 0, 0, 0],
              "reference_embeddings": [[0, 0, 1, 0]],
              "fine_text_embedding": [0, 0, 1, 0], "coarse_text_embedding": [1, 0, 0, 0]},
        "d": {"truth": "FAIL", "draft_embedding": [0, 1, 0, 0],
              "reference_embeddings": [[0, 0, 0, 1]],
              "fine_text_embedding": [0, 0, 0, 1], "coarse_text_embedding": [0, 1, 0, 0]},
    }}


def test_case_features_use_positive_minus_nearest_negative():
    rows = {r["case_id"]: r for r in reference_smoke.case_features(_artifact())}
    assert np.isclose(rows["a"]["prototype_margin"], 1.0)
    assert np.isclose(rows["c"]["prototype_margin"], -1.0)


def test_threshold_direction_is_low_score_means_failure():
    rows = reference_smoke.case_features(_artifact())
    result = reference_smoke.evaluate_feature(rows, "prototype_margin")
    assert result["in_sample"]["overall"]["tp"] == 2
    assert result["in_sample"]["overall"]["fp"] == 0


def test_evaluation_marks_primary_gate_without_hiding_raw_rows():
    result = reference_smoke.evaluate(_artifact())
    assert result["model"] == "fake"
    assert len(result["rows"]) == 4
    assert result["primary_gate"]["passed"] is False
    assert "Primary promotion gate" in reference_smoke.render(result)


def test_zero_embedding_is_rejected():
    bad = _artifact()
    bad["cases"]["a"]["draft_embedding"] = [0, 0, 0, 0]
    try:
        reference_smoke.case_features(bad)
    except ValueError as exc:
        assert "non-zero" in str(exc)
    else:
        raise AssertionError("zero embedding was accepted")


def test_pairwise_selector_uses_gain_and_counts_harm():
    artifact = _artifact()
    artifact["cases"]["a"]["candidate_embeddings"] = {
        "attempt_1": [0.8, 0.2, 0, 0]}
    artifact["cases"]["c"]["candidate_embeddings"] = {
        "attempt_1": [0, 0, 1, 0]}
    dino = {"cases": {
        "a": {"draft": 0.8, "attempt_1": 0.7},
        "c": {"draft": 0.2, "attempt_1": 0.8},
    }}
    result = reference_smoke.evaluate_pairwise(artifact, dino)
    primary = result["arms"]["prototype_margin"]
    assert primary["pairwise_sign_accuracy"] == 1.0
    assert primary["harmful_selections"] == 0
    assert result["primary_gate"]["passed"] is True
    assert "Pairwise repair selector" in reference_smoke.render_pairwise(result)
