"""Detector-only leave-one-concept-out evaluation (retrieved-reference Phase A).

No selector is evaluated here -- that path is already rejected. Only the routing
detector is fit and graded, so the study cannot be confounded by selector churn.
"""
from ragregen.detector_develop import develop


def _row(cid, truth_fail, semantic_ok, text, ref, cohort, oracle_ref=None):
    r = {"case_id": cid, "truth_fail": truth_fail, "semantic_ok": semantic_ok,
         "text_relevance": text, "reference_relevance": ref, "cohort": cohort}
    if oracle_ref is not None:
        r["oracle_reference_relevance"] = oracle_ref
    return r


def _separable_rows():
    """Retrieved reference still separates: FAILs low text+ref, PASS high both."""
    rows = []
    for i in range(6):
        rows.append(_row(f"f{i}", True, True, 0.10 + 0.01 * i,
                         0.05 + 0.01 * i, "bridge", oracle_ref=0.05))
    for i in range(6):
        rows.append(_row(f"p{i}", False, True, 0.80, 0.80, "common",
                         oracle_ref=0.80))
    return rows


def test_folds_exclude_the_heldout_case():
    result = develop(_separable_rows(), n_expected=12)
    assert len(result["folds"]) == 12
    for fold in result["folds"]:
        assert fold["case_id"] not in fold["fit_case_ids"]


def test_separable_rows_pass_all_gates():
    result = develop(_separable_rows(), n_expected=12)
    assert result["readiness"] == "PASS"
    conf = result["detector"]["retrieved_guarded"]
    assert conf["recall"] == 1.0 and conf["fp"] == 0
    assert result["gates"]  # non-empty
    assert all(f["feasible"] for f in result["folds"])


def test_baselines_reported_over_identical_rows():
    result = develop(_separable_rows(), n_expected=12)
    base = result["detector"]["baselines"]
    assert {"semantic_only", "retrieved_reference_only", "oracle_v2"} <= set(base)
    for row in base.values():
        assert row["tp"] + row["fp"] + row["tn"] + row["fn"] == 12


def test_oracle_dependence_shows_as_control_false_positives_and_fails():
    """Retrieved refs collapse: a correct control now scores low -> FP -> FAIL."""
    rows = _separable_rows()
    # Make three PASS controls look like failures on the retrieved reference only.
    for i in range(3):
        rows[6 + i]["reference_relevance"] = 0.02
        rows[6 + i]["text_relevance"] = 0.10
    result = develop(rows, n_expected=12)
    assert result["readiness"] == "FAIL"
    assert result["detector"]["retrieved_guarded"]["fp"] >= 2


def test_contamination_forces_inconclusive_not_fail():
    result = develop(_separable_rows(), n_expected=12, contaminated=["f0"])
    assert result["readiness"] == "INCONCLUSIVE"
    cov = [g for g in result["gates"] if g["name"] == "uncontaminated_coverage"][0]
    assert cov["passed"] is False


def test_missing_coverage_forces_inconclusive():
    result = develop(_separable_rows(), n_expected=13)  # one case absent
    assert result["readiness"] == "INCONCLUSIVE"
