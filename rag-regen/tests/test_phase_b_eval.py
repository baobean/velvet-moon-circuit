from ragregen.phase_b_eval import end_to_end_gates, readiness


def _agg(rare_mean, harmful_48, ctrl_ci_lo, preservation_min):
    return {"rare": {"arm3": {"mean_cropped_dino_delta": rare_mean}},
            "harmful_count_48": harmful_48,
            "control": {"arm3_minus_draft_ci": [ctrl_ci_lo, 0.05]},
            "preservation_min": preservation_min}


def test_gates_pass_on_clean_agg():
    g = end_to_end_gates(_agg(0.15, 1, -0.01, 1.0), visual_review={"pass": True})
    assert all(x["passed"] for x in g["gates"])


def test_preservation_below_one_fails():
    g = end_to_end_gates(_agg(0.15, 0, -0.0, 0.999), visual_review={"pass": True})
    assert any(x["name"] == "preservation_exact_1" and x["passed"] is False for x in g["gates"])


def test_control_ci_lower_below_margin_fails():
    g = end_to_end_gates(_agg(0.15, 0, -0.05, 1.0), visual_review={"pass": True})
    assert any(x["name"] == "control_non_inferiority" and x["passed"] is False for x in g["gates"])


def test_thin_flag_on_discrete_boundary():
    r = readiness({"readiness": "PASS", "recall_k": (18, 24), "fp_k": (1, 24)},
                  {"readiness": "PASS", "gates": []})
    assert r["readiness"] == "PASS" and r["thin"] is True
