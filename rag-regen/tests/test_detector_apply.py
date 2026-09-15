from ragregen.detector_apply import apply_frozen, detector_gate, degenerate_coverage

T = {"text": 0.5, "retrieved_reference": 0.25048828125}


def _row(cid, tf, sem_ok, text, ref, cohort):
    return {"case_id": cid, "truth_fail": tf, "semantic_ok": sem_ok,
            "text_relevance": text, "retrieved_reference_relevance": ref, "cohort": cohort}


def test_applies_frozen_rule_without_fitting():
    rows = [_row("f", True, True, 0.10, 0.10, "rare"),   # conjunction routes
            _row("p", False, True, 0.90, 0.90, "control")]  # clean control
    a = apply_frozen(rows, T)
    assert a["cases"]["f"]["routed"] is True
    assert a["cases"]["p"]["routed"] is False


def test_gate_passes_at_recall_18_and_fp_2():
    rows = [_row(f"f{i}", True, i >= 18, 0.9, 0.9, "rare") for i in range(24)]  # 18 semantic-caught
    rows += [_row(f"p{i}", False, True, 0.10 if i < 2 else 0.9, 0.10 if i < 2 else 0.9, "control")
             for i in range(24)]  # 2 controls routed -> 2 FP
    g = detector_gate(apply_frozen(rows, T), coverage_ok=True)
    assert g["recall_k"] == (18, 24) and g["fp_k"] == (2, 24)
    assert g["readiness"] == "PASS"


def test_missing_score_is_inconclusive():
    rows = [_row("f", True, True, float("nan"), 0.1, "rare")]
    assert degenerate_coverage(rows) is False
    g = detector_gate(apply_frozen([_row("f", True, True, 0.1, 0.1, "rare")], T), coverage_ok=False)
    assert g["readiness"] == "INCONCLUSIVE"
