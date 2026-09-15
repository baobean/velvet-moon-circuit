from ragregen import c1


def A(**phrases):
    return {p: {"sim": s, "kind": "subject"} for p, s in phrases.items()}


def B(ok, degenerate=False):
    return {"ok": ok, "degenerate": degenerate, "raw": "", "issues": []}


Y_TRUE = [True] * 15 + [False] * 7


def test_baselines_include_both_constants():
    base = c1.baseline_rows(Y_TRUE)
    assert set(base) == {"always-FAIL", "always-PASS"}
    assert base["always-FAIL"].recall == 1.0
    assert base["always-PASS"].recall == 0.0
    assert base["always-FAIL"].balanced_accuracy == 0.5


def test_best_tau_picks_the_maximising_value():
    sweep = c1.sweep_tau({"a": A(fox=0.10), "b": A(fox=0.90)},
                         {"a": B(ok=True), "b": B(ok=True)},
                         ["a", "b"], [True, False], grid=(0.05, 0.5))
    assert c1.best_tau(sweep, "grounded") == 0.5


def test_report_names_every_arm_and_both_baselines():
    sweep = c1.sweep_tau({"a": A(fox=0.10)}, {"a": B(ok=False)},
                         ["a"], [True], grid=(0.25,))
    md = c1.render_report(sweep=sweep, y_true=[True], column="verdict_identity",
                          excluded=0, degenerate=0, n_cases=1)
    for token in ("grounded", "semantic", "fused", "always-FAIL", "always-PASS"):
        assert token in md


def test_report_states_the_or_caveat_and_the_tau_fitting_caveat():
    """Both are the difference between a finding and a misleading number, so
    they are asserted rather than left to the writer's discretion."""
    sweep = c1.sweep_tau({"a": A(fox=0.10)}, {"a": B(ok=False)},
                         ["a"], [True], grid=(0.25,))
    md = c1.render_report(sweep=sweep, y_true=[True], column="verdict_identity",
                          excluded=0, degenerate=0, n_cases=1)
    low = md.lower()
    assert "or" in low and "recall" in low
    assert "fitted" in low or "same sample" in low


def test_report_surfaces_a_nonzero_degenerate_count():
    sweep = c1.sweep_tau({"a": A(fox=0.10)}, {"a": B(ok=False, degenerate=True)},
                         ["a"], [True], grid=(0.25,))
    md = c1.render_report(sweep=sweep, y_true=[True], column="verdict_identity",
                          excluded=0, degenerate=1, n_cases=1)
    assert "degenerate" in md.lower()
    assert "1" in md


def test_report_reports_exclusions():
    sweep = c1.sweep_tau({"a": A(fox=0.10)}, {"a": B(ok=False)},
                         ["a"], [True], grid=(0.25,))
    md = c1.render_report(sweep=sweep, y_true=[True], column="verdict",
                          excluded=3, degenerate=0, n_cases=1)
    assert "3" in md and "exclud" in md.lower()


def test_finegrained_section_names_the_verdict_and_the_missed_cases():
    stream_a = {
        "gross": {"p": {"sim": 0.10, "sim_coarse": 0.05}},
        "fine1": {"p": {"sim": 0.40, "sim_coarse": 0.38}},
        "fine2": {"p": {"sim": 0.41, "sim_coarse": 0.39}},
        "good":  {"p": {"sim": 0.80, "sim_coarse": 0.20}},
    }
    stream_b = {c: {"ok": True, "degenerate": False} for c in stream_a}
    ids = ["gross", "fine1", "fine2", "good"]
    y = [True, True, True, False]

    md, art = c1.render_finegrained(stream_a, stream_b, ids, y,
                                    column="verdict_identity")
    assert "fine1" in md and "fine2" in md
    assert "gross" in md
    assert art["baseline_caught"] == ["gross"]
    assert art["baseline_missed"] == ["fine1", "fine2"]
    assert "tau" in art and art["tau"] == c1.BASELINE_TAU


def test_finegrained_section_reports_no_band_when_the_rule_never_holds():
    stream_a = {"fine1": {"p": {"sim": 0.40, "sim_coarse": 0.38}},
                "good":  {"p": {"sim": 0.50, "sim_coarse": 0.52}}}
    stream_b = {c: {"ok": True, "degenerate": False} for c in stream_a}
    md, art = c1.render_finegrained(stream_a, stream_b, ["fine1", "good"],
                                    [True, False], column="verdict_identity")
    assert art["band"] == []
    assert "NOT MET" in md


def test_delta_table_prints_every_0_02_step_without_float_gaps():
    """Float modulo on this grid silently drops 6 of the 21 rows."""
    stream_a = {"c0": {"p": {"sim": 0.40, "sim_coarse": 0.55}}}
    stream_b = {"c0": {"ok": True, "degenerate": False}}
    md, _ = c1.render_finegrained(stream_a, stream_b, ["c0"], [True],
                                  column="verdict_identity")
    body = md.split("### Fused arm across delta")[1]
    rows = [ln for ln in body.splitlines()
            if ln.startswith("|") and "---" not in ln and "delta" not in ln]
    assert len(rows) == 21, f"expected 21 delta rows, got {len(rows)}"
    assert "| -0.2 |" in body and "| 0.2 |" in body


def test_prototype_section_is_empty_when_no_prototypes_were_scored():
    stream_a = {"c0": {"p": {"sim": 0.40, "sim_coarse": 0.55}}}
    stream_b = {"c0": {"ok": True, "degenerate": False}}
    md, art = c1.render_prototype(stream_a, stream_b, ["c0"], [True],
                                  column="verdict_identity")
    assert md == "" and art == {}


def _proto_concept(sim, proto, proto_coarse):
    """proto_margin reads candidates, not the top-level fields."""
    return {"sim": sim, "sim_coarse": 0.0,
            "candidates": [{"box": [0, 0, 1, 1], "dino_conf": 0.9, "sim": sim,
                            "sim_coarse": 0.0, "sim_proto": proto,
                            "sim_proto_coarse": proto_coarse}]}


def test_prototype_section_names_the_verdict_and_uses_the_proto_margin():
    stream_a = {
        "miss": {"p": _proto_concept(0.90, 0.10, 0.90)},
        "good": {"p": _proto_concept(0.90, 0.90, 0.10)},
    }
    stream_b = {c: {"ok": True, "degenerate": False} for c in stream_a}
    md, art = c1.render_prototype(stream_a, stream_b, ["miss", "good"],
                                  [True, False], column="verdict_identity")
    assert art["baseline_missed"] == ["miss"]
    assert art["mechanism"] == "prototype"
    assert "miss" in md
    assert art["tau"] == c1.BASELINE_TAU
