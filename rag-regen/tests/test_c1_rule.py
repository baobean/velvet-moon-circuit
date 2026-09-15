# tests/test_c1_rule.py
from ragregen import c1


def _streams(spec):
    """spec: {case_id: (sim, sim_coarse, semantic_ok)}"""
    a = {cid: {"p": {"sim": s, "sim_coarse": sc}} for cid, (s, sc, _) in spec.items()}
    b = {cid: {"ok": ok, "degenerate": False} for cid, (_, _, ok) in spec.items()}
    return a, b


def test_baseline_split_separates_caught_from_missed_with_delta_disabled():
    spec = {
        "gross": (0.10, 0.05, True),   # below tau -> caught by MISSING
        "fine":  (0.40, 0.38, True),   # above tau, margin +0.02 -> missed at delta=0
        "good":  (0.80, 0.20, True),   # a true pass
    }
    a, b = _streams(spec)
    caught, missed = c1.baseline_split(a, b, ["gross", "fine", "good"],
                                       [True, True, False])
    assert caught == ["gross"]
    assert missed == ["fine"]


def test_rule_passes_when_enough_misses_are_caught_without_false_positives():
    spec = {
        "gross": (0.10, 0.05, True),
        "fine1": (0.40, 0.38, True),   # margin +0.02: missed at 0, caught at 0.05
        "fine2": (0.41, 0.39, True),   # margin +0.02: missed at 0, caught at 0.05
        "good":  (0.80, 0.20, True),
    }
    ids = ["gross", "fine1", "fine2", "good"]
    y = [True, True, True, False]
    a, b = _streams(spec)
    caught, missed = c1.baseline_split(a, b, ids, y)
    r = c1.evaluate_rule(a, b, ids, y, caught, missed, delta=0.05, min_catch=2)
    assert r.caught == ["fine1", "fine2"]
    assert r.false_positives == []
    assert r.gross_retained is True
    assert r.passes is True


def test_rule_fails_when_a_true_pass_is_flagged():
    spec = {
        "fine1": (0.40, 0.38, True),   # margin +0.02: missed at 0, caught at 0.05
        "good":  (0.50, 0.52, True),   # margin -0.02, caught at delta=0.05
    }
    ids = ["fine1", "good"]
    y = [True, False]
    a, b = _streams(spec)
    caught, missed = c1.baseline_split(a, b, ids, y)
    r = c1.evaluate_rule(a, b, ids, y, caught, missed, delta=0.05, min_catch=1)
    assert r.false_positives == ["good"]
    assert r.passes is False, "precision must be 1.000"


def test_robust_band_requires_consecutive_passes():
    mk = lambda d, p: c1.RuleResult(delta=d, caught=[], false_positives=[],
                                    gross_retained=True, passes=p)
    results = [mk(0.00, True), mk(0.01, False), mk(0.02, True),
               mk(0.03, True), mk(0.04, True), mk(0.05, False)]
    band = c1.robust_band(results, min_consecutive=3)
    assert [r.delta for r in band] == [0.02, 0.03, 0.04]


def test_robust_band_is_empty_when_no_run_is_long_enough():
    mk = lambda d, p: c1.RuleResult(delta=d, caught=[], false_positives=[],
                                    gross_retained=True, passes=p)
    results = [mk(0.00, True), mk(0.01, True), mk(0.02, False)]
    assert c1.robust_band(results, min_consecutive=3) == []


def test_baseline_false_positives_names_the_threshold_driven_ones():
    stream_a = {"good": {"p": {"sim": 0.05, "sim_coarse": 0.0}},
                "fine": {"p": {"sim": 0.90, "sim_coarse": 0.0}}}
    stream_b = {c: {"ok": True, "degenerate": False} for c in stream_a}
    got = c1.baseline_false_positives(stream_a, stream_b, ["good", "fine"],
                                      [False, False], tau=0.25)
    assert got == ["good"]


def _concept(sim, proto, proto_coarse, sim_coarse=0.0):
    """A persisted concept carrying one candidate box.

    proto_margin reads the highest-confidence CANDIDATE, never the top-level
    fields, so fixtures must supply candidates or the margin is None.
    """
    return {"sim": sim, "sim_coarse": sim_coarse,
            "candidates": [{"box": [0, 0, 1, 1], "dino_conf": 0.9, "sim": sim,
                            "sim_coarse": sim_coarse, "sim_proto": proto,
                            "sim_proto_coarse": proto_coarse}]}


def test_a_false_positive_inherited_from_the_baseline_does_not_fail_the_rule():
    """tau = 0.25 imports FPs before delta is swept. Grading against *new*
    FPs measures the mechanism instead of the threshold it inherited."""
    stream_a = {
        "miss": {"p": _concept(0.90, 0.1, 0.9)},   # PRESENT, proto margin -0.8
        "fp":   {"p": _concept(0.05, 0.9, 0.1)},   # below tau -> MISSING
    }
    stream_b = {c: {"ok": True, "degenerate": False} for c in stream_a}
    ids, y = ["miss", "fp"], [True, False]
    caught0, missed0 = c1.baseline_split(stream_a, stream_b, ids, y, tau=0.25)
    assert missed0 == ["miss"] and caught0 == []

    r = c1.evaluate_rule(stream_a, stream_b, ids, y, caught0, missed0,
                         delta=0.0, tau=0.25, min_catch=1,
                         baseline_false_positives=["fp"],
                         margin_of=c1.proto_margin)
    assert r.caught == ["miss"]
    assert r.false_positives == []          # 'fp' is inherited, not new
    assert r.passes


def test_a_new_false_positive_still_fails_the_rule():
    stream_a = {
        "miss": {"p": _concept(0.90, 0.1, 0.9)},
        "ok":   {"p": _concept(0.90, 0.1, 0.9)},   # correct image, margin fires
    }
    stream_b = {c: {"ok": True, "degenerate": False} for c in stream_a}
    ids, y = ["miss", "ok"], [True, False]
    caught0, missed0 = c1.baseline_split(stream_a, stream_b, ids, y, tau=0.25)
    r = c1.evaluate_rule(stream_a, stream_b, ids, y, caught0, missed0,
                         delta=0.0, tau=0.25, min_catch=1,
                         baseline_false_positives=[],
                         margin_of=c1.proto_margin)
    assert r.false_positives == ["ok"]
    assert not r.passes
