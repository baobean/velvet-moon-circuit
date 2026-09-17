from ragregen.partgraph.decide import gain_verdict, PHASE0_GAIN_MARGIN


def test_margin_frozen():
    assert PHASE0_GAIN_MARGIN == 0.02


def test_clear_gain():
    r = gain_verdict([0.10, 0.12, 0.09, 0.11, 0.13, 0.10])
    assert r["verdict"] == "GAIN" and r["ci"][0] > 0


def test_null_when_ci_crosses_zero():
    r = gain_verdict([0.10, -0.08, 0.12, -0.09, 0.11, -0.10])
    assert r["verdict"] == "NO_LARGE_EFFECT"


def test_underpowered_below_min_ci_n():
    r = gain_verdict([0.10, 0.12])          # n < MIN_CI_N (5)
    assert r["verdict"] == "UNDERPOWERED" and r["ci"] is None
