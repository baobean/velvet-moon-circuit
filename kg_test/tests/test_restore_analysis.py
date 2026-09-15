from graft import restore_analysis as A


def _rows():
    r = []
    for cp, base in [(("C1", "leaf"), 0.30), (("C2", "leaf"), 0.40)]:
        c, p = cp
        for d in range(2):
            r += [
                {"concept": c, "part": p, "level": 0, "draw": d, "condition": "isolated", "k_eff": 2, "dino": base - 0.05, "siglip2": 0, "clip_i": 0},
                {"concept": c, "part": p, "level": 0, "draw": d, "condition": "rawnn", "k_eff": 2, "dino": base, "siglip2": 0, "clip_i": 0},
                {"concept": c, "part": p, "level": 0, "draw": d, "condition": "hub", "k_eff": 2, "dino": base + 0.04, "siglip2": 0, "clip_i": 0},
            ]
    return r


def test_make_or_break_positive_and_paired():
    mob = A.make_or_break(_rows(), metric="dino", levels=(0,))
    assert mob[0]["n"] == 2
    assert abs(mob[0]["mean_delta"] - 0.04) < 1e-9
    assert mob[0]["win_rate"] == 1.0


def test_borrowing_helps_hub_beats_isolated():
    bh = A.borrowing_helps(_rows(), level=0)
    assert bh["hub_minus_isolated"]["mean_delta"] > 0


def test_recovery_curve_has_all_conditions():
    rc = A.recovery_curve(_rows(), "dino")
    assert set(rc.keys()) == {"isolated", "rawnn", "hub"}
    assert 0 in rc["hub"]


def test_stratified_buckets_by_heldout_count():
    rows = _rows()
    held = ([{"concept": "C1", "part": "leaf"}] * 2) + ([{"concept": "C2", "part": "leaf"}] * 5)
    strat = A.stratified_delta(rows, held, "hub", "rawnn", level=0)
    assert sum(v["n"] for v in strat.values()) == 2
