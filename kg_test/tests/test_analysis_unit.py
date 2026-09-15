from graft.analysis import build_analysis, paired_analysis


def test_paired_analysis_delta_and_winrate():
    rows = [
        {"method": "ours", "species": "A", "dino": 0.30}, {"method": "b1", "species": "A", "dino": 0.20},
        {"method": "ours", "species": "B", "dino": 0.10}, {"method": "b1", "species": "B", "dino": 0.25},
    ]
    r = paired_analysis(rows, "dino")
    assert r["n"] == 2
    assert r["win_rate"] == 0.5                       # A: ours wins; B: ours loses
    assert abs(r["mean_delta"] - (-0.025)) < 1e-9     # ((0.30-0.20)+(0.10-0.25))/2


def _row(method, species, name_mode, ip, **metrics):
    return {"method": method, "species": species, "name_mode": name_mode, "ip_scale": ip, **metrics}


def test_build_analysis_picks_best_ip_by_neutral_ours_dino():
    rows = [
        _row("ours", "A", "neutral", 0.4, dino=0.10),
        _row("ours", "B", "neutral", 0.4, dino=0.20),   # mean 0.15
        _row("ours", "A", "neutral", 0.8, dino=0.50),
        _row("ours", "B", "neutral", 0.8, dino=0.60),   # mean 0.55 -> best
        # named rows and other methods must not steer the choice
        _row("ours", "A", "named", 0.4, dino=0.99),
        _row("b1", "A", "neutral", 0.4, dino=0.99),
    ]
    assert build_analysis(rows)["best_ip"] == 0.8


def test_build_analysis_pairs_ours_against_b1_and_notree_at_best_ip():
    rows = [
        _row("ours", "A", "neutral", 0.4, dino=0.10, siglip2=0.10, clip_i=0.10),
        _row("ours", "A", "neutral", 0.8, dino=0.50, siglip2=0.40, clip_i=0.30),
        _row("b1", "A", "neutral", 0.4, dino=0.90, siglip2=0.90, clip_i=0.90),
        _row("b1", "A", "neutral", 0.8, dino=0.20, siglip2=0.10, clip_i=0.10),
        _row("ours_notree", "A", "neutral", 0.8, dino=0.60, siglip2=0.40, clip_i=0.20),
    ]
    out = build_analysis(rows)
    assert out["best_ip"] == 0.8
    # ours-vs-b1 at ip 0.8 only: 0.50 - 0.20
    assert out["graft_vs_b1"]["dino"]["n"] == 1
    assert abs(out["graft_vs_b1"]["dino"]["mean_delta"] - 0.30) < 1e-9
    assert out["graft_vs_b1"]["dino"]["win_rate"] == 1.0
    # ours-vs-ours_notree at ip 0.8: 0.50 - 0.60 -> loss
    assert abs(out["graft_vs_notree"]["dino"]["mean_delta"] - (-0.10)) < 1e-9
    assert out["graft_vs_notree"]["dino"]["win_rate"] == 0.0
    assert set(out["graft_vs_b1"]) == {"dino", "siglip2", "clip_i"}


def test_build_analysis_named_minus_neutral_sign_and_missing_sides():
    rows = [
        _row("ours", "A", "neutral", 0.4, dino=0.20),
        _row("ours", "A", "named", 0.4, dino=0.50),     # +0.30
        _row("b0", "A", "neutral", 0.4, dino=0.10),     # no named side -> None
        _row("b2", "A", "named", 0.4),                  # metric key absent -> None
    ]
    out = build_analysis(rows)
    assert abs(out["named_minus_neutral"]["ours"]["dino"] - 0.30) < 1e-9
    assert out["named_minus_neutral"]["b0"]["dino"] is None
    assert out["named_minus_neutral"]["b2"]["dino"] is None


def test_build_analysis_tolerates_empty_and_untagged_rows():
    assert build_analysis([])["best_ip"] is None
    # rows with no ip_scale/name_mode tags at all must not crash
    out = build_analysis([{"method": "ours", "species": "A", "dino": 0.5}])
    assert out["best_ip"] is None
    assert out["graft_vs_b1"]["dino"]["n"] == 0
    # n == 0 means "nothing was compared", NOT "GRAFT tied B1 and never won".
    assert out["graft_vs_b1"]["dino"]["mean_delta"] is None
    assert out["graft_vs_b1"]["dino"]["win_rate"] is None


def test_paired_analysis_reports_none_when_nothing_pairs():
    r = paired_analysis([{"method": "ours", "species": "A", "dino": 0.5}], "dino")
    assert r["n"] == 0
    assert r["mean_delta"] is None and r["win_rate"] is None
