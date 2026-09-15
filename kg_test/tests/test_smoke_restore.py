import numpy as np
from graft import smoke_restore as S


def test_mask_valid_bounds():
    m = np.zeros((20, 20), bool); m[0:10, 0:10] = True     # frac 1.0 vs box 100
    assert S.check_mask_valid(m, (0, 0, 10, 10), 0.01, 0.9)[0] is False
    m2 = np.zeros((20, 20), bool); m2[0:5, 0:10] = True     # 50/100 = 0.5
    assert S.check_mask_valid(m2, (0, 0, 10, 10), 0.01, 0.9)[0] is True
    assert S.check_mask_valid(np.zeros((5, 5), bool), (0, 0, 5, 5), 0.01, 0.9)[0] is False


def test_k_eff_equal():
    rows = [{"condition": c, "k_eff": 2, "level": 0, "draw": 0} for c in ("random", "rawnn", "hub")]
    assert S.check_k_eff_equal(rows)[0] is True
    rows[0]["k_eff"] = 1
    assert S.check_k_eff_equal(rows)[0] is False


def test_no_contamination():
    assert S.check_no_contamination(["C2::leaf::0", "C3::leaf::1"], "C1", "C1/r0")[0] is True
    assert S.check_no_contamination(["C1::leaf::0"], "C1", "C1/r0")[0] is False


def test_actual_inpaint():
    base = np.full((40, 40, 3), 100, np.uint8); out = base.copy()
    out[10:30, 10:30] = 200
    assert S.check_actual_inpaint(base, out, (10, 10, 30, 30), outside_tol=2, inside_min=5)[0] is True
    assert S.check_actual_inpaint(base, base, (10, 10, 30, 30), outside_tol=2, inside_min=5)[0] is False


def test_no_oom():
    assert S.check_no_oom(peak_gb=12.0, budget_gb=23.0, per_gen_growth_gb=0.01, leak_tol=0.1)[0] is True
    assert S.check_no_oom(peak_gb=24.5, budget_gb=23.0, per_gen_growth_gb=0.01, leak_tol=0.1)[0] is False
    assert S.check_no_oom(peak_gb=12.0, budget_gb=23.0, per_gen_growth_gb=0.5, leak_tol=0.1)[0] is False


def test_run_gate_smoke():
    report = {"cells": [{"rows": [{"condition": c, "k_eff": 2, "level": 0, "draw": 0, "dino": 0.3}
                                  for c in ("random", "rawnn", "hub", "isolated")]}],
              "peak_gb": 12.0, "per_gen_growth_gb": 0.01}
    ok, msgs = S.run_gate(report)
    assert ok is True and msgs
