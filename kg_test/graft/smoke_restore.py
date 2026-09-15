"""Pure pass/fail checks for the Stage-A' smoke gate (spec §6). INFRA-ONLY: a failure
means fix the GPU path, never interpret it as evidence about the hypothesis. The GPU
smoke run assembles a `report` dict from a 1-concept pass and feeds it to run_gate."""
from __future__ import annotations
import numpy as np
from graft.parts import mask_area_fraction


def check_mask_valid(sam_mask, box, min_frac, max_frac):
    m = np.asarray(sam_mask, dtype=bool)
    if m.sum() == 0:
        return False, "empty SAM mask"
    frac = mask_area_fraction(m, box)
    ok = min_frac <= frac <= max_frac
    return ok, f"mask area frac {frac:.3f} in [{min_frac},{max_frac}]={ok}"


def check_k_eff_equal(rows_for_cell):
    borrow = [r for r in rows_for_cell if r["condition"] in ("random", "rawnn", "hub")]
    ks = {r["k_eff"] for r in borrow}
    return (len(ks) <= 1), f"borrowing-arm k_eff set={sorted(ks)} (want size<=1)"


def check_no_contamination(borrowed_ids, target_concept, target_ref):
    bad = [i for i in borrowed_ids if i.split("::")[0] == target_concept]
    return (not bad), f"contaminating borrowed ids={bad}"


def check_actual_inpaint(base_arr, out_arr, box, outside_tol, inside_min):
    a, o = np.asarray(base_arr, int), np.asarray(out_arr, int)
    x0, y0, x1, y1 = box
    inside = np.abs(a[y0:y1, x0:x1] - o[y0:y1, x0:x1]).mean() if (x1 > x0 and y1 > y0) else 0
    outmask = np.ones(a.shape[:2], bool)
    outmask[y0:y1, x0:x1] = False
    outside = np.abs(a[outmask] - o[outmask]).mean() if outmask.any() else 0
    ok = bool(inside >= inside_min and outside <= outside_tol)
    return ok, f"inside d={inside:.2f}(>= {inside_min}), outside d={outside:.2f}(<= {outside_tol})"


def check_arms_comparable(rows_for_cell, conditions=("isolated", "random", "rawnn", "hub")):
    keys = {(r["level"], r["draw"]) for r in rows_for_cell}
    for (lvl, d) in keys:
        cell = [r for r in rows_for_cell if r["level"] == lvl and r["draw"] == d]
        present = {r["condition"] for r in cell}
        finite = all(r["dino"] == r["dino"] for r in cell if "dino" in r)
        if not set(conditions).issubset(present) or not finite:
            return False, f"cell (L{lvl},d{d}) missing arms or non-finite: {sorted(present)}"
    return True, "all arms present + finite for every (level,draw)"


def check_no_oom(peak_gb, budget_gb, per_gen_growth_gb, leak_tol):
    ok = peak_gb <= budget_gb and per_gen_growth_gb <= leak_tol
    return ok, f"peak {peak_gb:.1f}GB<= {budget_gb}, per-gen growth {per_gen_growth_gb:.3f}<= {leak_tol}"


def run_gate(report):
    msgs, results = [], []
    for m, box in report.get("masks", []):
        ok, msg = check_mask_valid(m, box, report.get("min_frac", 0.01), report.get("max_frac", 0.9))
        results.append(ok); msgs.append("mask_valid: " + msg)
    for cell in report.get("cells", []):
        for chk in (check_k_eff_equal(cell["rows"]), check_arms_comparable(cell["rows"])):
            results.append(chk[0]); msgs.append("cell: " + chk[1])
        for bid in cell.get("borrowed", []):
            ok, msg = check_no_contamination(bid["ids"], bid["concept"], bid["ref"])
            results.append(ok); msgs.append("contam: " + msg)
    for inp in report.get("inpaints", []):
        ok, msg = check_actual_inpaint(inp["base"], inp["out"], inp["box"],
                                       report.get("outside_tol", 3), report.get("inside_min", 4))
        results.append(ok); msgs.append("inpaint: " + msg)
    if "peak_gb" in report:
        ok, msg = check_no_oom(report["peak_gb"], report.get("budget_gb", 23.0),
                               report.get("per_gen_growth_gb", 0.0), report.get("leak_tol", 0.1))
        results.append(ok); msgs.append("oom: " + msg)
    return (all(results) if results else False), msgs
