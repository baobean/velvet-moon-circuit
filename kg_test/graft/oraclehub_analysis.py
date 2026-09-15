"""Readout for the OracleHub experiment. Pairs arms at (concept, part, draw), aggregates to
the (concept, part) CELL (draws are correlated -> the cell is the independent unit), and reports
oraclehub-rawnn stratified by the cell's sibling NN-rank stratum (near/mid/far). The hypothesis
is supported ONLY if the win concentrates in `far` -- taxonomy helping exactly where the embedding
misses the relation -- not as a flat gain across strata."""
from __future__ import annotations
from collections import defaultdict
import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:                                    # pragma: no cover
    wilcoxon = None


def _paired_cell_deltas(rows, arm_a, arm_b, metric="dino"):
    """(concept,part) -> mean over draws of (arm_a - arm_b); also each cell's stratum."""
    bykey = defaultdict(dict)                          # (c,p,draw) -> {cond: val}
    strat = {}
    for r in rows:
        v = r.get(metric)
        if v is None or v != v:
            continue
        bykey[(r["concept"], r["part"], r["draw"])][r["condition"]] = v
        strat[(r["concept"], r["part"])] = r.get("stratum")
    cell = defaultdict(list)
    for (c, p, d), cd in bykey.items():
        if arm_a in cd and arm_b in cd:
            cell[(c, p)].append(cd[arm_a] - cd[arm_b])
    return {k: float(np.mean(v)) for k, v in cell.items() if v}, strat


def _stats(vals):
    vals = [v for v in vals]
    if not vals:
        return {"mean_delta": float("nan"), "win_rate": float("nan"), "n": 0, "wilcoxon_p": float("nan")}
    wins = sum(1 for v in vals if v > 0)
    if wilcoxon is not None and any(v != 0 for v in vals):
        try:
            p = float(wilcoxon(vals).pvalue)
        except Exception:
            p = float("nan")
    else:
        p = 1.0
    return {"mean_delta": float(np.mean(vals)), "win_rate": wins / len(vals),
            "n": len(vals), "wilcoxon_p": p}


def contrast(rows, arm_a, arm_b, metric="dino"):
    deltas, strat = _paired_cell_deltas(rows, arm_a, arm_b, metric)
    out = {"overall": _stats(list(deltas.values()))}
    for s in ("near", "mid", "far"):
        out[s] = _stats([d for k, d in deltas.items() if strat.get(k) == s])
    return out


def recovery_by_stratum(rows, metric="dino"):
    """mean metric per arm per stratum (cell-averaged)."""
    byarm = defaultdict(lambda: defaultdict(list))     # arm -> stratum -> [cell means]
    cell = defaultdict(list)                            # (c,p,cond) -> vals
    strat = {}
    for r in rows:
        v = r.get(metric)
        if v is None or v != v:
            continue
        cell[(r["concept"], r["part"], r["condition"])].append(v)
        strat[(r["concept"], r["part"])] = r.get("stratum")
    for (c, p, cond), vs in cell.items():
        byarm[cond][strat[(c, p)]].append(float(np.mean(vs)))
    return {arm: {s: float(np.mean(v)) for s, v in sd.items()} for arm, sd in byarm.items()}


def summarize(rows, metric="dino"):
    return {
        "n_rows": len(rows),
        "oraclehub_vs_rawnn": contrast(rows, "oraclehub", "rawnn", metric),
        "oraclehub_vs_random": contrast(rows, "oraclehub", "random", metric),
        "rawnn_vs_random": contrast(rows, "rawnn", "random", metric),
        "recovery_by_stratum": recovery_by_stratum(rows, metric),
    }
