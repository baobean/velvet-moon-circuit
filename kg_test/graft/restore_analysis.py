"""Analysis for the part-level restore matrix: recovery curve, the +Hub - +RawNN
make-or-break (paired per concept-part, with Wilcoxon), borrowing-helps vs isolated,
and reliability stratification by held-out P-crop count (sprint §0 lesson)."""
from __future__ import annotations
import statistics as st
from collections import defaultdict


def _by(rows, **f):
    return [r for r in rows if all(r.get(k) == v for k, v in f.items())]


def recovery_curve(rows, metric="dino"):
    out = defaultdict(dict)
    conds = {r["condition"] for r in rows}
    levels = sorted({r["level"] for r in rows})
    for c in conds:
        for lvl in levels:
            vals = [r[metric] for r in _by(rows, condition=c, level=lvl) if r[metric] == r[metric]]
            if vals:
                out[c][lvl] = st.mean(vals)
    return {k: dict(v) for k, v in out.items()}


def _pairs(rows, cond_a, cond_b, level, metric):
    keys = sorted({(r["concept"], r["part"]) for r in rows if r["level"] == level})
    deltas = []
    for (c, p) in keys:
        a = [r[metric] for r in _by(rows, concept=c, part=p, level=level, condition=cond_a) if r[metric] == r[metric]]
        b = [r[metric] for r in _by(rows, concept=c, part=p, level=level, condition=cond_b) if r[metric] == r[metric]]
        if a and b:
            deltas.append(st.mean(a) - st.mean(b))
    return deltas


def _summ(deltas):
    if not deltas:
        return {"mean_delta": None, "win_rate": None, "n": 0, "wilcoxon_p": None}
    wins = sum(1 for d in deltas if d > 0)
    p = None
    try:
        from scipy.stats import wilcoxon
        if any(d != 0 for d in deltas):
            p = float(wilcoxon(deltas).pvalue)
    except Exception:
        p = None
    return {"mean_delta": st.mean(deltas), "win_rate": wins / len(deltas), "n": len(deltas), "wilcoxon_p": p}


def make_or_break(rows, metric="dino", levels=(0, 1)):
    return {lvl: _summ(_pairs(rows, "hub", "rawnn", lvl, metric)) for lvl in levels}


def borrowing_helps(rows, metric="dino", level=0):
    return {"hub_minus_isolated": _summ(_pairs(rows, "hub", "isolated", level, metric)),
            "rawnn_minus_isolated": _summ(_pairs(rows, "rawnn", "isolated", level, metric))}


def _heldout_count(held_recs, concept, part):
    return sum(1 for h in held_recs if h["concept"] == concept and h["part"] == part)


def _bucket(n):
    if n <= 1:
        return "held<=1"
    if n <= 2:
        return "held2"
    if n <= 4:
        return "held3-4"
    return "held5+"


def stratified_delta(rows, held_recs, cond_a, cond_b, metric="dino", level=0):
    keys = sorted({(r["concept"], r["part"]) for r in rows if r["level"] == level})
    buckets = defaultdict(list)
    for (c, p) in keys:
        a = [r[metric] for r in _by(rows, concept=c, part=p, level=level, condition=cond_a) if r[metric] == r[metric]]
        b = [r[metric] for r in _by(rows, concept=c, part=p, level=level, condition=cond_b) if r[metric] == r[metric]]
        if a and b:
            buckets[_bucket(_heldout_count(held_recs, c, p))].append(st.mean(a) - st.mean(b))
    return {bk: _summ(v) for bk, v in buckets.items()}
