from __future__ import annotations
import statistics as st
from collections import defaultdict

def _by(rows, **f):
    return [r for r in rows if all(r[k] == v for k, v in f.items())]

def recovery_curve(rows, metric="dino"):
    out = defaultdict(dict)
    conds = {r["condition"] for r in rows}; levels = sorted({r["level"] for r in rows})
    for c in conds:
        for lvl in levels:
            vals = [r[metric] for r in _by(rows, condition=c, level=lvl)]
            if vals:
                out[c][lvl] = st.mean(vals)
    return dict(out)

def paired_delta(rows, cond_a="hub", cond_b="rawnn", level=1, metric="dino"):
    concepts = sorted({r["concept"] for r in rows if r["level"] == level})
    deltas = []
    for c in concepts:
        a = [r[metric] for r in _by(rows, concept=c, level=level, condition=cond_a)]
        b = [r[metric] for r in _by(rows, concept=c, level=level, condition=cond_b)]
        if a and b:
            deltas.append(st.mean(a) - st.mean(b))   # mean over draws, then pair
    wins = sum(1 for d in deltas if d > 0)
    return {"mean_delta": st.mean(deltas) if deltas else None,
            "win_rate": wins / len(deltas) if deltas else None,
            "n": len(deltas)}
