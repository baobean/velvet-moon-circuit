from __future__ import annotations
import math
from collections import Counter

SLOTS = ["primary_color", "secondary_color", "pattern_or_markings",
         "surface_texture", "overall_shape_or_form", "distinctive_feature"]
NOT_VISIBLE = {"not visible", "none", "n/a", ""}

def norm(v) -> str:
    return str(v).strip().lower()

def target_attributes(reads):
    out = {}
    for slot in SLOTS:
        vals = [norm(r.get(slot)) for r in reads]
        vis = [v for v in vals if v not in NOT_VISIBLE]
        if not vis:
            continue
        cnt = Counter(vis)
        value, support = cnt.most_common(1)[0]
        thr = max(2, math.ceil(0.5 * len(vis)))
        is_target = support >= thr and len(vis) >= 3
        out[slot] = {"value": value, "support": support,
                     "visible_count": len(vis), "is_target": is_target}
    return out

def n_targets(targets) -> int:
    return sum(1 for v in targets.values() if v["is_target"])

def decidable(targets) -> bool:
    return n_targets(targets) >= 2
