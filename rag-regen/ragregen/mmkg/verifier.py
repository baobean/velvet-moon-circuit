from __future__ import annotations
from ragregen.mmkg.schema import norm, NOT_VISIBLE, decidable

def attribute_states(targets, read, judge_fn):
    out = {}
    for slot, meta in targets.items():
        if not meta.get("is_target"):
            continue
        t = meta["value"]; r = read.get(slot, "not visible")
        if norm(r) in NOT_VISIBLE:
            out[slot] = {"target": t, "read": r, "present": False,
                         "match": None, "contradicted": False}
        elif t == norm(r):
            out[slot] = {"target": t, "read": r, "present": True,
                         "match": True, "contradicted": False}
        else:
            m = bool(judge_fn(t, r, slot))
            out[slot] = {"target": t, "read": r, "present": True,
                         "match": m, "contradicted": not m}
    return out

def case_verdict(targets, read, judge_fn):
    states = attribute_states(targets, read, judge_fn)
    n_present = sum(1 for s in states.values() if s["present"])
    n_contra = sum(1 for s in states.values() if s["contradicted"])
    dec = decidable(targets)
    if not dec or n_present == 0:
        verdict = "ABSTAIN"
    elif n_contra >= 1:
        verdict = "FAIL"
    else:
        verdict = "PASS"
    return {"verdict": verdict, "n_present": n_present, "n_contradicted": n_contra,
            "decidable": dec, "per_attribute": states}
