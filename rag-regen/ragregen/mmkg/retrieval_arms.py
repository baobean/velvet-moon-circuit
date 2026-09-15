from __future__ import annotations

def flat_arm(hits):
    pool = sorted(hits, key=lambda h: h.get("rank", 0))
    sel = max(hits, key=lambda h: h["score"]) if hits else None
    return {"candidate_pool": pool, "selected_ref": sel}

def mmkg_arm(instances, targets):
    # targets reserved for target-slot coverage selection in a later iteration; Phase-1 coverage groups by part_type
    # coverage-aware set: one representative (max score) per distinct part_type present
    by_pt = {}
    for it in instances:
        pt = it.get("part_type")
        if pt not in by_pt or it["score"] > by_pt[pt]["score"]:
            by_pt[pt] = it
    selected_set = list(by_pt.values())
    sel = max(instances, key=lambda h: h["score"]) if instances else None
    return {"candidate_pool": instances, "selected_set": selected_set, "selected_ref": sel}

def _paths(arm):
    return {h["path"] for h in arm["candidate_pool"]}

def divergence(flat, mmkg):
    pf, pm = _paths(flat), _paths(mmkg)
    union = pf | pm
    jac = (len(pf & pm) / len(union)) if union else 0.0
    fi = (flat["selected_ref"] or {}).get("path")
    mi = (mmkg["selected_ref"] or {}).get("path")
    return {"jaccard": jac, "n_flat": len(pf), "n_mmkg": len(pm),
            "selected_ref_identical": fi is not None and fi == mi}

def coverage(flat, mmkg):
    pts = {it.get("part_type") for it in mmkg["candidate_pool"] if it.get("part_type") is not None}
    fi = (flat["selected_ref"] or {}).get("path")
    return {"flat_unit_size": 1, "mmkg_unit_size": len(mmkg.get("selected_set", [])),
            "mmkg_part_types": len(pts), "flat_top1_in_mmkg_pool": fi in _paths(mmkg)}
