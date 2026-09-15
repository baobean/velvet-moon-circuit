from __future__ import annotations
import math
from collections import defaultdict

def _pred_fail(v): return v == "FAIL"           # ABSTAIN and PASS both -> not routed

def confusion(cases):
    tp = fp = tn = fn = nab = 0
    for c in cases:
        if c["verdict"] == "ABSTAIN": nab += 1
        pred, lab = _pred_fail(c["verdict"]), (c["label"] == "fail")
        if lab and pred: tp += 1
        elif lab and not pred: fn += 1
        elif not lab and pred: fp += 1
        else: tn += 1
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fp_rate = fp / (fp + tn) if (fp + tn) else float("nan")
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = ((tp*tn - fp*fn) / denom) if denom else 0.0
    n = tp + fp + tn + fn
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "recall": recall, "fp_rate": fp_rate,
            "mcc": mcc, "accuracy": (tp+tn)/n if n else float("nan"), "n_abstain": nab, "n": n}

def gate(conf):
    r = conf["recall"] >= 0.75
    f = conf["fp_rate"] <= 2 / 24  # spec §6.6: FP <= 0.083 (<= 2/24), exact at the boundary
    return {"recall_ok": bool(r), "fp_ok": bool(f),
            "decision": "PASS" if (r and f) else "NEGATIVE"}

def crosstab(cases):
    both = mmkg_only = sem_only = neither = 0
    for c in cases:
        if c["label"] != "fail": continue
        m = _pred_fail(c["verdict"]); srt = not bool(c.get("semantic_ok", True))  # semantic routes on NOT ok
        if m and srt: both += 1
        elif m and not srt: mmkg_only += 1
        elif not m and srt: sem_only += 1
        else: neither += 1
    return {"both": both, "mmkg_only_catch": mmkg_only, "semantic_only_catch": sem_only,
            "neither": neither}

def stratify(cases, key):
    groups = defaultdict(list)
    for c in cases:
        groups[c.get(key)].append(c)
    return {str(k): confusion(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}
