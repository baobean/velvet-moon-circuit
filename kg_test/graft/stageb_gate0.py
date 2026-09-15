"""Stage-B Gate 0 (premise): does taxonomy attribute-transfer predict a held-out target
concept's RARE attributes better than visual-NN retrieval? Pure logic + orchestration; every
VLM call lives in graft.stageb_vlm so this module is unit-tested with a mock VLM. Fully
pre-registered per docs/superpowers/specs/2026-09-08-stage-b-attribute-transfer-pilot-design.md."""
from __future__ import annotations
import glob, json, math, os
from collections import Counter, defaultdict
import numpy as np
from graft.dataset import load_species, split_refs
from graft import taxonomy
from scipy.stats import binomtest

NOT_VISIBLE = {"not visible", "none", "n/a", ""}
PART_ATTRS = [("leaf", "shape"), ("leaf", "texture"), ("bark", "texture"),
              ("bark", "color"), ("branching", "pattern")]
_PA = set(PART_ATTRS)


def norm(v) -> str:
    return str(v).strip().lower()


# ---- Task 1: loaders + build set ----
def load_build_attrs(kg_glob="outputs/*/kg.json"):
    out = {}
    for kp in glob.glob(kg_glob):
        kg = json.load(open(kp)); c = kg["concept"]; d = {}
        for pr in kg.get("parts", []):
            for at in pr.get("attributes", []):
                key = (pr["name"], at["name"])
                if key in _PA:
                    v = norm(at.get("value"))
                    if v not in NOT_VISIBLE:
                        d[key] = v
        out[c] = d
    return out


def load_exemplars(kg_glob="outputs/*/kg.json"):
    out = {}
    for kp in glob.glob(kg_glob):
        kg = json.load(open(kp)); c = kg["concept"]; d = {}
        for pr in kg.get("parts", []):
            v = pr.get("embeddings", {}).get("siglip2")
            if v:
                a = np.asarray(v, float); d[pr["name"]] = a / (np.linalg.norm(a) or 1.0)
        out[c] = d
    return out


def build_image_set(concept, root="data/treevill/rawdata2"):
    return set(split_refs(load_species(root, concept), 5, 0)[0])


# ---- Task 2: disjoint held-out majority label ----
def heldout_label(records, build_set, read_fn):
    seen, imgs = set(), []
    for r in records:
        p = r["ref_path"]
        if p in build_set or p in seen:
            continue
        seen.add(p); imgs.append(p)
    assert not (set(imgs) & set(build_set)), "held-out label images overlap build set"
    votes = defaultdict(Counter)
    for p in imgs:
        for key, val in read_fn(p).items():
            v = norm(val)
            if v not in NOT_VISIBLE:
                votes[key][v] += 1
    out = {}
    for key, c in votes.items():
        top = c.most_common()
        if len(top) == 1 or top[0][1] > top[1][1]:
            out[key] = top[0][0]
    return out


# ---- Task 3: rarity ----
def rare_values(labels_by_concept, part, attr):
    key = (part, attr)
    vals = [d[key] for d in labels_by_concept.values() if key in d]
    n_vis = len(vals)
    if n_vis == 0:
        return set()
    thr = max(2, math.ceil(0.15 * n_vis))
    cnt = Counter(vals)
    return {v for v, k in cnt.items() if k <= thr}


# ---- Task 4: arms ----
def _cos(a, b):
    a = a / (np.linalg.norm(a) or 1.0); b = b / (np.linalg.norm(b) or 1.0)
    return float(a @ b)


def retrieval_pred(concept, part, attr, exemplars, build_attrs):
    q = exemplars.get(concept, {}).get(part)
    if q is None:
        return None
    key = (part, attr)
    cands = [(o, _cos(q, exemplars[o][part])) for o in exemplars
             if o != concept and part in exemplars[o] and key in build_attrs.get(o, {})]
    if not cands:
        return None
    cands.sort(key=lambda x: -x[1])
    return build_attrs[cands[0][0]][key]


def mmkg_pred(concept, part, attr, exemplars, build_attrs):
    fam = taxonomy.family_of(concept)
    if fam is None:
        return None
    key = (part, attr)
    sibs = [o for o in build_attrs if o != concept
            and taxonomy.family_of(o) == fam and key in build_attrs[o]]
    if not sibs:
        return None
    cnt = Counter(build_attrs[o][key] for o in sibs)
    top = cnt.most_common()
    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0]
    tied = {v for v, k in top if k == top[0][1]}
    q = exemplars.get(concept, {}).get(part)
    best = max((o for o in sibs if build_attrs[o][key] in tied),
               key=lambda o: _cos(q, exemplars[o][part])
               if (q is not None and part in exemplars[o]) else -1.0)
    return build_attrs[best][key]


# ---- Task 6: cases, scoring, McNemar, gate ----
def build_cases(labels_by_concept, exemplars, build_attrs):
    cases, rare_cache = [], {}
    for c, lab in labels_by_concept.items():
        for (part, attr), val in lab.items():
            pr = retrieval_pred(c, part, attr, exemplars, build_attrs)
            pm = mmkg_pred(c, part, attr, exemplars, build_attrs)
            if pr is None or pm is None:
                continue
            if (part, attr) not in rare_cache:
                rare_cache[(part, attr)] = rare_values(labels_by_concept, part, attr)
            cases.append({"concept": c, "part": part, "attr": attr, "label": val,
                          "rare": val in rare_cache[(part, attr)],
                          "pred_retrieval": pr, "pred_mmkg": pm})
    return cases


def score_cases(cases, judge_fn):
    out = []
    for k in cases:
        a = f"{k['part']} {k['attr']}"
        r = dict(k)
        r["match_retrieval"] = bool(judge_fn(k["pred_retrieval"], k["label"], a))
        r["match_mmkg"] = bool(judge_fn(k["pred_mmkg"], k["label"], a))
        out.append(r)
    return out


def mcnemar_p(scored):
    rare = [s for s in scored if s["rare"]]
    b = sum(1 for s in rare if s["match_mmkg"] and not s["match_retrieval"])
    c = sum(1 for s in rare if s["match_retrieval"] and not s["match_mmkg"])
    if b + c == 0:
        return 1.0
    return float(binomtest(min(b, c), b + c, 0.5).pvalue)


def _recall(scored, arm, rare):
    sub = [s for s in scored if s["rare"] == rare]
    if not sub:
        return float("nan")
    return sum(1 for s in sub if s[f"match_{arm}"]) / len(sub)


def gate0(scored):
    n_rare = sum(1 for s in scored if s["rare"])
    rr_m = _recall(scored, "mmkg", True); rr_r = _recall(scored, "retrieval", True)
    delta = rr_m - rr_r; p = mcnemar_p(scored)
    decision = "PASS" if (delta >= 0.15 and n_rare >= 30 and p < 0.05) else "WIND_DOWN"
    return {"n_rare": n_rare, "n_total": len(scored),
            "rare_recall_mmkg": rr_m, "rare_recall_retrieval": rr_r, "delta": delta,
            "mcnemar_p": p, "common_recall_mmkg": _recall(scored, "mmkg", False),
            "common_recall_retrieval": _recall(scored, "retrieval", False),
            "decision": decision}


# ---- Task 7: feasibility + main ----
def feasibility(labels_by_concept, exemplars, build_attrs):
    cases = build_cases(labels_by_concept, exemplars, build_attrs)
    by_attr = Counter((c["part"], c["attr"]) for c in cases if c["rare"])
    return {"n_included": len(cases), "n_rare": sum(1 for c in cases if c["rare"]),
            "by_attr": {f"{p}.{a}": n for (p, a), n in by_attr.items()}}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-dir", default="outputs/mmkg_oh/heldout_parts")
    ap.add_argument("--kg-glob", default="outputs/*/kg.json")
    ap.add_argument("--root", default="data/treevill/rawdata2")
    ap.add_argument("--out", default="outputs/mmkg/stageb_gate0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    build_attrs = load_build_attrs(a.kg_glob)
    exemplars = load_exemplars(a.kg_glob)
    evalc = sorted(os.path.splitext(x)[0] for x in os.listdir(a.heldout_dir))

    proxy = {c: build_attrs.get(c, {}) for c in evalc}
    feas = feasibility(proxy, exemplars, build_attrs)
    print("feasibility (proxy):", feas, flush=True)
    if feas["n_rare"] < 30:
        json.dump({"decision": "WIND_DOWN_FEASIBILITY", "feasibility": feas},
                  open(os.path.join(a.out, "result.json"), "w"), indent=2)
        print("WIND_DOWN_FEASIBILITY", flush=True); return

    from graft.config import GraftConfig
    from graft.models import Models
    from graft import stageb_vlm
    cfg = GraftConfig.from_yaml("configs/oraclehub.yaml")
    models = Models(cfg)
    labels = {}
    for c in evalc:
        recs = json.load(open(os.path.join(a.heldout_dir, f"{c}.json")))
        B = build_image_set(c, a.root)
        labels[c] = heldout_label(recs, B, lambda p: stageb_vlm.read_attributes([p], models))
        print(f"[label] {c}: {len(labels[c])} attrs", flush=True)
    cases = build_cases(labels, exemplars, build_attrs)
    print(f"cases={len(cases)} rare={sum(1 for c in cases if c['rare'])}", flush=True)
    scored = score_cases(cases, lambda x, y, attr: stageb_vlm.judge_match(x, y, attr, models))
    out = gate0(scored)
    out["feasibility"] = feas
    json.dump({"result": out, "cases": scored},
              open(os.path.join(a.out, "result.json"), "w"), indent=2)
    print("DECISION:", out["decision"], "delta=%.3f" % out["delta"],
          "n_rare=%d" % out["n_rare"], "p=%.4f" % out["mcnemar_p"],
          "mmkg=%.3f retr=%.3f" % (out["rare_recall_mmkg"], out["rare_recall_retrieval"]), flush=True)


if __name__ == "__main__":
    main()
