"""OracleHub vs RawNN restore experiment (build_P=0). Tests whether a TAXONOMY hub -- an
oracle relation orthogonal to SigLIP2 -- recovers references that are taxonomically relevant
but embedding-distant, which RawNN misses.

Design (build_P=0, the maximal/cleanest deficit): the query keeps ZERO own crops, so restoration
conditions ONLY on borrowed crops from OTHER concepts. Consequences: (a) the query contributes no
conditioning crop, so ALL its unique images are held-out target+scoring (dissolves the unique-image
blocker); (b) no contamination path (a query never borrows from itself). Borrow pool = the per-part
kg.json EXEMPLAR of every concept (already on disk); taxonomy family labels are the ORACLE signal.
Arms {random, rawnn, oraclehub} are matched at k_eff = #same-family siblings (capped at k). Reuses
the Stage-A' inpaint/score consumer so only the selector differs. Resume-safe (existing PNG/row)."""
from __future__ import annotations
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
from PIL import Image
from graft.config import GraftConfig
from graft import parts, taxonomy
from graft.transfer import select_borrowed, reference_weights
from graft.worker_restore_cell import _inpaint, _score_restored
from graft.oraclehub_analysis import summarize

K_STRATA = ("near", "mid", "far")   # sibling NN-rank <=4 / 5-12 / >12


def load_pool(kg_glob="outputs/*/kg.json"):
    """part -> list of {concept, crop_path, siglip2(np, L2-normed), fam_id}; plus
    exemplar[concept][part] = the same normed vector (the query embedding)."""
    pool = defaultdict(list)
    exemplar = defaultdict(dict)
    for kp in glob.glob(kg_glob):
        with open(kp) as f:
            kg = json.load(f)
        c = kg["concept"]
        cdir = os.path.dirname(kp)
        for pr in kg.get("parts", []):
            v = pr.get("embeddings", {}).get("siglip2")
            if not v:
                continue
            a = np.asarray(v, float); a = a / (np.linalg.norm(a) or 1.0)
            crop = pr.get("exemplar_crop") or os.path.join(cdir, "crops", f"{pr['name']}.png")
            rec = {"concept": c, "crop_path": crop, "siglip2": a, "fam_id": taxonomy.family_id(c)}
            pool[pr["name"]].append(rec)
            exemplar[c][pr["name"]] = a
    return pool, exemplar


def _arm_select(kind, query_embed, pool_part, own_concept, own_fam_id, k_eff, seed):
    pe = np.array([r["siglip2"] for r in pool_part], float)
    pc = [r["concept"] for r in pool_part]
    pl = np.array([r["fam_id"] for r in pool_part])
    return select_borrowed(kind, query_embed=query_embed, pool_embeds=pe, pool_concepts=pc,
                           pool_labels=pl, own_concept=own_concept, hub_id=own_fam_id,
                           k=k_eff, seed=seed)


def sibling_rank(query_embed, pool_part, own_concept, own_fam_id):
    """Best (smallest) NN-rank of a same-family sibling among other-concept part exemplars,
    and the number of siblings. rank 1 = the embedding-nearest candidate."""
    others = [(i, float(r["siglip2"] @ query_embed)) for i, r in enumerate(pool_part)
              if r["concept"] != own_concept]
    others.sort(key=lambda x: -x[1])
    sib_ranks = [rank for rank, (i, _) in enumerate(others, 1)
                 if pool_part[i]["fam_id"] == own_fam_id and own_fam_id != -1]
    return (min(sib_ranks) if sib_ranks else None), len(sib_ranks)


def stratum(rank):
    return "near" if rank <= 4 else ("mid" if rank <= 12 else "far")


def iter_cells(pool, exemplar, held_recs, k, gate_k=4):
    """Gate-A (concept, part) cells: has a same-family sibling exemplar for the part AND the
    best sibling is NOT in NN top-`gate_k` (embedding-distant). k_eff = min(k, #siblings)."""
    held_by = defaultdict(list)
    for h in held_recs:
        held_by[(h["concept"], h["part"])].append(h)
    for (c, p), tgts in sorted(held_by.items()):
        if len(tgts) < 2 or p not in exemplar.get(c, {}):
            continue                                  # need >=2 held-out crops (target + score)
        fam_id = taxonomy.family_id(c)
        if fam_id == -1:
            continue
        q = exemplar[c][p]
        br, n_sib = sibling_rank(q, pool[p], c, fam_id)
        if br is None or br <= gate_k:
            continue                                  # not embedding-distant -> not a real test
        k_eff = min(k, n_sib)
        yield (c, p, len(tgts), k_eff, br)


def _tag(c, p, d, cond):
    return f"{c}_{p}_{d}_{cond}".replace(" ", "-")


def build_inputs(pool, held_by, c, p, tgt, cond, k_eff, q_embed, fam_id, seed):
    tgts = held_by[(c, p)]
    target = tgts[tgt]
    orig = Image.open(target["ref_path"]).convert("RGB")
    box = tuple(target["box"])
    sam_mask = np.load(target["mask_path"])
    masked_img, box_mask = parts.box_mask_image(orig, box)
    box_mask_pil = Image.fromarray((box_mask * 255).astype(np.uint8), mode="L")
    sel = _arm_select(cond, q_embed, pool[p], c, fam_id, k_eff, seed)
    ref_imgs = [Image.open(pool[p][i]["crop_path"]).convert("RGB") for i in sel]
    weights = reference_weights(0, len(ref_imgs)).tolist()
    scoring = [Image.open(h["crop_path"]).convert("RGB") for j, h in enumerate(tgts) if j != tgt]
    borrowed_ids = [f"{pool[p][i]['concept']}::{p}" for i in sel]
    return {"orig_img": orig, "masked_img": masked_img, "box_mask_pil": box_mask_pil,
            "ref_imgs": ref_imgs, "weights": weights, "box": box, "sam_mask": sam_mask,
            "scoring_crops": scoring, "k_eff": len(sel), "borrowed_ids": borrowed_ids}


def main():
    ap = argparse.ArgumentParser()
    for f in ("heldout-dir", "out", "config"):
        ap.add_argument(f"--{f}", required=True)
    ap.add_argument("--kg-glob", default="outputs/*/kg.json")
    ap.add_argument("--concepts", default="")
    a = ap.parse_args()
    cfg = GraftConfig.from_yaml(a.config)
    pool, exemplar = load_pool(a.kg_glob)

    held_recs = []
    for hp in glob.glob(os.path.join(a.heldout_dir, "*.json")):
        with open(hp) as f:
            held_recs.extend(json.load(f))
    if a.concepts:
        want = {c.strip() for c in a.concepts.split(",")}
        held_recs = [h for h in held_recs if h["concept"] in want]
    held_by = defaultdict(list)
    for h in held_recs:
        held_by[(h["concept"], h["part"])].append(h)

    conditions = ["random", "rawnn", "oraclehub"]
    cells = list(iter_cells(pool, exemplar, held_recs, cfg.restore_k))
    # (c, p, draw, cond, k_eff, sib_rank); one distinct target per draw
    plan = []
    for (c, p, n_tgt, k_eff, br) in cells:
        for d in range(min(cfg.restore_n_draws, n_tgt)):
            for cond in conditions:
                plan.append((c, p, d % n_tgt, cond, k_eff, br))
    os.makedirs(a.out, exist_ok=True)
    img_dir = os.path.join(a.out, "_imgs"); os.makedirs(img_dir, exist_ok=True)
    print(f"cells={len(cells)} plan={len(plan)} (build_P=0, arms={conditions})", flush=True)

    from graft.models import Models
    models = Models(cfg)

    # ---- Phase A: inpaint ----
    gen = models.generator
    for i, (c, p, d, cond, k_eff, br) in enumerate(plan):
        png = os.path.join(img_dir, _tag(c, p, d, cond) + ".png")
        if os.path.exists(png):
            continue
        fam_id = taxonomy.family_id(c)
        inp = build_inputs(pool, held_by, c, p, d, cond, k_eff, exemplar[c][p], fam_id, seed=d)
        img = _inpaint(gen, cfg, "a photo of a plant " + p, inp)
        img.save(png)
        import torch; torch.cuda.empty_cache()
        print(f"[inpaint {i+1}/{len(plan)}] {_tag(c,p,d,cond)} keff={k_eff}", flush=True)
    models.unload("generator")

    # ---- Phase B: score ----
    rows = []
    for i, (c, p, d, cond, k_eff, br) in enumerate(plan):
        rowp = os.path.join(a.out, _tag(c, p, d, cond) + ".json")
        if os.path.exists(rowp):
            with open(rowp) as f:
                rows.append(json.load(f)); continue
        fam_id = taxonomy.family_id(c)
        inp = build_inputs(pool, held_by, c, p, d, cond, k_eff, exemplar[c][p], fam_id, seed=d)
        img = Image.open(os.path.join(img_dir, _tag(c, p, d, cond) + ".png")).convert("RGB")
        scores = _score_restored(img, inp["box"], inp["sam_mask"], inp["scoring_crops"], models)
        row = {"concept": c, "part": p, "draw": d, "condition": cond, "k_eff": inp["k_eff"],
               "sib_rank": br, "stratum": stratum(br), **scores}
        with open(rowp, "w") as f:
            json.dump(row, f)
        rows.append(row)
        print(f"[score {i+1}/{len(plan)}] {_tag(c,p,d,cond)} dino={scores['dino']:.3f}", flush=True)

    out = summarize(rows)
    with open(os.path.join(a.out, "oraclehub_result.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("DONE", json.dumps(out.get("oraclehub_vs_rawnn", {}), indent=2), flush=True)


if __name__ == "__main__":
    main()
