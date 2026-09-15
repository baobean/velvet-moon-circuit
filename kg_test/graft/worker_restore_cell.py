"""GPU worker: one part-level restore cell. Pure input assembly (mask target, build
conditioning refs via restore.select_refs, gather scoring crops) is factored into
build_restore_inputs so the per-cell worker AND the resident runner share ONE consumer;
arms differ only in the selector (spec §3). GPU steps: inpaint the box, score the SAM crop."""
from __future__ import annotations
import argparse, json, os
import numpy as np
from PIL import Image
from graft.config import GraftConfig
from graft.graph_schema import PartTypeGraph
from graft import restore, parts


def _held_for(held_recs, concept, part):
    return [h for h in held_recs if h["concept"] == concept and h["part"] == part]


def build_restore_inputs(graph, held_recs, concept, part, target_idx, build_P, condition, k_eff, draw):
    cand = _held_for(held_recs, concept, part)
    target = cand[target_idx]
    base = Image.open(target["ref_path"]).convert("RGB")
    box = tuple(target["box"])
    sam_mask = np.load(target["mask_path"])
    masked_img, box_mask = parts.box_mask_image(base, box)
    box_mask_pil = Image.fromarray((box_mask * 255).astype(np.uint8), mode="L")
    scoring_crops = [Image.open(h["crop_path"]).convert("RGB")
                     for j, h in enumerate(cand) if j != target_idx]
    own_ids, bidx, pool = restore.select_refs(graph, concept, part, build_P, condition, k_eff, draw)
    ref_imgs, weights = restore.assemble_ref_images(own_ids, bidx, pool, graph)
    borrowed_ids = [pool[i].id for i in bidx]
    return {"orig_img": base, "masked_img": masked_img, "box_mask_pil": box_mask_pil,
            "ref_imgs": ref_imgs, "weights": weights, "scoring_crops": scoring_crops, "box": box,
            "sam_mask": sam_mask, "k_eff": len(bidx), "target_ref": target["ref_path"],
            "borrowed_ids": borrowed_ids}


def restore_row(concept, part, level, condition, draw, k_eff, scores):
    return {"concept": concept, "part": part, "level": level, "draw": draw,
            "condition": condition, "k_eff": k_eff, **scores}


def _score_restored(result_img, box, sam_mask, scoring_crops, models):
    from graft import metrics
    bb = parts.mask_bbox(sam_mask) or box
    restored_crop = result_img.crop(bb)
    if not scoring_crops:
        return {"dino": float("nan"), "siglip2": float("nan"), "clip_i": float("nan")}
    return {"dino": float(metrics.image_fidelity(restored_crop, scoring_crops, models.dino)),
            "siglip2": float(metrics.image_fidelity(restored_crop, scoring_crops, models.siglip)),
            "clip_i": float(metrics.image_fidelity(restored_crop, scoring_crops, models.clip))}


def _inpaint(gen, cfg, prompt, inp):
    """Per-part inpaint: fill ONLY the masked box (conditioned on the borrowed crops + real
    surrounding context) and composite it back onto the pristine original, so the edit is
    localized to the part (confound #2) and every coord stays in original-image space."""
    refs = inp["ref_imgs"] if inp["ref_imgs"] else [inp["masked_img"]]
    wts = inp["weights"] if inp["ref_imgs"] else [1.0]
    orig = inp["orig_img"]
    out = gen.inpaint_multi(prompt, orig, inp["box_mask_pil"], refs, wts, seed=0,
                            steps=cfg.restore_inpaint_steps, strength=cfg.restore_inpaint_strength,
                            guidance=cfg.restore_guidance)
    # SDXL-inpaint generates at 1024x1024 regardless of input size; map back to native size.
    if out.size != orig.size:
        out = out.resize(orig.size)
    x0, y0, x1, y1 = inp["box"]
    composited = orig.copy()
    composited.paste(out.crop((x0, y0, x1, y1)), (x0, y0))   # edit only the part region
    return composited


def main():
    ap = argparse.ArgumentParser()
    for f in ("concept", "part", "condition", "graph", "heldout", "out", "config"):
        ap.add_argument(f"--{f}", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--draw", type=int, required=True)
    ap.add_argument("--target-idx", type=int, required=True)
    a = ap.parse_args()
    cfg = GraftConfig.from_yaml(a.config)
    graph = PartTypeGraph.from_json(a.graph)
    with open(a.heldout) as f:
        held_recs = json.load(f)
    from graft.models import Models
    k_eff = restore.compute_k_eff(graph, a.concept, a.part, k=cfg.restore_k)
    inp = build_restore_inputs(graph, held_recs, a.concept, a.part, a.target_idx,
                               a.level, a.condition, k_eff, a.draw)
    models = Models(cfg)
    gen = models.generator
    prompt = "a photo of a plant " + a.part
    img = _inpaint(gen, cfg, prompt, inp)
    models.unload("generator")
    scores = _score_restored(img, inp["box"], inp["sam_mask"], inp["scoring_crops"], models)
    os.makedirs(a.out, exist_ok=True)
    row = restore_row(a.concept, a.part, a.level, a.condition, a.draw, inp["k_eff"], scores)
    with open(os.path.join(a.out, f"{a.concept}_{a.part}_{a.level}_{a.draw}_{a.condition}.json"), "w") as f:
        json.dump(row, f)


if __name__ == "__main__":
    main()
