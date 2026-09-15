"""GPU worker: one starvation cell. Build the (possibly starved) own-crop set for
one part-conditioned generation, borrow k crops via the chosen selector (or none
for 'isolated'), condition once via generate_multi, score vs held-out. All arms
share this single consumer; only select_borrowed(kind=...) differs (§7.2 invariant)."""
from __future__ import annotations
import argparse, json, os
import numpy as np
from PIL import Image
from graft.config import GraftConfig
from graft.dataset import load_species, split_refs
from graft.graph_schema import PartTypeGraph
from graft.schema import ConceptKG
from graft.starvation import keep_draw
from graft.transfer import select_borrowed, reference_weights
from graft.models import Models
from graft import metrics                              # metrics.image_fidelity(gen, refs, embedder)

K = 4

def _own_crops(graph, concept, kept_refs):
    """This concept's crops whose ref survived starvation (any part)."""
    return [i for i in graph.part_instances
            if i.concept == concept and i.ref_path in set(kept_refs)]

def _hub_of(graph, inst_id):
    for h in graph.part_types:
        if inst_id in h.member_ids:
            return h.id
    return None

def build_cell_inputs(graph, kg, concept, level, draw, condition, build_refs):
    """(ref_imgs, weights, prompt) for one starvation cell -- no GPU/models here.
    Shared by the per-cell worker and the resident runner so the k_eff / selection /
    weighting / part-restricted-pool logic lives in exactly ONE place."""
    kept = keep_draw(build_refs, level, seed=draw)
    own = _own_crops(graph, concept, kept)
    own_embeds = np.array([o.siglip2 for o in own], dtype=float) if own else np.zeros((0, 1))
    own_imgs = [Image.open(o.crop_path).convert("RGB") for o in own]

    borrowed_imgs = []
    # "isolated"/"oraclehub" borrow nothing; a concept with no surviving own crops also
    # skips borrowing (no anchor part, no query embedding).
    if condition not in ("isolated", "oraclehub") and own:
        own_part = own[0].part
        # pool restricted to the transfer's part for ALL arms (selection-only invariant + P5).
        pool = [p for p in graph.part_instances if p.part == own_part]
        if pool:
            pool_embeds = np.array([p.siglip2 for p in pool], dtype=float)
            pool_concepts = [p.concept for p in pool]
            pool_labels = np.array([_hub_of(graph, p.id) if _hub_of(graph, p.id) is not None else -1
                                    for p in pool])
            hub_id = _hub_of(graph, own[0].id)
            q = own_embeds[0]
            # hub bounds k_eff; +RawNN/+Random draw EXACTLY k_eff -> arms differ only in WHICH crops.
            hub_sel = select_borrowed("hub", query_embed=q, pool_embeds=pool_embeds,
                                      pool_concepts=pool_concepts, pool_labels=pool_labels,
                                      own_concept=concept, hub_id=hub_id, k=K, seed=draw) if hub_id is not None else []
            k_eff = len(hub_sel)
            if k_eff:
                sel = hub_sel if condition == "hub" else select_borrowed(
                    condition, query_embed=q, pool_embeds=pool_embeds,
                    pool_concepts=pool_concepts, pool_labels=pool_labels,
                    own_concept=concept, hub_id=hub_id, k=k_eff, seed=draw)
                borrowed_imgs = [Image.open(pool[i].crop_path).convert("RGB") for i in sel]

    ref_imgs = own_imgs + borrowed_imgs
    weights = [] if not ref_imgs else reference_weights(len(own_imgs), len(borrowed_imgs)).tolist()
    prompt = "a photo of a plant, " + ", ".join(kg.attribute_texts()[:8])
    return ref_imgs, weights, prompt

def main():
    ap = argparse.ArgumentParser()
    for f in ("concept","condition","graph","kg","root","out","config"):
        ap.add_argument(f"--{f}", required=True)
    ap.add_argument("--level", type=int, required=True); ap.add_argument("--draw", type=int, required=True)
    a = ap.parse_args()
    cfg = GraftConfig.from_yaml(a.config)
    graph = PartTypeGraph.from_json(a.graph); kg = ConceptKG.from_json(a.kg)
    sp = load_species(a.root, a.concept)
    build_refs, heldout = split_refs(sp, k_build=cfg.k_build_refs, seed=0)  # seed=0 matches run_eval_fast
    ref_imgs, weights, prompt = build_cell_inputs(
        graph, kg, a.concept, a.level, a.draw, a.condition, build_refs)
    models = Models(cfg)
    gen = models.generator                              # SdxlIpGenerator via the facade
    img = gen.generate_multi(prompt, ref_imgs, weights, seed=0) if ref_imgs else gen.generate(prompt, None, seed=0)
    models.unload("generator")                          # free SDXL+IP before loading scorers (single-GPU)
    held_imgs = [Image.open(p).convert("RGB") for p in heldout]
    scores = {                                          # metrics.image_fidelity = mean cosine vs held-out
        "dino": metrics.image_fidelity(img, held_imgs, models.dino),
        "siglip2": metrics.image_fidelity(img, held_imgs, models.siglip),
        "clip_i": metrics.image_fidelity(img, held_imgs, models.clip),
    }
    os.makedirs(a.out, exist_ok=True)
    row = {"concept": a.concept, "level": a.level, "draw": a.draw,
           "condition": a.condition, **scores}
    with open(os.path.join(a.out, f"{a.concept}_{a.level}_{a.draw}_{a.condition}.json"), "w") as f:
        json.dump(row, f)

if __name__ == "__main__":
    main()
