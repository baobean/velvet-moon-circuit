"""The ONE shared, pure builder for a part-level restore cell. Assembles the
conditioning references (own P crops starved to build_P + k_eff borrowed via the
arm's selector) so every arm goes through identical code and differs ONLY in the
selector (spec §3 invariant). No GPU here: masking/inpaint/scoring live in the worker."""
from __future__ import annotations
import numpy as np
from PIL import Image
from graft.transfer import select_borrowed, reference_weights
from graft.starvation import keep_draw


def hub_of(graph, inst_id):
    for h in graph.part_types:
        if inst_id in h.member_ids:
            return h.id
    return None


def own_part_crops(graph, concept, part):
    return [i for i in graph.part_instances if i.concept == concept and i.part == part]


def _part_pool(graph, part):
    return [p for p in graph.part_instances if p.part == part]


def compute_k_eff(graph, concept, part, k):
    own = own_part_crops(graph, concept, part)
    if not own:
        return 0
    hub_id = hub_of(graph, own[0].id)
    if hub_id is None:
        return 0
    pool = _part_pool(graph, part)
    pool_embeds = np.array([p.siglip2 for p in pool], dtype=float)
    pool_concepts = [p.concept for p in pool]
    pool_labels = np.array([hub_of(graph, p.id) if hub_of(graph, p.id) is not None else -1 for p in pool])
    sel = select_borrowed("hub", query_embed=own[0].siglip2, pool_embeds=pool_embeds,
                          pool_concepts=pool_concepts, pool_labels=pool_labels,
                          own_concept=concept, hub_id=hub_id, k=k, seed=0)
    return len(sel)


def select_refs(graph, concept, part, build_P, condition, k_eff, draw):
    own_all = own_part_crops(graph, concept, part)
    own_ids = [o.id for o in keep_draw(own_all, build_P, seed=draw)]  # starve own to build_P
    pool = _part_pool(graph, part)
    if condition == "isolated" or k_eff == 0 or not own_all:
        return own_ids, [], pool
    hub_id = hub_of(graph, own_all[0].id)
    pool_embeds = np.array([p.siglip2 for p in pool], dtype=float)
    pool_concepts = [p.concept for p in pool]
    pool_labels = np.array([hub_of(graph, p.id) if hub_of(graph, p.id) is not None else -1 for p in pool])
    bidx = select_borrowed(condition, query_embed=own_all[0].siglip2, pool_embeds=pool_embeds,
                           pool_concepts=pool_concepts, pool_labels=pool_labels,
                           own_concept=concept, hub_id=hub_id, k=k_eff, seed=draw)
    return own_ids, list(bidx), pool


def assemble_ref_images(own_ids, borrowed_pool_idx, pool, graph):
    by_id = graph.instances_by_id()
    own_imgs = [Image.open(by_id[i].crop_path).convert("RGB") for i in own_ids]
    borrowed_imgs = [Image.open(pool[i].crop_path).convert("RGB") for i in borrowed_pool_idx]
    ref_imgs = own_imgs + borrowed_imgs
    weights = [] if not ref_imgs else reference_weights(len(own_imgs), len(borrowed_imgs)).tolist()
    return ref_imgs, weights
