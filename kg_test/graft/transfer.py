from __future__ import annotations
import numpy as np
from graft.parttype_graph import centroid_prototype


def _capped_topk(idx, scores, concepts, *, k, max_per_concept) -> list[int]:
    """Greedy top-k over candidate indices ranked by `scores` (higher = better),
    capping how many can come from any one concept. Shared by hub (centroid score)
    and hubnn (query score) so the two differ ONLY in the ranking signal."""
    order = sorted(idx, key=lambda i: -float(scores[i]))
    chosen, per = [], {}
    for i in order:
        c = concepts[i]
        if per.get(c, 0) >= max_per_concept:
            continue
        chosen.append(i); per[c] = per.get(c, 0) + 1
        if len(chosen) == k:
            break
    return chosen


def select_borrowed(kind, *, query_embed, pool_embeds, pool_concepts,
                    pool_labels, own_concept, hub_id, k, seed) -> list[int]:
    pool_embeds = np.asarray(pool_embeds, dtype=float)
    others = [i for i in range(len(pool_concepts)) if pool_concepts[i] != own_concept]
    if kind == "hub":
        return centroid_prototype(pool_embeds, pool_labels, hub_id, pool_concepts,
                                  k=k, max_per_concept=2, exclude_concept=own_concept)
    if kind in ("hubnn", "oraclehub"):
        # Restrict to the query's group (pool_labels == hub_id) and rank by QUERY similarity,
        # 2/concept cap. `hubnn`: the group is the SigLIP2-induced HDBSCAN hub (labels = hub ids)
        # -> isolates hub-gated candidate generation from `hub`'s centroid re-rank. `oraclehub`:
        # the group is a TAXONOMIC family (labels = family ids, an oracle signal orthogonal to the
        # embedding) -> tests whether a relation the embedding can't see beats plain NN. Same code
        # so the two differ ONLY in what the caller passes as pool_labels/hub_id.
        labels = np.asarray(pool_labels)
        cand = [i for i in np.where(labels == hub_id)[0] if pool_concepts[i] != own_concept]
        q = np.asarray(query_embed, dtype=float)
        scores = pool_embeds @ q
        return _capped_topk(cand, scores, pool_concepts, k=k, max_per_concept=2)
    if kind == "rawnn":
        q = np.asarray(query_embed, dtype=float)
        order = sorted(others, key=lambda i: -float(pool_embeds[i] @ q))
        return order[:k]
    if kind == "random":
        rng = np.random.default_rng(seed)
        return list(rng.permutation(others)[:k])
    raise ValueError(f"unknown selector: {kind}")

def reference_weights(n_own, n_borrowed, *, own_per=2.0, borrowed_per=1.0,
                      cap=0.7) -> np.ndarray:
    w = np.array([own_per]*n_own + [borrowed_per]*n_borrowed, dtype=float)
    w = w / w.sum()
    if n_own and n_borrowed and w[n_own:].sum() > cap:
        w[n_own:] *= cap / w[n_own:].sum()
        w[:n_own] *= (1 - cap) / w[:n_own].sum()
    return w
