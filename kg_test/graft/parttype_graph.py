from __future__ import annotations
import numpy as np

def induce_parttypes(embeds, *, min_cluster_size: int = 3, min_samples: int = 1):
    from sklearn.cluster import HDBSCAN
    embeds = np.asarray(embeds, dtype=float)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                        metric="euclidean")  # euclidean on L2-normed ~ cosine
    return clusterer.fit_predict(embeds)

def hub_coherence(embeds, labels) -> dict[int, float]:
    embeds = np.asarray(embeds, dtype=float)
    labels = np.asarray(labels)
    out = {}
    for h in sorted({int(l) for l in labels if l != -1}):
        members = embeds[labels == h]
        centroid = members.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) or 1.0)
        out[h] = float((members @ centroid).mean())
    return out

def valid_hubs(labels, concepts, coherences, *, min_concepts: int = 2,
               min_coherence: float = 0.5) -> set[int]:
    labels = np.asarray(labels)
    concepts = list(concepts)
    valid = set()
    for h, coh in coherences.items():
        idx = np.where(labels == h)[0]
        n_concepts = len({concepts[i] for i in idx})
        if n_concepts >= min_concepts and coh >= min_coherence:
            valid.add(int(h))
    return valid

def centroid_prototype(embeds, labels, hub_id, concepts, *, k,
                       max_per_concept: int = 2, exclude_concept=None) -> list[int]:
    embeds = np.asarray(embeds, dtype=float); labels = np.asarray(labels)
    concepts = list(concepts)
    idx = [i for i in np.where(labels == hub_id)[0] if concepts[i] != exclude_concept]
    if not idx:
        return []
    members = embeds[idx]
    centroid = members.mean(axis=0); centroid /= (np.linalg.norm(centroid) or 1.0)
    order = sorted(idx, key=lambda i: -float(embeds[i] @ centroid))
    chosen, per = [], {}
    for i in order:
        c = concepts[i]
        if per.get(c, 0) >= max_per_concept:
            continue
        chosen.append(i); per[c] = per.get(c, 0) + 1
        if len(chosen) == k:
            break
    return chosen
