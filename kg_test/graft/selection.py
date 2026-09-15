"""Name-free exemplar selection: pick the most representative reference photo
as the medoid of that concept's reference embeddings. Pure numpy so both
generate (GRAFT) and baselines (B1) select the SAME exemplar for parity."""
from __future__ import annotations

import numpy as np


def medoid_index(embeddings) -> int:
    """Index of the reference whose embedding is closest to the mean of all
    references. Rows are assumed L2-normalized (as every graft Embedder
    guarantees), so closeness reduces to the largest dot product with the mean."""
    embs = np.asarray(embeddings, dtype=float)
    mean = embs.mean(axis=0)
    return int(np.argmax(embs @ mean))
