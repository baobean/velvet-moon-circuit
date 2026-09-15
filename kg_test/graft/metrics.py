"""Evaluation metrics: concept fidelity (primary), attribute accuracy,
prompt alignment (secondary) -- spec section 4."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from PIL import Image


def mean_cosine(gen_vec: np.ndarray, ref_mat: np.ndarray) -> float:
    """Mean cosine similarity between one embedding and a set of reference
    embeddings. Assumes every row is already L2-normalized (as every
    Embedder in graft.interfaces guarantees), so this is a plain dot
    product, not a full cosine computation."""
    return float((ref_mat @ gen_vec).mean())


def image_fidelity(gen: Image.Image, refs: Sequence[Image.Image], embedder) -> float:
    """Mean embedding similarity between a generated image and held-out real
    reference images -- the primary "did we render the rare concept" metric
    (spec section 4)."""
    gen_vec = embedder.embed_image([gen])[0]
    ref_mat = embedder.embed_image(list(refs))
    return mean_cosine(gen_vec, ref_mat)


def clip_t(gen: Image.Image, text: str, clip_embedder) -> float:
    """CLIP-I/T style text-image alignment (secondary; known-weak for rare
    concepts, spec section 4)."""
    img_vec = clip_embedder.embed_image([gen])[0]
    text_vec = clip_embedder.embed_text([text])[0]
    return float(np.dot(img_vec, text_vec))


def attribute_accuracy(gen: Image.Image, attr_texts: Sequence[str], vlm) -> float:
    """Fraction of KG attributes the VLM confirms are present -- TIFA-style
    checklist accuracy."""
    if not attr_texts:
        return 1.0
    passes = 0
    for attr in attr_texts:
        reply = vlm.ask(gen, f'Does this image show "{attr}"? Answer only "yes" or "no".')
        if reply.strip().lower().startswith("yes"):
            passes += 1
    return passes / len(attr_texts)
