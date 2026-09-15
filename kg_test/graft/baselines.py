"""B0/B1/B2 baselines for the eval driver (spec section 4).

B0: vanilla SDXL prompt -- no KG, no reference.
B1: ImageRAG-style flat retrieval -- one unstructured reference image (the
    SigLIP2 medoid of the pool, the SAME exemplar GRAFT selects -> exemplar
    parity) via IP-Adapter, no graph structure, no anchor.
B2: RAVEL/Context Canvas-style reproduction -- LLM-recalled symbolic
    attributes (no visual grounding) expanded into the prompt, text-only
    SDXL. This is the key ablation isolating GRAFT's "visually grounded +
    reference-conditioned" contribution.
"""
from __future__ import annotations

from typing import List, Sequence

from PIL import Image

from graft.prompt import compose_ravel_prompt, concept_token
from graft.selection import medoid_index

_DEFAULT_NEGATIVE = "monochrome, lowres, bad anatomy, worst quality, low quality"

RAVEL_ATTR_INSTRUCTION = (
    "List up to 8 concise visual attributes of a {concept} (its typical "
    "shape, color, texture, and distinguishing features), based on what you "
    "already know -- do not look at any image. Reply with ONLY a JSON list "
    'of short strings, e.g. ["...", "..."].'
)


def b0_vanilla(concept: str, models, cfg, seed: int) -> Image.Image:
    prompt = f"a photo of a {concept_token(concept, cfg.neutralize_name)}"
    return models.generator.generate(
        prompt=prompt, ip_image=None, seed=seed, negative=_DEFAULT_NEGATIVE, steps=50
    )


def b1_imagerag(concept: str, pool_paths: Sequence[str], models, cfg, seed: int) -> Image.Image:
    """Flat single-reference conditioning: the SigLIP2 medoid of the pool (the
    SAME exemplar GRAFT selects -> exemplar parity), IP-Adapter, no graph."""
    pool_images = [Image.open(p).convert("RGB") for p in pool_paths]
    ip_image = pool_images[medoid_index(models.siglip.embed_image(pool_images))]
    prompt = f"a photo of a {concept_token(concept, cfg.neutralize_name)}"
    return models.generator.generate(
        prompt=prompt, ip_image=ip_image, seed=seed, negative=_DEFAULT_NEGATIVE, steps=50
    )


def _parse_attr_list(raw: str) -> List[str]:
    import json

    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        items = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [str(x) for x in items]


def b2_ravel(concept: str, models, cfg, seed: int) -> Image.Image:
    """VLM lists attributes from parametric memory (no image shown) ->
    KG-shaped prompt template, but with zero visual grounding -> text-only
    SDXL. Reproduces RAVEL/Context Canvas's core mechanism. Attributes are
    still recalled by the real name; only the final prompt head is neutralized."""
    blank = Image.new("RGB", (16, 16), (128, 128, 128))
    raw = models.vlm.describe([blank], RAVEL_ATTR_INSTRUCTION.format(concept=concept))
    attrs = _parse_attr_list(raw)
    prompt = compose_ravel_prompt(concept, attrs, neutralize=cfg.neutralize_name)
    return models.generator.generate(
        prompt=prompt, ip_image=None, seed=seed, negative=_DEFAULT_NEGATIVE, steps=50
    )
