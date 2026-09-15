"""M2 Lever A: SDXL + IP-Adapter generation from a KG-structured prompt and
a medoid-selected exemplar reference image."""
from __future__ import annotations

from typing import Tuple

from PIL import Image

from graft.prompt import compose_prompt
from graft.schema import ConceptKG
from graft.selection import medoid_index

_DEFAULT_NEGATIVE = "monochrome, lowres, bad anatomy, worst quality, low quality"


def select_exemplar(kg: ConceptKG) -> str:
    """Name-free medoid of the concept's stored reference embeddings.

    A Phase-1 `kg.json` predates `ref_embeddings` and would reach
    `medoid_index([])` as an empty-matrix matmul: that surfaced as a worker
    crash, was reclassified as "no signal, try the next seed", exhausted the
    refine budget, and dropped the whole GRAFT row from the readout. Fail
    loudly and self-describingly instead."""
    if not kg.ref_embeddings or len(kg.ref_embeddings) != len(kg.ref_paths):
        raise ValueError(
            f"{kg.concept}: kg.json has no aligned ref_embeddings — rebuild it "
            f"(sprint spec section 6)"
        )
    return kg.ref_paths[medoid_index(kg.ref_embeddings)]


def generate_lever_a(
    kg: ConceptKG, models, cfg, seed: int
) -> Tuple[Image.Image, str]:
    """Compose the KG-structured (optionally name-neutralized) prompt, select
    the medoid exemplar, and generate one SDXL+IP-Adapter sample."""
    prompt = compose_prompt(kg, neutralize=cfg.neutralize_name)
    ip_image = Image.open(select_exemplar(kg)).convert("RGB")
    image = models.generator.generate(
        prompt=prompt,
        ip_image=ip_image,
        seed=seed,
        negative=_DEFAULT_NEGATIVE,
        steps=50,
    )
    return image, prompt
