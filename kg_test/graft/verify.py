"""M3: multimodal part-grounded verification.

Unlike RAVEL/Context Canvas's self-correction (SRD), which scores the whole
image from global VLM feedback and rewrites the prompt, this locates each KG
part in the generated image and scores it against that part's real exemplar
crop -- feedback is grounded per-region, not just per-image.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import numpy as np
from PIL import Image

from graft.schema import ConceptKG

ATTR_QUESTION = 'Does this image show "{attr}"? Answer only "yes" or "no".'


@dataclass
class PartScore:
    part: str
    sim: float
    present: bool


@dataclass
class VerifyReport:
    part_scores: List[PartScore]
    attr_pass: float
    failing_parts: List[str] = field(default_factory=list)
    ok: bool = False


def aggregate(
    part_scores: List[PartScore], attr_pass: float, sim_thresh: float, attr_thresh: float
) -> VerifyReport:
    failing = [p.part for p in part_scores if p.sim < sim_thresh]
    ok = (len(failing) == 0) and (attr_pass >= attr_thresh)
    return VerifyReport(part_scores=part_scores, attr_pass=attr_pass, failing_parts=failing, ok=ok)


def _part_phrase(part_name: str) -> str:
    return re.sub(r"_", " ", part_name)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def verify(image: Image.Image, kg: ConceptKG, models, cfg) -> VerifyReport:
    """Locate + score each KG part in `image`, and check each KG attribute
    with the VLM.

    Unloads detector/siglip/dino before the VLM attribute-checking phase
    (they're only used in the part-scoring phase above it): the two phases'
    models together run ~13GB, over this card's actually-available headroom
    under real contention from other researchers' jobs (see refine.py's
    docstring), even though neither phase alone is heavy. Callers still
    control model lifetime *across* the refine loop (i.e. whether verify's
    own models are unloaded before the next generate() call)."""
    part_scores: List[PartScore] = []
    if cfg.use_part_tree:
        scored = [p for p in kg.parts if p.embeddings]  # only parts with a real reference crop
        for part in scored:
            boxes = models.detector.detect(image, _part_phrase(part.name))
            if not boxes:
                part_scores.append(PartScore(part=part.name, sim=0.0, present=False))
                continue
            x0, y0, x1, y1 = boxes[0]
            crop = image.crop((x0, y0, x1, y1))
            sims = []
            if "siglip2" in part.embeddings:
                sims.append(_cosine(models.siglip.embed_image([crop])[0], np.array(part.embeddings["siglip2"])))
            if "dino" in part.embeddings:
                sims.append(_cosine(models.dino.embed_image([crop])[0], np.array(part.embeddings["dino"])))
            part_scores.append(PartScore(part=part.name, sim=float(np.mean(sims)) if sims else 0.0, present=True))
        if scored:
            models.unload("detector"); models.unload("siglip"); models.unload("dino")

    attr_texts = kg.attribute_texts()
    if attr_texts:
        passes = 0
        for attr in attr_texts:
            reply = models.vlm.ask(image, ATTR_QUESTION.format(attr=attr)).strip().lower()
            if reply.startswith("yes"):
                passes += 1
        attr_pass = passes / len(attr_texts)
    else:
        attr_pass = 1.0

    return aggregate(part_scores, attr_pass, cfg.part_sim_threshold, cfg.attr_pass_threshold)
