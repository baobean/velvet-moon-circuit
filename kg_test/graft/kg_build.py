"""M1: build a visually-grounded MMKG for one rare concept from real
reference images.

Every attribute value comes from the VLM *reading* the actual reference
images (source="vision"), not from its parametric memory -- this is the
wedge against RAVEL/Context Canvas (spec section 2), whose KG is
LLM-recalled symbolic text. When `cfg.neutralize_name` is set (the default),
the species name is withheld from the VLM entirely, so the attributes cannot
be name-recalled even in principle (sprint spec section 5).

The anchor/delta VLM pass is dropped (sprint spec section 5): it leaked the
name and fed nothing in scope. `ConceptKG.anchor`/`delta` are still written,
empty, so Phase-1 kg.json files keep deserializing.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Sequence, Tuple

from PIL import Image

from graft.attrfilter import count_generic
from graft.schema import AttributeNode, ConceptKG, PartNode
from graft.selection import medoid_index

PART_NAMES = ["leaf", "bark", "cone_or_flower", "branching"]

# The reply shape both instructions ask for. Plain single braces: it is
# concatenated, never .format()-ed (see _schema_instruction).
_SCHEMA_JSON_SHAPE = (
    '{"global": {"overall_form": "...", "canopy_or_silhouette": "...", '
    '"color_palette": "..."}, "parts": {"leaf": {"shape": "...", '
    '"texture": "..."}, "bark": {"texture": "...", "color": "..."}, '
    '"cone_or_flower": {"appearance": "..."}, "branching": {"pattern": '
    '"..."}}}'
)

# Named arm (cfg.neutralize_name=False): the Phase-1 instruction, kept so the
# named/neutralized delta is a config toggle rather than a code fork.
SCHEMA_INSTRUCTION = (
    "You are looking at reference photographs of a single plant/tree species: "
    "{concept}. Describe ONLY what is visibly present in these images -- do "
    "not rely on prior knowledge of the species. Reply with ONLY a JSON "
    "object of this exact shape, filling every field with what you observe "
    "(use \"not visible\" if a part is not shown):\n"
) + _SCHEMA_JSON_SHAPE

SCHEMA_INSTRUCTION_NEUTRAL = (
    "You are looking at reference photographs of a single plant species. "
    "Describe ONLY what is visibly present in these images. Be DISCRIMINATIVE: "
    "give the specific color, shape, texture, and count details that distinguish "
    "this plant from a generic tree. Reply with ONLY a JSON object of this exact "
    "shape (use \"not visible\" if a part is not shown):\n"
) + _SCHEMA_JSON_SHAPE

REASK_SUFFIX = (
    " Your previous answer was too generic. Replace any vague value (e.g. 'green', "
    "'full', 'tree') with a concrete, distinguishing observation from the images."
)


def should_reask(values: Sequence[str], threshold: float = 0.5) -> bool:
    """True when at least `threshold` of the attribute values are generic, i.e.
    the description carries too little discriminative signal to keep."""
    return bool(values) and (count_generic(values) / len(values)) >= threshold


def _schema_instruction(concept: str, neutralize: bool) -> str:
    if neutralize:
        return SCHEMA_INSTRUCTION_NEUTRAL
    # .replace, not .format: the JSON shape carries literal single braces.
    return SCHEMA_INSTRUCTION.replace("{concept}", concept)


def _extract_json(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object found in VLM reply: {raw!r}")
    return json.loads(raw[start : end + 1])


def parse_vlm_schema(raw: str) -> Tuple[List[AttributeNode], Dict[str, List[AttributeNode]]]:
    """Parse the schema-instruction reply into (global_attrs, part_attrs)."""
    data = _extract_json(raw)
    global_attrs = [
        AttributeNode(name=k, value=str(v), source="vision")
        for k, v in (data.get("global") or {}).items()
    ]
    part_attrs: Dict[str, List[AttributeNode]] = {}
    for part_name, attrs in (data.get("parts") or {}).items():
        part_attrs[part_name] = [
            AttributeNode(name=k, value=str(v), source="vision") for k, v in attrs.items()
        ]
    return global_attrs, part_attrs


def _part_phrase(part_name: str) -> str:
    return re.sub(r"_", " ", part_name)


def _renorm(v) -> List[float]:
    import numpy as np

    n = float(np.linalg.norm(v))
    return (v / n).tolist() if n > 0 else v.tolist()


_NOT_SHOWN = ("not visible", "none", "n/a")


def _attr_values(
    global_attrs: List[AttributeNode], part_attrs_by_name: Dict[str, List[AttributeNode]]
) -> List[str]:
    """The attribute values the re-ask gate should judge."""
    vals = [a.value for a in global_attrs] + [
        a.value for attrs in part_attrs_by_name.values() for a in attrs
    ]
    # A sanctioned "not visible" is honesty, not vagueness -- both instructions
    # ask for it by name. Counting it as generic would make an image-poor
    # species trip the re-ask on honesty alone, and REASK_SUFFIX would then
    # pressure the VLM to invent attributes for parts it cannot see, which is
    # exactly the LLM-recall failure this sprint exists to kill.
    return [v for v in vals if v.strip().lower() not in _NOT_SHOWN]


def build_kg(concept: str, build_refs: List[str], models, cfg, outputs_dir: str | None = None) -> ConceptKG:
    """Build and persist a ConceptKG for `concept` from its `build_refs`
    (real reference image paths). Releases the VLM before returning so the
    next pipeline stage can claim the GPU (spec's single-GPU-sequential
    constraint)."""
    outputs_dir = outputs_dir or os.path.join(cfg.outputs_dir, concept)
    crops_dir = os.path.join(outputs_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    ref_images = [Image.open(p).convert("RGB") for p in build_refs]

    instruction = _schema_instruction(concept, cfg.neutralize_name)
    raw = models.vlm.describe(ref_images, instruction)
    global_attrs, part_attrs_by_name = parse_vlm_schema(raw)

    # One-time re-ask when the description came back mostly generic
    # ("green", "full", "tree"): with the name withheld the VLM has to earn
    # the attributes from the pixels, and a vague first pass is recoverable.
    if should_reask(_attr_values(global_attrs, part_attrs_by_name)):
        try:
            raw = models.vlm.describe(ref_images, instruction + REASK_SUFFIX)
            global_attrs, part_attrs_by_name = parse_vlm_schema(raw)
        except ValueError:
            # A valid first pass is already in hand; an unparseable retry must
            # not fail a multi-GPU-minute build inside an unattended run.
            sys.stderr.write(
                f"[kg_build] {concept}: re-ask reply unparseable -- keeping first-pass attributes\n"
            )

    # anchor/delta dropped (sprint spec section 5): no second, name-leaking
    # VLM pass. The fields stay on ConceptKG, empty, for back-compat.
    anchor, delta = "", []
    # vlm (~5GB nf4) and detector+siglip+dino (~3GB together) are only ever
    # used in their own phase below -- unloading each phase before the next
    # keeps this function's peak GPU usage to a single phase's models (~5GB)
    # instead of holding everything at once, which matters on this card's
    # actual, observed contention from other researchers' jobs (see
    # refine.py's docstring).
    models.unload("vlm")

    # Per-ref SigLIP2 embeddings, aligned to build_refs order; they carry the
    # concept mean and the medoid (the name-free exemplar, spec section 6).
    ref_siglip = models.siglip.embed_image(ref_images)  # (N, D), L2-normed
    ref_embeddings = [v.tolist() for v in ref_siglip]
    crop_source = ref_images[medoid_index(ref_siglip)]

    parts: List[PartNode] = []
    for part_name in PART_NAMES:
        attrs = part_attrs_by_name.get(part_name, [])
        boxes = models.detector.detect(crop_source, _part_phrase(part_name))
        if boxes:
            x0, y0, x1, y1 = boxes[0]
            crop = crop_source.crop((x0, y0, x1, y1))
            crop_path = os.path.join(crops_dir, f"{part_name}.png")
            crop.save(crop_path)
            embeddings = {
                "siglip2": models.siglip.embed_image([crop])[0].tolist(),
                "dino": models.dino.embed_image([crop])[0].tolist(),
            }
            parts.append(PartNode(part_name, attrs, crop_path, embeddings))
        else:
            # No reliable box in the reference -> the part is unscoreable.
            # Embedding the whole reference instead would make verify compare
            # a whole image against a crop (spec section 8.1(b)); empty
            # embeddings exclude the part from the gate instead. Section
            # 8.1(b) also asks for the exclusion to be logged, not silent --
            # the audit needs the box hit-rate.
            sys.stderr.write(
                f"[kg_build] {concept}: no box for part '{part_name}' in medoid ref "
                f"-- part unscoreable\n"
            )
            parts.append(PartNode(part_name, attrs, None, {}))

    concept_embeddings = {
        "siglip2": _renorm(ref_siglip.mean(axis=0)),
        "dino": _renorm(models.dino.embed_image(ref_images).mean(axis=0)),
    }

    models.unload("detector")
    models.unload("siglip")
    models.unload("dino")
    # The reranker is no longer used here: medoid selection (spec section 6)
    # replaces text-keyed exemplar ranking, so ref_paths keeps its natural
    # order and ref_embeddings stays aligned to it.

    kg = ConceptKG(
        concept=concept,
        global_attrs=global_attrs,
        parts=parts,
        concept_embeddings=concept_embeddings,
        anchor=anchor,
        delta=delta,
        ref_paths=list(build_refs),
        ref_embeddings=ref_embeddings,
    )
    kg.to_json(os.path.join(outputs_dir, "kg.json"))
    return kg
