"""Turn a ConceptKG (or a flat attribute list, for the B2/RAVEL-style
baseline) into an SDXL text prompt."""
from __future__ import annotations

from typing import List, Sequence

from graft.schema import ConceptKG

_TEMPLATE = "a photo of a {concept}, {attrs}"
NEUTRAL_TOKEN = "plant"


def concept_token(concept: str, neutralize: bool) -> str:
    """Return the concept token for use in a prompt.
    If neutralize is True, return NEUTRAL_TOKEN; otherwise return concept."""
    return NEUTRAL_TOKEN if neutralize else concept


def compose_prompt(kg: ConceptKG, max_attrs: int = 8, neutralize: bool = True) -> str:
    """KG-structured prompt: concept + vision-sourced attribute clauses
    (attribute_texts() already orders vision-sourced values first)."""
    attrs = kg.attribute_texts()[:max_attrs]
    return _TEMPLATE.format(concept=concept_token(kg.concept, neutralize), attrs=", ".join(attrs))


def compose_ravel_prompt(concept: str, llm_attrs: Sequence[str], neutralize: bool = True) -> str:
    """B2 baseline form: same template, but over attributes supplied
    directly (e.g. from the VLM's parametric memory, with no visual
    grounding) -- reproduces RAVEL/Context Canvas's prompt-only steering."""
    return _TEMPLATE.format(concept=concept_token(concept, neutralize), attrs=", ".join(llm_attrs))
