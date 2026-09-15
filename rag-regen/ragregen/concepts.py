"""Prompt -> the concept phrases a verifier should check.

Deliberately rule-based and model-free. This runs before any model loads, it
must be deterministic for reproducible runs, and its output is the contract
between the prompt and both verification streams.

Counts and spatial relations get their own kinds because Stream A cannot check
them -- GroundingDINO boxes objects, not quantities or arrangements. Fusion
routes those to Stream B (spec §2).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

KINDS = ("subject", "count", "relation")

_NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "a pair of", "several", "many", "a few",
}

_RELATION_WORDS = {
    "behind", "in front of", "next to", "beside", "above", "below", "under",
    "on top of", "between", "inside", "outside", "left of", "right of",
    "beneath", "near", "against",
}


@dataclass(frozen=True)
class Concept:
    phrase: str
    kind: str
    #: Superordinate category for the target concept, e.g. "parrot" for
    #: "African grey parrot". None on non-target concepts and whenever the
    #: caller supplies no coarse term. Stream A skips the fine-grained
    #: margin test when this is None, which is what keeps the pre-contrastive
    #: behaviour intact for every concept the operator did not annotate.
    coarse: str | None = None


def _found(prompt_lc: str, needles: set[str]) -> list[str]:
    hits = []
    for n in needles:
        if re.search(rf"(?<![a-z]){re.escape(n)}(?![a-z])", prompt_lc):
            hits.append(n)
    return sorted(hits)


def parse(prompt: str, target: str, coarse: str | None = None) -> list[Concept]:
    """Return the concepts to verify, target first.

    >>> [c.kind for c in parse("three foxes behind a tree", target="fox")][0]
    'subject'
    """
    prompt_lc = prompt.lower()
    out: list[Concept] = [Concept(target, "subject", coarse)]
    seen = {target.lower()}

    for word in _found(prompt_lc, _NUMBER_WORDS):
        if word not in seen:
            out.append(Concept(word, "count"))
            seen.add(word)

    if re.search(r"(?<!\w)\d+(?!\w)", prompt_lc):
        digits = re.findall(r"(?<!\w)\d+(?!\w)", prompt_lc)
        for d in digits:
            if d not in seen:
                out.append(Concept(d, "count"))
                seen.add(d)

    for word in _found(prompt_lc, _RELATION_WORDS):
        if word not in seen:
            out.append(Concept(word, "relation"))
            seen.add(word)

    return out
