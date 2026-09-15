"""Which LAION rows become the retrieval corpus.

Two jobs, sized separately (doc 5 §4). The per-concept slice guarantees
retrieval has something to find -- in random web images a given ImageNet class
appears at roughly 1-in-10^4, so a random 100k yields single digits for a
common concept. The random bulk is distractor mass: a corpus of only the
seeded concepts would make retrieval trivially correct and measure nothing.

Pure functions over caption strings. Parquet and network live in
scripts/fetch_corpus.py so this stays testable without either.
"""
from __future__ import annotations

import re

import numpy as np


def caption_matches(caption: str, concept: str) -> bool:
    """Whether `caption` mentions `concept` as a whole word or phrase.

    Word-bounded on purpose: a plain substring test puts "foxglove" in the
    fox slice, and retrieval then looks broken for a reason that is ours
    rather than the method's.
    """
    pattern = r"\b" + r"\s+".join(
        re.escape(w) for w in concept.lower().split()) + r"\b"
    return re.search(pattern, caption.lower()) is not None


def count_per_concept(captions, concepts) -> dict[str, int]:
    """How many captions mention each concept. Published in the manifest."""
    caps = list(captions)
    return {c: sum(1 for cap in caps if caption_matches(cap, c))
            for c in concepts}


def select_rows(captions, concepts, *, n_random: int, n_per_concept: int,
                seed: int) -> list[int]:
    """Row indices to download: the seeded slice, plus random bulk.

    Deterministic for a seed, so a failed download resumes against the same
    sample instead of silently re-rolling the corpus.
    """
    caps = list(captions)
    rng = np.random.default_rng(seed)
    keep: set[int] = set()

    for concept in concepts:
        hits = [i for i, cap in enumerate(caps)
                if caption_matches(cap, concept)]
        #: Fewer hits than asked for is a finding, not an error -- raising
        #: would abort a 100k download over one rare word.
        take = min(n_per_concept, len(hits))
        if take:
            keep.update(int(i) for i in rng.choice(hits, size=take,
                                                   replace=False))

    if n_random:
        bulk = rng.choice(len(caps), size=min(n_random, len(caps)),
                          replace=False)
        keep.update(int(i) for i in bulk)

    return sorted(keep)
