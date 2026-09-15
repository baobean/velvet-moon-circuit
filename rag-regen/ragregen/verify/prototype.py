"""Image-side prototypes for the fine-grained contrast.

Phrase-level contrast failed because SigLIP's absolute score for a (crop, text)
pair is dominated by how caption-like the phrase is -- "Boston bull" 0.780 vs
"dog" 0.017 on the same crop -- so the margin measured the phrase pair, not the
image (findings/2026-07-27-finegrained-result.md §3a). Comparing the crop
against reference IMAGES removes the text side of that asymmetry entirely.

Ported from `../ImageRAG/scripts/finegrained_seg.py:70-105`, whose
nearest-prototype head measured 0.917 on breed-level box selection. The
arithmetic there is load-bearing and reproduced exactly: normalise each
reference, mean, renormalise.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class ImageEmbedder(Protocol):
    def encode_pil(self, images, batch_size: int = 32) -> np.ndarray: ...


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    # A zero vector has no direction; returning it unchanged keeps cosine at
    # 0.0 rather than producing nan and poisoning every downstream comparison.
    return v if n == 0.0 else v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(np.asarray(a, dtype=np.float64)),
                        _unit(np.asarray(b, dtype=np.float64))))


def build_prototype(images, embedder: ImageEmbedder) -> np.ndarray:
    """One unit-norm prototype from a set of reference images.

    Each reference is normalised BEFORE the mean. Skipping that lets a single
    high-magnitude embedding dominate the prototype, which is a silent quality
    failure rather than a crash.
    """
    images = list(images)
    if not images:
        raise ValueError("a prototype needs at least one reference image")
    feats = np.asarray(embedder.encode_pil(images), dtype=np.float64)
    stacked = np.stack([_unit(row) for row in feats])
    return _unit(stacked.mean(axis=0))


class PrototypeBank:
    """Phrase -> unit-norm prototype vector.

    `get` returns None for an unknown phrase rather than raising: a dataset may
    legitimately carry no references for some coarse term, and that concept
    must fall through to PRESENT rather than fail the case.
    """

    def __init__(self, vectors: dict[str, np.ndarray]):
        self._v = {k: _unit(np.asarray(v, dtype=np.float64))
                   for k, v in vectors.items()}

    def __contains__(self, phrase: str) -> bool:
        return phrase in self._v

    @property
    def phrases(self) -> list[str]:
        return sorted(self._v)

    def get(self, phrase: str) -> np.ndarray | None:
        return self._v.get(phrase)

    def score(self, crop_vec: np.ndarray, phrase: str) -> float | None:
        proto = self._v.get(phrase)
        return None if proto is None else cosine(crop_vec, proto)


#: The two prototype sources. "ceiling" uses ground-truth references and is a
#: diagnostic upper bound -- it is NEVER reportable as a verifier result,
#: because gt_refs are also the target of the DINO identity metric. "retrieved"
#: is the only configuration that can ship.
ARMS = ("ceiling", "retrieved")


def _open(path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def _maybe_build(paths, embedder, phrase=None, prepare=None) -> np.ndarray | None:
    paths = list(paths)
    if not paths:
        return None
    images = [_open(p) for p in paths]
    if prepare is not None:
        # A reference the preparer cannot handle still has to contribute
        # something; dropping it silently would shrink the prototype without
        # saying so.
        images = [prepare(im, phrase) or im for im in images]
    return build_prototype(images, embedder)


def bank_from_gt_refs(dataset, embedder: ImageEmbedder, exclude=(),
                      prepare=None) -> PrototypeBank:
    """Ceiling arm: fine prototypes from gt_refs, coarse from coarse_refs.

    `exclude` drops specific reference paths, so a case's own image can never
    enter its own prototype (finegrained_seg.py:85). A term whose references
    are all excluded is omitted entirely rather than built from nothing.

    `prepare(image, phrase) -> image | None` runs on each reference before it
    is embedded. Prototypes built from whole photographs encode framing and
    background -- "a photograph of a durian" rather than "a durian" -- so
    crops that look like photographs outscore tight object crops whatever they
    contain (findings/2026-07-28-masking-result.md). Passing
    `mask.crop_to_object` here is what makes an object prototype, and keeping
    it a callable is what keeps this module free of a masker dependency.
    """
    excluded = {str(p) for p in exclude}
    vectors: dict[str, np.ndarray] = {}
    for case in dataset.cases:
        v = _maybe_build([p for p in case.gt_refs if str(p) not in excluded],
                         embedder, case.concept, prepare)
        if v is not None:
            vectors[case.concept] = v
    for term, refs in dataset.coarse_refs.items():
        v = _maybe_build([p for p in refs if str(p) not in excluded],
                         embedder, term, prepare)
        if v is not None:
            vectors[term] = v
    return PrototypeBank(vectors)


class CropClassifier:
    """Adapts a PrototypeBank to the crop-scoring interface select_box wants.

    The bank compares vectors; box selection has crops. Embedding is this
    adapter's job, which keeps prototype.py a pure vector module and lets
    mask.py stay duck-typed against `.score(crop, phrase)`.
    """

    def __init__(self, bank: PrototypeBank, embedder: ImageEmbedder):
        self.bank = bank
        self.embedder = embedder

    def score(self, crop, phrase: str) -> float | None:
        # Checked before embedding: an unknown phrase cannot be scored, and a
        # forward pass to discover that is wasted work on a loaded encoder.
        if self.bank.get(phrase) is None:
            return None
        return self.bank.score(self.embedder.encode_pil([crop])[0], phrase)


def bank_from_retrieval(dataset, retriever, embedder: ImageEmbedder,
                        k: int = 8) -> PrototypeBank:
    """Retrieved arm: both sides come from the operator's corpus.

    Deliberately reads only `concept` and `coarse` from each case -- never
    `gt_refs`, which would leak the answer key into the shipping path.
    """
    vectors: dict[str, np.ndarray] = {}
    from ragregen.retrieve import reference_query

    for case in dataset.cases:
        for phrase in (case.concept, case.coarse):
            if phrase in vectors:
                continue
            coarse = case.coarse if phrase == case.concept else None
            query = reference_query(phrase, coarse)
            v = _maybe_build([h.path for h in retriever.search(query, k)],
                             embedder)
            if v is not None:
                vectors[phrase] = v
    return PrototypeBank(vectors)
