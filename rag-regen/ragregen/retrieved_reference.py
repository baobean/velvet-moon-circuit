"""Pure helpers for building the retrieved-reference reranker artifact.

Keeps the GPU script thin and testable. Two invariants matter: ``text_relevance``
is copied bit-for-bit from the oracle artifact (it is scored against a text-only
query, so it is reference-independent and must not drift), and contamination is an
*exact* SHA-256 overlap between a retrieved image and any of the case's
ground-truth references -- detected, never silently worked around.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ragregen.eval_manifest import sha256_file


def preserve_text(oracle_case: Mapping) -> float:
    """The oracle draft's text_relevance, unchanged. Raises if absent."""
    try:
        return oracle_case["scores"]["draft"]["text_relevance"]
    except (KeyError, TypeError) as exc:
        raise ValueError("oracle case has no draft text_relevance") from exc


def contamination(retrieved_path: Path, gt_ref_paths: Sequence[Path]) -> bool:
    """True iff the retrieved image is byte-identical to any ground-truth ref."""
    retrieved_hash = sha256_file(Path(retrieved_path))
    for ref in gt_ref_paths:
        ref = Path(ref)
        if ref.is_file() and sha256_file(ref) == retrieved_hash:
            return True
    return False


def build_case_record(oracle_case: Mapping, *,
                      retrieved_reference_relevance: float, hit: Mapping,
                      image_sha256: str, contaminated: bool) -> dict:
    """Assemble one retrieved-reference case record.

    Preserves the oracle text score, substitutes the retrieved reference score,
    and records full retrieval provenance.
    """
    text = preserve_text(oracle_case)
    return {
        "truth": oracle_case["truth"],
        "concept": oracle_case.get("concept"),
        "coarse": oracle_case.get("coarse"),
        "scores": {"draft": {
            "text_relevance": text,
            "reference_relevance": float(retrieved_reference_relevance)}},
        "retrieval": {
            "query": hit.get("query"),
            "path": hit.get("path"),
            "score": hit.get("score"),
            "rank": hit.get("rank"),
            "image_sha256": image_sha256,
        },
        "contaminated": bool(contaminated),
    }
