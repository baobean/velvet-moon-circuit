"""Pure helpers for the retrieved-reference reranker artifact."""
import pytest

from ragregen.retrieved_reference import (
    build_case_record, contamination, preserve_text)


def _oracle_case():
    return {"truth": "FAIL", "concept": "okapi", "coarse": "giraffe",
            "scores": {"draft": {"text_relevance": 0.3203125,
                                 "reference_relevance": 0.081}}}


def test_text_relevance_is_preserved_bit_for_bit():
    assert preserve_text(_oracle_case()) == 0.3203125


def test_preserve_text_raises_when_draft_text_missing():
    with pytest.raises(ValueError):
        preserve_text({"scores": {"draft": {}}})


def test_contamination_true_only_on_exact_hash_overlap(tmp_path):
    ref = tmp_path / "ref.jpg"
    ref.write_bytes(b"the-reference-bytes")
    same = tmp_path / "retrieved_same.jpg"
    same.write_bytes(b"the-reference-bytes")
    other = tmp_path / "retrieved_other.jpg"
    other.write_bytes(b"different")

    assert contamination(same, [ref]) is True
    assert contamination(other, [ref]) is False


def test_case_record_preserves_text_and_swaps_reference(tmp_path):
    img = tmp_path / "hit.jpg"
    img.write_bytes(b"hit-bytes")
    hit = {"query": "a real photograph of okapi", "path": str(img),
           "score": 0.42, "rank": 0}

    rec = build_case_record(_oracle_case(), retrieved_reference_relevance=0.19,
                            hit=hit, image_sha256="abc", contaminated=False)

    assert rec["truth"] == "FAIL"
    assert rec["scores"]["draft"]["text_relevance"] == 0.3203125  # preserved
    assert rec["scores"]["draft"]["reference_relevance"] == 0.19   # retrieved
    assert rec["retrieval"]["path"] == str(img)
    assert rec["retrieval"]["rank"] == 0
    assert rec["retrieval"]["image_sha256"] == "abc"
    assert rec["contaminated"] is False
