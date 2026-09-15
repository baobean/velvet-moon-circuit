import pytest

from graft.generate import select_exemplar
from graft.schema import ConceptKG


def test_select_exemplar_returns_medoid_path():
    # refs a,b cluster; c is an outlier -> medoid is a or b
    kg = ConceptKG("x", [], [], {}, "", [], ["a.jpg", "b.jpg", "c.jpg"],
                   ref_embeddings=[[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]])
    assert select_exemplar(kg) in ("a.jpg", "b.jpg")


def test_select_exemplar_rejects_missing_ref_embeddings():
    """A Phase-1 kg.json (no ref_embeddings) must fail loudly, not crash deep
    inside numpy -- an empty-matrix matmul used to surface as a refine-exhausted
    'likely repeated OOM' and silently dropped the whole GRAFT row."""
    stale = ConceptKG("Avocado", [], [], {}, "", [], ["a.jpg", "b.jpg"])
    with pytest.raises(ValueError, match="ref_embeddings"):
        select_exemplar(stale)


def test_select_exemplar_rejects_misaligned_ref_embeddings():
    kg = ConceptKG("Avocado", [], [], {}, "", [], ["a.jpg", "b.jpg"],
                   ref_embeddings=[[1.0, 0.0]])
    with pytest.raises(ValueError, match="ref_embeddings"):
        select_exemplar(kg)
