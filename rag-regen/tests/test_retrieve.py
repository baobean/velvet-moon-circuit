from pathlib import Path

import numpy as np
import pytest

from ragregen import retrieve


class FakeEncoder:
    """Maps a filename stem to a fixed unit vector; text maps by keyword."""

    dim = 4
    VECS = {
        "fox":    np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"),
        "panda":  np.array([0.0, 1.0, 0.0, 0.0], dtype="float32"),
        "durian": np.array([0.0, 0.0, 1.0, 0.0], dtype="float32"),
    }

    def encode_images(self, paths, batch_size=32):
        return np.stack([self.VECS[Path(p).stem] for p in paths])

    def encode_text(self, texts):
        out = []
        for t in texts:
            for key, vec in self.VECS.items():
                if key in t.lower():
                    out.append(vec)
                    break
            else:
                out.append(np.zeros(self.dim, dtype="float32"))
        return np.stack(out)


PATHS = [Path("/c/fox.jpg"), Path("/c/panda.jpg"), Path("/c/durian.jpg")]


def test_reference_query_disambiguates_with_visual_medium_and_coarse_type():
    assert retrieve.reference_query("love-in-a-mist", "flower") == (
        "a real photograph of love-in-a-mist, a type of flower, as the main "
        "subject")
    assert "real photograph" in retrieve.reference_query("axolotl")


def test_build_index_returns_the_count(tmp_path):
    n = retrieve.build_index(PATHS, FakeEncoder(), tmp_path / "i.faiss")
    assert n == 3
    assert (tmp_path / "i.faiss").is_file()
    assert (tmp_path / "i.faiss.paths.json").is_file()


def test_search_returns_the_matching_image_first(tmp_path):
    retrieve.build_index(PATHS, FakeEncoder(), tmp_path / "i.faiss")
    r = retrieve.Retriever.from_index(tmp_path / "i.faiss", FakeEncoder())
    hits = r.search("a fox in the snow", k=3)
    assert hits[0].path.stem == "fox"
    assert hits[0].rank == 0


def test_search_k_caps_the_result_count(tmp_path):
    retrieve.build_index(PATHS, FakeEncoder(), tmp_path / "i.faiss")
    r = retrieve.Retriever.from_index(tmp_path / "i.faiss", FakeEncoder())
    assert len(r.search("a panda", k=2)) == 2


def test_free_drops_the_encoder_even_without_its_own_free(tmp_path):
    """Without this, stage_with_model's getattr-guarded cleanup silently
    no-ops on a bare Retriever, and the encoder stays GPU-resident through
    the next stage's VRAM preflight."""
    retrieve.build_index(PATHS, FakeEncoder(), tmp_path / "i.faiss")
    r = retrieve.Retriever.from_index(tmp_path / "i.faiss", FakeEncoder())
    r.free()
    assert r.encoder is None


def test_free_calls_the_encoder_s_own_free_when_present(tmp_path):
    class FreeableEncoder(FakeEncoder):
        def __init__(self):
            self.freed = False

        def free(self):
            self.freed = True

    retrieve.build_index(PATHS, FakeEncoder(), tmp_path / "i.faiss")
    enc = FreeableEncoder()
    r = retrieve.Retriever.from_index(tmp_path / "i.faiss", enc)
    r.free()
    assert enc.freed is True
    assert r.encoder is None


def test_ranks_are_consecutive_from_zero(tmp_path):
    retrieve.build_index(PATHS, FakeEncoder(), tmp_path / "i.faiss")
    r = retrieve.Retriever.from_index(tmp_path / "i.faiss", FakeEncoder())
    assert [h.rank for h in r.search("a durian", k=3)] == [0, 1, 2]


def test_build_index_rejects_an_empty_corpus(tmp_path):
    with pytest.raises(ValueError, match="no images"):
        retrieve.build_index([], FakeEncoder(), tmp_path / "i.faiss")


def test_a_query_matching_nothing_scores_without_nan(tmp_path):
    """An unmatched query embeds to the zero vector. Without the zero-norm
    guard this divides by zero and every score comes back NaN, which ranks
    arbitrarily instead of failing loudly."""
    retrieve.build_index(PATHS, FakeEncoder(), tmp_path / "i.faiss")
    r = retrieve.Retriever.from_index(tmp_path / "i.faiss", FakeEncoder())
    hits = r.search("a spaceship", k=3)
    assert len(hits) == 3
    assert all(not np.isnan(h.score) for h in hits)


def test_k_larger_than_the_corpus_is_capped(tmp_path):
    retrieve.build_index(PATHS, FakeEncoder(), tmp_path / "i.faiss")
    r = retrieve.Retriever.from_index(tmp_path / "i.faiss", FakeEncoder())
    hits = r.search("a fox", k=99)
    assert len(hits) == 3
    assert [h.rank for h in hits] == [0, 1, 2]


def test_an_index_without_its_sidecar_names_the_rebuild_command(tmp_path):
    retrieve.build_index(PATHS, FakeEncoder(), tmp_path / "i.faiss")
    (tmp_path / "i.faiss.paths.json").unlink()
    with pytest.raises(FileNotFoundError, match="build-index"):
        retrieve.Retriever.from_index(tmp_path / "i.faiss", FakeEncoder())
