"""Text-to-image retrieval over a FAISS IndexFlatIP.

Flat/exact on purpose. At 500k images a 1152-dim fp32 index is ~2.3 GB and
brute force stays fast, so retrieval quality is never confounded by index
approximation (spec §9 R6). Do not swap in IVF or HNSW without a measured
reason to.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class TextImageEncoder(Protocol):
    dim: int

    def encode_images(self, paths, batch_size: int = 32) -> np.ndarray: ...
    def encode_text(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class Hit:
    path: Path
    score: float
    rank: int


def reference_query(concept: str, coarse: str | None = None) -> str:
    """Disambiguate a name into the kind of visual the editor can consume.

    Bare rare names are often non-visual or polysemous in web corpora:
    ``love-in-a-mist`` retrieved people in romantic flower scenes and
    ``axolotl`` retrieved crochet toys. The coarse type and photograph prior
    constrain retrieval without importing the target prompt's composition.
    """
    concept = concept.strip()
    coarse = (coarse or "").strip()
    if coarse and coarse.casefold() != concept.casefold():
        return (f"a real photograph of {concept}, a type of {coarse}, "
                f"as the main subject")
    return f"a real photograph of {concept} as the main subject"


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype="float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def build_index(image_paths, encoder: TextImageEncoder, out_path: Path,
                batch_size: int = 32) -> int:
    """Embed every image and write an exact inner-product index."""
    import faiss

    paths = [Path(p) for p in image_paths]
    if not paths:
        raise ValueError("no images to index -- check images_root in retrieval_db.yaml")

    mat = _l2_normalise(encoder.encode_images(paths, batch_size=batch_size))
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))
    out_path.with_suffix(out_path.suffix + ".paths.json").write_text(
        json.dumps([str(p) for p in paths], indent=2))
    return len(paths)


class Retriever:
    def __init__(self, index, paths: list[Path], encoder: TextImageEncoder):
        self.index = index
        self.paths = paths
        self.encoder = encoder

    @classmethod
    def from_index(cls, index_path: Path, encoder: TextImageEncoder) -> "Retriever":
        import faiss

        index_path = Path(index_path)
        index = faiss.read_index(str(index_path))
        meta = index_path.with_suffix(index_path.suffix + ".paths.json")
        if not meta.is_file():
            raise FileNotFoundError(
                f"index at {index_path} has no sidecar {meta.name}. "
                f"Rebuild with `./scripts/run.sh build-index`.")
        paths = [Path(p) for p in json.loads(meta.read_text())]
        return cls(index, paths, encoder)

    @property
    def dim(self) -> int:
        return self.index.d

    def search(self, query: str, k: int) -> list[Hit]:
        vec = _l2_normalise(self.encoder.encode_text([query]))
        k = min(k, len(self.paths))
        scores, idxs = self.index.search(vec, k)
        return [Hit(path=self.paths[int(i)], score=float(s), rank=rank)
                for rank, (s, i) in enumerate(zip(scores[0], idxs[0])) if i >= 0]

    def free(self) -> None:
        """Drop the encoder and return its VRAM. The FAISS index is CPU-only.

        Without this, `stage_with_model`'s cleanup finds no `.free` on a bare
        `Retriever` and silently does nothing -- the encoder stays GPU-resident
        through every later stage's preflight check.
        """
        enc_free = getattr(self.encoder, "free", None)
        if callable(enc_free):
            enc_free()
        self.encoder = None
