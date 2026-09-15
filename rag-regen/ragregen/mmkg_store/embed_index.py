"""MMKG species store embedding index — FAISS IndexFlatIP + sidecar.

Config is frozen (``LOCKED``): every index built by this module is a flat
inner-product FAISS index over L2-normalized SigLIP so400m/384 embeddings
(dim 1152). ``load_index`` refuses to open a sidecar whose recorded config
does not match ``LOCKED`` byte-for-byte -- an index built under a different
encoder/metric must never be silently reused.

``IndexBuilder`` itself is dim-agnostic: the dimension is fixed by the first
``add`` call and asserted on every later one. The real dim-1152 enforcement
happens at live build time (Task 8), not here -- this module only guarantees
internal consistency of whatever vectors it is given.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

LOCKED = {
    "encoder": "siglip_so400m_384",
    "dim": 1152,
    "normalized": True,
    "metric": "ip",
    "index": "IndexFlatIP",
}


def _l2_normalize(vec) -> np.ndarray:
    """Return ``vec`` as a float32 L2-normalized vector.

    Guards against a zero-norm vector by dividing by 1 instead of 0 (the
    vector is returned unchanged, still all-zero, rather than raising or
    producing NaNs).
    """
    arr = np.asarray(vec, dtype="float32")
    norm = float(np.linalg.norm(arr))
    return arr / (norm if norm > 0 else 1.0)


class IndexBuilder:
    """Accumulates L2-normalized embeddings into a flat IP FAISS index.

    Dim-agnostic: the first ``add`` fixes the dimension; every later ``add``
    must match it.
    """

    def __init__(self):
        self._dim: int | None = None
        self._vectors: list[np.ndarray] = []
        self._entries: dict[int, dict] = {}

    def add(self, vec, meta: dict) -> int:
        """L2-normalize ``vec``, append it, and store ``meta`` at its ref.

        Returns the integer ``embedding_ref`` (the vector's index in the
        eventual FAISS index) that keys ``meta`` in the sidecar's
        ``entries``.
        """
        normed = _l2_normalize(vec)
        if self._dim is None:
            self._dim = normed.shape[0]
        elif normed.shape[0] != self._dim:
            raise ValueError(
                f"embedding dim {normed.shape[0]} != established dim {self._dim}")

        ref = len(self._vectors)
        self._vectors.append(normed)
        self._entries[ref] = dict(meta)
        return ref

    def save(self, out_dir) -> None:
        """Write ``index.faiss`` and ``sidecar.json`` under ``out_dir``."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        dim = self._dim if self._dim is not None else 0
        index = faiss.IndexFlatIP(dim)
        if self._vectors:
            index.add(np.stack(self._vectors).astype("float32"))
        faiss.write_index(index, str(out_dir / "index.faiss"))

        sidecar = {"config": LOCKED, "entries": self._entries}
        (out_dir / "sidecar.json").write_text(json.dumps(sidecar))


def load_index(dir):
    """Load ``index.faiss`` + ``sidecar.json`` from ``dir``.

    Raises ``ValueError`` if the sidecar's recorded config does not exactly
    match ``LOCKED``.
    """
    dir = Path(dir)
    sidecar = json.loads((dir / "sidecar.json").read_text())
    if sidecar["config"] != LOCKED:
        raise ValueError(
            f"index config mismatch: expected {LOCKED}, got {sidecar['config']}")

    index = faiss.read_index(str(dir / "index.faiss"))
    return index, sidecar


def search(index, sidecar, vec, k, kind=None, allow=None):
    """Search ``index`` for the ``k`` nearest neighbors of ``vec`` by IP.

    ``vec`` is L2-normalized before searching. Results are filtered
    post-hoc by ``kind`` (exact match on ``meta["kind"]``) and/or ``allow``
    (a set of allowed ``meta["global_id"]`` values). Returns a list of
    ``(embedding_ref, score, meta)`` tuples, best match first.
    """
    query = _l2_normalize(vec).reshape(1, -1)
    n = index.ntotal
    if n == 0:
        return []

    search_k = min(max(k, 1) * 4 if (kind is not None or allow is not None) else k, n)
    scores, refs = index.search(query, search_k)

    entries = sidecar["entries"]
    results = []
    for ref, score in zip(refs[0], scores[0]):
        if ref < 0:
            continue
        meta = entries[str(ref)] if str(ref) in entries else entries.get(ref)
        if meta is None:
            continue
        if kind is not None and meta.get("kind") != kind:
            continue
        if allow is not None and meta.get("global_id") not in allow:
            continue
        results.append((int(ref), float(score), meta))
        if len(results) >= k:
            break

    return results
