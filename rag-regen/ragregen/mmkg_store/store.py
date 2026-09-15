"""MMKG species store -- persistence, inverted index, and filterable query API.

Records (§4 shape, produced by ``mmkg_store.schema.build_record``) are persisted
one JSON file per species under ``store/<dataset>/<species_key>.json``. Alongside
them, ``write_store`` builds a single ``inverted.json`` reverse index with three
sections:

- ``attributes``: ``"<slot>|<value>" -> [global_id, ...]``
- ``hubs``: ``hub_id -> [global_id, ...]`` (from each record's ``instance_of``
  relations -- namespaced genus/family hubs)
- ``parts``: ``part_type -> [embedding_ref, ...]`` (one entry per part crop)

The FAISS index (medoids/candidates/part-crop embeddings) is a separate,
optional artifact: ``write_store`` saves it only when given an
``embed_index.IndexBuilder``, and pure JSON/query-API tests never need faiss.

``Store`` is the read-side: ``get``/``query``/``members``/``provenance`` work
purely off the persisted JSON + inverted index (no FAISS required); only
``nearest_crops`` touches the FAISS index (via ``embed_index.search``), and is
a no-op (empty list) when no index was persisted.
"""
from __future__ import annotations

import json
from pathlib import Path

from ragregen.mmkg_store import embed_index as ei


def _dataset_and_species_key(record: dict) -> tuple[str, str]:
    """Split a record's ``global_id`` (``"<dataset>:<species_key>"``) in two."""
    dataset, _, species_key = record["global_id"].partition(":")
    return dataset, species_key


def write_store(records: list[dict], out_dir, index_builder=None) -> None:
    """Persist ``records`` + the inverted index (+ optional FAISS index).

    Writes ``store/<dataset>/<species_key>.json`` per record and one
    ``inverted.json`` under ``out_dir``. When ``index_builder`` (an
    ``embed_index.IndexBuilder``) is given, also saves ``index.faiss`` +
    ``sidecar.json`` via ``index_builder.save(out_dir)``.
    """
    out_dir = Path(out_dir)
    store_dir = out_dir / "store"
    store_dir.mkdir(parents=True, exist_ok=True)

    attributes_index: dict[str, list[str]] = {}
    hubs_index: dict[str, list[str]] = {}
    parts_index: dict[str, list] = {}

    for record in records:
        dataset, species_key = _dataset_and_species_key(record)
        gid = record["global_id"]

        dataset_dir = store_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / f"{species_key}.json").write_text(json.dumps(record))

        for slot, attr in record.get("attributes", {}).items():
            key = f"{slot}|{attr['value']}"
            attributes_index.setdefault(key, []).append(gid)

        for rel in record.get("relations", []):
            if rel.get("type") == "instance_of":
                hubs_index.setdefault(rel["target"], []).append(gid)

        for crop in record.get("part_crops", []):
            parts_index.setdefault(crop["part_type"], []).append(crop["embedding_ref"])

    inverted = {"attributes": attributes_index, "hubs": hubs_index, "parts": parts_index}
    (out_dir / "inverted.json").write_text(json.dumps(inverted))

    if index_builder is not None:
        index_builder.save(out_dir)


class Store:
    """Read-side view over a persisted store directory."""

    def __init__(self, out_dir, records: dict[str, dict], inverted: dict,
                 parts_gids: dict[str, list[str]], index=None, sidecar=None):
        self._out_dir = Path(out_dir)
        self._records = records
        self._inverted = inverted
        self._parts_gids = parts_gids
        self._index = index
        self._sidecar = sidecar

    @classmethod
    def load(cls, dir) -> "Store":
        """Load records + the inverted index (+ FAISS index, if present) from ``dir``."""
        dir = Path(dir)

        records: dict[str, dict] = {}
        store_dir = dir / "store"
        if store_dir.exists():
            for dataset_dir in sorted(store_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                for record_path in sorted(dataset_dir.glob("*.json")):
                    record = json.loads(record_path.read_text())
                    records[record["global_id"]] = record

        inverted_path = dir / "inverted.json"
        if inverted_path.exists():
            inverted = json.loads(inverted_path.read_text())
        else:
            inverted = {"attributes": {}, "hubs": {}, "parts": {}}

        # Derived (not persisted): part_type -> owning global_ids, built from the
        # loaded records themselves -- lets `query(part_type=...)` work without
        # requiring a FAISS sidecar (inverted.json's "parts" section keys by
        # embedding_ref, which is only resolvable back to a global_id via the
        # sidecar; `nearest_crops` uses that section instead, post-search).
        parts_gids: dict[str, list[str]] = {}
        for record in records.values():
            for crop in record.get("part_crops", []):
                parts_gids.setdefault(crop["part_type"], []).append(record["global_id"])

        index = sidecar = None
        if (dir / "index.faiss").exists() and (dir / "sidecar.json").exists():
            index, sidecar = ei.load_index(dir)

        return cls(dir, records, inverted, parts_gids, index, sidecar)

    def get(self, global_id: str) -> dict | None:
        """Return the full record for ``global_id``, or ``None`` if absent."""
        return self._records.get(global_id)

    def query(self, hub: str | None = None, attribute: tuple[str, str] | None = None,
              part_type: str | None = None) -> list[str]:
        """Filterable structured lookup: intersection of the given filters' hits.

        Each of ``hub``, ``attribute`` (a ``(slot, value)`` pair), and
        ``part_type`` narrows independently via the inverted index; when more
        than one is given, the result is their intersection. With no filters
        given, returns every global_id in the store.
        """
        result_sets = []
        if hub is not None:
            result_sets.append(set(self._inverted.get("hubs", {}).get(hub, [])))
        if attribute is not None:
            slot, value = attribute
            key = f"{slot}|{value}"
            result_sets.append(set(self._inverted.get("attributes", {}).get(key, [])))
        if part_type is not None:
            result_sets.append(set(self._parts_gids.get(part_type, [])))

        if not result_sets:
            return list(self._records.keys())

        hits = result_sets[0]
        for other in result_sets[1:]:
            hits &= other
        return list(hits)

    def members(self, hub_id: str) -> list[str]:
        """Global ids belonging to ``hub_id`` (equivalent to ``query(hub=hub_id)``)."""
        return list(self._inverted.get("hubs", {}).get(hub_id, []))

    def provenance(self, global_id: str) -> dict:
        """The ``provenance`` block of ``global_id``'s record.

        Raises ``KeyError`` if ``global_id`` is not in the store.
        """
        return self._records[global_id]["provenance"]

    def nearest_crops(self, vec, part_type: str | None = None, hub: str | None = None, k: int = 5):
        """FAISS nearest-neighbor search over the index, MMKG-filtered.

        Delegates to ``embed_index.search``; ``hub`` narrows via the allowed
        global_id set (``members(hub)``), ``part_type`` narrows post-search
        via the inverted index's ``parts`` section (embedding_ref -> only
        that part type). Returns ``[]`` when no FAISS index was persisted.
        """
        if self._index is None or self._sidecar is None:
            return []

        allow = set(self.members(hub)) if hub is not None else None
        fetch_k = k * 4 if part_type is not None else k
        results = ei.search(self._index, self._sidecar, vec, k=fetch_k, allow=allow)

        if part_type is not None:
            allowed_refs = set(self._inverted.get("parts", {}).get(part_type, []))
            results = [r for r in results if r[0] in allowed_refs][:k]

        return results
