"""MMKG species store manifest — eligibility, contamination, and dedup guard.

BLOCKING contamination policy: a candidate image never enters the eligible pool
if it is byte-identical to another already-kept image (a within-manifest
duplicate) or to a ground-truth evaluation reference (leakage). Both checks are
exact SHA-256 comparisons, reusing the same hashing/contamination primitives
the verifier-v2 pipeline already trusts (``eval_manifest.sha256_file``,
``retrieved_reference.contamination``) rather than reimplementing them.

An image that cannot be hashed (missing/unreadable file) is dropped rather than
raised -- the Stage-1 lesson: a corrupt or absent artifact should degrade the
pool, not crash the build.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from ragregen.eval_manifest import sha256_file
from ragregen.retrieved_reference import contamination


def load_manifest(path) -> dict[str, dict]:
    """Load and validate a species manifest.

    Schema: ``{species_key: {"eligible": [img paths], "eval_set": [paths]?}}``.
    Validates that every entry's ``eligible`` (and ``eval_set``, if present) is
    a list.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"manifest {path} must be a JSON object")

    for species_key, entry in data.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{species_key}: entry must be an object")
        if not isinstance(entry.get("eligible"), list):
            raise ValueError(f"{species_key}: 'eligible' must be a list")
        if "eval_set" in entry and not isinstance(entry["eval_set"], list):
            raise ValueError(f"{species_key}: 'eval_set' must be a list")

    return data


def eligible_pool(entry: Mapping, gt_eval_paths: Sequence = ()) -> tuple[list[str], list[dict]]:
    """Filter ``entry["eligible"]`` to the contamination/dedup-clean pool.

    Drops, in order: paths that cannot be hashed (unreadable), sha256
    duplicates of an already-kept path, and paths that are byte-identical to
    any of ``gt_eval_paths``. Returns ``(kept, excluded)`` where ``excluded``
    entries are ``{"path","reason"}`` with reason in
    {"unhashable","duplicate","contamination"}.

    Raises ``ValueError`` if any kept path is also listed in the manifest's
    own ``eval_set`` -- an eligible image must never double as its own
    evaluation reference.
    """
    kept: list[str] = []
    excluded: list[dict] = []
    seen_hashes: dict[str, str] = {}

    for path in entry.get("eligible", []):
        try:
            digest = sha256_file(Path(path))
        except OSError:
            excluded.append({"path": path, "reason": "unhashable"})
            continue

        if digest in seen_hashes:
            excluded.append({"path": path, "reason": "duplicate"})
            continue

        if contamination(Path(path), [Path(p) for p in gt_eval_paths]):
            excluded.append({"path": path, "reason": "contamination"})
            continue

        seen_hashes[digest] = path
        kept.append(path)

    eval_set = set(entry.get("eval_set", []))
    leaked = [path for path in kept if path in eval_set]
    if leaked:
        raise ValueError(
            f"eligible path(s) also listed in eval_set: {sorted(leaked)}")

    return kept, excluded


def provenance_block(entry: Mapping, manifest_path, kept: list[str],
                     excluded: list[dict]) -> dict:
    """Assemble the provenance record for one manifest entry's eligible pool.

    ``manifest_path`` is hashed so the record is traceable to the exact
    manifest build that produced ``kept``/``excluded``. When the manifest was
    generated in-memory (no on-disk file to hash -- the mmkg_store build
    generates both the Treevill and iNat manifests directly from
    ``kg.json``/``val.json`` each run, never a committed manifest file),
    ``manifest_path`` may be ``None``: ``build_manifest`` then carries
    ``{"path": None, "sha256": <sha256 of the entry's own JSON
    serialization>}`` -- a clear in-memory marker, still traceable to the
    exact entry, rather than a fabricated file path.
    """
    if manifest_path is None:
        serialized = json.dumps(entry, sort_keys=True, default=str).encode("utf-8")
        build_manifest = {"path": None, "sha256": hashlib.sha256(serialized).hexdigest()}
    else:
        manifest_path = Path(manifest_path)
        build_manifest = {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}
    return {
        "source_split": entry.get("source_split"),
        "build_manifest": build_manifest,
        "eligible_images": kept,
        "excluded_images": excluded,
    }
