"""Treevill build adapter — GRAFT ``kg.json`` -> mmkg_store record.

Re-embeds every eligible image (and part-crop exemplar) with the store's own
encoder via the ``embed_fn`` seam; kg.json's own SigLIP2 embeddings are never
read. Attributes are taken directly from each part's ``{"name","value","source"}``
consensus entries -- kg.json carries no image-level support/visible_count
statistics, so those fields are recorded as ``None`` rather than fabricated.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from ragregen.mmkg_store import schema

DEFAULT_TAXONOMY_PY = "/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/kg_test/graft/taxonomy.py"

# kg.json's `ref_paths`/`exemplar_crop` (e.g. "data/treevill/rawdata2/Akashmoni/1.jpg",
# "outputs/Akashmoni/crops/leaf.png") are relative to the kg_test repo root, NOT
# rag-regen -- every such path must be resolved against this root before it can be
# hashed/opened.
DEFAULT_TREEVILL_ROOT = "/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/kg_test"

# Normalized attribute values that mean "no consensus" -- dropped, not indexed.
NOT_VISIBLE = {"not visible", "none", "n/a", ""}


def load_taxonomy(taxonomy_py: str = DEFAULT_TAXONOMY_PY) -> dict[str, tuple[str, str]]:
    """Read the ``TAXONOMY`` dict literal out of ``taxonomy_py`` without importing it.

    Parses the module's source with ``ast`` and ``literal_eval``s the value
    assigned to the module-level ``TAXONOMY`` name (a plain or annotated
    assignment), so no cross-repo import of the GRAFT codebase is needed.

    Returns ``{concept: (scientific_name, family)}``.
    """
    tree = ast.parse(Path(taxonomy_py).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "TAXONOMY" for t in node.targets
        ):
            return ast.literal_eval(node.value)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "TAXONOMY"
            and node.value is not None
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"no top-level TAXONOMY assignment found in {taxonomy_py}")


def _centroid_nearest_index(vectors: np.ndarray) -> int:
    """Index of the row in ``vectors`` with max cosine similarity to their centroid."""
    centroid = vectors.mean(axis=0)
    norms = np.linalg.norm(vectors, axis=1)
    centroid_norm = np.linalg.norm(centroid)
    denom = norms * (centroid_norm if centroid_norm > 0 else 1.0)
    denom = np.where(denom == 0, 1.0, denom)
    sims = (vectors @ centroid) / denom
    return int(np.argmax(sims))


def _build_attributes(kg: dict) -> dict:
    """Attribute records built directly from parts' consensus entries.

    Keyed ``"<part>.<name>"``; drops any attribute whose normalized value is
    in ``NOT_VISIBLE``. ``support``/``visible_count`` are always ``None`` --
    kg.json has no per-image consensus counts, so none are fabricated.
    """
    attributes: dict = {}
    for part in kg.get("parts", []):
        part_name = part["name"]
        for attr in part.get("attributes", []):
            value = attr.get("value")
            if str(value).strip().lower() in NOT_VISIBLE:
                continue
            attributes[f"{part_name}.{attr['name']}"] = {
                "value": value,
                "support": None,
                "visible_count": None,
                "source": attr.get("source", "graft-kg"),
            }
    return attributes


def treevill_record(concept, kg, tax, eligible_paths, embed_fn, index_add, *,
                     treevill_root: str = DEFAULT_TREEVILL_ROOT,
                     provenance_extra: dict | None = None) -> dict:
    """Assemble one Treevill species record for ``concept``.

    ``eligible_paths`` (the manifest's contamination/dedup-clean pool) are
    re-embedded in one batch via ``embed_fn`` to pick the centroid-nearest
    medoid and to index every candidate; each part's ``exemplar_crop`` is
    re-embedded individually and indexed as a typed part crop. Every indexed
    vector goes through the ``index_add(vec, meta) -> embedding_ref`` seam.

    ``eligible_paths`` are assumed already resolved (the caller/manifest
    joins ``ref_paths`` against ``treevill_root``); ``exemplar_crop`` is
    resolved against ``treevill_root`` here since it is read straight off
    ``kg`` rather than the manifest. ``provenance_extra``, if given, is
    merged into the record's ``provenance`` (the §3/§4 fields --
    ``build_manifest``/``excluded_images``/``source_split`` -- assembled by
    the caller via ``manifest.provenance_block``).
    """
    scientific_name, family = tax[concept]
    genus = scientific_name.split()[0]
    taxonomy = {"genus": genus, "family": family}
    gid = schema.global_id("treevill", concept)

    eligible_paths = list(eligible_paths)
    embeddings = np.asarray(embed_fn(eligible_paths))
    medoid_idx = _centroid_nearest_index(embeddings)
    medoid_path = eligible_paths[medoid_idx]

    candidates = []
    medoid_ref = None
    for i, (path, vec) in enumerate(zip(eligible_paths, embeddings)):
        ref = index_add(vec, {"global_id": gid, "kind": "candidate", "path": path})
        candidates.append({"image_path": path, "embedding_ref": ref})
        if i == medoid_idx:
            medoid_ref = index_add(
                embeddings[medoid_idx],
                {"global_id": gid, "kind": "medoid", "path": medoid_path},
            )

    medoid = {
        "image_path": medoid_path,
        "embedding_ref": medoid_ref,
        "k_images": len(eligible_paths),
        "selection": "siglip-centroid-nearest",
    }

    part_crops = []
    for part in kg.get("parts", []):
        crop_path = part.get("exemplar_crop")
        if not crop_path:
            continue
        resolved_crop_path = str(Path(treevill_root) / crop_path)
        crop_vec = np.asarray(embed_fn([resolved_crop_path]))[0]
        ref = index_add(crop_vec, {"global_id": gid, "kind": "part_crop", "path": resolved_crop_path})
        part_crops.append({
            "part_type": part["name"],
            "image_path": resolved_crop_path,
            "embedding_ref": ref,
        })

    attributes = _build_attributes(kg)

    provenance = {"eligible_images": eligible_paths, "kg_concept": kg.get("concept")}
    if provenance_extra:
        provenance.update(provenance_extra)

    return schema.build_record(
        dataset="treevill",
        species_key=concept,
        scientific_name=scientific_name,
        common_name=concept,
        taxonomy=taxonomy,
        medoid=medoid,
        candidates=candidates,
        part_crops=part_crops,
        attributes=attributes,
        provenance=provenance,
    )
