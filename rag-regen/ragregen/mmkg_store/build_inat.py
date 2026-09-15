"""iNaturalist birds build adapter -- 25 pilot species (locked) -> mmkg_store record.

Species selection is delegated to ``ragregen.mmkg.inat_pilot.select_pilot_species``
(the frozen §2 selection, LOCKED to exactly 25 species in production) -- this
module never re-derives or expands the pilot set. For each selected species,
the eligible image pool comes from the manifest (already contamination/dedup
clean, §3), never from ``inat_pilot.species_image_paths`` directly. Every
eligible image is re-embedded via the store's own ``embed_fn`` seam (the
val.json/manifest carry no embeddings of their own) to pick the
centroid-nearest medoid; bird attributes reuse
``inat_pilot.bird_target_attributes`` -- the same name-blind consensus rule
validated by the pilot -- over one ``read_fn`` read per eligible image.

MVP scope: whole-image medoid + attributes only -- birds get no part crops
(``part_crops = []``), unlike the Treevill adapter.
"""
from __future__ import annotations

import numpy as np

from ragregen.mmkg import inat_pilot
from ragregen.mmkg_store import schema


def _centroid_nearest_index(vectors: np.ndarray) -> int:
    """Index of the row in ``vectors`` with max cosine similarity to their centroid."""
    centroid = vectors.mean(axis=0)
    norms = np.linalg.norm(vectors, axis=1)
    centroid_norm = np.linalg.norm(centroid)
    denom = norms * (centroid_norm if centroid_norm > 0 else 1.0)
    denom = np.where(denom == 0, 1.0, denom)
    sims = (vectors @ centroid) / denom
    return int(np.argmax(sims))


def _species_key(category: dict) -> str:
    """Zero-padded category id -- matches the ``image_dir_name`` prefix
    (e.g. ``"04486"``)."""
    return f"{int(category['id']):05d}"


def inat_record(species, manifest, embed_fn, read_fn, index_add, *, data_root=None,
                 provenance_extra: dict | None = None) -> dict:
    """Assemble one iNat species record for the selected-species dict ``species``.

    ``manifest[species_key]["eligible"]`` (the manifest's contamination/dedup
    -clean pool) is re-embedded in one batch via ``embed_fn`` to pick the
    centroid-nearest medoid and to index every candidate; every indexed
    vector goes through the ``index_add(vec, meta) -> embedding_ref`` seam,
    mirroring the Treevill adapter. Birds carry no part crops (MVP).
    ``provenance_extra``, if given, is merged into the record's
    ``provenance`` (the §3/§4 fields -- ``build_manifest``/``excluded_images``/
    ``source_split`` -- assembled by the caller via
    ``manifest.provenance_block``).
    """
    species_key = _species_key(species)
    entry = manifest[species_key]
    eligible = list(entry["eligible"])

    embeddings = np.asarray(embed_fn(eligible))
    medoid_idx = _centroid_nearest_index(embeddings)
    medoid_path = eligible[medoid_idx]

    gid = schema.global_id("inat", species_key)

    candidates = []
    medoid_ref = None
    for i, (path, vec) in enumerate(zip(eligible, embeddings)):
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
        "k_images": len(eligible),
        "selection": "siglip-centroid-nearest",
    }

    reads = [read_fn(path) for path in eligible]
    consensus = inat_pilot.bird_target_attributes(reads)
    attributes = schema.attribute_record(consensus, source="inat-consensus")

    taxonomy = {"genus": species.get("genus"), "family": species.get("family")}

    provenance = {
        "eligible_images": eligible,
        "category_id": species["id"],
        "data_root": data_root,
    }
    if provenance_extra:
        provenance.update(provenance_extra)

    return schema.build_record(
        dataset="inat",
        species_key=species_key,
        scientific_name=species.get("name"),
        common_name=species.get("common_name"),
        taxonomy=taxonomy,
        medoid=medoid,
        candidates=candidates,
        part_crops=[],
        attributes=attributes,
        provenance=provenance,
    )


def inat_records(val_json, data_root, manifest, embed_fn, read_fn, index_add, *,
                  provenance_extra_by_species: dict[str, dict] | None = None) -> list[dict]:
    """One record per pilot species selected by
    ``inat_pilot.select_pilot_species(val_json)`` -- the production selection
    seam (LOCKED to 25 species in production). Eligible images always come
    from ``manifest``, never from ``inat_pilot.species_image_paths`` directly.

    A selected species absent from ``manifest`` is skipped rather than
    KeyError'd -- the caller (``build.py``) omits a species whose eligible
    pool ended up empty after contamination/dedup filtering (I2: the build
    must never crash, and the adapter must never be called with an empty
    pool); that species is tallied by the caller instead.
    """
    species_list = inat_pilot.select_pilot_species(val_json)
    provenance_extra_by_species = provenance_extra_by_species or {}
    records = []
    for species in species_list:
        species_key = _species_key(species)
        if species_key not in manifest:
            continue
        records.append(inat_record(
            species, manifest, embed_fn, read_fn, index_add, data_root=data_root,
            provenance_extra=provenance_extra_by_species.get(species_key),
        ))
    return records
