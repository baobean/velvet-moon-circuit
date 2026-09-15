"""MMKG species store build orchestrator -- Treevill + iNat -> one store.

One shared ``embed_index.IndexBuilder``/encoder load re-embeds every Treevill
reference/crop and every iNat medoid candidate; one VLM load (only if the
iNat side actually runs and no ``read_fn`` override is given) reads bird
attribute slots. Everything funnels through ``store.write_store`` and a
**build summary** -- infrastructure validation only (species/dataset counts,
hub sizes, attribute coverage, #medoids/#crops indexed, eligible/excluded
tallies). No fidelity/DINO/CLIP/repair-success/NN-baseline metrics -- that is
a separate experiment and out of scope here.

``encoder_factory``/``vlm_factory`` are injectable seams; their defaults
import and construct the real ``QwenVLM``/SigLIP encoder lazily, INSIDE the
function body, only when actually called -- so ``import
ragregen.mmkg_store.build`` never loads a model (mirrors
``ragregen.mmkg.inat_pilot``'s own ``_default_vlm_factory``/
``_default_encoder_factory`` pattern).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from ragregen.mmkg import inat_pilot
from ragregen.mmkg.inat_pilot import N_BUILD
from ragregen.mmkg_store import build_inat, build_treevill, manifest, manifests, store
from ragregen.mmkg_store.embed_index import IndexBuilder, LOCKED

DEFAULT_KG_GLOB = "/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/kg_test/outputs/*/kg.json"
DEFAULT_VAL_JSON = inat_pilot.DEFAULT_VAL_JSON
DEFAULT_DATA_ROOT = inat_pilot.DEFAULT_DATA_ROOT
DEFAULT_TAXONOMY_PY = build_treevill.DEFAULT_TAXONOMY_PY
DEFAULT_TREEVILL_ROOT = build_treevill.DEFAULT_TREEVILL_ROOT
DEFAULT_OUT_DIR = "/mmlabworkspace_new/Students/tuanld/soict-2026-data/mmkg_store"
DEFAULT_N_BUILD = N_BUILD
# The store's FAISS sidecar always stamps `LOCKED` regardless of which
# encoder actually produced the vectors (embed_index.IndexBuilder.save), so
# the encoder used here must always equal LOCKED["encoder"] -- never
# retyped, and asserted in `main` below.
DEFAULT_ENCODER = LOCKED["encoder"]


# ---------------------------------------------------------------------------
# Default (real) factories -- imported lazily so importing this module pulls
# no torch/faiss model weights.
# ---------------------------------------------------------------------------

def _default_encoder_factory(name: str = DEFAULT_ENCODER):
    from ragregen.encoders import build_encoder

    return build_encoder(name)


def _default_vlm_factory():
    from ragregen.vlm import QwenVLM

    return QwenVLM()


# ---------------------------------------------------------------------------
# Eligibility accounting
# ---------------------------------------------------------------------------

def _tally_eligible(entry: dict) -> tuple[list, list, dict]:
    """Apply ``manifest.eligible_pool`` to one manifest entry.

    Returns ``(kept, excluded, tally)`` where ``tally`` is
    ``{"eligible": n, "excluded_total": n, "excluded_reasons": {reason: n}}``
    and ``excluded`` is the raw ``[{"path","reason"}, ...]`` list (kept
    alongside the tally so callers can wire it into a per-record
    ``manifest.provenance_block`` without re-running ``eligible_pool``). The
    entry's own ``eval_set`` (if declared) doubles as the contamination
    guard's ``gt_eval_paths`` -- an eligible image byte-identical to a
    declared held-out image is dropped, never silently kept.
    """
    kept, excluded = manifest.eligible_pool(entry, gt_eval_paths=entry.get("eval_set", ()))
    reasons: dict = {}
    for item in excluded:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    tally = {"eligible": len(kept), "excluded_total": len(excluded), "excluded_reasons": reasons}
    return kept, excluded, tally


def _merge_reason_tally(into: dict, reasons: dict) -> None:
    for reason, n in reasons.items():
        into[reason] = into.get(reason, 0) + n


# ---------------------------------------------------------------------------
# Embedding-dimension guard
# ---------------------------------------------------------------------------

def _dim_checked_embed_fn(embed_fn, expected_dim: int):
    """Wrap ``embed_fn`` so the FIRST real embedding it produces is asserted
    to be exactly ``expected_dim``-dimensional, raising ``ValueError`` on
    mismatch. Every later call passes through unchecked -- this is a
    startup guard against a silently swapped/misconfigured encoder, not a
    per-call invariant (``IndexBuilder.add`` already enforces internal
    consistency across every vector it accumulates).
    """
    state = {"checked": False}

    def wrapped(paths):
        vecs = np.asarray(embed_fn(paths))
        if not state["checked"]:
            if vecs.ndim != 2 or vecs.shape[1] != expected_dim:
                actual = vecs.shape[1] if vecs.ndim == 2 else vecs.shape
                raise ValueError(
                    f"encoder produced embedding dim {actual!r} on its first real call, "
                    f"expected {expected_dim} (LOCKED['dim']) -- the store's FAISS "
                    "sidecar always stamps LOCKED regardless of which encoder actually "
                    "produced the vectors, so a dimension mismatch must fail loudly here "
                    "rather than silently mislabel the store."
                )
            state["checked"] = True
        return vecs

    return wrapped


# ---------------------------------------------------------------------------
# Build summary -- infrastructure validation only
# ---------------------------------------------------------------------------

def _hub_sizes(records: list[dict]) -> dict:
    """``hub_id -> member count``, from every record's ``instance_of`` relations."""
    sizes: dict = {}
    for rec in records:
        for rel in rec.get("relations", []):
            if rel.get("type") == "instance_of":
                sizes[rel["target"]] = sizes.get(rel["target"], 0) + 1
    return sizes


def build_summary(records: list[dict], eligibility_tally: dict) -> dict:
    """Infrastructure-validation summary: "was the store built correctly?"

    Species/dataset counts, hub sizes, attribute coverage, #medoids/#crops/
    #candidates indexed, and the eligible/excluded tallies. Deliberately
    carries NO fidelity/DINO/CLIP/repair-success/NN-baseline numbers -- those
    belong to a separate experiment.
    """
    dataset_counts: dict = {}
    attribute_coverage: dict = {}
    n_medoids = n_crops = n_candidates = 0

    for rec in records:
        dataset = rec["dataset"]
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

        cov = attribute_coverage.setdefault(dataset, {
            "total_species": 0, "species_with_attributes": 0, "total_attribute_slots": 0,
        })
        cov["total_species"] += 1
        attrs = rec.get("attributes", {})
        if attrs:
            cov["species_with_attributes"] += 1
        cov["total_attribute_slots"] += len(attrs)

        if rec.get("medoid", {}).get("embedding_ref") is not None:
            n_medoids += 1
        n_crops += len(rec.get("part_crops", []))
        n_candidates += len(rec.get("candidates", []))

    return {
        "n_species": len(records),
        "dataset_counts": dataset_counts,
        "hub_sizes": _hub_sizes(records),
        "attribute_coverage": attribute_coverage,
        "n_medoids_indexed": n_medoids,
        "n_crops_indexed": n_crops,
        "n_candidates_indexed": n_candidates,
        "n_embeddings_indexed": n_medoids + n_crops + n_candidates,
        "eligibility": eligibility_tally,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out")
    p.add_argument("--kg-glob")
    p.add_argument("--val-json")
    p.add_argument("--data-root")
    p.add_argument("--taxonomy-py")
    p.add_argument("--treevill-root")
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--n-build", type=int)
    return p.parse_args(argv if argv is not None else [])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None, *, encoder_factory=None, vlm_factory=None, read_fn=None,
         kg_glob=None, val_json_path=None, data_root=None, taxonomy_py=None,
         n_build=None, out_dir=None, treevill_root=None, expected_dim=None) -> dict:
    """Build the mmkg_store: Treevill (from ``kg_glob``) + iNat pilot birds
    (from ``val_json_path``/``data_root``), through one shared encoder/index.

    Every path/seam kwarg overrides the matching ``argv`` CLI flag, which in
    turn overrides the module default -- so an offline test can drive the
    whole flow with fakes + tmp paths without touching ``argv`` parsing.

    ``treevill_root`` is the root kg.json's own ``ref_paths``/``exemplar_crop``
    are resolved against (kg_test repo root by default -- I1). ``expected_dim``
    defaults to ``LOCKED["dim"]`` (1152): the first real ``embed_fn`` call is
    asserted to produce exactly that many dims, raising ``ValueError`` on
    mismatch (fix-wave-2 #4) -- exposed as a kwarg purely so an offline test
    can drive the check against a smaller toy-encoder dimension; production
    callers should never override it.
    """
    args = _parse_args(argv)

    kg_glob = kg_glob or args.kg_glob or DEFAULT_KG_GLOB
    val_json_path = val_json_path or args.val_json or DEFAULT_VAL_JSON
    data_root = data_root or args.data_root or DEFAULT_DATA_ROOT
    taxonomy_py = taxonomy_py or args.taxonomy_py or DEFAULT_TAXONOMY_PY
    treevill_root = treevill_root or args.treevill_root or DEFAULT_TREEVILL_ROOT
    n_build = n_build if n_build is not None else (
        args.n_build if args.n_build is not None else DEFAULT_N_BUILD)
    out_dir = out_dir or args.out or DEFAULT_OUT_DIR
    encoder_name = args.encoder or DEFAULT_ENCODER
    expected_dim = expected_dim if expected_dim is not None else LOCKED["dim"]

    if encoder_name != LOCKED["encoder"]:
        raise ValueError(
            f"encoder {encoder_name!r} != locked encoder {LOCKED['encoder']!r} -- "
            "the store's FAISS sidecar always stamps the LOCKED config regardless "
            "of which encoder actually produced the vectors, so building with a "
            "different encoder would silently mislabel the store. The store must "
            "never be built with a non-LOCKED encoder."
        )

    with open(val_json_path) as f:
        val_json = json.load(f)

    treevill_manifest = manifests.generate_treevill_manifest(kg_glob, treevill_root=treevill_root)
    inat_manifest = manifests.generate_inat_manifest(val_json, data_root, n_build=n_build)

    tax = build_treevill.load_taxonomy(taxonomy_py)

    builder = IndexBuilder()
    encoder = (encoder_factory or (lambda: _default_encoder_factory(encoder_name)))()
    embed_fn = _dim_checked_embed_fn(encoder.encode_images, expected_dim)

    records: list[dict] = []
    eligibility_tally: dict = {}

    # --- Treevill --------------------------------------------------------
    kg_paths = sorted(glob.glob(kg_glob))
    treevill_eligible_total = 0
    treevill_excluded: dict = {}
    treevill_excluded_concepts: list = []
    for kg_path in kg_paths:
        kg = json.loads(Path(kg_path).read_text())
        concept = kg["concept"]
        entry = treevill_manifest.get(concept)
        if entry is None:
            continue

        # Real data: not every kg.json concept has a taxonomy.py entry (65
        # concepts vs 36 mapped, observed). `treevill_record` KeyErrors on a
        # missing entry -- skip and tally it rather than crashing a long GPU
        # build; self-documented in build_summary.json.
        if concept not in tax:
            treevill_excluded["no_taxonomy_entry"] = treevill_excluded.get("no_taxonomy_entry", 0) + 1
            treevill_excluded_concepts.append(concept)
            continue

        kept, excluded, tally = _tally_eligible(entry)
        treevill_eligible_total += tally["eligible"]
        _merge_reason_tally(treevill_excluded, tally["excluded_reasons"])

        # I2: an empty eligible pool (every image dropped by dedup/contamination,
        # or the concept simply had none) must never reach `_centroid_nearest_index`
        # (raises ValueError on an empty array) -- skip and tally, never crash.
        if not kept:
            treevill_excluded["empty_eligible_pool"] = treevill_excluded.get("empty_eligible_pool", 0) + 1
            treevill_excluded_concepts.append(concept)
            continue

        provenance_extra = manifest.provenance_block(entry, None, kept, excluded)
        try:
            rec = build_treevill.treevill_record(
                concept, kg, tax, kept, embed_fn, builder.add,
                treevill_root=treevill_root, provenance_extra=provenance_extra,
            )
        except KeyError:
            # Defensive backstop for any taxonomy-shaped KeyError the explicit
            # membership check above didn't catch -- never crash the build.
            treevill_excluded["no_taxonomy_entry"] = treevill_excluded.get("no_taxonomy_entry", 0) + 1
            treevill_excluded_concepts.append(concept)
            continue
        records.append(rec)

    eligibility_tally["treevill"] = {
        "n_concepts": len(kg_paths),
        "eligible": treevill_eligible_total,
        "excluded_reasons": treevill_excluded,
        "excluded_concepts": treevill_excluded_concepts,
    }

    # --- iNat --------------------------------------------------------------
    filtered_inat_manifest: dict = {}
    inat_provenance_extra: dict = {}
    inat_eligible_total = 0
    inat_excluded: dict = {}
    inat_excluded_species: list = []
    for species_key, entry in inat_manifest.items():
        kept, excluded, tally = _tally_eligible(entry)
        inat_eligible_total += tally["eligible"]
        _merge_reason_tally(inat_excluded, tally["excluded_reasons"])

        # I2: same empty-pool guard as Treevill -- a species tallied down to
        # zero eligible images must never reach `_centroid_nearest_index`
        # (raises ValueError on an empty array); skip it, tally it, never
        # call the adapter with an empty pool.
        if not kept:
            inat_excluded["empty_eligible_pool"] = inat_excluded.get("empty_eligible_pool", 0) + 1
            inat_excluded_species.append(species_key)
            continue

        filtered_inat_manifest[species_key] = {
            "eligible": kept, "eval_set": entry.get("eval_set", []),
        }
        inat_provenance_extra[species_key] = manifest.provenance_block(entry, None, kept, excluded)

    eligibility_tally["inat"] = {
        "n_species": len(inat_manifest),
        "eligible": inat_eligible_total,
        "excluded_reasons": inat_excluded,
        "excluded_species": inat_excluded_species,
    }

    if filtered_inat_manifest:
        if read_fn is not None:
            read = read_fn
        else:
            vlm = (vlm_factory or _default_vlm_factory)()
            read = lambda path: inat_pilot.read_bird_slots(path, vlm)  # noqa: E731

        inat_recs = build_inat.inat_records(
            val_json, data_root, filtered_inat_manifest, embed_fn, read, builder.add,
            provenance_extra_by_species=inat_provenance_extra,
        )
        records.extend(inat_recs)

    store.write_store(records, out_dir, index_builder=builder)

    summary = build_summary(records, eligibility_tally)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "build_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[mmkg_store.build] wrote {len(records)} records "
          f"({eligibility_tally['treevill']['n_concepts']} treevill concepts, "
          f"{eligibility_tally['inat']['n_species']} inat species) to {out_dir}")

    return summary


if __name__ == "__main__":
    main()
