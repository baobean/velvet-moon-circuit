"""MMKG species store manifest generators -- Treevill + iNat pilot.

Pure/file-reading: each function returns the manifest dict directly, in the
``{species_key: {"eligible": [...], "eval_set": [...]?}}`` shape
``manifest.load_manifest``/``manifest.eligible_pool`` consume. Writing the
result to disk (the "commit the manifest files" step) is the caller's job --
these functions never touch the filesystem except to read their inputs.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from ragregen.mmkg import inat_pilot
from ragregen.mmkg.inat_pilot import N_BUILD, N_TEST
from ragregen.mmkg_store.build_treevill import DEFAULT_TREEVILL_ROOT


def generate_treevill_manifest(kg_glob: str, treevill_root: str = DEFAULT_TREEVILL_ROOT) -> dict:
    """One manifest entry per Treevill concept, keyed by ``kg["concept"]``.

    ``eligible`` is the kg.json's own GRAFT build references
    (``kg["ref_paths"]``), each resolved against ``treevill_root`` -- kg.json
    stores these paths relative to the kg_test repo root (e.g.
    ``"data/treevill/rawdata2/Akashmoni/1.jpg"``), NOT rag-regen, so they must
    be joined against that root before they can be hashed/opened. ``eligible``
    is every image GRAFT itself used to build that concept's knowledge graph.

    ``source_split`` is explicitly ``"ref_paths_only"``: kg.json provides no
    build/held-out split of its own (I4) -- this is documented rather than
    left implicit, and never fabricated. No ``eval_set``: Treevill carries no
    held-out ground-truth split, so ``manifest.eligible_pool`` is later
    called with an empty ``gt_eval_paths`` and only intra-pool SHA-256 dedup
    applies.
    """
    manifest: dict = {}
    for kg_path in sorted(glob.glob(kg_glob)):
        kg = json.loads(Path(kg_path).read_text())
        concept = kg["concept"]
        manifest[concept] = {
            "eligible": [str(Path(treevill_root) / p) for p in kg.get("ref_paths", [])],
            "source_split": "ref_paths_only",
        }
    return manifest


def generate_inat_manifest(val_json: dict, data_root: str, n_build: int = N_BUILD) -> dict:
    """One manifest entry per pilot species (``inat_pilot.select_pilot_species``,
    LOCKED 25 in production; a monkeypatched seam in tests).

    Species key is the zero-padded category id (matches
    ``build_inat``'s ``species_key`` / the ``image_dir_name`` prefix, e.g.
    ``"04486"``). ``eligible`` is the species' first ``n_build`` val images
    (the build split); ``eval_set`` is the next ``N_TEST`` (spec-frozen 3,
    imported from ``inat_pilot`` rather than retyped) -- declared so
    ``manifest.eligible_pool``'s contamination guard can catch any
    byte-identical leak between the two splits. ``source_split`` records the
    actual split sizes used (``"build_<n_build>_eval_<N_TEST>"``) -- iNat, unlike
    Treevill, keeps its real build/eval split (pilot 7-build/3-eval by default).
    """
    manifest: dict = {}
    source_split = f"build_{n_build}_eval_{N_TEST}"
    for species in inat_pilot.select_pilot_species(val_json):
        species_key = f"{int(species['id']):05d}"
        paths = inat_pilot.species_image_paths(val_json, species["id"], data_root)
        manifest[species_key] = {
            "eligible": paths[:n_build],
            "eval_set": paths[n_build:n_build + N_TEST],
            "source_split": source_split,
        }
    return manifest
