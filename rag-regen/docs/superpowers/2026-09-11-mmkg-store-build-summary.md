# MMKG species store — build summary (infrastructure artifact)

**Date:** 2026-09-11
**What this is:** an **infrastructure validation** record of the closed-world species reference registry
built over Treevill + iNaturalist. It answers only *"was the store built correctly?"* — species/eligible/
excluded/medoid/crop counts, index config, and the adapter contract. **It is not a research result and carries
no fidelity/accuracy claim** (per the frozen spec; the MMKG's accuracy question was already answered negative
by GRAFT + the iNat pilot). Do not read build success as evidence of an MMKG accuracy improvement.

**Spec:** `docs/superpowers/specs/2026-09-10-mmkg-species-store-design.md` (frozen).
**Code:** `ragregen/mmkg_store/` (branch `mmkg-species-store`; 60 tests; whole-branch reviewed).
**Store:** `/mmlabworkspace_new/Students/tuanld/soict-2026-data/mmkg_store/` (`store/`, `index.faiss`,
`sidecar.json`, `inverted.json`, `build_summary.json`).

## Counts (from `build_summary.json`, verified via `Store.load`)

```
TreeVill:
  species built:        36   (of 65 kg.json concepts; 29 excluded)
  eligible images:      133
  excluded:             29 concepts, reason "no_taxonomy_entry"  (not in graft/taxonomy.py -> no hub)
                        0 empty-pool, 0 duplicate, 0 contamination
  medoids:              36   (siglip-centroid-nearest, one per species)
  part crops:           129  (leaf / bark / branching / cone_or_flower)
  attribute coverage:   34/36 species carry >=1 attribute; 151 attribute slots
                        support/visible_count = null  (GRAFT kg.json has no per-image counts; not fabricated)
  source_split:         "ref_paths_only"  (kg.json has no held-out split -> NO Treevill eval-set contamination
                        guard; intra-pool SHA-256 dedup still applied)

iNaturalist (birds):
  species built:        25   (exactly the locked pilot set; no auto-expand)
  eligible images:      175  (7 build images/species)
  excluded:             0
  medoids:              25   (siglip-centroid-nearest)
  part crops:           0    (MVP: birds carry medoid + attributes only)
  attribute coverage:   25/25 species carry attributes; 108 attribute slots
                        support/visible_count = integer counts; source "inat-consensus"
  source_split:         "build_7_eval_3"  (3 held-out images/species declared as eval_set and excluded from
                        the candidate pool; contamination-checked)

Index:
  vectors (embeddings): 498   (61 medoids + 129 part crops + 308 candidate images)
  dim:                  1152
  encoder:              siglip_so400m_384   (asserted == LOCKED on load)
  metric:               inner product (cosine on L2-normalised vectors)
  index:                IndexFlatIP
  (Treevill re-embedded with this encoder; kg.json's own SigLIP2-768 vectors ignored.)

Hubs (typed relations):  76 dataset-namespaced family/genus hubs (e.g. inat:family:Certhiidae -> 3 species).
```

## Adapter contract (verified, both datasets)

```
known species  ->  global_id ("<dataset>:<species_key>")
               ->  store.get(global_id)
               ->  record.medoid.image_path        (precomputed at build; deterministic)
               ->  PIL.Image (RGB)                  ->  existing prepare_reference / Inpainter (unchanged)
```

Verified: `reference_source.medoid_reference(store, "treevill:Akashmoni")` → RGB 256×256;
`("inat:03678")` → RGB 500×363; `("inat:99999")` (missing) → **`None`** (no fallback species).
The adapter uses **only** `store.get` — no FAISS, no `nearest_crops`, no NN, no filtering, no attribute text,
no multi-reference; the generator is untouched.

## Provenance & integrity

Every record carries `provenance = {build_manifest{path,sha256}, eligible_images, excluded_images(+reason),
source_split}`. The contamination/eligibility guard ran on every image (SHA-256 dedup + eval-set exclusion);
on this data it produced 0 duplicate/contamination exclusions. FAISS `sidecar.json` stores the locked config
and is asserted on `Store.load`; the build asserts the first real embedding is 1152-d.

## Known limitations (documented, not worked around)

- **Treevill has no held-out split** (`source_split: ref_paths_only`) — kg.json provides only `ref_paths`, so
  Treevill records have no eval-set and thus no leakage guard against a *future* Treevill evaluation (only
  intra-pool dedup). A held-out split must be declared then; it was **not** fabricated here.
- **29 Treevill concepts excluded** for lacking a `graft/taxonomy.py` entry (~45% of the 65) — a taxonomy
  data-completeness gap, listed in `build_summary.json`. Closing it would extend the Treevill set later.
- Portability required committing two pre-existing untracked shared deps (`ragregen/eval_manifest.py`,
  `ragregen/retrieved_reference.py`) — done in a dedicated dependency commit; a clean checkout now imports.

## Build notes

Attempt 1 was OOM-killed (SigLIP + Qwen VLM co-resident on a 23 GB host under GPU contention). Attempt 2
completed with the GPU free. One VLM load; ~500 SigLIP embeddings + 175 iNat name-blind attribute reads;
offline scoring. No repair-accuracy experiment was run.
