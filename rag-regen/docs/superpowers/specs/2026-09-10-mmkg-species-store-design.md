# MMKG structured species reference store + rag-regen adapter (design)

**Date:** 2026-09-10
**Status:** design, pending review (revised per review #1–#10). Author sign-off pending.
**What this is:** a **structured reference store / registry** — a lightweight, graph-shaped (typed) MMKG
representation, **not** a graph database and **not** a retrieval system. With the concept (species) known per
repair case, its repair job is a **keyed lookup**: `species → precomputed medoid image`. It makes **no fidelity
claim** (GRAFT + iNat established that MMKG structure does not beat retrieval); it *organizes and enriches*
references and serves structured/filterable **queries** for analysis and future work. **NN plays no role at
repair time.**

## 1. Purpose and MVP scope

A standalone structured species reference store over **Treevill** (trees) + **iNaturalist** (birds), **plus a
thin rag-regen reference-source adapter**. Closed-world, keyed by species (correct for repair — the concept is
known; we never NN-retrieve on the masked wrong region).

**In scope (MVP):**
1. The store: schema (§4), two domain adapters (§5), persistence + query API (§6).
2. The rag-regen adapter (§7): `species key → record.medoid → PIL Image`, returned through the **existing**
   repair reference interface, unchanged. No repair-time NN, no FAISS, no filtering.
3. The contamination / eligibility policy (§3) — **blocking**, built in from day one.

**Out of scope (later; schema is extensible to these but none is implemented or claimed in MVP):** open-world
on-the-fly build; cold-start composition; candidate filtering + re-ranking at repair time; a full species
repair *eval*; bird part-crop detection; any fidelity/accuracy experiment.

## 2. Non-goals / guardrails

- **No fidelity claim, no gate, no NN at repair time.** The repair adapter returns the *precomputed* medoid by
  key. Filtering/re-ranking candidates is a **separate future feature**, deliberately excluded from MVP because
  a filtered candidate set can change which image is the medoid — the store must return one deterministic,
  provenance-backed reference.
- The repair generator (`regen.Inpainter`, `regen.prepare_reference`) is **not modified**.
- Terminology is honest: "structured reference store / typed reference registry / lightweight MMKG
  representation" — never "graph DB", "retrieval system", or "cold-start-capable" (the last would invite a
  claim we are not making).

## 3. Contamination / eligibility policy (BLOCKING — review #3)

The store's trustworthiness for any later evaluation depends on this, so it is built in from the start, not
retrofitted.

- **Build manifest is explicit and required.** Store construction consumes an explicit per-species manifest of
  **eligible source images** (the candidate pool). The build uses **only** those images. There is no implicit
  "just glob the folder".
- **Evaluation/query images are excluded from the candidate pool.** For iNat, if a later repair eval uses `val`
  test images, those must not be in the store — so the manifest declares the eligible (build) split and the
  build asserts the eval/query split is disjoint from it. (Treevill: reuse GRAFT's `split_refs` build/held-out
  split; the store's candidate pool = build refs only.)
- **Contamination guard.** When an eval/reference set is declared, the build runs the existing SHA-256
  `retrieved_reference.contamination` check and **excludes** any candidate byte-identical to a declared eval
  image; dedup by SHA-256 within the pool.
- Every record's `provenance` (§4) records: `source_split`, `build_manifest` (path + sha256), `eligible_images`,
  `excluded_images` (with reason: duplicate | contamination | eval-split).

## 4. Schema (typed, persisted lightweight — review #4, #6, #7, #8, #9)

**Identity (review #8):** the primary key is the pair `(dataset, species_key)`, serialized as a global id
`"<dataset>:<species_key>"` (e.g. `treevill:Cannonball Tree`, `inat:04486`). All cross-references and index
entries use the global id to avoid cross-dataset collisions.

One record per species (`store/<dataset>/<species_key>.json`):

- **species:** `global_id`, `dataset` (`treevill`|`inat`), `species_key`, `scientific_name`,
  `common_name: str | null` (review #7 — do not assume it exists).
- **taxonomy + relations (review #6 — make the graph explicit without a graph DB):**
  ```
  taxonomy: { family, genus, order?, class? }
  relations: [ {type: "instance_of", target: "<dataset>:genus:<genus>"},
               {type: "instance_of", target: "<dataset>:family:<family>"} ]
  ```
  Hub ids are **dataset-namespaced** (`treevill:genus:Certhia`, `inat:family:Anatidae`) so they cannot collide
  across datasets — MVP keeps taxonomy per-dataset; a genuinely shared cross-dataset hub (with defined merge
  semantics) is deferred. The reverse index (§6) lists a hub's members. This is a typed, graph-shaped
  representation — not Neo4j.
- **medoid (precomputed at build — review #1):** `{image_path, embedding_ref, k_images, selection: "siglip-
  centroid-nearest"}`. Computed once at build from the eligible pool; this is exactly what the repair adapter
  returns. **No repair-time recomputation.**
- **candidates:** `[{image_path, embedding_ref}]` — the eligible member images (for the query API / analysis /
  future re-ranking). Not consulted by the repair path.
- **part_crops** (typed, domain vocab, optional): `[{part_type, crop_path, bbox, embedding_ref, source_image}]`.
- **attributes (review #9 — define fields, store verified output only, do not re-open verifier research):**
  ```
  <slot>: { value: str,            # the consensus plurality value (single value per slot)
            support: int,          # # eligible images agreeing on `value`  (COUNT, not fraction)
            visible_count: int,    # # eligible images where the slot was non-missing
            provenance: {image_paths} }
  ```
  Consensus is the **already-frozen** rule (`ragregen.mmkg.schema`: `visible_count>=3` AND `support >=
  max(2, ceil(0.5*visible_count))`); a slot with no consensus is simply absent. No thresholds, judges, or
  multi-value slots are introduced here — the store persists the verified build output, nothing more.
- **provenance:** §3 fields + `{encoder_id, vlm_id, built_at, build_config}`.

## 5. Domain adapters (unified ingester → identical §4 records)

- **Treevill:** ingest from GRAFT's already-built `kg.json` (`kg_test/outputs/<concept>/kg.json`) — medoid,
  part crops (`leaf`/`bark`/`branching`/`cone_or_flower`), attributes — taxonomy from `graft/taxonomy.py`. The
  eligible pool = GRAFT's **build** refs (`split_refs`, k_build); held-out images excluded (§3). Data reuse +
  reshape; no cross-repo code import required (read JSON, reference/copy crops), recording GRAFT build as
  provenance.
- **iNaturalist birds:** **MVP scope is locked to exactly the 25 pilot bird species already on disk** under
  `/mmlabworkspace_new/.../inat2021_birds/` (the `inat_pilot` §2 selection) — the build must **not** auto-expand
  to more of iNat (that would balloon into a new experiment). Build from the manifest-declared eligible `val`
  images. Per species: SigLIP medoid over the eligible pool
  (`ragregen.encoders.build_encoder("siglip_so400m_384")`), bird attributes via the name-blind read reused
  from `ragregen.mmkg.inat_pilot`, taxonomy from `val.json`. **MVP: whole-image medoid + attributes only — no
  bird part-crops.** The eligible split is declared in the manifest and asserted disjoint from any eval/query
  split.

## 6. Persistence + query API (review #4, #5, #6)

**Lightweight, not a graph DB:**
- Per-species JSON records (§4) under `store/`.
- One FAISS index over embeddings (medoids + candidates + part crops). **FAISS is a query/indexing facility,
  NOT part of the repair-time medoid path** (§7). `embedding_ref` is a **stable build-local embedding record
  id** (an opaque integer); its *current implementation* is the FAISS index id, but the schema does not depend
  on FAISS — a sidecar maps `embedding_ref -> {faiss_id, global_id, kind: medoid|candidate|part_crop, path,
  encoder_id, normalized:true, metric:"ip"}`, so the vector backend can change without touching records.
  **Locked at build:** encoder `siglip_so400m_384`, **L2-normalized** vectors, **inner-product (cosine)** metric,
  dim **1152**, **`IndexFlatIP`** (exact — matches `retrieve.py`'s deliberate flat/exact choice). This config is
  stored in the sidecar and asserted on load (a record's `embedding_ref` is meaningless without it).
- Reverse/inverted index: `attribute (slot,value) -> global_ids`, `hub (namespaced family|genus) -> global_ids`,
  `part_type -> crop ids`.

**Query API** (`ragregen/mmkg_store/store.py`) — the structured value; distinct from the repair path:
- `get(global_id) -> Record` — the full record (used by the repair adapter, §7).
- `query(hub=?, attribute=(slot,value)?, part_type=?) -> list[global_id]` — filterable structured lookup.
- `nearest_crops(embedding, part_type=?, hub=?, k=?) -> list[crop]` — FAISS NN, MMKG-filtered (query-side only).
- `members(hub) -> list[global_id]`, `provenance(global_id) -> dict`.

## 7. rag-regen reference-source adapter (review #1, #2 — simplified)

`ragregen/mmkg_store/reference_source.py` — produces the single `reference: Image.Image` that the existing
`regen.prepare_reference` / `Inpainter` consume. It does **not** change repair, and performs **no NN and no
FAISS**:

```
repair case (known species) ─▶ global_id
        ▼
store.get(global_id)
        ▼
record.medoid.image_path            # precomputed at build; deterministic
        ▼
PIL Image  ─▶ existing prepare_reference()  ─▶ existing Inpainter
```

That is the whole adapter. It uses **only** `store.get`; it must **not** call the §6 query API
(`nearest_crops`, `query`) or touch FAISS. No attribute text is fed, no multiple crops, no candidate filtering,
no NN on the masked region. If the species is absent from the store, the adapter returns `None` (the caller
falls back to its current behavior) — **surfaced, never silently substituted** (it must never fetch a
"nearest other species", which would make it a retrieval system and create an identity/contamination confound).
Unit-testable with a tiny fake store (no GPU, no FLUX): assert `global_id → record.medoid.image_path → PIL
Image`, and that a missing key returns `None`.

## 8. Build pipeline + artifacts

`ragregen/mmkg_store/build.py` — `build_treevill(kg_glob, manifest)`, `build_inat(manifest, val_json,
data_root, encoder, vlm)`, both emitting §4 records from **manifest-eligible images only**; `write_store(records,
out_dir)` persists records + FAISS + inverted index + the locked encoder sidecar. Contamination/dedup (§3) via
`ragregen.retrieved_reference` / `ragregen.eval_manifest`. Store under `/mmlabworkspace_new/.../mmkg_store/`.
A **build summary** (not a gate): species/dataset counts, hub sizes, attribute coverage, #medoids/#crops
indexed, and the eligible/excluded tallies from §3.

## 9. Reuse / tech / non-goals

Reuse: GRAFT `kg.json` + `graft/taxonomy.py`; `ragregen.mmkg.{schema, inat_pilot}`, `ragregen.encoders`,
`ragregen.retrieved_reference` / `eval_manifest`, FAISS. New code isolated under `ragregen/mmkg_store/`; tests
under `tests/mmkg_store/`. **Non-goals:** repair generator changes; open-world/cold-start (schema extensible,
not implemented or claimed); repair-time filtering/re-ranking/NN; a species repair eval; a graph DB.

## 10. Open items for planning

- Confirm the exact current reference-production seam the adapter parallels (where the pipeline hands `regen`
  its `reference` today) so the adapter is a true drop-in returning the same type.
- Treevill `kg.json` field mapping (medoid image path, crop paths, attribute shape, build-refs list) → §4.
- iNat manifest for the store: **exactly the 25 pilot birds** (locked, §5) — declare the eligible split and any
  eval-set to contamination-check against. Extension beyond 25 is a separate follow-on, not this MVP.
- Manifest file format (per-species eligible image list + optional declared eval set for the contamination
  check).
