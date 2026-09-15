# MMKG reference integration — design

**Date:** 2026-09-11
**Status:** design, approved for planning (brainstorm 2026-09-11)
**Scope:** wire the parked `mmkg-species-store` artifact into the rag-regen repair
pipeline as a *reference source*, and prove the plumbing with a CPU-only smoke
test. **Plumbing correctness only — no accuracy or fidelity claim.**

## 1. Goal and non-goals

**Goal.** Give the repair pipeline a second, deterministic way to obtain the
reference image it conditions on: an explicit species identity resolves through
the MMKG store to a precomputed medoid, which then feeds the *existing,
unmodified* repair stages. Prove end-to-end that

```
case.global_id → store.get() → precomputed medoid → PIL → existing repair
```

works, on CPU, deterministically, with no GPU and no FAISS.

**Non-goals (this work establishes none of these):**
- Retrieval / reference-selection *quality* (is the medoid a good reference?).
- Repair *fidelity* (does MMKG-referenced repair produce better images?).
- Any `concept → species_key` inference bridge (name-match or learned).
- Identity inferred from the generated image (the failed detector line).
- Re-validating the store itself (done; parked artifact is treated as finished).

## 2. Grounded facts about the current pipeline

- The reference enters repair as an in-memory PIL image at two sinks in
  `ragregen/regen.py`: `Inpainter.regen(draft, mask, reference, prompt, ...)`
  (line 385) and `stitch(draft, mask, cutout, ...)` (line 37).
- It is *sourced* three stages upstream:
  - **retrieve** — `_retrieve_one` (`scripts/run_pipeline.py:501`) runs
    `reference_query(concept, coarse)` → `Retriever.search()` (FAISS
    `IndexFlatIP`) → writes hit image paths to `refs.json`.
  - **mask** — `_mask_one` (`run_pipeline.py:427-453`) opens each ref, runs
    `mask_reference` + `prepare_reference` (crop to subject), saves
    `reference_i.png` / `cutout_i.png`.
  - **regen** — `_regen_one` (`run_pipeline.py:533`) opens
    `reference_{attempt}.png` and calls the sink.
- A case carries free-text `concept` and `coarse` only. **`species_key` /
  `global_id` appear nowhere outside `ragregen/mmkg_store/` and
  `ragregen/mmkg/`** (grep-confirmed). There is **no** `concept → species_key`
  bridge. This is the design gap; §3 resolves it by requiring an explicit key.
- The store adapter already exists: `mmkg_store.reference_source.medoid_reference
  (store, global_id)` (line 5) returns the medoid PIL (RGB) or `None`;
  `Store.get(global_id)` (`store.py:127`) returns the record or `None`; the
  record's `medoid.image_path` is the on-disk medoid.
- `global_id` format is `"<dataset>:<species_key>"` (`schema.py:4`).

## 3. Identity: explicit `global_id` on the case

The pipeline produces no species key, so the integration *supplies* one. Each
case in MMKG mode carries an explicit `global_id` string (e.g.
`"treevill:sp_00123"`). This is the honest form of an externally-supplied
identity: deterministic, auditable, zero inference.

- A name-match bridge (`concept`/`coarse` → `scientific_name`/`common_name`) is
  **out of scope**; it is itself a fuzzy matching step and belongs to a later
  study (see §8, H4).
- Identity inference from the draft is **out of scope** (revives the failed
  detector).

**No silent fallback.** If a case is in MMKG mode and its `global_id` is missing,
malformed, or absent from the store, the pipeline raises a clear error. It must
never fall back to FAISS retrieval or to a full ungrounded photo — that is the
exact silent path `_mask_one` already warns against (`run_pipeline.py:435-438`).

## 4. Integration seam: retrieve-stage reference swap

The mask and regen stages consume a reference path/PIL and are indifferent to
its origin, so the seam is the **retrieve stage**. In MMKG mode, the retrieve
step is *replaced* (not augmented) by a deterministic store lookup:

```
MMKG mode, per case:
  gid = case.global_id
  rec = store.get(gid)            # store.py:127; None → raise (§3)
  refs.json = [rec["medoid"]["image_path"]]   # single medoid, no FAISS
```

Everything downstream — `mask_reference`, `prepare_reference`, `stitch` /
`Inpainter.regen`, compositing — runs **byte-identically** to the FAISS path.
`prepare_reference` is *left in* for this work (it re-crops the medoid to its
subject); "skip prepare for medoids" is deferred (§8).

**Invariants:**
- MMKG mode calls only `Store.get` / `medoid_reference`. It never constructs a
  `Retriever`, reads a FAISS index, or calls `Store.nearest_crops`
  (`store.py:169`). No repair-time NN.
- No generator/inpainting internals change. `regen.py` is untouched.
- The run trace records the resolved `global_id` and `store.provenance(gid)` so
  each repair's reference is reconstructable.

### Data flow, before vs after

```
BEFORE:  concept+coarse → FAISS search → refs.json → mask/prepare → regen
AFTER :  case.global_id → store.get → medoid path → refs.json → mask/prepare → regen
                                                     └── (identical downstream) ──┘
```

## 5. Smoke test (the deliverable that proves §4)

CPU-only, deterministic, no GPU, no FLUX, no FAISS. Hermetic: builds a **fixture
store** in a temp dir via `mmkg_store.store.write_store`, so the test does **not**
depend on the parked artifact's on-disk paths.

**Fixture:** 2 synthetic records, `treevill:sp_A` and `treevill:sp_B`, each
`medoid.image_path` pointing at a distinct fixture PNG (e.g. solid-color 64×64).
A 128×128 draft PNG and a mask PNG with a white rectangle.

**Repair sink:** `stitch` (CPU, deterministic, no weights). A second assertion
uses `Inpainter(pipe=fake)` (injection supported at `regen.py:345`) with a fake
`pipe` that echoes a fixed image, to prove the reference *reaches*
`Inpainter.regen` without loading FLUX.

| Item | Expectation |
|---|---|
| Species key | `"treevill:sp_A"` (explicit on the case) |
| MMKG record | `store.get("treevill:sp_A")["medoid"]["image_path"]` == fixture-A path |
| Medoid | `medoid_reference(store, "treevill:sp_A")` → fixture-A PIL, RGB |
| Reference | the medoid (post `prepare_reference`) is what feeds the sink |
| Repair output | `stitch` `RegenResult`: `~mask` region bit-identical to draft (`regen.py:85`); masked region derived from fixture A |

**Assertions:**
1. `store.get(gid)` resolves and returns fixture-A's medoid path.
2. `medoid_reference` returns an RGB PIL matching fixture **A, not B** (identity
   discrimination — catches a "returns any record" bug).
3. `stitch` output outside the mask is bit-identical to the draft.
4. `stitch` output inside the mask is non-trivially changed.
5. **Negative control:** `medoid_reference(store, "treevill:MISSING")` → `None`,
   and MMKG mode raises rather than falling back; assert no `Retriever` / FAISS
   index is constructed in MMKG mode.
6. (inpaint) fake-`pipe` receives a non-None `image_reference` equal to the
   prepared medoid.

**Mocked:** FLUX pipe, any GPU. **Real:** store I/O, `medoid_reference`,
`stitch`, compositing, mask handling.

## 6. Baseline / control

Plumbing controls, not an evaluation:
- **Identity discrimination** — A selected, not B (assertion 2).
- **No silent fallback** — missing key raises; FAISS never touched (assertion 5).
- **Parity (optional)** — same draft+mask through FAISS path and MMKG path with a
  medoid set equal to a FAISS hit image; assert identical `RegenResult`, proving
  only the reference *source* changed.

No accuracy metric, cohort, or holdout.

## 7. Scientific boundary

| Layer | This work |
|---|---|
| Integration correctness | ✅ establishes |
| Infrastructure validation | reuses parked artifact; does not re-establish |
| Retrieval / reference-selection quality | ❌ cannot establish |
| Repair fidelity | ❌ cannot establish |

Given prior MMKG results (all verifier/detector holdouts FAILed), the boundary is
the point: success here means only "the pipe is connected."

## 8. Follow-up hypotheses (only if the smoke test passes) — deferred

- **H1 reproducibility:** explicit-key medoid makes the reference deterministic
  across runs vs FAISS. Baseline: FAISS arm. Experiment: hash `reference_i.png`
  across 2 runs, both arms. No GPU.
- **H2 non-inferiority:** a single medoid is no worse than top-1 FAISS as a
  reference. Baseline: `retrieved` arm. Experiment: paired fidelity metric on the
  frozen overlap. Allows a null result.
- **H3 provenance:** store-backed references record exact record + provenance per
  repair. Audit-only, no model run.
- **H4 the bridge is the bottleneck:** given a correct key, medoid repair is fine;
  producing the key is the open problem. Experiment: measure a name-match bridge's
  accuracy vs hand-labeled keys — string/embedding study, no repair.

Also deferred: "skip `prepare_reference` for medoids."

## 9. Risks / failure modes and mitigations

- **Namespace mismatch** (`concept` vs `global_id`) — explicit key; assert exact
  `dataset:species_key` format.
- **Identity ambiguity** — explicit key sidesteps it here.
- **Contamination leakage** — medoid byte-identical to a case's GT reference;
  reuse `retrieved_reference.contamination()` (SHA-256) as a gate before any real
  run (not needed for synthetic smoke test).
- **Accidental repair-time FAISS/NN** — assert `nearest_crops`/`Retriever`/index
  never touched in MMKG mode.
- **Hidden fallback** — `store.get` → `None` must raise, never drop to FAISS or a
  full photo.
- **Path mismatch** — real store medoid paths may not resolve on another host;
  fixture store for the test; verify resolution before any real run.
- **Split mismatch** — locked scope 36 TreeVill + 25 iNat; cases without a key are
  excluded explicitly, not defaulted.
- **Generator drift** — none intended; parity control guards it.
- **Coupling to the parked branch** — do not import/read the parked artifact in
  tests; work on a new branch off `main`, not on `mmkg-species-store` (must not
  merge).
- **Reproducibility/provenance** — record `global_id` + provenance in the trace.

## 10. Boundaries (hard)

- Treat the parked MMKG store as finished; do not expand its locked scope or
  modify its frozen rules.
- Do not modify generator/inpainting internals (`regen.py` sinks unchanged).
- No new accuracy metric or large evaluation.
- No GPU experiment.
- Do not build on / merge the `mmkg-species-store` branch.
