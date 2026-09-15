# MMKG Structured Species Reference Store + Adapter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build a closed-world, typed **reference registry** over Treevill + the 25 iNat pilot birds (per species: precomputed medoid + candidates + typed part-crops + verified attributes + taxonomy + provenance), a lightweight query API, and a thin rag-regen reference-source adapter that returns the precomputed medoid image by species key — no repair-time NN/FAISS/filtering, generator unchanged.

**Architecture:** A new isolated package `ragregen/mmkg_store/`. Pure decision/shape logic (schema, manifest/eligibility, store I/O, the adapter) is unit-tested with fakes; the FAISS index and the two dataset build adapters sit behind encoder/VLM seams so everything but the live build is offline-testable. The store persists per-species JSON records + one FAISS index (`IndexFlatIP`, locked config) + an inverted index.

**Tech stack:** Python 3.11 (`/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python`), numpy, faiss, Pillow; reuse `ragregen.encoders` (SigLIP), `ragregen.mmkg.{schema,inat_pilot}`, `ragregen.retrieved_reference`, `ragregen.eval_manifest`. Spec (frozen): `docs/superpowers/specs/2026-09-10-mmkg-species-store-design.md`.

## Global Constraints

- **Spec is frozen** — copy every value verbatim; do not change a rule or threshold.
- **Git:** rag-regen real repo. Before Task 1: `git checkout -b mmkg-species-store` off the current `mmkg-verifier-stage1` (it carries `ragregen.mmkg.{schema,inat_pilot}` this plan reuses). The tree has **pre-existing unrelated uncommitted changes** — every commit `git add` **only the files the task names**, never `-A`/`.`.
- **Locked embedding config (asserted on store load):** encoder `siglip_so400m_384`, vectors **L2-normalized**, metric **inner-product**, dim **1152**, FAISS **`IndexFlatIP`**. Stored in a sidecar; a record's `embedding_ref` is an **opaque build-local integer id** (current impl = FAISS id), never assume it *is* the FAISS id outside the sidecar.
- **Identity:** `global_id = "<dataset>:<species_key>"`. Hub ids **dataset-namespaced**: `"<dataset>:genus:<genus>"`, `"<dataset>:family:<family>"`.
- **Eligibility/contamination (BLOCKING):** build consumes an explicit per-species **manifest** of eligible images; build uses **only** those; any declared eval/query set is asserted **disjoint** and SHA-256 contamination-checked (`retrieved_reference.contamination`); dedup by `eval_manifest.sha256_file`. Provenance records `source_split, build_manifest(+sha256), eligible_images, excluded_images(reason)`.
- **Repair adapter:** `get(global_id) → record.medoid.image_path → PIL Image`. **No** FAISS, **no** `nearest_crops`/`query`, **no** NN, **no** filtering, **no** attribute text, **no** multi-crop. Missing species → return `None` (never a nearest-other-species). Generator (`regen`) unchanged.
- **iNat scope LOCKED to the 25 pilot birds** (`inat_pilot.select_pilot_species`); do **not** auto-expand.
- **Treevill embeddings:** reuse `kg.json` for structure/attributes/taxonomy/paths only; **RE-EMBED** images with `siglip_so400m_384` (do NOT reuse kg.json's SigLIP2-768 vectors — the index must be single-encoder).
- **Attributes:** reuse the frozen consensus (`ragregen.mmkg.schema`); persist one `value` per slot with integer `support`/`visible_count`; introduce no new thresholds/judges/multi-value.
- **Env/test:** `PYTHONPATH=. /mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest -q`.

## File Structure

- `ragregen/mmkg_store/__init__.py`
- `schema.py` — record shape, id/hub helpers, validation, attribute reshape.
- `manifest.py` — manifest load + eligibility + contamination/dedup.
- `embed_index.py` — FAISS build/load + sidecar + locked-config guard.
- `store.py` — persistence + query API + inverted index.
- `reference_source.py` — the repair adapter.
- `build_treevill.py`, `build_inat.py` — dataset adapters.
- `build.py` — orchestrator + summary + `main`.
- tests under `tests/mmkg_store/`.

---

### Task 1: schema — record shape, id/hub helpers, attribute reshape

**Files:** Create `ragregen/mmkg_store/__init__.py` (empty), `ragregen/mmkg_store/schema.py`, `tests/mmkg_store/__init__.py` (empty), `tests/mmkg_store/test_schema.py`.

**Interfaces (produces):**
- `global_id(dataset, species_key) -> str` = `f"{dataset}:{species_key}"`; `hub_id(dataset, rank, name) -> str` = `f"{dataset}:{rank}:{name}"` (rank ∈ {"genus","family"}).
- **Attribute record shape (uniform):** `{value: str, support: int|null, visible_count: int|null, source: str}`. `support`/`visible_count` are integer **image counts** when a real per-image consensus exists (iNat), and **`null`** when the source provides only a pre-consensused value with no image-level statistics (Treevill kg.json — verified: it stores `{name,value,source}` only, no counts). Fabricating counts is forbidden.
- `attribute_record(consensus, source) -> dict` — takes `bird_target_attributes` output `{slot:{value,support,visible_count,is_target}}`, returns `{slot:{value,support,visible_count,source}}` for `is_target` slots only (int counts; iNat path). Treevill builds records **directly** with `support=None, visible_count=None` (Task 6), not via this function.
- `build_record(*, dataset, species_key, scientific_name, common_name, taxonomy, medoid, candidates, part_crops, attributes, provenance) -> dict` — assembles the §4 record with `global_id`, `relations` (namespaced `instance_of` genus+family from `taxonomy`), and validates required keys (raises `ValueError` on missing `medoid.image_path` or empty `dataset`/`species_key`). `common_name` may be `None`.
- `REQUIRED_MEDOID_KEYS = {"image_path","embedding_ref","k_images","selection"}`.

- [ ] **Step 1: failing test**
```python
# tests/mmkg_store/test_schema.py
from ragregen.mmkg_store import schema as s
def test_ids_namespaced():
    assert s.global_id("inat","04486") == "inat:04486"
    assert s.hub_id("treevill","genus","Certhia") == "treevill:genus:Certhia"
def test_attribute_record_targets_only():
    cons = {"bark.texture":{"value":"rough","support":4,"visible_count":5,"is_target":True},
            "leaf.shape":{"value":"ovate","support":2,"visible_count":3,"is_target":False}}
    out = s.attribute_record(cons, source="inat-consensus")
    assert out == {"bark.texture":{"value":"rough","support":4,"visible_count":5,"source":"inat-consensus"}}
def test_build_record_relations_and_validation():
    rec = s.build_record(dataset="inat", species_key="04486", scientific_name="Phoebastria immutabilis",
        common_name=None, taxonomy={"family":"Diomedeidae","genus":"Phoebastria"},
        medoid={"image_path":"m.jpg","embedding_ref":0,"k_images":7,"selection":"siglip-centroid-nearest"},
        candidates=[{"image_path":"m.jpg","embedding_ref":0}], part_crops=[], attributes={}, provenance={})
    assert rec["global_id"] == "inat:04486" and rec["common_name"] is None
    assert {"type":"instance_of","target":"inat:family:Diomedeidae"} in rec["relations"]
    assert {"type":"instance_of","target":"inat:genus:Phoebastria"} in rec["relations"]
    import pytest
    with pytest.raises(ValueError):
        s.build_record(dataset="inat", species_key="x", scientific_name="", common_name=None,
            taxonomy={"family":"F","genus":"G"}, medoid={"image_path":""}, candidates=[], part_crops=[],
            attributes={}, provenance={})
```
- [ ] **Step 2: run → FAIL** (`PYTHONPATH=. .../python -m pytest tests/mmkg_store/test_schema.py -q`).
- [ ] **Step 3: implement** `schema.py` per the interfaces (pure; no external deps beyond stdlib).
- [ ] **Step 4: run → PASS.**
- [ ] **Step 5: commit** `git add ragregen/mmkg_store/__init__.py ragregen/mmkg_store/schema.py tests/mmkg_store/__init__.py tests/mmkg_store/test_schema.py` → `feat(mmkg_store): record shape, namespaced ids, attribute reshape`.

---

### Task 2: manifest — eligibility + contamination + dedup (BLOCKING policy)

**Files:** Create `ragregen/mmkg_store/manifest.py`, `tests/mmkg_store/test_manifest.py`.

**Interfaces:**
- `load_manifest(path) -> dict[str, dict]` — JSON: `{species_key: {"eligible": [img paths], "eval_set": [paths]?}}`; validates lists exist.
- `eligible_pool(entry, gt_eval_paths=()) -> tuple[list[str], list[dict]]` — returns `(kept, excluded)`: drop sha256 duplicates (`eval_manifest.sha256_file`), drop any path where `retrieved_reference.contamination(path, gt_eval_paths)` is True, and **assert** none of `kept` is in `entry.get("eval_set", [])` (raise `ValueError` if the manifest itself lists an eligible image in its own eval_set). `excluded` entries = `{"path","reason"}` with reason ∈ {"duplicate","contamination"}.
- `provenance_block(entry, manifest_path, kept, excluded) -> dict` — `{"source_split","build_manifest":{"path","sha256"},"eligible_images":kept,"excluded_images":excluded}`.

- [ ] **Step 1: failing test** (real temp files, mirroring the proven guard test):
```python
# tests/mmkg_store/test_manifest.py
from ragregen.mmkg_store import manifest as m
from PIL import Image
def test_eligible_pool_drops_dup_and_contaminated(tmp_path):
    keep = tmp_path/"a.jpg"; Image.new("RGB",(8,8),"red").save(keep)
    dup = tmp_path/"b.jpg"; dup.write_bytes(keep.read_bytes())          # sha dup -> dropped
    contam = tmp_path/"c.jpg"; Image.new("RGB",(8,8),"blue").save(contam)
    gt = tmp_path/"gt.jpg"; gt.write_bytes(contam.read_bytes())         # eval ref == contam
    kept, excl = m.eligible_pool({"eligible":[str(keep),str(dup),str(contam)]}, gt_eval_paths=[str(gt)])
    assert kept == [str(keep)]
    assert {e["reason"] for e in excl} == {"duplicate","contamination"}
def test_manifest_rejects_eligible_in_eval(tmp_path):
    p = tmp_path/"a.jpg"; Image.new("RGB",(8,8),"red").save(p)
    import pytest
    with pytest.raises(ValueError):
        m.eligible_pool({"eligible":[str(p)], "eval_set":[str(p)]})
```
- [ ] **Step 2: run → FAIL.**
- [ ] **Step 3: implement** using `ragregen.eval_manifest.sha256_file` + `ragregen.retrieved_reference.contamination` (drop-on-unhashable too, per the Stage-1 lesson).
- [ ] **Step 4: run → PASS.**
- [ ] **Step 5: commit** the two files → `feat(mmkg_store): manifest eligibility + contamination/dedup guard`.

---

### Task 3: embed_index — FAISS IndexFlatIP + sidecar + locked-config guard

**Files:** Create `ragregen/mmkg_store/embed_index.py`, `tests/mmkg_store/test_embed_index.py`.

**Interfaces:**
- `LOCKED = {"encoder":"siglip_so400m_384","dim":1152,"normalized":True,"metric":"ip","index":"IndexFlatIP"}`.
- `IndexBuilder` — `add(vec, meta) -> int` (L2-normalizes `vec`, appends to the flat IP index, stores `meta={global_id,kind,path}` at the returned integer `embedding_ref`); `save(out_dir)` writes `index.faiss` + `sidecar.json` (`{"config":LOCKED,"entries":{ref:meta}}`).
- `load_index(dir) -> (faiss_index, sidecar)` — **asserts** `sidecar["config"] == LOCKED` (raise `ValueError` on mismatch).
- `search(faiss_index, sidecar, vec, k, kind=None, allow=None) -> list[(embedding_ref, score, meta)]` — normalize query, IP search, filter by `kind`/`allow` (a set of global_ids) post-hoc.

- [ ] **Step 1: failing test** (uses faiss; kontext env has it — mark the test module import-guarded so non-faiss envs skip):
```python
# tests/mmkg_store/test_embed_index.py
import numpy as np, pytest
faiss = pytest.importorskip("faiss")
from ragregen.mmkg_store import embed_index as ei
def test_add_search_roundtrip_and_config_guard(tmp_path):
    b = ei.IndexBuilder()
    r0 = b.add(np.array([1.,0.],dtype="float32"), {"global_id":"d:x","kind":"medoid","path":"x.jpg"})
    r1 = b.add(np.array([0.,1.],dtype="float32"), {"global_id":"d:y","kind":"medoid","path":"y.jpg"})
    # (dim guard is checked on real 1152 vectors in the live build; here we assert id + meta wiring)
    b.save(str(tmp_path))
    idx, side = ei.load_index(str(tmp_path))
    hits = ei.search(idx, side, np.array([1.,0.],dtype="float32"), k=1)
    assert hits[0][0] == r0 and hits[0][2]["global_id"] == "d:x"
    # config guard
    import json; s=json.load(open(tmp_path/"sidecar.json")); s["config"]["encoder"]="other"; json.dump(s,open(tmp_path/"sidecar.json","w"))
    with pytest.raises(ValueError): ei.load_index(str(tmp_path))
```
> Implementer note: the toy test uses dim-2 vectors to exercise id/meta/guard wiring without a GPU encoder; the real store enforces dim 1152 at build (Task 8). Keep `IndexBuilder` dim-agnostic (first `add` fixes the dim); assert all later `add`s match it.
- [ ] **Step 2: run → FAIL.**   - [ ] **Step 3: implement.**   - [ ] **Step 4: run → PASS.**
- [ ] **Step 5: commit** → `feat(mmkg_store): faiss index + sidecar + locked-config guard`.

---

### Task 4: store — persistence + query API + inverted index

**Files:** Create `ragregen/mmkg_store/store.py`, `tests/mmkg_store/test_store.py`.

**Interfaces:**
- `write_store(records, out_dir, index_builder=None)` — writes `store/<dataset>/<species_key>.json` per record, builds the inverted index (`inverted.json`: `attributes {slot|value→[global_id]}`, `hubs {hub_id→[global_id]}`, `parts {part_type→[embedding_ref]}`), and saves the FAISS index if a builder is given.
- `Store.load(dir) -> Store`; `get(global_id) -> dict|None`; `query(hub=None, attribute=None, part_type=None) -> list[str]` (intersection of inverted-index hits); `members(hub_id) -> list[str]`; `provenance(global_id) -> dict`; `nearest_crops(vec, part_type=None, hub=None, k=5)` (uses `embed_index.search`, filtered).

- [ ] **Step 1: failing test** — build 3 synthetic records (2 same hub, differing attributes), `write_store` (no index), `Store.load`, assert `get`, `query(hub=…)` returns the 2, `query(attribute=("bark.texture","rough"))` filters, `members`, `provenance`. (No faiss needed if `nearest_crops` untested here; test it in a separate faiss-guarded case.)
- [ ] **Step 2: FAIL.**  - [ ] **Step 3: implement.**  - [ ] **Step 4: PASS.**
- [ ] **Step 5: commit** → `feat(mmkg_store): persistence + filterable query API + inverted index`.

---

### Task 5: reference_source — the repair adapter (guardrail-critical)

**Files:** Create `ragregen/mmkg_store/reference_source.py`, `tests/mmkg_store/test_reference_source.py`.

**Interfaces:**
- `medoid_reference(store, global_id) -> PIL.Image | None` — `rec = store.get(global_id)`; if `None` → return `None`; else open `rec["medoid"]["image_path"]` as RGB `Image` and return it. **Uses only `store.get`.** No FAISS, no `nearest_crops`, no filtering, no fallback to another species.

- [ ] **Step 1: failing test**
```python
# tests/mmkg_store/test_reference_source.py
from ragregen.mmkg_store import reference_source as rs
from PIL import Image
class FakeStore:
    def __init__(self, recs): self.recs=recs; self.calls=[]
    def get(self, gid): self.calls.append(gid); return self.recs.get(gid)
    def nearest_crops(self,*a,**k): raise AssertionError("adapter must not call nearest_crops")
def test_returns_medoid_image_and_none(tmp_path):
    m = tmp_path/"m.jpg"; Image.new("RGB",(8,8),"green").save(m)
    st = FakeStore({"inat:04486":{"medoid":{"image_path":str(m)}}})
    img = rs.medoid_reference(st, "inat:04486")
    assert img is not None and img.size == (8,8) and img.mode == "RGB"
    assert rs.medoid_reference(st, "inat:missing") is None      # never a fallback species
    assert st.calls == ["inat:04486","inat:missing"]            # only get(); nearest_crops would AssertionError
```
- [ ] **Step 2: FAIL.**  - [ ] **Step 3: implement** (≤10 lines).  - [ ] **Step 4: PASS.**
- [ ] **Step 5: commit** → `feat(mmkg_store): repair reference adapter (precomputed medoid, no NN/FAISS)`.

---

### Task 6: build_treevill — kg.json → record (re-embed with the store encoder)

**Files:** Create `ragregen/mmkg_store/build_treevill.py`, `tests/mmkg_store/test_build_treevill.py`.

**Interfaces:**
- `load_taxonomy(taxonomy_py="…/kg_test/graft/taxonomy.py") -> dict[str,(sci,family)]` — read the `TAXONOMY` dict literal (via `ast.literal_eval` of the assigned dict, or `runpy`), so no cross-repo import.
- `treevill_record(concept, kg, tax, eligible_paths, embed_fn, index_add) -> dict` — from a loaded `kg.json` dict + taxonomy: `scientific_name=tax[concept][0]`, `genus=sci.split()[0]`, `family=tax[concept][1]`, `common_name=concept`; candidates = `eligible_paths` (from the manifest = kg's build refs); **medoid** = `embed_fn(eligible_paths)` → centroid → nearest path; part_crops from `kg["parts"][*]["exemplar_crop"]` (+ `embed_fn` of the crop) typed by `part["name"]`; attributes built **directly** from the parts' `attributes` (see note); `index_add(vec, meta)` returns the `embedding_ref` for medoid/candidates/crops. Uses `mmkg_store.schema.build_record`.

> Implementer note (VERIFIED against real `kg.json`): each part attribute is `{name, value, source}` — a single GRAFT consensus value with **no image-level counts**. So build the attribute record **directly**, keyed `"<part>.<name>"`, `value=attr["value"]`, `support=None`, `visible_count=None`, `source=attr.get("source","graft-kg")`; **drop** attributes whose normalized value ∈ NOT_VISIBLE. Do **NOT** run them through `schema.attribute_record` and do **NOT** fabricate `1/1` counts — null counts honestly record that kg.json carries no consensus statistics. Do NOT re-run any VLM. Medoid/crop **embeddings are recomputed** with the store encoder (`embed_fn`); kg's own SigLIP2-768 embeddings are ignored.

- [ ] **Step 1: failing test** — synthetic `kg` dict (2 parts, exemplar_crop paths, attributes incl. a "not visible") + a fake `tax` + fake `embed_fn` (returns fixed vectors so the medoid is deterministic) + a fake `index_add`; assert the record has namespaced relations, medoid = the centroid-nearest eligible path, part_crops typed, attributes drop the not-visible one and carry `support is None`/`visible_count is None`/`source` (never fabricated counts), common_name==concept, scientific/genus/family correct.
- [ ] **Step 2: FAIL.**  - [ ] **Step 3: implement** (pure over `embed_fn`/`index_add` seams).  - [ ] **Step 4: PASS.**
- [ ] **Step 5: commit** → `feat(mmkg_store): treevill adapter (re-embed, medoid, crops, attrs, taxonomy)`.

---

### Task 7: build_inat — 25 pilot birds → record

**Files:** Create `ragregen/mmkg_store/build_inat.py`, `tests/mmkg_store/test_build_inat.py`.

**Interfaces:**
- `inat_records(val_json, data_root, manifest, embed_fn, read_fn, index_add) -> list[dict]` — for each species in `inat_pilot.select_pilot_species(val_json)` (LOCKED 25): eligible = `manifest[species_key]["eligible"]` (⊆ `inat_pilot.species_image_paths`); **medoid** = `embed_fn(eligible)` centroid-nearest; attributes = `schema.attribute_record(inat_pilot.bird_target_attributes([read_fn(p) for p in eligible]), source="inat-consensus")`; taxonomy from the val category (`family`, `genus`); part_crops = `[]` (MVP); `common_name` from the category (may be `None`). Uses `mmkg_store.schema.build_record`, dataset `"inat"`, `species_key = zero-padded category id` (matches `image_dir_name` prefix).
- [ ] **Step 1: failing test** — **monkeypatch `inat_pilot.select_pilot_species`** to return 3 small synthetic species dicts (exercise the production selection *seam* without conflicting with the locked-25 production scope — do NOT stuff 25 fakes into a fixture); fake `manifest`/`embed_fn`/`read_fn`/`index_add`; assert 3 records, medoid deterministic, bird attributes present with int counts + `source=="inat-consensus"`, `part_crops == []`, `dataset=="inat"`, relations namespaced `inat:…`.
- [ ] **Step 2: FAIL.**  - [ ] **Step 3: implement.**  - [ ] **Step 4: PASS.**
- [ ] **Step 5: commit** → `feat(mmkg_store): inat birds adapter (25 pilot species, medoid+attrs)`.

---

### Task 8: build orchestrator + offline smoke + live build + summary

**Files:** Create `ragregen/mmkg_store/build.py`, `scripts/mmkg_store_build.sh`, `tests/mmkg_store/test_build_smoke.py`; run artifacts under `/mmlabworkspace_new/.../mmkg_store/` + `docs/superpowers/2026-09-10-mmkg-store-build-summary.md`.

**Interfaces:** `main(argv=None, *, encoder_factory=None, vlm_factory=None, …)` with seams; flow: build the shared `IndexBuilder`; `build_treevill` over `kg_test/outputs/*/kg.json` + Treevill manifest; `build_inat` over the 25-bird manifest; `write_store`; write a **build summary** — **infrastructure validation only**, answering "was the store built correctly?": species/dataset counts, hub sizes, attribute coverage, #medoids/#crops indexed, eligible/excluded tallies. **NO** fidelity/DINO/CLIP/repair-success/NN-baseline metrics — that is a separate experiment and out of scope.

- [ ] **Step 1:** write `build.py` + the offline **smoke test** (`test_build_smoke.py`): monkeypatch `encoder_factory`/`vlm_factory`/read/kg-glob with fakes over 1 tree + 2 birds; run `main([...])`; assert store records + `inverted.json` + sidecar written and `reference_source.medoid_reference(Store.load(out), gid)` returns an Image. Assert `import ragregen.mmkg_store.build` does **not instantiate or load** the encoder or VLM (model loading is lazy, behind the `*_factory` seams) — the check is about model **loading**, not module imports (a module-level `faiss` import is fine; keep encoder/VLM instantiation lazy, e.g. inside the default factories and `IndexBuilder`/`load`). Commit.
- [ ] **Step 2:** **Manifests.** Generate the Treevill manifest (each concept's GRAFT build refs from `kg.json`/`split_refs`) and the iNat manifest (the 25 pilot birds' eligible `val` images; declare any eval_set to exclude). Commit the manifest builder + the manifest files (small JSON).
- [ ] **Step 3 (controller-driven live build):** run detached (setsid + swap, per the Stage-1 memory-guard lesson):
  `setsid bash scripts/mmkg_store_build.sh > outputs/mmkg_store_build.log 2>&1 < /dev/null &` — one encoder load (+VLM for iNat attrs), re-embed Treevill refs + iNat medoids, write the store to bulk storage. Watch for the summary; confirm no traceback.
- [ ] **Step 4:** verify the store: `Store.load` asserts the locked config; spot-check a Treevill and an iNat record (`get`), `query(hub=…)`, and `reference_source.medoid_reference` returns a real medoid Image for both datasets.
- [ ] **Step 5:** write `docs/superpowers/2026-09-10-mmkg-store-build-summary.md` (counts, hub sizes, attribute coverage, eligible/excluded tallies, the locked config, and the adapter contract). Update memory (`mmkg-to-ragregen-direction`) + ledger. Commit the summary.

---

## Self-Review

**Spec coverage:** §3 eligibility/contamination → Task 2 + Task 8 manifests. §4 schema/ids/relations/attributes → Task 1 (+ adapters 6/7 assemble via `build_record`). §5 adapters → Tasks 6/7 (Treevill re-embed noted). §6 persistence/query/locked-index/embedding_ref → Tasks 3+4. §7 repair adapter (no NN/FAISS/fallback) → Task 5. §8 build+summary → Task 8. Non-goals (no generator change, no repair-time NN/filter, iNat 25 locked, no eval) → Global Constraints + Task 5/7 tests. **No gaps.**

**Placeholder scan:** none — pure tasks carry test+impl contracts; build tasks carry concrete step lists and seams. Live build (Task 8 steps 3–5) is controller-driven, each step a concrete command/artifact.

**Type consistency:** `target_attributes`/`bird_target_attributes` output shape → `schema.attribute_record` (Tasks 6/7 consume it) → `build_record.attributes`. `IndexBuilder.add → embedding_ref (int)` used by adapters and stored in records + sidecar; `load_index` asserts `LOCKED`. `Store.get` return consumed by `reference_source.medoid_reference` (keys `medoid.image_path`). `global_id`/`hub_id` produced in Task 1, used across store/inverted-index/adapters. Manifest `eligible` list → adapters' `eligible_paths`. Encoder `siglip_so400m_384` single-sourced in `embed_index.LOCKED` + build seam.
