# MMKG Reference Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the repair pipeline obtain its reference image from the MMKG store via an explicit species `global_id`, and prove the plumbing with a CPU-only smoke test — no fidelity claim.

**Architecture:** Add an optional `global_id` to a `Case`; add a `--reference-source mmkg` mode to `run_pipeline` that, per case, resolves `store.get(global_id)` and writes the precomputed medoid path into `refs.json` (skipping FAISS). Everything downstream (mask, prepare, stitch/inpaint) is unchanged. A standalone smoke test proves `key → store → medoid → repair` on CPU.

**Tech Stack:** Python, PIL, numpy, pytest. Existing modules: `ragregen.mmkg_store.{store,schema,reference_source}`, `ragregen.regen`, `ragregen.config`, `scripts/run_pipeline.py`.

**Spec:** `docs/superpowers/specs/2026-09-11-mmkg-reference-integration-design.md`

## Global Constraints

- Plumbing correctness ONLY — no accuracy/fidelity metric, no cohort, no holdout.
- No GPU, no FLUX load, no FAISS in MMKG mode. Never call `Store.nearest_crops` / build a `Retriever` in MMKG mode.
- No silent fallback: a missing/malformed `global_id` in MMKG mode raises; never fall back to FAISS or a full photo.
- Do not modify generator/inpainting internals: `ragregen/regen.py` is read-only for this plan.
- `global_id` format is exactly `"<dataset>:<species_key>"` (`ragregen/mmkg_store/schema.py:4`).
- Work on a NEW branch off `main` (e.g. `mmkg-reference-integration`). Do NOT build on or merge `mmkg-species-store`. Do not import/read the parked artifact in tests — build fixture stores.
- **Committing:** the plan uses per-task commits, but the user gates the first commit given the parked-branch context — confirm the branch before committing.

---

### Task 1: Adapter → repair-sink contract smoke test (CPU, no pipeline changes)

Proves `global_id → store.get → medoid → PIL → stitch/inpaint` against the *existing* code, composed together. These tests exercise code that already exists (`medoid_reference`, `Store`, `regen.stitch`, `Inpainter` with an injected pipe), so they should pass on first run — that passing IS the integration proof. A failure here is a real contract gap, not a red-green step.

**Files:**
- Create: `tests/test_mmkg_reference_integration.py`
- Read-only refs: `ragregen/mmkg_store/reference_source.py:5`, `ragregen/mmkg_store/store.py:127`, `ragregen/regen.py:37,345,385`

**Interfaces:**
- Consumes: `schema.build_record(...)`, `store.write_store(records, out_dir)`, `Store.load(dir)`, `reference_source.medoid_reference(store, global_id)`, `regen.stitch(draft, mask, cutout, ...)`, `regen.Inpainter(pipe=...)`.
- Produces: nothing importable; this is a leaf test.

- [ ] **Step 1: Write the smoke test file**

```python
"""Smoke test: species key -> store -> precomputed medoid -> existing repair.

CPU-only, deterministic, no GPU/FLUX/FAISS. Proves the MMKG reference-source
contract end to end. Spec:
docs/superpowers/specs/2026-09-11-mmkg-reference-integration-design.md
"""
import numpy as np
from PIL import Image

from ragregen import regen
from ragregen.mmkg_store import reference_source as rs
from ragregen.mmkg_store import schema as s
from ragregen.mmkg_store import store as st


def _fixture_store(tmp_path):
    """A 2-species store on disk; medoids are distinct solid-colour PNGs."""
    a = tmp_path / "sp_A_medoid.png"
    b = tmp_path / "sp_B_medoid.png"
    Image.new("RGB", (64, 64), (200, 0, 0)).save(a)   # red
    Image.new("RGB", (64, 64), (0, 0, 200)).save(b)   # blue

    def rec(key, path):
        return s.build_record(
            dataset="treevill", species_key=key,
            scientific_name=f"Genus {key}", common_name=key,
            taxonomy={"genus": "Genus", "family": "Fam"},
            medoid={"image_path": str(path), "embedding_ref": 0,
                    "k_images": 1, "selection": "siglip-centroid-nearest"},
            candidates=[], part_crops=[], attributes={},
            provenance={"source_split": "build"},
        )

    out = tmp_path / "store_out"
    st.write_store([rec("sp_A", a), rec("sp_B", b)], out)
    return st.Store.load(out)


def _draft(size=(64, 64)):
    return Image.new("RGB", size, (10, 20, 30))


def _mask(size=(64, 64), box=(16, 16, 48, 48)):
    m = Image.new("L", size, 0)
    m.paste(255, box)
    return m


def test_key_selects_the_right_medoid_not_any(tmp_path):
    store = _fixture_store(tmp_path)
    img = rs.medoid_reference(store, "treevill:sp_A")
    assert img is not None and img.mode == "RGB"
    assert np.asarray(img)[0, 0].tolist() == [200, 0, 0]        # A is red
    other = rs.medoid_reference(store, "treevill:sp_B")
    assert np.asarray(other)[0, 0].tolist() == [0, 0, 200]      # B is blue


def test_missing_key_returns_none_never_a_fallback(tmp_path):
    store = _fixture_store(tmp_path)
    assert rs.medoid_reference(store, "treevill:MISSING") is None


def test_no_faiss_index_persisted_so_no_repair_time_nn(tmp_path):
    store = _fixture_store(tmp_path)
    # write_store received no index_builder, so nearest_crops is inert.
    assert store.nearest_crops([0.0], k=5) == []


def test_medoid_feeds_stitch_and_preserves_outside_the_mask(tmp_path):
    store = _fixture_store(tmp_path)
    ref = rs.medoid_reference(store, "treevill:sp_A").convert("RGBA")
    draft, mask = _draft(), _mask()
    result = regen.stitch(draft, mask, ref, feather_px=0, fill_residual=False)
    assert result.mechanism == "stitch"
    d, out = np.asarray(draft), np.asarray(result.image)
    m = np.asarray(mask) > 127
    assert np.array_equal(out[~m], d[~m])          # outside mask: bit-identical
    assert not np.array_equal(out[m], d[m])        # inside mask: changed


def test_medoid_reaches_inpainter_without_loading_flux(tmp_path):
    store = _fixture_store(tmp_path)
    ref = rs.medoid_reference(store, "treevill:sp_A")
    seen = {}

    class FakePipe:
        def set_progress_bar_config(self, **k):
            pass

        def __call__(self, **call):
            seen.update(call)

            class R:
                images = [Image.new("RGB", call["image"].size, (5, 5, 5))]

            return R()

    eng = regen.Inpainter(pipe=FakePipe())
    eng.regen(_draft(), _mask(), ref,
              "Replace only the tree with a Genus sp_A.")
    assert seen["image_reference"] is not None
    assert seen["image_reference"].size == ref.size
```

- [ ] **Step 2: Run the smoke test**

Run: `pytest tests/test_mmkg_reference_integration.py -v`
Expected: all 5 tests PASS. (If any FAIL, stop — it is a real contract gap; do not patch `regen.py`; report it.)

- [ ] **Step 3: Commit** (confirm branch first — see Global Constraints)

```bash
git add tests/test_mmkg_reference_integration.py
git commit -m "test(mmkg): CPU smoke test for species-key -> medoid -> repair contract"
```

---

### Task 2: Add optional `global_id` to `Case` and parse it

**Files:**
- Modify: `ragregen/config.py:22` (Case dataclass), `ragregen/config.py:149` (load_dataset case construction)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: existing `load_dataset(path)` / `Case`.
- Produces: `Case.global_id: str | None` (default `None`); when present in YAML it is validated to match `"<dataset>:<species_key>"` (a non-empty prefix, a single `:`, a non-empty suffix).

- [ ] **Step 1: Write the failing test**

```python
# in tests/test_config.py
import pytest
from ragregen import config


def _dataset_yaml(tmp_path, extra_case_lines=""):
    p = tmp_path / "ds.yaml"
    p.write_text(
        "name: t\n"
        f"images_root: {tmp_path}\n"
        "cases:\n"
        "  - id: c1\n"
        "    prompt: a photo of a tree\n"
        "    concept: oak\n"
        "    coarse: tree\n"
        "    gt_refs: [r.jpg]\n"
        f"{extra_case_lines}"
    )
    (tmp_path / "r.jpg").write_bytes(b"x")
    return p


def test_global_id_defaults_to_none(tmp_path):
    ds = config.load_dataset(_dataset_yaml(tmp_path))
    assert ds.cases[0].global_id is None


def test_global_id_is_parsed_when_present(tmp_path):
    p = _dataset_yaml(tmp_path, "    global_id: treevill:sp_00123\n")
    ds = config.load_dataset(p)
    assert ds.cases[0].global_id == "treevill:sp_00123"


def test_malformed_global_id_raises(tmp_path):
    p = _dataset_yaml(tmp_path, "    global_id: no_colon_here\n")
    with pytest.raises(ValueError, match="global_id"):
        config.load_dataset(p)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_config.py -k global_id -v`
Expected: FAIL (`Case` has no `global_id` / not parsed).

- [ ] **Step 3: Add the field and parsing**

In `ragregen/config.py`, add to `Case` (after `gt_refs`, line 37):

```python
    #: MMKG species identity, "<dataset>:<species_key>". None unless the case
    #: is authored for the mmkg reference source. No concept->key inference
    #: exists; this is supplied explicitly (spec 2026-09-11).
    global_id: str | None = None
```

In `load_dataset`, before `cases.append(Case(`, add:

```python
        gid = raw.get("global_id")
        if gid is not None:
            parts = str(gid).split(":")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(
                    f"{path}: case {cid!r} has global_id {gid!r}; expected "
                    f"'<dataset>:<species_key>' with both parts non-empty.")
```

and pass it in the `Case(...)` call:

```python
            gt_refs=[root / r for r in refs],
            global_id=gid,
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (all config tests, including the 3 new ones).

- [ ] **Step 5: Commit**

```bash
git add ragregen/config.py tests/test_config.py
git commit -m "feat(config): optional global_id on Case for mmkg reference source"
```

---

### Task 3: `--reference-source mmkg` mode in run_pipeline

Replaces the FAISS retrieve step, per case, with a deterministic store lookup that writes the medoid path into `refs.json`. A small pure helper carries the logic so it is unit-testable without the scheduler.

**Files:**
- Modify: `scripts/run_pipeline.py` — argparse (near line 148-162), the retrieve stage dispatch (near line 310-328), and add helper `_reference_mmkg_one`.
- Test: `tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: `Case.global_id` (Task 2), `mmkg_store.store.Store.load`, `mmkg_store.reference_source.medoid_reference`, the existing `run_dir.case_dir(cid)` and `refs.json` contract (`run_pipeline.py:504`).
- Produces: `_reference_mmkg_one(store, run_dir, case) -> None` — writes `refs.json` (a single-element list: the medoid `image_path`) and `retrieval.json` (`{"source": "mmkg", "global_id": ..., "medoid_path": ...}`); raises `ValueError` if `case.global_id` is falsy or absent from the store. Never constructs a `Retriever`.

- [ ] **Step 1: Write the failing test**

```python
# in tests/test_run_pipeline.py
import json
import numpy as np
import pytest
from PIL import Image

import scripts.run_pipeline as rp
from ragregen import config
from ragregen.mmkg_store import schema as s
from ragregen.mmkg_store import store as st


class _RunDir:
    def __init__(self, root):
        self.root = root

    def case_dir(self, cid):
        d = self.root / cid
        d.mkdir(parents=True, exist_ok=True)
        return d


def _store(tmp_path):
    med = tmp_path / "m.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(med)
    rec = s.build_record(
        dataset="treevill", species_key="sp_A",
        scientific_name="Genus sp", common_name="sp_A",
        taxonomy={"genus": "Genus", "family": "Fam"},
        medoid={"image_path": str(med), "embedding_ref": 0,
                "k_images": 1, "selection": "siglip-centroid-nearest"},
        candidates=[], part_crops=[], attributes={},
        provenance={"source_split": "build"},
    )
    out = tmp_path / "store_out"
    st.write_store([rec], out)
    return st.Store.load(out), str(med)


def _case(gid):
    return config.Case(id="c1", prompt="a photo of a tree", concept="oak",
                       coarse="tree", global_id=gid)


def test_mmkg_reference_writes_medoid_path_to_refs(tmp_path):
    store, med = _store(tmp_path)
    run_dir = _RunDir(tmp_path / "run")
    rp._reference_mmkg_one(store, run_dir, _case("treevill:sp_A"))
    refs = json.loads((run_dir.case_dir("c1") / "refs.json").read_text())
    assert refs == [med]
    meta = json.loads((run_dir.case_dir("c1") / "retrieval.json").read_text())
    assert meta["source"] == "mmkg" and meta["global_id"] == "treevill:sp_A"


def test_mmkg_reference_missing_key_raises_no_fallback(tmp_path):
    store, _ = _store(tmp_path)
    run_dir = _RunDir(tmp_path / "run")
    with pytest.raises(ValueError, match="global_id"):
        rp._reference_mmkg_one(store, run_dir, _case("treevill:MISSING"))
    with pytest.raises(ValueError, match="global_id"):
        rp._reference_mmkg_one(store, run_dir, _case(None))
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_run_pipeline.py -k mmkg_reference -v`
Expected: FAIL (`run_pipeline` has no `_reference_mmkg_one`).

- [ ] **Step 3: Implement the helper and wire the mode**

In `scripts/run_pipeline.py`, add the helper (near `_retrieve_one`, ~line 501):

```python
def _reference_mmkg_one(store, run_dir, case) -> None:
    """MMKG reference source: resolve global_id -> medoid path. No FAISS, no NN.

    Writes the same refs.json contract the FAISS path writes (a list of image
    paths), so every downstream stage is unchanged. Raises rather than falling
    back when the identity is absent -- a silent fallback is the exact failure
    the spec forbids.
    """
    from ragregen.mmkg_store import reference_source as rs

    gid = case.global_id
    if not gid:
        raise ValueError(
            f"case {case.id!r} has no global_id but --reference-source mmkg "
            f"was requested; author an explicit '<dataset>:<species_key>'.")
    img = rs.medoid_reference(store, gid)
    if img is None:
        raise ValueError(
            f"case {case.id!r}: global_id {gid!r} is not in the MMKG store. "
            f"No fallback -- fix the key or the store path.")
    medoid_path = store.get(gid)["medoid"]["image_path"]

    d = run_dir.case_dir(case.id)
    (d / "refs.json").write_text(json.dumps([str(medoid_path)]))
    (d / "retrieval.json").write_text(json.dumps({
        "source": "mmkg", "global_id": gid, "medoid_path": str(medoid_path),
    }, indent=2))
```

Add the CLI flags in the argparser (near line 148):

```python
    ap.add_argument("--reference-source", choices=("faiss", "mmkg"),
                    default="faiss",
                    help="where repair references come from. 'mmkg' resolves "
                         "case.global_id to a precomputed medoid; no FAISS.")
    ap.add_argument("--mmkg-store", default=None,
                    help="path to the MMKG store dir (required for "
                         "--reference-source mmkg).")
```

In the retrieve stage block (the `if ... "retrieve"` section around line 310-328), branch on the mode so FAISS is never touched in mmkg mode. Replace the retrieve stage body with:

```python
        if args.reference_source == "mmkg":
            if not args.mmkg_store:
                raise ValueError("--reference-source mmkg needs --mmkg-store")
            from ragregen.mmkg_store import store as _store_mod
            store = _store_mod.Store.load(args.mmkg_store)
            for cid in list(queue.pending("retrieve")):
                _reference_mmkg_one(store, run_dir, by_id[cid])
                queue.mark(cid, "retrieve", "done")
        else:
            # existing FAISS path, unchanged (stage_with_model + _retrieve_one)
            ...
```

(Keep the existing FAISS branch verbatim in the `else`.)

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_run_pipeline.py -v`
Expected: PASS (new mmkg tests plus existing run_pipeline tests unaffected).

- [ ] **Step 5: Full suite + no-regression check**

Run: `pytest tests/test_regen.py tests/mmkg_store/ tests/test_config.py tests/test_run_pipeline.py tests/test_mmkg_reference_integration.py -q`
Expected: PASS. Confirms nothing downstream of the reference source changed behavior.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_pipeline.py tests/test_run_pipeline.py
git commit -m "feat(pipeline): --reference-source mmkg (deterministic medoid, no FAISS)"
```

---

## Self-Review

**Spec coverage:**
- §3 explicit `global_id`, no fallback → Task 2 (field + validation), Task 3 (raise on miss/absent). ✅
- §4 retrieve-stage seam, no FAISS/NN, unchanged downstream → Task 3 (branch + refs.json contract), Task 3 Step 5 (no-regression). ✅
- §4 provenance in trace → Task 3 writes `retrieval.json` with `source`/`global_id`/`medoid_path`. ✅ (Full `store.provenance()` into the run trace is deferred — spec §8/§9 list it as optional; note below.)
- §5 CPU smoke test, fixture store, stitch + fake-pipe inpaint, identity discrimination, negative control → Task 1. ✅
- §6 no-silent-fallback + no-NN controls → Task 1 (`nearest_crops==[]`, missing→None), Task 3 (raise). ✅ Parity control is optional (§6) and omitted to keep scope minimal.
- §7 boundary (no fidelity metric) → no task introduces a metric. ✅
- §10 boundaries (regen.py untouched, no GPU, don't build on parked branch) → Global Constraints + Task 1 note. ✅

**Placeholder scan:** the only `...` are explicit "keep existing code verbatim" markers in Task 3 Step 3, with the surrounding new code fully specified. No TBD/TODO. ✅

**Type consistency:** `global_id: str | None` used identically in config (Task 2) and consumed in Task 3; `_reference_mmkg_one(store, run_dir, case)` signature matches its test and call site; `refs.json` is a `list[str]` in both the FAISS path and the mmkg helper. ✅

**Deferred (tracked, not gaps):** recording full `store.provenance(gid)` into `trace.json`; the parity control; skipping `prepare_reference` for medoids. All are spec §8/§9 "optional/deferred".
