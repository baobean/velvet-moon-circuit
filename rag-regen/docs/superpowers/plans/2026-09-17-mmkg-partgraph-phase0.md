# PartGraph-RAG Phase 0 (through the wk2 GO/NO-GO gate) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and pilot the core mechanism — condition FLUX repair on a *composed multi-part MMKG reference* — and decide GO/NO-GO against the single-medoid baseline with a pre-registered rule, before spending the CVPR runway.

**Architecture:** A small new package `ragregen/partgraph/` sits *upstream* of the existing, unmodified repair engine (`ragregen/regen.py`). It turns a species' MMKG subgraph into ONE reference image (select part crops → compose a canvas), which the existing `Inpainter.regen` / `stitch` consume byte-for-byte like any other reference. A pilot harness runs the `single_medoid` and `partgraph` arms over ~15 species and applies a frozen GAIN rule. Everything except the FLUX call itself is CPU/fixture-testable.

**Tech Stack:** Python 3.11 (`kontext` conda env), PIL, numpy, existing `ragregen.{mmkg_store,regen,mask,metrics,draft,trace,config,env}`, FAISS (already in store), FLUX.1-Kontext (existing `regen.Inpainter`).

## Global Constraints

- **Python:** always `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python` — never bare `python` (a conda 3.13 with no faiss). Tests: `.../kontext/bin/python -m pytest`.
- **Do not modify the generator.** `ragregen/regen.py` sinks (`Inpainter.regen`, `stitch`) stay unchanged; the mechanism only changes the *reference source* upstream.
- **Pre-registration (freeze before any scoring):** Phase 0 GAIN rule — `GAIN` iff `mean(Δ) ≥ +0.02` DINO cosine **and** `CI_low > 0`; `LOSS` iff `mean(Δ) ≤ −0.02` **and** `CI_high < 0`; else `NO_LARGE_EFFECT`. Δ = `partgraph − single_medoid` per species. The `+0.02` margin is `PHASE0_GAIN_MARGIN` (Task 5), fixed before data. (Note: `metrics.MARGIN = -0.02` is the *non-inferiority* margin — do not reuse it for the GAIN test.)
- **Encoder hygiene:** the eval encoder must differ from any retrieval/crop encoder (already enforced by `config.load_pipeline`). DINO (`facebook/dinov3-vitl16...`) is the primary metric; it must not equal `retriever`/`crop_scorer`.
- **Held-out refs:** score only against references the repair never saw — use `metrics.split_refs(gt_refs, heldout)` and `metrics.dino_identity(output_crop, held_out_crops, encoder)`. Scoring with the edit pool is self-marking.
- **Preservation guard:** every composite case reports `metrics.preservation(...)`; for `stitch`/composited `inpaint` it must be `1.000` outside the mask. A gain from repainting the whole canvas is not a gain.
- **No repair-time NN for the arms under test** beyond the declared reference source; `partgraph` uses `store.get` + part-crop `image_path`s (and optionally `store.nearest_crops` for part selection, declared per run).
- **GPU:** runs must be `setsid`-detached and resumable (write per-case results row-by-row); pin the card (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n>`); machines `.200`/`.202`/`.245` (see `docs/superpowers/2026-09-17-cvpr-partgraph-handoff.md` infra section — `.245` needs `CUDA_VISIBLE_DEVICES=2` and no flock). `regen.check_vram()` enforces the 12 GB floor.
- **Commits:** small, one per task; branch off `main` (e.g. `partgraph-phase0`), never commit to `main`.

---

### Task 1: `select_part_images` — gather a species' part crops from the Store

**Files:**
- Create: `ragregen/partgraph/__init__.py` (empty)
- Create: `ragregen/partgraph/select.py`
- Test: `tests/partgraph/test_select.py` (create `tests/partgraph/__init__.py` too)

**Interfaces:**
- Consumes: `mmkg_store.store.Store` (`.get(global_id) -> record | None`); a part-crop record is `{"part_type": str, "image_path": str, "embedding_ref": int}` (confirmed in `build_treevill.py:144-147`).
- Produces: `select_part_images(store, global_id, *, parts=None) -> dict[str, PIL.Image.Image]` — one RGB image per part_type (first crop of each type; deterministic by list order). Empty dict if the record is missing or has no part_crops. `parts` optionally restricts to a subset of part_type names.

- [ ] **Step 1: Write the failing test**

```python
# tests/partgraph/test_select.py
from pathlib import Path
from PIL import Image
from ragregen.partgraph.select import select_part_images


class _FakeStore:
    def __init__(self, records): self._r = records
    def get(self, gid): return self._r.get(gid)


def _png(tmp_path, name, color):
    p = tmp_path / name
    Image.new("RGB", (32, 32), color).save(p)
    return str(p)


def test_selects_one_image_per_part_type(tmp_path):
    leaf = _png(tmp_path, "leaf.png", (0, 128, 0))
    bark = _png(tmp_path, "bark.png", (128, 64, 0))
    store = _FakeStore({"ds:sp_A": {"part_crops": [
        {"part_type": "leaf", "image_path": leaf, "embedding_ref": 0},
        {"part_type": "bark", "image_path": bark, "embedding_ref": 1},
    ]}})
    got = select_part_images(store, "ds:sp_A")
    assert set(got) == {"leaf", "bark"}
    assert got["leaf"].mode == "RGB" and got["leaf"].size == (32, 32)


def test_missing_record_returns_empty(tmp_path):
    assert select_part_images(_FakeStore({}), "ds:nope") == {}


def test_parts_filter(tmp_path):
    leaf = _png(tmp_path, "leaf.png", (0, 128, 0))
    bark = _png(tmp_path, "bark.png", (128, 64, 0))
    store = _FakeStore({"ds:sp_A": {"part_crops": [
        {"part_type": "leaf", "image_path": leaf, "embedding_ref": 0},
        {"part_type": "bark", "image_path": bark, "embedding_ref": 1},
    ]}})
    got = select_part_images(store, "ds:sp_A", parts=["leaf"])
    assert set(got) == {"leaf"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/partgraph/test_select.py -v`
Expected: FAIL with `ModuleNotFoundError: ragregen.partgraph.select`.

- [ ] **Step 3: Write minimal implementation**

```python
# ragregen/partgraph/select.py
"""Gather a species' part-crop images from the MMKG store (no FAISS, no NN)."""
from __future__ import annotations

from PIL import Image


def select_part_images(store, global_id, *, parts=None):
    """One RGB image per part_type for a species. Empty dict if absent.

    First crop of each part_type wins (deterministic by record list order).
    `parts` optionally restricts to a subset of part_type names.
    """
    rec = store.get(global_id)
    if rec is None:
        return {}
    out: dict[str, Image.Image] = {}
    for crop in rec.get("part_crops", []):
        pt = crop["part_type"]
        if parts is not None and pt not in parts:
            continue
        if pt in out:
            continue
        out[pt] = Image.open(crop["image_path"]).convert("RGB")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/partgraph/test_select.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add ragregen/partgraph/__init__.py ragregen/partgraph/select.py tests/partgraph/__init__.py tests/partgraph/test_select.py
git commit -m "feat(partgraph): select part-crop images from MMKG store"
```

---

### Task 2: `compose_reference` — one canvas from many part crops

**Files:**
- Create: `ragregen/partgraph/compose.py`
- Test: `tests/partgraph/test_compose.py`

**Interfaces:**
- Consumes: `dict[str, PIL.Image.Image]` (part_type -> image), as produced by Task 1.
- Produces: `compose_reference(part_images, *, tile=384, order=None) -> PIL.Image.Image` — a single RGB canvas, each part resized to fit a `tile`×`tile` cell, laid out in a near-square grid in `order` (default: sorted part_type names, for determinism). Raises `ValueError` on empty input. This is the mechanism's core novelty for the pilot: v1 is a deterministic grid; *relation-driven placement* is a later ablation rung (Phase 1), noted so it is not conflated with v1.

- [ ] **Step 1: Write the failing test**

```python
# tests/partgraph/test_compose.py
import math
import pytest
from PIL import Image
from ragregen.partgraph.compose import compose_reference


def _imgs():
    return {
        "leaf": Image.new("RGB", (100, 50), (0, 128, 0)),
        "bark": Image.new("RGB", (40, 90), (128, 64, 0)),
        "flower": Image.new("RGB", (60, 60), (200, 0, 120)),
    }


def test_single_canvas_rgb_grid_size():
    out = compose_reference(_imgs(), tile=384)
    assert out.mode == "RGB"
    cols = math.ceil(math.sqrt(3))          # 2
    rows = math.ceil(3 / cols)              # 2
    assert out.size == (cols * 384, rows * 384)


def test_deterministic_default_order():
    a = compose_reference(_imgs(), tile=128)
    b = compose_reference(_imgs(), tile=128)
    assert list(a.getdata()) == list(b.getdata())


def test_empty_raises():
    with pytest.raises(ValueError):
        compose_reference({}, tile=128)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/partgraph/test_compose.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# ragregen/partgraph/compose.py
"""Compose one reference canvas from several part crops (the mechanism core).

v1 = deterministic near-square grid, one part per cell (fit-inside, centred on
a neutral-grey cell so a cropped part is not read as an object edge). Relation-
driven placement is a Phase-1 ablation, deliberately NOT done here."""
from __future__ import annotations

import math

from PIL import Image

_GREY = (127, 127, 127)


def compose_reference(part_images, *, tile: int = 384, order=None):
    if not part_images:
        raise ValueError("compose_reference: no part images supplied")
    keys = list(order) if order is not None else sorted(part_images)
    n = len(keys)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    canvas = Image.new("RGB", (cols * tile, rows * tile), _GREY)
    for i, k in enumerate(keys):
        img = part_images[k].convert("RGB")
        w, h = img.size
        s = min(tile / w, tile / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        cell = img.resize((nw, nh), Image.LANCZOS)
        cx, cy = (i % cols) * tile, (i // cols) * tile
        canvas.paste(cell, (cx + (tile - nw) // 2, cy + (tile - nh) // 2))
    return canvas
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/partgraph/test_compose.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add ragregen/partgraph/compose.py tests/partgraph/test_compose.py
git commit -m "feat(partgraph): compose one reference canvas from part crops"
```

---

### Task 3: `reference_for` — the arms as one interface

**Files:**
- Create: `ragregen/partgraph/arms.py`
- Test: `tests/partgraph/test_arms.py`

**Interfaces:**
- Consumes: `mmkg_store.reference_source.medoid_reference(store, gid) -> Image | None`; Task 1 `select_part_images`; Task 2 `compose_reference`.
- Produces: `ARMS = ("single_medoid", "partgraph")`; `reference_for(arm, store, global_id, *, tile=384) -> PIL.Image.Image` — the reference image for that arm. Raises `ValueError` for an unknown arm; raises `LookupError` if the store cannot supply the arm's reference (no silent fallback — the exact failure mode the reference-integration design §3 requires).

- [ ] **Step 1: Write the failing test**

```python
# tests/partgraph/test_arms.py
import pytest
from PIL import Image
from ragregen.partgraph.arms import reference_for, ARMS


class _Store:
    def __init__(self, rec, medoid=None): self._rec, self._medoid = rec, medoid
    def get(self, gid): return self._rec


def _crop(tmp_path, name, color):
    p = tmp_path / name
    Image.new("RGB", (40, 40), color).save(p)
    return str(p)


def test_arms_registry():
    assert ARMS == ("single_medoid", "partgraph")


def test_partgraph_arm_composes(tmp_path):
    rec = {"part_crops": [
        {"part_type": "leaf", "image_path": _crop(tmp_path, "l.png", (0, 128, 0)), "embedding_ref": 0},
        {"part_type": "bark", "image_path": _crop(tmp_path, "b.png", (128, 64, 0)), "embedding_ref": 1},
    ]}
    out = reference_for("partgraph", _Store(rec), "ds:sp_A", tile=64)
    assert out.mode == "RGB" and out.size == (128, 128)   # 2x2 grid of 64


def test_partgraph_no_parts_raises(tmp_path):
    with pytest.raises(LookupError):
        reference_for("partgraph", _Store({"part_crops": []}), "ds:sp_A")


def test_unknown_arm_raises(tmp_path):
    with pytest.raises(ValueError):
        reference_for("magic", _Store({}), "ds:sp_A")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/partgraph/test_arms.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# ragregen/partgraph/arms.py
"""The reference source per arm — single interface, no silent fallback."""
from __future__ import annotations

from ragregen.mmkg_store.reference_source import medoid_reference
from ragregen.partgraph.compose import compose_reference
from ragregen.partgraph.select import select_part_images

ARMS = ("single_medoid", "partgraph")


def reference_for(arm, store, global_id, *, tile: int = 384):
    if arm == "single_medoid":
        ref = medoid_reference(store, global_id)
        if ref is None:
            raise LookupError(f"no medoid for {global_id!r}")
        return ref
    if arm == "partgraph":
        parts = select_part_images(store, global_id)
        if not parts:
            raise LookupError(f"no part crops for {global_id!r}")
        return compose_reference(parts, tile=tile)
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/partgraph/test_arms.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add ragregen/partgraph/arms.py tests/partgraph/test_arms.py
git commit -m "feat(partgraph): unified per-arm reference source (medoid vs partgraph)"
```

---

### Task 4: `gain_verdict` — the frozen GO/NO-GO decision rule

**Files:**
- Create: `ragregen/partgraph/decide.py`
- Test: `tests/partgraph/test_decide.py`

**Interfaces:**
- Consumes: `metrics.paired_bootstrap_ci(deltas) -> (lo, hi) | None`.
- Produces: `PHASE0_GAIN_MARGIN = 0.02`; `gain_verdict(deltas, *, seed=0) -> dict` with keys `mean`, `ci` (`[lo, hi]` or `None`), `n`, `verdict` ∈ {`GAIN`, `LOSS`, `NO_LARGE_EFFECT`, `UNDERPOWERED`}. `UNDERPOWERED` when `paired_bootstrap_ci` returns `None` (n < `metrics.MIN_CI_N`). Rule: `GAIN` iff `mean ≥ +MARGIN and lo > 0`; `LOSS` iff `mean ≤ −MARGIN and hi < 0`; else `NO_LARGE_EFFECT`.

- [ ] **Step 1: Write the failing test**

```python
# tests/partgraph/test_decide.py
from ragregen.partgraph.decide import gain_verdict, PHASE0_GAIN_MARGIN


def test_margin_frozen():
    assert PHASE0_GAIN_MARGIN == 0.02


def test_clear_gain():
    r = gain_verdict([0.10, 0.12, 0.09, 0.11, 0.13, 0.10])
    assert r["verdict"] == "GAIN" and r["ci"][0] > 0


def test_null_when_ci_crosses_zero():
    r = gain_verdict([0.10, -0.08, 0.12, -0.09, 0.11, -0.10])
    assert r["verdict"] == "NO_LARGE_EFFECT"


def test_underpowered_below_min_ci_n():
    r = gain_verdict([0.10, 0.12])          # n < MIN_CI_N (5)
    assert r["verdict"] == "UNDERPOWERED" and r["ci"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/partgraph/test_decide.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# ragregen/partgraph/decide.py
"""The pre-registered Phase-0 GO/NO-GO rule. Frozen before data (spec §8)."""
from __future__ import annotations

import numpy as np

from ragregen import metrics

#: GAIN margin in DINO cosine, fixed before any scoring. Do not tune.
PHASE0_GAIN_MARGIN = 0.02


def gain_verdict(deltas, *, seed: int = 0) -> dict:
    vals = [d for d in deltas if d is not None]
    mean = float(np.mean(vals)) if vals else float("nan")
    ci = metrics.paired_bootstrap_ci(vals, seed=seed)
    if ci is None:
        return {"mean": mean, "ci": None, "n": len(vals), "verdict": "UNDERPOWERED"}
    lo, hi = ci
    if mean >= PHASE0_GAIN_MARGIN and lo > 0:
        verdict = "GAIN"
    elif mean <= -PHASE0_GAIN_MARGIN and hi < 0:
        verdict = "LOSS"
    else:
        verdict = "NO_LARGE_EFFECT"
    return {"mean": mean, "ci": [lo, hi], "n": len(vals), "verdict": verdict}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/partgraph/test_decide.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add ragregen/partgraph/decide.py tests/partgraph/test_decide.py
git commit -m "feat(partgraph): frozen Phase-0 GAIN decision rule"
```

---

### Task 5: `run_case` — one species through one arm (draft→mask→ref→regen→score)

**Files:**
- Create: `ragregen/partgraph/pilot.py`
- Test: `tests/partgraph/test_pilot.py`

**Interfaces:**
- Consumes: `regen.stitch(draft, mask, cutout) -> RegenResult` and `regen.Inpainter(pipe=...)` (injectable pipe — CPU-testable, `regen.py:345`); `metrics.crop_to_mask(image, mask)`, `metrics.split_refs(gt_refs, heldout)`, `metrics.dino_identity(output_crop, ref_crops, encoder)`, `metrics.preservation(draft, output, alpha)`; Task 3 `reference_for`.
- Produces: `run_case(*, arm, store, global_id, draft, mask, gt_refs, encoder, sink, heldout=1, tile=384) -> dict` with keys `arm`, `global_id`, `dino`, `preservation`, `ref_size`. `sink` is a callable `(draft, mask, reference) -> RegenResult` (so the FLUX pipe stays injectable and the test uses `stitch`). Scores the output crop against **held-out** ref crops only.

- [ ] **Step 1: Write the failing test**

```python
# tests/partgraph/test_pilot.py
from pathlib import Path
from PIL import Image
from ragregen.partgraph.pilot import run_case
from ragregen import regen


class _Store:
    def __init__(self, rec): self._rec = rec
    def get(self, gid): return self._rec


class _ConstEncoder:      # deterministic, offline stand-in for DinoEncoder
    def embed(self, image):
        import numpy as np
        a = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
        return a / (np.linalg.norm(a) + 1e-12)


def _png(p, color, size=(64, 64)):
    Image.new("RGB", size, color).save(p); return p


def test_run_case_stitch_arm_scores_and_preserves(tmp_path):
    leaf = tmp_path / "leaf.png"; _png(leaf, (0, 200, 0))
    rec = {"part_crops": [{"part_type": "leaf", "image_path": str(leaf), "embedding_ref": 0}],
           "medoid": {"image_path": str(leaf)}}
    draft = Image.new("RGB", (128, 128), (10, 10, 10))
    mask = Image.new("L", (128, 128), 0)
    for x in range(40, 88):
        for y in range(40, 88):
            mask.putpixel((x, y), 255)
    r1 = tmp_path / "r1.png"; _png(r1, (0, 200, 0))
    r2 = tmp_path / "r2.png"; _png(r2, (0, 190, 0))

    def sink(d, m, ref):
        return regen.stitch(d, m, ref)

    out = run_case(arm="partgraph", store=_Store(rec), global_id="ds:sp_A",
                   draft=draft, mask=mask, gt_refs=[Path(r1), Path(r2)],
                   encoder=_ConstEncoder(), sink=sink, heldout=1, tile=64)
    assert out["arm"] == "partgraph"
    assert 0.0 <= out["dino"] <= 1.0
    assert abs(out["preservation"] - 1.0) < 1e-6      # stitch preserves ~mask exactly
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/partgraph/test_pilot.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# ragregen/partgraph/pilot.py
"""Run one species through one arm and score it (held-out DINO + preservation)."""
from __future__ import annotations

from PIL import Image

from ragregen import metrics
from ragregen.partgraph.arms import reference_for


def run_case(*, arm, store, global_id, draft, mask, gt_refs, encoder, sink,
             heldout: int = 1, tile: int = 384) -> dict:
    reference = reference_for(arm, store, global_id, tile=tile)
    result = sink(draft, mask, reference)
    output = result.image

    _edit_pool, held = metrics.split_refs(gt_refs, heldout)
    out_crop = metrics.crop_to_mask(output, mask)
    held_crops = [metrics.crop_to_mask(Image.open(p).convert("RGB"),
                                       Image.new("L", Image.open(p).size, 255))
                  for p in held]
    dino = metrics.dino_identity(out_crop, held_crops, encoder)
    pres = metrics.preservation(draft, output, result.alpha)
    return {"arm": arm, "global_id": global_id, "dino": dino,
            "preservation": pres["score"], "ref_size": list(reference.size)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/partgraph/test_pilot.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ragregen/partgraph/pilot.py tests/partgraph/test_pilot.py
git commit -m "feat(partgraph): single-case arm runner (held-out DINO + preservation)"
```

---

### Task 6: Dataset → MMKG store build (PlantCLEF + CUB, part crops from native labels)

> **Data dependency — do the reconnaissance step first.** Exact code cannot be written blind: it depends on the on-disk layout of the downloaded PlantCLEF / CUB. Step 1 inspects the real files and records the layout; Steps 2-5 build on the confirmed shape, reusing the existing `build_treevill.py` pattern.

**Files:**
- Create: `ragregen/mmkg_store/build_plantclef.py` (mirror `build_treevill.py`)
- Create: `ragregen/mmkg_store/build_cub.py` (mirror `build_treevill.py`)
- Test: `tests/mmkg_store/test_build_plantclef.py`, `tests/mmkg_store/test_build_cub.py`
- Reuse (unchanged): `mmkg_store.schema.build_record`, `mmkg_store.store.write_store`, `mmkg_store.embed_index.IndexBuilder`.

**Interfaces:**
- Consumes: `schema.build_record(dataset=, species_key=, scientific_name=, common_name=, taxonomy={genus,family}, medoid={image_path,embedding_ref,k_images,selection}, candidates=[], part_crops=[{part_type,image_path,embedding_ref}], attributes={}, provenance={})`; `embed_index.IndexBuilder().add(vec, meta) -> embedding_ref`; `store.write_store(records, out_dir, index_builder)`.
- Produces: an on-disk store loadable by `Store.load(dir)` where each species record has `part_crops` keyed by the dataset's native part types (PlantCLEF: `leaf/flower/fruit/bark/branch`; CUB: derived crops for the 15 part locations, grouped into `head/breast/wing/tail/leg` or similar — decide the grouping in Step 1 and freeze it).

- [ ] **Step 1: Reconnaissance — record the real layout (no code committed yet).** Run against the actual downloaded data with the env python: list the directory tree, the metadata/label file format (PlantCLEF content-type field; CUB `parts/part_locs.txt`, `image_class_labels.txt`, `attributes/`), and image resolutions. Write findings into `docs/superpowers/2026-09-17-cvpr-partgraph-handoff.md` under a new "Dataset layouts" subsection. **Confirm each dataset passes the wk1 rarity/headroom screen (Task 7) before building the full store.**

- [ ] **Step 2: Write the failing test** — a hermetic build test that constructs a tiny fixture tree (2 species, 2 part-typed images each) in `tmp_path`, runs the build entrypoint against it, then `Store.load`s the result and asserts: 2 records, each with `part_crops` of the expected part types, a medoid `image_path`, and taxonomy hubs present. Mirror `tests/mmkg_store/` existing build tests.

- [ ] **Step 3: Implement `build_plantclef.py`** following `build_treevill.py` structure: iterate species → per part-type content label, pick the crop image(s) → `part_crops.append({"part_type": <label>, "image_path": <resolved>, "embedding_ref": index_builder.add(embed(img), meta)})`; pick medoid via SigLIP centroid-nearest (reuse the existing helper in `build.py`); `build_record(...)`; `write_store(records, out_dir, index_builder)`. Then `build_cub.py` likewise, deriving part crops from `part_locs.txt` bounding boxes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/mmkg_store/test_build_plantclef.py tests/mmkg_store/test_build_cub.py -v`

- [ ] **Step 5: Build the real ~15-species pilot stores** (GPU for embeddings; resumable) for both datasets, into `/mmlabworkspace_new/.../partgraph_store/{plantclef,cub}/`. Record counts + provenance. Commit the build code (not the data).

```bash
git add ragregen/mmkg_store/build_plantclef.py ragregen/mmkg_store/build_cub.py tests/mmkg_store/test_build_plantclef.py tests/mmkg_store/test_build_cub.py
git commit -m "feat(mmkg_store): build part-structured stores from PlantCLEF + CUB"
```

---

### Task 7: Pilot runner + rarity screen → the GO/NO-GO artifact

**Files:**
- Create: `scripts/partgraph_pilot.py` (mirror `scripts/screen_premise.py` structure: `env.exit_now(main())`, `trace.open_run`, row-by-row writes, `print_gpu_info`)
- Reuse (unchanged): `scripts/screen_premise.py` for the wk1 rarity/headroom screen; `ragregen.draft`, `ragregen.mask`, `regen.Inpainter`.
- Test: `tests/partgraph/test_pilot_runner.py` (CPU: drive `main` with a tiny fixture dataset + fixture store + `stitch` sink + a monkeypatched encoder; assert `result.json` written with a `verdict`).

**Interfaces:**
- Consumes: Task 5 `run_case`, Task 4 `gain_verdict`, Task 3 `ARMS`; `config.load_dataset`, `config.load_pipeline`, `metrics.EvalEncoders.load`, `mmkg_store.store.Store.load`; each pilot case carries an explicit `global_id` (reference-integration design §3 — supplied on the case, no inference).
- Produces: `outputs/partgraph_pilot/<ts>/result.json` = `{per_case: [...both arms...], deltas: [...], decision: gain_verdict(deltas)}` + per-case traces (subgraph parts used, composed-canvas image, both arm outputs, scores). Exit non-zero if any case dropped.

- [ ] **Step 1: Freeze the run config** — ~15 species/dataset with explicit `global_id`, `steps`/`seed` from `pipeline.yaml`, `heldout_refs ≥ 1`, DINO eval encoder ≠ retriever/crop encoder. Write the frozen knobs into the run's `args`.

- [ ] **Step 2: Write the failing test** — CPU-only: fixture dataset (2 cases w/ `global_id` + `gt_refs`), fixture store (part crops + medoid), `stitch` sink, deterministic offline encoder (as in Task 5); run `main`; assert `result.json` exists with `per_case` (4 rows = 2 species × 2 arms), a `deltas` list, and a `decision` dict whose `verdict` ∈ the four values.

- [ ] **Step 3: Implement `main`** — for each species, run both `ARMS` through `run_case` (draft once per species via `draft.Drafter`; mask via `mask.mask_draft`; `Inpainter` sink for the real run, `stitch` for the CPU test); write each row immediately (resumable); compute `deltas = partgraph_dino − single_medoid_dino` per species; `decision = gain_verdict(deltas)`; `run.finish("ok", {...})`; `env.reclaim_gpu()`.

- [ ] **Step 4: Run the test**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/partgraph/test_pilot_runner.py -v`

- [ ] **Step 5: Run the real pilot (the wk2 gate)** on a reserved GPU (`setsid nohup`, pinned card), both datasets. Read `decision.verdict`:
  - **GAIN** → proceed to the Phase 1 plan (full runs, IP-Adapter/ImageRAG/RAVEL-text-KG baselines, ablation ladder, human study).
  - **NO_LARGE_EFFECT / LOSS / UNDERPOWERED** → **stop and pivot the mechanism** (spec §8); do not scale a dead mechanism. Record the verdict as a finding under `docs/superpowers/`.

```bash
git add scripts/partgraph_pilot.py tests/partgraph/test_pilot_runner.py
git commit -m "feat(partgraph): wk2 GO/NO-GO pilot runner + frozen decision artifact"
```

---

## Self-review notes

- **Spec coverage:** §5 mechanism → Tasks 1-3,5; §8 gates → Tasks 4,7 (+ reuse `screen_premise.py` for wk1); §4 datasets → Task 6; §6 baselines → `single_medoid` vs `partgraph` here (IP-Adapter/ImageRAG/RAVEL-text-KG deferred to Phase 1, as this plan is scoped to the go/no-go gate); §7 eval protocol → held-out DINO + preservation + frozen rule in Tasks 4-5,7. Multi-instance / relational-arrangement ablations are Phase 1 (noted in Task 2).
- **Type consistency:** `reference_for(arm, store, global_id, *, tile)` used identically in Tasks 3,5,7; `gain_verdict(deltas)` in Tasks 4,7; part-crop record `{part_type,image_path,embedding_ref}` in Tasks 1,6 (matches `build_treevill.py:144`); `run_case(...)->{arm,global_id,dino,preservation,ref_size}` in Tasks 5,7.
- **Known honest gap:** Task 6 code is layout-dependent → its Step 1 is reconnaissance, not fabricated code. This is deliberate, not a placeholder.
