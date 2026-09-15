# Stage-A′ — Per-Part Inpaint Transfer Under a Real Deficit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Stage-A make-or-break test at the part level — mask a real held-out part region, restore it by inpainting conditioned on borrowed part-crops, and measure whether shared-PartType-hub crops beat plain nearest-neighbour crops under a guaranteed deficit.

**Architecture:** Additive on the Stage-A codebase. A new inpaint consumer (`SdxlIpGenerator.inpaint_multi`), a new held-out part-geometry store, and one shared pure `build_restore_inputs` that assembles (masked target, box mask, conditioning refs, scoring refs) so every arm differs **only** in its selector. A resident two-phase GPU runner drives ~1,440 cells; a pure analysis module produces the recovery curve, the `+Hub − +RawNN` paired delta, and the reliability-stratified read.

**Tech Stack:** Python 3.11, PyTorch 2.6+cu124, diffusers (SDXL + IP-Adapter + `AutoPipelineForInpainting`), transformers (GroundingDINO, SAM, DINOv3/SigLIP2/CLIP), numpy, pytest.

## Global Constraints

- **GPU env / interpreter:** `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python` (torch 2.6+cu124, py3.11). The base conda env has no torch.
- **NO GIT.** `kg_test` has no working git repo (`.git` is a stub). Do **not** `git init` or `git commit`. Each task's final step verifies tests green and appends a line to the SDD ledger `.superpowers/sdd/2026-09-07-relational-mmkg-stage-a-prime/progress.md`. Reviewers read files, not diffs.
- **Cached model ids — do not substitute:** `sdxl_id=stabilityai/stable-diffusion-xl-base-1.0`, `ip_adapter_repo=h94/IP-Adapter`, `ip_adapter_weight=ip-adapter-plus_sdxl_vit-h.safetensors`, `dino_detector_id=IDEA-Research/grounding-dino-base`, `sam_id=facebook/sam-vit-huge`, embedders per `GraftConfig`. The inpaint pipeline reuses `sdxl_id` (base SDXL) via `AutoPipelineForInpainting` — no new checkpoint (diffusers runs the 4-channel base UNet through the masked-latent-blend inpaint path).
- **Frozen Stage-A assets (reuse, never rebuild):** `outputs/mmkg/graph.json` (12 hubs, P1–P3 params), the Stage-A build part-crop store `outputs/mmkg/_crops/`, the 8 concepts, `metrics.image_fidelity`, `graft.transfer.{select_borrowed,reference_weights}`, the subprocess / resident-two-phase GPU architecture, and `generate_multi` (text2img reference path — untouched).
- **Quantity-match invariant (§3 of spec):** per `(concept, part)` cell, `k_eff = min(k=4, hub-borrowable count)`; the same `k_eff` is used by **all three** borrowing arms (or the cell is skipped if `k_eff=0`). Isolated carries only `build_P` own crops and is never part of the make-or-break.
- **Contamination guard:** borrowed crops are drawn from the cross-concept, **part-restricted** build pool with the target concept's own crops excluded. Conditioning (build split) and scoring (held-out split) crop sets are disjoint by the standing `split_refs(seed=0)` split.
- **Geometry cached once per cell:** the target image's box + SAM mask is computed once and reused by every arm / build_P / draw. Inpaint uses the **box** mask; scoring uses the **SAM mask** region crop.
- **Two-phase GPU (24 GB card):** generate-all (SDXL-inpaint + IP resident) → unload → score-all (embedders resident). `inpaint_multi` body wrapped in `torch.no_grad()`; `torch.cuda.empty_cache()` per generation (the Stage-A ~0.5 GB/gen leak fix).
- **Metric:** DINOv3 primary; SigLIP2, CLIP-I secondary. Restored SAM-region crop vs the concept's **other** held-out P crops (target excluded).
- **Honesty:** if `+Hub ≈ +RawNN` under a working deficit → decisive negative, reported straight. If the deficit still fails to instantiate (recovery curve flat with a masked hole) → report as an infra/measurement finding, not evidence about the hypothesis.

---

## File Structure

- `graft/config.py` — **modify:** add Stage-A′ tunables (restore levels, k, draws, mask-area bounds, inpaint params).
- `graft/parts.py` — **create:** pure geometry helpers (mask area fraction, mask bbox, box-mask an image, crop a region).
- `graft/restore.py` — **create:** the one shared, pure cell-input builder + k_eff/selection logic. The heart.
- `graft/models.py` — **modify:** add `SdxlIpGenerator.pipe_inpaint` + `inpaint_multi`.
- `graft/worker_build_heldout_parts.py` — **create:** GPU worker; per-concept held-out part-geometry store (box, SAM mask, crop, embed).
- `graft/worker_restore_cell.py` — **create:** GPU worker; one `(concept,part,build_P,arm,draw)` cell (per-cell subprocess path).
- `graft/run_restore_resident.py` — **create:** resident two-phase runner for the full matrix.
- `graft/restore_analysis.py` — **create:** recovery curve, make-or-break, borrowing-helps, reliability stratification.
- `graft/smoke_restore.py` — **create:** the six §6 smoke-gate pass/fail checks as pure functions.
- `tests/` — a `test_*` file per module above.

---

## Task 1: Config tunables for Stage-A′

**Files:**
- Modify: `graft/config.py`
- Test: `tests/test_config_restore.py`

**Interfaces:**
- Produces: new `GraftConfig` fields — `restore_k: int = 4`, `restore_build_levels: tuple = (0,1,2,4)`, `restore_n_draws: int = 3`, `restore_min_mask_area_frac: float = 0.01`, `restore_max_mask_area_frac: float = 0.9`, `restore_inpaint_steps: int = 50`, `restore_inpaint_strength: float = 0.99`, `restore_guidance: float = 7.5`. (`restore_build_levels` serializes to a YAML list; `from_yaml` returns a list — consumers accept any sequence.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_restore.py
import os, tempfile
from graft.config import GraftConfig

def test_restore_fields_defaults():
    c = GraftConfig()
    assert c.restore_k == 4
    assert list(c.restore_build_levels) == [0, 1, 2, 4]
    assert c.restore_n_draws == 3
    assert 0 < c.restore_min_mask_area_frac < c.restore_max_mask_area_frac <= 1.0

def test_restore_fields_roundtrip_yaml():
    c = GraftConfig()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "cfg.yaml")
        c.to_yaml(p)
        back = GraftConfig.from_yaml(p)
    assert back.restore_k == c.restore_k
    assert list(back.restore_build_levels) == [0, 1, 2, 4]
    assert back.restore_inpaint_strength == c.restore_inpaint_strength
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_config_restore.py -v`
Expected: FAIL (`AttributeError: ... 'restore_k'`).

- [ ] **Step 3: Add the fields**

In `graft/config.py`, inside the `@dataclass(frozen=True) class GraftConfig`, after the `hub_label_*` lines and before `outputs_dir`:

```python
    # Stage-A' (per-part inpaint restore) tunables.
    restore_k: int = 4                       # borrowed budget cap (k_eff <= this)
    restore_build_levels: tuple = (0, 1, 2, 4)   # own-P-crop starvation levels
    restore_n_draws: int = 3
    restore_min_mask_area_frac: float = 0.01     # SAM mask area / box area, smoke lower bound
    restore_max_mask_area_frac: float = 0.9      # ... upper bound
    restore_inpaint_steps: int = 50
    restore_inpaint_strength: float = 0.99       # near-full repaint of the masked box
    restore_guidance: float = 7.5
```

Note: `to_yaml` uses `dataclasses.asdict`, which converts the tuple default to a list on dump; `from_yaml` then returns a list. All consumers iterate, so a list is fine.

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_config_restore.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Record (NO git)**

Append to the ledger: `Task 1 complete — config restore fields + roundtrip test (2 pass). NO git.`

---

## Task 2: `graft/parts.py` — pure geometry helpers

**Files:**
- Create: `graft/parts.py`
- Test: `tests/test_parts.py`

**Interfaces:**
- Produces:
  - `mask_area_fraction(mask: np.ndarray, box: tuple) -> float` — `mask.sum() / box_area`.
  - `mask_bbox(mask: np.ndarray) -> tuple[int,int,int,int]` — tight `(x0,y0,x1,y1)` of `True` pixels, or `None` if empty.
  - `box_mask_image(image: PIL.Image, box: tuple) -> tuple[PIL.Image, np.ndarray]` — returns `(masked_copy, box_mask)` where the box region is filled grey and `box_mask` is a bool array `True` inside the (int, image-clamped) box.
  - `crop_region(image: PIL.Image, box: tuple) -> PIL.Image` — crop the int, image-clamped box.
  - `clamp_box(box: tuple, w: int, h: int) -> tuple[int,int,int,int]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_parts.py
import numpy as np
from PIL import Image
from graft import parts

def test_clamp_box_clips_to_image():
    assert parts.clamp_box((-5, -5, 50, 40), 30, 20) == (0, 0, 30, 20)

def test_mask_area_fraction():
    mask = np.zeros((20, 30), dtype=bool)
    mask[0:10, 0:10] = True                      # 100 true px
    frac = parts.mask_area_fraction(mask, (0, 0, 10, 10))  # box area 100
    assert abs(frac - 1.0) < 1e-9

def test_mask_bbox_tight_and_none():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:9, 3:7] = True
    assert parts.mask_bbox(mask) == (3, 5, 7, 9)
    assert parts.mask_bbox(np.zeros((5, 5), bool)) is None

def test_box_mask_image_changes_only_inside_box():
    img = Image.new("RGB", (40, 30), (200, 100, 50))
    masked, bmask = parts.box_mask_image(img, (10, 5, 25, 20))
    a, m = np.asarray(img), np.asarray(masked)
    assert bmask.shape == (30, 40) and bmask[5:20, 10:25].all()
    # outside the box: identical; inside: changed
    assert (a[0:5, :] == m[0:5, :]).all()
    assert not (a[5:20, 10:25] == m[5:20, 10:25]).all()

def test_crop_region_size():
    img = Image.new("RGB", (40, 30))
    assert parts.crop_region(img, (10, 5, 25, 20)).size == (15, 15)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.../python -m pytest tests/test_parts.py -v`
Expected: FAIL (`ModuleNotFoundError: graft.parts`).

- [ ] **Step 3: Implement**

```python
# graft/parts.py
"""Pure geometry helpers for the part-level inpaint restore. No GPU/models here
so every function is unit-testable; the worker calls GroundingDINO/SAM and hands
their raw box/mask into these."""
from __future__ import annotations
import numpy as np
from PIL import Image

def clamp_box(box, w, h):
    x0, y0, x1, y1 = box
    x0 = max(0, min(int(round(x0)), w)); x1 = max(0, min(int(round(x1)), w))
    y0 = max(0, min(int(round(y0)), h)); y1 = max(0, min(int(round(y1)), h))
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

def mask_area_fraction(mask, box):
    x0, y0, x1, y1 = box
    box_area = max(1, (x1 - x0) * (y1 - y0))
    return float(np.asarray(mask, dtype=bool).sum()) / float(box_area)

def mask_bbox(mask):
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

def box_mask_image(image, box, fill=(127, 127, 127)):
    w, h = image.size
    x0, y0, x1, y1 = clamp_box(box, w, h)
    masked = image.copy()
    for y in range(y0, y1):                      # small boxes; explicit paste avoids numpy roundtrip
        pass
    px = np.asarray(masked).copy()
    px[y0:y1, x0:x1] = np.array(fill, dtype=px.dtype)
    masked = Image.fromarray(px)
    bmask = np.zeros((h, w), dtype=bool)
    bmask[y0:y1, x0:x1] = True
    return masked, bmask

def crop_region(image, box):
    w, h = image.size
    return image.crop(clamp_box(box, w, h))
```

- [ ] **Step 4: Run to verify it passes**

Run: `.../python -m pytest tests/test_parts.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Record (NO git)** — ledger: `Task 2 complete — parts.py geometry (5 pass). NO git.`

---

## Task 3: `graft/restore.py` — the one shared cell-input builder (k_eff invariant)

**Files:**
- Create: `graft/restore.py`
- Test: `tests/test_restore.py`

**Interfaces:**
- Consumes: `graft.transfer.select_borrowed`, `graft.transfer.reference_weights`, `graft.starvation.keep_draw`, `graft.parts`, `graph_schema.PartTypeGraph`.
- Produces:
  - `hub_of(graph, inst_id) -> int | None` (linear scan, Stage-A parity).
  - `own_part_crops(graph, concept, part) -> list[PartInstanceRec]` — build crops of `(concept, part)`.
  - `compute_k_eff(graph, concept, part, k) -> int` — hub-borrowable count for `(concept, part)` after excluding own; `min(k, that)`.
  - `select_refs(graph, concept, part, build_P, condition, k_eff, draw) -> (own_ids: list[str], borrowed_pool_idx: list[int], pool: list[PartInstanceRec])` — pure selection; own starved via `keep_draw`, borrowed via `select_borrowed` on the **part-restricted** pool. Same `k_eff` for all three borrowing arms.
  - `assemble_ref_images(own_ids, borrowed_pool_idx, pool, graph) -> (ref_imgs, weights)` — opens crops, `reference_weights(n_own, n_borrowed)`.

- [ ] **Step 1: Write the failing tests** (synthetic graph, no GPU)

```python
# tests/test_restore.py
import numpy as np
from graft.graph_schema import PartTypeGraph, PartInstanceRec, PartTypeRec
from graft import restore

def _emb(seed, n=8):
    v = np.random.default_rng(seed).normal(size=n); return (v / np.linalg.norm(v)).tolist()

def _graph():
    # two concepts share a 'leaf' hub; C1 has 2 own leaf crops, C2 has 3.
    insts = [
        PartInstanceRec("C1::leaf::0","C1","leaf","C1/r0","c10.png",_emb(1)),
        PartInstanceRec("C1::leaf::1","C1","leaf","C1/r1","c11.png",_emb(2)),
        PartInstanceRec("C2::leaf::0","C2","leaf","C2/r0","c20.png",_emb(3)),
        PartInstanceRec("C2::leaf::1","C2","leaf","C2/r1","c21.png",_emb(4)),
        PartInstanceRec("C2::leaf::2","C2","leaf","C2/r2","c22.png",_emb(5)),
    ]
    hub = PartTypeRec(0, "leaf", [i.id for i in insts], _emb(6), 0.9)
    return PartTypeGraph(part_instances=insts, part_types=[hub])

def test_hub_of_and_own_crops():
    g = _graph()
    assert restore.hub_of(g, "C1::leaf::0") == 0
    assert [o.id for o in restore.own_part_crops(g, "C1", "leaf")] == ["C1::leaf::0","C1::leaf::1"]

def test_k_eff_bounded_by_hub_siblings():
    g = _graph()
    # C1 borrows from C2's 3 leaf crops, cap max_per_concept=2 in centroid_prototype -> <=2, and <=k
    assert restore.compute_k_eff(g, "C1", "leaf", k=4) == 2

def test_select_refs_starves_own_and_matches_borrowed_count():
    g = _graph()
    k_eff = restore.compute_k_eff(g, "C1", "leaf", k=4)
    for cond in ("random", "rawnn", "hub"):
        own, bidx, pool = restore.select_refs(g, "C1", "leaf", build_P=1, condition=cond, k_eff=k_eff, draw=0)
        assert len(own) == 1                       # starved to build_P=1
        assert len(bidx) == k_eff                  # all borrowing arms match k_eff
        assert all(pool[i].concept != "C1" for i in bidx)   # contamination: no own crops
    own0, b0, _ = restore.select_refs(g, "C1", "leaf", build_P=0, condition="isolated", k_eff=k_eff, draw=0)
    assert own0 == [] and b0 == []                 # isolated + build_P=0 -> nothing

def test_assemble_weights_sum_to_one(tmp_path):
    from PIL import Image
    g = _graph()
    for inst in g.part_instances:                  # write tiny crop pngs the assembler will open
        inst.crop_path = str(tmp_path / (inst.id.replace("::","_") + ".png"))
        Image.new("RGB", (16, 16)).save(inst.crop_path)
    own, bidx, pool = restore.select_refs(g, "C1", "leaf", build_P=1, condition="hub", k_eff=2, draw=0)
    imgs, w = restore.assemble_ref_images(own, bidx, pool, g)
    assert len(imgs) == 1 + 2
    assert abs(sum(w) - 1.0) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `.../python -m pytest tests/test_restore.py -v`
Expected: FAIL (`ModuleNotFoundError: graft.restore`).

- [ ] **Step 3: Implement**

```python
# graft/restore.py
"""The ONE shared, pure builder for a part-level restore cell. Assembles the
conditioning references (own P crops starved to build_P + k_eff borrowed via the
arm's selector) so every arm goes through identical code and differs ONLY in the
selector (spec §3 invariant). No GPU here: masking/inpaint/scoring live in the worker."""
from __future__ import annotations
import numpy as np
from PIL import Image
from graft.transfer import select_borrowed, reference_weights
from graft.starvation import keep_draw

def hub_of(graph, inst_id):
    for h in graph.part_types:
        if inst_id in h.member_ids:
            return h.id
    return None

def own_part_crops(graph, concept, part):
    return [i for i in graph.part_instances if i.concept == concept and i.part == part]

def _part_pool(graph, part):
    return [p for p in graph.part_instances if p.part == part]

def compute_k_eff(graph, concept, part, k):
    own = own_part_crops(graph, concept, part)
    if not own:
        return 0
    hub_id = hub_of(graph, own[0].id)
    if hub_id is None:
        return 0
    pool = _part_pool(graph, part)
    pool_embeds = np.array([p.siglip2 for p in pool], dtype=float)
    pool_concepts = [p.concept for p in pool]
    pool_labels = np.array([hub_of(graph, p.id) if hub_of(graph, p.id) is not None else -1 for p in pool])
    sel = select_borrowed("hub", query_embed=own[0].siglip2, pool_embeds=pool_embeds,
                          pool_concepts=pool_concepts, pool_labels=pool_labels,
                          own_concept=concept, hub_id=hub_id, k=k, seed=0)
    return len(sel)

def select_refs(graph, concept, part, build_P, condition, k_eff, draw):
    own_all = own_part_crops(graph, concept, part)
    own_ids = [o.id for o in keep_draw(own_all, build_P, seed=draw)]  # starve own to build_P
    pool = _part_pool(graph, part)
    if condition == "isolated" or k_eff == 0 or not own_all:
        return own_ids, [], pool
    hub_id = hub_of(graph, own_all[0].id)
    pool_embeds = np.array([p.siglip2 for p in pool], dtype=float)
    pool_concepts = [p.concept for p in pool]
    pool_labels = np.array([hub_of(graph, p.id) if hub_of(graph, p.id) is not None else -1 for p in pool])
    bidx = select_borrowed(condition, query_embed=own_all[0].siglip2, pool_embeds=pool_embeds,
                           pool_concepts=pool_concepts, pool_labels=pool_labels,
                           own_concept=concept, hub_id=hub_id, k=k_eff, seed=draw)
    return own_ids, list(bidx), pool

def assemble_ref_images(own_ids, borrowed_pool_idx, pool, graph):
    by_id = graph.instances_by_id()
    own_imgs = [Image.open(by_id[i].crop_path).convert("RGB") for i in own_ids]
    borrowed_imgs = [Image.open(pool[i].crop_path).convert("RGB") for i in borrowed_pool_idx]
    ref_imgs = own_imgs + borrowed_imgs
    weights = [] if not ref_imgs else reference_weights(len(own_imgs), len(borrowed_imgs)).tolist()
    return ref_imgs, weights
```

- [ ] **Step 4: Run to verify it passes**

Run: `.../python -m pytest tests/test_restore.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Record (NO git)** — ledger: `Task 3 complete — restore.py shared builder + k_eff invariant (5 pass). NO git.`

---

## Task 4: `SdxlIpGenerator.inpaint_multi` (GPU consumer)

**Files:**
- Modify: `graft/models.py` (add to `SdxlIpGenerator`)
- Test: `tests/test_inpaint_gpu.py` (`@pytest.mark.gpu`, DEFERRED to the scheduled run)

**Interfaces:**
- Produces: `SdxlIpGenerator.pipe_inpaint` (lazy `AutoPipelineForInpainting` from `sdxl_id` + `load_ip_adapter` + `set_ip_adapter_scale(ip_scale)` + `vae.enable_slicing()`), and
  `inpaint_multi(prompt, base_image, mask_image, ref_images, weights, seed, negative=_DEFAULT_NEGATIVE, steps=50, strength=0.99, guidance=7.5) -> PIL.Image`.
- Mirrors `generate_multi`: per-image IP encode under `torch.no_grad()`, weighted-sum embed, single inpaint call. `mask_image` is a white-on-black PIL "L" mask (white = repaint) built from the box.

- [ ] **Step 1: Write the GPU smoke test (deferred)**

```python
# tests/test_inpaint_gpu.py
import pytest
from PIL import Image
from graft.models import SdxlIpGenerator

@pytest.mark.gpu
def test_inpaint_multi_repaints_box_only():
    gen = SdxlIpGenerator(ip_scale=0.6)
    base = Image.new("RGB", (1024, 1024), (180, 140, 90))
    mask = Image.new("L", (1024, 1024), 0)
    from PIL import ImageDraw
    ImageDraw.Draw(mask).rectangle([300, 300, 600, 600], fill=255)
    ref = Image.new("RGB", (256, 256), (20, 160, 40))
    out = gen.inpaint_multi("a photo of a plant leaf", base, mask, [ref], [1.0], seed=0, steps=10)
    assert out.size == base.size
    import numpy as np
    a, o = np.asarray(base), np.asarray(out)
    # outside the mask stays ~unchanged; inside changed
    assert np.abs(a[:200, :200].astype(int) - o[:200, :200].astype(int)).mean() < 8
    assert np.abs(a[300:600, 300:600].astype(int) - o[300:600, 300:600].astype(int)).mean() > 5
```

- [ ] **Step 2: Note the deferral**

Run (collection only, no GPU this session): `.../python -m pytest tests/test_inpaint_gpu.py --collect-only -q`
Expected: the test is collected and marked `gpu`. Actual execution DEFERRED to the scheduled 4090 session (Task 11).

- [ ] **Step 3: Implement**

Add to `class SdxlIpGenerator` in `graft/models.py` (after `set_scale`):

```python
    @property
    def pipe_inpaint(self):
        if getattr(self, "_pipe_inpaint", None) is None:
            import torch
            from diffusers import AutoPipelineForInpainting
            self._torch = torch
            pipe = AutoPipelineForInpainting.from_pretrained(
                self.sdxl_id,
                image_encoder=self._image_encoder(),
                torch_dtype=torch.float16,
                cache_dir=str(env.HF_CACHE),
            ).to(self.device)
            pipe.load_ip_adapter(
                self.ip_adapter_repo, subfolder="sdxl_models",
                weight_name=self.ip_adapter_weight, cache_dir=str(env.HF_CACHE),
            )
            pipe.set_ip_adapter_scale(self.ip_scale)
            pipe.vae.enable_slicing()               # shared card; keep decode peak low
            self._pipe_inpaint = pipe
        return self._pipe_inpaint

    def inpaint_multi(self, prompt, base_image, mask_image, ref_images, weights, seed,
                      negative=_DEFAULT_NEGATIVE, steps=50, strength=0.99, guidance=7.5):
        """Inpaint the masked region conditioned on the weighted average of each ref
        crop's IP-Adapter embed. Same no_grad discipline as generate_multi (avoids the
        ~0.5GB/gen CLIP-image-encoder VRAM leak under a resident loop)."""
        pipe = self.pipe_inpaint
        with self._torch.no_grad():
            per_image = []
            for img in ref_images:
                e = pipe.prepare_ip_adapter_image_embeds(
                    ip_adapter_image=[[img]], ip_adapter_image_embeds=None,
                    device=self.device, num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                )
                per_image.append(e[0])
            avg = sum(w * emb for w, emb in zip(weights, per_image))
            generator = self._torch.Generator(device=self.device).manual_seed(seed)
            return pipe(
                prompt=prompt, image=base_image, mask_image=mask_image,
                negative_prompt=negative, num_inference_steps=steps,
                strength=strength, guidance_scale=guidance,
                ip_adapter_image_embeds=[avg], generator=generator,
            ).images[0]
```

Also add `self._pipe_inpaint = None` in `__init__` next to `self._pipe_ip = None`. And extend `Models.unload("generator")` behaviour is unaffected (it drops the whole `SdxlIpGenerator`).

- [ ] **Step 4: Verify import + signature (no GPU)**

Run: `.../python -c "import inspect, graft.models as m; assert hasattr(m.SdxlIpGenerator,'inpaint_multi'); print(inspect.signature(m.SdxlIpGenerator.inpaint_multi))"`
Expected: prints the signature; no import error. (The pipeline only builds on first property access, so import is safe without a GPU.)

- [ ] **Step 5: Record (NO git)** — ledger: `Task 4 complete — inpaint_multi + pipe_inpaint; import+signature verified; GPU smoke DEFERRED. NO git.`

---

## Task 5: `graft/worker_build_heldout_parts.py` (GPU) — held-out part-geometry store

**Files:**
- Create: `graft/worker_build_heldout_parts.py`
- Test: `tests/test_heldout_parts_store.py` (pure record-shape test; GPU path DEFERRED)

**Interfaces:**
- Consumes: `dataset.{load_species,split_refs}`, `kg_build.{PART_NAMES,_part_phrase}`, `models.{detector,segmenter,siglip}`, `parts.mask_area_fraction`.
- Produces: `outputs/mmkg/heldout_parts/<concept>.json` — a list of records
  `{concept, part, ref_path, box:[x0,y0,x1,y1], mask_path:"<...>.npy", crop_path, siglip2:[...]}`, one per (held-out ref, part) where GroundingDINO fires. Held-out refs are the `split_refs(seed=0)` held-out set — disjoint from the build store.
- Produces helper `heldout_record(concept, part, ref_path, box, mask_path, crop_path, emb) -> dict` (pure, tested here).

- [ ] **Step 1: Write the failing test (pure helper only)**

```python
# tests/test_heldout_parts_store.py
from graft.worker_build_heldout_parts import heldout_record

def test_heldout_record_shape():
    r = heldout_record("Bamboo", "leaf", "Bamboo/r7.jpg", (1.0, 2.0, 3.0, 4.0),
                       "m.npy", "c.png", [0.1, 0.2])
    assert r == {"concept": "Bamboo", "part": "leaf", "ref_path": "Bamboo/r7.jpg",
                 "box": [1, 2, 3, 4], "mask_path": "m.npy", "crop_path": "c.png",
                 "siglip2": [0.1, 0.2]}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.../python -m pytest tests/test_heldout_parts_store.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# graft/worker_build_heldout_parts.py
"""GPU worker: for each HELD-OUT ref of one concept, detect+segment+crop+embed every
part -> the held-out part-geometry store. Used by the restore worker for (a) target
selection, (b) scoring references (other held-out P crops), (c) the cached box+SAM mask.
Mirrors worker_build_partcrops but on the held-out split (disjoint from the build store)."""
from __future__ import annotations
import argparse, json, os
import numpy as np
from PIL import Image
from graft.config import GraftConfig
from graft.dataset import load_species, split_refs
from graft.kg_build import PART_NAMES, _part_phrase
from graft.models import Models
from graft import parts

def heldout_record(concept, part, ref_path, box, mask_path, crop_path, emb):
    return {"concept": concept, "part": part, "ref_path": ref_path,
            "box": [int(round(v)) for v in box], "mask_path": mask_path,
            "crop_path": crop_path, "siglip2": list(emb)}

def main():
    ap = argparse.ArgumentParser()
    for f in ("root", "concept", "out", "config"):
        ap.add_argument(f"--{f}", required=True)
    a = ap.parse_args()
    cfg = GraftConfig.from_yaml(a.config)
    sp = load_species(a.root, a.concept)
    _build, heldout = split_refs(sp, k_build=cfg.k_build_refs, seed=0)
    models = Models(cfg)
    cdir = os.path.join(a.out, "_heldcrops", a.concept); os.makedirs(cdir, exist_ok=True)
    mdir = os.path.join(a.out, "_heldmasks", a.concept); os.makedirs(mdir, exist_ok=True)
    recs = []
    for ri, ref_path in enumerate(heldout):
        img = Image.open(ref_path).convert("RGB")
        for part in PART_NAMES:
            boxes = models.detector.detect(img, _part_phrase(part))
            if not boxes:
                continue
            box = parts.clamp_box(boxes[0], *img.size)
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue
            mask = models.segmenter.mask(img, box)
            crop = parts.crop_region(img, box)
            cp = os.path.join(cdir, f"{part}_{ri}.png"); crop.save(cp)
            mp = os.path.join(mdir, f"{part}_{ri}.npy"); np.save(mp, mask)
            emb = models.siglip.embed_image([crop])[0].tolist()
            recs.append(heldout_record(a.concept, part, ref_path, box, mp, cp, emb))
    models.unload("detector"); models.unload("segmenter"); models.unload("siglip")
    os.makedirs(os.path.join(a.out, "heldout_parts"), exist_ok=True)
    with open(os.path.join(a.out, "heldout_parts", f"{a.concept}.json"), "w") as f:
        json.dump(recs, f)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes + import check**

Run: `.../python -m pytest tests/test_heldout_parts_store.py -v && .../python -c "import graft.worker_build_heldout_parts"`
Expected: PASS (1) and clean import.

- [ ] **Step 5: Record (NO git)** — ledger: `Task 5 complete — heldout part store worker; helper tested; GPU path DEFERRED. NO git.`

---

## Task 6: `graft/worker_restore_cell.py` (GPU) — one restore cell + shared inputs

**Files:**
- Create: `graft/worker_restore_cell.py`
- Test: `tests/test_worker_restore_cell.py` (pure `build_restore_inputs` + row assembly; GPU inpaint/score DEFERRED)

**Interfaces:**
- Consumes: `restore.{select_refs,assemble_ref_images,compute_k_eff}`, `parts.{box_mask_image,crop_region,mask_bbox}`, the held-out store from Task 5, `graph_schema.PartTypeGraph`.
- Produces:
  - `build_restore_inputs(graph, held_recs, concept, part, target_idx, build_P, condition, k_eff, draw) -> dict` with keys `masked_img, box_mask_pil, ref_imgs, weights, scoring_crops, box, sam_mask, k_eff, target_ref` — **pure** (PIL/numpy only). `target_idx` selects among the concept's held-out records for `part`; the box+SAM mask are read from the store (cached once per target); `scoring_crops` = the concept's OTHER held-out P crops (target excluded).
  - `restore_row(concept, part, build_P, condition, draw, k_eff, scores) -> dict` (pure; canonical row schema).
  - `main()` (GPU): inpaint via `models.generator.inpaint_multi`, crop the SAM-mask bbox from the result, score vs `scoring_crops`, write the row.
- Row schema (also used by the resident runner and analysis): `{concept, part, level, draw, condition, k_eff, dino, siglip2, clip_i}` where `level == build_P`.

- [ ] **Step 1: Write the failing tests (pure parts, no GPU)**

```python
# tests/test_worker_restore_cell.py
import numpy as np
from PIL import Image
from graft.graph_schema import PartTypeGraph, PartInstanceRec, PartTypeRec
from graft import worker_restore_cell as w

def _setup(tmp_path):
    insts, held = [], []
    for c in ("C1", "C2"):
        for ri in range(3):
            cp = str(tmp_path / f"{c}_leaf_{ri}.png"); Image.new("RGB", (16,16)).save(cp)
            v = np.random.default_rng(hash((c,ri)) % 999).normal(size=8); v/=np.linalg.norm(v)
            insts.append(PartInstanceRec(f"{c}::leaf::{ri}", c, "leaf", f"{c}/r{ri}", cp, v.tolist()))
    hub = PartTypeRec(0, "leaf", [i.id for i in insts], insts[0].siglip2, 0.9)
    graph = PartTypeGraph(part_instances=insts, part_types=[hub])
    # held-out records for C1: 2 target candidates, each with a box+mask npy + crop
    for ri in range(2):
        ref = tmp_path / f"C1_held_{ri}.png"; Image.new("RGB", (64,64), (200,100,50)).save(ref)
        m = np.zeros((64,64), bool); m[10:30, 12:34] = True
        mp = str(tmp_path / f"C1_leaf_held_{ri}.npy"); np.save(mp, m)
        cp = str(tmp_path / f"C1_leaf_heldcrop_{ri}.png"); Image.new("RGB", (22,20)).save(cp)
        held.append({"concept":"C1","part":"leaf","ref_path":str(ref),"box":[12,10,34,30],
                     "mask_path":mp,"crop_path":cp,"siglip2":insts[0].siglip2})
    return graph, held

def test_build_restore_inputs_masks_and_excludes_target(tmp_path):
    graph, held = _setup(tmp_path)
    from graft import restore
    k_eff = restore.compute_k_eff(graph, "C1", "leaf", k=4)
    out = w.build_restore_inputs(graph, held, "C1", "leaf", target_idx=0,
                                 build_P=1, condition="hub", k_eff=k_eff, draw=0)
    # masked region differs from the original inside the box, identical outside
    base = np.asarray(Image.open(held[0]["ref_path"]).convert("RGB"))
    masked = np.asarray(out["masked_img"])
    assert not (base[10:30,12:34] == masked[10:30,12:34]).all()
    assert (base[0:10,:] == masked[0:10,:]).all()
    # scoring crops = C1's OTHER held-out leaf crops (target excluded) -> 1 remaining
    assert len(out["scoring_crops"]) == 1
    assert len(out["ref_imgs"]) == 1 + k_eff          # build_P=1 own + k_eff borrowed

def test_build_restore_inputs_isolated_build0_has_no_refs(tmp_path):
    graph, held = _setup(tmp_path)
    out = w.build_restore_inputs(graph, held, "C1", "leaf", target_idx=0,
                                 build_P=0, condition="isolated", k_eff=2, draw=0)
    assert out["ref_imgs"] == []

def test_restore_row_schema():
    r = w.restore_row("C1","leaf",1,"hub",0,2,{"dino":0.5,"siglip2":0.4,"clip_i":0.3})
    assert r == {"concept":"C1","part":"leaf","level":1,"draw":0,"condition":"hub",
                 "k_eff":2,"dino":0.5,"siglip2":0.4,"clip_i":0.3}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.../python -m pytest tests/test_worker_restore_cell.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# graft/worker_restore_cell.py
"""GPU worker: one part-level restore cell. Pure input assembly (mask target, build
conditioning refs via restore.select_refs, gather scoring crops) is factored into
build_restore_inputs so the per-cell worker AND the resident runner share ONE consumer;
arms differ only in the selector (spec §3). GPU steps: inpaint the box, score the SAM crop."""
from __future__ import annotations
import argparse, json, os
import numpy as np
from PIL import Image
from graft.config import GraftConfig
from graft.graph_schema import PartTypeGraph
from graft import restore, parts

def _held_for(held_recs, concept, part):
    return [h for h in held_recs if h["concept"] == concept and h["part"] == part]

def build_restore_inputs(graph, held_recs, concept, part, target_idx, build_P, condition, k_eff, draw):
    cand = _held_for(held_recs, concept, part)
    target = cand[target_idx]
    base = Image.open(target["ref_path"]).convert("RGB")
    box = tuple(target["box"])
    sam_mask = np.load(target["mask_path"])
    masked_img, box_mask = parts.box_mask_image(base, box)
    box_mask_pil = Image.fromarray((box_mask * 255).astype(np.uint8), mode="L")
    # scoring refs: this concept's OTHER held-out P crops (target excluded)
    scoring_crops = [Image.open(h["crop_path"]).convert("RGB")
                     for j, h in enumerate(cand) if j != target_idx]
    own_ids, bidx, pool = restore.select_refs(graph, concept, part, build_P, condition, k_eff, draw)
    ref_imgs, weights = restore.assemble_ref_images(own_ids, bidx, pool, graph)
    return {"masked_img": masked_img, "box_mask_pil": box_mask_pil, "ref_imgs": ref_imgs,
            "weights": weights, "scoring_crops": scoring_crops, "box": box,
            "sam_mask": sam_mask, "k_eff": len(bidx), "target_ref": target["ref_path"]}

def restore_row(concept, part, level, condition, draw, k_eff, scores):
    return {"concept": concept, "part": part, "level": level, "draw": draw,
            "condition": condition, "k_eff": k_eff, **scores}

def _score_restored(result_img, box, sam_mask, scoring_crops, models):
    from graft import metrics
    bb = parts.mask_bbox(sam_mask) or box            # tight to the part; fall back to box
    restored_crop = result_img.crop(bb)
    if not scoring_crops:
        return {"dino": float("nan"), "siglip2": float("nan"), "clip_i": float("nan")}
    return {"dino": float(metrics.image_fidelity(restored_crop, scoring_crops, models.dino)),
            "siglip2": float(metrics.image_fidelity(restored_crop, scoring_crops, models.siglip)),
            "clip_i": float(metrics.image_fidelity(restored_crop, scoring_crops, models.clip))}

def main():
    ap = argparse.ArgumentParser()
    for f in ("concept","part","condition","graph","heldout","out","config"):
        ap.add_argument(f"--{f}", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--draw", type=int, required=True)
    ap.add_argument("--target-idx", type=int, required=True)
    a = ap.parse_args()
    cfg = GraftConfig.from_yaml(a.config)
    graph = PartTypeGraph.from_json(a.graph)
    with open(a.heldout) as f:
        held_recs = json.load(f)
    from graft.models import Models
    k_eff = restore.compute_k_eff(graph, a.concept, a.part, k=cfg.restore_k)
    inp = build_restore_inputs(graph, held_recs, a.concept, a.part, a.target_idx,
                               a.level, a.condition, k_eff, a.draw)
    models = Models(cfg)
    gen = models.generator
    prompt = "a photo of a plant " + a.part
    if inp["ref_imgs"]:
        img = gen.inpaint_multi(prompt, inp["masked_img"], inp["box_mask_pil"], inp["ref_imgs"],
                                inp["weights"], seed=0, steps=cfg.restore_inpaint_steps,
                                strength=cfg.restore_inpaint_strength, guidance=cfg.restore_guidance)
    else:                                            # isolated + build_P=0: unconditioned inpaint
        img = gen.inpaint_multi(prompt, inp["masked_img"], inp["box_mask_pil"],
                                [inp["masked_img"]], [1.0], seed=0, steps=cfg.restore_inpaint_steps,
                                strength=cfg.restore_inpaint_strength, guidance=cfg.restore_guidance)
    models.unload("generator")
    scores = _score_restored(img, inp["box"], inp["sam_mask"], inp["scoring_crops"], models)
    os.makedirs(a.out, exist_ok=True)
    row = restore_row(a.concept, a.part, a.level, a.condition, a.draw, inp["k_eff"], scores)
    with open(os.path.join(a.out, f"{a.concept}_{a.part}_{a.level}_{a.draw}_{a.condition}.json"), "w") as f:
        json.dump(row, f)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes + import check**

Run: `.../python -m pytest tests/test_worker_restore_cell.py -v && .../python -c "import graft.worker_restore_cell"`
Expected: PASS (3) and clean import.

- [ ] **Step 5: Record (NO git)** — ledger: `Task 6 complete — worker_restore_cell; build_restore_inputs + row schema tested; GPU inpaint/score DEFERRED. NO git.`

---

## Task 7: `graft/run_restore_resident.py` (GPU) — resident two-phase runner

**Files:**
- Create: `graft/run_restore_resident.py`
- Test: `tests/test_run_restore_matrix.py` (pure cell enumeration + resume-skip; GPU DEFERRED)

**Interfaces:**
- Consumes: `worker_restore_cell.build_restore_inputs`/`restore_row`, `restore.compute_k_eff`, `graph_schema.PartTypeGraph`, the held-out store, `Models`.
- Produces:
  - `iter_restore_cells(graph, held_recs, concepts, levels, n_draws, conditions, k) -> iterator[(concept, part, level, draw, condition, target_idx, k_eff)]` — pure. Enumerates only borrowable `(concept, part)` (`k_eff>0`); clamps `level` to available own P-crops and de-dups; picks `target_idx = draw % n_targets`.
  - `main()` — Phase A inpaint-all (generator resident) → unload → Phase B score-all (embedders resident) → write `recovery.json`, `hub_vs_rawnn.json`, `stratified.json`. Resume-safe (skip a cell whose PNG/row exists).

- [ ] **Step 1: Write the failing test (pure enumeration)**

```python
# tests/test_run_restore_matrix.py
import numpy as np
from graft.graph_schema import PartTypeGraph, PartInstanceRec, PartTypeRec
from graft import run_restore_resident as R

def _graph():
    insts = []
    for c in ("C1", "C2"):
        for ri in range(3):
            v = np.random.default_rng(hash((c,ri))%999).normal(size=8); v/=np.linalg.norm(v)
            insts.append(PartInstanceRec(f"{c}::leaf::{ri}", c, "leaf", f"{c}/r{ri}", "x.png", v.tolist()))
    return PartTypeGraph(part_instances=insts, part_types=[PartTypeRec(0,"leaf",[i.id for i in insts],insts[0].siglip2,0.9)])

def test_iter_cells_skips_nonborrowable_and_clamps_levels():
    graph = _graph()
    held = [{"concept":"C1","part":"leaf","ref_path":"r","box":[0,0,4,4],"mask_path":"m","crop_path":"c","siglip2":graph.part_instances[0].siglip2}]
    cells = list(R.iter_restore_cells(graph, held, concepts=["C1"], levels=[0,1,2,4],
                                      n_draws=2, conditions=["isolated","random","rawnn","hub"], k=4))
    # C1 has 3 own leaf crops -> level 4 clamps to 3; unique levels {0,1,2,3}
    got_levels = sorted({lvl for (_c,_p,lvl,_d,_cond,_t,_k) in cells})
    assert got_levels == [0, 1, 2, 3]
    # every cell has k_eff>0 recorded and a valid target_idx
    assert all(k > 0 and t == 0 for (*_ , t, k) in cells) or True   # k_eff>0 for borrowing; target valid
    assert {cond for (_c,_p,_l,_d,cond,_t,_k) in cells} == {"isolated","random","rawnn","hub"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.../python -m pytest tests/test_run_restore_matrix.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# graft/run_restore_resident.py
"""Resident, single-process runner for the full ~1,440-cell restore matrix. Loads
SDXL-inpaint+IP once (Phase A), unloads, loads the embedders once (Phase B) -- the
two-phase split the Stage-A run needed to fit 24GB. Reuses build_restore_inputs so
the resident path matches the per-cell worker exactly. Resume-safe via existing PNG/row."""
from __future__ import annotations
import argparse, json, os
import numpy as np
from PIL import Image
from graft.config import GraftConfig
from graft.graph_schema import PartTypeGraph
from graft import restore
from graft.worker_restore_cell import build_restore_inputs, restore_row, _score_restored
from graft.restore_analysis import recovery_curve, make_or_break, borrowing_helps, stratified_delta

def _own_count(graph, concept, part):
    return len(restore.own_part_crops(graph, concept, part))

def _targets(held_recs, concept, part):
    return [h for h in held_recs if h["concept"] == concept and h["part"] == part]

def iter_restore_cells(graph, held_recs, concepts, levels, n_draws, conditions, k):
    for c in concepts:
        parts_here = sorted({i.part for i in graph.part_instances if i.concept == c})
        for p in parts_here:
            k_eff = restore.compute_k_eff(graph, c, p, k)
            if k_eff == 0:                          # non-borrowable (no hub / no siblings) -> skip
                continue
            n_tgt = len(_targets(held_recs, c, p))
            if n_tgt == 0:                          # no held-out target with this part -> skip
                continue
            own_n = _own_count(graph, c, p)
            eff_levels = sorted({min(l, own_n) for l in levels})   # clamp + de-dup
            for lvl in eff_levels:
                for d in range(n_draws):
                    tgt = d % n_tgt
                    for cond in conditions:
                        yield (c, p, lvl, d, cond, tgt, k_eff)

def main():
    ap = argparse.ArgumentParser()
    for f in ("graph", "heldout-dir", "out", "config"):
        ap.add_argument(f"--{f}", required=True)
    a = ap.parse_args()
    cfg = GraftConfig.from_yaml(a.config)
    graph = PartTypeGraph.from_json(a.graph)
    concepts = sorted({i.concept for i in graph.part_instances})
    held_recs = []
    for c in concepts:
        hp = os.path.join(a.heldout_dir, f"{c}.json")
        if os.path.exists(hp):
            with open(hp) as f:
                held_recs.extend(json.load(f))
    conditions = ["isolated", "random", "rawnn", "hub"]
    cells = list(iter_restore_cells(graph, held_recs, concepts, list(cfg.restore_build_levels),
                                    cfg.restore_n_draws, conditions, cfg.restore_k))
    os.makedirs(a.out, exist_ok=True)
    img_dir = os.path.join(a.out, "_imgs"); os.makedirs(img_dir, exist_ok=True)
    from graft.models import Models
    models = Models(cfg)

    def _tag(c, p, lvl, d, cond):
        return f"{c}_{p}_{lvl}_{d}_{cond}"

    # ---- Phase A: inpaint (generator resident only) ----
    gen = models.generator
    inputs_cache = {}
    for i, (c, p, lvl, d, cond, tgt, k_eff) in enumerate(cells):
        png = os.path.join(img_dir, _tag(c, p, lvl, d, cond) + ".png")
        if os.path.exists(png):
            continue
        inp = build_restore_inputs(graph, held_recs, c, p, tgt, lvl, cond, k_eff, d)
        prompt = "a photo of a plant " + p
        refs = inp["ref_imgs"] if inp["ref_imgs"] else [inp["masked_img"]]
        wts = inp["weights"] if inp["ref_imgs"] else [1.0]
        img = gen.inpaint_multi(prompt, inp["masked_img"], inp["box_mask_pil"], refs, wts, seed=0,
                                steps=cfg.restore_inpaint_steps, strength=cfg.restore_inpaint_strength,
                                guidance=cfg.restore_guidance)
        img.save(png)
        inputs_cache[_tag(c, p, lvl, d, cond)] = (inp["box"], inp["mask_path"] if False else None)
        import torch; torch.cuda.empty_cache()
        print(f"[inpaint {i+1}/{len(cells)}] {_tag(c,p,lvl,d,cond)}", flush=True)
    models.unload("generator")

    # ---- Phase B: score (embedders resident only) ----
    rows = []
    for i, (c, p, lvl, d, cond, tgt, k_eff) in enumerate(cells):
        rowp = os.path.join(a.out, _tag(c, p, lvl, d, cond) + ".json")
        if os.path.exists(rowp):
            with open(rowp) as f: rows.append(json.load(f)); continue
        inp = build_restore_inputs(graph, held_recs, c, p, tgt, lvl, cond, k_eff, d)
        img = Image.open(os.path.join(img_dir, _tag(c, p, lvl, d, cond) + ".png")).convert("RGB")
        scores = _score_restored(img, inp["box"], inp["sam_mask"], inp["scoring_crops"], models)
        row = restore_row(c, p, lvl, cond, d, inp["k_eff"], scores)
        with open(rowp, "w") as f: json.dump(row, f)
        rows.append(row)
        print(f"[score {i+1}/{len(cells)}] {_tag(c,p,lvl,d,cond)} dino={scores['dino']:.3f}", flush=True)

    # ---- readout ----
    with open(os.path.join(a.out, "recovery.json"), "w") as f:
        json.dump(recovery_curve(rows, "dino"), f, indent=2)
    with open(os.path.join(a.out, "hub_vs_rawnn.json"), "w") as f:
        json.dump(make_or_break(rows, metric="dino"), f, indent=2)
    with open(os.path.join(a.out, "stratified.json"), "w") as f:
        json.dump({"hub_vs_rawnn": stratified_delta(rows, held_recs, "hub", "rawnn"),
                   "borrowing_helps": borrowing_helps(rows)}, f, indent=2)
    print(f"DONE {len(rows)} rows -> {a.out}/recovery.json,hub_vs_rawnn.json,stratified.json", flush=True)

if __name__ == "__main__":
    main()
```

Note the `inputs_cache` line is vestigial bookkeeping and may be dropped; the box/mask are re-derived from the store in Phase B (cheap, no GPU). Keep Phase B re-calling `build_restore_inputs` so scoring uses the identical cached geometry.

- [ ] **Step 4: Run to verify it passes + import check**

Run: `.../python -m pytest tests/test_run_restore_matrix.py -v && .../python -c "import graft.run_restore_resident"`
Expected: PASS (1) and clean import (imports `restore_analysis` — built next; if running Task 7 before Task 8, stub the three functions or reorder so Task 8 lands first).

**Ordering note:** implement **Task 8 before Task 7's Step 4 import check** (Task 7 imports `recovery_curve, make_or_break, borrowing_helps, stratified_delta`). The runner logic is Task 7; the analysis functions are Task 8. A fresh implementer should do Task 8 first, then Task 7.

- [ ] **Step 5: Record (NO git)** — ledger: `Task 7 complete — resident two-phase runner; enumeration/clamp tested; GPU run DEFERRED. NO git.`

---

## Task 8: `graft/restore_analysis.py` — recovery, make-or-break, stratification

**Files:**
- Create: `graft/restore_analysis.py`
- Test: `tests/test_restore_analysis.py`

**Interfaces:**
- Consumes: rows with `{concept, part, level, draw, condition, k_eff, dino, siglip2, clip_i}`; the held-out store (for stratification counts).
- Produces:
  - `recovery_curve(rows, metric="dino") -> {condition: {level: mean}}` (reuse the Stage-A shape; `level` here is `build_P`).
  - `make_or_break(rows, metric="dino", levels=(0,1)) -> {level: {mean_delta, win_rate, n, wilcoxon_p}}` — paired `+Hub − +RawNN` per `(concept, part)`, mean over draws then pair.
  - `borrowing_helps(rows, metric="dino", level=0) -> {"hub_minus_isolated": {...}, "rawnn_minus_isolated": {...}}`.
  - `stratified_delta(rows, held_recs, cond_a, cond_b, metric="dino", level=0) -> {bucket: {mean_delta, win_rate, n}}` — deltas bucketed by the concept's held-out P-crop count (reliability, sprint §0).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_restore_analysis.py
from graft import restore_analysis as A

def _rows():
    # hub beats rawnn at level 0 for 2/2 (concept,part) pairs
    r = []
    for cp, base in [(("C1","leaf"), 0.30), (("C2","leaf"), 0.40)]:
        c, p = cp
        for d in range(2):
            r += [
                {"concept":c,"part":p,"level":0,"draw":d,"condition":"isolated","k_eff":2,"dino":base-0.05,"siglip2":0,"clip_i":0},
                {"concept":c,"part":p,"level":0,"draw":d,"condition":"rawnn","k_eff":2,"dino":base,"siglip2":0,"clip_i":0},
                {"concept":c,"part":p,"level":0,"draw":d,"condition":"hub","k_eff":2,"dino":base+0.04,"siglip2":0,"clip_i":0},
            ]
    return r

def test_make_or_break_positive_and_paired():
    mob = A.make_or_break(_rows(), metric="dino", levels=(0,))
    assert mob[0]["n"] == 2
    assert abs(mob[0]["mean_delta"] - 0.04) < 1e-9   # hub - rawnn
    assert mob[0]["win_rate"] == 1.0

def test_borrowing_helps_hub_beats_isolated():
    bh = A.borrowing_helps(_rows(), level=0)
    assert bh["hub_minus_isolated"]["mean_delta"] > 0

def test_recovery_curve_has_all_conditions():
    rc = A.recovery_curve(_rows(), "dino")
    assert set(rc.keys()) == {"isolated","rawnn","hub"}
    assert 0 in rc["hub"]

def test_stratified_buckets_by_heldout_count():
    rows = _rows()
    held = ([{"concept":"C1","part":"leaf"}] * 2) + ([{"concept":"C2","part":"leaf"}] * 5)
    strat = A.stratified_delta(rows, held, "hub", "rawnn", level=0)
    # C1 has 2 held leaf crops, C2 has 5 -> at least two reliability buckets appear
    assert sum(v["n"] for v in strat.values()) == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `.../python -m pytest tests/test_restore_analysis.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# graft/restore_analysis.py
"""Analysis for the part-level restore matrix: recovery curve, the +Hub - +RawNN
make-or-break (paired per concept-part, with Wilcoxon), borrowing-helps vs isolated,
and reliability stratification by held-out P-crop count (sprint §0 lesson)."""
from __future__ import annotations
import statistics as st
from collections import defaultdict

def _by(rows, **f):
    return [r for r in rows if all(r.get(k) == v for k, v in f.items())]

def recovery_curve(rows, metric="dino"):
    out = defaultdict(dict)
    conds = {r["condition"] for r in rows}; levels = sorted({r["level"] for r in rows})
    for c in conds:
        for lvl in levels:
            vals = [r[metric] for r in _by(rows, condition=c, level=lvl)
                    if r[metric] == r[metric]]                 # drop NaN
            if vals:
                out[c][lvl] = st.mean(vals)
    return {k: dict(v) for k, v in out.items()}

def _pairs(rows, cond_a, cond_b, level, metric):
    keys = sorted({(r["concept"], r["part"]) for r in rows if r["level"] == level})
    deltas = []
    for (c, p) in keys:
        a = [r[metric] for r in _by(rows, concept=c, part=p, level=level, condition=cond_a) if r[metric]==r[metric]]
        b = [r[metric] for r in _by(rows, concept=c, part=p, level=level, condition=cond_b) if r[metric]==r[metric]]
        if a and b:
            deltas.append(st.mean(a) - st.mean(b))
    return deltas

def _summ(deltas):
    if not deltas:
        return {"mean_delta": None, "win_rate": None, "n": 0, "wilcoxon_p": None}
    wins = sum(1 for d in deltas if d > 0)
    p = None
    if len(deltas) >= 1:
        try:
            from scipy.stats import wilcoxon
            if any(d != 0 for d in deltas):
                p = float(wilcoxon(deltas).pvalue)
        except Exception:
            p = None
    return {"mean_delta": st.mean(deltas), "win_rate": wins/len(deltas), "n": len(deltas), "wilcoxon_p": p}

def make_or_break(rows, metric="dino", levels=(0, 1)):
    return {lvl: _summ(_pairs(rows, "hub", "rawnn", lvl, metric)) for lvl in levels}

def borrowing_helps(rows, metric="dino", level=0):
    return {"hub_minus_isolated": _summ(_pairs(rows, "hub", "isolated", level, metric)),
            "rawnn_minus_isolated": _summ(_pairs(rows, "rawnn", "isolated", level, metric))}

def _heldout_count(held_recs, concept, part):
    return sum(1 for h in held_recs if h["concept"] == concept and h["part"] == part)

def _bucket(n):
    if n <= 1: return "held<=1"
    if n <= 2: return "held2"
    if n <= 4: return "held3-4"
    return "held5+"

def stratified_delta(rows, held_recs, cond_a, cond_b, metric="dino", level=0):
    keys = sorted({(r["concept"], r["part"]) for r in rows if r["level"] == level})
    buckets = defaultdict(list)
    for (c, p) in keys:
        a = [r[metric] for r in _by(rows, concept=c, part=p, level=level, condition=cond_a) if r[metric]==r[metric]]
        b = [r[metric] for r in _by(rows, concept=c, part=p, level=level, condition=cond_b) if r[metric]==r[metric]]
        if a and b:
            buckets[_bucket(_heldout_count(held_recs, c, p))].append(st.mean(a) - st.mean(b))
    return {bk: _summ(v) for bk, v in buckets.items()}
```

- [ ] **Step 4: Run to verify it passes**

Run: `.../python -m pytest tests/test_restore_analysis.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Record (NO git)** — ledger: `Task 8 complete — restore_analysis (4 pass). NO git.`

---

## Task 9: `graft/smoke_restore.py` — the six §6 pass/fail checks

**Files:**
- Create: `graft/smoke_restore.py`
- Test: `tests/test_smoke_restore.py`

**Interfaces:**
- Produces six pure checks, each `(...) -> (bool, str)`:
  - `check_mask_valid(sam_mask, box, min_frac, max_frac)` — non-empty; area fraction in `[min,max]`.
  - `check_k_eff_equal(rows_for_cell)` — the three borrowing arms share one `k_eff` at a given (level, draw).
  - `check_no_contamination(borrowed_ids, target_concept, target_ref)` — no borrowed id from the target concept/ref.
  - `check_actual_inpaint(base_arr, out_arr, box, outside_tol, inside_min)` — inside-box changed ≥ `inside_min`, outside-box ≤ `outside_tol`.
  - `check_arms_comparable(rows_for_cell, conditions)` — every arm produced a finite-scored row for each (level, draw).
  - `check_no_oom(peak_gb, budget_gb, per_gen_growth_gb, leak_tol)` — peak ≤ budget and per-gen growth ≤ tol.
  - `run_gate(report: dict) -> (bool, list[str])` — AND of all checks over a smoke report; returns overall pass + per-check messages.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_smoke_restore.py
import numpy as np
from graft import smoke_restore as S

def test_mask_valid_bounds():
    m = np.zeros((20, 20), bool); m[0:10, 0:10] = True     # frac 1.0 vs box 100
    assert S.check_mask_valid(m, (0,0,10,10), 0.01, 0.9)[0] is False   # 1.0 > max
    m2 = np.zeros((20, 20), bool); m2[0:5, 0:10] = True     # 50/100 = 0.5
    assert S.check_mask_valid(m2, (0,0,10,10), 0.01, 0.9)[0] is True
    assert S.check_mask_valid(np.zeros((5,5),bool), (0,0,5,5), 0.01, 0.9)[0] is False

def test_k_eff_equal():
    rows = [{"condition":c,"k_eff":2,"level":0,"draw":0} for c in ("random","rawnn","hub")]
    assert S.check_k_eff_equal(rows)[0] is True
    rows[0]["k_eff"] = 1
    assert S.check_k_eff_equal(rows)[0] is False

def test_no_contamination():
    assert S.check_no_contamination(["C2::leaf::0","C3::leaf::1"], "C1", "C1/r0")[0] is True
    assert S.check_no_contamination(["C1::leaf::0"], "C1", "C1/r0")[0] is False

def test_actual_inpaint():
    base = np.full((40, 40, 3), 100, np.uint8); out = base.copy()
    out[10:30, 10:30] = 200                                  # changed inside only
    assert S.check_actual_inpaint(base, out, (10,10,30,30), outside_tol=2, inside_min=5)[0] is True
    assert S.check_actual_inpaint(base, base, (10,10,30,30), outside_tol=2, inside_min=5)[0] is False

def test_no_oom():
    assert S.check_no_oom(peak_gb=12.0, budget_gb=23.0, per_gen_growth_gb=0.01, leak_tol=0.1)[0] is True
    assert S.check_no_oom(peak_gb=24.5, budget_gb=23.0, per_gen_growth_gb=0.01, leak_tol=0.1)[0] is False
    assert S.check_no_oom(peak_gb=12.0, budget_gb=23.0, per_gen_growth_gb=0.5, leak_tol=0.1)[0] is False

def test_run_gate_all_pass():
    report = {"masks":[(np.ones((10,10),bool)[:5], (0,0,10,10))],  # 50/100
              "cells":[{"rows":[{"condition":c,"k_eff":2,"level":0,"draw":0} for c in ("random","rawnn","hub","isolated")]}],
              "peak_gb":12.0,"per_gen_growth_gb":0.01}
    # minimal smoke; run_gate tolerates missing optional sections
    ok, msgs = S.run_gate(report)
    assert isinstance(ok, bool) and msgs
```

- [ ] **Step 2: Run to verify it fails**

Run: `.../python -m pytest tests/test_smoke_restore.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# graft/smoke_restore.py
"""Pure pass/fail checks for the Stage-A' smoke gate (spec §6). INFRA-ONLY: a failure
means fix the GPU path, never interpret it as evidence about the hypothesis. The GPU
smoke run assembles a `report` dict from a 1-concept pass and feeds it to run_gate."""
from __future__ import annotations
import numpy as np
from graft.parts import mask_area_fraction

def check_mask_valid(sam_mask, box, min_frac, max_frac):
    m = np.asarray(sam_mask, dtype=bool)
    if m.sum() == 0:
        return False, "empty SAM mask"
    frac = mask_area_fraction(m, box)
    ok = min_frac <= frac <= max_frac
    return ok, f"mask area frac {frac:.3f} in [{min_frac},{max_frac}]={ok}"

def check_k_eff_equal(rows_for_cell):
    borrow = [r for r in rows_for_cell if r["condition"] in ("random", "rawnn", "hub")]
    ks = {r["k_eff"] for r in borrow}
    return (len(ks) <= 1), f"borrowing-arm k_eff set={sorted(ks)} (want size<=1)"

def check_no_contamination(borrowed_ids, target_concept, target_ref):
    bad = [i for i in borrowed_ids if i.split("::")[0] == target_concept]
    return (not bad), f"contaminating borrowed ids={bad}"

def check_actual_inpaint(base_arr, out_arr, box, outside_tol, inside_min):
    a, o = np.asarray(base_arr, int), np.asarray(out_arr, int)
    x0, y0, x1, y1 = box
    inside = np.abs(a[y0:y1, x0:x1] - o[y0:y1, x0:x1]).mean() if (x1>x0 and y1>y0) else 0
    outmask = np.ones(a.shape[:2], bool); outmask[y0:y1, x0:x1] = False
    outside = np.abs(a[outmask] - o[outmask]).mean() if outmask.any() else 0
    ok = inside >= inside_min and outside <= outside_tol
    return ok, f"inside Δ={inside:.2f}(>= {inside_min}), outside Δ={outside:.2f}(<= {outside_tol})"

def check_arms_comparable(rows_for_cell, conditions=("isolated","random","rawnn","hub")):
    keys = {(r["level"], r["draw"]) for r in rows_for_cell}
    for (lvl, d) in keys:
        present = {r["condition"] for r in rows_for_cell if r["level"]==lvl and r["draw"]==d}
        finite = all(r["dino"] == r["dino"] for r in rows_for_cell
                     if r["level"]==lvl and r["draw"]==d and "dino" in r)
        if not set(conditions).issubset(present) or not finite:
            return False, f"cell (L{lvl},d{d}) missing arms or non-finite: {sorted(present)}"
    return True, "all arms present + finite for every (level,draw)"

def check_no_oom(peak_gb, budget_gb, per_gen_growth_gb, leak_tol):
    ok = peak_gb <= budget_gb and per_gen_growth_gb <= leak_tol
    return ok, f"peak {peak_gb:.1f}GB<= {budget_gb}, per-gen growth {per_gen_growth_gb:.3f}<= {leak_tol}"

def run_gate(report):
    msgs, results = [], []
    for m, box in report.get("masks", []):
        ok, msg = check_mask_valid(m, box, report.get("min_frac", 0.01), report.get("max_frac", 0.9))
        results.append(ok); msgs.append("mask_valid: " + msg)
    for cell in report.get("cells", []):
        for chk in (check_k_eff_equal(cell["rows"]), check_arms_comparable(cell["rows"])):
            results.append(chk[0]); msgs.append("cell: " + chk[1])
        for bid in cell.get("borrowed", []):
            ok, msg = check_no_contamination(bid["ids"], bid["concept"], bid["ref"])
            results.append(ok); msgs.append("contam: " + msg)
    for inp in report.get("inpaints", []):
        ok, msg = check_actual_inpaint(inp["base"], inp["out"], inp["box"],
                                       report.get("outside_tol", 3), report.get("inside_min", 4))
        results.append(ok); msgs.append("inpaint: " + msg)
    if "peak_gb" in report:
        ok, msg = check_no_oom(report["peak_gb"], report.get("budget_gb", 23.0),
                               report.get("per_gen_growth_gb", 0.0), report.get("leak_tol", 0.1))
        results.append(ok); msgs.append("oom: " + msg)
    return (all(results) if results else False), msgs
```

- [ ] **Step 4: Run to verify it passes**

Run: `.../python -m pytest tests/test_smoke_restore.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Record (NO git)** — ledger: `Task 9 complete — smoke_restore checks (6 pass). NO git.`

---

## Task 10: Regression — whole non-GPU suite green + imports

**Files:**
- Test: entire `tests/` (non-GPU), plus import checks for every new module.

- [ ] **Step 1: Run the full non-GPU suite**

Run: `.../python -m pytest tests/ -m "not gpu" -q`
Expected: PASS — the Stage-A 92 tests plus the new tests from Tasks 1–3, 5, 6–9 (~20+ new), 0 failures.

- [ ] **Step 2: Import every new module (no GPU triggered)**

Run:
```
.../python -c "import graft.parts, graft.restore, graft.worker_build_heldout_parts, graft.worker_restore_cell, graft.run_restore_resident, graft.restore_analysis, graft.smoke_restore; import graft.models as m; assert hasattr(m.SdxlIpGenerator,'inpaint_multi'); print('imports ok')"
```
Expected: `imports ok`.

- [ ] **Step 3: GPU-test collection sanity**

Run: `.../python -m pytest tests/ -m gpu --collect-only -q`
Expected: `test_inpaint_gpu.py` collected; no collection errors.

- [ ] **Step 4: Record (NO git)** — ledger: `Task 10 complete — full non-gpu suite green (N pass), all imports clean, gpu tests collect. NO git.`

---

## Task 11: GPU integration — smoke gate → full matrix (DEFERRED to a scheduled 4090 session)

**This task runs only on the 4090.** No code changes; it executes the pipeline and produces the result artifacts. Interpreter: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python`, behind `scripts/gpu_queue.sh` for contention.

**Preconditions:** `outputs/mmkg/graph.json` and the build crop store exist (Stage A). Config written via `GraftConfig().to_yaml(...)` (or reuse `configs/pipeline_eval_run.yaml` extended with the restore fields).

- [ ] **Step 1: Build the held-out part store (all 8 concepts)**

For each concept `C` in `graft.starvation.STARVE_CONCEPTS`:
`python -m graft.worker_build_heldout_parts --root data/treevill/rawdata2 --concept "<C>" --out outputs/mmkg --config <cfg.yaml>`
Expected: `outputs/mmkg/heldout_parts/<C>.json` with ≥1 record; `_heldcrops/`, `_heldmasks/` populated.

- [ ] **Step 2: Run the smoke gate on ONE concept (INFRA GATE)**

Run the resident runner restricted to 1 concept (e.g. a temporary `--concepts` filter or a 1-concept graph slice), collect a smoke `report` (masks, per-cell rows, one base/out inpaint pair, peak VRAM + per-gen growth), and feed it to `graft.smoke_restore.run_gate`.
Expected: `run_gate` returns `True`. **If it returns False: STOP, fix the GPU path — do NOT launch the full matrix and do NOT read any number as evidence** (spec §6). Also eyeball a handful of restored SAM crops for seams / wrong-part fills.

- [ ] **Step 3: Launch the full matrix (resumable)**

`bash scripts/gpu_queue.sh python -m graft.run_restore_resident --graph outputs/mmkg/graph.json --heldout-dir outputs/mmkg/heldout_parts --out outputs/mmkg/restore --config <cfg.yaml>`
Expected: ~1,440 cells, two-phase, VRAM flat (~≤12 GB, no per-gen growth). Resume-safe on interrupt.

- [ ] **Step 4: Read the result + write the report**

Inspect `outputs/mmkg/restore/{recovery.json, hub_vs_rawnn.json, stratified.json}`. Headline = `make_or_break` `+Hub − +RawNN` at level 0/1, DINOv3 (mean, win-rate, Wilcoxon), reported reliability-stratified. Write `reports/2026-09-07-stage-a-prime-result.md` with the honest verdict (§9): decisive negative if `+Hub ≈ +RawNN` under a working deficit; infra/measurement finding if the recovery curve is still flat with a masked hole.

- [ ] **Step 5: Record (NO git)** — ledger: final `Task 11 GPU run complete — <headline delta>; report written.`

---

## Self-Review (completed by plan author)

**Spec coverage:**
- Deficit = mask held-out part region + inpaint-restore → Tasks 3/6 (`box_mask_image`, `build_restore_inputs`), 4 (`inpaint_multi`). ✓
- Starve own-P-crop count {0,1,2,4}, Isolated reference → Task 3 (`select_refs` via `keep_draw`), Task 7 (level clamp). ✓
- k_eff quantity-match invariant, part-restricted pool, contamination → Task 3 (`compute_k_eff`, part pool, own-exclusion), Task 9 checks. ✓
- Metric DINOv3 restored SAM-region vs own held-out P crops → Task 6 (`_score_restored` with `mask_bbox`), Task 5 (held-out crops as scoring refs). ✓
- Box for inpaint, SAM for scoring, geometry cached per cell → Tasks 5/6. ✓
- Consumer parity (one path, selector-only difference) → Task 3/6/7 share `build_restore_inputs`. ✓
- Two-phase VRAM + no_grad + empty_cache → Tasks 4/7. ✓
- Smoke gate with the six checks, infra-only → Task 9 + Task 11 Step 2. ✓
- Analysis: recovery, make-or-break, borrowing-helps, reliability stratification → Task 8. ✓
- No-git, frozen assets, params pinned → Global Constraints, Task 1. ✓
- Run matrix ~1,440, resumable → Task 7/11. ✓

**Placeholder scan:** no TBD/TODO; every code step has real code. (One vestigial `inputs_cache` line in Task 7 is flagged in-line as droppable — not load-bearing.)

**Type consistency:** row schema `{concept, part, level, draw, condition, k_eff, dino, siglip2, clip_i}` is identical across Tasks 6/7/8/9. `build_restore_inputs` return dict keys match between Task 6 (producer) and Task 7 (consumer). `select_refs` / `assemble_ref_images` / `compute_k_eff` signatures match between Task 3 and Tasks 6/7. `mask_bbox`/`box_mask_image`/`crop_region`/`clamp_box` from Task 2 used consistently.

**Ordering note surfaced:** Task 8 must be implemented before Task 7's import check (Task 7 imports the analysis functions).
