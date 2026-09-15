# Fast Eval Driver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A batched eval driver that cuts model loads from ~880 to ~14 (SDXL loaded once, each embedder once) so the sprint matrix runs in ~1–2h instead of 27h, producing outputs identical in shape to the slow path.

**Architecture:** Four sequential phases, each a subprocess holding one model family for its whole life (no cross-model churn): (1) build KGs + recall b2 attrs, (2) generate every image with SDXL resident, (3) score every image one-embedder-per-process, (4) aggregate pure-Python and reuse `analysis.build_analysis`/`summarize`. A new parallel path; the slow `run_eval.py` is untouched and stays the trusted reference.

**Tech Stack:** Python, pytest (unit default; `@pytest.mark.gpu` deselected), numpy, PIL, HuggingFace (SDXL+IP-Adapter, SigLIP2, DINOv3, CLIP, Qwen2.5-VL, GroundingDINO).

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-09-03-fast-eval-driver-design.md`.
- **Slow path is frozen:** do NOT modify `graft/run_eval.py`, `graft/worker_baseline.py`, `graft/worker_metrics.py`, `graft/refine.py`. Reuse `graft.run_eval.summarize` and `graft.analysis.build_analysis` by import.
- **GPU env:** `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python` (torch 2.6+cu124). Base conda env has no torch. Unit tests run under either; gpu tests only under kontext.
- **Every GPU worker holds ONE model family for life** — never load a different model into a process that already loaded one. `env.setup()` is the first call in each worker's `main`.
- **Row schema is exactly the slow path's:** `{method, species, name_mode, ip_scale, n_heldout, dino, siglip2, clip_i, clip_t, attribute_accuracy}`. `ip_scale` is `None` for b0/b2.
- **Cached model ids fixed** (`graft/config.py`); never substitute.
- **Neutral token is `"plant"`; `neutralize_name` drives the prompt head.**
- **Image id / filename:** `f"{species}|{method}|{name_mode}|ip{ip_scale}|seed{seed}"` is the score-table key; the PNG basename is the same with `|`→`_`, `/`→`_`, plus `.png`, under `outputs/<species>/gen_fast/`.
- **TDD, DRY, YAGNI, frequent commits.** Pure logic tested with in-memory fakes; anything needing weights marked `@pytest.mark.gpu`.
- **Run unit tests:** `cd ndbao_hbngoc/kg_test && python -m pytest -q` (gpu auto-deselected).
- **No git repo at this path** (`.git` is an empty stub). Perform the "commit" steps as `git add`/`git commit`; they no-op-fail visibly — flag to the user rather than skipping the checkpoint. (Every prior sprint task did this.)

---

### Task 1: `eval_common` — image ids + KG-usable check

**Files:**
- Create: `graft/eval_common.py`
- Test: `tests/test_eval_common.py`

**Interfaces:**
- Produces: `image_id(species, method, name_mode, ip_scale, seed) -> str`; `id_to_filename(image_id) -> str`; `kg_is_usable(kg_path) -> bool` (True iff the file exists and its `ConceptKG` has non-empty `ref_embeddings` aligned to `ref_paths`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_common.py
import json
from graft.eval_common import image_id, id_to_filename, kg_is_usable
from graft.schema import ConceptKG

def test_image_id_and_filename():
    iid = image_id("Egyptian lotus", "ours", "neutral", 0.6, 1)
    assert iid == "Egyptian lotus|ours|neutral|ip0.6|seed1"
    assert id_to_filename(iid) == "Egyptian lotus_ours_neutral_ip0.6_seed1.png"
    # None ip_scale (b0/b2) stays literal
    assert image_id("Guava", "b0", "named", None, 0) == "Guava|b0|named|ipNone|seed0"

def test_kg_is_usable(tmp_path):
    good = ConceptKG("x", [], [], {}, "", [], ["a.jpg", "b.jpg"], ref_embeddings=[[0.1], [0.2]])
    p = tmp_path / "kg.json"; good.to_json(str(p))
    assert kg_is_usable(str(p)) is True
    bad = ConceptKG("x", [], [], {}, "", [], ["a.jpg", "b.jpg"])  # no ref_embeddings
    q = tmp_path / "bad.json"; bad.to_json(str(q))
    assert kg_is_usable(str(q)) is False
    assert kg_is_usable(str(tmp_path / "missing.json")) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_common.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/eval_common.py
"""Pure helpers shared by the fast eval driver (spec 2026-09-03). No torch at
import time -- the GPU workers import their model code lazily inside main()."""
from __future__ import annotations

import os
from typing import List, Optional


def image_id(species: str, method: str, name_mode: str, ip_scale: Optional[float], seed: int) -> str:
    return f"{species}|{method}|{name_mode}|ip{ip_scale}|seed{seed}"


def id_to_filename(iid: str) -> str:
    return iid.replace("|", "_").replace("/", "_") + ".png"


def kg_is_usable(kg_path: str) -> bool:
    """True iff kg_path exists and holds ref_embeddings aligned to ref_paths.
    A Phase-1 kg.json (no ref_embeddings) is NOT usable -- it must be rebuilt
    (spec 2026-08-30 C1)."""
    if not os.path.exists(kg_path):
        return False
    try:
        from graft.schema import ConceptKG

        kg = ConceptKG.from_json(kg_path)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return bool(kg.ref_embeddings) and len(kg.ref_embeddings) == len(kg.ref_paths)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_common.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/eval_common.py tests/test_eval_common.py
git commit -m "feat(eval_common): image ids + kg_is_usable for the fast driver"
```

---

### Task 2: `eval_common.build_manifest` — expand the cell/seed matrix

**Files:**
- Modify: `graft/eval_common.py`
- Test: `tests/test_eval_common.py`

**Interfaces:**
- Consumes: `image_id` (Task 1).
- Produces: `SWEPT_METHODS = ("ours", "ours_notree", "b1")`; `build_manifest(species_names, methods, name_modes, ip_scales, seeds) -> List[dict]` where each row is `{"image_id", "species", "method", "name_mode", "ip_scale", "seed"}`. Swept methods get every `ip_scale`; b0/b2 get one cell per name_mode with `ip_scale=None`. Every cell fans out to `seeds` seed-rows (0..seeds-1).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_common.py  (append)
from graft.eval_common import build_manifest, SWEPT_METHODS

def test_build_manifest_swept_vs_unswept_and_seeds():
    m = build_manifest(
        species_names=["A"], methods=["ours", "b0", "b1"],
        name_modes=["neutral", "named"], ip_scales=[0.4, 0.6], seeds=2,
    )
    # ours,b1 swept over 2 ip x 2 name x 2 seeds = 8 each; b0 unswept: 2 name x 2 seeds = 4
    n = lambda meth: sum(1 for r in m if r["method"] == meth)
    assert n("ours") == 8 and n("b1") == 8 and n("b0") == 4
    # b0 rows carry ip_scale None
    assert all(r["ip_scale"] is None for r in m if r["method"] == "b0")
    # ids are unique and self-describing
    ids = [r["image_id"] for r in m]
    assert len(ids) == len(set(ids))
    assert "A|b0|neutral|ipNone|seed1" in ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_common.py -q`
Expected: FAIL (`build_manifest` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/eval_common.py  (append)
SWEPT_METHODS = ("ours", "ours_notree", "b1")


def build_manifest(species_names, methods, name_modes, ip_scales, seeds) -> List[dict]:
    """Flat (species,method,name_mode,ip_scale,seed) rows. Swept methods span
    every ip_scale; b0/b2 ignore ip (one cell per name_mode, ip_scale=None)."""
    rows: List[dict] = []
    for species in species_names:
        for method in methods:
            ips = list(ip_scales) if method in SWEPT_METHODS else [None]
            for name_mode in name_modes:
                for ip in ips:
                    for seed in range(seeds):
                        rows.append({
                            "image_id": image_id(species, method, name_mode, ip, seed),
                            "species": species, "method": method,
                            "name_mode": name_mode, "ip_scale": ip, "seed": seed,
                        })
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_common.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/eval_common.py tests/test_eval_common.py
git commit -m "feat(eval_common): build_manifest cell/seed expansion"
```

---

### Task 3: `eval_common.prompt_for` — shared prompt + exemplar selection

**Files:**
- Modify: `graft/eval_common.py`
- Test: `tests/test_eval_common.py`

**Interfaces:**
- Consumes: `graft.prompt.compose_prompt/compose_ravel_prompt/concept_token`, `graft.generate.select_exemplar`, `graft.selection.medoid_index`.
- Produces: `prompt_for(method, concept, kg, b2_attrs, neutralize) -> str`; `exemplar_for(method, kg, pool_paths, pool_embed_fn) -> Optional[str]`. `prompt_for` reproduces the four methods' heads exactly as `graft/baselines.py` + `graft/generate.py` do. `exemplar_for` returns the medoid ref path for ours/ours_notree (from `kg`) and b1 (from the pool via `pool_embed_fn`), `None` for b0/b2.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_common.py  (append)
import numpy as np
from graft.eval_common import prompt_for, exemplar_for
from graft.schema import AttributeNode, ConceptKG

def _kg():
    return ConceptKG("monkey puzzle tree",
                     [AttributeNode("silhouette", "symmetric candelabra", "vision")],
                     [], {}, "", [], ["a.jpg", "b.jpg", "c.jpg"],
                     ref_embeddings=[[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]])

def test_prompt_for_heads():
    kg = _kg()
    assert prompt_for("b0", "Avocado", kg, [], neutralize=True) == "a photo of a plant"
    assert prompt_for("b0", "Avocado", kg, [], neutralize=False) == "a photo of a Avocado"
    assert prompt_for("ours", "Avocado", kg, [], neutralize=True).startswith("a photo of a plant")
    assert "symmetric candelabra" in prompt_for("ours", "Avocado", kg, [], neutralize=True)
    assert prompt_for("b2", "durian", kg, ["spiky husk"], neutralize=True) == "a photo of a plant, spiky husk"

def test_exemplar_for_medoid_and_none():
    kg = _kg()  # medoid of the 3 rows is index 0 or 1
    assert exemplar_for("ours", kg, kg.ref_paths, None) in ("a.jpg", "b.jpg")
    assert exemplar_for("b0", kg, kg.ref_paths, None) is None
    # b1 selects via the pool embed fn (same medoid rule)
    embed = lambda paths: np.array([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]])[: len(paths)]
    assert exemplar_for("b1", kg, ["a.jpg", "b.jpg", "c.jpg"], embed) in ("a.jpg", "b.jpg")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_common.py -q`
Expected: FAIL (`prompt_for` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/eval_common.py  (append)
def prompt_for(method: str, concept: str, kg, b2_attrs, neutralize: bool) -> str:
    from graft.prompt import compose_prompt, compose_ravel_prompt, concept_token

    if method in ("ours", "ours_notree"):
        return compose_prompt(kg, neutralize=neutralize)
    if method == "b2":
        return compose_ravel_prompt(concept, b2_attrs, neutralize=neutralize)
    # b0, b1: bare neutralized head (matches graft/baselines.py)
    return f"a photo of a {concept_token(concept, neutralize)}"


def exemplar_for(method: str, kg, pool_paths, pool_embed_fn):
    from graft.selection import medoid_index

    if method in ("ours", "ours_notree"):
        from graft.generate import select_exemplar
        return select_exemplar(kg)
    if method == "b1":
        return pool_paths[medoid_index(pool_embed_fn(pool_paths))]
    return None  # b0, b2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_common.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/eval_common.py tests/test_eval_common.py
git commit -m "feat(eval_common): prompt_for + exemplar_for shared across methods"
```

---

### Task 4: `eval_common.rows_from_tables` — best-of-k aggregation

**Files:**
- Modify: `graft/eval_common.py`
- Test: `tests/test_eval_common.py`

**Interfaces:**
- Produces: `rows_from_tables(manifest, tables, n_heldout_by_species, use_part_tree=False) -> (rows, missing_cells)`. `tables` is `{image_id: {metric: value}}` merged from every embedder. For each **cell** (species,method,name_mode,ip_scale), pick the seed with the highest verify score — `attr_pass` when `use_part_tree=False`, else `0.5·mean_part_sim + 0.5·attr_pass` — ties broken by lowest seed. Emit one row per cell tagged `{method,species,name_mode,ip_scale,n_heldout}` + the winning seed's `{dino,siglip2,clip_i,clip_t,attribute_accuracy}`. A cell whose seeds are all absent from `tables` (or missing required metrics) goes to `missing_cells`. `attr_pass == attribute_accuracy` (one VLM pass).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_common.py  (append)
from graft.eval_common import build_manifest, rows_from_tables

def test_rows_from_tables_picks_best_seed_and_flags_missing():
    man = build_manifest(["A", "B"], ["ours"], ["neutral"], [0.6], seeds=2)
    def metr(dino, aa):
        return {"dino": dino, "siglip2": 0.5, "clip_i": 0.5, "clip_t": 0.1, "attribute_accuracy": aa}
    tables = {
        "A|ours|neutral|ip0.6|seed0": metr(0.30, 0.4),
        "A|ours|neutral|ip0.6|seed1": metr(0.90, 0.8),  # higher attr -> this seed wins
        # species B: no images at all -> its cell is missing
    }
    rows, missing = rows_from_tables(man, tables, {"A": 16, "B": 10}, use_part_tree=False)
    assert len(rows) == 1
    r = rows[0]
    assert r["species"] == "A" and r["method"] == "ours" and r["ip_scale"] == 0.6
    assert r["n_heldout"] == 16
    assert abs(r["dino"] - 0.90) < 1e-9  # winning seed's metrics
    assert {"species": "B", "method": "ours", "name_mode": "neutral", "ip_scale": 0.6} in missing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_common.py -q`
Expected: FAIL (`rows_from_tables` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/eval_common.py  (append)
_METRIC_KEYS = ("dino", "siglip2", "clip_i", "clip_t", "attribute_accuracy")


def _verify_score(m: dict, use_part_tree: bool) -> float:
    attr = m.get("attribute_accuracy", 0.0)
    if not use_part_tree:
        return attr
    sims = [m[k] for k in ("part_sim_dino", "part_sim_siglip") if k in m]
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    return 0.5 * mean_sim + 0.5 * attr


def rows_from_tables(manifest, tables, n_heldout_by_species, use_part_tree=False):
    # group seed-rows by cell
    cells: dict = {}
    for row in manifest:
        key = (row["species"], row["method"], row["name_mode"], row["ip_scale"])
        cells.setdefault(key, []).append(row)

    rows, missing = [], []
    for (species, method, name_mode, ip_scale), seed_rows in cells.items():
        scored = [
            (r["seed"], tables[r["image_id"]])
            for r in seed_rows
            if r["image_id"] in tables
            and all(k in tables[r["image_id"]] for k in _METRIC_KEYS)
        ]
        if not scored:
            missing.append({"species": species, "method": method,
                            "name_mode": name_mode, "ip_scale": ip_scale})
            continue
        # best verify score, lowest seed on tie
        seed, best = min(scored, key=lambda sm: (-_verify_score(sm[1], use_part_tree), sm[0]))
        rows.append({
            "method": method, "species": species, "name_mode": name_mode,
            "ip_scale": ip_scale, "n_heldout": n_heldout_by_species[species],
            **{k: best[k] for k in _METRIC_KEYS},
        })
    return rows, missing
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_common.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/eval_common.py tests/test_eval_common.py
git commit -m "feat(eval_common): rows_from_tables best-of-k aggregation"
```

---

### Task 5: `Models.generator.set_scale` — vary ip_scale without reload

**Files:**
- Modify: `graft/models.py` (`SdxlIpGenerator`)
- Test: `tests/test_models_setscale_unit.py`

**Interfaces:**
- Produces: `SdxlIpGenerator.set_scale(scale: float)` — updates `self.ip_scale`; if `_pipe_ip` is already built, also calls `self._pipe_ip.set_ip_adapter_scale(scale)`. So the fast generator can switch ip per cell without rebuilding the pipe. Idempotent, safe before the pipe is loaded.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_setscale_unit.py
from graft.models import SdxlIpGenerator

class _FakePipe:
    def __init__(self): self.scale = None
    def set_ip_adapter_scale(self, s): self.scale = s

def test_set_scale_before_and_after_pipe_load():
    g = SdxlIpGenerator.__new__(SdxlIpGenerator)  # bypass heavy __init__
    g.ip_scale = 0.6
    g._pipe_ip = None
    g.set_scale(0.4)                     # before load: just records
    assert g.ip_scale == 0.4
    g._pipe_ip = _FakePipe()
    g.set_scale(0.8)                     # after load: pushes to the pipe
    assert g.ip_scale == 0.8
    assert g._pipe_ip.scale == 0.8
```

(`SdxlIpGenerator` stores `self.ip_scale` at `graft/models.py:343` and lazy-builds `self._pipe_ip`; the test bypasses `__init__` so no weights load.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models_setscale_unit.py -q`
Expected: FAIL (`set_scale` missing).

- [ ] **Step 3: Write minimal implementation**

In `graft/models.py`, add to `SdxlIpGenerator` (right after `generate`):

```python
    def set_scale(self, scale: float) -> None:
        """Change the IP-Adapter scale without rebuilding the pipe. Safe before
        pipe_ip is loaded (records it; pipe_ip's constructor applies it)."""
        self.ip_scale = scale
        if self._pipe_ip is not None:
            self._pipe_ip.set_ip_adapter_scale(scale)
```

(Confirm the generator uses `self._pipe_ip` as its cached attribute name — it does at `graft/models.py`'s `pipe_ip` property.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_models_setscale_unit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/models.py tests/test_models_setscale_unit.py
git commit -m "feat(models): SdxlIpGenerator.set_scale to vary ip without reload"
```

---

### Task 6: `worker_recall_b2` — recall b2 attrs for all species (VLM resident)

**Files:**
- Create: `graft/worker_recall_b2.py`
- Test: `tests/test_worker_recall_b2_unit.py`

**Interfaces:**
- Produces: pure `recall_all(species_names, describe_fn) -> dict` mapping species→attr list, where `describe_fn(concept) -> List[str]` runs one VLM recall (blank image + `RAVEL_ATTR_INSTRUCTION`, parsed by `baselines._parse_attr_list`). A `main(argv)` loads the VLM once, calls `recall_all` over all species, writes `<out_json>`. b2 attrs depend only on the concept name (not name_mode), so one list per species.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_worker_recall_b2_unit.py
from graft.worker_recall_b2 import recall_all

def test_recall_all_maps_species_to_attrs():
    calls = []
    def describe(concept):
        calls.append(concept)
        return [f"{concept}-attr1", f"{concept}-attr2"]
    out = recall_all(["Guava", "Mango"], describe)
    assert out == {"Guava": ["Guava-attr1", "Guava-attr2"],
                   "Mango": ["Mango-attr1", "Mango-attr2"]}
    assert calls == ["Guava", "Mango"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_worker_recall_b2_unit.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/worker_recall_b2.py
#!/usr/bin/env python
"""Phase 1b of the fast driver: recall b2 (RAVEL-style) attributes for every
species with the VLM resident, so the SDXL generation phase never needs the
VLM. b2 attrs depend only on the concept name. One VLM load for all species.

Usage: python -m graft.worker_recall_b2 <cfg_yaml> <out_json> <species>...
"""
from __future__ import annotations

import json
import sys
from typing import Dict, List

from graft import env

env.setup()


def recall_all(species_names, describe_fn) -> Dict[str, List[str]]:
    return {name: describe_fn(name) for name in species_names}


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: worker_recall_b2.py <cfg_yaml> <out_json> <species>...", file=sys.stderr)
        return 2
    cfg_yaml, out_json, *species = argv

    from PIL import Image

    from graft.baselines import RAVEL_ATTR_INSTRUCTION, _parse_attr_list
    from graft.config import GraftConfig
    from graft.models import Models

    cfg = GraftConfig.from_yaml(cfg_yaml)
    models = Models(cfg)
    blank = Image.new("RGB", (16, 16), (128, 128, 128))

    def describe(concept: str):
        raw = models.vlm.describe([blank], RAVEL_ATTR_INSTRUCTION.format(concept=concept))
        return _parse_attr_list(raw)

    out = recall_all(species, describe)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_worker_recall_b2_unit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/worker_recall_b2.py tests/test_worker_recall_b2_unit.py
git commit -m "feat(fast): worker_recall_b2 -- one-VLM-load b2 attribute recall"
```

---

### Task 7: `worker_generate_many` — SDXL-resident generation of all images

**Files:**
- Create: `graft/worker_generate_many.py`
- Test: `tests/test_worker_generate_many_unit.py` (pure driver), `tests/test_generate_many_gpu.py` (`@pytest.mark.gpu`)

**Interfaces:**
- Consumes: `eval_common.{prompt_for,exemplar_for,id_to_filename}`, `graft.selection.medoid_index`.
- Produces: pure `generate_plan(manifest, kg_by_species, b2_attrs, species_pool, out_dir, exists_fn) -> List[dict]` returning, for each seed-row **not** already on disk, the `{image_id, prompt, ip_path, seed, out_path, method}` needed to generate — resolving prompt + exemplar via `eval_common`. b0/b2 across ip are deduped by `out_path` (same file). `main(argv)` loads `Models(cfg)`, keeps SDXL resident, calls `generate_plan`, and for each item `set_scale`s (parsed from the id), generates, saves. One SDXL load; per-item OOM logs + skips.

- [ ] **Step 1: Write the failing test (pure plan)**

```python
# tests/test_worker_generate_many_unit.py
from graft.worker_generate_many import generate_plan
from graft.schema import ConceptKG

def _kg(concept):
    return ConceptKG(concept, [], [], {}, "", [], ["a.jpg", "b.jpg"],
                     ref_embeddings=[[1.0, 0.0], [0.9, 0.2]])

def test_generate_plan_resolves_prompts_and_skips_existing():
    manifest = [
        {"image_id": "A|ours|neutral|ip0.6|seed0", "species": "A", "method": "ours",
         "name_mode": "neutral", "ip_scale": 0.6, "seed": 0},
        {"image_id": "A|b0|neutral|ipNone|seed0", "species": "A", "method": "b0",
         "name_mode": "neutral", "ip_scale": None, "seed": 0},
    ]
    done = {"A_b0_neutral_ipNone_seed0.png"}  # b0 already generated
    plan = generate_plan(
        manifest, kg_by_species={"A": _kg("A")}, b2_attrs={"A": []},
        species_pool={"A": ["a.jpg", "b.jpg"]}, out_dir="OUT",
        exists_fn=lambda p: p.split("/")[-1] in done,
    )
    ids = [p["image_id"] for p in plan]
    assert ids == ["A|ours|neutral|ip0.6|seed0"]        # b0 skipped as existing
    item = plan[0]
    assert item["prompt"].startswith("a photo of a plant")
    assert item["ip_path"] in ("a.jpg", "b.jpg")        # medoid exemplar
    assert item["out_path"].endswith("A_ours_neutral_ip0.6_seed0.png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_worker_generate_many_unit.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/worker_generate_many.py
#!/usr/bin/env python
"""Phase 2 of the fast driver: generate every manifest image with SDXL+IP-Adapter
resident (one load for the whole run). See spec 2026-09-03 section 3.2.

Usage:
    python -m graft.worker_generate_many <cfg_yaml> <manifest_json> <kg_dir> \
        <b2_attrs_json> <pool_json> <out_dir>
"""
from __future__ import annotations

import json
import os
import sys

from graft import env

env.setup()

from graft.eval_common import exemplar_for, id_to_filename, prompt_for

_NEG = "monochrome, lowres, bad anatomy, worst quality, low quality"


def generate_plan(manifest, kg_by_species, b2_attrs, species_pool, out_dir, exists_fn):
    """Pure: resolve prompt+exemplar+out_path for each seed-row not on disk.
    b0/b2 collapse across ip_scale to the same out file, so a duplicate out_path
    seen earlier in this plan is skipped too."""
    import numpy as np  # noqa: F401  (only used by callers' embed fns)

    plan, planned_paths = [], set()
    for row in manifest:
        sp, method = row["species"], row["method"]
        out_path = os.path.join(out_dir, sp, "gen_fast", id_to_filename(row["image_id"]))
        if exists_fn(out_path) or out_path in planned_paths:
            continue
        kg = kg_by_species.get(sp)
        prompt = prompt_for(method, sp, kg, b2_attrs.get(sp, []),
                            neutralize=(row["name_mode"] == "neutral"))
        ip_path = exemplar_for(method, kg, species_pool.get(sp, []), _pool_embedder(method))
        planned_paths.add(out_path)
        plan.append({"image_id": row["image_id"], "prompt": prompt, "ip_path": ip_path,
                     "seed": row["seed"], "ip_scale": row["ip_scale"],
                     "out_path": out_path, "method": method})
    return plan


def _pool_embedder(method):
    # b1 needs a SigLIP embedder over the pool; resolved lazily in main so the
    # pure planner can be called with a stub. Placeholder here; main injects.
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 6:
        print("usage: worker_generate_many.py <cfg_yaml> <manifest_json> <kg_dir> "
              "<b2_attrs_json> <pool_json> <out_dir>", file=sys.stderr)
        return 2
    cfg_yaml, manifest_json, kg_dir, b2_json, pool_json, out_dir = argv

    from PIL import Image

    from graft.config import GraftConfig
    from graft.models import Models
    from graft.schema import ConceptKG

    cfg = GraftConfig.from_yaml(cfg_yaml)
    manifest = json.load(open(manifest_json))
    b2_attrs = json.load(open(b2_json))
    pool = json.load(open(pool_json))
    species = sorted({r["species"] for r in manifest})
    kg_by_species = {
        s: ConceptKG.from_json(os.path.join(kg_dir, s, "kg.json"))
        for s in species
        if os.path.exists(os.path.join(kg_dir, s, "kg.json"))
    }
    models = Models(cfg)

    def pool_embed(paths):
        imgs = [Image.open(p).convert("RGB") for p in paths]
        return models.siglip.embed_image(imgs)

    # inject the real embedder into the planner
    import graft.worker_generate_many as self_mod
    self_mod._pool_embedder = lambda method: pool_embed if method == "b1" else None

    plan = generate_plan(manifest, kg_by_species, b2_attrs, pool, out_dir,
                         exists_fn=os.path.exists)
    for item in plan:
        os.makedirs(os.path.dirname(item["out_path"]), exist_ok=True)
        ip_image = Image.open(item["ip_path"]).convert("RGB") if item["ip_path"] else None
        if item["ip_scale"] is not None:
            models.generator.set_scale(item["ip_scale"])
        try:
            img = models.generator.generate(prompt=item["prompt"], ip_image=ip_image,
                                            seed=item["seed"], negative=_NEG, steps=50)
        except Exception as exc:  # noqa: BLE001 -- one bad cell must not kill the phase
            sys.stderr.write(f"[gen] {item['image_id']}: FAILED {type(exc).__name__}: {exc}\n")
            continue
        img.save(item["out_path"])
        print(f"[gen] {item['image_id']} -> {item['out_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_worker_generate_many_unit.py -q`
Expected: PASS.
Then add `tests/test_generate_many_gpu.py` (`pytestmark = pytest.mark.gpu`) that builds a 1-species manifest for `{ours,b0,b1}`, runs `main` against real weights on `data/treevill/rawdata2` (using an existing usable `kg.json`), and asserts every planned PNG exists and is ≥768². Run manually: `python -m pytest tests/test_generate_many_gpu.py -q -m gpu`.

- [ ] **Step 5: Commit**

```bash
git add graft/worker_generate_many.py tests/test_worker_generate_many_unit.py tests/test_generate_many_gpu.py
git commit -m "feat(fast): worker_generate_many -- SDXL-resident batched generation"
```

---

### Task 8: `worker_score_many` — one-embedder scoring of all images

**Files:**
- Create: `graft/worker_score_many.py`
- Test: `tests/test_worker_score_many_unit.py` (pure), `tests/test_score_many_gpu.py` (`@pytest.mark.gpu`)

**Interfaces:**
- Consumes: `graft.metrics.{image_fidelity,clip_t,attribute_accuracy}`, `eval_common.id_to_filename`.
- Produces: pure `score_plan(manifest, out_dir, exists_fn) -> List[dict]` (per manifest row that has a generated PNG on disk: `{image_id, species, img_path}`); `main(argv)` takes `<embedder> <cfg_yaml> <manifest_json> <kg_dir> <heldout_json> <out_dir> <out_table_json>` and, with **one** model resident (`dino|siglip|clip|vlm`), computes only that model's metrics for every scored image, writing `{image_id: {metric: value}}`. dino/siglip → `{"dino"|"siglip2": heldout_fidelity}`; clip → `{"clip_i", "clip_t"}`; vlm → `{"attribute_accuracy"}`.

- [ ] **Step 1: Write the failing test (pure plan)**

```python
# tests/test_worker_score_many_unit.py
from graft.worker_score_many import score_plan

def test_score_plan_only_includes_generated_images():
    manifest = [
        {"image_id": "A|ours|neutral|ip0.6|seed0", "species": "A"},
        {"image_id": "A|ours|neutral|ip0.6|seed1", "species": "A"},
    ]
    on_disk = {"A_ours_neutral_ip0.6_seed0.png"}  # seed1 never generated
    plan = score_plan(manifest, out_dir="OUT",
                      exists_fn=lambda p: p.split("/")[-1] in on_disk)
    assert [p["image_id"] for p in plan] == ["A|ours|neutral|ip0.6|seed0"]
    assert plan[0]["img_path"].endswith("A/gen_fast/A_ours_neutral_ip0.6_seed0.png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_worker_score_many_unit.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/worker_score_many.py
#!/usr/bin/env python
"""Phase 3 of the fast driver: score every generated image with ONE embedder
resident. Invoked once per embedder (dino|siglip|clip|vlm), so no process ever
churns two models -- the safety envelope the passing gpu tests validate.

Usage:
    python -m graft.worker_score_many <embedder> <cfg_yaml> <manifest_json> \
        <kg_dir> <heldout_json> <out_dir> <out_table_json>
"""
from __future__ import annotations

import json
import os
import sys

from graft import env

env.setup()

from graft.eval_common import id_to_filename


def score_plan(manifest, out_dir, exists_fn):
    plan = []
    for row in manifest:
        sp = row["species"]
        img_path = os.path.join(out_dir, sp, "gen_fast", id_to_filename(row["image_id"]))
        if exists_fn(img_path):
            plan.append({"image_id": row["image_id"], "species": sp, "img_path": img_path})
    return plan


def main(argv: list[str]) -> int:
    if len(argv) != 7:
        print("usage: worker_score_many.py <embedder> <cfg_yaml> <manifest_json> "
              "<kg_dir> <heldout_json> <out_dir> <out_table_json>", file=sys.stderr)
        return 2
    embedder, cfg_yaml, manifest_json, kg_dir, heldout_json, out_dir, out_table = argv

    from PIL import Image

    from graft.config import GraftConfig
    from graft.metrics import attribute_accuracy, clip_t, image_fidelity
    from graft.models import Models
    from graft.schema import ConceptKG

    cfg = GraftConfig.from_yaml(cfg_yaml)
    manifest = json.load(open(manifest_json))
    heldout = json.load(open(heldout_json))  # {species: [heldout paths]}
    plan = score_plan(manifest, out_dir, os.path.exists)
    species = sorted({p["species"] for p in plan})
    kg_by = {s: ConceptKG.from_json(os.path.join(kg_dir, s, "kg.json")) for s in species}
    heldout_imgs = {s: [Image.open(p).convert("RGB") for p in heldout[s]] for s in species}
    models = Models(cfg)

    table: dict = {}
    for item in plan:
        sp = item["species"]
        try:
            img = Image.open(item["img_path"]).convert("RGB")
            if embedder == "dino":
                table[item["image_id"]] = {"dino": image_fidelity(img, heldout_imgs[sp], models.dino)}
            elif embedder == "siglip":
                table[item["image_id"]] = {"siglip2": image_fidelity(img, heldout_imgs[sp], models.siglip)}
            elif embedder == "clip":
                table[item["image_id"]] = {
                    "clip_i": image_fidelity(img, heldout_imgs[sp], models.clip),
                    "clip_t": clip_t(img, f"a photo of a {sp}", models.clip),
                }
            elif embedder == "vlm":
                table[item["image_id"]] = {
                    "attribute_accuracy": attribute_accuracy(img, kg_by[sp].attribute_texts(), models.vlm)
                }
            else:
                print(f"unknown embedder '{embedder}'", file=sys.stderr)
                return 2
        except Exception as exc:  # noqa: BLE001 -- one bad image must not kill the phase
            sys.stderr.write(f"[score:{embedder}] {item['image_id']}: FAILED {type(exc).__name__}: {exc}\n")
            continue
    with open(out_table, "w") as f:
        json.dump(table, f, indent=2)
    print(f"[score:{embedder}] wrote {len(table)} rows -> {out_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_worker_score_many_unit.py -q`
Expected: PASS.
Then add `tests/test_score_many_gpu.py` (`pytestmark = pytest.mark.gpu`): given a small manifest + generated PNGs (from Task 7's gpu run), run `main` for each of `dino,siglip,clip,vlm` and assert each table covers every scored image with its metric keys and values in `[-1, 1]` (fidelity) / `[0, 1]` (attr). Run manually.

- [ ] **Step 5: Commit**

```bash
git add graft/worker_score_many.py tests/test_worker_score_many_unit.py tests/test_score_many_gpu.py
git commit -m "feat(fast): worker_score_many -- one-embedder-resident scoring"
```

---

### Task 9: `run_eval_fast` — orchestrator + Phase 4 + CLI

**Files:**
- Create: `graft/run_eval_fast.py`
- Test: `tests/test_run_eval_fast_unit.py`

**Interfaces:**
- Consumes: everything above; `graft.dataset.{list_species,load_species,split_refs}`, `graft.run_eval.{rank_species_by_unique_count,summarize,_markdown_table}`, `graft.analysis.build_analysis`, `graft.pipeline._build_kg_subprocess`, `eval_common.kg_is_usable`.
- Produces: pure `assemble(manifest, tables, n_heldout_by_species, use_part_tree) -> dict` (= `{"rows", "summary", "expected_cells", "missing_cells", "analysis"}`, reusing `summarize`+`build_analysis`); `main(argv)` runs the four phases via `subprocess.run` and writes `results.json`/`results.md`/`analysis.json` under `--out`. CLI + defaults per spec section 6.

- [ ] **Step 1: Write the failing test (pure assembly)**

```python
# tests/test_run_eval_fast_unit.py
from graft.eval_common import build_manifest
from graft.run_eval_fast import assemble

def test_assemble_reuses_summarize_and_build_analysis():
    man = build_manifest(["A"], ["ours", "b1"], ["neutral"], [0.6], seeds=1)
    def metr(d): return {"dino": d, "siglip2": 0.7, "clip_i": 0.7, "clip_t": 0.2, "attribute_accuracy": 0.5}
    tables = {
        "A|ours|neutral|ip0.6|seed0": metr(0.6),
        "A|b1|neutral|ip0.6|seed0": metr(0.4),
    }
    out = assemble(man, tables, {"A": 10}, use_part_tree=False)
    assert out["expected_cells"] == 2 and out["missing_cells"] == []
    # summarize groups by method, excludes tags
    assert abs(out["summary"]["ours"]["dino"] - 0.6) < 1e-9
    # analysis wiring: ours beats b1 on this one species
    gv = out["analysis"]["graft_vs_b1"]["dino"]
    assert gv["n"] == 1 and abs(gv["mean_delta"] - 0.2) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_eval_fast_unit.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/run_eval_fast.py
#!/usr/bin/env python
"""Fast eval driver (spec 2026-09-03): four batched phases, ~14 model loads
instead of ~880. Slow run_eval.py stays the trusted reference; this reuses its
summarize + analysis.build_analysis so outputs match in shape.

Phases: (1) build KGs [worker_build_kg] + recall b2 [worker_recall_b2];
(2) generate all [worker_generate_many]; (3) score all, one embedder each
[worker_score_many]; (4) assemble (pure)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from graft import env


def assemble(manifest, tables, n_heldout_by_species, use_part_tree):
    from graft.analysis import build_analysis
    from graft.eval_common import rows_from_tables
    from graft.run_eval import summarize

    rows, missing = rows_from_tables(manifest, tables, n_heldout_by_species, use_part_tree)
    return {
        "rows": rows,
        "summary": summarize(rows),
        "expected_cells": len({(r["species"], r["method"], r["name_mode"], r["ip_scale"])
                               for r in manifest}),
        "missing_cells": missing,
        "analysis": build_analysis(rows),
    }


def _merge_tables(paths):
    merged: dict = {}
    for p in paths:
        if os.path.exists(p):
            for iid, m in json.load(open(p)).items():
                merged.setdefault(iid, {}).update(m)
    return merged


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--species", default=None, help="comma list; bypasses --select-by")
    ap.add_argument("--n-species", type=int, default=8)
    ap.add_argument("--select-by", choices=("alpha", "unique"), default="unique")
    ap.add_argument("--methods", default="ours,b0,b1,b2")
    ap.add_argument("--ip-scales", default="0.6")
    ap.add_argument("--name-modes", default="neutral")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--use-part-tree", action="store_true")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    env.setup()
    from graft.config import GraftConfig
    from graft.dataset import list_species, load_species, split_refs
    from graft.eval_common import build_manifest, kg_is_usable
    from graft.pipeline import _build_kg_subprocess
    from graft.run_eval import _markdown_table, rank_species_by_unique_count

    methods = [m for m in args.methods.split(",") if m]
    name_modes = [m.strip() for m in args.name_modes.split(",") if m.strip()]
    ip_scales = [float(s) for s in args.ip_scales.split(",") if s.strip()]
    bad = set(name_modes) - {"neutral", "named"}
    if bad:
        ap.error(f"--name-modes: unknown {sorted(bad)}")
    if not ip_scales or not name_modes or not methods:
        ap.error("--methods/--ip-scales/--name-modes must be non-empty")

    cfg = GraftConfig.from_yaml(args.config) if os.path.exists(args.config) else GraftConfig()
    if args.species:
        names = [s.strip() for s in args.species.split(",") if s.strip()]
    elif args.select_by == "unique":
        names = rank_species_by_unique_count(args.root, limit=args.n_species)
    else:
        names = list_species(args.root)[: args.n_species]

    species_list = [load_species(args.root, n) for n in names]
    build_cfg = __import__("dataclasses").replace(cfg, neutralize_name=True)
    out_dir = args.out or os.path.join(cfg.outputs_dir, "eval_fast")
    os.makedirs(out_dir, exist_ok=True)
    work = os.path.join(out_dir, "_work")
    os.makedirs(work, exist_ok=True)

    # split + skip species that cannot form build+heldout
    n_heldout, pool, heldout, usable = {}, {}, {}, []
    for sp in species_list:
        b, h = split_refs(sp, k_build=cfg.k_build_refs, seed=0)
        if not b or not h:
            print(f"[fast] {sp.name}: no build+heldout split, skipping")
            continue
        usable.append(sp.name)
        n_heldout[sp.name], pool[sp.name], heldout[sp.name] = len(h), b, h

    # ---- Phase 1: build KGs (rebuild stale) + recall b2 ----
    cfg_yaml = os.path.join(work, "cfg.yaml"); cfg.to_yaml(cfg_yaml)
    kg_dir = cfg.outputs_dir
    for name in usable:
        kg_path = os.path.join(kg_dir, name, "kg.json")
        if not kg_is_usable(kg_path):
            print(f"[fast] {name}: building KG")
            _build_kg_subprocess(name, pool[name], build_cfg, os.path.join(kg_dir, name))
    b2_json = os.path.join(work, "b2_attrs.json")
    subprocess.run([sys.executable, "-m", "graft.worker_recall_b2", cfg_yaml, b2_json, *usable], check=True)

    # ---- manifest + side inputs ----
    manifest = [r for r in build_manifest(usable, methods, name_modes, ip_scales, args.seeds)]
    man_json = os.path.join(work, "manifest.json"); json.dump(manifest, open(man_json, "w"))
    pool_json = os.path.join(work, "pool.json"); json.dump(pool, open(pool_json, "w"))
    heldout_json = os.path.join(work, "heldout.json"); json.dump(heldout, open(heldout_json, "w"))

    # ---- Phase 2: generate all (SDXL resident) ----
    subprocess.run([sys.executable, "-m", "graft.worker_generate_many",
                    cfg_yaml, man_json, kg_dir, b2_json, pool_json, kg_dir], check=True)

    # ---- Phase 3: score all (one embedder each) ----
    embedders = ["dino", "siglip", "clip", "vlm"]
    table_paths = []
    for emb in embedders:
        tp = os.path.join(work, f"table_{emb}.json")
        subprocess.run([sys.executable, "-m", "graft.worker_score_many",
                        emb, cfg_yaml, man_json, kg_dir, heldout_json, kg_dir, tp], check=True)
        table_paths.append(tp)

    # ---- Phase 4: assemble ----
    tables = _merge_tables(table_paths)
    result = assemble(manifest, tables, n_heldout, use_part_tree=args.use_part_tree)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({k: result[k] for k in ("rows", "summary", "expected_cells", "missing_cells")}, f, indent=2)
    with open(os.path.join(out_dir, "analysis.json"), "w") as f:
        json.dump(result["analysis"], f, indent=2)
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(_markdown_table(result["summary"]) + "\n")
    if result["missing_cells"]:
        print(f"[fast] INCOMPLETE: {len(result['missing_cells'])}/{result['expected_cells']} cells missing",
              file=sys.stderr)
    print(_markdown_table(result["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_run_eval_fast_unit.py -q`
Expected: PASS. Also confirm torch-free import: `python -c "import graft.run_eval_fast, graft.eval_common, graft.worker_generate_many, graft.worker_score_many, graft.worker_recall_b2"` and `python -m graft.run_eval_fast --help`.

- [ ] **Step 5: Commit**

```bash
git add graft/run_eval_fast.py tests/test_run_eval_fast_unit.py
git commit -m "feat(fast): run_eval_fast orchestrator + Phase-4 assemble (reuses summarize/build_analysis)"
```

---

## Post-implementation: validate then scale (manual, on the shared 4090)

Not coding tasks — the trust gate and the payoff:

1. **Cross-check fast vs slow on 1–2 species** (spec §5): run the existing slow `run_eval --species Ashok --ip-scales 0.6 --name-modes neutral --methods ours,b0,b1,b2` and `run_eval_fast --species Ashok --ip-scales 0.6 --name-modes neutral --methods ours,b0,b1,b2 --seeds 2`, and diff the per-cell metrics. They should agree within float tolerance (only the best-of-k seed pick may differ). If they diverge, STOP and debug before scaling.
2. **Time it** against the slow path's ~10 min/`ours`-cell to confirm the ~10× win.
3. **Scale:** `run_eval_fast --n-species 30 --select-by unique` (or the full 66) for a statistically real GRAFT-vs-B1 n. Then re-run the audit and refresh `reports/sprint-report.md` with the larger-n result and a proper significance statement.

---

## Self-Review

**Spec coverage:** §3.1 manifest → Tasks 1,2. §3.2 generate → Tasks 5,7. §3.3 score → Task 8. §3.4 aggregate → Tasks 4,9. b2 VLM-at-gen → Task 6 (folded into Phase 1). §4 safety (one model per process) → Tasks 6,7,8 each load a single family; §5 validation → post-impl step 1 + the gpu smokes in Tasks 7,8. §6 CLI/defaults → Task 9. All covered.

**Placeholder scan:** none. `_pool_embedder` in Task 7 is a real injection seam (the planner takes a stub in the unit test; `main` overwrites it with the SigLIP embedder), not a TODO — documented in its docstring and exercised by both the unit test (stub) and the gpu test (real).

**Type consistency:** `image_id`/`id_to_filename` used identically across Tasks 1,2,7,8. `build_manifest` row shape (Task 2) consumed unchanged by Tasks 7,8,9. `rows_from_tables(manifest,tables,n_heldout,use_part_tree)` (Task 4) called by `assemble` (Task 9) with matching args. `tables` = `{image_id: {metric: value}}` produced by Task 8, merged in Task 9, consumed in Task 4. `set_scale` (Task 5) called in Task 7. Row schema matches `summarize`/`build_analysis` (slow path) exactly.
