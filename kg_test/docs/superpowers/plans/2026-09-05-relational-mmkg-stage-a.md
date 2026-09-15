# Relational MMKG — Stage A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build shared **PartType** hubs over all concepts, transfer their cross-concept exemplar prototype into a data-poor concept's IP-Adapter conditioning, and prove — via a controlled starvation/recovery experiment with quantity-matched controls — that hub transfer beats both the isolated per-concept representation and plain nearest-neighbour retrieval.

**Architecture:** A new relational layer on top of the existing per-concept `ConceptKG`. A GPU pass crops every part from **all** build references of every concept and embeds them (SigLIP2) into a **crop store**. Pure code induces PartType hubs (HDBSCAN), gates them (coherence + ≥2 concepts), and selects a hub's centroid prototype. A single shared **transfer consumer** turns {own crops + k borrowed crops} into weighted IP-Adapter embeddings; the three arms (+Hub-k / +RawNN-k / +Random-k) differ **only** in the selection function. A starvation harness sweeps build sizes {1,2,3,5} on the 8 reliably-measured concepts and scores fidelity against their fixed held-out sets.

**Tech Stack:** Python 3.11, numpy 2.4, `sklearn.cluster.HDBSCAN` (sklearn 1.9), torch 2.6 + diffusers (SDXL + IP-Adapter-plus ViT-H), existing `graft` models (SigLIP2, DINOv3, GroundingDINO). Tests: pytest (`*_unit.py` pure, `*_gpu.py` marked `@pytest.mark.gpu`).

**Spec:** `docs/superpowers/specs/2026-09-05-relational-mmkg-parttype-transfer-design.md` (LOCKED). Section refs below (§, P#) point at it.

## Global Constraints

- **Python interpreter (all commands):** `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python` — call it `$PY`. The base conda env has no numpy/sklearn/torch. Run tests as `$PY -m pytest ...`.
- **GPU discipline (unchanged):** single shared RTX 4090; every GPU-touching phase runs in its own disposable subprocess (`graft/worker_*.py`), staged/resumable. Never hold VLM + detector + SDXL resident together. Reuse the `models.Models` lazy facade + `models.unload(...)`.
- **Model ids are fixed** — do not substitute (see `graft/config.py`): SigLIP2 `google/siglip2-base-patch16-384`, DINOv3 `facebook/dinov3-vitl16-pretrain-lvd1689m`, GroundingDINO `IDEA-Research/grounding-dino-base`, SDXL `stabilityai/stable-diffusion-xl-base-1.0`, IP-Adapter `h94/IP-Adapter` weight `ip-adapter-plus_sdxl_vit-h.safetensors`.
- **Clustering space is SigLIP2, never DINOv3** (P1) — DINOv3 is the fidelity metric; clustering in it would confound the eval.
- **Membership is SigLIP2 + HDBSCAN ONLY (P1); no LLM/VLM touches membership.** The VLM (P10) runs *only after* hubs are formed, to attach a human-readable `PartTypeRec.label`. That label is **non-functional**: nothing in the transfer consumer or generation reads it, and clustering never sees it. `assemble_graph` (Task 6) forms hubs with `label=""`; labeling (Task 6b) is a separate, skippable GPU step. Skipping it changes no result.
- **Hub labeling model + decoding are locked (P10):** `Qwen2.5-VL-7B-Instruct` (`GraftConfig.vlm_id`) via `models.vlm.describe`, on the hub's top-4 centroid-nearest crops, blind/name-free instruction. Decoding params are **explicit, config-backed** — new `GraftConfig` fields `hub_label_do_sample = False` and `hub_label_max_new_tokens = 512`, **passed explicitly** to `describe(...)` (NOT relying on `QwenVLM` defaults), so they serialize into `cfg.yaml` and cannot drift.
- **Selection-only invariant (§7.2, P5):** `+Hub-k`, `+RawNN-k`, `+Random-k` share ONE consumer path — identical `k` (P4=4), identical weighting/cap (P8), identical IP-Adapter conditioning. Only the selector that returns the k crop ids may differ. A unit test must assert this.
- **Parity/contamination (§7.4):** borrowed crops must exclude the target concept entirely AND any held-out image. Held-out images never enter the crop store.
- **Locked params:** P1 HDBSCAN(min_cluster_size=3, min_samples=1) on L2-normed SigLIP2; P2 hub valid iff ≥2 distinct concepts AND mean member↔centroid cosine ≥0.50; P3 prototype = k nearest hub centroid, ≤2 per source concept; P4 k=4; P6 concepts=8 (Bamboo, Ashok, Egyptian lotus, Nageshore, Avocado, Camphor Tree, Hijol, Ashore), build levels {1,2,3,5}, 3 keep-draws/level, transfer forced on; P7 τ<3 (deploy only); P8 own:borrowed=2:1 per crop, borrowed aggregate cap 0.7; P9 separate store referencing `ConceptKG` by concept id.
- **Data root:** `data/treevill/rawdata2` (as in the fast-eval runs). Dedup + adaptive split live in `graft/dataset.py`.

---

## File Structure

- `graft/graph_schema.py` (new) — `PartInstanceRec`, `PartTypeRec`, `PartTypeGraph` dataclasses + JSON load/save. One responsibility: the on-disk relational store (P9).
- `graft/parttype_graph.py` (new) — pure induction/gating/prototype: `induce_parttypes`, `hub_coherence`, `valid_hubs`, `centroid_prototype`. numpy + sklearn only.
- `graft/transfer.py` (new) — pure selection dispatch `select_borrowed` (the selection-only seam) + weighting `reference_weights` (P8).
- `graft/starvation.py` (new) — pure experiment scaffolding: `keep_draw`, constants `STARVE_CONCEPTS/STARVE_LEVELS/N_DRAWS`, per-cell iterator.
- `graft/starvation_analysis.py` (new) — pure readout: recovery curves + paired `+Hub-k`−`+RawNN-k` deltas, reusing the `scripts/stratify_heldout.py` idiom.
- `graft/models.py` (modify, `SdxlIpGenerator`) — add `generate_multi(prompt, ref_images, weights, seed, ...)` weighted-embeds conditioning.
- `graft/worker_build_partcrops.py` (new) — GPU worker: crop every part from every build ref of one concept, embed SigLIP2, emit `PartInstanceRec`s.
- `graft/build_graph.py` (new) — driver: run the crop-store workers (all concepts), then induce+gate hubs → write `PartTypeGraph` (hubs get `label=""`). Also holds the pure `top_centroid_members` helper (Task 6b).
- `graft/worker_label_hubs.py` (new) — GPU worker: attach a P10 VLM label to each already-formed hub. **Interpretability-only, non-functional**, skippable; never read by the consumer.
- `graft/worker_starve_cell.py` (new) — GPU worker: one (concept, level, draw, condition) → generate + score, emit a row.
- `graft/run_starvation.py` (new) — driver: enumerate cells (P6), dispatch workers via `scripts/gpu_queue.sh`, collect rows, call analysis.
- Tests: matching `tests/test_*_unit.py` (pure) and `tests/test_*_gpu.py` (GPU).

---

### Task 1: Graph store schema (`graph_schema.py`)

**Files:**
- Create: `graft/graph_schema.py`
- Test: `tests/test_graph_schema.py`

**Interfaces:**
- Produces:
  - `PartInstanceRec(id: str, concept: str, part: str, ref_path: str, crop_path: str, siglip2: list[float])`
  - `PartTypeRec(id: int, part: str, member_ids: list[str], centroid: list[float], coherence: float, label: str)`
  - `PartTypeGraph(part_instances: list[PartInstanceRec], part_types: list[PartTypeRec])` with `to_json(path)` / `from_json(path)` and helper `instances_by_id() -> dict[str, PartInstanceRec]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_schema.py
from graft.graph_schema import PartInstanceRec, PartTypeRec, PartTypeGraph

def test_graph_roundtrips_through_json(tmp_path):
    inst = PartInstanceRec(id="Bamboo::leaf::0", concept="Bamboo", part="leaf",
                           ref_path="a.jpg", crop_path="a_leaf.png", siglip2=[0.1, 0.2])
    hub = PartTypeRec(id=3, part="leaf", member_ids=["Bamboo::leaf::0"],
                      centroid=[0.1, 0.2], coherence=0.71, label="")
    g = PartTypeGraph(part_instances=[inst], part_types=[hub])
    p = tmp_path / "graph.json"
    g.to_json(str(p))
    g2 = PartTypeGraph.from_json(str(p))
    assert g2.part_types[0].coherence == 0.71
    assert g2.instances_by_id()["Bamboo::leaf::0"].concept == "Bamboo"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_graph_schema.py -v`
Expected: FAIL (`ModuleNotFoundError: graft.graph_schema`).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/graph_schema.py
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field

@dataclass
class PartInstanceRec:
    id: str
    concept: str
    part: str
    ref_path: str
    crop_path: str
    siglip2: list[float]

@dataclass
class PartTypeRec:
    id: int
    part: str
    member_ids: list[str]
    centroid: list[float]
    coherence: float
    label: str = ""

@dataclass
class PartTypeGraph:
    part_instances: list[PartInstanceRec] = field(default_factory=list)
    part_types: list[PartTypeRec] = field(default_factory=list)

    def instances_by_id(self) -> dict[str, PartInstanceRec]:
        return {i.id: i for i in self.part_instances}

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"part_instances": [asdict(i) for i in self.part_instances],
                       "part_types": [asdict(h) for h in self.part_types]}, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "PartTypeGraph":
        with open(path) as f:
            raw = json.load(f)
        return cls(
            part_instances=[PartInstanceRec(**i) for i in raw["part_instances"]],
            part_types=[PartTypeRec(**h) for h in raw["part_types"]],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_graph_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/graph_schema.py tests/test_graph_schema.py
git commit -m "feat(mmkg): PartType graph store schema (P9)"
```

---

### Task 2: Hub induction, coherence & validity (`parttype_graph.py`)

**Files:**
- Create: `graft/parttype_graph.py`
- Test: `tests/test_parttype_graph_unit.py`

**Interfaces:**
- Consumes: nothing from prior tasks (operates on numpy arrays).
- Produces:
  - `induce_parttypes(embeds: np.ndarray, *, min_cluster_size: int = 3, min_samples: int = 1) -> np.ndarray` — HDBSCAN labels (`-1` = noise), embeds assumed L2-normalized.
  - `hub_coherence(embeds: np.ndarray, labels: np.ndarray) -> dict[int, float]` — mean member↔centroid cosine per non-noise label.
  - `valid_hubs(labels, concepts, coherences, *, min_concepts=2, min_coherence=0.5) -> set[int]` (P2).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parttype_graph_unit.py
import numpy as np
from graft.parttype_graph import induce_parttypes, hub_coherence, valid_hubs

def _norm(a): return a / np.linalg.norm(a, axis=1, keepdims=True)

def test_two_tight_groups_induce_two_hubs():
    rng = np.random.default_rng(0)
    a = _norm(np.array([1.,0,0]) + 0.01*rng.standard_normal((6,3)))
    b = _norm(np.array([0,1.,0]) + 0.01*rng.standard_normal((6,3)))
    labels = induce_parttypes(np.vstack([a,b]), min_cluster_size=3)
    assert len({l for l in labels if l != -1}) == 2

def test_valid_hubs_apply_concept_and_coherence_gates():
    labels = np.array([0,0,0, 1,1,1])
    concepts = ["X","X","X", "P","Q","R"]          # hub 0 = 1 concept; hub 1 = 3 concepts
    coh = {0: 0.9, 1: 0.9}
    assert valid_hubs(labels, concepts, coh) == {1}   # hub 0 fails >=2 concepts
    coh2 = {0: 0.9, 1: 0.4}
    assert valid_hubs(labels, concepts, coh2) == set()  # hub 1 now fails coherence 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_parttype_graph_unit.py -v`
Expected: FAIL (`ModuleNotFoundError: graft.parttype_graph`).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/parttype_graph.py
from __future__ import annotations
import numpy as np

def induce_parttypes(embeds, *, min_cluster_size: int = 3, min_samples: int = 1):
    from sklearn.cluster import HDBSCAN
    embeds = np.asarray(embeds, dtype=float)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                        metric="euclidean")  # euclidean on L2-normed ~ cosine
    return clusterer.fit_predict(embeds)

def hub_coherence(embeds, labels) -> dict[int, float]:
    embeds = np.asarray(embeds, dtype=float)
    labels = np.asarray(labels)
    out = {}
    for h in sorted({int(l) for l in labels if l != -1}):
        members = embeds[labels == h]
        centroid = members.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) or 1.0)
        out[h] = float((members @ centroid).mean())
    return out

def valid_hubs(labels, concepts, coherences, *, min_concepts: int = 2,
               min_coherence: float = 0.5) -> set[int]:
    labels = np.asarray(labels)
    concepts = list(concepts)
    valid = set()
    for h, coh in coherences.items():
        idx = np.where(labels == h)[0]
        n_concepts = len({concepts[i] for i in idx})
        if n_concepts >= min_concepts and coh >= min_coherence:
            valid.add(int(h))
    return valid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_parttype_graph_unit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/parttype_graph.py tests/test_parttype_graph_unit.py
git commit -m "feat(mmkg): HDBSCAN hub induction + coherence/validity gates (P1,P2)"
```

---

### Task 3: Hub prototype + selection dispatch + weighting (`transfer.py` + prototype)

**Files:**
- Create: `graft/transfer.py`
- Modify: `graft/parttype_graph.py` (add `centroid_prototype`)
- Test: `tests/test_transfer_unit.py`

**Interfaces:**
- Consumes: `hub_coherence`, `valid_hubs` (Task 2).
- Produces (in `parttype_graph.py`):
  - `centroid_prototype(embeds, labels, hub_id, concepts, *, k, max_per_concept=2, exclude_concept=None) -> list[int]` — indices of the k members nearest the hub centroid, ≤`max_per_concept` per source concept, excluding `exclude_concept` (P3).
- Produces (in `transfer.py`):
  - `select_borrowed(kind, *, query_embed, pool_embeds, pool_concepts, pool_labels, own_concept, hub_id, k, seed) -> list[int]` — dispatch over `"hub"|"rawnn"|"random"`; ALL exclude `own_concept`; returns ≤k indices into `pool_*`.
  - `reference_weights(n_own, n_borrowed, *, own_per=2.0, borrowed_per=1.0, cap=0.7) -> np.ndarray` — normalized weights over `[own..., borrowed...]` (P8), borrowed aggregate rescaled to ≤`cap`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_unit.py
import numpy as np
from graft.transfer import select_borrowed, reference_weights

def _norm(a): return a / np.linalg.norm(a, axis=1, keepdims=True)

def test_weights_cap_and_ratio():
    # 1 own + 4 borrowed: raw 2 vs 1each -> own 2/6=0.333, borrowed 0.667 (< cap 0.7)
    w = reference_weights(1, 4)
    assert abs(w.sum() - 1.0) < 1e-9
    assert abs(w[0] - 2/6) < 1e-6
    # 1 own + 20 borrowed would be borrowed 20/22=0.909 -> capped to 0.7
    w2 = reference_weights(1, 20)
    assert abs(w2[1:].sum() - 0.7) < 1e-6
    assert abs(w2[0] - 0.3) < 1e-6

def test_selection_only_differs_by_kind_not_count():
    rng = np.random.default_rng(1)
    pool = _norm(rng.standard_normal((30, 8)))
    concepts = ["own"]*5 + [f"c{i}" for i in range(25)]
    labels = np.array([0]*15 + [1]*15)
    q = pool[0]
    kw = dict(query_embed=q, pool_embeds=pool, pool_concepts=concepts,
              pool_labels=labels, own_concept="own", hub_id=0, k=4, seed=7)
    hub = select_borrowed("hub", **kw)
    raw = select_borrowed("rawnn", **kw)
    rnd = select_borrowed("random", **kw)
    assert len(hub) == len(raw) == len(rnd) == 4          # same count (P4)
    for sel in (hub, raw, rnd):
        assert all(concepts[i] != "own" for i in sel)     # never borrow from own concept
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_transfer_unit.py -v`
Expected: FAIL (`ModuleNotFoundError: graft.transfer`).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/parttype_graph.py  (append)
def centroid_prototype(embeds, labels, hub_id, concepts, *, k,
                       max_per_concept: int = 2, exclude_concept=None) -> list[int]:
    import numpy as np
    embeds = np.asarray(embeds, dtype=float); labels = np.asarray(labels)
    concepts = list(concepts)
    idx = [i for i in np.where(labels == hub_id)[0] if concepts[i] != exclude_concept]
    if not idx:
        return []
    members = embeds[idx]
    centroid = members.mean(axis=0); centroid /= (np.linalg.norm(centroid) or 1.0)
    order = sorted(idx, key=lambda i: -float(embeds[i] @ centroid))
    chosen, per = [], {}
    for i in order:
        c = concepts[i]
        if per.get(c, 0) >= max_per_concept:
            continue
        chosen.append(i); per[c] = per.get(c, 0) + 1
        if len(chosen) == k:
            break
    return chosen
```

```python
# graft/transfer.py
from __future__ import annotations
import numpy as np
from graft.parttype_graph import centroid_prototype

def select_borrowed(kind, *, query_embed, pool_embeds, pool_concepts,
                    pool_labels, own_concept, hub_id, k, seed) -> list[int]:
    pool_embeds = np.asarray(pool_embeds, dtype=float)
    others = [i for i in range(len(pool_concepts)) if pool_concepts[i] != own_concept]
    if kind == "hub":
        return centroid_prototype(pool_embeds, pool_labels, hub_id, pool_concepts,
                                  k=k, max_per_concept=2, exclude_concept=own_concept)
    if kind == "rawnn":
        q = np.asarray(query_embed, dtype=float)
        order = sorted(others, key=lambda i: -float(pool_embeds[i] @ q))
        return order[:k]
    if kind == "random":
        rng = np.random.default_rng(seed)
        return list(rng.permutation(others)[:k])
    raise ValueError(f"unknown selector: {kind}")

def reference_weights(n_own, n_borrowed, *, own_per=2.0, borrowed_per=1.0,
                      cap=0.7) -> np.ndarray:
    w = np.array([own_per]*n_own + [borrowed_per]*n_borrowed, dtype=float)
    w = w / w.sum()
    if n_borrowed and w[n_own:].sum() > cap:
        w[n_own:] *= cap / w[n_own:].sum()
        if n_own:
            w[:n_own] *= (1 - cap) / w[:n_own].sum()
    return w
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_transfer_unit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/transfer.py graft/parttype_graph.py tests/test_transfer_unit.py
git commit -m "feat(mmkg): centroid prototype + selection dispatch + P8 weighting"
```

---

### Task 4: Weighted multi-reference IP-Adapter conditioning (`SdxlIpGenerator.generate_multi`)

**Files:**
- Modify: `graft/models.py` (`SdxlIpGenerator`, after `generate`, ~line 433)
- Test: `tests/test_generate_multi_gpu.py`

**Interfaces:**
- Consumes: `reference_weights` (Task 3).
- Produces: `SdxlIpGenerator.generate_multi(prompt: str, ref_images: list[Image.Image], weights: Sequence[float], seed: int, negative: str = _DEFAULT_NEGATIVE, steps: int = 50) -> Image.Image` — conditions on the weighted average of per-image IP-Adapter embeds. `weights` aligns to `ref_images`, sums to 1.

- [ ] **Step 1: Write the failing test** (GPU)

```python
# tests/test_generate_multi_gpu.py
import pytest
from PIL import Image
from graft.models import SdxlIpGenerator

@pytest.mark.gpu
def test_generate_multi_runs_and_is_deterministic():
    gen = SdxlIpGenerator(ip_scale=0.6)
    refs = [Image.new("RGB", (224, 224), c) for c in ((0,120,0), (10,90,10))]
    a = gen.generate_multi("a photo of a plant", refs, [0.6, 0.4], seed=0, steps=6)
    b = gen.generate_multi("a photo of a plant", refs, [0.6, 0.4], seed=0, steps=6)
    assert a.size == (1024, 1024)
    assert list(a.getdata()) == list(b.getdata())   # same seed -> identical
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_generate_multi_gpu.py -v -m gpu`
Expected: FAIL (`AttributeError: 'SdxlIpGenerator' object has no attribute 'generate_multi'`).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/models.py  (inside SdxlIpGenerator, after generate())
    def generate_multi(self, prompt, ref_images, weights, seed,
                       negative: str = _DEFAULT_NEGATIVE, steps: int = 50):
        """Condition on the weighted average of each ref image's IP-Adapter
        embed. weights align to ref_images and sum to 1 (see transfer.reference_weights)."""
        import numpy as np
        pipe = self.pipe_ip
        # One embed per reference image (list -> stacked along dim 1 by diffusers).
        embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=[[img] for img in ref_images],
            ip_adapter_image_embeds=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )  # list len = n_refs; each tensor [2, 1, seq, dim] (neg, cond)
        w = self._torch.tensor(np.asarray(weights, dtype="float32"),
                               device=self.device, dtype=embeds[0].dtype)
        stacked = self._torch.stack(embeds, dim=0)              # [n_refs, 2, 1, seq, dim]
        avg = (stacked * w.view(-1, 1, 1, 1, 1)).sum(dim=0)      # [2, 1, seq, dim]
        generator = self._torch.Generator(device=self.device).manual_seed(seed)
        return pipe(
            prompt=prompt, negative_prompt=negative, num_inference_steps=steps,
            ip_adapter_image_embeds=[avg], generator=generator,
        ).images[0]
```

- [ ] **Step 4: Run test to verify it passes** (needs the 4090; behind `scripts/gpu_queue.sh` if contended)

Run: `$PY -m pytest tests/test_generate_multi_gpu.py -v -m gpu`
Expected: PASS.
> If `prepare_ip_adapter_image_embeds` returns a single tensor for a multi-image list on this diffusers version, fall back to encoding each image separately (`ip_adapter_image=[img]` per call) and stacking — the weighted-average and `ip_adapter_image_embeds=[avg]` contract stays identical.

- [ ] **Step 5: Commit**

```bash
git add graft/models.py tests/test_generate_multi_gpu.py
git commit -m "feat(mmkg): weighted multi-reference IP-Adapter conditioning (P5)"
```

---

### Task 5: Part-crop store GPU worker (`worker_build_partcrops.py`)

**Files:**
- Create: `graft/worker_build_partcrops.py`
- Test: `tests/test_worker_build_partcrops_gpu.py`

**Interfaces:**
- Consumes: `graft.models.Models`, `graft.kg_build.PART_NAMES` and `_part_phrase`, `graft.dataset` split, `PartInstanceRec` (Task 1).
- Produces: a JSON list of `PartInstanceRec` for one concept, written to `<out>/<concept>.json`. CLI: `python -m graft.worker_build_partcrops --root <data> --concept <name> --out <dir> --config <yaml>`. Crops every part from **every build ref** (never held-out, §7.4), embeds SigLIP2.

- [ ] **Step 1: Write the failing test** (GPU smoke on one concept)

```python
# tests/test_worker_build_partcrops_gpu.py
import json, subprocess, sys, pytest
from pathlib import Path

@pytest.mark.gpu
def test_partcrops_worker_emits_multi_ref_instances(tmp_path):
    out = tmp_path / "crops"
    subprocess.run([sys.executable, "-m", "graft.worker_build_partcrops",
                    "--root", "data/treevill/rawdata2", "--concept", "Bamboo",
                    "--out", str(out), "--config", "configs/pipeline_eval_run.yaml"],
                   check=True)
    recs = json.loads((out / "Bamboo.json").read_text())
    assert len(recs) >= 5                      # multiple parts x multiple build refs
    assert {r["concept"] for r in recs} == {"Bamboo"}
    assert len(recs[0]["siglip2"]) > 0
    # more than one distinct ref contributed (all-build-refs, not just medoid)
    assert len({r["ref_path"] for r in recs}) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_worker_build_partcrops_gpu.py -v -m gpu`
Expected: FAIL (`No module named graft.worker_build_partcrops`).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/worker_build_partcrops.py
"""GPU worker: crop every part from every BUILD reference of one concept and
embed each crop with SigLIP2 -> PartInstanceRec list. Held-out refs are never
touched (spec §7.4). Own subprocess per the graft GPU discipline."""
from __future__ import annotations
import argparse, json, os
from dataclasses import asdict
from graft.config import GraftConfig
from graft.dataset import load_species, split_refs    # split_refs -> (build_refs, heldout_refs)
from graft.kg_build import PART_NAMES, _part_phrase
from graft.graph_schema import PartInstanceRec
from graft.models import Models
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--concept", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = GraftConfig.from_yaml(a.config)
    sp = load_species(a.root, a.concept)
    build_refs, _heldout = split_refs(sp, k_build=cfg.k_build_refs, seed=0)  # seed=0 matches run_eval_fast
    models = Models(cfg)
    crops_dir = os.path.join(a.out, "_crops", a.concept); os.makedirs(crops_dir, exist_ok=True)
    recs = []
    for ri, ref_path in enumerate(build_refs):
        img = Image.open(ref_path).convert("RGB")
        for part in PART_NAMES:
            boxes = models.detector.detect(img, _part_phrase(part))
            if not boxes:
                continue
            x0, y0, x1, y1 = boxes[0]
            crop = img.crop((x0, y0, x1, y1))
            cp = os.path.join(crops_dir, f"{part}_{ri}.png"); crop.save(cp)
            emb = models.siglip.embed_image([crop])[0].tolist()   # L2-normed
            recs.append(PartInstanceRec(id=f"{a.concept}::{part}::{ri}", concept=a.concept,
                                        part=part, ref_path=ref_path, crop_path=cp, siglip2=emb))
    models.unload("detector"); models.unload("siglip")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.concept}.json"), "w") as f:
        json.dump([asdict(r) for r in recs], f)

if __name__ == "__main__":
    main()
```
> RESOLVED: the split is `load_species(root, name)` → `split_refs(sp, k_build=cfg.k_build_refs, seed=0)` returning `(build_refs, heldout_refs)` (deduped; `seed=0` is what `run_eval_fast` uses, so build/held-out are consistent across tasks). Do not re-implement dedup.

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_worker_build_partcrops_gpu.py -v -m gpu`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/worker_build_partcrops.py tests/test_worker_build_partcrops_gpu.py
git commit -m "feat(mmkg): all-build-refs part-crop store GPU worker (P1,§7.4)"
```

---

### Task 6: Graph build driver (`build_graph.py`)

**Files:**
- Create: `graft/build_graph.py`
- Test: `tests/test_build_graph_unit.py`

**Interfaces:**
- Consumes: `PartInstanceRec`/`PartTypeGraph` (Task 1), `induce_parttypes`/`hub_coherence`/`valid_hubs` (Task 2).
- Produces:
  - `assemble_graph(instances: list[PartInstanceRec]) -> PartTypeGraph` — pure: per part category, cluster member SigLIP2 embeds, compute coherence, keep only valid hubs (P2), emit `PartTypeRec`s (id namespaced per part) and set each instance's membership. Instances in invalid/noise hubs are retained in the store but belong to no `PartTypeRec`.
  - `main()` CLI: read all `<crops_dir>/*.json` worker outputs → `assemble_graph` → write `<out>/graph.json`.

- [ ] **Step 1: Write the failing test** (pure — synthetic instances, no GPU)

```python
# tests/test_build_graph_unit.py
import numpy as np
from graft.graph_schema import PartInstanceRec
from graft.build_graph import assemble_graph

def _rec(cid, part, ri, vec):
    return PartInstanceRec(id=f"{cid}::{part}::{ri}", concept=cid, part=part,
                           ref_path=f"{cid}{ri}.jpg", crop_path="x.png",
                           siglip2=list(vec/np.linalg.norm(vec)))

def test_assemble_builds_valid_cross_concept_leaf_hub():
    rng = np.random.default_rng(0)
    insts = []
    # 6 leaves near [1,0,0] across 3 concepts -> one valid hub (>=2 concepts, coherent)
    for i, c in enumerate(["A","A","B","B","C","C"]):
        insts.append(_rec(c, "leaf", i, np.array([1.,0,0]) + 0.01*rng.standard_normal(3)))
    g = assemble_graph(insts)
    leaf_hubs = [h for h in g.part_types if h.part == "leaf"]
    assert len(leaf_hubs) == 1
    assert len({insts[i].concept for i in range(6)}) == 3
    assert leaf_hubs[0].coherence >= 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_build_graph_unit.py -v`
Expected: FAIL (`No module named graft.build_graph`).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/build_graph.py
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from graft.graph_schema import PartInstanceRec, PartTypeRec, PartTypeGraph
from graft.parttype_graph import induce_parttypes, hub_coherence, valid_hubs

def assemble_graph(instances: list[PartInstanceRec]) -> PartTypeGraph:
    part_types: list[PartTypeRec] = []
    next_id = 0
    for part in sorted({i.part for i in instances}):
        members = [i for i in instances if i.part == part]
        if len(members) < 3:
            continue
        embeds = np.array([m.siglip2 for m in members], dtype=float)
        concepts = [m.concept for m in members]
        labels = induce_parttypes(embeds)
        coh = hub_coherence(embeds, labels)
        keep = valid_hubs(labels, concepts, coh)
        for h in sorted(keep):
            idx = np.where(labels == h)[0]
            centroid = embeds[idx].mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) or 1.0)
            part_types.append(PartTypeRec(
                id=next_id, part=part, member_ids=[members[i].id for i in idx],
                centroid=list(centroid), coherence=float(coh[h])))
            next_id += 1
    return PartTypeGraph(part_instances=list(instances), part_types=part_types)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    insts = []
    for p in sorted(glob.glob(os.path.join(a.crops_dir, "*.json"))):
        insts += [PartInstanceRec(**r) for r in json.loads(open(p).read())]
    os.makedirs(a.out, exist_ok=True)
    assemble_graph(insts).to_json(os.path.join(a.out, "graph.json"))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_build_graph_unit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/build_graph.py tests/test_build_graph_unit.py
git commit -m "feat(mmkg): graph build driver (induce+gate hubs -> PartTypeGraph)"
```

---

### Task 6b: Hub semantic labeling (interpretability-only, GPU) (`worker_label_hubs.py`)

> This task adds **non-functional** metadata (P10). It runs *after* Task 6 and reads nothing back into the pipeline. The transfer consumer (Task 8) never reads `label`. Membership is already fixed by Task 6 (SigLIP2+HDBSCAN); this only names the finished hubs. Skipping this task entirely leaves `label=""` and changes no result.

**Files:**
- Modify: `graft/config.py` (add locked decoding fields `hub_label_do_sample`, `hub_label_max_new_tokens`)
- Modify: `graft/models.py` (`QwenVLM.describe`/`generate` accept explicit `do_sample`/`max_new_tokens`)
- Modify: `graft/build_graph.py` (add pure helper `top_centroid_members`)
- Create: `graft/worker_label_hubs.py`
- Test: `tests/test_config.py` (new field defaults), `tests/test_label_hubs_unit.py` (helper, pure), `tests/test_label_hubs_gpu.py` (worker, GPU)

**Interfaces:**
- Consumes: `PartTypeGraph`/`PartTypeRec`/`PartInstanceRec` (Task 1), `graft.models.Models` VLM.
- Produces:
  - `GraftConfig.hub_label_do_sample: bool = False`, `GraftConfig.hub_label_max_new_tokens: int = 512` (P10, serialized into `cfg.yaml`).
  - `QwenVLM.describe(images, instruction, *, do_sample: bool | None = None, max_new_tokens: int | None = None) -> str` — forwards explicit decoding (falls back to instance defaults when `None`, so `kg_build` is unaffected).
  - `top_centroid_members(graph: PartTypeGraph, hub: PartTypeRec, k: int = 4) -> list[str]` — the k member ids whose SigLIP2 embed is nearest the hub centroid (pure).
  - CLI `python -m graft.worker_label_hubs --graph <graph.json> --config <yaml>` — sets each `PartTypeRec.label` via the locked P10 setup (decoding passed **explicitly** from config) and rewrites `graph.json` in place.

- [ ] **Step 1: Lock decoding params in config + forward them through `describe` (with a pure config test)**

Add to `graft/config.py` `GraftConfig` (near the other tunables):
```python
    hub_label_do_sample: bool = False        # P10: locked, serialized into cfg.yaml
    hub_label_max_new_tokens: int = 512       # P10: locked
```

Extend `graft/models.py` `QwenVLM` so decoding is explicit (keep current behavior as the fallback so `kg_build` is untouched):
```python
    # QwenVLM.describe(...): thread through explicit overrides
    def describe(self, images, instruction, *, do_sample=None, max_new_tokens=None) -> str:
        ...  # build inputs as today, then:
        out = self.model.generate(
            **inputs,
            do_sample=self.do_sample if do_sample is None else do_sample,
            max_new_tokens=self.max_new_tokens if max_new_tokens is None else max_new_tokens,
        )
        ...  # decode as today
```
(If `QwenVLM.__init__` has no `self.do_sample`, add `self.do_sample = False` there; `self.max_new_tokens` already exists.)

Config test:
```python
# tests/test_config.py  (add)
from graft.config import GraftConfig
def test_hub_label_decoding_defaults_are_locked():
    c = GraftConfig()
    assert c.hub_label_do_sample is False
    assert c.hub_label_max_new_tokens == 512
```
Run: `$PY -m pytest tests/test_config.py -v` → PASS. (The `describe` signature change is covered by the Task-6b GPU smoke, Step 5.)

- [ ] **Step 2: Write the failing unit test (helper)**

```python
# tests/test_label_hubs_unit.py
import numpy as np
from graft.graph_schema import PartInstanceRec, PartTypeRec, PartTypeGraph
from graft.build_graph import top_centroid_members

def test_top_centroid_members_picks_nearest():
    c = [1.0, 0.0]
    insts = [
        PartInstanceRec("m0","A","leaf","a.jpg","a.png", [0.99, 0.14]),   # near centroid
        PartInstanceRec("m1","B","leaf","b.jpg","b.png", [0.98, 0.20]),   # near
        PartInstanceRec("m2","C","leaf","c.jpg","c.png", [0.20, 0.98]),   # far
    ]
    hub = PartTypeRec(id=0, part="leaf", member_ids=["m0","m1","m2"], centroid=c, coherence=0.7)
    g = PartTypeGraph(part_instances=insts, part_types=[hub])
    assert top_centroid_members(g, hub, k=2) == ["m0", "m1"]
```

- [ ] **Step 3: Run to verify it fails**

Run: `$PY -m pytest tests/test_label_hubs_unit.py -v`
Expected: FAIL (`cannot import name 'top_centroid_members'`).

- [ ] **Step 4: Implement the pure helper + the GPU worker**

```python
# graft/build_graph.py  (append)
def top_centroid_members(graph, hub, k: int = 4) -> list[str]:
    import numpy as np
    by_id = graph.instances_by_id()
    c = np.asarray(hub.centroid, dtype=float)
    scored = [(mid, float(np.asarray(by_id[mid].siglip2, dtype=float) @ c))
              for mid in hub.member_ids if mid in by_id]
    scored.sort(key=lambda t: -t[1])
    return [mid for mid, _ in scored[:k]]
```

```python
# graft/worker_label_hubs.py
"""GPU worker: attach a human-readable label to each already-formed PartType hub
(spec P10). INTERPRETABILITY ONLY -- membership is fixed (SigLIP2+HDBSCAN, Task 6)
and no label is ever read by the transfer consumer. Deterministic Qwen2.5-VL."""
from __future__ import annotations
import argparse
from PIL import Image
from graft.config import GraftConfig
from graft.graph_schema import PartTypeGraph
from graft.build_graph import top_centroid_members
from graft.models import Models

INSTRUCTION = ("These are cropped photos of the same plant part from different plants. "
               "In 3-8 words, name the part type by its visible shape/texture/arrangement. "
               "Reply with only the phrase.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True); ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = GraftConfig.from_yaml(a.config)
    graph = PartTypeGraph.from_json(a.graph)
    by_id = graph.instances_by_id()
    models = Models(cfg)
    for hub in graph.part_types:
        crops = [Image.open(by_id[m].crop_path).convert("RGB")
                 for m in top_centroid_members(graph, hub, k=4)]
        hub.label = models.vlm.describe(              # decoding passed EXPLICITLY from config (P10)
            crops, INSTRUCTION,
            do_sample=cfg.hub_label_do_sample,
            max_new_tokens=cfg.hub_label_max_new_tokens,
        ).strip()
    models.unload("vlm")
    graph.to_json(a.graph)                         # rewrite in place; membership untouched

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the helper unit test (pure)**

Run: `$PY -m pytest tests/test_label_hubs_unit.py -v`
Expected: PASS.

- [ ] **Step 6: Write + run the GPU smoke test**

```python
# tests/test_label_hubs_gpu.py
import subprocess, sys, shutil, pytest
from pathlib import Path
from graft.graph_schema import PartTypeGraph

@pytest.mark.gpu
def test_labeling_sets_labels_without_changing_membership(tmp_path):
    src = Path("outputs/mmkg/graph.json")
    g = tmp_path / "graph.json"; shutil.copy(src, g)
    before = PartTypeGraph.from_json(str(g))
    subprocess.run([sys.executable, "-m", "graft.worker_label_hubs",
                    "--graph", str(g), "--config", "configs/pipeline_eval_run.yaml"], check=True)
    after = PartTypeGraph.from_json(str(g))
    # membership identical; labels now populated
    assert [h.member_ids for h in after.part_types] == [h.member_ids for h in before.part_types]
    assert any(h.label for h in after.part_types)
```

Run: `$PY -m pytest tests/test_label_hubs_gpu.py -v -m gpu`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add graft/config.py graft/models.py graft/build_graph.py graft/worker_label_hubs.py \
        tests/test_config.py tests/test_label_hubs_unit.py tests/test_label_hubs_gpu.py
git commit -m "feat(mmkg): interpretability-only hub labeling w/ config-locked decoding (P10)"
```

---

### Task 7: Starvation scaffolding (`starvation.py`)

**Files:**
- Create: `graft/starvation.py`
- Test: `tests/test_starvation_unit.py`

**Interfaces:**
- Produces:
  - `STARVE_CONCEPTS: list[str]` (the 8, P6), `STARVE_LEVELS = [1,2,3,5]`, `N_DRAWS = 3`, `CONDITIONS = ["isolated","random","rawnn","hub"]`. (`+OracleHub-k` is a deferred follow-up — it needs a taxonomy map that does not exist yet; adding it as a stub would duplicate `hub`, so it is out of the Stage-A run.)
  - `keep_draw(build_refs: list[str], keep_n: int, seed: int) -> list[str]` — deterministic subset; `keep_n >= len(build_refs)` returns all (un-starved ceiling).
  - `iter_cells() -> Iterator[tuple[str,int,int]]` — `(concept, level, draw_index)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_starvation_unit.py
from graft.starvation import keep_draw, iter_cells, STARVE_CONCEPTS, STARVE_LEVELS, N_DRAWS

def test_keep_draw_deterministic_and_sized():
    refs = [f"r{i}.jpg" for i in range(5)]
    assert keep_draw(refs, 2, seed=0) == keep_draw(refs, 2, seed=0)   # deterministic
    assert len(keep_draw(refs, 2, seed=0)) == 2
    assert keep_draw(refs, 9, seed=0) == refs                          # ceiling
    assert keep_draw(refs, 2, seed=0) != keep_draw(refs, 2, seed=1)    # seed varies draw

def test_cells_cover_grid():
    cells = list(iter_cells())
    assert len(cells) == len(STARVE_CONCEPTS) * len(STARVE_LEVELS) * N_DRAWS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_starvation_unit.py -v`
Expected: FAIL (`No module named graft.starvation`).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/starvation.py
from __future__ import annotations
import numpy as np

STARVE_CONCEPTS = ["Bamboo","Ashok","Egyptian lotus","Nageshore",
                   "Avocado","Camphor Tree","Hijol","Ashore"]
STARVE_LEVELS = [1, 2, 3, 5]
N_DRAWS = 3
CONDITIONS = ["isolated","random","rawnn","hub"]   # +OracleHub-k deferred (needs taxonomy map)

def keep_draw(build_refs, keep_n, seed):
    refs = list(build_refs)
    if keep_n >= len(refs):
        return refs
    rng = np.random.default_rng(seed)
    idx = sorted(rng.permutation(len(refs))[:keep_n].tolist())
    return [refs[i] for i in idx]

def iter_cells():
    for c in STARVE_CONCEPTS:
        for lvl in STARVE_LEVELS:
            for d in range(N_DRAWS):
                yield (c, lvl, d)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_starvation_unit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/starvation.py tests/test_starvation_unit.py
git commit -m "feat(mmkg): starvation grid scaffolding (P6)"
```

---

### Task 8: Starvation cell worker (`worker_starve_cell.py`)

**Files:**
- Create: `graft/worker_starve_cell.py`
- Test: `tests/test_worker_starve_cell_gpu.py`

**Interfaces:**
- Consumes: `keep_draw`/`CONDITIONS` (Task 7), `PartTypeGraph` (Task 1), `select_borrowed`/`reference_weights` (Task 3), `SdxlIpGenerator.generate_multi` (Task 4), `graft.eval_common` scoring (`score_fidelity` against held-out — reuse the existing fast-eval scorer), `graft.dataset` split, `ConceptKG` prompt (`attribute_texts`).
- Produces: CLI `python -m graft.worker_starve_cell --concept C --level L --draw D --condition K --graph <graph.json> --kg <kg.json> --root <data> --out <dir> --config <yaml>`, writing one row `{concept, level, draw, condition, dino, siglip2, clip_i}` to `<out>/<concept>_<L>_<D>_<K>.json`. The five conditions share ONE consumer call; only `select_borrowed(kind=...)` differs (isolated = no borrowed; oraclehub = hub_id from taxonomy map, deferred stub returns isolated until the oracle map exists).

- [ ] **Step 1: Write the failing test** (GPU, one cell)

```python
# tests/test_worker_starve_cell_gpu.py
import json, subprocess, sys, pytest
from pathlib import Path

@pytest.mark.gpu
def test_starve_cell_emits_scored_row(tmp_path):
    out = tmp_path / "rows"
    subprocess.run([sys.executable, "-m", "graft.worker_starve_cell",
        "--concept", "Bamboo", "--level", "1", "--draw", "0", "--condition", "hub",
        "--graph", "outputs/mmkg/graph.json", "--kg", "outputs/Bamboo/kg.json",
        "--root", "data/treevill/rawdata2", "--out", str(out),
        "--config", "configs/pipeline_eval_run.yaml"], check=True)
    row = json.loads((out / "Bamboo_1_0_hub.json").read_text())
    assert row["condition"] == "hub" and row["level"] == 1
    assert 0.0 <= row["dino"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_worker_starve_cell_gpu.py -v -m gpu`
Expected: FAIL (`No module named graft.worker_starve_cell`).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/worker_starve_cell.py
"""GPU worker: one starvation cell. Build the (possibly starved) own-crop set for
one part-conditioned generation, borrow k crops via the chosen selector (or none
for 'isolated'), condition once via generate_multi, score vs held-out. All arms
share this single consumer; only select_borrowed(kind=...) differs (§7.2 invariant)."""
from __future__ import annotations
import argparse, json, os
import numpy as np
from PIL import Image
from graft.config import GraftConfig
from graft.dataset import load_species, split_refs
from graft.graph_schema import PartTypeGraph
from graft.schema import ConceptKG
from graft.starvation import keep_draw
from graft.transfer import select_borrowed, reference_weights
from graft.models import Models
from graft import metrics                              # metrics.image_fidelity(gen, refs, embedder)

K = 4

def _own_crops(graph, concept, kept_refs):
    """This concept's crops whose ref survived starvation (any part)."""
    return [i for i in graph.part_instances
            if i.concept == concept and i.ref_path in set(kept_refs)]

def _hub_of(graph, inst_id):
    for h in graph.part_types:
        if inst_id in h.member_ids:
            return h.id
    return None

def main():
    ap = argparse.ArgumentParser()
    for f in ("concept","condition","graph","kg","root","out","config"):
        ap.add_argument(f"--{f}", required=True)
    ap.add_argument("--level", type=int, required=True); ap.add_argument("--draw", type=int, required=True)
    a = ap.parse_args()
    cfg = GraftConfig.from_yaml(a.config)
    graph = PartTypeGraph.from_json(a.graph); kg = ConceptKG.from_json(a.kg)
    sp = load_species(a.root, a.concept)
    build_refs, heldout = split_refs(sp, k_build=cfg.k_build_refs, seed=0)  # seed=0 matches run_eval_fast
    kept = keep_draw(build_refs, a.level, seed=a.draw)

    own = _own_crops(graph, a.concept, kept)
    own_embeds = np.array([o.siglip2 for o in own], dtype=float) if own else np.zeros((0, 1))
    own_imgs = [Image.open(o.crop_path).convert("RGB") for o in own]

    borrowed_imgs = []
    # "isolated" borrows nothing; "oraclehub" is a deferred stub (needs a taxonomy map
    # that doesn't exist yet) and behaves as isolated; a concept with no surviving own
    # crops (own empty) also skips borrowing (no anchor part, no query embedding).
    if a.condition not in ("isolated", "oraclehub") and own:
        own_part = own[0].part
        # Pool MUST be restricted to the transfer's part for ALL arms -- otherwise
        # +RawNN/+Random borrow off-part crops while +Hub (hubs are within-part) does not,
        # breaking the selection-only invariant + P5.
        pool = [p for p in graph.part_instances if p.part == own_part]
        if pool:
            pool_embeds = np.array([p.siglip2 for p in pool], dtype=float)
            pool_concepts = [p.concept for p in pool]
            pool_labels = np.array([_hub_of(graph, p.id) if _hub_of(graph, p.id) is not None else -1
                                    for p in pool])
            hub_id = _hub_of(graph, own[0].id)
            q = own_embeds[0]                          # own non-empty here, so this is safe
            # SELECTION-ONLY INVARIANT: all borrowing arms use the SAME realized count.
            # The hub bounds it (coherent, <=2/concept, own-excluded members); +RawNN/+Random
            # then draw EXACTLY k_eff crops, so the arms differ only in WHICH crops, never how
            # many. No valid hub -> k_eff=0 and no arm borrows (cell behaves like isolated).
            hub_sel = select_borrowed("hub", query_embed=q, pool_embeds=pool_embeds,
                                      pool_concepts=pool_concepts, pool_labels=pool_labels,
                                      own_concept=a.concept, hub_id=hub_id, k=K, seed=a.draw) if hub_id is not None else []
            k_eff = len(hub_sel)
            if k_eff:
                sel = hub_sel if a.condition == "hub" else select_borrowed(
                    a.condition, query_embed=q, pool_embeds=pool_embeds,
                    pool_concepts=pool_concepts, pool_labels=pool_labels,
                    own_concept=a.concept, hub_id=hub_id, k=k_eff, seed=a.draw)
                borrowed_imgs = [Image.open(pool[i].crop_path).convert("RGB") for i in sel]

    ref_imgs = own_imgs + borrowed_imgs
    if not ref_imgs:                                   # fully starved, no borrow: fall back to clean
        ref_imgs, weights = [], []
    else:
        weights = reference_weights(len(own_imgs), len(borrowed_imgs)).tolist()

    prompt = "a photo of a plant, " + ", ".join(kg.attribute_texts()[:8])
    models = Models(cfg)
    gen = models.generator                              # SdxlIpGenerator via the facade
    if ref_imgs:
        img = gen.generate_multi(prompt, ref_imgs, weights, seed=0)
    else:
        img = gen.generate(prompt, None, seed=0)
    models.unload("generator")                          # free SDXL+IP before loading scorers (single-GPU)
    held_imgs = [Image.open(p).convert("RGB") for p in heldout]
    scores = {                                          # metrics.image_fidelity = mean cosine vs held-out
        "dino": metrics.image_fidelity(img, held_imgs, models.dino),
        "siglip2": metrics.image_fidelity(img, held_imgs, models.siglip),
        "clip_i": metrics.image_fidelity(img, held_imgs, models.clip),
    }
    os.makedirs(a.out, exist_ok=True)
    row = {"concept": a.concept, "level": a.level, "draw": a.draw,
           "condition": a.condition, **scores}
    with open(os.path.join(a.out, f"{a.concept}_{a.level}_{a.draw}_{a.condition}.json"), "w") as f:
        json.dump(row, f)

if __name__ == "__main__":
    main()
```
> RESOLVED accessors (verified against the codebase — use exactly these, do not re-implement): SDXL-IP generator = `models.generator` (a `SdxlIpGenerator`); split = `load_species(root, name)` → `split_refs(sp, k_build=cfg.k_build_refs, seed=0)`; fidelity = `metrics.image_fidelity(gen_img, held_imgs, embedder)` computed once per embedder over `models.dino` / `models.siglip` / `models.clip`. Keep the single-`generate_multi` consumer for ALL borrowing arms; the arms differ only in the crop set the selector returns, and all use the same `k_eff` (above).
>
> STAGE-A SIMPLIFICATION (known, revisit at the scheduled GPU run): this worker conditions on the concept's whole own-crop set plus a single hub's borrowed prototype (the hub of `own[0]`), rather than doing per-part transfer for every part independently. That is an acceptable first integration for the starvation signal, but the per-part-transfer modeling should be validated/expanded when the GPU run is scheduled. The `@pytest.mark.gpu` smoke test here needs the real 4090 plus upstream artifacts (`outputs/mmkg/graph.json` from Task 6 over the 8 concepts, and each concept's `outputs/<concept>/kg.json`) — it is written now but executed during the scheduled integration run, not inline.

- [ ] **Step 4: Run test to verify it passes** (requires Tasks 5–6 artifacts: `outputs/mmkg/graph.json`, a built `outputs/Bamboo/kg.json`)

Run: `$PY -m pytest tests/test_worker_starve_cell_gpu.py -v -m gpu`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/worker_starve_cell.py tests/test_worker_starve_cell_gpu.py
git commit -m "feat(mmkg): starvation cell worker w/ shared consumer (§7.2 invariant)"
```

---

### Task 9: Analysis / readout (`starvation_analysis.py`)

**Files:**
- Create: `graft/starvation_analysis.py`
- Test: `tests/test_starvation_analysis_unit.py`

**Interfaces:**
- Produces:
  - `recovery_curve(rows, metric="dino") -> dict[str, dict[int, float]]` — per condition, mean metric per build level (averaged over concepts and draws).
  - `paired_delta(rows, cond_a="hub", cond_b="rawnn", level=1, metric="dino") -> dict` — per-concept paired mean over draws, then `{mean_delta, win_rate, n}` (the make-or-break `+Hub-k` − `+RawNN-k`, §7.2/§10).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_starvation_analysis_unit.py
from graft.starvation_analysis import recovery_curve, paired_delta

ROWS = [
    {"concept":"A","level":1,"draw":0,"condition":"hub","dino":0.5},
    {"concept":"A","level":1,"draw":0,"condition":"rawnn","dino":0.4},
    {"concept":"B","level":1,"draw":0,"condition":"hub","dino":0.6},
    {"concept":"B","level":1,"draw":0,"condition":"rawnn","dino":0.6},
    {"concept":"A","level":5,"draw":0,"condition":"hub","dino":0.7},
]

def test_recovery_curve_groups_by_level():
    curve = recovery_curve(ROWS, "dino")
    assert curve["hub"][1] == (0.5 + 0.6) / 2
    assert curve["hub"][5] == 0.7

def test_paired_delta_hub_vs_rawnn():
    d = paired_delta(ROWS, "hub", "rawnn", level=1, metric="dino")
    assert d["n"] == 2 and abs(d["mean_delta"] - 0.05) < 1e-9   # (+0.1, 0.0)/2
    assert d["win_rate"] == 0.5                                  # A wins, B ties
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_starvation_analysis_unit.py -v`
Expected: FAIL (`No module named graft.starvation_analysis`).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/starvation_analysis.py
from __future__ import annotations
import statistics as st
from collections import defaultdict

def _by(rows, **f):
    return [r for r in rows if all(r[k] == v for k, v in f.items())]

def recovery_curve(rows, metric="dino"):
    out = defaultdict(dict)
    conds = {r["condition"] for r in rows}; levels = sorted({r["level"] for r in rows})
    for c in conds:
        for lvl in levels:
            vals = [r[metric] for r in _by(rows, condition=c, level=lvl)]
            if vals:
                out[c][lvl] = st.mean(vals)
    return dict(out)

def paired_delta(rows, cond_a="hub", cond_b="rawnn", level=1, metric="dino"):
    concepts = sorted({r["concept"] for r in rows if r["level"] == level})
    deltas = []
    for c in concepts:
        a = [r[metric] for r in _by(rows, concept=c, level=level, condition=cond_a)]
        b = [r[metric] for r in _by(rows, concept=c, level=level, condition=cond_b)]
        if a and b:
            deltas.append(st.mean(a) - st.mean(b))   # mean over draws, then pair
    wins = sum(1 for d in deltas if d > 0)
    return {"mean_delta": st.mean(deltas) if deltas else None,
            "win_rate": wins / len(deltas) if deltas else None,
            "n": len(deltas)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_starvation_analysis_unit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/starvation_analysis.py tests/test_starvation_analysis_unit.py
git commit -m "feat(mmkg): starvation recovery curves + Hub-vs-RawNN paired delta (§10)"
```

---

### Task 10: Starvation run driver (`run_starvation.py`)

**Files:**
- Create: `graft/run_starvation.py`
- Test: `tests/test_run_starvation_unit.py`

**Interfaces:**
- Consumes: `iter_cells`/`CONDITIONS` (Task 7), `recovery_curve`/`paired_delta` (Task 9).
- Produces:
  - `plan_commands(graph, kg_dir, root, out, config) -> list[list[str]]` — one `worker_starve_cell` argv per (cell × condition); pure, testable.
  - `collect(out) -> list[dict]` — read all row json.
  - `main()` — resumable dispatch of `plan_commands` through `scripts/gpu_queue.sh` (skip cells whose row json exists), then write `recovery.json` (curves) + `hub_vs_rawnn.json` (paired deltas at level 1) + a `results.md`.

- [ ] **Step 1: Write the failing test** (pure — command planning + resume)

```python
# tests/test_run_starvation_unit.py
from graft.run_starvation import plan_commands
from graft.starvation import CONDITIONS, iter_cells

def test_plan_one_command_per_cell_and_condition():
    cmds = plan_commands("g.json", "outputs", "data/treevill/rawdata2", "out", "c.yaml")
    assert len(cmds) == len(list(iter_cells())) * len(CONDITIONS)
    sample = cmds[0]
    assert "--condition" in sample and "--graph" in sample and "g.json" in sample
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_run_starvation_unit.py -v`
Expected: FAIL (`No module named graft.run_starvation`).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/run_starvation.py
from __future__ import annotations
import argparse, glob, json, os, subprocess, sys
from graft.starvation import iter_cells, CONDITIONS
from graft.starvation_analysis import recovery_curve, paired_delta

def plan_commands(graph, kg_dir, root, out, config):
    cmds = []
    for (concept, level, draw) in iter_cells():
        for cond in CONDITIONS:
            cmds.append([sys.executable, "-m", "graft.worker_starve_cell",
                "--concept", concept, "--level", str(level), "--draw", str(draw),
                "--condition", cond, "--graph", graph,
                "--kg", os.path.join(kg_dir, concept, "kg.json"),
                "--root", root, "--out", out, "--config", config])
    return cmds

def _row_path(out, argv):                 # row filename mirrors the worker's output name
    c = argv[argv.index("--concept")+1]; l = argv[argv.index("--level")+1]
    dr = argv[argv.index("--draw")+1]; k = argv[argv.index("--condition")+1]
    return os.path.join(out, f"{c}_{l}_{dr}_{k}.json")

def collect(out):
    return [json.loads(open(p).read()) for p in sorted(glob.glob(os.path.join(out, "*_*_*_*.json")))]

def main():
    ap = argparse.ArgumentParser()
    for f in ("graph","kg-dir","root","out","config"):
        ap.add_argument(f"--{f}", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for argv in plan_commands(a.graph, a.kg_dir, a.root, a.out, a.config):
        if os.path.exists(_row_path(a.out, argv)):     # resume: skip done cells
            continue
        subprocess.run(["bash", "scripts/gpu_queue.sh", *argv], check=True)
    rows = collect(a.out)
    json.dump(recovery_curve(rows, "dino"), open(os.path.join(a.out, "recovery.json"), "w"), indent=2)
    json.dump(paired_delta(rows, "hub", "rawnn", level=1), open(os.path.join(a.out, "hub_vs_rawnn.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_run_starvation_unit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/run_starvation.py tests/test_run_starvation_unit.py
git commit -m "feat(mmkg): resumable starvation run driver + readout"
```

---

### Task 11: Full-suite regression + end-to-end dry check

**Files:**
- Modify: none (verification task)

- [ ] **Step 1: Run the full non-GPU unit suite**

Run: `$PY -m pytest tests -v -m "not gpu"`
Expected: PASS (all new `*_unit.py` + existing units green — no regressions in `schema`, `selection`, `eval_common`).

- [ ] **Step 2: Build the graph on the 8 starvation concepts (GPU, staged)**

Run:
```bash
for c in Bamboo Ashok "Egyptian lotus" Nageshore Avocado "Camphor Tree" Hijol Ashore; do
  bash scripts/gpu_queue.sh $PY -m graft.worker_build_partcrops \
    --root data/treevill/rawdata2 --concept "$c" --out outputs/mmkg/crops \
    --config configs/pipeline_eval_run.yaml
done
$PY -m graft.build_graph --crops-dir outputs/mmkg/crops --out outputs/mmkg
```
Expected: `outputs/mmkg/graph.json` exists with ≥1 valid cross-concept hub per common part (inspect coherence ≥0.5, ≥2 concepts).

- [ ] **Step 3: Run one full starvation cell per condition (GPU smoke) and read `hub_vs_rawnn.json`**

Run: `$PY -m graft.run_starvation --graph outputs/mmkg/graph.json --kg-dir outputs --root data/treevill/rawdata2 --out outputs/mmkg/starve --config configs/pipeline_eval_run.yaml`
Expected: `recovery.json` shows a curve rising with build level; `hub_vs_rawnn.json` reports `mean_delta`/`win_rate`/`n=8` at level 1 — **the make-or-break number** (§10). Report it honestly whatever the sign.

- [ ] **Step 4: Commit any fixups**

```bash
git add -A && git commit -m "test(mmkg): stage-A regression + end-to-end verification"
```

---

## Self-Review

**Spec coverage:** §4 schema → Tasks 1/5/6; §5 construction (visual/HDBSCAN, ≥2-concept, coherence) → Tasks 2/5/6 (P1,P2); §5.1 membership-is-visual-only + validation diagnostics → Task 6 (hubs formed with `label=""`) + Task 9; §5.1/P10 interpretability-only VLM labeling (membership untouched) → Task 6b; §6 transfer consumer + guard → Tasks 3/4/8 (P3,P5,P8); §7.1 starvation → Tasks 7/8/10 (P6); §7.2 conditions incl. quantity-matched controls + selection-only invariant → Tasks 3/8 (P4,P5); §7.4 parity → Task 5 (held-out excluded) + Task 3/8 (own concept excluded); §10 success/null (Hub vs RawNN) → Task 9. **Deferred by design:** oracle-hub taxonomy edges — `+OracleHub-k` is left out of the Stage-A `CONDITIONS` (Task 7) rather than stubbed, since without a taxonomy map it would duplicate `+Hub-k`; it becomes a real arm once a taxonomy map is added (a small follow-up). AttributeValue/Stage B (spec §8).

**Placeholder scan:** no "TBD"/"handle errors" left. Two explicit *verify-the-accessor* notes (Tasks 5/8) name the exact real functions to reuse (`dataset` split, `eval_common.score_fidelity`, `Models.sdxl_ip`) rather than inventing them — resolve at implementation, do not re-implement.

**Type consistency:** `PartInstanceRec`/`PartTypeRec`/`PartTypeGraph` used identically across Tasks 1/5/6/8; `select_borrowed(kind=...)` and `reference_weights(n_own,n_borrowed)` signatures match between Tasks 3 and 8; `k=4` (P4) fixed in Task 3 test and Task 8 `K`; row schema `{concept,level,draw,condition,dino,siglip2,clip_i}` consistent across Tasks 8/9/10.

**Known follow-ups (not blockers for the locked question):** the `+OracleHub-k` taxonomy arm and Stage-B AttributeValue layer are separate small plans, gated on this Stage-A `+Hub-k > +RawNN-k` result per spec §8/§10.
