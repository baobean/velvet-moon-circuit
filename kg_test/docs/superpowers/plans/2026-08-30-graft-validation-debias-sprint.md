# GRAFT Validation & De-bias Sprint — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the GRAFT eval trustworthy before the full Treevill benchmark — remove the sample-size threshold, fully neutralize the concept name, give B1/GRAFT identical name-free exemplar selection, fix two MMKG correctness bugs, and add the audit + prune-ablation + `ip_scale` sweep the sprint needs to decide the part-tree.

**Architecture:** Pure decision logic lives in small modules that take `graft.interfaces` Protocol instances (unit-tested with in-memory fakes); only `graft.models` touches HuggingFace weights. Every GPU-touching phase runs in its own disposable subprocess (`graft/worker_*.py`). This sprint adds config knobs that select behavior (`neutralize_name`, `exemplar_selection`, `use_part_tree`) so the neutral/named and full/pruned arms are config toggles, not code forks.

**Tech Stack:** Python, pytest (unit tests default; `-m gpu` tests deselected by default and run manually on the shared RTX 4090), numpy, PIL, HuggingFace (SDXL+IP-Adapter, Qwen2.5-VL, SigLIP2, DINOv3, CLIP, GroundingDINO).

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-08-30-graft-validation-debias-sprint-design.md`. This sprint gates the full run in `docs/superpowers/specs/2026-08-30-graft-phase2-design.md` §6.
- **Cached model ids are fixed** — never substitute a model id (see `graft/config.py` and `configs/pipeline.yaml`).
- **Neutral token is exactly `"plant"`.**
- **`GraftConfig.from_yaml` rejects unknown keys** — every new config field must also be acceptable in the two existing yamls (`configs/pipeline.yaml`, `configs/pipeline_eval_run.yaml`); adding fields with defaults is fine, renaming existing keys is not.
- **`k_build_refs` keeps its name but is reinterpreted as the build-size *cap*** (do not rename it — the yamls set it).
- **Single-GPU discipline unchanged:** models load/unload per phase; each GPU phase is its own subprocess.
- **TDD, DRY, YAGNI, frequent commits.** Unit tests use Protocol fakes; mark any test needing real weights with `@pytest.mark.gpu`.
- **Run unit tests with:** `cd ndbao_hbngoc/kg_test && python -m pytest -q` (gpu tests auto-deselected).
- **Commits:** there is currently no working git repo at this path (`.git` is an empty stub). Before Task 1, initialize one if the user has approved it; otherwise perform the "commit" steps as `git add`/`git commit` and they will no-op-fail visibly — flag to the user rather than skipping the checkpoint.

---

### Task 1: Config knobs

**Files:**
- Modify: `graft/config.py` (add three fields; document `k_build_refs` as a cap)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `GraftConfig.neutralize_name: bool = True`, `GraftConfig.exemplar_selection: str = "medoid"`, `GraftConfig.use_part_tree: bool = True`. `k_build_refs: int = 5` now means "max build refs (cap)".

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py  (append)
from graft.config import GraftConfig

def test_new_debias_defaults():
    cfg = GraftConfig()
    assert cfg.neutralize_name is True
    assert cfg.exemplar_selection == "medoid"
    assert cfg.use_part_tree is True

def test_from_yaml_accepts_new_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("neutralize_name: false\nexemplar_selection: text\nuse_part_tree: false\n")
    cfg = GraftConfig.from_yaml(str(p))
    assert cfg.neutralize_name is False and cfg.exemplar_selection == "text" and cfg.use_part_tree is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -q`
Expected: FAIL (`neutralize_name` not an attribute / unknown key).

- [ ] **Step 3: Write minimal implementation**

In `graft/config.py`, inside the `# Tunables.` block, add:

```python
    neutralize_name: bool = True          # replace the concept name with NEUTRAL_TOKEN everywhere
    exemplar_selection: str = "medoid"    # "medoid" (name-free) | "text" (Phase-1 reranker/text)
    use_part_tree: bool = True            # score part crops in verify (False = attribute-gate only)
    k_build_refs: int = 5                 # CAP on build refs; effective build = min(cap, n_unique - 1)
```

(Keep the existing `k_build_refs` line — just update its comment to the one above.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/config.py tests/test_config.py
git commit -m "feat(config): add neutralize_name, exemplar_selection, use_part_tree; k_build_refs is now a cap"
```

---

### Task 2: Adaptive dataset split (removes the ≥6-image gate)

**Files:**
- Modify: `graft/dataset.py` (`split_refs`)
- Modify: `graft/run_eval.py` (skip guard)
- Test: `tests/test_dataset.py`

**Interfaces:**
- Produces: `split_refs(sp, k_build, seed, n_heldout_min=1) -> (build, heldout)`; effective build = `min(k_build, max(0, n - n_heldout_min))`.
- Consumes (run_eval): a species is skipped when `not build_refs or not heldout_refs`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py  (append)
from graft.dataset import Species, split_refs

def test_adaptive_split_includes_small_species():
    two = Species("two", ["a.jpg", "b.jpg"])
    b, h = split_refs(two, k_build=5, seed=0)
    assert len(b) == 1 and len(h) == 1            # build n-1, hold out 1

    one = Species("one", ["only.jpg"])
    b1, h1 = split_refs(one, k_build=5, seed=0)
    assert b1 == [] and len(h1) == 1              # can't build+hold-out => empty build (eval skips)

def test_adaptive_split_caps_large_species():
    big = Species("big", [f"{i}.jpg" for i in range(15)])
    b, h = split_refs(big, k_build=5, seed=0)
    assert len(b) == 5 and len(h) == 10           # unchanged vs Phase 1 for n>=6
```

(The existing `test_split_is_disjoint_and_deterministic` on the 3-image fixture still asserts `len(b1)==2 and len(h1)==1` — with `k_build=2`, `min(2, 3-1)=2`, so it stays green.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset.py -q`
Expected: FAIL on the two new tests (current split takes `shuffled[:k_build]` fixed).

- [ ] **Step 3: Write minimal implementation**

Replace `split_refs` body in `graft/dataset.py`:

```python
def split_refs(sp, k_build, seed, n_heldout_min=1):
    """Adaptive split into disjoint (build_refs, heldout_refs). `k_build` is a
    CAP: effective build = min(k_build, n_unique - n_heldout_min), always
    reserving n_heldout_min held-out images. A 1-image species yields an empty
    build (the eval driver then skips it)."""
    rng = random.Random(seed)
    shuffled = list(sp.images)
    rng.shuffle(shuffled)
    n = len(shuffled)
    build_n = min(k_build, max(0, n - n_heldout_min))
    return shuffled[:build_n], shuffled[build_n:]
```

In `graft/run_eval.py`, change the guard (currently `if not heldout_refs:`):

```python
        build_refs, heldout_refs = split_refs(sp, k_build=cfg.k_build_refs, seed=0)
        if not build_refs or not heldout_refs:
            print(f"[eval] {sp.name}: cannot form build+heldout split (n_unique={len(sp.images)}), skipping")
            continue
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dataset.py -q`
Expected: PASS (all, including the unchanged deterministic test).

- [ ] **Step 5: Commit**

```bash
git add graft/dataset.py graft/run_eval.py tests/test_dataset.py
git commit -m "feat(dataset): adaptive build/heldout split, floor drops from 6 to 2 unique images"
```

---

### Task 3: Medoid selection helper (pure)

**Files:**
- Create: `graft/selection.py`
- Test: `tests/test_selection_medoid.py`

**Interfaces:**
- Produces: `medoid_index(embeddings) -> int` — index of the row closest to the mean (rows assumed L2-normalized, so closeness = max dot with the mean).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_selection_medoid.py
import numpy as np
from graft.selection import medoid_index

def test_medoid_picks_central_row():
    # rows 0,1 cluster together; row 2 is an outlier -> medoid is 0 or 1, not 2
    embs = np.array([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]])
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    assert medoid_index(embs) in (0, 1)
    assert medoid_index(embs) != 2

def test_medoid_single_row():
    assert medoid_index(np.array([[0.1, 0.2, 0.3]])) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_selection_medoid.py -q`
Expected: FAIL (module `graft.selection` does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/selection.py
"""Name-free exemplar selection: pick the most representative reference photo
as the medoid of that concept's reference embeddings. Pure numpy so both
generate (GRAFT) and baselines (B1) select the SAME exemplar for parity."""
from __future__ import annotations

import numpy as np


def medoid_index(embeddings) -> int:
    """Index of the reference whose embedding is closest to the mean of all
    references. Rows are assumed L2-normalized (as every graft Embedder
    guarantees), so closeness reduces to the largest dot product with the mean."""
    embs = np.asarray(embeddings, dtype=float)
    mean = embs.mean(axis=0)
    return int(np.argmax(embs @ mean))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_selection_medoid.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/selection.py tests/test_selection_medoid.py
git commit -m "feat(selection): name-free medoid exemplar selection helper"
```

---

### Task 4: Schema — store per-reference embeddings

**Files:**
- Modify: `graft/schema.py` (`ConceptKG`)
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces: `ConceptKG.ref_embeddings: List[List[float]]` (default `[]`), aligned index-for-index with `ref_paths`; SigLIP2 per-reference vectors. `from_json` tolerates its absence (Phase-1 `kg.json`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schema.py  (append)
from graft.schema import ConceptKG

def test_ref_embeddings_roundtrip_and_backcompat(tmp_path):
    kg = ConceptKG("x", [], [], {}, "", [], ["a.jpg", "b.jpg"], ref_embeddings=[[0.1, 0.2], [0.3, 0.4]])
    p = tmp_path / "kg.json"
    kg.to_json(str(p))
    back = ConceptKG.from_json(str(p))
    assert back.ref_embeddings == [[0.1, 0.2], [0.3, 0.4]]

    # Phase-1 kg.json without the field still deserializes to [].
    import json
    d = json.loads(p.read_text()); d.pop("ref_embeddings")
    (tmp_path / "old.json").write_text(json.dumps(d))
    assert ConceptKG.from_json(str(tmp_path / "old.json")).ref_embeddings == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_schema.py -q`
Expected: FAIL (`ref_embeddings` not a field).

- [ ] **Step 3: Write minimal implementation**

In `graft/schema.py`, add the field to `ConceptKG` (after `ref_paths`):

```python
    ref_paths: List[str]
    ref_embeddings: List[List[float]] = field(default_factory=list)  # SigLIP2, aligned to ref_paths
```

And in `from_json`, add `ref_embeddings` to the constructor call:

```python
        return cls(
            concept=raw["concept"],
            global_attrs=global_attrs,
            parts=parts,
            concept_embeddings=raw.get("concept_embeddings", {}),
            anchor=raw["anchor"],
            delta=raw["delta"],
            ref_paths=raw["ref_paths"],
            ref_embeddings=raw.get("ref_embeddings", []),
        )
```

(`asdict` in `to_json` already serializes the new field. `field` is already imported.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_schema.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/schema.py tests/test_schema.py
git commit -m "feat(schema): store per-reference SigLIP2 embeddings for medoid selection"
```

---

### Task 5: Prompt neutralization

**Files:**
- Modify: `graft/prompt.py`
- Test: `tests/test_prompt.py`

**Interfaces:**
- Produces: `NEUTRAL_TOKEN = "plant"`; `concept_token(concept, neutralize) -> str`; `compose_prompt(kg, max_attrs=8, neutralize=True) -> str`; `compose_ravel_prompt(concept, llm_attrs, neutralize=True) -> str`.

- [ ] **Step 1: Write the failing test**

Update the two existing tests to pass `neutralize=False` where they assert the real concept, and add neutralization tests:

```python
# tests/test_prompt.py  (edit the two existing asserts + append)
from graft.prompt import NEUTRAL_TOKEN, compose_prompt, compose_ravel_prompt, concept_token

# in test_prompt_mentions_concept_and_vision_attrs_first: call compose_prompt(kg, neutralize=False)
# in test_ravel_prompt_uses_given_attrs: call compose_ravel_prompt("durian", [...], neutralize=False)

def test_neutralize_replaces_head_but_keeps_attrs():
    from graft.schema import AttributeNode, ConceptKG, PartNode
    kg = ConceptKG("monkey puzzle tree",
                   [AttributeNode("silhouette", "symmetric candelabra", "vision")],
                   [], {}, "", [], ["r.jpg"])
    p = compose_prompt(kg, neutralize=True)
    assert p.lower().startswith(f"a photo of a {NEUTRAL_TOKEN}")
    assert "monkey puzzle tree" not in p.lower()
    assert "symmetric candelabra" in p            # ref-derived attrs stay

def test_concept_token():
    assert concept_token("Avocado", True) == "plant"
    assert concept_token("Avocado", False) == "Avocado"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prompt.py -q`
Expected: FAIL (`neutralize` kwarg / `NEUTRAL_TOKEN` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/prompt.py
_TEMPLATE = "a photo of a {concept}, {attrs}"
NEUTRAL_TOKEN = "plant"

def concept_token(concept: str, neutralize: bool) -> str:
    return NEUTRAL_TOKEN if neutralize else concept

def compose_prompt(kg, max_attrs: int = 8, neutralize: bool = True) -> str:
    attrs = kg.attribute_texts()[:max_attrs]
    return _TEMPLATE.format(concept=concept_token(kg.concept, neutralize), attrs=", ".join(attrs))

def compose_ravel_prompt(concept, llm_attrs, neutralize: bool = True) -> str:
    return _TEMPLATE.format(concept=concept_token(concept, neutralize), attrs=", ".join(llm_attrs))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_prompt.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/prompt.py tests/test_prompt.py
git commit -m "feat(prompt): neutralize concept name to 'plant' (kept behind neutralize flag)"
```

---

### Task 6: Wire medoid + neutralization into generate

**Files:**
- Modify: `graft/generate.py`
- Test: `tests/test_selection.py`

**Interfaces:**
- Consumes: `ConceptKG.ref_embeddings`/`ref_paths` (Task 4), `medoid_index` (Task 3), `compose_prompt(..., neutralize=)` (Task 5), `cfg.neutralize_name`.
- Produces: `select_exemplar(kg) -> str` (medoid path — signature changes, no reranker/`candidate_paths`); `generate_lever_a(kg, models, cfg, seed)` unchanged signature.

- [ ] **Step 1: Write the failing test**

Rewrite `tests/test_selection.py` for the new signature:

```python
# tests/test_selection.py
from graft.generate import select_exemplar
from graft.schema import ConceptKG

def test_select_exemplar_returns_medoid_path():
    # refs a,b cluster; c is an outlier -> medoid is a or b
    kg = ConceptKG("x", [], [], {}, "", [], ["a.jpg", "b.jpg", "c.jpg"],
                   ref_embeddings=[[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]])
    assert select_exemplar(kg) in ("a.jpg", "b.jpg")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_selection.py -q`
Expected: FAIL (old `select_exemplar` requires `candidate_paths, reranker`).

- [ ] **Step 3: Write minimal implementation**

Replace the top of `graft/generate.py`:

```python
from graft.prompt import compose_prompt
from graft.schema import ConceptKG
from graft.selection import medoid_index

_DEFAULT_NEGATIVE = "monochrome, lowres, bad anatomy, worst quality, low quality"


def select_exemplar(kg: ConceptKG) -> str:
    """Name-free medoid of the concept's stored reference embeddings."""
    return kg.ref_paths[medoid_index(kg.ref_embeddings)]


def generate_lever_a(kg, models, cfg, seed):
    prompt = compose_prompt(kg, neutralize=cfg.neutralize_name)
    ip_image = Image.open(select_exemplar(kg)).convert("RGB")
    image = models.generator.generate(
        prompt=prompt, ip_image=ip_image, seed=seed, negative=_DEFAULT_NEGATIVE, steps=50
    )
    return image, prompt
```

(Delete the reranker load/unload and `_open` injection — no longer used. Keep `from PIL import Image`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_selection.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/generate.py tests/test_selection.py
git commit -m "feat(generate): medoid exemplar + neutralized prompt; drop reranker from M2"
```

---

### Task 7: `is_generic` attribute filter (pure)

**Files:**
- Create: `graft/attrfilter.py`
- Test: `tests/test_attrfilter.py`

**Interfaces:**
- Produces: `is_generic(value: str) -> bool`; `count_generic(values: Sequence[str]) -> int`. Used by kg_build (Task 10) to decide a one-time re-ask.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_attrfilter.py
from graft.attrfilter import is_generic, count_generic

def test_flags_generic_values():
    for v in ["green", "full", "tree", "not visible", "normal", ""]:
        assert is_generic(v) is True

def test_keeps_discriminative_values():
    for v in ["symmetric candelabra silhouette", "stiff triangular scales", "reddish flaking bark"]:
        assert is_generic(v) is False

def test_count_generic():
    assert count_generic(["green", "stiff triangular scales", "full"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_attrfilter.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/attrfilter.py
"""Flag low-information ("generic") attribute values so M1 can re-ask for
discriminative ones. Pure and unit-tested; see spec section 7 (folds Phase 2
design section 5.1)."""
from __future__ import annotations

from typing import Sequence

_GENERIC = {
    "", "green", "full", "tree", "plant", "leaf", "leaves", "normal", "typical",
    "standard", "not visible", "none", "n/a", "na", "unknown", "brown", "grey", "gray",
}


def is_generic(value: str) -> bool:
    v = " ".join(value.strip().lower().split())
    if v in _GENERIC:
        return True
    return len(v.split()) < 2  # a single bare word is not discriminative enough


def count_generic(values: Sequence[str]) -> int:
    return sum(1 for v in values if is_generic(v))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_attrfilter.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/attrfilter.py tests/test_attrfilter.py
git commit -m "feat(attrfilter): is_generic/count_generic for M1 discriminative re-ask"
```

---

### Task 8: Verify — part-tree toggle + crop-consistency fix

**Files:**
- Modify: `graft/verify.py`
- Test: `tests/test_verify_unit.py`

**Interfaces:**
- Consumes: `cfg.use_part_tree`; a `PartNode` with **empty** `embeddings` means "no reliable reference crop — do not score this part".
- Produces: `verify(image, kg, models, cfg)` — scores only parts that have stored (box-derived) embeddings; a part with a reference embedding but no box in the generated image scores `sim=0.0, present=False` (no whole-image fallback); with `use_part_tree=False` no part is scored and the gate is attribute-only.

- [ ] **Step 1: Write the failing test**

Add fakes + two behavior tests (keep the existing `aggregate` tests):

```python
# tests/test_verify_unit.py  (append)
import numpy as np
from graft.verify import verify
from graft.schema import AttributeNode, ConceptKG, PartNode

class _FakeVLM:
    def ask(self, image, question): return "yes"
class _FakeDetector:
    def __init__(self, found): self.found = found
    def detect(self, image, phrase): return [(0, 0, 4, 4)] if self.found else []
class _FakeEmb:
    def embed_image(self, images): return np.ones((len(images), 3)) / np.sqrt(3)
class _FakeModels:
    def __init__(self, found):
        self.vlm, self.detector = _FakeVLM(), _FakeDetector(found)
        self.siglip = self.dino = _FakeEmb()
    def unload(self, name): pass

class _Cfg:
    part_sim_threshold = 0.5; attr_pass_threshold = 0.6
    def __init__(self, use_part_tree): self.use_part_tree = use_part_tree

def _kg_with_one_scored_part():
    part = PartNode("leaf", [AttributeNode("shape", "stiff scales", "vision")],
                    "crop.png", {"siglip2": [1/np.sqrt(3)]*3, "dino": [1/np.sqrt(3)]*3})
    unscored = PartNode("bark", [], None, {})  # no box at build -> empty embeddings
    return ConceptKG("x", [], [part, unscored], {}, "", [], ["r.jpg"])

def test_use_part_tree_false_skips_part_scoring():
    r = verify(object(), _kg_with_one_scored_part(), _FakeModels(found=True), _Cfg(False))
    assert r.part_scores == [] and r.failing_parts == []

def test_scores_only_box_derived_parts_and_no_whole_image_fallback():
    r = verify(object(), _kg_with_one_scored_part(), _FakeModels(found=False), _Cfg(True))
    names = [p.part for p in r.part_scores]
    assert names == ["leaf"]                         # 'bark' (empty embeddings) never scored
    leaf = r.part_scores[0]
    assert leaf.present is False and leaf.sim == 0.0  # no box in gen -> 0, not a whole-image sim
```

(`verify` calls `image.crop(...)` only when a box is found, so passing `object()` as the image is safe in these two paths.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_verify_unit.py -q`
Expected: FAIL (current verify scores all parts and falls back to whole-image).

- [ ] **Step 3: Write minimal implementation**

Replace the part-scoring block in `graft/verify.py`'s `verify`:

```python
    part_scores: List[PartScore] = []
    if cfg.use_part_tree:
        scored = [p for p in kg.parts if p.embeddings]  # only parts with a real reference crop
        for part in scored:
            boxes = models.detector.detect(image, _part_phrase(part.name))
            if not boxes:
                part_scores.append(PartScore(part=part.name, sim=0.0, present=False))
                continue
            x0, y0, x1, y1 = boxes[0]
            crop = image.crop((x0, y0, x1, y1))
            sims = []
            if "siglip2" in part.embeddings:
                sims.append(_cosine(models.siglip.embed_image([crop])[0], np.array(part.embeddings["siglip2"])))
            if "dino" in part.embeddings:
                sims.append(_cosine(models.dino.embed_image([crop])[0], np.array(part.embeddings["dino"])))
            part_scores.append(PartScore(part=part.name, sim=float(np.mean(sims)) if sims else 0.0, present=True))
        if scored:
            models.unload("detector"); models.unload("siglip"); models.unload("dino")
```

(The attribute-checklist block below it is unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_verify_unit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/verify.py tests/test_verify_unit.py
git commit -m "fix(verify): score only box-derived parts (no whole-image fallback); add use_part_tree toggle"
```

---

### Task 9: Baselines — neutralize + B1 medoid parity

**Files:**
- Modify: `graft/baselines.py`
- Test: `tests/test_baselines_unit.py`

**Interfaces:**
- Consumes: `concept_token` (Task 5), `medoid_index` (Task 3), `compose_ravel_prompt(..., neutralize=)`, `cfg.neutralize_name`.
- Produces: `b0_vanilla`, `b1_imagerag`, `b2_ravel` with neutralized heads; B1 selects its reference by **SigLIP2 medoid** over the pool (same rule GRAFT uses → same exemplar) and drops the "According to this image" scaffold.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baselines_unit.py
import numpy as np
from PIL import Image
from graft.baselines import b0_vanilla, b1_imagerag

class _CaptureGen:
    def __init__(self): self.prompt = self.ip_image = None
    def generate(self, prompt, ip_image, seed, negative, steps):
        self.prompt, self.ip_image = prompt, ip_image
        return Image.new("RGB", (8, 8))

class _Siglip:
    # first two pool images cluster, third is an outlier -> medoid index 0 or 1
    def embed_image(self, images):
        m = np.array([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]])[: len(images)]
        return m / np.linalg.norm(m, axis=1, keepdims=True)

class _Models:
    def __init__(self): self.generator, self.siglip = _CaptureGen(), _Siglip()
    def unload(self, name): pass

class _Cfg:
    def __init__(self, neutralize): self.neutralize_name = neutralize

def test_b0_neutralized_head():
    m = _Models(); b0_vanilla("Avocado", m, _Cfg(True), seed=0)
    assert m.generator.prompt == "a photo of a plant"

def test_b1_uses_medoid_and_neutral_scaffold(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.png"; Image.new("RGB", (8, 8)).save(p); paths.append(str(p))
    m = _Models(); b1_imagerag("Avocado", paths, m, _Cfg(True), seed=0)
    assert m.generator.prompt == "a photo of a plant"          # no "According to this image"
    assert "according to this image" not in m.generator.prompt.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_baselines_unit.py -q`
Expected: FAIL (current b0 embeds the name; b1 uses text retrieval + "According to this image").

- [ ] **Step 3: Write minimal implementation**

In `graft/baselines.py`: import helpers, rewrite the three functions, delete `_top1_by_siglip_text` and `RAVEL_ATTR_INSTRUCTION`'s prompt coupling stays but the compose call gains `neutralize`:

```python
from graft.prompt import compose_ravel_prompt, concept_token
from graft.selection import medoid_index

def b0_vanilla(concept, models, cfg, seed):
    prompt = f"a photo of a {concept_token(concept, cfg.neutralize_name)}"
    return models.generator.generate(prompt=prompt, ip_image=None, seed=seed,
                                     negative=_DEFAULT_NEGATIVE, steps=50)

def b1_imagerag(concept, pool_paths, models, cfg, seed):
    """Flat single-reference conditioning: the SigLIP2 medoid of the pool (the
    SAME exemplar GRAFT selects -> exemplar parity), IP-Adapter, no graph."""
    pool_images = [Image.open(p).convert("RGB") for p in pool_paths]
    ip_image = pool_images[medoid_index(models.siglip.embed_image(pool_images))]
    prompt = f"a photo of a {concept_token(concept, cfg.neutralize_name)}"
    return models.generator.generate(prompt=prompt, ip_image=ip_image, seed=seed,
                                     negative=_DEFAULT_NEGATIVE, steps=50)

def b2_ravel(concept, models, cfg, seed):
    blank = Image.new("RGB", (16, 16), (128, 128, 128))
    raw = models.vlm.describe([blank], RAVEL_ATTR_INSTRUCTION.format(concept=concept))
    attrs = _parse_attr_list(raw)
    prompt = compose_ravel_prompt(concept, attrs, neutralize=cfg.neutralize_name)
    return models.generator.generate(prompt=prompt, ip_image=None, seed=seed,
                                     negative=_DEFAULT_NEGATIVE, steps=50)
```

(Keep `_parse_attr_list` and `RAVEL_ATTR_INSTRUCTION` — B2 still recalls attributes *by name*; only the prompt head is neutralized. Remove `numpy` import only if now unused.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_baselines_unit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/baselines.py tests/test_baselines_unit.py
git commit -m "feat(baselines): neutralize heads; B1 uses SigLIP2 medoid for exemplar parity with GRAFT"
```

---

### Task 10: M1 build overhaul (neutralize, drop anchor, medoid crop, ref embeddings, box-only parts, re-ask)

**Files:**
- Modify: `graft/kg_build.py`
- Test: `tests/test_kg_build_unit.py` (pure parts), `tests/test_kg_build_gpu.py` (integration, `@pytest.mark.gpu`)

**Interfaces:**
- Consumes: `count_generic` (Task 7), `medoid_index` (Task 3), `cfg.neutralize_name`.
- Produces: a `ConceptKG` where — attributes are read **without** the species name when `cfg.neutralize_name`; `anchor=""`, `delta=[]`; part crops come from the **medoid** reference; `ref_embeddings` holds per-ref SigLIP2 vectors aligned to `ref_paths`; a part whose box is **not** found stores **empty** embeddings (unscoreable); if too many attribute values are generic, M1 re-asks once with a sharper instruction.

- [ ] **Step 1: Write the failing test (pure re-ask decision)**

Extract the re-ask decision as a pure function and test it:

```python
# tests/test_kg_build_unit.py  (append)
from graft.kg_build import should_reask

def test_should_reask_when_mostly_generic():
    assert should_reask(["green", "full", "tree", "stiff scales"], threshold=0.5) is True   # 3/4 generic
    assert should_reask(["stiff triangular scales", "reddish flaking bark"], threshold=0.5) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_kg_build_unit.py -q`
Expected: FAIL (`should_reask` missing).

- [ ] **Step 3: Write minimal implementation**

Add to `graft/kg_build.py`:

```python
from graft.attrfilter import count_generic
from graft.selection import medoid_index

SCHEMA_INSTRUCTION_NEUTRAL = (
    "You are looking at reference photographs of a single plant species. "
    "Describe ONLY what is visibly present in these images. Be DISCRIMINATIVE: "
    "give the specific color, shape, texture, and count details that distinguish "
    "this plant from a generic tree. Reply with ONLY a JSON object of this exact "
    "shape (use \"not visible\" if a part is not shown):\n" + _SCHEMA_JSON_SHAPE
)
# (_SCHEMA_JSON_SHAPE = the JSON-shape tail currently embedded in SCHEMA_INSTRUCTION;
#  factor it out so both the named and neutral instructions share it.)

REASK_SUFFIX = (
    " Your previous answer was too generic. Replace any vague value (e.g. 'green', "
    "'full', 'tree') with a concrete, distinguishing observation from the images."
)

def should_reask(values, threshold: float = 0.5) -> bool:
    return bool(values) and (count_generic(values) / len(values)) >= threshold

def _schema_instruction(concept: str, neutralize: bool) -> str:
    return SCHEMA_INSTRUCTION_NEUTRAL if neutralize else SCHEMA_INSTRUCTION.format(concept=concept)
```

Then in `build_kg`, apply the six changes (real model calls):

```python
    raw = models.vlm.describe(ref_images, _schema_instruction(concept, cfg.neutralize_name))
    global_attrs, part_attrs_by_name = parse_vlm_schema(raw)
    values = [a.value for a in global_attrs] + [a.value for p in part_attrs_by_name.values() for a in p]
    if should_reask(values):
        raw = models.vlm.describe(ref_images, _schema_instruction(concept, cfg.neutralize_name) + REASK_SUFFIX)
        global_attrs, part_attrs_by_name = parse_vlm_schema(raw)
    # anchor/delta dropped (spec section 5): no ANCHOR_INSTRUCTION call.
    anchor, delta = "", []
    models.unload("vlm")

    # per-ref SigLIP2 embeddings (also the concept mean) -> medoid reference for crops.
    ref_siglip = models.siglip.embed_image(ref_images)          # (N, D), L2-normed
    ref_embeddings = [v.tolist() for v in ref_siglip]
    medoid = medoid_index(ref_siglip)
    crop_source = ref_images[medoid]

    parts = []
    for part_name in PART_NAMES:
        attrs = part_attrs_by_name.get(part_name, [])
        boxes = models.detector.detect(crop_source, _part_phrase(part_name))
        if boxes:
            x0, y0, x1, y1 = boxes[0]
            crop = crop_source.crop((x0, y0, x1, y1))
            crop_path = os.path.join(crops_dir, f"{part_name}.png"); crop.save(crop_path)
            embeddings = {"siglip2": models.siglip.embed_image([crop])[0].tolist(),
                          "dino": models.dino.embed_image([crop])[0].tolist()}
            parts.append(PartNode(part_name, attrs, crop_path, embeddings))
        else:
            parts.append(PartNode(part_name, attrs, None, {}))   # unscoreable: empty embeddings

    concept_embeddings = {"siglip2": _renorm(ref_siglip.mean(axis=0)),
                          "dino": _renorm(models.dino.embed_image(ref_images).mean(axis=0))}
    models.unload("detector"); models.unload("siglip"); models.unload("dino")
    # reranker no longer used (medoid replaces it) — do not load it.

    kg = ConceptKG(concept=concept, global_attrs=global_attrs, parts=parts,
                   concept_embeddings=concept_embeddings, anchor=anchor, delta=delta,
                   ref_paths=list(build_refs), ref_embeddings=ref_embeddings)
    kg.to_json(os.path.join(outputs_dir, "kg.json"))
    return kg
```

Delete `ANCHOR_INSTRUCTION`, `parse_anchor_reply`, and the reranker block. Move the JSON-shape tail of the old `SCHEMA_INSTRUCTION` into `_SCHEMA_JSON_SHAPE` and keep a named `SCHEMA_INSTRUCTION` (for `neutralize_name=False` builds).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_kg_build_unit.py -q`
Expected: PASS.
Then update `tests/test_kg_build_gpu.py` to assert: `kg.anchor == ""`, `len(kg.ref_embeddings) == len(kg.ref_paths)`, and every `PartNode` either has both siglip2+dino embeddings or empty `{}`. Run manually on the card: `python -m pytest tests/test_kg_build_gpu.py -q -m gpu`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/kg_build.py tests/test_kg_build_unit.py tests/test_kg_build_gpu.py
git commit -m "feat(kg_build): name-blind M1, drop anchor, medoid crops, per-ref embeddings, box-only parts, generic re-ask"
```

---

### Task 11: Pruned-GRAFT arm (`ours_notree`)

**Files:**
- Modify: `graft/worker_baseline.py`
- Test: `tests/test_worker_baseline_unit.py`

**Interfaces:**
- Produces: a pure `cfg_for_method(cfg, method)` that returns `cfg` unchanged for most methods and `replace(cfg, use_part_tree=False)` for `"ours_notree"`; the worker treats both `"ours"` and `"ours_notree"` as the GRAFT refine path.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_worker_baseline_unit.py
from graft.config import GraftConfig
from graft.worker_baseline import cfg_for_method

def test_cfg_for_method_prunes_only_notree():
    base = GraftConfig()
    assert cfg_for_method(base, "ours").use_part_tree is True
    assert cfg_for_method(base, "ours_notree").use_part_tree is False
    assert cfg_for_method(base, "b1").use_part_tree is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_worker_baseline_unit.py -q`
Expected: FAIL (`cfg_for_method` missing).

- [ ] **Step 3: Write minimal implementation**

In `graft/worker_baseline.py`, add and use the helper:

```python
from dataclasses import replace

def cfg_for_method(cfg, method):
    return replace(cfg, use_part_tree=False) if method == "ours_notree" else cfg
```

In `main`, change the branch:

```python
    cfg = cfg_for_method(GraftConfig.from_yaml(cfg_yaml), method)
    ...
    if method in ("ours", "ours_notree"):
        from graft.refine import refine
        from graft.schema import ConceptKG
        kg = ConceptKG.from_json(kg_json_or_dash)
        image, _report, _prompt = refine(kg, None, cfg)
    else:
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_worker_baseline_unit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/worker_baseline.py tests/test_worker_baseline_unit.py
git commit -m "feat(eval): ours_notree = GRAFT with part-tree pruned for the ablation"
```

---

### Task 12: Refine tolerates a worker crash (Phase 2 §6.1)

**Files:**
- Modify: `graft/refine.py`
- Test: `tests/test_refine_unit.py`

**Interfaces:**
- Produces: `classify_returncode(code) -> str` returning `"ok"` (0), `"error"` (2 = usage/config bug → still raises), or `"skip"` (3 = OOM, or any other nonzero = crash/segfault → try the next seed). `_run_attempt` uses it so a worker crash no longer fails the whole case.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_refine_unit.py  (append)
from graft.refine import classify_returncode

def test_classify_returncode():
    assert classify_returncode(0) == "ok"
    assert classify_returncode(2) == "error"     # real usage/config bug -> raise
    assert classify_returncode(3) == "skip"      # OOM -> next seed
    assert classify_returncode(-11) == "skip"    # SIGSEGV crash -> next seed (was: fail case)
    assert classify_returncode(1) == "skip"      # generic crash -> next seed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_refine_unit.py -q`
Expected: FAIL (`classify_returncode` missing).

- [ ] **Step 3: Write minimal implementation**

In `graft/refine.py`:

```python
def classify_returncode(code: int) -> str:
    if code == 0:
        return "ok"
    if code == 2:
        return "error"   # usage/config bug — a real problem, surface it
    return "skip"        # 3 (OOM) or any crash/segfault — no signal, try the next seed
```

Rewrite the tail of `_run_attempt`:

```python
    kind = classify_returncode(proc.returncode)
    if kind == "error":
        return None, f"worker_generate_verify usage/config error (seed={seed}):\n{proc.stderr}"
    if kind == "skip":
        sys.stderr.write(f"[refine] seed {seed} produced no signal (code={proc.returncode}); trying next\n")
        return None, None
    # kind == "ok": load image + report as before
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_refine_unit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graft/refine.py tests/test_refine_unit.py
git commit -m "fix(refine): a worker crash retries the next seed like OOM, not fail the whole case"
```

---

### Task 13: MMKG audit script

**Files:**
- Create: `graft/audit.py`
- Test: `tests/test_audit_unit.py`

**Interfaces:**
- Produces: pure `summarize_audit(records) -> dict` where each record is `{"part": str, "ref_box": bool, "gen_box": bool, "sim": float|None}`; returns per-part hit-rates and sim summary. A `main(argv)` runs GroundingDINO over a subset's build refs + one generated image per species and writes the records (GPU; smoke-run manually).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audit_unit.py
from graft.audit import summarize_audit

def test_summarize_audit_hit_rates():
    recs = [
        {"part": "leaf", "ref_box": True,  "gen_box": True,  "sim": 0.7},
        {"part": "leaf", "ref_box": True,  "gen_box": False, "sim": None},
        {"part": "bark", "ref_box": False, "gen_box": False, "sim": None},
    ]
    s = summarize_audit(recs)
    assert s["leaf"]["ref_hit_rate"] == 1.0
    assert s["leaf"]["gen_hit_rate"] == 0.5
    assert s["leaf"]["mean_sim"] == 0.7           # only scored (sim not None) count
    assert s["bark"]["ref_hit_rate"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_audit_unit.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/audit.py
"""M1/M3 instrumentation (spec section 8.2): GroundingDINO per-part hit-rate
and part-similarity distribution on a validation subset, so the prune decision
(section 8.3) rests on evidence rather than assumption."""
from __future__ import annotations

from typing import List, Optional


def summarize_audit(records: List[dict]) -> dict:
    parts = sorted({r["part"] for r in records})
    out = {}
    for part in parts:
        rs = [r for r in records if r["part"] == part]
        sims = [r["sim"] for r in rs if r.get("sim") is not None]
        out[part] = {
            "n": len(rs),
            "ref_hit_rate": sum(r["ref_box"] for r in rs) / len(rs),
            "gen_hit_rate": sum(r["gen_box"] for r in rs) / len(rs),
            "mean_sim": (sum(sims) / len(sims)) if sims else None,
            "n_scored": len(sims),
        }
    return out
```

Add a `main(argv)` that, given `--root`, the subset species, and their `kg.json` + one generated image each, builds the `records` (real GroundingDINO detect on ref crop_source and the generated image, SigLIP2/DINOv3 sims where both boxes exist) and writes `records` + `summarize_audit(records)` to `outputs/audit/audit.json`. Guard the model imports inside `main` (the pure function must import with no torch).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_audit_unit.py -q`
Expected: PASS. (Model path: run `python -m graft.audit --root <treevill> --species <subset>` manually on the card after Task 14 produces images.)

- [ ] **Step 5: Commit**

```bash
git add graft/audit.py tests/test_audit_unit.py
git commit -m "feat(audit): GroundingDINO hit-rate + part-sim distribution instrumentation"
```

---

### Task 14: Sprint matrix driver + paired analysis

**Files:**
- Create: `graft/analysis.py`
- Modify: `graft/run_eval.py` (subset-by-unique-count, matrix loop, per-species held-out count, write analysis)
- Test: `tests/test_analysis_unit.py`, `tests/test_eval_unit.py`

**Interfaces:**
- Consumes: everything above; `cfg.neutralize_name`, `cfg.ip_scale`, method list including `ours` and `ours_notree`.
- Produces: `analysis.paired_analysis(rows, metric, a="ours", b="b1") -> {"per_species": [...], "mean_delta": float, "win_rate": float, "n": int}`; `run_eval` writes `results.json` (rows tagged with `method, species, name_mode, ip_scale, n_heldout`), `results.md`, and `analysis.json`.

- [ ] **Step 1: Write the failing test (pure analysis)**

```python
# tests/test_analysis_unit.py
from graft.analysis import paired_analysis

def test_paired_analysis_delta_and_winrate():
    rows = [
        {"method": "ours", "species": "A", "dino": 0.30}, {"method": "b1", "species": "A", "dino": 0.20},
        {"method": "ours", "species": "B", "dino": 0.10}, {"method": "b1", "species": "B", "dino": 0.25},
    ]
    r = paired_analysis(rows, "dino")
    assert r["n"] == 2
    assert r["win_rate"] == 0.5                       # A: ours wins; B: ours loses
    assert abs(r["mean_delta"] - (-0.025)) < 1e-9     # ((0.30-0.20)+(0.10-0.25))/2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_analysis_unit.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/analysis.py
"""Paired, per-species GRAFT-vs-baseline comparison (spec section 11) — the
central read now that exemplar selection and the name are no longer confounds."""
from __future__ import annotations

from typing import List


def paired_analysis(rows: List[dict], metric: str, a: str = "ours", b: str = "b1") -> dict:
    by = {}
    for r in rows:
        if metric in r:
            by.setdefault(r["species"], {})[r["method"]] = r[metric]
    per = []
    for sp, m in sorted(by.items()):
        if a in m and b in m:
            per.append({"species": sp, a: m[a], b: m[b], "delta": m[a] - m[b]})
    n = len(per)
    return {
        "per_species": per,
        "n": n,
        "mean_delta": (sum(p["delta"] for p in per) / n) if n else 0.0,
        "win_rate": (sum(1 for p in per if p["delta"] >= 0) / n) if n else 0.0,
    }
```

Then extend `graft/run_eval.py`:
- Add `rank_species_by_unique_count(root, limit=None) -> List[str]`: `load_species` each name (hashes/dedupes — note the NAS cost in the docstring), keep those with `len(images) >= 2`, sort by `len(images)` descending, return names.
- In `main`, add args: `--select-by {alpha,unique}` (default `unique`), `--ip-scales "0.4,0.6,0.8"`, `--name-modes "neutral,named"`, and change `--methods` default to `"ours,ours_notree,b0,b1,b2"`. Build the species subset via `rank_species_by_unique_count(root)[:n_species]` when `--select-by unique`.
- In `evaluate`, wrap the per-method loop in `for name_mode in name_modes: for ip_scale in ip_scales:` and set the case cfg via `dataclasses.replace(cfg, neutralize_name=(name_mode=="neutral"), ip_scale=ip_scale)`. **Build the KG once per species with `neutralize_name=True`** (a fixed build cfg) and reuse it across all cells. Sweep `ip_scale`/`name_mode` only for `{ours, ours_notree, b1}`; run `b0`/`b2` once (they ignore `ip_scale`). Tag each row with `name_mode`, `ip_scale`, and `row["n_heldout"] = len(heldout_refs)`.
- After collecting rows, write `analysis.json` = `{metric: paired_analysis(neutral_rows_at_best_ip, metric) for metric in ("dino","siglip2","clip_i")}`, plus the named−neutral delta per method.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_analysis_unit.py tests/test_eval_unit.py -q`
Expected: PASS. Extend `tests/test_eval_unit.py` to assert `summarize` still groups correctly with the extra tag keys present (they are non-numeric except `ip_scale`/`n_heldout`; ensure `summarize` skips `name_mode`/`species`/`method` and that tag numerics don't pollute metric means — exclude `ip_scale` and `n_heldout` from the averaged metric set).

- [ ] **Step 5: Commit**

```bash
git add graft/analysis.py graft/run_eval.py tests/test_analysis_unit.py tests/test_eval_unit.py
git commit -m "feat(eval): subset-by-unique-count, neutral/named x ip_scale matrix, paired GRAFT-vs-B1 analysis"
```

---

## Post-implementation: run the sprint (manual, on the shared 4090)

Not a coding task — the deliverable run, staged/resumable via the existing `gpu_queue.sh` + per-case subprocess pattern:

0. **Clear the pre-sprint outputs first (REQUIRED):**

```bash
rm -rf outputs/*/kg.json outputs/*/crops outputs/*/eval_images
```

   The four Phase-1 KGs on disk (`outputs/{Akashmoni,Ashok,Ashore,Avocado}`) are stale in
   four separate ways and `run_eval` reuses any `kg.json` it finds: (a) M1 attributes were
   read with the species name visible (name leak), (b) `anchor`/`delta` are set and the sprint
   drops them, (c) part crops came from `ref_images[0]` rather than the medoid, (d) part
   embeddings are whole-image fallbacks, not box-derived. `run_eval` now detects the missing
   `ref_embeddings` and rebuilds those KGs automatically, but (a)–(d) are invisible to that
   check — only deleting them guarantees a clean, name-blind build. The stale `eval_images`
   likewise carry Phase-1 filenames and would be mistaken for sprint output by `--resume`.
1. Pick the subset: `python -m graft.run_eval --root <treevill> --select-by unique --n-species 8 ... ` (dry-run the selection first to confirm the species + their unique counts).
2. Run the matrix (`ours,ours_notree,b0,b1,b2` × `neutral,named` × `ip_scale∈{0.4,0.6,0.8}`).
   Stage it species-by-species with `--species <name>` (skips the multi-minute NAS ranking
   hash) and restart an interrupted stage with `--resume` (re-scores cells whose
   `eval_images/*.png` already exists instead of regenerating them). After each stage check
   `results.json`'s `missing_cells` — a non-empty list means the matrix has holes and the
   paired `n` is smaller than it looks.
3. Run the audit on the same cell the readout uses: `python -m graft.audit --root <treevill> --species <subset> --name-mode neutral --ip-scale 0.6` (set `--ip-scale` to `analysis.json`'s `best_ip`).
4. Read out (from `outputs/eval/analysis.json` + `outputs/audit/audit.json`): prune decision (§8.3 — `ours` vs `ours_notree` paired), best `ip_scale` (§9), named−neutral delta (§5), and the confound-free GRAFT-vs-B1 paired result.
5. Write a short sprint report under `reports/` and set `use_part_tree`'s permanent default from the prune outcome; hand the validated config to Phase 2 §6.

---

## Self-Review

**Spec coverage:**
- §4 adaptive split → Task 2. §5 neutralization (prompt/retrieval/reranker/M1) → Tasks 5,6,9,10; named arm → Task 14. §6 medoid parity → Tasks 3,6,9. §7 reconciliation / is_generic re-ask → Tasks 7,10. §8.1 bugs → Task 10 (medoid crop), Task 8 (crop-consistency). §8.2 audit → Task 13. §8.3 prune ablation → Tasks 8,11,14. §9 ip_scale sweep → Task 14. §10 subset-by-unique-count + refine crash tolerance → Tasks 14,12. §11 metrics/paired/held-out count → Task 14. All covered.
- CLIP-T demotion (§5/§11): no code change needed — `worker_metrics` already computes `clip_t`; it is simply reported as a diagnostic, not dropped. Noted, no task required.
- `configs/pipeline.yaml` / `pipeline_eval_run.yaml`: the new fields have defaults and need not be added to the yamls; if a run wants the named arm or a fixed `ip_scale`, the driver sets them via `replace` (Task 14), so no yaml edit is required.

**Placeholder scan:** no "TBD/handle appropriately"; the one earlier default-TBD (`use_part_tree` permanent value) is explicitly resolved by the §8.3 outcome in the post-implementation run, with `True` as the sprint default.

**Type consistency:** `medoid_index(embeddings)->int` used identically in Tasks 6, 9, 10. `select_exemplar(kg)->str` (Task 6) matches its test (Task 6). `ref_embeddings: List[List[float]]` produced in Task 4/10, consumed in Task 6. `cfg_for_method` (Task 11), `classify_returncode` (Task 12), `paired_analysis`/`summarize_audit`/`is_generic`/`count_generic`/`should_reask` signatures match their tests.
