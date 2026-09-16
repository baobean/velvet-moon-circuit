# MMKG Medoid Fidelity Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-testable, pre-registered harness that runs a within-species matched pilot — precomputed SigLIP-centroid medoid vs a deterministic non-medoid same-species image, as the repair reference — over the 25 iNaturalist bird species, and report a frozen DINO verdict.

**Architecture:** All statistics, selection rules, and guards are pure CPU functions in one module (`ragregen/mmkg/medoid_fidelity_pilot.py`), unit-tested and committed *before* any repair is generated (that commit IS the pre-registration). A thin GPU orchestrator wires the existing draft/mask/inpaint/encode calls around those pure units via dependency injection, so the full pipeline is proven on CPU with fakes and only the final task touches a GPU.

**Tech Stack:** Python, PIL, numpy, pytest (stdlib `math` for the sign test; no scipy). Existing modules: `ragregen.mmkg_store.{store,reference_source}`, `ragregen.metrics` (`dino_identity`, `DinoEncoder`, `crop_to_mask`), `ragregen.{draft,mask,regen}`, and the committed `scripts/run_pipeline._reference_mmkg_one` contract.

**Spec:** `docs/superpowers/specs/2026-09-11-mmkg-medoid-fidelity-pilot-design.md` (LOCKED)

## Global Constraints

- Substrate = the **25 iNaturalist pilot bird species** only. TreeVill is **excluded** (no held-out split). (spec §3)
- Statistical unit = **species**; **n = 25**; **1 draft per species**; 25 matched pairs. (spec §3, §8)
- Arm A reference = `record["medoid"]["image_path"]` (precomputed). Arm B = the frozen §4 rule. (spec §3, §4)
- Ground truth = the species' **3 held-out eval images** (`source_split: build_7_eval_3`). (spec §5)
- Aggregation: `dino_identity` means cosine over the 3 GT → one per-species score; `delta_s = score_A_s − score_B_s`; the 25 `delta_s` are the sample. (spec §6)
- **Primary metric = DINO** (decision). **Secondary = SigLIP** — descriptive, selection-favorable, **never** in the decision. (spec §6)
- **MARGIN = 0.02** DINO cosine. `MARGIN_SOURCE = ragregen/metrics.py:195` (`metrics.MARGIN = -0.02`, pre-data non-inferiority margin). Frozen before any repair is generated/scored; never tuned after seeing deltas. (spec §7)
- Decision (pre-registered): **GAIN** = mean ≥ +0.02 and 95% CI low > 0; **LOSS** = mean ≤ −0.02 and 95% CI high < 0; else **NO_LARGE_EFFECT** → close the line. (spec §7)
- **No-rescue rule:** never add seeds/drafts because a result looks flat. No signal → close; signal → a *separate* follow-up. (spec §8)
- **No repair-time NN, either arm:** references resolve to fixed paths before inpaint. No FAISS / `nearest_crops` / `Retriever`. (spec §9, §11)
- ❌ attributes, ❌ graph/relations, ❌ reranking, ❌ multi-reference, ❌ VLM judge. `ragregen/regen.py` is read-only. (spec §9)
- **No fallback:** missing `global_id`, or any build/eval overlap, **aborts** — never silently substituted. (spec §5, §9)
- **Claim boundary (spec §12):** a GAIN may be worded only as "precomputed SigLIP-centroid medoids improve repair fidelity in this controlled setup (25 iNat birds, DINO)"; never "MMKG improves generation"; never generalized beyond birds.
- **GPU appears only in the final task (Task 8)**, and only after the pre-registration commit. Tasks 1–7 are CPU-only.
- Code lives in `ragregen/mmkg/` (following the `inat_pilot.py` precedent); tests in `tests/mmkg/`. Branch: `mmkg-reference-integration` (do not merge to the parked branch).

---

### Task 1: Frozen baseline-selection rule

Deterministic non-medoid same-species build image. This is spec §4 as code — no eyeballing, no randomness, no NN.

**Files:**
- Create: `ragregen/mmkg/medoid_fidelity_pilot.py`
- Test: `tests/mmkg/test_medoid_fidelity_pilot.py`
- Read-only ref: `tests/mmkg_store/test_build_inat.py:132-133` (candidate item = `{"image_path": str, "embedding_ref": int}`; `candidates` covers the full eligible build pool, medoid included).

**Interfaces:**
- Consumes: a store record dict (`record["medoid"]["image_path"]`, `record["candidates"]` = list of `{"image_path": ...}`).
- Produces: `baseline_reference(record: dict) -> str` — the chosen image path.

- [ ] **Step 1: Write the failing test**

```python
# tests/mmkg/test_medoid_fidelity_pilot.py
import pytest
from ragregen.mmkg import medoid_fidelity_pilot as p


def _rec(medoid_path, candidate_paths, gid="inat:00001"):
    return {
        "global_id": gid,
        "medoid": {"image_path": medoid_path},
        "candidates": [{"image_path": c, "embedding_ref": i}
                       for i, c in enumerate(candidate_paths)],
    }


def test_baseline_is_lexicographic_first_non_medoid():
    # candidates cover the full build pool INCLUDING the medoid image
    rec = _rec("sp/c.jpg", ["sp/c.jpg", "sp/b.jpg", "sp/a.jpg", "sp/d.jpg"])
    assert p.baseline_reference(rec) == "sp/a.jpg"


def test_baseline_excludes_the_medoid_even_if_lexicographically_first():
    rec = _rec("sp/a.jpg", ["sp/a.jpg", "sp/b.jpg", "sp/c.jpg"])
    assert p.baseline_reference(rec) == "sp/b.jpg"


def test_baseline_raises_when_only_the_medoid_exists():
    rec = _rec("sp/a.jpg", ["sp/a.jpg"])
    with pytest.raises(ValueError, match="baseline"):
        p.baseline_reference(rec)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mmkg/test_medoid_fidelity_pilot.py -k baseline -v`
Expected: FAIL (module / function does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# ragregen/mmkg/medoid_fidelity_pilot.py
"""Matched fidelity pilot: precomputed medoid vs deterministic non-medoid
same-species reference. Pure logic (CPU) + a DI orchestrator (GPU at run).

Spec: docs/superpowers/specs/2026-09-11-mmkg-medoid-fidelity-pilot-design.md
Pre-registration: committing this module (baseline rule, MARGIN, aggregation,
decision) freezes the analysis BEFORE any repair is generated. Do not tune.
"""
from __future__ import annotations


def baseline_reference(record: dict) -> str:
    """Frozen spec-§4 rule: deterministic non-medoid same-species build image.

    eligible = record["candidates"] image paths (the full build pool, medoid
    included) -> exclude the medoid path -> sort ascending -> take the first.
    No randomness, no eyeballing, no NN. Raises if nothing remains.
    """
    medoid_path = record["medoid"]["image_path"]
    paths = sorted(c["image_path"] for c in record["candidates"]
                   if c["image_path"] != medoid_path)
    if not paths:
        raise ValueError(
            f"{record.get('global_id')}: no non-medoid build image for the "
            f"baseline arm (candidates={len(record['candidates'])}).")
    return paths[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mmkg/test_medoid_fidelity_pilot.py -k baseline -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add ragregen/mmkg/medoid_fidelity_pilot.py tests/mmkg/test_medoid_fidelity_pilot.py
git commit -m "feat(mmkg-pilot): frozen deterministic baseline-reference rule"
```

---

### Task 2: Frozen per-species prompt template

One deterministic generation prompt per species, built from the stored names. Frozen before any draft is generated.

**Files:**
- Modify: `ragregen/mmkg/medoid_fidelity_pilot.py`
- Test: `tests/mmkg/test_medoid_fidelity_pilot.py`

**Interfaces:**
- Consumes: `record["common_name"]`, `record["scientific_name"]`.
- Produces: `species_prompt(record: dict) -> str`.

- [ ] **Step 1: Write the failing test**

```python
def test_species_prompt_is_deterministic_and_uses_both_names():
    rec = {"common_name": "Brown Creeper",
           "scientific_name": "Certhia americana"}
    out = p.species_prompt(rec)
    assert out == p.species_prompt(rec)          # deterministic
    assert "Brown Creeper" in out and "Certhia americana" in out
    assert out.startswith("a photo of ")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mmkg/test_medoid_fidelity_pilot.py -k prompt -v`
Expected: FAIL (`species_prompt` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
#: Frozen prompt template. A GENERATION prompt (not a verifier input); it may
#: name the species. Fixed before any draft is generated (spec §11).
PROMPT_TEMPLATE = "a photo of a {common} ({scientific}) bird perched on a branch"


def species_prompt(record: dict) -> str:
    return PROMPT_TEMPLATE.format(common=record["common_name"],
                                  scientific=record["scientific_name"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mmkg/test_medoid_fidelity_pilot.py -k prompt -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ragregen/mmkg/medoid_fidelity_pilot.py tests/mmkg/test_medoid_fidelity_pilot.py
git commit -m "feat(mmkg-pilot): frozen per-species generation prompt template"
```

---

### Task 3: Contamination / disjointness guard

Neither arm's reference may appear in its own scoring set. Abort, never score through overlap (spec §5).

**Files:**
- Modify: `ragregen/mmkg/medoid_fidelity_pilot.py`
- Test: `tests/mmkg/test_medoid_fidelity_pilot.py`

**Interfaces:**
- Consumes: two reference paths and the list of GT paths.
- Produces: `assert_disjoint(ref_paths: list[str], gt_paths: list[str]) -> None` — raises `ValueError` on any overlap (compared by resolved absolute path).

- [ ] **Step 1: Write the failing test**

```python
def test_disjoint_guard_passes_when_build_and_eval_are_separate():
    p.assert_disjoint(["/b/medoid.jpg", "/b/base.jpg"],
                      ["/e/g1.jpg", "/e/g2.jpg", "/e/g3.jpg"])  # no raise


def test_disjoint_guard_aborts_on_any_overlap():
    with pytest.raises(ValueError, match="overlap"):
        p.assert_disjoint(["/b/medoid.jpg", "/e/g2.jpg"],
                          ["/e/g1.jpg", "/e/g2.jpg", "/e/g3.jpg"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mmkg/test_medoid_fidelity_pilot.py -k disjoint -v`
Expected: FAIL (`assert_disjoint` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
import os


def assert_disjoint(ref_paths: list[str], gt_paths: list[str]) -> None:
    """Abort if any reference path is also a ground-truth path (spec §5)."""
    refs = {os.path.abspath(x) for x in ref_paths}
    gts = {os.path.abspath(x) for x in gt_paths}
    both = refs & gts
    if both:
        raise ValueError(
            f"reference/GT overlap -- would self-mark the score: {sorted(both)}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mmkg/test_medoid_fidelity_pilot.py -k disjoint -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ragregen/mmkg/medoid_fidelity_pilot.py tests/mmkg/test_medoid_fidelity_pilot.py
git commit -m "feat(mmkg-pilot): build/eval disjointness guard (abort on overlap)"
```

---

### Task 4: Secondary SigLIP image-identity metric

Mirror `metrics.dino_identity` (mean cosine over held-out crops) but with the SigLIP encoder's `encode_pil`. Descriptive only — never in the decision (spec §6).

**Files:**
- Modify: `ragregen/mmkg/medoid_fidelity_pilot.py`
- Test: `tests/mmkg/test_medoid_fidelity_pilot.py`
- Read-only ref: `ragregen/metrics.py:136-148` (`dino_identity`), `ragregen/metrics.py:158` (`encode_pil` interface).

**Interfaces:**
- Consumes: an output crop `Image`, a list of GT ref `Image`s, a SigLIP encoder exposing `encode_pil([img]) -> array`.
- Produces: `siglip_identity(output_crop, ref_crops, siglip_encoder) -> float` — mean cosine of L2-normalized SigLIP embeddings.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from PIL import Image


class _FakeSiglip:
    """encode_pil returns a fixed vector per pixel-mean, so cosines are known."""
    def encode_pil(self, imgs):
        out = []
        for im in imgs:
            v = np.asarray(im, dtype=np.float32).reshape(-1)[:3]
            out.append(v)
        return np.asarray(out, dtype=np.float32)


def test_siglip_identity_is_mean_cosine_over_refs():
    enc = _FakeSiglip()
    out = Image.new("RGB", (4, 4), (255, 0, 0))
    same = Image.new("RGB", (4, 4), (255, 0, 0))   # cosine 1.0
    orth = Image.new("RGB", (4, 4), (0, 255, 0))   # cosine 0.0
    val = p.siglip_identity(out, [same, orth], enc)
    assert abs(val - 0.5) < 1e-6                    # mean(1.0, 0.0)


def test_siglip_identity_raises_without_refs():
    with pytest.raises(ValueError):
        p.siglip_identity(Image.new("RGB", (4, 4)), [], _FakeSiglip())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mmkg/test_medoid_fidelity_pilot.py -k siglip_identity -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
import numpy as np


def _unit(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-12)


def siglip_identity(output_crop, ref_crops, siglip_encoder) -> float:
    """Mean cosine (SigLIP) between the output crop and each held-out ref.

    Secondary/descriptive only. SigLIP also SELECTED the medoid, so this is
    selection-favorable by construction and never enters the decision (§6).
    """
    refs = list(ref_crops)
    if not refs:
        raise ValueError("no held-out references to score against")
    o = _unit(siglip_encoder.encode_pil([output_crop])[0])
    return float(np.mean([float(o @ _unit(siglip_encoder.encode_pil([r])[0]))
                          for r in refs]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mmkg/test_medoid_fidelity_pilot.py -k siglip_identity -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ragregen/mmkg/medoid_fidelity_pilot.py tests/mmkg/test_medoid_fidelity_pilot.py
git commit -m "feat(mmkg-pilot): secondary SigLIP image-identity metric (descriptive)"
```

---

### Task 5: Pre-registered statistical decision

The frozen §7 read-out over the 25 per-species deltas: mean, paired 95% CI (t, df=n−1), sign split, verdict. Committing this IS the pre-registration.

**Files:**
- Modify: `ragregen/mmkg/medoid_fidelity_pilot.py`
- Test: `tests/mmkg/test_medoid_fidelity_pilot.py`

**Interfaces:**
- Consumes: `deltas: list[float]` (one per species), `margin: float`.
- Produces: `MARGIN: float = 0.02`; `decide(deltas, margin=MARGIN) -> dict` with keys `n, mean, ci_low, ci_high, n_positive, verdict` where `verdict ∈ {"GAIN","LOSS","NO_LARGE_EFFECT"}`.

- [ ] **Step 1: Write the failing test**

```python
def test_margin_matches_the_frozen_project_value():
    from ragregen import metrics
    assert p.MARGIN == abs(metrics.MARGIN) == 0.02   # source: metrics.py:195


def test_decide_flags_gain_when_ci_clears_zero_and_mean_exceeds_margin():
    d = p.decide([0.10] * 25)          # tight, clearly positive
    assert d["verdict"] == "GAIN" and d["ci_low"] > 0 and d["mean"] >= 0.02


def test_decide_flags_loss_symmetrically():
    d = p.decide([-0.10] * 25)
    assert d["verdict"] == "LOSS" and d["ci_high"] < 0


def test_decide_defaults_to_no_large_effect_when_ci_straddles_a_boundary():
    # mean ~0 with spread -> CI includes 0 -> not a large effect -> close
    d = p.decide([0.01, -0.01] * 12 + [0.0])
    assert d["verdict"] == "NO_LARGE_EFFECT"


def test_decide_reports_n_and_sign_split():
    d = p.decide([0.2, -0.1, 0.05])
    assert d["n"] == 3 and d["n_positive"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mmkg/test_medoid_fidelity_pilot.py -k "margin or decide" -v`
Expected: FAIL (`MARGIN` / `decide` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
import math

#: Frozen non-inferiority margin, DINO cosine. SOURCE: ragregen/metrics.py:195
#: (metrics.MARGIN = -0.02), fixed pre-data by doc 5 §2 (~3% of the 0.626
#: operating point). Applied symmetrically here. NEVER tuned after seeing
#: deltas (spec §7). n=25 powers only a LARGE effect; a non-large result reads
#: as NO_LARGE_EFFECT -> close the line (spec §8), not as proven equivalence.
MARGIN = 0.02

#: Two-sided t critical value, df = 24 (n = 25). Hardcoded so the decision has
#: no scipy dependency; if n ever changes, this must change with it.
_T_CRIT_DF24 = 2.0639


def decide(deltas, margin: float = MARGIN) -> dict:
    n = len(deltas)
    if n < 2:
        raise ValueError("need >=2 species to form a CI")
    mean = sum(deltas) / n
    var = sum((x - mean) ** 2 for x in deltas) / (n - 1)
    half = _T_CRIT_DF24 * math.sqrt(var / n)
    ci_low, ci_high = mean - half, mean + half
    n_pos = sum(1 for x in deltas if x > 0)

    if mean >= margin and ci_low > 0:
        verdict = "GAIN"
    elif mean <= -margin and ci_high < 0:
        verdict = "LOSS"
    else:
        verdict = "NO_LARGE_EFFECT"
    return {"n": n, "mean": mean, "ci_low": ci_low, "ci_high": ci_high,
            "n_positive": n_pos, "verdict": verdict}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mmkg/test_medoid_fidelity_pilot.py -k "margin or decide" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ragregen/mmkg/medoid_fidelity_pilot.py tests/mmkg/test_medoid_fidelity_pilot.py
git commit -m "feat(mmkg-pilot): pre-registered DINO decision rule (frozen MARGIN=0.02)"
```

---

### Task 6: Per-species record + delta

One record per species capturing every path and score, plus the per-species delta. Pure serialization; CPU.

**Files:**
- Modify: `ragregen/mmkg/medoid_fidelity_pilot.py`
- Test: `tests/mmkg/test_medoid_fidelity_pilot.py`

**Interfaces:**
- Consumes: `global_id`, `prompt`, path strings (`draft`, `mask`, `ref_medoid`, `ref_baseline`, `repair_medoid`, `repair_baseline`, `gt` list), and four floats (`dino_medoid`, `dino_baseline`, `siglip_medoid`, `siglip_baseline`).
- Produces: `species_record(...) -> dict` including `delta_dino = dino_medoid − dino_baseline` and `delta_siglip = siglip_medoid − siglip_baseline`.

- [ ] **Step 1: Write the failing test**

```python
def test_species_record_carries_paths_scores_and_dino_delta():
    r = p.species_record(
        global_id="inat:00001", prompt="a photo ...",
        draft="/r/d.png", mask="/r/m.png",
        ref_medoid="/b/med.jpg", ref_baseline="/b/base.jpg",
        repair_medoid="/r/A.png", repair_baseline="/r/B.png",
        gt=["/e/g1.jpg", "/e/g2.jpg", "/e/g3.jpg"],
        dino_medoid=0.70, dino_baseline=0.66,
        siglip_medoid=0.80, siglip_baseline=0.79)
    assert r["global_id"] == "inat:00001"
    assert abs(r["delta_dino"] - 0.04) < 1e-9
    assert abs(r["delta_siglip"] - 0.01) < 1e-9
    assert r["ref_medoid"] == "/b/med.jpg" and len(r["gt"]) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mmkg/test_medoid_fidelity_pilot.py -k species_record -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
def species_record(*, global_id, prompt, draft, mask, ref_medoid,
                    ref_baseline, repair_medoid, repair_baseline, gt,
                    dino_medoid, dino_baseline, siglip_medoid,
                    siglip_baseline) -> dict:
    return {
        "global_id": global_id, "prompt": prompt,
        "draft": draft, "mask": mask,
        "ref_medoid": ref_medoid, "ref_baseline": ref_baseline,
        "repair_medoid": repair_medoid, "repair_baseline": repair_baseline,
        "gt": list(gt),
        "dino_medoid": dino_medoid, "dino_baseline": dino_baseline,
        "siglip_medoid": siglip_medoid, "siglip_baseline": siglip_baseline,
        "delta_dino": dino_medoid - dino_baseline,
        "delta_siglip": siglip_medoid - siglip_baseline,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mmkg/test_medoid_fidelity_pilot.py -k species_record -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ragregen/mmkg/medoid_fidelity_pilot.py tests/mmkg/test_medoid_fidelity_pilot.py
git commit -m "feat(mmkg-pilot): per-species record with dino/siglip deltas"
```

---

### Task 7: Orchestrator (dependency-injected) + full-wiring CPU proof

Ties Tasks 1–6 around the model calls via injected collaborators, so the entire pilot — species → prompt → draft → mask → both arms → score → record → decide — is proven on CPU with fakes. Only Task 8 supplies real models. Enforces the no-repair-time-NN invariant.

**Files:**
- Modify: `ragregen/mmkg/medoid_fidelity_pilot.py`
- Test: `tests/mmkg/test_medoid_fidelity_pilot.py`
- Read-only refs: `ragregen/draft.py:55` (`Drafter.draft(prompt)`), `ragregen/mask.py:167` (`mask_draft(image, coarse, masker, ...) -> MaskResult`), `ragregen/regen.py:354` (`Inpainter.regen(draft, mask, reference, prompt, *, seed=...)`), `ragregen/metrics.py:136` (`dino_identity(output_crop, ref_crops, encoder)`), `ragregen/metrics.py` (`crop_to_mask`).

**Interfaces:**
- Consumes: a `deps` object (duck-typed) exposing:
  - `deps.draft(prompt, seed) -> Image`
  - `deps.mask(draft_image) -> mask Image | None` (None = ungroundable → drop)
  - `deps.repair(draft, mask, reference_image, prompt, seed) -> Image`
  - `deps.dino_score(repair_crop, gt_images) -> float`
  - `deps.siglip_score(repair_crop, gt_images) -> float`
  - `deps.crop(image, mask) -> Image`
  - `deps.open(path) -> Image`
  - `deps.assert_no_nn()` — raises if any FAISS/NN was invoked (invariant §9).
- Produces: `run_species(store, gid, gt_paths, deps, out_dir, seed) -> dict | None` (None on drop) and `run_pilot(store, species, deps, out_dir, seed_base) -> dict` (writes records + the `decide` verdict; returns the report dict).

- [ ] **Step 1: Write the failing test (full wiring, CPU, fakes only)**

```python
import json
from PIL import Image


class _FakeStore:
    def __init__(self, recs): self._r = recs
    def get(self, gid): return self._r.get(gid)


class _Deps:
    """Deterministic fakes. medoid ref -> higher dino, to prove the wiring
    surfaces a per-species delta with the right sign (NOT a real result)."""
    def __init__(self, tmp): self.tmp = tmp; self.nn_called = False
    def draft(self, prompt, seed): return Image.new("RGB", (16, 16), (10, 10, 10))
    def mask(self, img):
        m = Image.new("L", (16, 16), 0); m.paste(255, (4, 4, 12, 12)); return m
    def repair(self, draft, mask, reference_image, prompt, seed):
        # encode the reference's red channel into the output so scores differ
        r = reference_image.getpixel((0, 0))[0]
        return Image.new("RGB", (16, 16), (r, 0, 0))
    def crop(self, image, mask): return image.crop((4, 4, 12, 12))
    def open(self, path): return Image.new("RGB", (16, 16),
                                           (200 if "med" in path else 100, 0, 0))
    def dino_score(self, crop, gts):
        return crop.getpixel((0, 0))[0] / 255.0     # medoid(200) > baseline(100)
    def siglip_score(self, crop, gts): return 0.5
    def assert_no_nn(self):
        if self.nn_called: raise AssertionError("repair-time NN ran")


def _rec(gid):
    return {"global_id": gid,
            "common_name": "Brown Creeper", "scientific_name": "Certhia sp",
            "medoid": {"image_path": f"/b/{gid}_med.jpg"},
            "candidates": [{"image_path": f"/b/{gid}_med.jpg"},
                           {"image_path": f"/b/{gid}_base.jpg"}]}


def test_run_species_produces_a_signed_record_without_nn(tmp_path):
    store = _FakeStore({"inat:1": _rec("inat:1")})
    deps = _Deps(tmp_path)
    rec = p.run_species(store, "inat:1",
                        gt_paths=["/e/1.jpg", "/e/2.jpg", "/e/3.jpg"],
                        deps=deps, out_dir=tmp_path, seed=7)
    assert rec["delta_dino"] > 0          # medoid ref scored higher, wiring OK
    assert rec["ref_medoid"].endswith("_med.jpg")
    assert rec["ref_baseline"].endswith("_base.jpg")


def test_run_pilot_writes_records_and_a_verdict(tmp_path):
    store = _FakeStore({f"inat:{i}": _rec(f"inat:{i}") for i in range(1, 4)})
    species = [(f"inat:{i}", ["/e/1.jpg", "/e/2.jpg", "/e/3.jpg"])
               for i in range(1, 4)]
    report = p.run_pilot(store, species, _Deps(tmp_path), tmp_path, seed_base=0)
    assert report["n"] == 3 and "verdict" in report
    assert (tmp_path / "result.json").exists()
    saved = json.loads((tmp_path / "result.json").read_text())
    assert len(saved["records"]) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mmkg/test_medoid_fidelity_pilot.py -k "run_species or run_pilot" -v`
Expected: FAIL (`run_species` / `run_pilot` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
import json
from pathlib import Path


def run_species(store, gid, gt_paths, deps, out_dir, seed) -> dict | None:
    rec = store.get(gid)
    if rec is None:
        raise ValueError(f"{gid} not in store -- no fallback (spec §9)")

    ref_medoid = rec["medoid"]["image_path"]
    ref_baseline = baseline_reference(rec)
    assert_disjoint([ref_medoid, ref_baseline], gt_paths)   # spec §5

    prompt = species_prompt(rec)
    draft = deps.draft(prompt, seed)
    mask = deps.mask(draft)
    if mask is None:
        return None                                          # drop, reported

    d = Path(out_dir) / gid.replace(":", "_")
    d.mkdir(parents=True, exist_ok=True)
    draft.save(d / "draft.png"); mask.save(d / "mask.png")

    gts = [deps.open(g) for g in gt_paths]
    scores = {}
    for arm, ref_path in (("medoid", ref_medoid), ("baseline", ref_baseline)):
        repair = deps.repair(draft, mask, deps.open(ref_path), prompt, seed)
        repair.save(d / f"repair_{arm}.png")
        crop = deps.crop(repair, mask)
        scores[f"dino_{arm}"] = deps.dino_score(crop, gts)
        scores[f"siglip_{arm}"] = deps.siglip_score(crop, gts)

    deps.assert_no_nn()                                      # invariant §9
    return species_record(
        global_id=gid, prompt=prompt,
        draft=str(d / "draft.png"), mask=str(d / "mask.png"),
        ref_medoid=ref_medoid, ref_baseline=ref_baseline,
        repair_medoid=str(d / "repair_medoid.png"),
        repair_baseline=str(d / "repair_baseline.png"),
        gt=gt_paths, **scores)


def run_pilot(store, species, deps, out_dir, seed_base) -> dict:
    records, dropped = [], []
    for i, (gid, gt_paths) in enumerate(species):
        rec = run_species(store, gid, gt_paths, deps, out_dir, seed_base + i)
        (records if rec is not None else dropped).append(rec or gid)

    report = decide([r["delta_dino"] for r in records])
    report["dropped"] = [g for g in dropped]
    report["records"] = records
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "result.json").write_text(json.dumps(report, indent=2))
    return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mmkg/test_medoid_fidelity_pilot.py -v`
Expected: PASS (all tasks' tests, full module).

- [ ] **Step 5: No-regression check**

Run: `pytest tests/mmkg/ tests/mmkg_store/ tests/test_config.py tests/test_run_pipeline.py tests/test_mmkg_reference_integration.py -q`
Expected: PASS (pilot module added nothing that touches the plumbing; existing suites unaffected). Note: `tests/test_regen.py` has 2 pre-existing `cv2`-import failures unrelated to this work — do not treat them as regressions.

- [ ] **Step 6: Commit — THIS IS THE PRE-REGISTRATION**

```bash
git add ragregen/mmkg/medoid_fidelity_pilot.py tests/mmkg/test_medoid_fidelity_pilot.py
git commit -m "feat(mmkg-pilot): DI orchestrator + full-wiring CPU proof (pre-registration)"
```

After this commit the baseline rule, MARGIN, aggregation, and decision are frozen in git history. Any later change to them invalidates the pre-registration.

---

### Task 8: GATED GPU execution + finding (requires explicit user go-ahead)

**Do NOT start this task without the user's explicit approval to spend GPU.** Tasks 1–7 must be committed first (pre-registration). This task writes the real `deps` adapter (thin wrappers over the existing GPU functions), runs the 25-species pilot on the real store, and records the outcome. No new decision logic — it only supplies real models to Task 7's harness.

**Files:**
- Create: `scripts/run_medoid_fidelity_pilot.py` (CLI: `--mmkg-store`, `--out`, `--device`, `--seed-base`; builds real `deps`, resolves the 25 iNat species + their 3 held-out eval paths from the store/build manifest, calls `run_pilot`).
- Create (after the run): `docs/superpowers/<date>-mmkg-medoid-fidelity-finding.md`.
- Read-only refs: `ragregen/draft.py`, `ragregen/mask.py`, `ragregen/regen.py`, `ragregen/metrics.py`, the built store at `/mmlabworkspace_new/Students/tuanld/soict-2026-data/mmkg_store/`.

**Interfaces:**
- The real `deps` implements Task 7's duck-typed contract:
  - `draft` → `ragregen.draft.Drafter(...).draft(prompt)` with a fixed generator seed;
  - `mask` → `ragregen.mask.mask_draft(img, coarse="bird", masker=...)`, returning `None` when grounding fails;
  - `repair` → `ragregen.regen.Inpainter(...).regen(draft, mask, reference, prompt, seed=seed)`;
  - `crop` → `ragregen.metrics.crop_to_mask`;
  - `dino_score(crop, gts)` → `ragregen.metrics.dino_identity(crop, gts, DinoEncoder)` (means over the 3 GT — spec §6);
  - `siglip_score(crop, gts)` → `medoid_fidelity_pilot.siglip_identity(crop, gts, siglip_encoder)`;
  - `open` → `PIL.Image.open(path).convert("RGB")`;
  - `assert_no_nn` → asserts no `Retriever`/`nearest_crops` was constructed (the adapter never imports them).

- [ ] **Step 1: STOP — confirm GPU go-ahead with the user.** Present that Tasks 1–7 are committed (pre-registration frozen) and ask for explicit approval to run on GPU. Do not proceed otherwise.

- [ ] **Step 2: Resolve the 25 species + held-out GT paths.** From the store, list the `inat:*` species (assert exactly 25); for each, resolve its 3 held-out eval image paths from the record's `provenance` / build manifest (the eval split declared at build). Assert 3 GT per species and build/eval disjointness up front. If the store does not persist eval paths, resolve them from the recorded `build_manifest` — do not fabricate.

- [ ] **Step 3: Write `scripts/run_medoid_fidelity_pilot.py`** wiring the real `deps` and calling `run_pilot(store, species, deps, out_dir, seed_base)`.

- [ ] **Step 4: Dry-run wiring on CPU-safe fakes** (import the script, run `run_pilot` with the Task 7 `_Deps` against a 2-species fixture store) to confirm the CLI path constructs and writes `result.json` before any weights load.

- [ ] **Step 5: Execute on GPU** (single run, 25 species, 1 draft each — the frozen n). Capture `result.json` + per-species artifacts under the run dir. Report drops.

- [ ] **Step 6: Write the finding doc.** Record the measured `mean`, 95% CI, sign split, and the §7 verdict, worded within the §12 claim boundary. In the expected NO_LARGE_EFFECT case, state explicitly that Stage-1's *predicted* parity is now a *measured* "no large effect," closing the medoid fidelity direction. **Do not add seeds/drafts to change the result (no-rescue rule, §8).**

- [ ] **Step 7: Commit** the script, the run artifact pointer, and the finding.

```bash
git add scripts/run_medoid_fidelity_pilot.py docs/superpowers/*-mmkg-medoid-fidelity-finding.md
git commit -m "feat(mmkg-pilot): GPU run of medoid-vs-baseline fidelity pilot + finding"
```

---

## Self-Review

**Spec coverage:**
- §3 substrate/unit/n/1-draft → Global Constraints + Task 7 `run_pilot` (n from species list) + Task 8 Step 2 (assert 25). ✅
- §4 frozen baseline rule → Task 1. ✅
- §5 GT + disjointness abort → Task 3 (`assert_disjoint`), Task 7 (called per species), Task 8 Step 2. ✅
- §6 aggregation (dino_identity means over 3 GT), DINO primary / SigLIP secondary → Task 4 (`siglip_identity`), Task 7 (scores), Task 8 (`dino_score` means over GT). ✅
- §7 MARGIN + cited source + decision rule → Task 5 (`MARGIN`, `decide`, test asserts `== abs(metrics.MARGIN)`). ✅
- §8 no-rescue + NO_LARGE_EFFECT semantics → Task 5 (verdict), Task 8 Step 6 (explicit instruction). ✅
- §9/§11 no repair-time NN, no fallback, regen.py read-only → Task 7 (`assert_no_nn`, raise on missing gid; only draft/mask/inpaint/score called), Task 8 (adapter never imports Retriever). ✅
- §12 claim boundary → Global Constraints + Task 8 Step 6. ✅
- Pre-registration (freeze before GPU) → Tasks 1–7 committed before Task 8; Task 7 Step 6 labeled the pre-registration commit; Task 8 Step 1 gate. ✅

**Placeholder scan:** no TBD/TODO; every code step has real code. Task 8 legitimately defers the finding's numbers (they do not exist until the run) and the eval-path resolution detail (spec §11 open item), but the interfaces and asserts are fully specified. ✅

**Type consistency:** `baseline_reference(record)->str`, `species_prompt(record)->str`, `assert_disjoint(list,list)->None`, `siglip_identity(crop,refs,enc)->float`, `MARGIN=0.02`, `decide(list,margin)->dict`, `species_record(...)->dict`, `run_species(...)->dict|None`, `run_pilot(...)->dict` are used identically in their tests, in `run_species`, and in Task 8's adapter contract. `deps` duck-typed contract in Task 7 matches the real adapter enumerated in Task 8. ✅
