# Prototype Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Stream A an image-side contrastive test — score the detected crop against reference *images* of the fine and coarse concepts instead of against their *names*.

**Architecture:** A new `ragregen/verify/prototype.py` builds one mean-embedding prototype per phrase (ported from `../ImageRAG/scripts/finegrained_seg.py:70-105`, which measured 0.917). `GroundedVerifier` scores every candidate box against both prototypes and persists all of them, while its live selection rule and `state` logic stay byte-for-byte unchanged so the pinned τ = 0.25 baseline still reproduces. `c1.py` generalises its state recovery over an explicit *margin*, so the same tested rule machinery grades either mechanism.

**Tech Stack:** Python 3.11, numpy, transformers 5.14.1 (SigLIP-SO400M-384), pytest. No new dependencies.

## Global Constraints

- `$PY=/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python`. Bare `python` is conda 3.13, cannot import faiss, and fails 14 tests.
- **τ is pinned at 0.25 and is never fitted.** `delta_proto` is the only knob selected on this data.
- **`MISSING` is decided by the text `sim` only, and always takes precedence over `FINE_MISMATCH`.** This is what keeps the pre-registered baseline comparable. Never derive `MISSING` from a prototype score.
- **`ABSTAIN` is not `MISSING`.** No box means the detector may have failed.
- `DELTA_OFF = -1.0` is the only value that disables a contrastive test. `0.0` is an aggressive grid point, not "off".
- **The graded box-selection rule for the prototype margin is detector confidence** (`dino_conf` argmax), fixed before any number is looked at. Other rules are diagnostics only.
- **Dependency injection is mandatory.** No module-level model loading, no network at import.
- The 228 existing tests must stay green, and the τ = 0.25 baseline split must reproduce **bit for bit**.
- A case's own image must never enter its own prototype. The `retrieved` arm must never read `gt_refs`.
- Coarse references are authored as a **spread across the category**, never several photographs of one member.
- `git add <exact paths>` — never `git add -A`. `outputs/` is gitignored.
- **Environment caveat:** `import torch` currently segfaults (corrupt `libc10.so`). Until it is repaired, run `$PY -m pytest --ignore=tests/test_draft.py`; `tests/test_draft.py` is the only suite file that imports torch. Task 10 re-runs the complete suite.

---

### Task 1: `coarse_refs` in the dataset schema

**Files:**
- Modify: `ragregen/config.py:21-27` (`Case`), `ragregen/config.py:71-111` (`load_dataset`)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `DatasetConfig.coarse_refs: dict[str, list[Path]]`, keyed by **coarse term** (not case id) so the four shared terms are authored once. Absent key yields `{}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_coarse_refs_are_keyed_by_term_and_resolved_against_images_root(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    cfg = tmp_path / "d.yaml"
    cfg.write_text(f"""
name: t
images_root: {tmp_path}
coarse_refs:
  parrot: [a.jpg]
cases:
  - id: c1
    prompt: p
    concept: "African grey parrot"
    coarse: parrot
    gt_refs: [a.jpg]
""")
    ds = config.load_dataset(cfg)
    assert ds.coarse_refs == {"parrot": [tmp_path / "a.jpg"]}


def test_coarse_refs_absent_is_not_an_error(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    cfg = tmp_path / "d.yaml"
    cfg.write_text(f"""
name: t
images_root: {tmp_path}
cases:
  - id: c1
    prompt: p
    concept: "African grey parrot"
    coarse: parrot
    gt_refs: [a.jpg]
""")
    assert config.load_dataset(cfg).coarse_refs == {}


def test_coarse_refs_for_an_unknown_term_names_the_term(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    cfg = tmp_path / "d.yaml"
    cfg.write_text(f"""
name: t
images_root: {tmp_path}
coarse_refs:
  wombat: [a.jpg]
cases:
  - id: c1
    prompt: p
    concept: "African grey parrot"
    coarse: parrot
    gt_refs: [a.jpg]
""")
    with pytest.raises(ValueError, match="wombat"):
        config.load_dataset(cfg)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_config.py -k coarse_refs -v`
Expected: FAIL — `AttributeError: 'DatasetConfig' object has no attribute 'coarse_refs'`

- [ ] **Step 3: Implement**

In `ragregen/config.py`, add the field to `DatasetConfig` (after `cases`):

```python
    #: Reference images per COARSE term, for the prototype contrast. Keyed by
    #: term rather than by case so the terms several cases share ("dog",
    #: "tree", "bird", "bear") are authored once.
    coarse_refs: dict[str, list[Path]] = field(default_factory=dict)
```

In `load_dataset`, after the `for raw in raw_cases:` loop completes and before the `return`:

```python
    known = {c.coarse for c in cases}
    coarse_refs: dict[str, list[Path]] = {}
    for term, refs in (data.get("coarse_refs") or {}).items():
        if term not in known:
            raise ValueError(
                f"{path}: coarse_refs has an entry for '{term}', which is not "
                f"the coarse term of any case. Known terms: "
                f"{', '.join(sorted(known))}.")
        coarse_refs[term] = [root / r for r in refs]

    return DatasetConfig(name=_require(data, "name", path), images_root=root,
                         cases=cases, coarse_refs=coarse_refs)
```

Delete the old `return DatasetConfig(...)` line it replaces.

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Run the suite**

Run: `$PY -m pytest --ignore=tests/test_draft.py`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add ragregen/config.py tests/test_config.py
git commit -m "feat: coarse_refs in the dataset schema, keyed by coarse term"
```

---

### Task 2: `encode_pil` on the encoder

**Files:**
- Modify: `ragregen/encoders.py:104-118` (`encode_images`)
- Test: `tests/test_encoders.py`

**Interfaces:**
- Produces: `HFEncoder.encode_pil(images, batch_size: int = 32) -> np.ndarray` of shape `(n, d)`, **unnormalised**. `encode_images(paths, ...)` opens the paths and delegates to it.
- Consumed by Task 3. Prototypes are built from files; crops are PIL objects that never touch disk.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_encoders.py`:

```python
def test_encode_images_delegates_to_encode_pil(monkeypatch, tmp_path):
    from PIL import Image
    from ragregen import encoders

    p = tmp_path / "x.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(p)

    enc = encoders.HFEncoder.__new__(encoders.HFEncoder)
    seen = {}

    def fake_encode_pil(images, batch_size=32):
        seen["n"] = len(images)
        seen["mode"] = images[0].mode
        return np.zeros((len(images), 4), dtype=np.float32)

    enc.encode_pil = fake_encode_pil
    out = enc.encode_images([p])
    assert out.shape == (1, 4)
    assert seen == {"n": 1, "mode": "RGB"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_encoders.py -k delegates -v`
Expected: FAIL — `encode_images` opens and encodes inline, so `encode_pil` is never called and `seen` stays empty.

- [ ] **Step 3: Implement**

Replace `HFEncoder.encode_images` (`ragregen/encoders.py:104-118`) with:

```python
    def encode_images(self, paths, batch_size: int = 32) -> np.ndarray:
        from PIL import Image

        paths = list(paths)
        return self.encode_pil([Image.open(p).convert("RGB") for p in paths],
                               batch_size=batch_size)

    def encode_pil(self, images, batch_size: int = 32) -> np.ndarray:
        """Embed already-open PIL images.

        Crops never reach disk, so the prototype path needs an entry point
        that does not go through `Image.open`.
        """
        chunks = []
        images = list(images)
        for start in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[start:start + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                feats = _pooled(self.model.get_image_features(**inputs))
            chunks.append(feats.float().cpu().numpy())
            if start % (batch_size * 20) == 0:
                print(f"  [encode] {start + len(batch)}/{len(images)}", flush=True)
        return np.concatenate(chunks, axis=0)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_encoders.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ragregen/encoders.py tests/test_encoders.py
git commit -m "feat: encode_pil for embedding crops that never reach disk"
```

---

### Task 3: The prototype module

**Files:**
- Create: `ragregen/verify/prototype.py`
- Test: `tests/test_prototype.py`

**Interfaces:**
- Produces:
  - `ImageEmbedder` Protocol with `encode_pil(images, batch_size=32) -> np.ndarray`
  - `build_prototype(images, embedder) -> np.ndarray` — `(d,)`, unit norm
  - `PrototypeBank` with `.get(phrase) -> np.ndarray | None` and `.score(crop_vec, phrase) -> float | None`
  - `cosine(a, b) -> float`
- Consumed by Tasks 4 and 5.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prototype.py`:

```python
import numpy as np
import pytest

from ragregen.verify import prototype


class FakeEmbedder:
    """Returns a preset row per image, in call order."""

    def __init__(self, rows):
        self.rows = [np.asarray(r, dtype=np.float32) for r in rows]
        self.calls = 0

    def encode_pil(self, images, batch_size=32):
        n = len(list(images))
        out = np.stack(self.rows[self.calls:self.calls + n])
        self.calls += n
        return out


def test_build_prototype_normalises_before_and_after_the_mean():
    # Two vectors of very different magnitude pointing 90 degrees apart. If the
    # magnitudes were not divided out first, the long one would dominate and
    # the prototype would sit near the x axis instead of the diagonal.
    emb = FakeEmbedder([[100.0, 0.0], [0.0, 1.0]])
    p = prototype.build_prototype([object(), object()], emb)
    assert p == pytest.approx([2 ** -0.5, 2 ** -0.5], abs=1e-6)


def test_build_prototype_returns_unit_norm():
    emb = FakeEmbedder([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]])
    p = prototype.build_prototype([object()] * 3, emb)
    assert float(np.linalg.norm(p)) == pytest.approx(1.0, abs=1e-6)


def test_build_prototype_of_a_single_reference_is_that_reference_normalised():
    emb = FakeEmbedder([[0.0, 5.0]])
    assert prototype.build_prototype([object()], emb) == pytest.approx([0.0, 1.0])


def test_build_prototype_rejects_an_empty_reference_set():
    with pytest.raises(ValueError, match="at least one reference"):
        prototype.build_prototype([], FakeEmbedder([]))


def test_bank_returns_none_for_an_unknown_phrase():
    bank = prototype.PrototypeBank({"parrot": np.array([1.0, 0.0])})
    assert bank.get("wombat") is None
    assert bank.score(np.array([1.0, 0.0]), "wombat") is None


def test_bank_scores_cosine_against_the_named_prototype():
    bank = prototype.PrototypeBank({"parrot": np.array([1.0, 0.0])})
    assert bank.score(np.array([1.0, 0.0]), "parrot") == pytest.approx(1.0)
    assert bank.score(np.array([0.0, 1.0]), "parrot") == pytest.approx(0.0)


def test_cosine_is_invariant_to_magnitude():
    a, b = np.array([2.0, 0.0]), np.array([9.0, 0.0])
    assert prototype.cosine(a, b) == pytest.approx(1.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_prototype.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragregen.verify.prototype'`

- [ ] **Step 3: Implement**

Create `ragregen/verify/prototype.py`:

```python
"""Image-side prototypes for the fine-grained contrast.

Phrase-level contrast failed because SigLIP's absolute score for a (crop, text)
pair is dominated by how caption-like the phrase is -- "Boston bull" 0.780 vs
"dog" 0.017 on the same crop -- so the margin measured the phrase pair, not the
image (findings/2026-07-27-finegrained-result.md §3a). Comparing the crop
against reference IMAGES removes the text side of that asymmetry entirely.

Ported from `../ImageRAG/scripts/finegrained_seg.py:70-105`, whose
nearest-prototype head measured 0.917 on breed-level box selection. The
arithmetic there is load-bearing and reproduced exactly: normalise each
reference, mean, renormalise.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class ImageEmbedder(Protocol):
    def encode_pil(self, images, batch_size: int = 32) -> np.ndarray: ...


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    # A zero vector has no direction; returning it unchanged keeps cosine at
    # 0.0 rather than producing nan and poisoning every downstream comparison.
    return v if n == 0.0 else v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(np.asarray(a, dtype=np.float64)),
                        _unit(np.asarray(b, dtype=np.float64))))


def build_prototype(images, embedder: ImageEmbedder) -> np.ndarray:
    """One unit-norm prototype from a set of reference images.

    Each reference is normalised BEFORE the mean. Skipping that lets a single
    high-magnitude embedding dominate the prototype, which is a silent quality
    failure rather than a crash.
    """
    images = list(images)
    if not images:
        raise ValueError("a prototype needs at least one reference image")
    feats = np.asarray(embedder.encode_pil(images), dtype=np.float64)
    stacked = np.stack([_unit(row) for row in feats])
    return _unit(stacked.mean(axis=0))


class PrototypeBank:
    """Phrase -> unit-norm prototype vector.

    `get` returns None for an unknown phrase rather than raising: a dataset may
    legitimately carry no references for some coarse term, and that concept
    must fall through to PRESENT rather than fail the case.
    """

    def __init__(self, vectors: dict[str, np.ndarray]):
        self._v = {k: _unit(np.asarray(v, dtype=np.float64))
                   for k, v in vectors.items()}

    def __contains__(self, phrase: str) -> bool:
        return phrase in self._v

    @property
    def phrases(self) -> list[str]:
        return sorted(self._v)

    def get(self, phrase: str) -> np.ndarray | None:
        return self._v.get(phrase)

    def score(self, crop_vec: np.ndarray, phrase: str) -> float | None:
        proto = self._v.get(phrase)
        return None if proto is None else cosine(crop_vec, proto)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_prototype.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ragregen/verify/prototype.py tests/test_prototype.py
git commit -m "feat: image-side prototype construction and scoring"
```

---

### Task 4: The two prototype sources

**Files:**
- Modify: `ragregen/verify/prototype.py`
- Test: `tests/test_prototype.py`

**Interfaces:**
- Produces:
  - `bank_from_gt_refs(dataset, embedder, exclude=()) -> PrototypeBank` — the **ceiling** arm. Fine prototypes from `Case.gt_refs`, coarse from `DatasetConfig.coarse_refs`.
  - `bank_from_retrieval(dataset, retriever, embedder, k=8) -> PrototypeBank` — the **retrieved** arm. Both sides from `retriever.search(phrase, k)`.
  - `ARMS = ("ceiling", "retrieved")`
- `Retriever.search(query, k) -> list[Hit]` with `Hit.path` already exists in `ragregen/retrieve.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_prototype.py`:

```python
from pathlib import Path

from ragregen.config import Case, DatasetConfig


class RecordingEmbedder:
    """Embeds by filename stem, so a prototype's provenance is inspectable."""

    def __init__(self):
        self.seen = []

    def encode_pil(self, images, batch_size=32):
        images = list(images)
        self.seen.extend(images)
        return np.stack([np.array([1.0, 0.0]) for _ in images])


class ExplodingRefs(list):
    def __iter__(self):
        raise AssertionError("the retrieved arm must not read gt_refs")


def _ds(tmp_path, coarse_refs=None):
    (tmp_path / "f.png").write_bytes(b"")
    return DatasetConfig(
        name="t", images_root=tmp_path,
        cases=[Case(id="c1", prompt="p", concept="African grey parrot",
                    coarse="parrot", gt_refs=[tmp_path / "f.png"])],
        coarse_refs=coarse_refs or {},
    )


def test_ceiling_bank_covers_the_fine_and_coarse_terms(tmp_path, monkeypatch):
    monkeypatch.setattr(prototype, "_open", lambda p: object())
    ds = _ds(tmp_path, {"parrot": [tmp_path / "f.png"]})
    bank = prototype.bank_from_gt_refs(ds, RecordingEmbedder())
    assert bank.phrases == ["African grey parrot", "parrot"]


def test_ceiling_bank_omits_a_coarse_term_with_no_references(tmp_path, monkeypatch):
    monkeypatch.setattr(prototype, "_open", lambda p: object())
    bank = prototype.bank_from_gt_refs(_ds(tmp_path), RecordingEmbedder())
    assert bank.phrases == ["African grey parrot"]
    assert bank.get("parrot") is None


def test_ceiling_bank_never_builds_a_prototype_from_an_excluded_image(
        tmp_path, monkeypatch):
    monkeypatch.setattr(prototype, "_open", lambda p: object())
    ds = _ds(tmp_path)
    bank = prototype.bank_from_gt_refs(
        ds, RecordingEmbedder(), exclude={tmp_path / "f.png"})
    assert bank.get("African grey parrot") is None


def test_retrieved_bank_does_not_touch_gt_refs(tmp_path, monkeypatch):
    monkeypatch.setattr(prototype, "_open", lambda p: object())

    class FakeRetriever:
        def search(self, query, k):
            return [type("H", (), {"path": tmp_path / "f.png"})() for _ in range(k)]

    ds = DatasetConfig(
        name="t", images_root=tmp_path,
        cases=[Case(id="c1", prompt="p", concept="African grey parrot",
                    coarse="parrot", gt_refs=ExplodingRefs())],
    )
    bank = prototype.bank_from_retrieval(ds, FakeRetriever(), RecordingEmbedder(), k=2)
    assert bank.phrases == ["African grey parrot", "parrot"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_prototype.py -k "ceiling or retrieved" -v`
Expected: FAIL — `AttributeError: module 'ragregen.verify.prototype' has no attribute 'bank_from_gt_refs'`

- [ ] **Step 3: Implement**

Append to `ragregen/verify/prototype.py`:

```python
#: The two prototype sources. "ceiling" uses ground-truth references and is a
#: diagnostic upper bound -- it is NEVER reportable as a verifier result,
#: because gt_refs are also the target of the DINO identity metric. "retrieved"
#: is the only configuration that can ship.
ARMS = ("ceiling", "retrieved")


def _open(path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def _maybe_build(paths, embedder) -> np.ndarray | None:
    paths = list(paths)
    if not paths:
        return None
    return build_prototype([_open(p) for p in paths], embedder)


def bank_from_gt_refs(dataset, embedder: ImageEmbedder,
                      exclude=()) -> PrototypeBank:
    """Ceiling arm: fine prototypes from gt_refs, coarse from coarse_refs.

    `exclude` drops specific reference paths, so a case's own image can never
    enter its own prototype (finegrained_seg.py:85). A term whose references
    are all excluded is omitted entirely rather than built from nothing.
    """
    excluded = {str(p) for p in exclude}
    vectors: dict[str, np.ndarray] = {}
    for case in dataset.cases:
        v = _maybe_build([p for p in case.gt_refs if str(p) not in excluded],
                         embedder)
        if v is not None:
            vectors[case.concept] = v
    for term, refs in dataset.coarse_refs.items():
        v = _maybe_build([p for p in refs if str(p) not in excluded], embedder)
        if v is not None:
            vectors[term] = v
    return PrototypeBank(vectors)


def bank_from_retrieval(dataset, retriever, embedder: ImageEmbedder,
                        k: int = 8) -> PrototypeBank:
    """Retrieved arm: both sides come from the operator's corpus.

    Deliberately reads only `concept` and `coarse` from each case -- never
    `gt_refs`, which would leak the answer key into the shipping path.
    """
    vectors: dict[str, np.ndarray] = {}
    for case in dataset.cases:
        for phrase in (case.concept, case.coarse):
            if phrase in vectors:
                continue
            v = _maybe_build([h.path for h in retriever.search(phrase, k)],
                             embedder)
            if v is not None:
                vectors[phrase] = v
    return PrototypeBank(vectors)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_prototype.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ragregen/verify/prototype.py tests/test_prototype.py
git commit -m "feat: ceiling and retrieved prototype sources with leakage guards"
```

---

### Task 5: Persist every candidate box

**Files:**
- Modify: `ragregen/verify/grounded.py:36-113`
- Test: `tests/test_grounded.py`

**Interfaces:**
- Produces: `CandidateScore` dataclass; `ConceptScore.candidates: tuple[CandidateScore, ...]`, `.sim_proto`, `.sim_proto_coarse`. `GroundedVerifier.__init__` gains `prototypes: PrototypeBank | None = None` and `embedder: ImageEmbedder | None = None`.
- **The live selection rule and `state` logic do not change.** Prototype scores are recorded, never acted on in-process.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_grounded.py`:

```python
def test_every_candidate_box_is_persisted_not_just_the_winner():
    det = FakeDetector([((0, 0, 4, 4), 0.9), ((0, 0, 8, 8), 0.3)])
    v = grounded.GroundedVerifier(det, FakeScorer({"cat": [0.4, 0.8]}), tau=0.25)
    out = v.score(FakeImage(), [Concept("cat", "object", coarse="animal")])
    assert len(out["cat"].candidates) == 2
    assert [c.dino_conf for c in out["cat"].candidates] == [0.9, 0.3]


def test_prototype_scores_are_recorded_for_every_candidate():
    import numpy as np
    from ragregen.verify import prototype

    det = FakeDetector([((0, 0, 4, 4), 0.9), ((0, 0, 8, 8), 0.3)])
    bank = prototype.PrototypeBank({"cat": np.array([1.0, 0.0]),
                                    "animal": np.array([0.0, 1.0])})

    class Emb:
        def encode_pil(self, images, batch_size=32):
            return np.stack([np.array([1.0, 0.0])] * len(list(images)))

    v = grounded.GroundedVerifier(det, FakeScorer({"cat": [0.4, 0.8]}),
                                  tau=0.25, prototypes=bank, embedder=Emb())
    got = v.score(FakeImage(), [Concept("cat", "object", coarse="animal")])["cat"]
    assert [c.sim_proto for c in got.candidates] == pytest.approx([1.0, 1.0])
    assert [c.sim_proto_coarse for c in got.candidates] == pytest.approx([0.0, 0.0])


def test_wiring_prototypes_does_not_move_the_state_or_the_selected_box():
    """The pinned tau = 0.25 baseline must reproduce bit for bit."""
    import numpy as np
    from ragregen.verify import prototype

    det = FakeDetector([((0, 0, 4, 4), 0.9), ((0, 0, 8, 8), 0.3)])
    scorer = FakeScorer({"cat": [0.4, 0.8], "animal": [0.1]})
    plain = grounded.GroundedVerifier(det, scorer, tau=0.25)
    before = plain.score(FakeImage(), [Concept("cat", "object", coarse="animal")])["cat"]

    class Emb:
        def encode_pil(self, images, batch_size=32):
            return np.stack([np.array([0.0, 1.0])] * len(list(images)))

    det2 = FakeDetector([((0, 0, 4, 4), 0.9), ((0, 0, 8, 8), 0.3)])
    scorer2 = FakeScorer({"cat": [0.4, 0.8], "animal": [0.1]})
    withp = grounded.GroundedVerifier(
        det2, scorer2, tau=0.25,
        prototypes=prototype.PrototypeBank({"cat": np.array([1.0, 0.0])}),
        embedder=Emb())
    after = withp.score(FakeImage(), [Concept("cat", "object", coarse="animal")])["cat"]

    assert (after.state, after.box, after.sim, after.sim_coarse) == \
           (before.state, before.box, before.sim, before.sim_coarse)
```

If `tests/test_grounded.py` has no `FakeDetector` / `FakeScorer` / `FakeImage` returning per-call
scores in order, add them at the top of the file:

```python
class FakeImage:
    def crop(self, box):
        return ("crop", box)


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def all_boxes(self, image, phrase, max_boxes=8):
        return list(self.boxes)[:max_boxes]


class FakeScorer:
    """Returns queued scores per phrase, in call order."""

    def __init__(self, by_phrase):
        self.by_phrase = {k: list(v) for k, v in by_phrase.items()}

    def score(self, crop, phrase):
        queue = self.by_phrase.get(phrase)
        if not queue:
            return 0.0
        return queue.pop(0) if len(queue) > 1 else queue[0]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_grounded.py -k "candidate or prototype or wiring" -v`
Expected: FAIL — `TypeError: ConceptScore.__init__() got an unexpected keyword argument 'candidates'` or `AttributeError: 'ConceptScore' object has no attribute 'candidates'`.

- [ ] **Step 3: Implement**

In `ragregen/verify/grounded.py`, add above `ConceptScore`:

```python
@dataclass(frozen=True)
class CandidateScore:
    """One detector proposal, scored every way we know how.

    The previous finding could not size the box-selection asymmetry because
    `stream_a.json` kept only the winning box's score. Persisting all of them
    makes the selection rule an offline choice that can be varied and reported.
    """
    box: Box
    dino_conf: float
    sim: float
    sim_coarse: float | None = None
    sim_proto: float | None = None
    sim_proto_coarse: float | None = None
```

Add to `ConceptScore`:

```python
    sim_proto: float | None = None
    sim_proto_coarse: float | None = None
    candidates: tuple[CandidateScore, ...] = ()
```

Extend `__init__` (`grounded.py:62-71`) — add the two parameters and store them:

```python
    def __init__(self, detector: Detector, scorer: CropScorer, tau: float,
                 delta: float = 0.0, prototypes=None, embedder=None):
```

and after `self.delta = delta`:

```python
        # Recorded, never acted on in-process: the live state logic must stay
        # byte-for-byte what the pinned tau = 0.25 baseline was graded against.
        self.prototypes = prototypes
        self.embedder = embedder
```

Replace the body of `_score_one` from `best_box, best_conf, best_sim = ...` (line 87) through the
`sim_coarse` block (line 98) with:

```python
        crops = [image.crop((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
                 for b, _ in candidates]

        proto_f: list[float | None] = [None] * len(candidates)
        proto_c: list[float | None] = [None] * len(candidates)
        if self.prototypes is not None and self.embedder is not None:
            vecs = self.embedder.encode_pil(crops)
            for i, vec in enumerate(vecs):
                proto_f[i] = self.prototypes.score(vec, concept.phrase)
                if concept.coarse:
                    proto_c[i] = self.prototypes.score(vec, concept.coarse)

        scored = []
        for i, (box, conf) in enumerate(candidates):
            sim = float(self.scorer.score(crops[i], concept.phrase))
            coarse = (float(self.scorer.score(crops[i], concept.coarse))
                      if concept.coarse else None)
            scored.append(CandidateScore(box=box, dino_conf=conf, sim=sim,
                                         sim_coarse=coarse,
                                         sim_proto=proto_f[i],
                                         sim_proto_coarse=proto_c[i]))

        # Unchanged selection: argmax of the fine-phrase similarity, first one
        # winning ties. Changing this would move `sim`, hence MISSING at
        # tau = 0.25, hence the pre-registered baseline.
        best = max(scored, key=lambda c: c.sim)
        best_box, best_conf, best_sim = best.box, best.dino_conf, best.sim
        sim_coarse = best.sim_coarse
```

Replace the final `return` with:

```python
        return ConceptScore(concept.phrase, concept.kind, state,
                            box=best_box, dino_conf=best_conf, sim=best_sim,
                            sim_coarse=sim_coarse,
                            sim_proto=best.sim_proto,
                            sim_proto_coarse=best.sim_proto_coarse,
                            candidates=tuple(scored))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_grounded.py -v`
Expected: PASS.

- [ ] **Step 5: Run the suite**

Run: `$PY -m pytest --ignore=tests/test_draft.py`
Expected: green. **If any existing grounded test fails, the selection rule moved — revert and fix, do not update the expectation.**

- [ ] **Step 6: Commit**

```bash
git add ragregen/verify/grounded.py tests/test_grounded.py
git commit -m "feat: score and persist every candidate box, including prototypes"
```

---

### Task 6: Serialise candidates and prototype scores

**Files:**
- Modify: `scripts/score_a.py:31-53` (`score_case`), and its argument parser
- Test: `tests/test_score_a.py`

**Interfaces:**
- Produces: each concept in `stream_a.json` gains `sim_proto`, `sim_proto_coarse`, and `candidates: [{box, dino_conf, sim, sim_coarse, sim_proto, sim_proto_coarse}, ...]`. Still **no `state` field** — τ and δ stay offline.
- New CLI flag `--proto-arm {none,ceiling,retrieved}` (default `none`). The arm is recorded in the run trace via `vars(args)`, **not** as a key inside `stream_a.json` — see Step 3.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_score_a.py`:

```python
def test_score_case_serialises_every_candidate():
    from ragregen.verify.grounded import CandidateScore, ConceptScore

    class V:
        def score(self, image, concepts):
            return {"cat": ConceptScore(
                "cat", "object", "PRESENT", box=(0, 0, 1, 1), dino_conf=0.9,
                sim=0.8, sim_coarse=0.1, sim_proto=0.7, sim_proto_coarse=0.2,
                candidates=(
                    CandidateScore((0, 0, 1, 1), 0.9, 0.8, 0.1, 0.7, 0.2),
                    CandidateScore((0, 0, 2, 2), 0.4, 0.3, 0.2, 0.5, 0.4),
                ))}

    got = score_a.score_case(V(), object(), "p", "cat", "animal")
    assert got["cat"]["sim_proto"] == 0.7
    assert got["cat"]["sim_proto_coarse"] == 0.2
    assert len(got["cat"]["candidates"]) == 2
    assert got["cat"]["candidates"][1]["dino_conf"] == 0.4
    assert "state" not in got["cat"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_score_a.py -k serialises_every_candidate -v`
Expected: FAIL — `KeyError: 'sim_proto'`

- [ ] **Step 3: Implement**

In `scripts/score_a.py`, inside `score_case`'s per-concept dict, alongside the existing
`"sim_coarse": s.sim_coarse,` add:

```python
            "sim_proto": s.sim_proto,
            "sim_proto_coarse": s.sim_proto_coarse,
            "candidates": [
                {"box": list(c.box), "dino_conf": c.dino_conf, "sim": c.sim,
                 "sim_coarse": c.sim_coarse, "sim_proto": c.sim_proto,
                 "sim_proto_coarse": c.sim_proto_coarse}
                for c in s.candidates
            ],
```

Add the flag next to the existing `--device` argument:

```python
    ap.add_argument("--proto-arm", choices=("none", "ceiling", "retrieved"),
                    default="none",
                    help="prototype source. 'ceiling' uses gt_refs and is a "
                         "diagnostic upper bound, never a verifier result; "
                         "'retrieved' is the shipping configuration.")
```

In `main`, build the bank after `pipe_cfg` is loaded and before the `GroundedVerifier(...)` line.
The dataset is already loaded as `ds` and the pipeline config as `pipe_cfg` — use those names:

```python
    bank = embedder = None
    if args.proto_arm != "none":
        from ragregen import encoders
        from ragregen.verify import prototype

        embedder = encoders.build_encoder(pipe_cfg.retriever, device=args.device)
        if args.proto_arm == "ceiling":
            bank = prototype.bank_from_gt_refs(ds, embedder)
        else:
            from ragregen import retrieve
            db = config.load_retrieval_db()
            retriever = retrieve.Retriever.from_index(db.index_path, embedder)
            bank = prototype.bank_from_retrieval(ds, retriever, embedder)
        print(f"[score-a] prototype arm '{args.proto_arm}': "
              f"{len(bank.phrases)} prototypes", flush=True)
```

Then change the verifier construction to:

```python
    verifier = GroundedVerifier(detector, scorer, tau=SCORING_TAU,
                                prototypes=bank, embedder=embedder)
```

**Do not add a `_meta` key to the written dict.** `stream_a.json` maps case id to concept scores, and
`c1.abstain_rate` iterates `stream_a.values()` then `.values()` again — a `_meta` entry would be
walked as if it were a case and crash on `str.get`. The arm is already recorded: `trace.open_run`
persists `vars(args)`, which includes `proto_arm`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_score_a.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/score_a.py tests/test_score_a.py
git commit -m "feat: persist candidates and prototype scores in stream_a.json"
```

---

### Task 7: Generalise state recovery over an explicit margin

**Files:**
- Modify: `ragregen/c1.py:173-222` (`state_at`, `grounded_fails`, `fused_fails`, `abstain_rate`)
- Test: `tests/test_c1_states.py` (create if absent)

**Interfaces:**
- Produces:
  - `state_at_margin(sim, margin, tau, delta) -> str`
  - `text_margin(concept: dict) -> float | None` and `proto_margin(concept: dict) -> float | None`
  - `grounded_fails(case_scores, tau, delta=DELTA_OFF, margin_of=text_margin)`; `fused_fails` gains the same keyword and forwards it.
- `state_at` keeps its exact signature and behaviour — it becomes a thin wrapper.

- [ ] **Step 1: Write the failing test**

Create/append `tests/test_c1_states.py`:

```python
import pytest

from ragregen import c1


def test_state_at_margin_matches_state_at_for_the_text_mechanism():
    for sim in (0.0, 0.1, 0.25, 0.6, 0.9):
        for coarse in (None, 0.0, 0.3, 0.95):
            for delta in (-1.0, -0.2, 0.0, 0.2):
                margin = None if coarse is None else sim - coarse
                assert c1.state_at(sim, coarse, 0.25, delta) == \
                       c1.state_at_margin(sim, margin, 0.25, delta)


def test_missing_still_wins_over_a_failing_prototype_margin():
    # sim below tau, prototype margin far below delta: MISSING regardless.
    assert c1.state_at_margin(0.10, -0.9, 0.25, 0.0) == "MISSING"


def _cand(conf, proto, proto_coarse, sim=0.9, sim_coarse=0.0):
    return {"box": [0, 0, 1, 1], "dino_conf": conf, "sim": sim,
            "sim_coarse": sim_coarse, "sim_proto": proto,
            "sim_proto_coarse": proto_coarse}


def test_proto_margin_reads_the_prototype_fields_not_the_text_ones():
    concept = {"sim": 0.8, "sim_coarse": 0.1,
               "candidates": [_cand(0.9, 0.3, 0.5)]}
    assert c1.text_margin(concept) == pytest.approx(0.7)
    assert c1.proto_margin(concept) == pytest.approx(-0.2)


def test_proto_margin_uses_the_highest_confidence_box_not_the_selected_one():
    """The graded selection rule is detector confidence, and it is phrase-neutral.

    The top-level sim_proto belongs to the argmax-of-fine-phrase box. Grading
    on that would reinherit asymmetry (b) -- selection favouring the fine side
    before the margin is taken -- which is precisely what this mechanism exists
    to remove. This test is the guard against a refactor quietly switching back.
    """
    concept = {
        "sim": 0.9, "sim_coarse": 0.0,
        "sim_proto": 0.90, "sim_proto_coarse": 0.10,   # selected box: +0.80
        "candidates": [
            _cand(0.20, 0.90, 0.10),   # high fine-phrase sim, LOW confidence
            _cand(0.95, 0.10, 0.90),   # highest confidence: margin -0.80
        ],
    }
    assert c1.proto_margin(concept) == pytest.approx(-0.80)


def test_margins_are_none_when_the_coarse_side_is_absent():
    assert c1.text_margin({"sim": 0.8, "sim_coarse": None}) is None
    assert c1.proto_margin({"sim": 0.8, "candidates": []}) is None
    assert c1.proto_margin({"sim": 0.8}) is None
    assert c1.proto_margin(
        {"sim": 0.8, "candidates": [_cand(0.9, 0.5, None)]}) is None


def test_grounded_fails_can_be_driven_by_the_prototype_margin():
    scores = {"cat": {"sim": 0.8, "sim_coarse": 0.1,
                      "candidates": [_cand(0.9, 0.3, 0.5)]}}
    assert not c1.grounded_fails(scores, tau=0.25, delta=0.0)
    assert c1.grounded_fails(scores, tau=0.25, delta=0.0,
                             margin_of=c1.proto_margin)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_c1_states.py -v`
Expected: FAIL — `AttributeError: module 'ragregen.c1' has no attribute 'state_at_margin'`

- [ ] **Step 3: Implement**

In `ragregen/c1.py`, replace `state_at` (lines 173-190) with:

```python
def state_at_margin(sim: float | None, margin: float | None,
                    tau: float, delta: float) -> str:
    """Recover a state from a similarity and an explicit contrastive margin.

    Which margin -- text or prototype -- is the caller's choice. MISSING is
    always decided by `sim`, the text similarity, and always takes precedence:
    that is what keeps the pinned tau = 0.25 baseline comparable across both
    mechanisms.
    """
    if sim is None:
        return "ABSTAIN"
    if sim < tau:
        return "MISSING"
    # round() mirrors grounded.py exactly. Without it, 0.60 - 0.50 is
    # 0.09999999999999998 and a margin that should sit exactly on delta falls
    # below it. The two implementations must agree bit for bit or the report
    # grades the verifier against a rule it does not use.
    if margin is not None and round(margin, 9) < delta:
        return "FINE_MISMATCH"
    return "PRESENT"


def state_at(sim: float | None, sim_coarse: float | None,
             tau: float, delta: float) -> str:
    """The text mechanism. Mirrors GroundedVerifier._score_one exactly."""
    margin = None if (sim is None or sim_coarse is None) else sim - sim_coarse
    return state_at_margin(sim, margin, tau, delta)


def text_margin(concept: dict) -> float | None:
    """sim - sim_coarse, or None when either side is absent."""
    sim, coarse = concept.get("sim"), concept.get("sim_coarse")
    return None if (sim is None or coarse is None) else sim - coarse


def proto_margin(concept: dict) -> float | None:
    """sim_proto - sim_proto_coarse at the HIGHEST-CONFIDENCE candidate box.

    Deliberately not the top-level sim_proto. That belongs to the box chosen by
    argmax of the fine-phrase similarity, so the selection has already favoured
    the fine side before the margin is taken -- asymmetry (b) in
    findings/2026-07-27-finegrained-result.md §3b, which prototype scoring does
    NOT remove on its own. Detector confidence is neutral between the fine and
    coarse terms, so both sides are selected the same way and then measured the
    same way.

    This is the pre-registered graded rule. Other selection rules may be
    reported as diagnostics; none may replace this one.
    """
    scored = [c for c in (concept.get("candidates") or [])
              if c.get("sim_proto") is not None
              and c.get("sim_proto_coarse") is not None]
    if not scored:
        return None
    best = max(scored, key=lambda c: c["dino_conf"])
    return best["sim_proto"] - best["sim_proto_coarse"]
```

Replace `grounded_fails`, `fused_fails` and `abstain_rate` with margin-aware versions:

```python
def grounded_fails(case_scores: dict, tau: float, delta: float = DELTA_OFF,
                   margin_of=text_margin) -> bool:
    """Stream A fails a case iff some concept is MISSING or FINE_MISMATCH."""
    return any(
        state_at_margin(s.get("sim"), margin_of(s), tau, delta) in FAILING_STATES
        for s in case_scores.values()
    )


def fused_fails(case_scores: dict, case_b: dict, tau: float,
                delta: float = DELTA_OFF, margin_of=text_margin) -> bool:
    """Mirrors fusion.fuse: ok = not (grounded_failed or semantic_failed)."""
    return (grounded_fails(case_scores, tau, delta, margin_of)
            or semantic_fails(case_b))


def abstain_rate(stream_a: dict, tau: float, delta: float = DELTA_OFF) -> float:
    """Fraction of scored concepts that abstained.

    Reported because a high rate means Stream A is inert and `fused` is
    silently just Stream B wearing a second name. Independent of delta and of
    the margin source; the parameter exists so callers can pass a sweep row
    through uniformly.
    """
    states = [state_at_margin(s.get("sim"), text_margin(s), tau, delta)
              for case in stream_a.values() for s in case.values()]
    if not states:
        return 0.0
    return sum(1 for s in states if s == "ABSTAIN") / len(states)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_c1_states.py -v`
Expected: PASS.

- [ ] **Step 5: Verify the published numbers still reproduce**

Run: `$PY -m pytest --ignore=tests/test_draft.py`
Expected: green, including every existing `test_c1_*` test. **These pin the published C1 table; a
failure here means the refactor changed the mechanism, not the tests.**

- [ ] **Step 6: Commit**

```bash
git add ragregen/c1.py tests/test_c1_states.py
git commit -m "refactor: recover states from an explicit margin, text or prototype"
```

---

### Task 8: Grade the prototype rule

**Files:**
- Modify: `ragregen/c1.py` (`baseline_split`, `evaluate_rule`, `sweep_delta`)
- Test: `tests/test_c1_rule.py`

**Interfaces:**
- Produces: `evaluate_rule(..., baseline_false_positives=(), margin_of=text_margin)`. With the default empty tuple, behaviour is **identical to today** — any false positive fails the rule.
- `baseline_split` and `sweep_delta` gain `margin_of` and forward it.
- New: `baseline_false_positives(stream_a, stream_b, case_ids, y_true, tau) -> list[str]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_c1_rule.py`:

```python
def test_baseline_false_positives_names_the_threshold_driven_ones():
    stream_a = {"good": {"p": {"sim": 0.05, "sim_coarse": 0.0}},
                "fine": {"p": {"sim": 0.90, "sim_coarse": 0.0}}}
    stream_b = {c: {"ok": True, "degenerate": False} for c in stream_a}
    got = c1.baseline_false_positives(stream_a, stream_b, ["good", "fine"],
                                      [False, False], tau=0.25)
    assert got == ["good"]


def _concept(sim, proto, proto_coarse, sim_coarse=0.0):
    """A persisted concept carrying one candidate box.

    proto_margin reads the highest-confidence CANDIDATE, never the top-level
    fields, so fixtures must supply candidates or the margin is None.
    """
    return {"sim": sim, "sim_coarse": sim_coarse,
            "candidates": [{"box": [0, 0, 1, 1], "dino_conf": 0.9, "sim": sim,
                            "sim_coarse": sim_coarse, "sim_proto": proto,
                            "sim_proto_coarse": proto_coarse}]}


def test_a_false_positive_inherited_from_the_baseline_does_not_fail_the_rule():
    """tau = 0.25 imports FPs before delta is swept. Grading against *new*
    FPs measures the mechanism instead of the threshold it inherited."""
    stream_a = {
        "miss": {"p": _concept(0.90, 0.1, 0.9)},   # PRESENT, proto margin -0.8
        "fp":   {"p": _concept(0.05, 0.9, 0.1)},   # below tau -> MISSING
    }
    stream_b = {c: {"ok": True, "degenerate": False} for c in stream_a}
    ids, y = ["miss", "fp"], [True, False]
    caught0, missed0 = c1.baseline_split(stream_a, stream_b, ids, y, tau=0.25)
    assert missed0 == ["miss"] and caught0 == []

    r = c1.evaluate_rule(stream_a, stream_b, ids, y, caught0, missed0,
                         delta=0.0, tau=0.25, min_catch=1,
                         baseline_false_positives=["fp"],
                         margin_of=c1.proto_margin)
    assert r.caught == ["miss"]
    assert r.false_positives == []          # 'fp' is inherited, not new
    assert r.passes


def test_a_new_false_positive_still_fails_the_rule():
    stream_a = {
        "miss": {"p": _concept(0.90, 0.1, 0.9)},
        "ok":   {"p": _concept(0.90, 0.1, 0.9)},   # correct image, margin fires
    }
    stream_b = {c: {"ok": True, "degenerate": False} for c in stream_a}
    ids, y = ["miss", "ok"], [True, False]
    caught0, missed0 = c1.baseline_split(stream_a, stream_b, ids, y, tau=0.25)
    r = c1.evaluate_rule(stream_a, stream_b, ids, y, caught0, missed0,
                         delta=0.0, tau=0.25, min_catch=1,
                         baseline_false_positives=[],
                         margin_of=c1.proto_margin)
    assert r.false_positives == ["ok"]
    assert not r.passes
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_c1_rule.py -k "baseline_false_positives or inherited or new_false" -v`
Expected: FAIL — `AttributeError: module 'ragregen.c1' has no attribute 'baseline_false_positives'`

- [ ] **Step 3: Implement**

In `ragregen/c1.py`, add `margin_of=text_margin` to `baseline_split`'s signature and forward it to
`fused_fails`. Do the same for `sweep_delta`. Then add:

```python
def baseline_false_positives(stream_a: dict, stream_b: dict,
                             case_ids: list[str], y_true: list[bool],
                             tau: float = BASELINE_TAU,
                             margin_of=text_margin) -> list[str]:
    """Correct images the delta-disabled baseline already fails.

    tau = 0.25 imports two of these on the identity column (stack_of_books,
    sushi), both flagged MISSING by the threshold rather than by any
    contrastive rule. Requiring zero false positives absolutely made the
    pre-registered criterion unsatisfiable before delta was ever swept, so the
    rule is graded against *new* false positives instead.
    """
    return [cid for cid, is_fail in zip(case_ids, y_true)
            if not is_fail
            and fused_fails(stream_a[cid], stream_b[cid], tau, DELTA_OFF,
                            margin_of)]
```

Replace `evaluate_rule` with:

```python
def evaluate_rule(stream_a: dict, stream_b: dict, case_ids: list[str],
                  y_true: list[bool], baseline_caught: list[str],
                  baseline_missed: list[str], delta: float,
                  tau: float = BASELINE_TAU, min_catch: int = 4,
                  baseline_false_positives=(),
                  margin_of=text_margin) -> RuleResult:
    """The three pre-registered criteria at one delta.

    `baseline_false_positives` defaults to empty, which reproduces the original
    zero-FP criterion exactly.
    """
    fails = {cid: fused_fails(stream_a[cid], stream_b[cid], tau, delta, margin_of)
             for cid in case_ids}
    inherited = set(baseline_false_positives)

    caught = [cid for cid in baseline_missed if fails[cid]]
    false_positives = [cid for cid, is_fail in zip(case_ids, y_true)
                       if not is_fail and fails[cid] and cid not in inherited]
    gross_retained = all(fails[cid] for cid in baseline_caught)

    passes = (len(caught) >= min_catch
              and not false_positives
              and gross_retained)
    return RuleResult(delta=delta, caught=caught,
                      false_positives=false_positives,
                      gross_retained=gross_retained, passes=passes)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_c1_rule.py -v`
Expected: PASS, including every pre-existing rule test.

- [ ] **Step 5: Commit**

```bash
git add ragregen/c1.py tests/test_c1_rule.py
git commit -m "feat: grade the rule against new false positives, over either margin"
```

---

### Task 9: Report the prototype verdict

**Files:**
- Modify: `ragregen/c1.py` (add `render_prototype`), `scripts/c1_report.py` (`main`)
- Test: `tests/test_c1_render.py`

**Interfaces:**
- Produces: `render_prototype(stream_a, stream_b, case_ids, y_true, column, tau=BASELINE_TAU) -> tuple[str, dict]`, appended to `c1.md` and stored under `artifact[column]["prototype"]`.
- Returns `("", {})` when no case carries prototype scores, so a `--proto-arm none` run is unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_c1_render.py`:

```python
def test_prototype_section_is_empty_when_no_prototypes_were_scored():
    stream_a = {"c0": {"p": {"sim": 0.40, "sim_coarse": 0.55}}}
    stream_b = {"c0": {"ok": True, "degenerate": False}}
    md, art = c1.render_prototype(stream_a, stream_b, ["c0"], [True],
                                  column="verdict_identity")
    assert md == "" and art == {}


def _proto_concept(sim, proto, proto_coarse):
    """proto_margin reads candidates, not the top-level fields."""
    return {"sim": sim, "sim_coarse": 0.0,
            "candidates": [{"box": [0, 0, 1, 1], "dino_conf": 0.9, "sim": sim,
                            "sim_coarse": 0.0, "sim_proto": proto,
                            "sim_proto_coarse": proto_coarse}]}


def test_prototype_section_names_the_verdict_and_uses_the_proto_margin():
    stream_a = {
        "miss": {"p": _proto_concept(0.90, 0.10, 0.90)},
        "good": {"p": _proto_concept(0.90, 0.90, 0.10)},
    }
    stream_b = {c: {"ok": True, "degenerate": False} for c in stream_a}
    md, art = c1.render_prototype(stream_a, stream_b, ["miss", "good"],
                                  [True, False], column="verdict_identity")
    assert art["baseline_missed"] == ["miss"]
    assert art["mechanism"] == "prototype"
    assert "miss" in md
    assert art["tau"] == c1.BASELINE_TAU
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_c1_render.py -k prototype -v`
Expected: FAIL — `AttributeError: module 'ragregen.c1' has no attribute 'render_prototype'`

- [ ] **Step 3: Implement**

Append to `ragregen/c1.py`:

```python
def _has_prototypes(stream_a: dict) -> bool:
    return any(proto_margin(s) is not None
               for case in stream_a.values() for s in case.values())


def render_prototype(stream_a: dict, stream_b: dict, case_ids: list[str],
                     y_true: list[bool], column: str,
                     tau: float = BASELINE_TAU) -> tuple[str, dict]:
    """The prototype section: same rule machinery, image-side margin.

    Returns empty output when nothing carries prototype scores, so a run
    without --proto-arm is byte-for-byte what it was before.
    """
    if not _has_prototypes(stream_a):
        return "", {}

    caught0, missed0 = baseline_split(stream_a, stream_b, case_ids, y_true, tau)
    inherited = baseline_false_positives(stream_a, stream_b, case_ids, y_true, tau)
    results = [evaluate_rule(stream_a, stream_b, case_ids, y_true,
                             caught0, missed0, delta=d, tau=tau,
                             baseline_false_positives=inherited,
                             margin_of=proto_margin)
               for d in DELTA_GRID]
    band = robust_band(results)
    met = bool(band)
    chosen = band[len(band) // 2] if met else None

    L = [f"## Prototype verification — `{column}`", "",
         f"tau pinned at **{tau}** (not fitted). delta_proto swept over "
         f"{len(DELTA_GRID)} values from {DELTA_GRID[0]} to {DELTA_GRID[-1]}.", "",
         f"**Baseline at tau = {tau}, delta disabled:** "
         f"{len(caught0)} caught, {len(missed0)} missed.", "",
         f"- caught: {', '.join(caught0) if caught0 else '(none)'}",
         f"- missed: {', '.join(missed0) if missed0 else '(none)'}",
         f"- false positives inherited from the threshold: "
         f"{', '.join(inherited) if inherited else '(none)'}", ""]

    if met:
        L += [f"### RULE MET across delta_proto {band[0].delta} to "
              f"{band[-1].delta} ({len(band)} consecutive values)", "",
              f"At delta_proto = {chosen.delta}: **{len(chosen.caught)} of "
              f"{len(missed0)}** previously-missed cases now caught "
              f"({', '.join(chosen.caught) if chosen.caught else 'none'}), "
              f"**{len(chosen.false_positives)} new false positives**, "
              f"gross-category catches retained: "
              f"**{'yes' if chosen.gross_retained else 'NO'}**.", ""]
    else:
        best = max(results, key=lambda r: len(r.caught))
        L += ["### RULE NOT MET", "",
              "No band of 3 or more consecutive delta_proto values satisfies "
              "all three criteria. The best single delta_proto was "
              f"{best.delta}, catching {len(best.caught)} of {len(missed0)} "
              f"with {len(best.false_positives)} new false positives. "
              "Per the spec the conclusion is that image-side contrast is "
              "insufficient, not that delta_proto needs more tuning.", ""]

    artifact = {
        "mechanism": "prototype",
        "tau": tau,
        "baseline_caught": caught0,
        "baseline_missed": missed0,
        "inherited_false_positives": inherited,
        "rule_met": met,
        "band": [r.delta for r in band],
        "chosen_delta": chosen.delta if met else None,
        "caught_at_chosen": chosen.caught if met else [],
        "new_false_positives_at_chosen": chosen.false_positives if met else [],
    }
    return "\n".join(L), artifact
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_c1_render.py -v`
Expected: PASS.

- [ ] **Step 5: Wire it into the report**

In `scripts/c1_report.py`, inside `main`'s `for column in COLUMNS:` loop, after the existing
`sections.append(fg_md)`:

```python
        pr_md, pr_art = c1.render_prototype(stream_a, stream_b, case_ids,
                                            y_true, column=column)
        if pr_md:
            sections.append(pr_md)
```

and inside the `artifact[column] = {...}` dict:

```python
            "prototype": pr_art,
```

- [ ] **Step 6: Run the suite**

Run: `$PY -m pytest --ignore=tests/test_draft.py`
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add ragregen/c1.py scripts/c1_report.py tests/test_c1_render.py
git commit -m "feat: report the prototype rule verdict alongside the phrase one"
```

---

### Task 10: Operator documentation and the full-suite gate

**Files:**
- Modify: `docs/RUNBOOK.md` §3.1 and §4, `configs/dataset.example.yaml`, `ragregen/validate.py`
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes: everything above. Produces no new API.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_validate.py`:

```python
def test_validate_warns_when_a_coarse_term_is_thinly_referenced(tmp_path):
    from ragregen import validate
    from ragregen.config import Case, DatasetConfig

    ds = DatasetConfig(
        name="t", images_root=tmp_path,
        cases=[Case(id="c1", prompt="p", concept="African grey parrot",
                    coarse="parrot", gt_refs=[tmp_path / "a.png"])],
        coarse_refs={"parrot": [tmp_path / "a.png", tmp_path / "b.png"]},
    )
    warnings = validate.check_coarse_refs(ds)
    assert any("parrot" in w and "2" in w for w in warnings)


def test_validate_is_quiet_when_coarse_refs_are_absent_entirely(tmp_path):
    from ragregen import validate
    from ragregen.config import Case, DatasetConfig

    ds = DatasetConfig(
        name="t", images_root=tmp_path,
        cases=[Case(id="c1", prompt="p", concept="c", coarse="k",
                    gt_refs=[tmp_path / "a.png"])])
    assert validate.check_coarse_refs(ds) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_validate.py -k coarse_refs -v`
Expected: FAIL — `AttributeError: module 'ragregen.validate' has no attribute 'check_coarse_refs'`

- [ ] **Step 3: Implement**

Add to `ragregen/validate.py`:

```python
#: Below this many references a prototype is dominated by individual
#: photographs rather than the category. A warning, not an error: the operator
#: may be running the phrase mechanism only.
MIN_COARSE_REFS = 3


def check_coarse_refs(dataset) -> list[str]:
    """Warn about thin coarse reference sets.

    Spread cannot be checked mechanically -- three photographs of the same
    parrot satisfy any count -- so this catches only the countable half and
    the RUNBOOK carries the rest.
    """
    out = []
    for term, refs in sorted(dataset.coarse_refs.items()):
        if len(refs) < MIN_COARSE_REFS:
            out.append(
                f"coarse term '{term}' has only {len(refs)} reference "
                f"image(s); {MIN_COARSE_REFS}+ recommended, spread across the "
                f"category rather than several photographs of one member.")
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_validate.py -v`
Expected: PASS.

- [ ] **Step 5: Document it**

Add to `docs/RUNBOOK.md` §3.1, after the existing `coarse` entry:

```markdown
**`coarse_refs`** (optional, top level). Reference images per coarse term, used
by the prototype verifier. Keyed by the term itself, so terms several cases
share — `dog`, `tree`, `bird`, `bear` — are authored once:

```yaml
coarse_refs:
  parrot: [parrot_generic_0.jpg, parrot_generic_1.jpg, parrot_generic_2.jpg]
  dog:    [dog_generic_0.jpg, dog_generic_1.jpg, dog_generic_2.jpg]
```

Three or more per term, and **spread across the category** — three photographs
of the same parrot make a second fine-grained prototype, not a superordinate
one, and the contrast stops measuring what it claims. `validate` checks the
count; nothing can check the spread.
```

Add to `docs/RUNBOOK.md` §4, replacing the rows for the stages that do not exist
(`pilot`, `full`, `report`) with the stages that do, and adding:

```markdown
| `score-a --proto-arm ceiling` | yes | ~5 min | Stream A with prototypes from `gt_refs`. **A diagnostic ceiling — never quote it as a verifier result**, because `gt_refs` are also the DINO metric's target. |
| `score-a --proto-arm retrieved` | yes | ~5 min | Stream A with prototypes from your corpus. This is the reportable configuration; needs `build-index` first. |
```

Add the `coarse_refs` block to `configs/dataset.example.yaml`, with the same
authoring warning as a comment.

- [ ] **Step 6: Run the complete suite**

Run: `$PY -m pytest`
Expected: all green with **no `--ignore`**. If `tests/test_draft.py` still segfaults, the
`libc10.so` repair has not been applied — stop and report rather than skipping it.

- [ ] **Step 7: Commit**

```bash
git add docs/RUNBOOK.md configs/dataset.example.yaml ragregen/validate.py tests/test_validate.py
git commit -m "docs: coarse_refs authoring contract and the two prototype arms"
```

---

## Notes for the executor

- **Do not re-tune τ.** If a task tempts you to move it, the plan is wrong — stop and say so.
- **The baseline is the contract.** Any change that moves the τ = 0.25 split of 9 caught / 6 missed is a defect, even if every test still passes.
- **Six plan-authored fixture defects were found across the previous ten tasks.** Verify fixture arithmetic by running it, never by reasoning about it. If a brief's expected value disagrees with what the code produces, check which is wrong before assuming it is the code.
- Neither arm can be measured in this worktree: `coarse_refs` and a FAISS index are both operator inputs that do not exist yet. Tasks end at "implemented and tested".
