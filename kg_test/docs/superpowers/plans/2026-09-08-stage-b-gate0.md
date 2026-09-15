# Stage-B Gate 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Gate 0 of the Stage-B pilot — does taxonomy attribute-transfer predict a held-out target concept's *rare* attributes better than visual-NN retrieval, under the fully pre-registered protocol.

**Architecture:** Two new modules. `graft/stageb_vlm.py` isolates every VLM call (held-out attribute read + bidirectional equivalence judge) behind small functions so all decision logic in `graft/stageb_gate0.py` (loaders, disjoint held-out label, rarity, the two arms, McNemar, gate decision) is pure and unit-tested with a mock VLM. `main()` wires the pieces and writes `result.json`.

**Tech Stack:** Python 3.11, numpy, scipy.stats (binomtest for exact McNemar), Pillow; project models (`graft.models.Models`: `vlm` = Qwen2.5-VL-7B, `siglip` = SigLIP2). Reuse `graft.taxonomy`, `graft.kg_build`, `graft.dataset.split_refs`.

## Global Constraints

- **Spec is frozen** — `docs/superpowers/specs/2026-09-08-stage-b-attribute-transfer-pilot-design.md`. Every knob below is copied from it; do not change any threshold, prompt, or rule.
- **NO GIT** — repo `.git` is a stub. Replace every "commit" step with: append a one-line progress note to `.superpowers/sdd/2026-09-07-relational-mmkg-stage-a-prime/progress.md`. No `git` commands.
- **Attributes (exactly 5, part-level):** `leaf.shape`, `leaf.texture`, `bark.texture`, `bark.color`, `branching.pattern`.
- **NOT_VISIBLE set:** `{"not visible","none","n/a",""}` (compare on `str(v).strip().lower()`).
- **Rarity:** value's evaluable-concept count `≤ max(2, ceil(0.15 * N_visible))`, exact-normalized strings.
- **Judge prompt (frozen, verbatim):** `You are comparing two descriptions of a plant's {A}. A: "{x}". B: "{y}". Do A and B describe essentially the same {A}? Answer with only 'yes' or 'no'.` — `do_sample=False`, both orders, match iff **both** start with "yes".
- **Held-out read instruction:** `kg_build.SCHEMA_INSTRUCTION_NEUTRAL` (name-blind), `do_sample=False`, `max_new_tokens=512`, parsed by `kg_build.parse_vlm_schema`.
- **Concept universe:** evaluable = 25 concepts in `outputs/mmkg_oh/heldout_parts/`; prediction pool = all concepts in `outputs/*/kg.json`.
- **Build/held-out disjointness:** `B(C) = set(split_refs(load_species("data/treevill/rawdata2", C), k_build=5, seed=0)[0])`; label read only from held-out `ref_path ∉ B(C)`; assert disjoint.
- **PASS bar:** `rare_recall(MMKG) − rare_recall(retrieval) ≥ 0.15` AND `n_rare ≥ 30` AND `McNemar p < 0.05`. Else wind down.
- **Env python:** `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python`. Run tests with `PYTHONPATH=.`.

---

### Task 1: Attribute + exemplar loaders and build-image set

**Files:**
- Create: `graft/stageb_gate0.py`
- Test: `tests/test_stageb_gate0.py`

**Interfaces:**
- Produces:
  - `norm(v: str) -> str` — `str(v).strip().lower()`.
  - `NOT_VISIBLE: set[str]`.
  - `PART_ATTRS: list[tuple[str,str]]` — the 5 (part, attr) pairs.
  - `load_build_attrs(kg_glob="outputs/*/kg.json") -> dict[str, dict[tuple[str,str], str]]` — concept → {(part,attr): normed value}, visible only.
  - `load_exemplars(kg_glob="outputs/*/kg.json") -> dict[str, dict[str, np.ndarray]]` — concept → {part: L2-normed siglip2}.
  - `build_image_set(concept, root="data/treevill/rawdata2") -> set[str]` — `set(split_refs(load_species(root, concept), 5, 0)[0])`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stageb_gate0.py
import json, numpy as np
from graft import stageb_gate0 as g

def test_load_build_attrs_visible_only(tmp_path):
    kg = {"concept": "X", "parts": [
        {"name": "bark", "attributes": [{"name": "texture", "value": "Rough "},
                                        {"name": "color", "value": "not visible"}]},
        {"name": "leaf", "attributes": [{"name": "shape", "value": "Elliptical"}]}]}
    d = tmp_path / "X"; d.mkdir(); (d / "kg.json").write_text(json.dumps(kg))
    out = g.load_build_attrs(str(tmp_path / "*" / "kg.json"))
    assert out["X"][("bark", "texture")] == "rough"      # normalized
    assert ("bark", "color") not in out["X"]              # not-visible dropped
    assert out["X"][("leaf", "shape")] == "elliptical"

def test_load_exemplars_l2_normed(tmp_path):
    kg = {"concept": "X", "parts": [
        {"name": "leaf", "embeddings": {"siglip2": [3.0, 4.0]}}]}
    d = tmp_path / "X"; d.mkdir(); (d / "kg.json").write_text(json.dumps(kg))
    out = g.load_exemplars(str(tmp_path / "*" / "kg.json"))
    assert np.allclose(np.linalg.norm(out["X"]["leaf"]), 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. /mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_stageb_gate0.py -q`
Expected: FAIL (module `stageb_gate0` not found).

- [ ] **Step 3: Write minimal implementation**

```python
# graft/stageb_gate0.py
from __future__ import annotations
import glob, json, os
import numpy as np
from graft.dataset import load_species, split_refs

NOT_VISIBLE = {"not visible", "none", "n/a", ""}
PART_ATTRS = [("leaf", "shape"), ("leaf", "texture"), ("bark", "texture"),
              ("bark", "color"), ("branching", "pattern")]
_PA = set(PART_ATTRS)

def norm(v) -> str:
    return str(v).strip().lower()

def load_build_attrs(kg_glob="outputs/*/kg.json"):
    out = {}
    for kp in glob.glob(kg_glob):
        kg = json.load(open(kp)); c = kg["concept"]; d = {}
        for pr in kg.get("parts", []):
            for at in pr.get("attributes", []):
                key = (pr["name"], at["name"])
                if key in _PA:
                    v = norm(at.get("value"))
                    if v not in NOT_VISIBLE:
                        d[key] = v
        out[c] = d
    return out

def load_exemplars(kg_glob="outputs/*/kg.json"):
    out = {}
    for kp in glob.glob(kg_glob):
        kg = json.load(open(kp)); c = kg["concept"]; d = {}
        for pr in kg.get("parts", []):
            v = pr.get("embeddings", {}).get("siglip2")
            if v:
                a = np.asarray(v, float); d[pr["name"]] = a / (np.linalg.norm(a) or 1.0)
        out[c] = d
    return out

def build_image_set(concept, root="data/treevill/rawdata2"):
    return set(split_refs(load_species(root, concept), 5, 0)[0])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .../python -m pytest tests/test_stageb_gate0.py -q`
Expected: PASS.

- [ ] **Step 5: Record progress** (no git) — append `Task 1 done: stageb_gate0 loaders + build_image_set.` to the ledger.

---

### Task 2: Disjoint held-out label (majority over reads)

**Files:**
- Modify: `graft/stageb_gate0.py`
- Test: `tests/test_stageb_gate0.py`

**Interfaces:**
- Consumes: `norm`, `NOT_VISIBLE`.
- Produces: `heldout_label(records, build_set, read_fn) -> dict[tuple[str,str], str]` — records = list of held-out store dicts (each has `ref_path`); `read_fn(ref_path) -> dict[(part,attr), str]`; returns per-(part,attr) strict-plurality normed value over UNIQUE `ref_path ∉ build_set`, excluding NOT_VISIBLE; ties or no-visible → key omitted. Asserts read images are disjoint from `build_set`.

- [ ] **Step 1: Write the failing test**

```python
def test_heldout_label_majority_disjoint_and_ties():
    records = [{"ref_path": "a.jpg"}, {"ref_path": "a.jpg"},   # dup -> read once
               {"ref_path": "b.jpg"}, {"ref_path": "c.jpg"},
               {"ref_path": "BUILD.jpg"}]                       # excluded (in build set)
    reads = {
        "a.jpg": {("bark", "texture"): "rough", ("leaf", "shape"): "ovate"},
        "b.jpg": {("bark", "texture"): "rough", ("leaf", "shape"): "elliptical"},
        "c.jpg": {("bark", "texture"): "not visible", ("leaf", "shape"): "ovate"},
        "BUILD.jpg": {("bark", "texture"): "smooth"},           # must NOT be read
    }
    calls = []
    def read_fn(p): calls.append(p); return reads[p]
    lab = g.heldout_label(records, build_set={"BUILD.jpg"}, read_fn=read_fn)
    assert "BUILD.jpg" not in calls                              # disjointness enforced
    assert sorted(calls) == ["a.jpg", "b.jpg", "c.jpg"]          # deduped, build excluded
    assert lab[("bark", "texture")] == "rough"                  # 2 rough vs 1 not-visible(dropped)
    assert ("leaf", "shape") not in lab                          # ovate 2 vs elliptical 1 -> wait: tie? see note
```

> Note for implementer: in this fixture `leaf.shape` = ovate(a), elliptical(b), ovate(c) → ovate 2, elliptical 1 → **plurality ovate, not a tie**. Fix the test's final assert to `assert lab[("leaf","shape")] == "ovate"`. (Kept here to force you to reason about the count before coding.)

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .../python -m pytest tests/test_stageb_gate0.py::test_heldout_label_majority_disjoint_and_ties -q`
Expected: FAIL (`heldout_label` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
from collections import Counter, defaultdict

def heldout_label(records, build_set, read_fn):
    seen, imgs = set(), []
    for r in records:
        p = r["ref_path"]
        if p in build_set or p in seen:
            continue
        seen.add(p); imgs.append(p)
    assert not (set(imgs) & set(build_set)), "held-out label images overlap build set"
    votes = defaultdict(Counter)
    for p in imgs:
        for key, val in read_fn(p).items():
            v = norm(val)
            if v not in NOT_VISIBLE:
                votes[key][v] += 1
    out = {}
    for key, c in votes.items():
        top = c.most_common()
        if len(top) == 1 or top[0][1] > top[1][1]:      # strict plurality; ties dropped
            out[key] = top[0][0]
    return out
```

- [ ] **Step 4: Run test to verify it passes** (after fixing the final assert per the note).

Run: `PYTHONPATH=. .../python -m pytest tests/test_stageb_gate0.py -q` → PASS.

- [ ] **Step 5: Record progress** — `Task 2 done: disjoint held-out majority label.`

---

### Task 3: Rarity

**Files:**
- Modify: `graft/stageb_gate0.py`
- Test: `tests/test_stageb_gate0.py`

**Interfaces:**
- Consumes: `norm`.
- Produces: `rare_values(labels_by_concept, part, attr) -> set[str]` — over evaluable concepts' labels (dict concept→{(part,attr):value}); value counts by exact-normed string; rare = count `≤ max(2, ceil(0.15 * N_visible))`, where `N_visible` = #concepts with a visible label for (part,attr).

- [ ] **Step 1: Write the failing test**

```python
import math
def test_rare_values_threshold():
    # 20 concepts visible for (bark,texture): 'rough' x16, 'flaky' x2, 'peeling' x2
    labels = {}
    for i in range(16): labels[f"r{i}"] = {("bark","texture"): "rough"}
    for i in range(2):  labels[f"f{i}"] = {("bark","texture"): "flaky"}
    for i in range(2):  labels[f"p{i}"] = {("bark","texture"): "peeling"}
    labels["nv"] = {}                                    # no visible value -> ignored
    rare = g.rare_values(labels, "bark", "texture")
    # N_visible=20, thr=max(2, ceil(0.15*20)=3)=3 -> rough(16) common; flaky(2),peeling(2) rare
    assert rare == {"flaky", "peeling"}
```

- [ ] **Step 2: Run to verify fail.** Expected: FAIL (`rare_values` undefined).

- [ ] **Step 3: Implement**

```python
import math

def rare_values(labels_by_concept, part, attr):
    key = (part, attr)
    vals = [d[key] for d in labels_by_concept.values() if key in d]
    n_vis = len(vals)
    if n_vis == 0:
        return set()
    thr = max(2, math.ceil(0.15 * n_vis))
    cnt = Counter(vals)
    return {v for v, k in cnt.items() if k <= thr}
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Record progress** — `Task 3 done: rarity.`

---

### Task 4: The two arms (retrieval NN, MMKG taxonomy)

**Files:**
- Modify: `graft/stageb_gate0.py`
- Test: `tests/test_stageb_gate0.py`

**Interfaces:**
- Consumes: `load_exemplars`/`load_build_attrs` shapes, `graft.taxonomy.family_of`.
- Produces:
  - `retrieval_pred(concept, part, attr, exemplars, build_attrs) -> str | None` — among concepts (≠concept) with a visible build value for (part,attr) AND a part exemplar, the one with max cosine to `concept`'s part exemplar; its value. None if none exist or concept lacks the part exemplar.
  - `mmkg_pred(concept, part, attr, exemplars, build_attrs) -> str | None` — siblings = concepts (≠concept) with `family_of == family_of(concept)` (not None) and a visible build value; plurality value; ties → value of the sibling with max part-exemplar cosine to `concept`; None if no sibling with a visible value.

- [ ] **Step 1: Write the failing test**

```python
def test_arms_retrieval_vs_taxonomy(monkeypatch):
    import graft.taxonomy as tax
    monkeypatch.setattr(tax, "family_of",
                        lambda c: {"C":"F","S1":"F","S2":"F","NN":"G"}.get(c))
    ex = {  # part 'bark' exemplars in R^2 (unnormed ok; fn normalizes if needed -> normalize here)
        "C":  {"bark": np.array([1.0, 0.0])},
        "NN": {"bark": np.array([0.99, 0.01])},   # visually nearest, different family
        "S1": {"bark": np.array([0.2, 1.0])},     # sibling, far
        "S2": {"bark": np.array([0.3, 1.0])},     # sibling, far
    }
    ba = {"NN": {("bark","texture"): "smooth"},
          "S1": {("bark","texture"): "peeling"},
          "S2": {("bark","texture"): "peeling"},
          "C":  {("bark","texture"): "rough"}}     # C's own value never used
    assert g.retrieval_pred("C","bark","texture", ex, ba) == "smooth"   # nearest = NN
    assert g.mmkg_pred("C","bark","texture", ex, ba) == "peeling"       # sibling plurality
    # no-sibling case -> None
    monkeypatch.setattr(tax, "family_of", lambda c: {"C":"F"}.get(c))
    assert g.mmkg_pred("C","bark","texture", ex, ba) is None
```

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement**

```python
from graft import taxonomy

def _cos(a, b):
    a = a / (np.linalg.norm(a) or 1.0); b = b / (np.linalg.norm(b) or 1.0)
    return float(a @ b)

def retrieval_pred(concept, part, attr, exemplars, build_attrs):
    q = exemplars.get(concept, {}).get(part)
    if q is None:
        return None
    key = (part, attr)
    cands = [(o, _cos(q, exemplars[o][part])) for o in exemplars
             if o != concept and part in exemplars[o] and key in build_attrs.get(o, {})]
    if not cands:
        return None
    cands.sort(key=lambda x: -x[1])
    return build_attrs[cands[0][0]][key]

def mmkg_pred(concept, part, attr, exemplars, build_attrs):
    fam = taxonomy.family_of(concept)
    if fam is None:
        return None
    key = (part, attr)
    sibs = [o for o in build_attrs if o != concept
            and taxonomy.family_of(o) == fam and key in build_attrs[o]]
    if not sibs:
        return None
    cnt = Counter(build_attrs[o][key] for o in sibs)
    top = cnt.most_common()
    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0]
    tied = {v for v, k in top if k == top[0][1]}          # tie-break by exemplar cosine
    q = exemplars.get(concept, {}).get(part)
    best = max((o for o in sibs if build_attrs[o][key] in tied),
               key=lambda o: _cos(q, exemplars[o][part]) if (q is not None and part in exemplars[o]) else -1.0)
    return build_attrs[best][key]
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Record progress** — `Task 4 done: arms.`

---

### Task 5: VLM module — held-out read + bidirectional judge

**Files:**
- Create: `graft/stageb_vlm.py`
- Test: `tests/test_stageb_vlm.py`

**Interfaces:**
- Produces:
  - `read_attributes(image_paths, models) -> dict[tuple[str,str], str]` — calls `models.vlm.describe(images, kg_build.SCHEMA_INSTRUCTION_NEUTRAL)`, parses via `kg_build.parse_vlm_schema`, returns {(part,attr): value} for the 5 PART_ATTRS present.
  - `judge_match(x, y, attr_name, models) -> bool` — builds the frozen prompt for (x,y) and (y,x), calls `models.vlm.describe([], prompt)` (text-only) with the same neutral decoding, returns True iff both replies `.strip().lower().startswith("yes")`. If either `x`/`y` normed ∈ NOT_VISIBLE → return False without calling the VLM.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stageb_vlm.py
from graft import stageb_vlm as sv

class FakeVLM:
    def __init__(self, replies): self.replies = replies; self.prompts = []
    def describe(self, images, instruction):
        self.prompts.append(instruction); return self.replies.pop(0)

class FakeModels:
    def __init__(self, vlm): self.vlm = vlm

def test_read_attributes_parses_schema():
    reply = ('{"global":{}, "parts":{"bark":{"texture":"rough","color":"brown"},'
             '"leaf":{"shape":"ovate","texture":"smooth"},'
             '"branching":{"pattern":"irregular"},"cone_or_flower":{"appearance":"not visible"}}}')
    m = FakeModels(FakeVLM([reply]))
    out = sv.read_attributes(["a.jpg"], m)
    assert out[("bark","texture")] == "rough" and out[("leaf","shape")] == "ovate"
    assert ("cone_or_flower","appearance") not in out       # not one of the 5 PART_ATTRS

def test_judge_match_bidirectional_and():
    m = FakeModels(FakeVLM(["Yes", "yes"]));  assert sv.judge_match("rough","rough, flaky","bark texture", m) is True
    m = FakeModels(FakeVLM(["yes", "no"]));   assert sv.judge_match("rough","smooth","bark texture", m) is False
    m = FakeModels(FakeVLM([]));              assert sv.judge_match("not visible","rough","bark texture", m) is False  # no VLM call
```

- [ ] **Step 2: Run to verify fail.** Expected: FAIL (`stageb_vlm` not found).

- [ ] **Step 3: Implement**

```python
# graft/stageb_vlm.py
from __future__ import annotations
from graft import kg_build
from graft.stageb_gate0 import PART_ATTRS, NOT_VISIBLE, norm

_PA = set(PART_ATTRS)
_JUDGE = ('You are comparing two descriptions of a plant\'s {A}. A: "{x}". B: "{y}". '
          "Do A and B describe essentially the same {A}? Answer with only 'yes' or 'no'.")

def read_attributes(image_paths, models):
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    raw = models.vlm.describe(imgs, kg_build.SCHEMA_INSTRUCTION_NEUTRAL)
    _global, parts = kg_build.parse_vlm_schema(raw)
    out = {}
    for part, nodes in parts.items():
        for n in nodes:
            if (part, n.name) in _PA:
                out[(part, n.name)] = n.value
    return out

def _yes(reply): return str(reply).strip().lower().startswith("yes")

def judge_match(x, y, attr_name, models):
    if norm(x) in NOT_VISIBLE or norm(y) in NOT_VISIBLE:
        return False
    a = models.vlm.describe([], _JUDGE.format(A=attr_name, x=x, y=y))
    b = models.vlm.describe([], _JUDGE.format(A=attr_name, x=y, y=x))
    return _yes(a) and _yes(b)
```

> Implementer note: confirm `models.vlm.describe` accepts an empty image list for the text-only judge. If it requires ≥1 image, pass a fixed 1×1 white `Image.new("RGB",(8,8),"white")` and keep decoding identical — record the choice in the ledger; it does not change the frozen prompt.

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Record progress** — `Task 5 done: stageb_vlm read + judge.`

---

### Task 6: Case construction, McNemar, gate decision

**Files:**
- Modify: `graft/stageb_gate0.py`
- Test: `tests/test_stageb_gate0.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `Case` dict shape: `{"concept","part","attr","label","rare":bool,"pred_retrieval","pred_mmkg"}`.
  - `build_cases(labels_by_concept, exemplars, build_attrs) -> list[Case]` — for each evaluable concept & (part,attr) with a valid label, retrieval pred, AND mmkg pred (both non-None): a Case; `rare` from `rare_values`.
  - `score_cases(cases, judge_fn) -> list[dict]` — adds `match_retrieval`, `match_mmkg` bools via `judge_fn(pred, label, "<part> <attr>")`.
  - `mcnemar_p(scored) -> float` — exact two-sided binomial on discordant pairs (b = MMKG-only-correct, c = retrieval-only-correct) over the RARE scored cases; `binomtest(min(b,c), b+c, 0.5).pvalue` (p=1.0 if b+c==0).
  - `gate0(scored) -> dict` — rare recalls per arm, delta, n_rare, mcnemar_p, and `decision` ("PASS"/"WIND_DOWN") per the bar. Also common-case recalls (sanity).

- [ ] **Step 1: Write the failing test**

```python
def test_gate0_decision_and_mcnemar():
    # 30 rare cases: MMKG right on 24, retrieval right on 12 (delta +0.40)
    labels_by_concept = {}; exemplars = {}; build_attrs = {}  # not used by scorer here
    cases = []
    for i in range(30):
        mmkg_ok = i < 24; retr_ok = i < 12
        cases.append({"concept": f"C{i}", "part": "bark", "attr": "texture",
                      "label": "L", "rare": True,
                      "pred_mmkg": "L" if mmkg_ok else "X",
                      "pred_retrieval": "L" if retr_ok else "Y"})
    judge = lambda pred, lab, a: pred == lab                 # exact for the test
    scored = g.score_cases(cases, judge)
    out = g.gate0(scored)
    assert out["n_rare"] == 30
    assert abs(out["rare_recall_mmkg"] - 24/30) < 1e-9
    assert abs(out["rare_recall_retrieval"] - 12/30) < 1e-9
    assert out["delta"] > 0.15
    assert out["mcnemar_p"] < 0.05
    assert out["decision"] == "PASS"
```

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement**

```python
from scipy.stats import binomtest

def build_cases(labels_by_concept, exemplars, build_attrs):
    cases = []
    rare_cache = {}
    for c, lab in labels_by_concept.items():
        for (part, attr), val in lab.items():
            pr = retrieval_pred(c, part, attr, exemplars, build_attrs)
            pm = mmkg_pred(c, part, attr, exemplars, build_attrs)
            if pr is None or pm is None:
                continue
            if (part, attr) not in rare_cache:
                rare_cache[(part, attr)] = rare_values(labels_by_concept, part, attr)
            cases.append({"concept": c, "part": part, "attr": attr, "label": val,
                          "rare": val in rare_cache[(part, attr)],
                          "pred_retrieval": pr, "pred_mmkg": pm})
    return cases

def score_cases(cases, judge_fn):
    out = []
    for k in cases:
        a = f"{k['part']} {k['attr']}"
        r = dict(k)
        r["match_retrieval"] = bool(judge_fn(k["pred_retrieval"], k["label"], a))
        r["match_mmkg"] = bool(judge_fn(k["pred_mmkg"], k["label"], a))
        out.append(r)
    return out

def mcnemar_p(scored):
    rare = [s for s in scored if s["rare"]]
    b = sum(1 for s in rare if s["match_mmkg"] and not s["match_retrieval"])
    c = sum(1 for s in rare if s["match_retrieval"] and not s["match_mmkg"])
    if b + c == 0:
        return 1.0
    return float(binomtest(min(b, c), b + c, 0.5).pvalue)

def _recall(scored, arm, rare):
    sub = [s for s in scored if s["rare"] == rare]
    if not sub:
        return float("nan")
    return sum(1 for s in sub if s[f"match_{arm}"]) / len(sub)

def gate0(scored):
    n_rare = sum(1 for s in scored if s["rare"])
    rr_m = _recall(scored, "mmkg", True); rr_r = _recall(scored, "retrieval", True)
    delta = rr_m - rr_r; p = mcnemar_p(scored)
    decision = "PASS" if (delta >= 0.15 and n_rare >= 30 and p < 0.05) else "WIND_DOWN"
    return {"n_rare": n_rare, "n_total": len(scored),
            "rare_recall_mmkg": rr_m, "rare_recall_retrieval": rr_r, "delta": delta,
            "mcnemar_p": p, "common_recall_mmkg": _recall(scored, "mmkg", False),
            "common_recall_retrieval": _recall(scored, "retrieval", False),
            "decision": decision}
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Record progress** — `Task 6 done: cases + McNemar + gate0.`

---

### Task 7: Feasibility pre-check + `main()` orchestration

**Files:**
- Modify: `graft/stageb_gate0.py`
- Test: `tests/test_stageb_gate0.py` (feasibility only; `main` is the live run)

**Interfaces:**
- Consumes: all above; `graft.stageb_vlm`, `graft.models.Models`, `graft.config.GraftConfig`.
- Produces:
  - `feasibility(labels_by_concept, exemplars, build_attrs) -> dict` — using given labels as proxy, counts included cases and rare cases; `{"n_included","n_rare","by_attr"}`.
  - `main()` — CLI: `--heldout-dir outputs/mmkg_oh/heldout_parts --kg-glob "outputs/*/kg.json" --root data/treevill/rawdata2 --out outputs/mmkg/stageb_gate0`. Flow: load build_attrs+exemplars; **proxy feasibility** (build_attrs as labels) → if `n_rare < 30`, write `result.json` with `{"decision":"WIND_DOWN_FEASIBILITY", ...}` and STOP before loading the VLM; else load `Models`, read held-out labels per evaluable concept (`stageb_vlm.read_attributes` on disjoint images), `build_cases`, `score_cases` with `stageb_vlm.judge_match`, `gate0`, write `result.json`.

- [ ] **Step 1: Write the failing test**

```python
def test_feasibility_counts(monkeypatch):
    import graft.taxonomy as tax
    monkeypatch.setattr(tax, "family_of", lambda c: "F")     # all same family -> siblings exist
    ex = {c: {"bark": np.array([1.0, float(i)])} for i, c in enumerate(["A","B","C","D"])}
    # 4 concepts, (bark,texture): rough x2, peeling x1, flaky x1 -> thr=max(2,ceil(0.6))=2 -> peeling,flaky rare
    ba = {"A": {("bark","texture"):"rough"}, "B": {("bark","texture"):"rough"},
          "C": {("bark","texture"):"peeling"}, "D": {("bark","texture"):"flaky"}}
    labels = {c: ba[c] for c in ba}
    out = g.feasibility(labels, ex, ba)
    assert out["n_included"] == 4 and out["n_rare"] == 2
```

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement**

```python
def feasibility(labels_by_concept, exemplars, build_attrs):
    cases = build_cases(labels_by_concept, exemplars, build_attrs)
    by_attr = Counter((c["part"], c["attr"]) for c in cases if c["rare"])
    return {"n_included": len(cases), "n_rare": sum(1 for c in cases if c["rare"]),
            "by_attr": {f"{p}.{a}": n for (p, a), n in by_attr.items()}}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-dir", default="outputs/mmkg_oh/heldout_parts")
    ap.add_argument("--kg-glob", default="outputs/*/kg.json")
    ap.add_argument("--root", default="data/treevill/rawdata2")
    ap.add_argument("--out", default="outputs/mmkg/stageb_gate0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    build_attrs = load_build_attrs(a.kg_glob)
    exemplars = load_exemplars(a.kg_glob)
    evalc = sorted(os.path.splitext(x)[0] for x in os.listdir(a.heldout_dir))

    proxy = {c: build_attrs.get(c, {}) for c in evalc}
    feas = feasibility(proxy, exemplars, build_attrs)
    if feas["n_rare"] < 30:
        json.dump({"decision": "WIND_DOWN_FEASIBILITY", "feasibility": feas},
                  open(os.path.join(a.out, "result.json"), "w"), indent=2)
        print("WIND_DOWN_FEASIBILITY", feas); return

    from graft.config import GraftConfig
    from graft.models import Models
    from graft import stageb_vlm
    cfg = GraftConfig.from_yaml("configs/oraclehub.yaml")
    models = Models(cfg)
    labels = {}
    for c in evalc:
        recs = json.load(open(os.path.join(a.heldout_dir, f"{c}.json")))
        B = build_image_set(c, a.root)
        labels[c] = heldout_label(recs, B, lambda p: stageb_vlm.read_attributes([p], models))
    cases = build_cases(labels, exemplars, build_attrs)
    scored = score_cases(cases, lambda x, y, attr: stageb_vlm.judge_match(x, y, attr, models))
    out = gate0(scored)
    out["feasibility"] = feas
    json.dump({"result": out, "cases": scored}, open(os.path.join(a.out, "result.json"), "w"), indent=2)
    print("DECISION:", out["decision"], "delta=%.3f" % out["delta"],
          "n_rare=%d" % out["n_rare"], "p=%.4f" % out["mcnemar_p"])

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run → PASS** the feasibility test; full suite `PYTHONPATH=. .../python -m pytest -q -k "not gpu"`.
- [ ] **Step 5: Record progress** — `Task 7 done: feasibility + main.`

---

### Task 8: Live Gate-0 run + result report

**Files:**
- Create: `reports/2026-09-08-stageb-gate0-result.md`
- Run artifact: `outputs/mmkg/stageb_gate0/result.json`

- [ ] **Step 1:** Run detached (VLM load spike → memory guard; mirror the OracleHub pattern):
`setsid .../python -m graft.stageb_gate0 --heldout-dir outputs/mmkg_oh/heldout_parts --out outputs/mmkg/stageb_gate0 > outputs/mmkg/stageb_gate0.log 2>&1 < /dev/null &`
- [ ] **Step 2:** Watch for `result.json` (or the process to exit); confirm no traceback in the log.
- [ ] **Step 3:** Read `result.json`: `decision`, `delta`, `n_rare`, `mcnemar_p`, both recalls, common-case sanity recalls.
- [ ] **Step 4:** Write `reports/2026-09-08-stageb-gate0-result.md`: the pre-registered bar, the numbers, the PASS/WIND_DOWN decision, the VLM-bias limitation (§5), and — on PASS — that Gate 1 is unlocked; on FAIL — wind down with the standing verdict intact.
- [ ] **Step 5:** Update memory (`mmkg-to-ragregen-direction`) + ledger with the Gate-0 outcome.

---

## Self-Review

**Spec coverage:** §2.0 universe → Task 7 (`evalc` + pool). §2.1 attrs → `PART_ATTRS` (Task 1). §2.2 label + disjointness → Task 2 + `build_image_set` (Task 1) + Task 7 wiring. §2.3 rarity → Task 3. §2.4 arms → Task 4. §2.5 inclusion → `build_cases` (Task 6). §2.6 judge → Task 5. §3 metric/McNemar/bar → Task 6 (`gate0`). Feasibility pre-check → Task 7. Run/report → Task 8. Gate 1 → out of scope (downstream), per plan scope. **No gaps.**

**Placeholder scan:** none — every step has concrete code. The two implementer notes (heldout_label test count; empty-image judge call) are reasoning prompts, not deferred work.

**Type consistency:** `norm`, `NOT_VISIBLE`, `PART_ATTRS` defined in Task 1 and imported by `stageb_vlm` (Task 5). Case dict keys (`pred_retrieval`/`pred_mmkg`/`match_*`/`rare`) consistent across Tasks 6–7. `judge_fn(pred, label, "<part> <attr>")` signature matches `judge_match(x, y, attr_name, models)` via the Task 7 lambda. `read_fn(ref_path)->dict` matches the Task 7 lambda wrapping `read_attributes([p], models)`.
