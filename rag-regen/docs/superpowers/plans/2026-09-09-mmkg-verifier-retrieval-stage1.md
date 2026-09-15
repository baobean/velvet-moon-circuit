# MMKG Verifier + Organized Retrieval — Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an MMKG per holdout concept and test, offline over one trace, whether its fine-grained attribute verifier detects the identity failures rag-regen's semantic verifier misses (Role 2, scored on the frozen 48-case human-labelled holdout), while logging a genuine flat-vs-organized retrieval comparison (Role 1).

**Architecture:** Pure decision logic (schema/consensus, verifier verdict, retrieval-arm divergence, metrics) lives in small `ragregen/mmkg/` modules unit-tested with a mock VLM. Every VLM call (name-blind slot read + bidirectional judge) sits behind a thin interface so all scoring is deterministic and offline. One batch orchestrator builds the graphs, runs a single VLM read pass, writes one trace record per case, then a pure offline evaluator applies frozen gates.

**Tech Stack:** Python 3.11 (`/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python`), numpy, Pillow, faiss (via `ragregen.retrieve`), Qwen2.5-VL via `ragregen.vlm.QwenVLM`. Reuse `ragregen.retrieve`, `ragregen.retrieved_reference.contamination`, `ragregen.eval_manifest.sha256_file`. Spec: `docs/superpowers/specs/2026-09-09-mmkg-verifier-retrieval-stage1-design.md` (frozen).

## Global Constraints

- **Spec is frozen.** Every threshold/prompt/rule below is copied from the spec; do not change any.
- **Git.** rag-regen is a real repo currently on branch `mmkg-db-repair-stage1` with **pre-existing unrelated uncommitted changes** (`.gitignore`, `configs/*`, `ragregen/c1.py`, …). Before Task 1, branch: `git checkout -b mmkg-verifier-stage1`. In every commit step, `git add` **only the files that task names** — **never** `git add -A` or `.` — so the unrelated dirty files are never swept in.
- **SLOTS (frozen, exactly 6):** `primary_color`, `secondary_color`, `pattern_or_markings`, `surface_texture`, `overall_shape_or_form`, `distinctive_feature`.
- **NOT_VISIBLE:** `{"not visible","none","n/a",""}` (compare on `str(v).strip().lower()`).
- **Consensus (target attribute):** slot S is a target of concept C iff `V_vis(S) ≥ 3` AND a strict-plurality normalized value has count `≥ max(2, ceil(0.5·V_vis(S)))`; target value = that plurality value. C is **decidable** iff it has `≥ 2` target attributes.
- **Verdict:** per present target attr, `consistent` iff `norm(target)==norm(read)` OR judge returns match (exact-string short-circuit blessed 2026-09-09); else `contradicted`; `missing` (`norm∈NOT_VISIBLE`) abstains. Case **ABSTAIN** iff C undecidable OR `n_present==0`; else **FAIL** iff `n_contradicted ≥ 1` else **PASS**. Primary metric maps **ABSTAIN→PASS over all 48**.
- **Judge (frozen, verbatim):** `You are comparing two descriptions of an object's {slot}. A: "{x}". B: "{y}". Do A and B describe essentially the same {slot}? Answer with only 'yes' or 'no'.` — evaluate **both orders**, match iff **both** `.strip().lower().startswith("yes")`. Text-only ⇒ pass a blank `Image.new("RGB",(8,8),"white")` (QwenVLM needs ≥1 image); decoding greedy.
- **Read prompt (frozen, verbatim):** `Describe ONLY the single main subject in this image. Do not name it. Return a JSON object with exactly these keys: primary_color, secondary_color, pattern_or_markings, surface_texture, overall_shape_or_form, distinctive_feature. Each value is a short phrase, or "not visible" if you cannot tell. Output only the JSON.`
- **Gates (frozen):** failure recall `≥ 0.75` (≥18/24) AND FP rate `≤ 0.083` (≤2/24) ⇒ PASS; else honest negative, no integration.
- **Slice retrieval K (frozen):** `K_SLICE = 40`, reusing the **stored query string** in `outputs/phase_b/retrieval.json` verbatim against `data/laion100k/index.faiss` (parity); dedup by sha256; contamination-guard each image.
- **No new image generation. No DINO-sign as ground truth. Ground truth = human labels** (`outputs/phase_b/labels_holdout48.csv`, column `verdict_identity` ∈ {fail,pass}).
- **Test/run:** `PYTHONPATH=. /mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest -q` (gpu deselected by default). Live VLM/build steps are scripts, not tests.

---

### Task 1: Schema, normalization, and target-attribute consensus

**Files:**
- Create: `ragregen/mmkg/__init__.py` (empty), `ragregen/mmkg/schema.py`
- Test: `tests/mmkg/test_schema.py` (create `tests/mmkg/__init__.py` if the suite needs it — mirror existing `tests/` layout)

**Interfaces:**
- Produces:
  - `SLOTS: list[str]` (the 6, in the frozen order), `NOT_VISIBLE: set[str]`, `norm(v)->str`.
  - `target_attributes(reads: list[dict[str,str]]) -> dict[str, dict]` — input = per-image `{slot: value}` reads; output = `{slot: {"value": str, "support": int, "visible_count": int, "is_target": bool}}` for every slot with `visible_count>0`. Consensus per Global Constraints.
  - `n_targets(targets: dict) -> int` — count of `is_target`; `decidable(targets) -> bool` (`n_targets ≥ 2`).

- [ ] **Step 1: Write the failing test**

```python
# tests/mmkg/test_schema.py
from ragregen.mmkg import schema as s

def test_norm_and_slots():
    assert s.SLOTS[0] == "primary_color" and len(s.SLOTS) == 6
    assert s.norm(" Rough ") == "rough"
    assert "" in s.NOT_VISIBLE and "n/a" in s.NOT_VISIBLE

def test_target_attributes_consensus_and_decidable():
    # primary_color: red x3, blue x1 over 4 visible -> thr=max(2,ceil(2.0))=2 -> red target
    # surface_texture: rough x2 over 2 visible -> V_vis=2 (<3) -> NOT a target
    reads = [
        {"primary_color": "red",  "surface_texture": "rough"},
        {"primary_color": "Red ", "surface_texture": "rough"},
        {"primary_color": "red",  "surface_texture": "not visible"},
        {"primary_color": "blue", "surface_texture": "none"},
    ]
    t = s.target_attributes(reads)
    assert t["primary_color"]["is_target"] is True
    assert t["primary_color"]["value"] == "red" and t["primary_color"]["support"] == 3
    assert t["primary_color"]["visible_count"] == 4
    assert t["surface_texture"]["is_target"] is False   # V_vis=2 < 3
    assert s.n_targets(t) == 1 and s.decidable(t) is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. .../python -m pytest tests/mmkg/test_schema.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Write minimal implementation**

```python
# ragregen/mmkg/schema.py
from __future__ import annotations
import math
from collections import Counter

SLOTS = ["primary_color", "secondary_color", "pattern_or_markings",
         "surface_texture", "overall_shape_or_form", "distinctive_feature"]
NOT_VISIBLE = {"not visible", "none", "n/a", ""}

def norm(v) -> str:
    return str(v).strip().lower()

def target_attributes(reads):
    out = {}
    for slot in SLOTS:
        vals = [norm(r.get(slot)) for r in reads]
        vis = [v for v in vals if v not in NOT_VISIBLE]
        if not vis:
            continue
        cnt = Counter(vis)
        value, support = cnt.most_common(1)[0]
        thr = max(2, math.ceil(0.5 * len(vis)))
        out[slot] = {"value": value, "support": support,
                     "visible_count": len(vis), "is_target": support >= thr}
    return out

def n_targets(targets) -> int:
    return sum(1 for v in targets.values() if v["is_target"])

def decidable(targets) -> bool:
    return n_targets(targets) >= 2
```

- [ ] **Step 4: Run to verify it passes.** Run the file's tests → PASS.
- [ ] **Step 5: Commit**

```bash
git add ragregen/mmkg/__init__.py ragregen/mmkg/schema.py tests/mmkg/__init__.py tests/mmkg/test_schema.py
git commit -m "feat(mmkg): frozen slot schema + target-attribute consensus"
```

---

### Task 2: VLM read + bidirectional judge (behind a thin interface)

**Files:**
- Create: `ragregen/mmkg/vlm_read.py`
- Test: `tests/mmkg/test_vlm_read.py`

**Interfaces:**
- Consumes: `schema.SLOTS`, `schema.norm`, `schema.NOT_VISIBLE`; an `asker` object with `.ask(image, prompt)->str` (satisfied by `ragregen.vlm.QwenVLM`).
- Produces:
  - `READ_PROMPT: str`, `JUDGE_PROMPT: str` (frozen verbatim, Global Constraints).
  - `read_slots(image_path, asker) -> dict[str,str]` — opens image RGB, asks `READ_PROMPT`, parses the JSON object, returns `{slot: value}` for all 6 SLOTS (missing/garbage → `"not visible"`).
  - `judge_match(x, y, slot, asker) -> bool` — if `norm(x)∈NOT_VISIBLE or norm(y)∈NOT_VISIBLE` → False without asking; else ask both orders on a blank 8×8 white image, return `both startswith("yes")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/mmkg/test_vlm_read.py
from ragregen.mmkg import vlm_read as vr

class FakeAsker:
    def __init__(self, replies): self.replies = list(replies); self.calls = []
    def ask(self, image, prompt): self.calls.append((image, prompt)); return self.replies.pop(0)

def test_read_slots_parses_json_and_fills_missing(tmp_path):
    from PIL import Image
    p = tmp_path / "x.png"; Image.new("RGB", (8, 8), "white").save(p)
    reply = '{"primary_color":"Red","surface_texture":"rough"}'   # 4 slots missing
    out = vr.read_slots(str(p), FakeAsker([reply]))
    assert out["primary_color"] == "Red" and out["surface_texture"] == "rough"
    assert out["distinctive_feature"] == "not visible"            # missing key filled
    assert set(out) == set(vr.__import__("ragregen.mmkg.schema", fromlist=["SLOTS"]).SLOTS)

def test_judge_bidirectional_and_notvisible_shortcircuit():
    a = FakeAsker(["Yes", "yes"]); assert vr.judge_match("red","crimson","primary_color", a) is True
    a = FakeAsker(["yes", "no"]);  assert vr.judge_match("red","blue","primary_color", a) is False
    a = FakeAsker([]);             assert vr.judge_match("not visible","red","primary_color", a) is False
    assert a.calls == []                                          # no VLM call when a value is not-visible
```

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL (module missing).

- [ ] **Step 3: Write minimal implementation**

```python
# ragregen/mmkg/vlm_read.py
from __future__ import annotations
import json, re
from ragregen.mmkg.schema import SLOTS, NOT_VISIBLE, norm

READ_PROMPT = ('Describe ONLY the single main subject in this image. Do not name it. '
               'Return a JSON object with exactly these keys: primary_color, secondary_color, '
               'pattern_or_markings, surface_texture, overall_shape_or_form, distinctive_feature. '
               'Each value is a short phrase, or "not visible" if you cannot tell. Output only the JSON.')
JUDGE_PROMPT = ('You are comparing two descriptions of an object\'s {slot}. A: "{x}". B: "{y}". '
                "Do A and B describe essentially the same {slot}? Answer with only 'yes' or 'no'.")

def _parse_json(text):
    m = re.search(r"\{.*\}", str(text), re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def read_slots(image_path, asker):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    obj = _parse_json(asker.ask(img, READ_PROMPT))
    return {slot: (str(obj[slot]) if slot in obj and str(obj[slot]).strip() else "not visible")
            for slot in SLOTS}

def _yes(r): return str(r).strip().lower().startswith("yes")

def judge_match(x, y, slot, asker):
    if norm(x) in NOT_VISIBLE or norm(y) in NOT_VISIBLE:
        return False
    from PIL import Image
    blank = Image.new("RGB", (8, 8), "white")
    a = asker.ask(blank, JUDGE_PROMPT.format(slot=slot, x=x, y=y))
    b = asker.ask(blank, JUDGE_PROMPT.format(slot=slot, x=y, y=x))
    return _yes(a) and _yes(b)
```

- [ ] **Step 4: Run to verify it passes** → PASS.
- [ ] **Step 5: Commit**

```bash
git add ragregen/mmkg/vlm_read.py tests/mmkg/test_vlm_read.py
git commit -m "feat(mmkg): name-blind slot read + frozen bidirectional judge"
```

---

### Task 3: Role 2 — per-attribute states and case verdict (pure)

**Files:**
- Create: `ragregen/mmkg/verifier.py`
- Test: `tests/mmkg/test_verifier.py`

**Interfaces:**
- Consumes: `schema` (`norm`, `NOT_VISIBLE`, `decidable`); a `judge_fn(x, y, slot) -> bool`.
- Produces:
  - `attribute_states(targets, read, judge_fn) -> dict[str, dict]` — for each `is_target` slot: `{"target","read","present":bool,"match":bool|None,"contradicted":bool}`; `present=False` when `norm(read)∈NOT_VISIBLE` (then `match=None, contradicted=False`); else `match=judge_fn(target,read,slot)`, `contradicted = not match`.
  - `case_verdict(targets, read, judge_fn) -> dict` — `{"verdict":"FAIL"|"PASS"|"ABSTAIN","n_present":int,"n_contradicted":int,"decidable":bool,"per_attribute":{...}}`. ABSTAIN iff `not decidable(targets)` OR `n_present==0`; else FAIL iff `n_contradicted≥1` else PASS.

- [ ] **Step 1: Write the failing test**

```python
# tests/mmkg/test_verifier.py
from ragregen.mmkg import verifier as vf

def _targets(**kv):  # helper: mark given slots as targets
    return {s: {"value": v, "is_target": True} for s, v in kv.items()}

def test_fail_on_one_contradiction():
    targets = _targets(primary_color="red", surface_texture="rough")
    read = {"primary_color": "red", "surface_texture": "smooth"}
    judge = lambda x, y, slot: x == y                      # exact-match stand-in
    out = vf.case_verdict(targets, read, judge)
    assert out["decidable"] and out["n_present"] == 2 and out["n_contradicted"] == 1
    assert out["verdict"] == "FAIL"

def test_pass_when_all_present_consistent():
    targets = _targets(primary_color="red", surface_texture="rough")
    read = {"primary_color": "red", "surface_texture": "rough"}
    assert vf.case_verdict(targets, read, lambda x,y,s: x==y)["verdict"] == "PASS"

def test_abstain_when_undecidable_or_nothing_present():
    one = _targets(primary_color="red")                    # 1 target -> undecidable
    assert vf.case_verdict(one, {"primary_color":"red"}, lambda x,y,s:x==y)["verdict"] == "ABSTAIN"
    two = _targets(primary_color="red", surface_texture="rough")
    read = {"primary_color": "not visible", "surface_texture": "none"}   # nothing present
    v = vf.case_verdict(two, read, lambda x,y,s:x==y)
    assert v["n_present"] == 0 and v["verdict"] == "ABSTAIN"

def test_missing_never_contradicts():
    targets = _targets(primary_color="red", surface_texture="rough")
    read = {"primary_color": "red", "surface_texture": "not visible"}    # texture missing
    v = vf.case_verdict(targets, read, lambda x,y,s: False)               # judge would say mismatch if asked
    assert v["n_present"] == 1 and v["n_contradicted"] == 0 and v["verdict"] == "PASS"
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Write minimal implementation**

```python
# ragregen/mmkg/verifier.py
from __future__ import annotations
from ragregen.mmkg.schema import norm, NOT_VISIBLE, decidable

def attribute_states(targets, read, judge_fn):
    out = {}
    for slot, meta in targets.items():
        if not meta.get("is_target"):
            continue
        t = meta["value"]; r = read.get(slot, "not visible")
        if norm(r) in NOT_VISIBLE:
            out[slot] = {"target": t, "read": r, "present": False,
                         "match": None, "contradicted": False}
        else:
            m = bool(judge_fn(t, r, slot))
            out[slot] = {"target": t, "read": r, "present": True,
                         "match": m, "contradicted": not m}
    return out

def case_verdict(targets, read, judge_fn):
    states = attribute_states(targets, read, judge_fn)
    n_present = sum(1 for s in states.values() if s["present"])
    n_contra = sum(1 for s in states.values() if s["contradicted"])
    dec = decidable(targets)
    if not dec or n_present == 0:
        verdict = "ABSTAIN"
    elif n_contra >= 1:
        verdict = "FAIL"
    else:
        verdict = "PASS"
    return {"verdict": verdict, "n_present": n_present, "n_contradicted": n_contra,
            "decidable": dec, "per_attribute": states}
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit**

```bash
git add ragregen/mmkg/verifier.py tests/mmkg/test_verifier.py
git commit -m "feat(mmkg): Role 2 case-level identity verdict (frozen rules)"
```

---

### Task 4: Role 1 — flat vs MMKG-organized retrieval, divergence + coverage (pure)

**Files:**
- Create: `ragregen/mmkg/retrieval_arms.py`
- Test: `tests/mmkg/test_retrieval_arms.py`

**Interfaces:**
- Consumes: nothing from prior tasks (operates on plain dicts of retrieval hits + MMKG instances).
- Produces:
  - `flat_arm(hits) -> dict` — `hits` = list of `{"path","score","rank"}` (from `retrieval.json.cases[c]["hits"]`); returns `{"candidate_pool": hits_sorted_by_rank, "selected_ref": top_by_score}`.
  - `mmkg_arm(instances, targets) -> dict` — `instances` = list of `{"path","score","part_type"|None}` (deduped, quality-filtered slice with per-slot medoid tag); selection = one medoid per target-attribute slot (coverage-aware), reported as a `selected_set` plus `selected_ref` = highest-score member; returns `{"candidate_pool":instances,"selected_set":[...],"selected_ref":...}`.
  - `divergence(flat, mmkg) -> dict` — `{"jaccard": float, "n_flat":int,"n_mmkg":int,"selected_ref_identical":bool}` (Jaccard over candidate-pool paths).
  - `coverage(flat, mmkg) -> dict` — `{"flat_unit_size":1,"mmkg_unit_size":int,"mmkg_part_types":int,"flat_top1_in_mmkg_pool":bool}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/mmkg/test_retrieval_arms.py
from ragregen.mmkg import retrieval_arms as ra

def test_flat_and_mmkg_arms_and_divergence():
    hits = [{"path": "a.jpg", "score": 0.9, "rank": 0},
            {"path": "b.jpg", "score": 0.5, "rank": 1}]
    flat = ra.flat_arm(hits)
    assert flat["selected_ref"]["path"] == "a.jpg"                # top by score
    inst = [{"path": "b.jpg", "score": 0.5, "part_type": "body"},
            {"path": "c.jpg", "score": 0.7, "part_type": "head"}]
    targets = {"primary_color": {"is_target": True}, "surface_texture": {"is_target": True}}
    mmkg = ra.mmkg_arm(inst, targets)
    assert mmkg["selected_ref"]["path"] == "c.jpg"               # highest score in pool
    d = ra.divergence(flat, mmkg)
    assert d["n_flat"] == 2 and d["n_mmkg"] == 2
    assert abs(d["jaccard"] - 1/3) < 1e-9                        # {a,b} vs {b,c} -> 1/3
    assert d["selected_ref_identical"] is False
    cov = ra.coverage(flat, mmkg)
    assert cov["flat_unit_size"] == 1 and cov["mmkg_part_types"] == 2
    assert cov["flat_top1_in_mmkg_pool"] is False               # a.jpg not in {b,c}
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Write minimal implementation**

```python
# ragregen/mmkg/retrieval_arms.py
from __future__ import annotations

def flat_arm(hits):
    pool = sorted(hits, key=lambda h: h.get("rank", 0))
    sel = max(hits, key=lambda h: h["score"]) if hits else None
    return {"candidate_pool": pool, "selected_ref": sel}

def mmkg_arm(instances, targets):
    # coverage-aware set: one representative (max score) per distinct part_type present
    by_pt = {}
    for it in instances:
        pt = it.get("part_type")
        if pt not in by_pt or it["score"] > by_pt[pt]["score"]:
            by_pt[pt] = it
    selected_set = list(by_pt.values())
    sel = max(instances, key=lambda h: h["score"]) if instances else None
    return {"candidate_pool": instances, "selected_set": selected_set, "selected_ref": sel}

def _paths(arm):
    return {h["path"] for h in arm["candidate_pool"]}

def divergence(flat, mmkg):
    pf, pm = _paths(flat), _paths(mmkg)
    union = pf | pm
    jac = (len(pf & pm) / len(union)) if union else 0.0
    fi = (flat["selected_ref"] or {}).get("path")
    mi = (mmkg["selected_ref"] or {}).get("path")
    return {"jaccard": jac, "n_flat": len(pf), "n_mmkg": len(pm),
            "selected_ref_identical": fi is not None and fi == mi}

def coverage(flat, mmkg):
    pts = {it.get("part_type") for it in mmkg["candidate_pool"] if it.get("part_type") is not None}
    fi = (flat["selected_ref"] or {}).get("path")
    return {"flat_unit_size": 1, "mmkg_unit_size": len(mmkg.get("selected_set", [])),
            "mmkg_part_types": len(pts), "flat_top1_in_mmkg_pool": fi in _paths(mmkg)}
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit**

```bash
git add ragregen/mmkg/retrieval_arms.py tests/mmkg/test_retrieval_arms.py
git commit -m "feat(mmkg): Role 1 flat vs organized retrieval divergence + coverage"
```

---

### Task 5: Offline evaluation — confusion matrix, gates, cross-tab, stratification (pure)

**Files:**
- Create: `ragregen/mmkg/evaluate.py`
- Test: `tests/mmkg/test_evaluate.py`

**Interfaces:**
- Consumes: nothing from prior tasks (operates on per-case dicts).
- Produces:
  - `confusion(cases) -> dict` — `cases` = list of `{"verdict":FAIL|PASS|ABSTAIN,"label":fail|pass}`; ABSTAIN→PASS; returns `{"tp","fp","tn","fn","recall","fp_rate","mcc","accuracy","n_abstain","n"}`. `recall` over label==fail; `fp_rate` over label==pass.
  - `gate(conf) -> dict` — `{"recall_ok":bool,"fp_ok":bool,"decision":"PASS"|"NEGATIVE"}` per frozen gates (recall≥0.75, fp_rate≤0.083).
  - `crosstab(cases) -> dict` — each case also has `semantic_ok:bool`; 2×2 of (mmkg FAIL vs semantic-routed) on label==fail, to show unique catches.
  - `stratify(cases, key) -> dict[str, dict]` — group cases by `case[key]` (e.g. `"n_target_attributes"` bucket or `"cohort"`), `confusion` per group.

- [ ] **Step 1: Write the failing test**

```python
# tests/mmkg/test_evaluate.py
import math
from ragregen.mmkg import evaluate as ev

def _mk(n_fail_caught, n_fail, n_fp, n_pass):
    cases = []
    for i in range(n_fail):
        cases.append({"verdict": "FAIL" if i < n_fail_caught else "PASS", "label": "fail"})
    for i in range(n_pass):
        cases.append({"verdict": "FAIL" if i < n_fp else "PASS", "label": "pass"})
    return cases

def test_confusion_and_gate_pass():
    conf = ev.confusion(_mk(20, 24, 1, 24))            # recall 20/24=.833, fp 1/24=.042
    assert conf["tp"] == 20 and conf["fn"] == 4 and conf["fp"] == 1 and conf["tn"] == 23
    assert abs(conf["recall"] - 20/24) < 1e-9 and abs(conf["fp_rate"] - 1/24) < 1e-9
    g = ev.gate(conf); assert g["recall_ok"] and g["fp_ok"] and g["decision"] == "PASS"

def test_gate_negative_and_abstain_counts_as_pass():
    cases = _mk(12, 24, 4, 24)                          # the detector's numbers: recall .50, fp .167
    # turn 3 missed fails into ABSTAIN -> still count as PASS (miss)
    misses = [c for c in cases if c["label"] == "fail" and c["verdict"] == "PASS"][:3]
    for c in misses: c["verdict"] = "ABSTAIN"
    conf = ev.confusion(cases)
    assert conf["n_abstain"] == 3 and conf["tp"] == 12       # abstain not a catch
    assert ev.gate(conf)["decision"] == "NEGATIVE"
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Write minimal implementation**

```python
# ragregen/mmkg/evaluate.py
from __future__ import annotations
import math
from collections import defaultdict

def _pred_fail(v): return v == "FAIL"           # ABSTAIN and PASS both -> not routed

def confusion(cases):
    tp = fp = tn = fn = nab = 0
    for c in cases:
        if c["verdict"] == "ABSTAIN": nab += 1
        pred, lab = _pred_fail(c["verdict"]), (c["label"] == "fail")
        if lab and pred: tp += 1
        elif lab and not pred: fn += 1
        elif not lab and pred: fp += 1
        else: tn += 1
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fp_rate = fp / (fp + tn) if (fp + tn) else float("nan")
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = ((tp*tn - fp*fn) / denom) if denom else 0.0
    n = tp + fp + tn + fn
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "recall": recall, "fp_rate": fp_rate,
            "mcc": mcc, "accuracy": (tp+tn)/n if n else float("nan"), "n_abstain": nab, "n": n}

def gate(conf):
    r = conf["recall"] >= 0.75
    f = conf["fp_rate"] <= 0.083
    return {"recall_ok": bool(r), "fp_ok": bool(f),
            "decision": "PASS" if (r and f) else "NEGATIVE"}

def crosstab(cases):
    both = mmkg_only = sem_only = neither = 0
    for c in cases:
        if c["label"] != "fail": continue
        m = _pred_fail(c["verdict"]); srt = not bool(c.get("semantic_ok", True))  # semantic routes on NOT ok
        if m and srt: both += 1
        elif m and not srt: mmkg_only += 1
        elif not m and srt: sem_only += 1
        else: neither += 1
    return {"both": both, "mmkg_only_catch": mmkg_only, "semantic_only_catch": sem_only,
            "neither": neither}

def stratify(cases, key):
    groups = defaultdict(list)
    for c in cases:
        groups[c.get(key)].append(c)
    return {str(k): confusion(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit**

```bash
git add ragregen/mmkg/evaluate.py tests/mmkg/test_evaluate.py
git commit -m "feat(mmkg): offline confusion/gates/cross-tab/stratification"
```

---

### Task 6: Trace record assembly + IO (with provenance)

**Files:**
- Create: `ragregen/mmkg/trace.py`
- Test: `tests/mmkg/test_trace.py`

**Interfaces:**
- Consumes: `ragregen.eval_manifest.sha256_file`.
- Produces:
  - `build_trace_record(case_id, concept, cohort, label, draft, retrieval, mmkg_concept, attribute_reads, verifier, scores_reused, provenance) -> dict` — assembles the §3 schema shape; validates required keys present (raises `ValueError` on a missing top-level section).
  - `write_trace(record, out_dir) -> str` — writes `<out_dir>/trace/<case_id>.json` (pretty), returns path.
  - `load_labels(csv_path) -> dict[str, str]` — parse `labels_holdout48.csv` → `{case_id: verdict_identity}` (lowercased; blank → raise).

- [ ] **Step 1: Write the failing test**

```python
# tests/mmkg/test_trace.py
import json
from ragregen.mmkg import trace as tr

def test_build_and_write_roundtrip(tmp_path):
    rec = tr.build_trace_record(
        "agaric", "agaric", "rare", "fail",
        draft={"image_path": "d.png", "prompt": "a agaric"},
        retrieval={"flat": {}, "mmkg": {}, "pool_overlap": {}},
        mmkg_concept={"target_attributes": {}, "n_target_attributes": 0},
        attribute_reads={"draft": {}},
        verifier={"verdict": "ABSTAIN"},
        scores_reused={"semantic_ok": True},
        provenance={"vlm_id": "Qwen2.5-VL"})
    p = tr.write_trace(rec, str(tmp_path))
    back = json.load(open(p))
    assert back["concept"] == "agaric" and back["verifier"]["verdict"] == "ABSTAIN"

def test_build_trace_record_requires_sections():
    import pytest
    with pytest.raises(ValueError):
        tr.build_trace_record("c","c","rare","fail", draft={}, retrieval={},
                              mmkg_concept={}, attribute_reads={}, verifier={},
                              scores_reused={}, provenance={})  # empty draft/etc -> missing keys

def test_load_labels(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("case_id,prompt,concept,draft_path,verdict,verdict_identity,notes\n"
                 "agaric,x,agaric,d.png,,fail,\n")
    assert tr.load_labels(str(p)) == {"agaric": "fail"}
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Write minimal implementation**

```python
# ragregen/mmkg/trace.py
from __future__ import annotations
import csv, json, os

def build_trace_record(case_id, concept, cohort, label, *, draft, retrieval,
                       mmkg_concept, attribute_reads, verifier, scores_reused, provenance):
    if not (draft.get("image_path") and "flat" in retrieval and "target_attributes" in mmkg_concept
            and "draft" in attribute_reads and verifier.get("verdict")):
        raise ValueError("trace record missing required sections/keys")
    return {"case_id": case_id, "concept": concept, "cohort": cohort, "human_label": label,
            "draft": draft, "retrieval": retrieval, "mmkg_concept": mmkg_concept,
            "attribute_reads": attribute_reads, "verifier": verifier,
            "scores_reused": scores_reused, "provenance": provenance}

def write_trace(record, out_dir):
    d = os.path.join(out_dir, "trace"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{record['case_id']}.json")
    json.dump(record, open(p, "w"), indent=2)
    return p

def load_labels(csv_path):
    out = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            v = (row.get("verdict_identity") or "").strip().lower()
            if v not in {"fail", "pass"}:
                raise ValueError(f"{row.get('case_id')}: bad verdict_identity {v!r}")
            out[row["case_id"]] = v
    return out
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit**

```bash
git add ragregen/mmkg/trace.py tests/mmkg/test_trace.py
git commit -m "feat(mmkg): trace record assembly, IO, and label loader"
```

---

### Task 7: Build MMKGs — larger-K slice + contamination + feasibility pre-check

**Files:**
- Create: `ragregen/mmkg/build.py`
- Test: `tests/mmkg/test_build.py` (feasibility logic only; the live slice+VLM build is a script step)

**Interfaces:**
- Consumes: `schema.target_attributes/decidable`, `ragregen.retrieve.Retriever`, `ragregen.retrieved_reference.contamination`, `ragregen.eval_manifest.sha256_file`.
- Produces:
  - `dedup_and_guard(paths, gt_ref_paths) -> list[str]` — drop sha256 duplicates and any image `contamination(path, gt_ref_paths)` flags; preserve order.
  - `feasibility(concept_reads: dict[str, list[dict]]) -> dict` — per concept run `target_attributes`; `{"decidable_concepts":int,"undecidable":[names],"by_concept":{c:n_targets}}`.
  - `build_concept(concept, query, index_slice_paths, gt_ref_paths, read_fn) -> dict` — dedup+guard slice, `read_fn(path)->{slot:val}` over it, `target_attributes`, return `{"built_from_slice","slice_size","contamination_excluded","target_attributes","n_target_attributes"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/mmkg/test_build.py
from ragregen.mmkg import build as b

def test_feasibility_counts_decidable():
    reads = {
        # decidable: 2 slots each with 3 visible agreeing
        "A": [{"primary_color":"red","surface_texture":"rough"}]*3,
        # undecidable: only 1 slot reliable
        "B": [{"primary_color":"red"}]*3 + [{"surface_texture":"x"}],
    }
    out = b.feasibility(reads)
    assert out["by_concept"]["A"] == 2 and out["by_concept"]["B"] == 1
    assert out["decidable_concepts"] == 1 and out["undecidable"] == ["B"]

def test_build_concept_uses_readfn_and_targets():
    reads = {"s1.jpg": {"primary_color":"red","surface_texture":"rough"},
             "s2.jpg": {"primary_color":"red","surface_texture":"rough"},
             "s3.jpg": {"primary_color":"red","surface_texture":"smooth"}}
    rec = b.build_concept("A", "q", ["s1.jpg","s2.jpg","s3.jpg"], gt_ref_paths=[],
                          read_fn=lambda p: reads[p])
    assert rec["slice_size"] == 3
    assert rec["target_attributes"]["primary_color"]["is_target"] is True
    assert rec["n_target_attributes"] >= 1
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Write minimal implementation** (feasibility + build_concept pure over `read_fn`; `dedup_and_guard` uses project helpers)

```python
# ragregen/mmkg/build.py
from __future__ import annotations
from ragregen.mmkg.schema import target_attributes, n_targets, decidable
from ragregen.retrieved_reference import contamination
from ragregen.eval_manifest import sha256_file

def dedup_and_guard(paths, gt_ref_paths):
    seen, out = set(), []
    for p in paths:
        try:
            h = sha256_file(p)
        except Exception:
            continue
        if h in seen or contamination(p, gt_ref_paths):
            continue
        seen.add(h); out.append(p)
    return out

def feasibility(concept_reads):
    by = {c: n_targets(target_attributes(rs)) for c, rs in concept_reads.items()}
    undec = sorted([c for c, n in by.items() if n < 2])
    return {"decidable_concepts": sum(1 for n in by.values() if n >= 2),
            "undecidable": undec, "by_concept": by}

def build_concept(concept, query, index_slice_paths, gt_ref_paths, read_fn):
    kept = dedup_and_guard(index_slice_paths, gt_ref_paths)
    excluded = [p for p in index_slice_paths if p not in set(kept)]
    reads = [read_fn(p) for p in kept]
    tgt = target_attributes(reads)
    return {"concept": concept, "query": query, "built_from_slice": kept,
            "slice_size": len(kept), "contamination_excluded": excluded,
            "target_attributes": tgt, "n_target_attributes": n_targets(tgt),
            "decidable": decidable(tgt)}
```

- [ ] **Step 4: Run → PASS** the feasibility/build tests. Full suite: `PYTHONPATH=. .../python -m pytest -q`.
- [ ] **Step 5: Commit**

```bash
git add ragregen/mmkg/build.py tests/mmkg/test_build.py
git commit -m "feat(mmkg): slice dedup+contamination guard, feasibility, concept build"
```

---

### Task 8: Orchestrator `run_stage1` + live run + result report

**Files:**
- Create: `ragregen/mmkg/run_stage1.py`, `scripts/mmkg_stage1.sh`
- Create (run artifacts): `outputs/mmkg_stage1/` (trace/, result.json), `docs/superpowers/2026-09-09-mmkg-stage1-finding.md`

**Interfaces:**
- Consumes: everything above; `ragregen.vlm.QwenVLM`; `ragregen.retrieve.Retriever`; holdout artifacts under `outputs/phase_b/`.
- Produces: `main()` — the single batch + offline eval + report.

Flow (documented in the module docstring and `main`):
1. Load `outputs/phase_b/holdout.json` (concept lists per cohort), `retrieval.json` (per-concept stored `query` + gt reference paths), `labels_holdout48.csv` (via `trace.load_labels`), `retrieved_reranker.json`/semantic verdicts + `dino_reserves.json` (for `scores_reused` cross-tab only — never as ground truth).
2. **Slice + feasibility (CPU, no VLM yet):** for each concept, reuse the stored `query` verbatim; `Retriever.from_index(data/laion100k/index.faiss, encoder).search(query, K_SLICE=40)`; dedup+guard. Do a **proxy feasibility** by reading slots on a *small* subset? No — feasibility needs reads. Instead: after the read pass (step 3) compute `build.feasibility`; **but** first log slice sizes and, if any concept's kept slice `< 3`, record it. (The true decidable count is known only post-read; report it, per spec — do not silently under-power.)
3. **VLM read pass (one GPU load):** instantiate `QwenVLM`; for each concept, `vlm_read.read_slots` over its kept slice → `build.build_concept`. Then read the **draft** of each case once. Persist all reads.
4. **Retrieval arms (CPU):** `retrieval_arms.flat_arm(retrieval hits)` and `mmkg_arm(mmkg instances, targets)`; `divergence`, `coverage`.
5. **Verdict (CPU, judge is VLM):** for each case, `verifier.case_verdict(targets, draft_read, judge_fn)` where `judge_fn = lambda x,y,slot: vlm_read.judge_match(x,y,slot,vlm)`.
6. **Trace:** `trace.build_trace_record` + `write_trace` per case (provenance = sha256 of each reused source + `VLM_ID`).
7. **Offline eval (CPU):** assemble `{"verdict","label","semantic_ok","cohort","n_target_attributes"}` per case → `evaluate.confusion/gate/crosstab/stratify` (stratify by cohort and by `n_target_attributes` bucket). Write `outputs/mmkg_stage1/result.json` (Role 1 divergence/coverage aggregates + Role 2 confusion/gate/crosstab/strata + decidable count + abstain count).
8. **Report** `docs/superpowers/2026-09-09-mmkg-stage1-finding.md`: the frozen gates, the numbers, PASS/NEGATIVE decision, the cross-tab vs the semantic branch's 12 misses, the decidable/abstain accounting, Role 1 divergence/coverage, and the VLM-as-detector/human-as-truth limitation note.

- [ ] **Step 1: Write `run_stage1.py` and `scripts/mmkg_stage1.sh`** (wire the flow above; keep all thresholds imported from the frozen modules — no re-declared constants).
- [ ] **Step 2: Dry-run the offline path on a tiny fake** — add `tests/mmkg/test_run_smoke.py` that imports `run_stage1`, monkeypatches the VLM asker + `Retriever` with fakes over 2 concepts / 4 cases, runs `main(argv=[...])`, and asserts `result.json` + one trace file are written and `gate` keys exist. Run: `PYTHONPATH=. .../python -m pytest tests/mmkg/test_run_smoke.py -q` → PASS. Commit.
- [ ] **Step 3: Feasibility gate (live, CPU + one VLM load)** — run the slice + read pass; inspect `result.json`'s decidable count. If `decidable_concepts` is low enough that recall cannot reach 18/24 even at perfect precision (`decidable rare < 18`), **stop and report the data limit** (like the GRAFT OracleHub/Stage-B blockers) rather than reporting an under-powered gate.
- [ ] **Step 4: Full live run** (detached; mirror GRAFT's VLM-load memory guard):
`setsid .../python -m ragregen.mmkg.run_stage1 --out outputs/mmkg_stage1 > outputs/mmkg_stage1/run.log 2>&1 < /dev/null &`
Watch for `result.json`; confirm no traceback.
- [ ] **Step 5: Write the finding report** from `result.json` (numbers, decision, cross-tab, limitations), then update memory (`mmkg-to-ragregen-direction`) + the SDD ledger with the outcome. Commit report.

```bash
git add ragregen/mmkg/run_stage1.py scripts/mmkg_stage1.sh tests/mmkg/test_run_smoke.py docs/superpowers/2026-09-09-mmkg-stage1-finding.md
git commit -m "feat(mmkg): Stage-1 orchestrator, one-batch trace, offline gates + finding"
```

---

## Self-Review

**Spec coverage:** §1 two roles/one trace → Tasks 3–6 + 8. §2 frozen defs → Global Constraints + Tasks 1/2. §2 reference parity + contamination → Task 7 (`dedup_and_guard`) + Task 8 step 2. §3 trace schema → Task 6 + Task 8 step 6. §4 build + concept-general slots + consensus → Tasks 1, 2, 7. §5 Role 1 intervention/divergence/coverage → Task 4 + Task 8 step 4. §6 verdict + gates + cross-tab → Tasks 3, 5 + Task 8 steps 5,7. §7 one batch + offline + stratification + re-run-not-splice → Task 8 (reads holdout read-only; writes new `outputs/mmkg_stage1/`). §9 success/negative → Task 5 `gate` + Task 8 report. §10 open items (VLM entry, slice feasibility, reuse boundary, coarse pool) → resolved: `QwenVLM.ask`/`ask_images` (Task 2), larger-K slice + feasibility (Tasks 7–8), logic reimplemented in-repo not cross-imported (all tasks), part_type-optional coverage (Task 4). **No gaps.**

**Placeholder scan:** none — every code step has concrete content. Task 8 steps 1/4/5 are live-run/report actions (no unit test asserted for the GPU pass by design), each with an exact command; the offline path is smoke-tested in step 2.

**Type consistency:** `target_attributes` output shape (`{slot:{value,support,visible_count,is_target}}`) is produced in Task 1 and consumed by Tasks 3 (`meta["value"]`,`is_target`), 4 (`is_target`), 7 (`n_targets`). `case_verdict` return keys (`verdict`,`n_present`,`n_contradicted`,`decidable`,`per_attribute`) match Task 8 step 5 and Task 5's `{"verdict","label"}` consumer. `judge_fn(x,y,slot)->bool` signature matches `judge_match(x,y,slot,asker)` via the Task 8 lambda. `flat_arm`/`mmkg_arm` output keys (`candidate_pool`,`selected_ref`,`selected_set`) match `divergence`/`coverage` consumers. `confusion`→`gate` keys (`recall`,`fp_rate`) consistent. Trace `build_trace_record` required keys match the sections Task 8 assembles.
