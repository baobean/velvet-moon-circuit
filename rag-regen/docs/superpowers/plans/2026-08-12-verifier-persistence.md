# Verifier Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pipeline persist the numbers behind its verdicts, stop the fused score from fabricating a value it does not have, and re-grade the 92 finished doc-5 cases without regenerating an image.

**Architecture:** A single serialiser in `ragregen/verify/record.py` becomes the only way a `ConceptScore` becomes JSON, adopted by both `scripts/score_a.py` (unchanged output) and `scripts/run_pipeline.py` (which currently persists only the state string). `fusion.fuse` returns `score: float | None` so an absent Stream A reads as absent rather than as 1.0. The already-written-but-never-constructed `ragregen/trace.py` gets wired. A new `scripts/reverify.py` re-runs the two verify stages over images already on disk, into a new run directory that never writes to the source.

**Tech Stack:** Python 3.11, numpy, transformers 5.14.1 (GroundingDINO, SigLIP-SO400M-384, Qwen-VL nf4), pytest. No new dependencies.

## Global Constraints

- `$PY=/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python`. **Never bare `python`** — it is conda 3.13 and has no faiss.
- **The 520 existing tests must stay green.** Run `$PY -m pytest -q`. The suite sometimes dumps core *after* printing its summary — that is the known native-imaging shutdown flake, not a failure.
- **`scripts/score_a.py`'s output must not change by one byte.** `c1.py` consumes `stream_a.json` and the pinned τ = 0.25 baseline must keep reproducing. Task 1 pins this with a golden fixture.
- **`grounded.py`'s live state logic stays byte-for-byte unchanged.** Prototype scores are recorded, never acted on in-process (`grounded.py:87`).
- **Both new flags default off.** A resumed doc-5 campaign must behave identically to today.
- **Back-compat is mandatory in two places:** `streams.json` holding bare state strings, and `queue.json` at `schema: 1`. `--resume` is the guarantee doc 3 exists to make.
- **Never write to `outputs/doc5_20260810_183939/`.** It is evidence. Operator standing rule, 2026-08-10.
- `git add <exact paths>` — never `git add -A`. `outputs/` is gitignored.
- Dependency injection throughout. No module-level model loading, no network at import.
- Branch: cut `fix-verifier-persistence` from `doc5-laion-benchmark` **after** Task 0 commits the pre-existing working-tree fixes.

---

## File Structure

| file | responsibility | task |
|---|---|---|
| `ragregen/verify/record.py` | **create** — the only `ConceptScore` ⇄ JSON converter | 1 |
| `scripts/score_a.py:30-59` | modify — delegate to `record.to_dict(with_state=False)` | 1 |
| `ragregen/verify/fusion.py:44-76` | modify — `score` becomes `float \| None` | 2 |
| `ragregen/schedule.py` (`select_best`, `CaseState`, `SCHEMA`) | modify — skip `None` scores; `healthy` status | 2, 6 |
| `scripts/run_pipeline.py:311-343` | modify — full persistence, back-compat read | 3 |
| `scripts/run_pipeline.py:280-308` | modify — `CaseTrace`, Stream B `issues`/`raw` | 4 |
| `scripts/run_pipeline.py` (arg parsing, `_verify_round`, `_mask_one`) | modify — prototype flags, `mask.json` | 5 |
| `scripts/reverify.py` | **create** — re-verify a finished run into a new run dir | 7 |
| `docs/RUNBOOK.md`, `configs/retrieval_db.yaml` | modify — the query-step lie, CWD-relative index path | 8 |

---

### Task 0: Commit the pre-existing working tree, then branch

**Files:** no code changes — this task only lands what is already on disk.

- [ ] **Step 1: Confirm the suite is green before touching anything**

Run: `$PY -m pytest -q 2>&1 | tail -5`
Expected: `520 passed, 11 deselected` (a core dump *after* that line is the known flake).

- [ ] **Step 2: Commit the campaign fixes already in the tree**

```bash
git add ragregen/encoders.py ragregen/env.py scripts/build_index.py scripts/campaign.sh \
        scripts/repair_case.py scripts/report.py scripts/run_pipeline.py \
        scripts/screen_premise.py scripts/supervise.sh \
        tests/test_encoders.py tests/test_env.py tests/test_report.py
git commit -m "fix: the campaign's own repairs, committed before new work lands on top"
git add configs/dataset_common.yaml docs/RUN_ON_SECOND_MACHINE.md scripts/gpu_wait.sh \
        tests/test_campaign_lock.py tests/test_campaign_run_dir.py tests/test_gpu_wait.py
git commit -m "feat: the untracked files the campaign could not run without"
```

- [ ] **Step 3: Cut the branch**

```bash
git checkout -b fix-verifier-persistence
git status --short   # expected: clean except docs/run_when_free_gpu.txt
```

- [ ] **Step 4: Commit**

Nothing further to commit; the branch point is the deliverable.

---

### Task 1: One serialiser, owned by the package

**Files:**
- Create: `ragregen/verify/record.py`
- Create: `tests/test_record.py`
- Modify: `scripts/score_a.py:30-59`

**Interfaces:**
- Consumes: `ragregen.verify.grounded.ConceptScore`, `CandidateScore`
- Produces: `record.to_dict(score, *, with_state) -> dict`, `record.from_dict(d) -> ConceptScore`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_record.py
import pytest

from ragregen.verify import record
from ragregen.verify.grounded import CandidateScore, ConceptScore


def _full() -> ConceptScore:
    cand = CandidateScore(box=(1.0, 2.0, 3.0, 4.0), dino_conf=0.9, sim=0.7,
                          sim_coarse=0.2, sim_proto=0.55, sim_proto_coarse=0.11)
    return ConceptScore("Boston bull", "subject", "PRESENT",
                        box=(1.0, 2.0, 3.0, 4.0), dino_conf=0.9, sim=0.7,
                        sim_coarse=0.2, sim_proto=0.55, sim_proto_coarse=0.11,
                        candidates=(cand,))


def test_round_trip_preserves_every_field():
    s = _full()
    assert record.from_dict(record.to_dict(s, with_state=True)) == s


def test_round_trip_preserves_nones_and_empty_candidates():
    s = ConceptScore("three", "count", "ABSTAIN")
    assert record.from_dict(record.to_dict(s, with_state=True)) == s


def test_with_state_false_omits_state_only():
    d = record.to_dict(_full(), with_state=False)
    assert "state" not in d
    assert d["sim"] == 0.7 and d["kind"] == "subject"


def test_a_bare_state_string_rehydrates_as_the_lossy_old_shape():
    """Old streams.json holds {'phrase': 'MISSING'}. A resume must not die."""
    s = record.from_dict("MISSING")
    assert s.state == "MISSING"
    assert s.sim is None and s.candidates == ()


def test_from_dict_rejects_an_unknown_state():
    with pytest.raises(ValueError, match="not a verifier state"):
        record.from_dict({"state": "BROKEN", "phrase": "x", "kind": "subject"})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_record.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragregen.verify.record'`

- [ ] **Step 3: Write the implementation**

```python
# ragregen/verify/record.py
"""The one way a ConceptScore becomes JSON, and comes back.

Two callers persisted Stream A independently and discarded opposite halves:
score_a.py kept every similarity and no verdict, run_pipeline.py kept the
verdict and no similarity. The pipeline's half is why 202 verdicts of GPU time
cannot be re-graded at any other tau. One serialiser, two callers, no drift.
"""
from __future__ import annotations

from ragregen.verify.grounded import STATES, CandidateScore, ConceptScore


def _cand_to_dict(c: CandidateScore) -> dict:
    return {"box": list(c.box), "dino_conf": c.dino_conf, "sim": c.sim,
            "sim_coarse": c.sim_coarse, "sim_proto": c.sim_proto,
            "sim_proto_coarse": c.sim_proto_coarse}


def _cand_from_dict(d: dict) -> CandidateScore:
    return CandidateScore(box=tuple(d["box"]), dino_conf=d["dino_conf"],
                          sim=d["sim"], sim_coarse=d.get("sim_coarse"),
                          sim_proto=d.get("sim_proto"),
                          sim_proto_coarse=d.get("sim_proto_coarse"))


def to_dict(score: ConceptScore, *, with_state: bool) -> dict:
    """Serialise one concept.

    `with_state=False` is score_a.py's contract: its tau is arbitrary, so a
    verdict in that file would be a number nobody should trust. The pipeline
    passes True because the scheduler resumes on verdicts.
    """
    out = {
        "phrase": score.phrase,
        "kind": score.kind,
        "sim": score.sim,
        "sim_coarse": score.sim_coarse,
        "sim_proto": score.sim_proto,
        "sim_proto_coarse": score.sim_proto_coarse,
        "box": list(score.box) if score.box is not None else None,
        "dino_conf": score.dino_conf,
        "candidates": [_cand_to_dict(c) for c in score.candidates],
    }
    if with_state:
        out["state"] = score.state
    return out


def from_dict(d: dict | str) -> ConceptScore:
    """Rehydrate. A bare string is an old streams.json entry, not an error.

    Runs produced before 2026-08-12 persisted `{phrase: "MISSING"}` and nothing
    else. Those directories must stay resumable, so a string rehydrates to the
    lossy score it honestly is.
    """
    if isinstance(d, str):
        if d not in STATES:
            raise ValueError(f"{d!r} is not a verifier state")
        return ConceptScore("", "subject", d)

    state = d.get("state", "PRESENT")
    if state not in STATES:
        raise ValueError(f"{state!r} is not a verifier state")
    box = d.get("box")
    return ConceptScore(
        d.get("phrase", ""), d.get("kind", "subject"), state,
        box=tuple(box) if box is not None else None,
        dino_conf=d.get("dino_conf"), sim=d.get("sim"),
        sim_coarse=d.get("sim_coarse"), sim_proto=d.get("sim_proto"),
        sim_proto_coarse=d.get("sim_proto_coarse"),
        candidates=tuple(_cand_from_dict(c) for c in d.get("candidates", ())),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_record.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Capture score_a's current output as a golden fixture**

Run:
```bash
$PY -c "
import json, shutil
src = 'outputs/screen_20260726_233320/stream_a.json'
shutil.copy(src, 'tests/fixtures/stream_a_golden.json')
print(len(json.load(open(src))), 'cases captured')
"
```
Expected: `22 cases captured`. Create `tests/fixtures/` first if absent.

- [ ] **Step 6: Write the failing golden test**

```python
# append to tests/test_record.py
import json
from pathlib import Path

from ragregen.verify import record as _record


def test_score_a_serialisation_is_unchanged(tmp_path):
    """The tau = 0.25 C1 baseline reproduces only if this file's shape holds."""
    golden = json.loads(
        Path("tests/fixtures/stream_a_golden.json").read_text())
    case = golden["boston_bull"]
    phrase, payload = next(iter(case.items()))
    rebuilt = _record.to_dict(_record.from_dict({**payload, "phrase": phrase}),
                              with_state=False)
    assert set(rebuilt) == set(payload) | {"phrase"}
    for k in payload:
        assert rebuilt[k] == payload[k], k
```

- [ ] **Step 7: Run it, then make `score_a.py` delegate**

Run: `$PY -m pytest tests/test_record.py::test_score_a_serialisation_is_unchanged -q`

Replace the dict comprehension body of `score_case` (`scripts/score_a.py:41-59`) with:

```python
    parsed = concepts.parse(prompt, target=concept, coarse=coarse)
    scores = verifier.score(image, parsed)
    return {phrase: {k: v for k, v in record.to_dict(s, with_state=False).items()
                     if k != "phrase"}
            for phrase, s in scores.items()}
```

and add `from ragregen.verify import record` to its imports. The `phrase` key is
stripped because `stream_a.json` already keys by phrase; adding it inside would
change the file.

- [ ] **Step 8: Verify byte-identical output on real data**

Run:
```bash
cp outputs/screen_20260726_233320/stream_a.json /tmp/claude-1004/stream_a_before.json
$PY -m pytest tests/test_record.py -q
diff <($PY -c "import json;print(json.dumps(json.load(open('/tmp/claude-1004/stream_a_before.json')),indent=2,sort_keys=True))") \
     <($PY -c "import json;print(json.dumps(json.load(open('tests/fixtures/stream_a_golden.json')),indent=2,sort_keys=True))") && echo IDENTICAL
```
Expected: `IDENTICAL`

- [ ] **Step 9: Full suite, then commit**

Run: `$PY -m pytest -q 2>&1 | tail -3`
Expected: `526 passed` (520 + 6 new)

```bash
git add ragregen/verify/record.py tests/test_record.py tests/fixtures/stream_a_golden.json scripts/score_a.py
git commit -m "feat: one serialiser, so the pipeline stops throwing away its evidence"
```

---

### Task 2: A score that can be absent

**Files:**
- Modify: `ragregen/verify/fusion.py:29-76`
- Modify: `ragregen/schedule.py` (`select_best`)
- Modify: `tests/test_fusion.py`, `tests/test_schedule.py`

**Interfaces:**
- Consumes: Task 1's `record` (not directly — this task is pure logic)
- Produces: `Verdict.score: float | None`; `select_best(draft_score: float | None, attempts)` where `attempts` is `(label, ok, score | None)`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_fusion.py
from ragregen.verify.fusion import fuse
from ragregen.verify.grounded import ConceptScore
from ragregen.verify.semantic import SemanticVerdict


def test_score_is_none_when_stream_a_contributed_nothing():
    """Every concept abstained. 1.0 would be a number we do not have."""
    scores = {"three": ConceptScore("three", "count", "ABSTAIN")}
    v = fuse(scores, SemanticVerdict(ok=True, raw=""))
    assert v.score is None
    assert v.ok is True


def test_score_is_the_weakest_concept_when_stream_a_spoke():
    scores = {
        "fox": ConceptScore("fox", "subject", "PRESENT", sim=0.8),
        "tree": ConceptScore("tree", "subject", "PRESENT", sim=0.3),
    }
    assert fuse(scores, SemanticVerdict(ok=True, raw="")).score == 0.3
```

```python
# append to tests/test_schedule.py
from ragregen.schedule import select_best


def test_a_scoreless_failing_attempt_never_displaces_the_draft():
    """The anas_platyrhynchos shape: the draft failed, the attempt failed, and
    the attempt carries no Stream A evidence at all."""
    assert select_best(0.0, [("attempt_1", False, None)]) == "draft"


def test_a_higher_scoring_failing_attempt_still_wins_when_it_has_evidence():
    assert select_best(0.1, [("attempt_1", False, 0.4)]) == "attempt_1"


def test_a_none_draft_score_does_not_crash_selection():
    assert select_best(None, [("attempt_1", False, 0.4)]) == "attempt_1"
    assert select_best(None, [("attempt_1", False, None)]) == "draft"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_fusion.py tests/test_schedule.py -q`
Expected: FAIL — `assert 1.0 is None` and `assert 'attempt_1' == 'draft'`

- [ ] **Step 3: Implement in `fusion.py`**

Change the `Verdict` dataclass field to `score: float | None`, and replace the score block:

```python
    #: min over the concepts Stream A could actually see. None -- not 1.0 --
    #: when it saw none: a fabricated perfect score outranks every real one in
    #: select_best, which is how three failing attempts were promoted over
    #: their drafts in outputs/doc5_20260810_183939.
    sims = [s.sim for s in grounded_scores.values()
            if s.state != "ABSTAIN" and s.sim is not None]
    score = float(min(sims)) if sims else None
```

- [ ] **Step 4: Implement in `schedule.select_best`**

```python
def select_best(draft_score: float | None, attempts) -> str:
    """The candidate to report, from {draft, attempt_1 .. attempt_N}.

    `attempts` is (label, ok, score) in attempt order; score may be None,
    meaning Stream A produced no evidence for that candidate. A candidate with
    no evidence can never displace the draft -- it is not better, it is
    unmeasured.
    """
    for label, ok, _ in attempts:
        if ok:
            return label

    best_label, best_score = "draft", draft_score
    for label, _, score in attempts:
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_label, best_score = label, score
    return best_label
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_fusion.py tests/test_schedule.py -q`
Expected: PASS

- [ ] **Step 6: Mutation-test the guard**

Temporarily change `if score is None: continue` to `if False: continue`. Run
`$PY -m pytest tests/test_schedule.py -q` — **exactly one** test must fail
(`test_a_scoreless_failing_attempt_never_displaces_the_draft`). Revert.

- [ ] **Step 7: Full suite, then commit**

Run: `$PY -m pytest -q 2>&1 | tail -3`

```bash
git add ragregen/verify/fusion.py ragregen/schedule.py tests/test_fusion.py tests/test_schedule.py
git commit -m "fix: an absent score reads as absent, not as perfect"
```

---

### Task 3: The pipeline persists what it computed

**Files:**
- Modify: `scripts/run_pipeline.py:311-343` (`_stash`, `_record_score`)
- Modify: `tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: `record.to_dict`, `record.from_dict` (Task 1); `fuse` returning `score: float | None` (Task 2)
- Produces: `streams.json` entries shaped `{"<attempt>": {"grounded": {phrase: {...full dict...}}, "semantic": {...}}}`; `scores.json` entries `[ok, score|null]`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_run_pipeline.py
import json

import run_pipeline  # the suite already imports scripts/ this way


def test_grounded_stash_persists_similarities_not_just_states(tmp_path, fake_run_dir):
    """202 verdicts of GPU time were persisted as 202 bare strings."""
    from ragregen.verify.grounded import ConceptScore
    scores = {"fox": ConceptScore("fox", "subject", "MISSING", sim=0.12,
                                  sim_coarse=0.03)}
    run_pipeline._stash_grounded(fake_run_dir, "fox_case", 0, scores)
    data = json.loads(
        (fake_run_dir.case_dir("fox_case") / "streams.json").read_text())
    assert data["0"]["grounded"]["fox"]["sim"] == 0.12
    assert data["0"]["grounded"]["fox"]["state"] == "MISSING"


def test_record_score_reads_an_old_bare_string_streams_file(fake_run_dir):
    """A doc-5 run dir must stay resumable."""
    both = {"grounded": {"fox": "MISSING"},
            "semantic": {"ok": True, "degenerate": False}}
    ok = run_pipeline._record_score(fake_run_dir, "fox_case", "draft", both)
    assert ok is False
    scored = json.loads(
        (fake_run_dir.case_dir("fox_case") / "scores.json").read_text())
    assert scored["draft"] == [False, None]


def test_record_score_uses_the_real_similarity_when_it_is_there(fake_run_dir):
    both = {"grounded": {"fox": {"phrase": "fox", "kind": "subject",
                                 "state": "MISSING", "sim": 0.12,
                                 "candidates": []}},
            "semantic": {"ok": True, "degenerate": False}}
    run_pipeline._record_score(fake_run_dir, "fox_case", "draft", both)
    scored = json.loads(
        (fake_run_dir.case_dir("fox_case") / "scores.json").read_text())
    assert scored["draft"] == [False, 0.12]
```

Add this fixture near the top of the file if the suite has no equivalent:

```python
@pytest.fixture
def fake_run_dir(tmp_path):
    class _RD:
        path = tmp_path

        def case_dir(self, cid):
            d = tmp_path / cid
            d.mkdir(exist_ok=True)
            return d
    return _RD()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_run_pipeline.py -q -k "stash or record_score"`
Expected: FAIL — `AttributeError: module 'run_pipeline' has no attribute '_stash_grounded'`

- [ ] **Step 3: Implement**

Add beside `_stash`:

```python
def _stash_grounded(run_dir, case_id: str, attempt: int, scores: dict) -> None:
    """Persist Stream A in full -- every similarity, every candidate.

    The old shape was {phrase: state}, which cannot be re-graded at any other
    tau. See docs/superpowers/specs/2026-08-12-verifier-persistence-design.md.
    """
    _stash(run_dir, case_id, attempt, "grounded",
           {phrase: record.to_dict(s, with_state=True)
            for phrase, s in scores.items()})
```

Change `score_grounded` (`run_pipeline.py:373-375`) to call it:

```python
        scores = verifier.score(image_of(cid), parsed)
        _stash_grounded(run_dir, cid, attempt, scores)
```

Change `_record_score`'s rehydration (`run_pipeline.py:338-339`) to:

```python
    scores = {phrase: record.from_dict(payload)
              for phrase, payload in both["grounded"].items()}
```

and its write to tolerate `None`:

```python
    data[label] = [bool(verdict.ok),
                   None if verdict.score is None else float(verdict.score)]
```

Add `from ragregen.verify import record` to the module imports.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_run_pipeline.py -q -k "stash or record_score"`
Expected: PASS (3 tests)

- [ ] **Step 5: Prove the doc-5 queue still resumes**

Run:
```bash
$PY -c "
import json, sys
sys.path.insert(0, 'scripts')
import run_pipeline
both = json.load(open('outputs/doc5_20260810_183939/amur_leopard/streams.json'))['0']
scores = {p: __import__('ragregen.verify.record', fromlist=['x']).from_dict(v)
          for p, v in both['grounded'].items()}
print('rehydrated:', {p: s.state for p, s in scores.items()})
"
```
Expected: `rehydrated: {'Amur leopard': 'PRESENT'}` — the real campaign file, read without error.

- [ ] **Step 6: Full suite, then commit**

```bash
git add scripts/run_pipeline.py tests/test_run_pipeline.py
git commit -m "fix: the pipeline persists the numbers behind its verdicts"
```

---

### Task 4: Wire the trace that was written and never used

**Files:**
- Modify: `scripts/run_pipeline.py` (`_verify_round`, `_retrieve_one`)
- Modify: `tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: `ragregen.trace.CaseTrace` (`vlm(stage, raw)`, `grounded(scores)`, `hits(list)`, `save(path)`)
- Produces: `<case>/trace.json` per case

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_run_pipeline.py
def test_the_raw_vlm_reply_reaches_the_trace(fake_run_dir):
    run_pipeline._trace_vlm(fake_run_dir, "fox_case", 0,
                            '{"verdict": "FAIL", "issues": []}')
    data = json.loads(
        (fake_run_dir.case_dir("fox_case") / "trace.json").read_text())
    assert data["vlm"][0]["raw"] == '{"verdict": "FAIL", "issues": []}'
    assert data["vlm"][0]["stage"] == "semantic@0"


def test_a_garbage_reply_is_flagged_in_the_trace(fake_run_dir):
    """One doc-5 verdict was degenerate and nothing recorded what Qwen said."""
    from ragregen.trace import GARBAGE_SIGNATURE
    run_pipeline._trace_vlm(fake_run_dir, "duck", 0, GARBAGE_SIGNATURE)
    data = json.loads((fake_run_dir.case_dir("duck") / "trace.json").read_text())
    assert data["vlm"][0]["garbage"] is True


def test_semantic_stash_keeps_issues_and_raw(fake_run_dir):
    from ragregen.verify.semantic import Issue, SemanticVerdict
    v = SemanticVerdict(ok=False, raw="{...}", degenerate=False,
                        issues=[Issue(concept="fox", problem="wrong ears")])
    run_pipeline._stash_semantic(fake_run_dir, "fox_case", 0, v)
    data = json.loads(
        (fake_run_dir.case_dir("fox_case") / "streams.json").read_text())
    assert data["0"]["semantic"]["issues"] == [
        {"concept": "fox", "problem": "wrong ears"}]
    assert data["0"]["semantic"]["raw"] == "{...}"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_run_pipeline.py -q -k "trace or semantic_stash"`
Expected: FAIL — `AttributeError: module 'run_pipeline' has no attribute '_trace_vlm'`

- [ ] **Step 3: Implement**

```python
def _trace(run_dir, case_id: str) -> "trace.CaseTrace":
    """Load-or-start this case's trace. Appended to, never overwritten."""
    from ragregen.trace import CaseTrace
    path = run_dir.case_dir(case_id) / "trace.json"
    t = CaseTrace(case_id=case_id)
    if path.is_file():
        old = json.loads(path.read_text())
        t.vlm_calls = old.get("vlm", [])
        t.grounded_scores = old.get("grounded", [])
        t.retrieval_hits = old.get("retrieval", [])
    return t


def _trace_vlm(run_dir, case_id: str, attempt: int, raw: str) -> None:
    """The raw reply, with garbage flagged.

    Parent spec §11: the cause is almost never the algorithm, it is one
    silently-garbage intermediate that nothing logged.
    """
    t = _trace(run_dir, case_id)
    t.vlm(f"semantic@{attempt}", raw)
    t.save(run_dir.case_dir(case_id) / "trace.json")


def _stash_semantic(run_dir, case_id: str, attempt: int, verdict) -> None:
    _stash(run_dir, case_id, attempt, "semantic",
           {"ok": verdict.ok, "degenerate": verdict.degenerate,
            "raw": verdict.raw,
            "issues": [{"concept": i.concept, "problem": i.problem}
                       for i in verdict.issues]})
```

In `_verify_round`'s `judge`, replace the `_stash(...)` call with:

```python
        verdict = verifier.judge(image_of(cid), case.prompt)
        _stash_semantic(run_dir, cid, attempt, verdict)
        _trace_vlm(run_dir, cid, attempt, verdict.raw)
```

In `score_grounded`, after `_stash_grounded(...)`, add:

```python
        t = _trace(run_dir, cid)
        t.grounded({k: v.state for k, v in scores.items()})
        t.save(run_dir.case_dir(cid) / "trace.json")
```

In `_retrieve_one`, after writing `refs.json`:

```python
    t = _trace(run_dir, case.id)
    t.hits([h.path for h in hits])
    t.save(run_dir.case_dir(case.id) / "trace.json")
```

`_retrieve_one` gains no new parameter — it already takes `run_dir`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_run_pipeline.py -q -k "trace or semantic_stash"`
Expected: PASS (3 tests)

- [ ] **Step 5: Full suite, then commit**

```bash
git add scripts/run_pipeline.py tests/test_run_pipeline.py
git commit -m "feat: the trace module stops being dead code"
```

---

### Task 5: Prototypes become reachable, and masking says how it chose

**Files:**
- Modify: `scripts/run_pipeline.py` (arg parsing, `_verify_round`, `_mask_one`)
- Modify: `tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: `prototype.bank_from_gt_refs(ds, embedder)`, `prototype.bank_from_retrieval(ds, retriever, embedder, k=8)`, `mask.mask_draft(image, coarse, masker, *, prototypes, score_phrase, dilate_px)`
- Produces: CLI flags `--proto-arm {none,ceiling,retrieved}` and `--mask-prototypes`; `<case>/mask.json`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_run_pipeline.py
def test_both_prototype_flags_default_off(monkeypatch):
    """A resumed doc-5 run must behave exactly as it did."""
    args = run_pipeline._parse_args(["--arm", "oracle"])
    assert args.proto_arm == "none"
    assert args.mask_prototypes is False


def test_mask_draft_gets_no_bank_when_the_flag_is_off(fake_run_dir, monkeypatch):
    seen = {}

    def _fake_mask_draft(image, coarse, masker, **kw):
        seen.update(kw)
        return None

    monkeypatch.setattr(run_pipeline.mask, "mask_draft", _fake_mask_draft)
    case = run_pipeline_case_stub()          # defined in Step 3
    with pytest.raises(ValueError):
        run_pipeline._mask_one(object(), fake_run_dir, case, _blank_image(),
                               _pipe_cfg_stub(), prototypes=None)
    assert seen["prototypes"] is None


def test_mask_json_records_how_the_box_was_chosen(fake_run_dir, monkeypatch):
    """doc 1 §6 called selected_by the diagnostic that makes a masking failure
    separable from a generation failure. The pipeline never wrote it."""
    from ragregen.mask import MaskResult
    import numpy as np

    result = MaskResult(mask=np.ones((4, 4), dtype=float), box=(0, 0, 4, 4),
                        score=0.9, n_candidates=3, selected_by="confidence")
    monkeypatch.setattr(run_pipeline.mask, "mask_draft", lambda *a, **k: result)
    monkeypatch.setattr(run_pipeline.mask, "mask_reference", lambda *a, **k: None)
    case = run_pipeline_case_stub()
    (fake_run_dir.case_dir(case.id) / "refs.json").write_text("[]")
    run_pipeline._mask_one(object(), fake_run_dir, case, _blank_image(),
                           _pipe_cfg_stub(), prototypes=None)
    meta = json.loads((fake_run_dir.case_dir(case.id) / "mask.json").read_text())
    assert meta == {"selected_by": "confidence", "n_candidates": 3,
                    "box": [0, 0, 4, 4], "score": 0.9}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_run_pipeline.py -q -k "prototype or mask_json"`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'proto_arm'`

- [ ] **Step 3: Add the test helpers**

```python
# near the fixtures in tests/test_run_pipeline.py
from PIL import Image

from ragregen.config import Case


def _blank_image():
    return Image.new("RGB", (8, 8), "white")


def run_pipeline_case_stub():
    return Case(id="fox_case", prompt="a fox in grass", concept="fox",
                coarse="canine", gt_refs=[], kind="target", cohort="bridge")


def _pipe_cfg_stub():
    class _Cfg:
        mask_dilate_px = 12
        retry_budget = 3
        tau = 0.25
        crop_scorer = "siglip_so400m_384"
    return _Cfg()
```

If `Case`'s constructor rejects any of those keyword names, read
`ragregen/config.py`'s `Case` dataclass and use its real field names — do not
invent fields.

- [ ] **Step 4: Implement the flags**

In the argument parser:

```python
    ap.add_argument("--proto-arm", choices=("none", "ceiling", "retrieved"),
                    default="none",
                    help="build a prototype bank and record image-side scores "
                         "in streams.json. Records only -- the live verdict "
                         "logic is unchanged (grounded.py:87).")
    ap.add_argument("--mask-prototypes", action="store_true",
                    help="ALSO use that bank to pick the mask box. This "
                         "changes mask geometry, so it changes results.")
```

Add the bank builder:

```python
def _build_bank(arm: str, ds, device: str):
    """None unless an arm was asked for. Two arms, two sources, never mixed."""
    if arm == "none":
        return None, None
    from ragregen import models, retrieve
    from ragregen.verify import prototype
    embedder = models.build_embedder(device=device)
    if arm == "ceiling":
        return prototype.bank_from_gt_refs(ds, embedder), embedder
    return prototype.bank_from_retrieval(ds, retrieve.Retriever.open(), embedder), embedder
```

Confirm the real constructor names for `models.build_embedder` and
`retrieve.Retriever` by reading those modules before writing this — use what is
there, do not invent.

Pass the bank into `GroundedVerifier(..., prototypes=bank, embedder=embedder)`
inside `load_grounded`, and give `_mask_one` a `prototypes` keyword that it
forwards to `mask.mask_draft(..., prototypes=prototypes)`. The driver passes
`bank if args.mask_prototypes else None`.

- [ ] **Step 5: Write `mask.json` in `_mask_one`**

Immediately after the `drafted is None` check:

```python
    (d / "mask.json").write_text(json.dumps({
        "selected_by": drafted.selected_by,
        "n_candidates": drafted.n_candidates,
        "box": list(drafted.box),
        "score": drafted.score,
    }, indent=2))
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_run_pipeline.py -q -k "prototype or mask_json"`
Expected: PASS (3 tests)

- [ ] **Step 7: Full suite, then commit**

```bash
git add scripts/run_pipeline.py tests/test_run_pipeline.py
git commit -m "feat: the prototype bank is reachable, and the masker says how it chose"
```

---

### Task 6: `healthy` is not `unrepaired`

**Files:**
- Modify: `ragregen/schedule.py` (`CaseState`, `SCHEMA`)
- Modify: `scripts/run_pipeline.py` (`_resolve`, `_verify_round`)
- Modify: `tests/test_schedule.py`, `tests/test_run_pipeline.py`

**Interfaces:**
- Produces: `CaseState.status ∈ {pending, healthy, repaired, unrepaired, failed}`; `SCHEMA = 2`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_schedule.py
def test_a_schema_1_queue_still_opens(tmp_path):
    """The doc-5 queue must stay resumable across the schema bump."""
    import json
    from ragregen.schedule import Queue
    p = tmp_path / "queue.json"
    p.write_text(json.dumps({
        "schema": 1, "retry_budget": 3, "screen_run": "outputs/screen_x",
        "cases": {"fox": {"stages": {"grounded@0": "done"}, "attempt": 0,
                          "status": "unrepaired", "best": "draft",
                          "error": None}}}))
    q = Queue.open(p, ["fox"], retry_budget=3, screen_run="ignored")
    assert q.cases["fox"].status == "unrepaired"
    assert q.cases["fox"].best == "draft"
```

```python
# append to tests/test_run_pipeline.py
def test_a_draft_that_passed_is_healthy_not_unrepaired():
    """Two code paths wrote 'unrepaired' meaning opposite things
    (findings/2026-07-29-orchestration-result.md §2)."""
    q = _queue_stub(["fox"])
    run_pipeline._resolve(q, "fox", "draft")
    assert q.cases["fox"].status == "healthy"


def test_a_repair_that_worked_is_repaired():
    q = _queue_stub(["fox"])
    run_pipeline._resolve(q, "fox", "attempt_2")
    assert q.cases["fox"].status == "repaired"


def test_a_resolved_case_is_not_re_fused(fake_run_dir, monkeypatch):
    """A resolved case had its scores.json rewritten on every resume."""
    calls = []
    monkeypatch.setattr(run_pipeline, "_record_score",
                        lambda *a, **k: calls.append(a) or True)
    q = _queue_stub(["fox"])
    q.cases["fox"].status = "healthy"
    run_pipeline._fuse_pending(q, fake_run_dir, {"fox": None}, attempt=1)
    assert calls == []
```

Add `_queue_stub` beside the other helpers:

```python
def _queue_stub(ids):
    from ragregen.schedule import CaseState, Queue
    q = Queue(Path("/dev/null"), {i: CaseState(case_id=i) for i in ids}, 3, "x")
    q.save = lambda: None
    return q
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_schedule.py tests/test_run_pipeline.py -q -k "healthy or schema_1 or re_fused or repaired"`
Expected: FAIL — `assert 'unrepaired' == 'healthy'`

- [ ] **Step 3: Implement**

In `schedule.py`: `SCHEMA = 2`, and update `CaseState.status`'s comment to
`# pending|healthy|repaired|unrepaired|failed`. `Queue.open` needs **no** change —
it already reads whatever statuses the file holds, which is what keeps schema 1
resumable; the test above pins that.

In `run_pipeline._resolve`:

```python
    queue.cases[case_id].best = label
    queue.cases[case_id].status = "healthy" if label == "draft" else "repaired"
```

Extract the fuse loop at the end of `_verify_round` into a named function so it
is testable, and skip resolved cases:

```python
#: Statuses meaning the case has left the loop. Re-fusing one rewrites
#: scores.json for finished work on every resume
#: (findings/2026-07-29-orchestration-result.md §3).
RESOLVED = ("healthy", "repaired", "unrepaired", "failed")


def _fuse_pending(queue, run_dir, by_id, *, attempt: int) -> None:
    for cid in list(by_id):
        st = queue.cases.get(cid)
        if st is not None and st.status in RESOLVED:
            continue
        streams = run_dir.case_dir(cid) / "streams.json"
        if not streams.is_file():
            continue
        both = json.loads(streams.read_text()).get(str(attempt), {})
        if "grounded" not in both or "semantic" not in both:
            continue
        label = "draft" if attempt == 0 else f"attempt_{attempt}"
        if _record_score(run_dir, cid, label, both):
            _resolve(queue, cid, label)
```

and call `_fuse_pending(queue, run_dir, by_id, attempt=attempt)` from
`_verify_round`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_schedule.py tests/test_run_pipeline.py -q -k "healthy or schema_1 or re_fused or repaired"`
Expected: PASS (5 tests)

- [ ] **Step 5: Check `report.py` does not read `status`**

Run: `grep -n "status" scripts/report.py`
Expected: no partitioning on `status` — doc 4 §4 partitions on `scores.json`'s
draft verdict. If any line does partition on it, fix it to use the draft verdict
and note it in the commit message.

- [ ] **Step 6: Full suite, then commit**

```bash
git add ragregen/schedule.py scripts/run_pipeline.py tests/test_schedule.py tests/test_run_pipeline.py
git commit -m "fix: a case that never needed repair is healthy, not unrepaired"
```

---

### Task 7: `scripts/reverify.py`

**Files:**
- Create: `scripts/reverify.py`
- Create: `tests/test_reverify.py`
- Modify: `scripts/run.sh` (add a `reverify` stage)

**Interfaces:**
- Consumes: everything from Tasks 1-6; `trace.open_run(tag, argv, args)`, `schedule.Queue`, `schedule.select_best`, `run_pipeline._verify_round`
- Produces: a new run directory with `streams.json`, `scores.json`, `queue.json`, `run.json` carrying `source_run`, and symlinked pixels

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reverify.py
import json
import os
from pathlib import Path

import pytest

import reverify


def _finished_run(tmp_path):
    src = tmp_path / "doc5_src"
    for cid in ("fox", "duck"):
        d = src / cid
        d.mkdir(parents=True)
        (d / "mask.png").write_bytes(b"PNG")
        (d / "attempt_1.png").write_bytes(b"PNG")
        (d / "scores.json").write_text(json.dumps({"draft": [False, 0.0],
                                                   "attempt_1": [False, 1.0]}))
    (src / "queue.json").write_text(json.dumps({
        "schema": 1, "retry_budget": 3, "screen_run": "outputs/screen_x",
        "cases": {"fox": {"stages": {}, "attempt": 1, "status": "repaired",
                          "best": "attempt_1", "error": None},
                  "duck": {"stages": {}, "attempt": 1, "status": "repaired",
                           "best": "attempt_1", "error": None}}}))
    return src


def test_the_source_run_is_never_written(tmp_path):
    src = _finished_run(tmp_path)
    before = {p: p.stat().st_mtime_ns for p in src.rglob("*") if p.is_file()}
    reverify.plan_targets(src)
    after = {p: p.stat().st_mtime_ns for p in src.rglob("*") if p.is_file()}
    assert before == after


def test_targets_are_the_draft_and_every_attempt(tmp_path):
    src = _finished_run(tmp_path)
    targets = reverify.plan_targets(src)
    assert targets["fox"] == [0, 1]


def test_pixels_are_symlinked_not_copied(tmp_path):
    src = _finished_run(tmp_path)
    dst = tmp_path / "out"
    dst.mkdir()
    reverify.link_pixels(src, _RunDirStub(dst), ["fox"])
    linked = dst / "fox" / "attempt_1.png"
    assert linked.is_symlink()
    assert linked.resolve() == (src / "fox" / "attempt_1.png").resolve()


def test_reselection_drops_a_promoted_failing_attempt(tmp_path):
    """anas_platyrhynchos: draft [false, 0.0], attempt_1 [false, 1.0]."""
    scored = {"draft": [False, 0.0], "attempt_1": [False, None]}
    assert reverify.reselect(scored) == "draft"


def test_reselection_keeps_a_genuine_pass():
    scored = {"draft": [False, 0.0], "attempt_1": [True, 0.4]}
    assert reverify.reselect(scored) == "attempt_1"


class _RunDirStub:
    def __init__(self, path):
        self.path = path

    def case_dir(self, cid):
        d = self.path / cid
        d.mkdir(parents=True, exist_ok=True)
        return d
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_reverify.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'reverify'`

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python
"""Re-verify a finished run's images into a NEW run directory.

Regenerates nothing. The source run is evidence and is never written to --
operator standing rule, 2026-08-10: when a run's provenance is compromised,
re-run it cleanly rather than hand-edit its metadata.

    ./scripts/run.sh reverify --run outputs/doc5_20260810_183939 \
        --dataset configs/dataset_common.yaml [--proto-arm retrieved]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ragregen import config, schedule, trace                    # noqa: E402
import run_pipeline                                             # noqa: E402


def plan_targets(src: Path) -> dict[str, list[int]]:
    """{case_id: [0, 1, 2...]} -- attempt 0 is the draft. Read-only."""
    out: dict[str, list[int]] = {}
    for case_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        attempts = sorted(int(p.stem.split("_")[1])
                          for p in case_dir.glob("attempt_*.png"))
        out[case_dir.name] = [0] + attempts
    return out


def link_pixels(src: Path, run_dir, case_ids: list[str]) -> None:
    """Symlink, never copy. These pixels are the one thing this run did not
    produce, and claiming authorship of them is the error being corrected."""
    for cid in case_ids:
        d = run_dir.case_dir(cid)
        for pattern in ("mask.png", "attempt_*.png", "cutout_*.png"):
            for f in sorted((src / cid).glob(pattern)):
                target = d / f.name
                if not target.exists():
                    target.symlink_to(f.resolve())


def reselect(scored: dict) -> str:
    """Re-run select_best over a corrected scores.json."""
    draft_ok, draft_score = scored.get("draft", [False, None])
    attempts = [(label, bool(v[0]), v[1])
                for label, v in sorted(scored.items()) if label != "draft"]
    if draft_ok:
        return "draft"
    return schedule.select_best(draft_score, attempts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True,
                    help="a finished pipeline run directory")
    ap.add_argument("--dataset", type=Path,
                    default=Path("configs/dataset_common.yaml"))
    ap.add_argument("--screen", type=Path, default=Path("outputs/screen_latest"))
    ap.add_argument("--proto-arm", choices=("none", "ceiling", "retrieved"),
                    default="none")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="reverify")
    args = ap.parse_args()

    src = args.run.resolve()
    targets = plan_targets(src)
    ds = config.load_dataset(args.dataset)
    by_id = {c.id: c for c in ds.cases if c.id in targets}

    run_dir = trace.open_run(args.tag, argv=sys.argv, args=vars(args))
    run_dir.write_json("run.json", {"source_run": str(src),
                                    "proto_arm": args.proto_arm})
    link_pixels(src, run_dir, list(by_id))

    queue = schedule.Queue.open(run_dir.path / "queue.json", list(by_id),
                                retry_budget=max(len(v) for v in targets.values()),
                                screen_run=str(args.screen))
    queue.save()

    max_attempt = max(max(v) for v in targets.values())
    for attempt in range(0, max_attempt + 1):
        def image_of(cid, _a=attempt):
            if _a == 0:
                return _open(args.screen / cid / "draft.png")
            return _open(src / cid / f"attempt_{_a}.png")

        run_pipeline._verify_round(queue, run_dir, by_id, _cfg(args),
                                   attempt=attempt, device=args.device,
                                   image_of=image_of)

    for cid in by_id:
        path = run_dir.case_dir(cid) / "scores.json"
        if not path.is_file():
            continue
        best = reselect(json.loads(path.read_text()))
        queue.cases[cid].best = best
        queue.cases[cid].status = ("healthy" if best == "draft"
                                   and json.loads(path.read_text())["draft"][0]
                                   else "repaired" if best != "draft"
                                   else "unrepaired")
    queue.save()
    run_dir.finish("ok", {"cases": len(by_id)})
    return 0


def _open(path: Path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def _cfg(args):
    return config.load_pipeline(Path("configs/pipeline.yaml"))


if __name__ == "__main__":
    raise SystemExit(main())
```

Before writing this, read `ragregen/config.py` for the real loader names
(`load_dataset` / `load_pipeline` may differ) and `ragregen/trace.py:59` for
`write_json`'s signature. Use what is there.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_reverify.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Add the `run.sh` stage**

Add a `reverify)` case beside the existing `pipeline)` case, forwarding `"$@"` to
`scripts/reverify.py`, and add it to the usage string.

- [ ] **Step 6: Dry check against the real run, no GPU**

Run:
```bash
$PY -c "
import sys; sys.path.insert(0, 'scripts')
import reverify
from pathlib import Path
t = reverify.plan_targets(Path('outputs/doc5_20260810_183939'))
print('cases:', len(t), 'verdicts:', sum(len(v) for v in t.values()))
"
```
Expected: `cases: 92 verdicts: 202`

- [ ] **Step 7: Full suite, then commit**

```bash
git add scripts/reverify.py tests/test_reverify.py scripts/run.sh
git commit -m "feat: re-verify a finished run without regenerating a pixel"
```

---

### Task 8: The documentation stops claiming a step that does not exist

**Files:**
- Modify: `docs/RUNBOOK.md` (§1 pipeline diagram, §3 stage list)
- Modify: `configs/retrieval_db.yaml` or the two path derivations that read it

- [ ] **Step 1: Fix the query-step lie**

`RUNBOOK.md` §1 documents *"2. QUERY — the VLM writes a search caption for the
missing concept."* `ragregen/query.py` does not exist and
`run_pipeline._retrieve_one` searches `case.concept` verbatim (deliberately
deferred, orchestration design §9). Replace that block with:

```
  ├─ 2. RETRIEVE ........ SigLIP searches YOUR retrieval database for the
  │                       concept name itself, returning the top N matches.
  │                       (A VLM-written search caption is designed but not
  │                       built -- orchestration design §9. The concept name
  │                       is what is searched today.)
```

and renumber the steps below it.

- [ ] **Step 2: Document the two new flags and the reverify stage**

Add to §3 a row for `reverify`, and for `--proto-arm` / `--mask-prototypes` a
note that both default off and that `--mask-prototypes` changes results while
`--proto-arm` only records.

- [ ] **Step 3: Fix the CWD dependence**

`configs/retrieval_db.yaml` mixes an absolute `images_root` with a relative
`index_path`, and both `validate_all` and `report.main` derive the manifest from
`index_path.parent` — so both silently depend on the working directory (ledger
MINOR 13). Resolve `index_path` against `env.PROJECT_ROOT` when it is relative,
in the config loader, so every entry point agrees.

- [ ] **Step 4: Write the failing test**

```python
# append to tests/test_config.py
def test_a_relative_index_path_resolves_against_the_project_root(tmp_path, monkeypatch):
    from ragregen import config, env
    monkeypatch.chdir(tmp_path)                 # not the repo root
    db = config.load_retrieval_db(env.PROJECT_ROOT / "configs/retrieval_db.yaml")
    assert db.index_path.is_absolute()
```

- [ ] **Step 5: Run it, implement, re-run**

Run: `$PY -m pytest tests/test_config.py -q -k relative_index_path`
Expected: FAIL, then PASS after the loader change.

- [ ] **Step 6: Full suite, then commit**

```bash
git add docs/RUNBOOK.md ragregen/config.py tests/test_config.py
git commit -m "docs: the runbook stops promising a query step nobody built"
```

---

### Task 9: The re-grade — REQUIRES USER APPROVAL AND A FREE GPU

> **Do not start this task without explicit approval.** It books the shared card
> for roughly 3 hours. Every prior task is CPU-only. This mirrors the doc-5
> plan's Task 13 boundary.

**Files:**
- Create: `docs/findings/2026-08-12-verifier-persistence-result.md`

- [ ] **Step 1: Grade the prototype rule on the 22 labelled drafts (~10 min GPU)**

```bash
./scripts/run.sh score-a --labels outputs/screen_20260726_233320/labels.csv \
                         --proto-arm retrieved --force
./scripts/run.sh c1 --run outputs/screen_20260726_233320
```

Read the prototype section of `c1.md`. The pre-registration of
`specs/2026-07-28-prototype-verifier-design.md` §7 is graded **as written**:
τ pinned at 0.25, `delta_proto` over the 81-point grid, graded target the 6 cases
the τ = 0.25 baseline missed, MET requiring ≥4 of 6 newly caught with **no new**
false positives over a band of ≥3 consecutive δ, box selection by detector
confidence. If it fails, the conclusion is that image-side contrast is
insufficient — **not** that the grid needs widening.

- [ ] **Step 2: Confirm the τ = 0.25 baseline still reproduces**

The same `c1.md` must still report grounded TP 5 / recall 0.333 / bal-acc 0.667 /
MCC +0.370 at τ\* = 0.02. If it moved, stop and diagnose before reading any
prototype number — that is the `f472538` failure recurring.

- [ ] **Step 3: Re-verify the 92 cases (~2.5 h GPU)**

```bash
./scripts/gpu_wait.sh --need 17 -- ./scripts/run.sh reverify \
    --run outputs/doc5_20260810_183939 --dataset configs/dataset_common.yaml
```

- [ ] **Step 4: Report on the re-verified run**

```bash
./scripts/run.sh report --run outputs/reverify_latest --dataset configs/dataset_common.yaml
```

- [ ] **Step 5: Write the findings doc**

`docs/findings/2026-08-12-verifier-persistence-result.md`, in the house style of
the other five. It must carry:

- what the corrected selection did to `anas_platyrhynchos`, `tiger` and
  `guacamole`, and to the headline deltas — the doc-5 `target/bridge` +0.047
  [+0.000, +0.114] is the number that may move;
- **the Stream B reproduction check** (risk V1): the campaign's per-case
  `semantic.ok` beside the re-verified one, and every case that moved named. If
  many moved, the re-grade is measuring nf4 nondeterminism as well as the fix,
  and the doc says so;
- whether the prototype rule was MET, with the δ band or the reason there is none;
- the distribution of Stream A `sim` now that it exists — the campaign's
  MISSING 135 / PRESENT 64 / FINE_MISMATCH 2 / ABSTAIN 41 with actual numbers
  behind it;
- what this does **not** establish: masks were cut by confidence and cannot be
  re-cut without regenerating (V4), no new images exist, and the `oracle` and
  `ceiling` arms remain unrun.

**The existing report and findings are not edited.** They stand as what was true
when written.

- [ ] **Step 6: Commit**

```bash
git add docs/findings/2026-08-12-verifier-persistence-result.md
git commit -m "docs: what the verifier was actually saying all along"
```

---

## Self-Review

**Spec coverage:** §4.1 → Task 1. §4.2 → Task 2. §4.3 → Task 4. §4.4 → Task 5.
§4.5 → Task 7. §4.6 → Task 9 Step 1. §5 (status, re-fuse, CWD, RUNBOOK) →
Tasks 6 and 8. §7 testing → distributed across every task. §8 risks: V1 → Task 9
Step 5; V2 → Task 9 Step 5; V3 → inherited, stated in Task 9 Step 1; V4 → Task 9
Step 5; V5 → Task 9 Step 3's `gpu_wait.sh`. §9 deliverables → Tasks 0-9.

**Placeholder scan:** none. Every code step carries real code. Three steps
(Task 5 Step 4, Task 7 Step 3, Task 8 Step 3) instruct the implementer to read a
named module for exact constructor names rather than trusting the plan's guess —
that is deliberate, and the modules and line numbers are named.

**Type consistency:** `record.to_dict(score, *, with_state)` / `from_dict(d)`
used identically in Tasks 1, 3, 7. `Verdict.score: float | None` introduced in
Task 2 and consumed in Tasks 3 and 7. `select_best(draft_score: float | None,
attempts)` defined in Task 2, called in Task 7's `reselect`. `_stash_grounded` /
`_stash_semantic` / `_trace_vlm` / `_fuse_pending` defined in Tasks 3, 4, 6 and
used consistently. `MaskResult.selected_by` / `.n_candidates` are pre-existing
fields (`ragregen/mask.py`), not invented here.
