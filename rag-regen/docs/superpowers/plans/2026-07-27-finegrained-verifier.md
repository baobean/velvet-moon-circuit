# Fine-Grained Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Stream A a contrastive `FINE_MISMATCH` state so it catches the fine-grained identity errors C1 showed it passes, and re-run the C1 report to decide whether the repair loop is unblocked.

**Architecture:** `GroundedVerifier` scores the best crop twice — against the concept phrase and against its superordinate `coarse` term — and reports `FINE_MISMATCH` when the margin falls below δ. `score_a.py` persists both similarities and never a verdict, so every (τ, δ) pair is recovered offline by arithmetic in `c1.py`. Stream B is untouched and its cached JSON is reused.

**Tech Stack:** Python 3.11.15 in the `kontext` conda env; pytest; PyYAML; PIL; transformers 5.14.1 (GPU tasks only). No new dependencies.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-27-finegrained-verifier-design.md`. Read it before Task 1.
- **Worktree:** all work happens in `.claude/worktrees/plan1-verifier` on branch `worktree-plan1-verifier`.
- **Interpreter.** The project runs on the `kontext` conda env, **not** whatever `python` resolves to on `PATH`. Bare `python` is base conda 3.13, cannot import `faiss`, and fails 14 tests for reasons unrelated to your change. Export this once per shell and use `$PY` in every command below:

  ```bash
  export PY=/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python
  ```

  It is the same interpreter `scripts/run.sh:5` pins.
- **Run tests with:** `$PY -m pytest` from the worktree root. The default suite is CPU-only; GPU tests are marked `gpu` and deselected by default (`pytest.ini`).
- **Baseline to preserve:** **188 passed, 9 deselected** — measured 2026-07-27, immediately before Task 1. (Plan 1's ledger records 127/8; that predates the C1 plan's tests.) Never finish a task with a red suite.
- **τ is pinned at 0.25** for all fine-grained reporting. δ is the only fitted knob. Do not sweep τ to a peak.
- **The coarse term is authored from the concept name alone**, never from the observed draft. Fitting it to a known failure is answer-key leakage.
- **`FINE_MISMATCH` never overrides `MISSING`.** If `sim < τ`, the state is `MISSING` regardless of the margin.
- **`ABSTAIN` is not `MISSING`** (parent spec §9 R3). δ must not change abstention behaviour.
- **Tests must fail for the right reason.** Four times on Plan 1 a test passed vacuously. Build fakes from the real signature (`inspect.signature`), and mutation-test every assertion that claims to pin behaviour — flip the operator, run pytest, and paste the **real** failure output. Hand-typed simulated output is not evidence.
- **Commit after every task.** Use `git add <exact paths>`, never `git add -A`.

---

## File Structure

| file | responsibility | task |
|---|---|---|
| `ragregen/concepts.py` | `Concept.coarse`; `parse(prompt, target, coarse)` attaches it to the target only | 1 |
| `ragregen/config.py` | `Case.coarse`; presence and `coarse != concept` validation at load | 2 |
| `ragregen/verify/grounded.py` | `FINE_MISMATCH` state, `delta`, second scorer call, `ConceptScore.sim_coarse` | 3 |
| `ragregen/verify/fusion.py` | treat `FINE_MISMATCH` as failure; carry `sim_coarse` into evidence | 4 |
| `scripts/score_a.py` | join `dataset.yaml` for `coarse`; persist `sim_coarse` | 5 |
| `ragregen/c1.py` | `state_at(sim, sim_coarse, τ, δ)`, `DELTA_GRID`, `sweep_delta` | 6 |
| `ragregen/c1.py` | baseline split, `evaluate_rule`, `robust_band` | 7 |
| `ragregen/c1.py`, `scripts/c1_report.py` | fine-grained report section appended to the existing τ report | 8 |
| `configs/dataset.yaml`, `configs/dataset.example.yaml`, `docs/RUNBOOK.md` | the 22 coarse terms and operator instructions | 9 |
| `docs/findings/2026-07-27-finegrained-result.md` | the measured outcome and the rule verdict | 10 |

**Deviation from spec §3, stated deliberately.** The spec places `coarse` validation in `validate.py`. This plan puts it in `config.load_dataset` instead, matching the existing precedent — the `gt_refs` requirement is enforced there, not in `validate.py`. The spec's intent (fail before any GPU work) is satisfied either way, because `load_dataset` runs first in every entry point.

---

## Task 1: `Concept.coarse` and `parse(..., coarse=...)`

**Files:**
- Modify: `ragregen/concepts.py:31-33` (dataclass), `ragregen/concepts.py:44-51` (`parse`)
- Test: `tests/test_concepts.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Concept(phrase: str, kind: str, coarse: str | None = None)` and
  `parse(prompt: str, target: str, coarse: str | None = None) -> list[Concept]`.
  The target concept (index 0, `kind == "subject"`) carries `coarse`; every other
  concept has `coarse is None`. Tasks 3 and 5 depend on exactly this.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_concepts.py`:

```python
def test_target_concept_carries_the_coarse_term():
    cs = concepts.parse("an African grey parrot on a branch",
                        target="African grey parrot", coarse="parrot")
    assert cs[0].phrase == "African grey parrot"
    assert cs[0].coarse == "parrot"


def test_non_target_concepts_have_no_coarse_term():
    cs = concepts.parse("an African grey parrot on a wooden branch",
                        target="African grey parrot", coarse="parrot")
    others = [c for c in cs[1:]]
    assert others, "prompt should yield at least one non-target concept"
    assert all(c.coarse is None for c in others)


def test_coarse_defaults_to_none_when_not_supplied():
    cs = concepts.parse("a fox in the snow", target="fox")
    assert all(c.coarse is None for c in cs)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_concepts.py -k coarse -v`
Expected: FAIL — `TypeError: parse() got an unexpected keyword argument 'coarse'`

- [ ] **Step 3: Add the field and the parameter**

In `ragregen/concepts.py`, extend the dataclass:

```python
@dataclass(frozen=True)
class Concept:
    phrase: str
    kind: str
    #: Superordinate category for the target concept, e.g. "parrot" for
    #: "African grey parrot". None on non-target concepts and whenever the
    #: caller supplies no coarse term. Stream A skips the fine-grained
    #: margin test when this is None, which is what keeps the pre-contrastive
    #: behaviour intact for every concept the operator did not annotate.
    coarse: str | None = None
```

Change the `parse` signature and its first line only:

```python
def parse(prompt: str, target: str, coarse: str | None = None) -> list[Concept]:
```

```python
    out: list[Concept] = [Concept(target, "subject", coarse)]
```

Leave the rest of `parse` untouched: every other `Concept(...)` construction passes two arguments and therefore gets `coarse=None` for free.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_concepts.py -v`
Expected: PASS, including the pre-existing tests.

- [ ] **Step 5: Mutation-check the second test**

Temporarily change the `parse` line to `Concept(target, "subject", coarse)` → apply `coarse` to *every* concept by editing the other construction sites to pass `coarse`. Run:

Run: `$PY -m pytest tests/test_concepts.py::test_non_target_concepts_have_no_coarse_term -v`
Expected: FAIL. Paste the real assertion output into the task report, then revert the mutation.

- [ ] **Step 6: Run the full suite**

Run: `$PY -m pytest`
Expected: 191 passed, 9 deselected.

- [ ] **Step 7: Commit**

```bash
git add ragregen/concepts.py tests/test_concepts.py
git commit -m "feat: carry a coarse superordinate term on the target concept"
```

---

## Task 2: `Case.coarse` with load-time validation

**Files:**
- Modify: `ragregen/config.py:21-26` (dataclass), `ragregen/config.py:86-100` (loader)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Case(id, prompt, concept, coarse, gt_refs)` — `coarse` is a required
  `str` positioned **before** `gt_refs` so the defaulted field stays last.
  `load_dataset` raises `ValueError` when `coarse` is absent or equals `concept`.
  Task 5 reads `Case.coarse`; Task 9 supplies the values.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`. Follow the file's existing fixture style for writing a temp YAML — if it has a helper for that, use it; the code below writes the file directly so it works either way.

```python
def _dataset_yaml(tmp_path, coarse_line="    coarse: parrot"):
    img = tmp_path / "imgs"
    img.mkdir(exist_ok=True)
    (img / "ref0.jpg").write_bytes(b"x")
    text = (
        "name: t\n"
        f"images_root: {img}\n"
        "cases:\n"
        "  - id: parrot_case\n"
        "    prompt: an African grey parrot on a branch\n"
        "    concept: African grey parrot\n"
        f"{coarse_line}\n"
        "    gt_refs:\n"
        "      - ref0.jpg\n"
    )
    p = tmp_path / "dataset.yaml"
    p.write_text(text)
    return p


def test_case_carries_the_coarse_term(tmp_path):
    ds = config.load_dataset(_dataset_yaml(tmp_path))
    assert ds.cases[0].coarse == "parrot"


def test_missing_coarse_is_an_actionable_error(tmp_path):
    path = _dataset_yaml(tmp_path, coarse_line="    notes: none")
    with pytest.raises(ValueError) as e:
        config.load_dataset(path)
    assert "coarse" in str(e.value)
    assert "parrot_case" in str(e.value)


def test_coarse_equal_to_concept_is_rejected(tmp_path):
    path = _dataset_yaml(tmp_path, coarse_line="    coarse: African grey parrot")
    with pytest.raises(ValueError) as e:
        config.load_dataset(path)
    assert "superordinate" in str(e.value)


def test_coarse_equal_to_concept_is_rejected_case_insensitively(tmp_path):
    path = _dataset_yaml(tmp_path, coarse_line="    coarse: '  african GREY parrot '")
    with pytest.raises(ValueError) as e:
        config.load_dataset(path)
    assert "superordinate" in str(e.value)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_config.py -k coarse -v`
Expected: FAIL — `AttributeError: 'Case' object has no attribute 'coarse'` on the first, and `Failed: DID NOT RAISE` on the rest.

- [ ] **Step 3: Add the field**

In `ragregen/config.py`:

```python
@dataclass(frozen=True)
class Case:
    id: str
    prompt: str
    concept: str
    coarse: str
    gt_refs: list[Path] = field(default_factory=list)
```

- [ ] **Step 4: Validate and populate it in `load_dataset`**

Replace the `for key in ("prompt", "concept"):` loop with:

```python
        for key in ("prompt", "concept", "coarse"):
            if not raw.get(key):
                raise ValueError(f"{path}: case '{cid}' has no {key}")
        if raw["coarse"].strip().lower() == raw["concept"].strip().lower():
            raise ValueError(
                f"{path}: case '{cid}' has coarse == concept "
                f"('{raw['concept']}'). The coarse term must be the concept's "
                f"superordinate category -- concept 'African grey parrot' -> "
                f"coarse 'parrot'. Identical terms make the fine-grained "
                f"margin identically zero, silently disabling the test."
            )
```

and add the field to the `Case(...)` construction:

```python
        cases.append(Case(
            id=cid,
            prompt=raw["prompt"],
            concept=raw["concept"],
            coarse=raw["coarse"],
            gt_refs=[root / r for r in refs],
        ))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 6: Fix the fallout across the suite**

`Case` now has a required field. Any other test or module constructing `Case(...)` will fail.

Run: `$PY -m pytest`
Expected: some failures outside `test_config.py`. Add `coarse=` to each construction site. Do **not** give `coarse` a default value to avoid this work — a silent default is exactly how a case ships with the fine-grained test disabled.

Re-run: `$PY -m pytest`
Expected: 195 passed, 9 deselected.

- [ ] **Step 7: Commit**

```bash
git add ragregen/config.py tests/test_config.py
git commit -m "feat: require a coarse superordinate term on every dataset case"
```

Include any other files you had to touch in Step 6 in the same `git add`.

---

## Task 3: `FINE_MISMATCH` in `GroundedVerifier`

**Files:**
- Modify: `ragregen/verify/grounded.py:16` (`STATES`), `:36-42` (`ConceptScore`), `:45-60` (`__init__`), `:68-85` (`_score_one`)
- Test: `tests/test_grounded.py`

**Interfaces:**
- Consumes: `Concept.coarse` from Task 1.
- Produces: `GroundedVerifier(detector, scorer, tau: float, delta: float = 0.0)`
  and `ConceptScore(phrase, kind, state, box=None, dino_conf=None, sim=None, sim_coarse=None)`
  where `state` is one of `PRESENT`, `MISSING`, `FINE_MISMATCH`, `ABSTAIN`.
  Tasks 4, 5 and 6 depend on this state name and on `sim_coarse`.

The existing `FakeDetector` and `FakeScorer` at the top of `tests/test_grounded.py` already
dispatch by phrase, so a coarse term is just another entry in the `sims` dict. Reuse them;
do not write new fakes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_grounded.py`:

```python
def _gv(sims, tau=0.25, delta=0.0):
    return grounded.GroundedVerifier(
        FakeDetector({"African grey parrot": [(BOX, 0.9)]}),
        FakeScorer(sims), tau=tau, delta=delta)


PARROT = Concept("African grey parrot", "subject", "parrot")


def test_coarse_term_winning_by_more_than_delta_is_a_fine_mismatch():
    # margin = 0.40 - 0.55 = -0.15, which is below delta=0.05
    gv = _gv({"African grey parrot": 0.40, "parrot": 0.55}, delta=0.05)
    out = gv.score(IMG, [PARROT])
    assert out["African grey parrot"].state == "FINE_MISMATCH"
    assert out["African grey parrot"].sim_coarse == 0.55


def test_margin_exactly_at_delta_is_present():
    # margin = 0.60 - 0.50 = 0.10 == delta, and the test is strict `<`
    gv = _gv({"African grey parrot": 0.60, "parrot": 0.50}, delta=0.10)
    out = gv.score(IMG, [PARROT])
    assert out["African grey parrot"].state == "PRESENT"


def test_margin_just_below_delta_is_a_fine_mismatch():
    # margin = 0.599 - 0.50 = 0.099 < delta = 0.10
    gv = _gv({"African grey parrot": 0.599, "parrot": 0.50}, delta=0.10)
    out = gv.score(IMG, [PARROT])
    assert out["African grey parrot"].state == "FINE_MISMATCH"


def test_missing_takes_precedence_over_fine_mismatch():
    # sim 0.10 < tau 0.25, and the margin 0.10-0.90 is also below delta
    gv = _gv({"African grey parrot": 0.10, "parrot": 0.90}, delta=0.05)
    out = gv.score(IMG, [PARROT])
    assert out["African grey parrot"].state == "MISSING"


def test_no_box_still_abstains_regardless_of_delta():
    gv = grounded.GroundedVerifier(
        FakeDetector({}), FakeScorer({"parrot": 0.9}), tau=0.25, delta=0.5)
    out = gv.score(IMG, [PARROT])
    assert out["African grey parrot"].state == "ABSTAIN"
    assert out["African grey parrot"].sim_coarse is None


def test_concept_without_a_coarse_term_is_never_a_fine_mismatch():
    gv = grounded.GroundedVerifier(
        FakeDetector({"fox": [(BOX, 0.9)]}), FakeScorer({"fox": 0.30}),
        tau=0.25, delta=0.9)
    out = gv.score(IMG, [Concept("fox", "subject")])
    assert out["fox"].state == "PRESENT"
    assert out["fox"].sim_coarse is None


@pytest.mark.parametrize("bad", [-1.5, 1.5])
def test_delta_outside_the_closed_unit_range_is_rejected(bad):
    with pytest.raises(ValueError, match="delta"):
        grounded.GroundedVerifier(FakeDetector({}), FakeScorer({}),
                                  tau=0.25, delta=bad)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_grounded.py -k "fine_mismatch or delta or coarse" -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'delta'`

- [ ] **Step 3: Add the state, the field, and the delta parameter**

In `ragregen/verify/grounded.py`:

```python
STATES = ("PRESENT", "MISSING", "FINE_MISMATCH", "ABSTAIN")
```

```python
@dataclass(frozen=True)
class ConceptScore:
    phrase: str
    kind: str
    state: str
    box: Box | None = None
    dino_conf: float | None = None
    sim: float | None = None
    #: Similarity of the same crop to the concept's superordinate term. None
    #: when the concept carries no coarse term or when no box was found.
    sim_coarse: float | None = None
```

Extend `__init__`:

```python
    def __init__(self, detector: Detector, scorer: CropScorer, tau: float,
                 delta: float = 0.0):
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        if not -1.0 <= delta <= 1.0:
            raise ValueError(f"delta must be in [-1, 1], got {delta}")
        self.detector = detector
        self.scorer = scorer
        self.tau = tau
        self.delta = delta
```

Update the class docstring's `Args:` block to document `delta`:

```
        delta:    a crop whose similarity to the concept exceeds its
                  similarity to the concept's coarse term by less than this
                  is FINE_MISMATCH. Only applies to concepts carrying a
                  coarse term. Defaults to 0.0.
```

- [ ] **Step 4: Score the coarse term and branch**

Replace the tail of `_score_one` (currently lines 83-85) with:

```python
        sim_coarse = None
        if concept.coarse and best_box is not None:
            crop = image.crop((int(best_box[0]), int(best_box[1]),
                               int(best_box[2]), int(best_box[3])))
            sim_coarse = float(self.scorer.score(crop, concept.coarse))

        if best_sim < self.tau:
            state = "MISSING"
        elif sim_coarse is not None and best_sim - sim_coarse < self.delta:
            state = "FINE_MISMATCH"
        else:
            state = "PRESENT"

        return ConceptScore(concept.phrase, concept.kind, state,
                            box=best_box, dino_conf=best_conf, sim=best_sim,
                            sim_coarse=sim_coarse)
```

The `MISSING` branch comes first because a crop that fails the absolute presence test is not a
fine-grained question — reversing the order would relabel gross-category errors as fine-grained
ones and corrupt the mechanism split Task 8 reports.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_grounded.py -v`
Expected: PASS, including all pre-existing tests.

- [ ] **Step 6: Mutation-check the boundary pair**

Change `best_sim - sim_coarse < self.delta` to `<= self.delta`. Run:

Run: `$PY -m pytest tests/test_grounded.py -v`
Expected: exactly one failure — `test_margin_exactly_at_delta_is_present`. If zero tests fail, the boundary is not pinned and the pair is vacuous. If two or more fail, the tests overlap and are not isolating the boundary.

Then change it back and instead swap the branch order (test `FINE_MISMATCH` before `MISSING`). Run:

Run: `$PY -m pytest tests/test_grounded.py::test_missing_takes_precedence_over_fine_mismatch -v`
Expected: FAIL.

Paste both real pytest transcripts into the task report, then revert both mutations.

- [ ] **Step 7: Run the full suite**

Run: `$PY -m pytest`
Expected: 203 passed, 9 deselected.

- [ ] **Step 8: Commit**

```bash
git add ragregen/verify/grounded.py tests/test_grounded.py
git commit -m "feat: FINE_MISMATCH state from contrastive crop scoring"
```

---

## Task 4: Fusion treats `FINE_MISMATCH` as failure

**Files:**
- Modify: `ragregen/verify/fusion.py:33-35` (failure set), `:48-50` (target), `:63-66` (evidence)
- Test: `tests/test_fusion.py`

**Interfaces:**
- Consumes: `ConceptScore.state == "FINE_MISMATCH"` and `.sim_coarse` from Task 3.
- Produces: `fusion.FAILING_STATES = ("MISSING", "FINE_MISMATCH")`. `fuse()` keeps its
  signature and `Verdict` shape; `evidence["grounded"][phrase]` gains a `sim_coarse` key.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fusion.py`, following the file's existing helpers for building
`ConceptScore` and `SemanticVerdict`:

```python
def test_fine_mismatch_fails_the_case():
    scores = {"African grey parrot": ConceptScore(
        "African grey parrot", "subject", "FINE_MISMATCH",
        box=(0, 0, 8, 8), sim=0.40, sim_coarse=0.55)}
    v = fusion.fuse(scores, SemanticVerdict(ok=True, degenerate=False, issues=[]))
    assert v.ok is False
    assert v.agreement == "GROUNDED_ONLY"
    assert v.target == "African grey parrot"


def test_fine_mismatch_competes_with_missing_for_the_target():
    scores = {
        "parrot_a": ConceptScore("parrot_a", "subject", "FINE_MISMATCH",
                                 box=(0, 0, 8, 8), sim=0.40, sim_coarse=0.55),
        "parrot_b": ConceptScore("parrot_b", "object", "MISSING",
                                 box=(0, 0, 8, 8), sim=0.10),
    }
    v = fusion.fuse(scores, SemanticVerdict(ok=True, degenerate=False, issues=[]))
    assert v.target == "parrot_b", "lowest sim among all failing states wins"


def test_evidence_carries_sim_coarse():
    scores = {"African grey parrot": ConceptScore(
        "African grey parrot", "subject", "FINE_MISMATCH",
        box=(0, 0, 8, 8), sim=0.40, sim_coarse=0.55)}
    v = fusion.fuse(scores, SemanticVerdict(ok=True, degenerate=False, issues=[]))
    assert v.evidence["grounded"]["African grey parrot"]["sim_coarse"] == 0.55


def test_fine_mismatch_with_a_none_box_still_fails():
    """A contract-violating scorer must not be rescued by box inspection."""
    scores = {"p": ConceptScore("p", "subject", "FINE_MISMATCH",
                                box=None, sim=0.40, sim_coarse=0.55)}
    v = fusion.fuse(scores, SemanticVerdict(ok=True, degenerate=False, issues=[]))
    assert v.ok is False
```

Adjust the `SemanticVerdict(...)` construction to match its real signature — check it with
`$PY -c "import inspect, ragregen.verify.semantic as s; print(inspect.signature(s.SemanticVerdict))"`
before writing the tests, and use what it prints rather than what this plan guesses.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_fusion.py -k "fine_mismatch or sim_coarse" -v`
Expected: FAIL — `assert True is False`, because `FINE_MISMATCH` is not yet in the failure set.

- [ ] **Step 3: Widen the failure set**

In `ragregen/verify/fusion.py`, add the constant below `AGREEMENTS`:

```python
#: Grounded states that fail a case. FINE_MISMATCH joins MISSING here rather
#: than being folded into it so the C1 report can attribute each catch to a
#: mechanism (spec §2).
FAILING_STATES = ("MISSING", "FINE_MISMATCH")
```

Replace the first two lines of `fuse`:

```python
    failing = [s for s in grounded_scores.values() if s.state in FAILING_STATES]
    grounded_failed = bool(failing)
```

and the target selection:

```python
    if grounded_failed:
        target = min(failing, key=lambda s: s.sim if s.sim is not None else 0.0).phrase
```

and the evidence dict:

```python
            "grounded": {p: {"state": s.state, "sim": s.sim,
                             "sim_coarse": s.sim_coarse, "kind": s.kind}
                         for p, s in grounded_scores.items()},
```

Update the module docstring's last paragraph to name the new state:

```
Failure is decided on `ConceptScore.state` alone, never on whether a box is
present -- see FAILING_STATES. A scorer that violates its contract can return
MISSING or FINE_MISMATCH with `box=None`, which is indistinguishable from
ABSTAIN if this module pattern-matches on the box (Task 6 carry-forward).
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_fusion.py -v`
Expected: PASS.

- [ ] **Step 5: Mutation-check the failure set**

Temporarily set `FAILING_STATES = ("MISSING",)`. Run:

Run: `$PY -m pytest tests/test_fusion.py -v`
Expected: FAIL on the fine-mismatch tests. Paste the real output, then revert.

- [ ] **Step 6: Run the full suite**

Run: `$PY -m pytest`
Expected: 207 passed, 9 deselected.

- [ ] **Step 7: Commit**

```bash
git add ragregen/verify/fusion.py tests/test_fusion.py
git commit -m "feat: fuse FINE_MISMATCH as a grounded failure"
```

---

## Task 5: `score_a.py` persists `sim_coarse`

**Files:**
- Modify: `scripts/score_a.py:28-40` (`score_case`), `:50-90` (`main`)
- Test: `tests/test_score_a.py`

**Interfaces:**
- Consumes: `parse(..., coarse=...)` (Task 1), `Case.coarse` (Task 2), `ConceptScore.sim_coarse` (Task 3).
- Produces: `score_case(verifier, image, prompt, concept, coarse) -> dict` where each
  phrase entry gains `"sim_coarse"`. `stream_a.json` entries therefore carry
  `{"sim", "sim_coarse", "box", "dino_conf", "kind", "state"}`. Task 6 reads
  `sim` and `sim_coarse` from this shape.

**Cache note:** the existing `outputs/screen_20260726_233320/stream_a.json` predates
`sim_coarse`. It must be regenerated with `--force` in Task 10. Do not attempt to patch the
old file — the coarse similarities were never computed and cannot be recovered offline.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_score_a.py`:

```python
class RecordingVerifier:
    """Captures the parsed concepts so the test can assert what was scored."""

    def __init__(self):
        self.seen = None

    def score(self, image, parsed):
        self.seen = parsed
        return {
            c.phrase: ConceptScore(c.phrase, c.kind, "PRESENT",
                                   box=(0.0, 0.0, 8.0, 8.0), dino_conf=0.9,
                                   sim=0.7, sim_coarse=0.3 if c.coarse else None)
            for c in parsed
        }


def test_score_case_forwards_the_coarse_term_to_the_parser():
    v = RecordingVerifier()
    score_a.score_case(v, IMG, "an African grey parrot on a branch",
                       "African grey parrot", "parrot")
    target = v.seen[0]
    assert target.phrase == "African grey parrot"
    assert target.coarse == "parrot"


def test_score_case_persists_sim_coarse():
    v = RecordingVerifier()
    out = score_a.score_case(v, IMG, "an African grey parrot on a branch",
                             "African grey parrot", "parrot")
    assert out["African grey parrot"]["sim_coarse"] == 0.3
    assert "sim" in out["African grey parrot"]
```

Reuse whatever image constant and imports `tests/test_score_a.py` already defines; add
`from ragregen.verify.grounded import ConceptScore` if it is not already imported.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_score_a.py -k coarse -v`
Expected: FAIL — `TypeError: score_case() takes 4 positional arguments but 5 were given`

- [ ] **Step 3: Thread `coarse` through `score_case`**

```python
def score_case(verifier, image, prompt: str, concept: str, coarse: str) -> dict:
    parsed = concepts.parse(prompt, target=concept, coarse=coarse)
    scores = verifier.score(image, parsed)
    return {
        phrase: {
            "sim": s.sim,
            "sim_coarse": s.sim_coarse,
            "box": list(s.box) if s.box is not None else None,
            "dino_conf": s.dino_conf,
            "kind": s.kind,
            "state": s.state,
        }
        for phrase, s in scores.items()
    }
```

- [ ] **Step 4: Join `dataset.yaml` in `main`**

Add the argument, next to `--pipeline`:

```python
    ap.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET_PATH,
                    help="dataset.yaml supplying each case's coarse term")
```

After the draft-existence loop (currently ending at line 76) and **before** any model loads,
insert the coverage check:

```python
    ds = config.load_dataset(args.dataset)
    coarse_by_id = {c.id: c.coarse for c in ds.cases}
    absent = sorted({r.case_id for r in todo} - set(coarse_by_id))
    if absent:
        print(f"[error] no coarse term in {args.dataset} for: {', '.join(absent)}. "
              f"Every labelled case needs one -- see docs/RUNBOOK.md.")
        return 2
```

Placing it before `models.DinoDetector(...)` means a missing coarse term costs a second, not a
five-minute model load.

Change the scoring call:

```python
        existing[r.case_id] = score_case(verifier, image, r.prompt, r.concept,
                                         coarse_by_id[r.case_id])
```

Update the module docstring's second paragraph:

```
Writes raw similarities for both the concept and its coarse superordinate,
never thresholded verdicts: tau and delta are recovered offline by c1.state_at,
so calibration is arithmetic instead of thousands of GPU passes.
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_score_a.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `$PY -m pytest`
Expected: 209 passed, 9 deselected.

- [ ] **Step 7: Commit**

```bash
git add scripts/score_a.py tests/test_score_a.py
git commit -m "feat: score-a persists coarse similarities from dataset.yaml"
```

---

## Task 6: Offline state recovery over (τ, δ)

**Files:**
- Modify: `ragregen/c1.py:149-192` (`state_at_tau`, `grounded_fails`, `fused_fails`, `abstain_rate`), and add `sweep_delta`
- Test: `tests/test_c1_arms.py`

**Interfaces:**
- Consumes: the `stream_a.json` shape from Task 5, and `fusion.FAILING_STATES` semantics from Task 4.
- Produces:
  - `c1.BASELINE_TAU = 0.25`
  - `c1.DELTA_GRID: tuple[float, ...]` — −0.20 to +0.20 step 0.005, 81 values
  - `c1.state_at(sim, sim_coarse, tau, delta) -> str`
  - `c1.grounded_fails(case_scores, tau, delta=0.0) -> bool`
  - `c1.fused_fails(case_scores, case_b, tau, delta=0.0) -> bool`
  - `c1.abstain_rate(stream_a, tau, delta=0.0) -> float`
  - `c1.sweep_delta(stream_a, stream_b, case_ids, y_true, tau=BASELINE_TAU, grid=DELTA_GRID) -> list[dict]`
    with rows `{"delta", "arms", "abstain_rate"}`, mirroring `sweep_tau`'s row shape so the
    existing rendering helpers apply unchanged.

  `state_at_tau` is kept as a thin delegate so the existing τ report and its tests are untouched.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_c1_arms.py`:

```python
def test_state_at_matches_the_verifier_on_the_same_inputs():
    """The report must not grade the verifier against a rule it does not use."""
    from ragregen.verify import grounded as g

    class D:
        def all_boxes(self, image, phrase, max_boxes=8):
            return [((0.0, 0.0, 8.0, 8.0), 0.9)]

    class S:
        def score(self, crop, phrase):
            return {"fine": 0.40, "coarse": 0.55}[phrase]

    from PIL import Image
    gv = g.GroundedVerifier(D(), S(), tau=0.25, delta=0.05)
    out = gv.score(Image.new("RGB", (16, 16)),
                   [Concept("fine", "subject", "coarse")])
    live = out["fine"]
    assert c1.state_at(live.sim, live.sim_coarse, 0.25, 0.05) == live.state


def test_state_at_is_missing_below_tau_even_when_the_margin_is_wide():
    assert c1.state_at(0.10, 0.05, 0.25, 0.0) == "MISSING"


def test_state_at_abstains_on_a_none_similarity():
    assert c1.state_at(None, None, 0.25, 0.5) == "ABSTAIN"


def test_state_at_ignores_delta_when_no_coarse_similarity_exists():
    assert c1.state_at(0.30, None, 0.25, 0.9) == "PRESENT"


def test_fine_mismatch_makes_the_grounded_arm_fail():
    case = {"p": {"sim": 0.40, "sim_coarse": 0.55}}
    assert c1.grounded_fails(case, tau=0.25, delta=0.05) is True
    assert c1.grounded_fails(case, tau=0.25, delta=-0.5) is False


def test_delta_grid_spans_both_signs_and_includes_zero():
    assert c1.DELTA_GRID[0] == -0.2
    assert c1.DELTA_GRID[-1] == 0.2
    assert 0.0 in c1.DELTA_GRID
    assert len(c1.DELTA_GRID) == 81


def test_sweep_delta_returns_one_row_per_grid_value():
    stream_a = {"c0": {"p": {"sim": 0.40, "sim_coarse": 0.55}}}
    stream_b = {"c0": {"ok": True, "degenerate": False}}
    rows = c1.sweep_delta(stream_a, stream_b, ["c0"], [True],
                          grid=(-0.1, 0.0, 0.1))
    assert [r["delta"] for r in rows] == [-0.1, 0.0, 0.1]
    assert rows[0]["arms"]["grounded"].recall == 0.0
    assert rows[2]["arms"]["grounded"].recall == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_c1_arms.py -k "state_at or delta" -v`
Expected: FAIL — `AttributeError: module 'ragregen.c1' has no attribute 'state_at'`

- [ ] **Step 3: Add the constants and `state_at`**

In `ragregen/c1.py`, below `TAU_GRID`:

```python
#: tau is pinned here for all fine-grained reporting. C1 showed tau* was fitted
#: to noise -- a two-cell peak at the edge of the grid -- so delta is the only
#: knob selected on this data (fine-grained spec §5).
BASELINE_TAU = 0.25

#: delta values swept: -0.20 to +0.20 in steps of 0.005. Negative delta fails
#: only when the coarse term strictly beats the fine term by a margin; zero
#: fails when coarse >= fine; positive delta is more aggressive.
DELTA_GRID: tuple[float, ...] = tuple(round(0.005 * i, 3) for i in range(-40, 41))

#: Grounded states that fail a case. Mirrors fusion.FAILING_STATES; diverging
#: would grade the verifier against a rule it does not use.
FAILING_STATES = ("MISSING", "FINE_MISMATCH")


def state_at(sim: float | None, sim_coarse: float | None,
             tau: float, delta: float) -> str:
    """Recover a ConceptScore state from cached similarities.

    Mirrors GroundedVerifier._score_one exactly, including the precedence of
    MISSING over FINE_MISMATCH and the strict `<` on the margin.
    """
    if sim is None:
        return "ABSTAIN"
    if sim < tau:
        return "MISSING"
    # round() mirrors grounded.py exactly. Without it, 0.60 - 0.50 is
    # 0.09999999999999998 and a margin that should sit exactly on delta falls
    # below it. The two implementations must agree bit for bit or the report
    # grades the verifier against a rule it does not use.
    if sim_coarse is not None and round(sim - sim_coarse, 9) < delta:
        return "FINE_MISMATCH"
    return "PRESENT"
```

- [ ] **Step 4: Delegate `state_at_tau` and thread `delta` through the arms**

```python
def state_at_tau(sim: float | None, tau: float) -> str:
    """Pre-contrastive state recovery, kept for the tau report.

    Equivalent to state_at with no coarse similarity, which can never yield
    FINE_MISMATCH.
    """
    return state_at(sim, None, tau, 0.0)


def grounded_fails(case_scores: dict, tau: float, delta: float = 0.0) -> bool:
    """Stream A fails a case iff some concept is MISSING or FINE_MISMATCH."""
    return any(
        state_at(s.get("sim"), s.get("sim_coarse"), tau, delta) in FAILING_STATES
        for s in case_scores.values()
    )


def fused_fails(case_scores: dict, case_b: dict, tau: float,
                delta: float = 0.0) -> bool:
    """Mirrors fusion.fuse: ok = not (grounded_failed or semantic_failed)."""
    return grounded_fails(case_scores, tau, delta) or semantic_fails(case_b)


def abstain_rate(stream_a: dict, tau: float, delta: float = 0.0) -> float:
    """Fraction of scored concepts that abstained.

    Reported because a high rate means Stream A is inert and `fused` is
    silently just Stream B wearing a second name. Independent of delta;
    the parameter exists so callers can pass a sweep row through uniformly.
    """
    states = [state_at(s.get("sim"), s.get("sim_coarse"), tau, delta)
              for case in stream_a.values() for s in case.values()]
    if not states:
        return 0.0
    return sum(1 for s in states if s == "ABSTAIN") / len(states)
```

- [ ] **Step 5: Add `sweep_delta`**

Directly after `sweep_tau`:

```python
def sweep_delta(stream_a: dict, stream_b: dict, case_ids: list[str],
                y_true: list[bool], tau: float = BASELINE_TAU,
                grid=DELTA_GRID) -> list[dict]:
    """One row per delta at a pinned tau. Row shape mirrors sweep_tau."""
    for cid in case_ids:
        if cid not in stream_a:
            raise KeyError(f"case '{cid}' missing from stream_a")
        if cid not in stream_b:
            raise KeyError(f"case '{cid}' missing from stream_b")

    rows = []
    for delta in grid:
        preds = {
            "grounded": [grounded_fails(stream_a[c], tau, delta) for c in case_ids],
            "semantic": [semantic_fails(stream_b[c]) for c in case_ids],
            "fused": [fused_fails(stream_a[c], stream_b[c], tau, delta)
                      for c in case_ids],
        }
        rows.append({
            "delta": delta,
            "arms": {name: confusion(y_true, p) for name, p in preds.items()},
            "abstain_rate": abstain_rate(
                {c: stream_a[c] for c in case_ids}, tau, delta),
        })
    return rows
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_c1_arms.py tests/test_c1_metrics.py -v`
Expected: PASS, including every pre-existing τ test.

- [ ] **Step 7: Mutation-check the parity test**

Change `state_at`'s margin comparison to `<=`. Run:

Run: `$PY -m pytest tests/test_c1_arms.py::test_state_at_matches_the_verifier_on_the_same_inputs -v`
Expected: this specific test still passes (the fixture is not on the boundary), but
`test_fine_mismatch_makes_the_grounded_arm_fail` may not catch it either. **If neither fails,
add a boundary case to the parity test** where `sim - sim_coarse == delta` exactly, so the
mutation is caught. Paste the real failure output, then revert.

- [ ] **Step 8: Run the full suite**

Run: `$PY -m pytest`
Expected: 216 passed, 9 deselected.

- [ ] **Step 9: Commit**

```bash
git add ragregen/c1.py tests/test_c1_arms.py
git commit -m "feat: offline state recovery over tau and delta"
```

---

## Task 7: The pre-registered decision rule

**Files:**
- Modify: `ragregen/c1.py` (append after `sweep_delta`)
- Test: `tests/test_c1_rule.py` (create)

**Interfaces:**
- Consumes: `state_at`, `grounded_fails`, `BASELINE_TAU`, `DELTA_GRID` (Task 6).
- Produces:
  - `c1.baseline_split(stream_a, stream_b, case_ids, y_true, tau=BASELINE_TAU) -> tuple[list[str], list[str]]`
    returning `(caught_ids, missed_ids)` among the true-fail cases, with δ disabled.
  - `c1.RuleResult` dataclass: `delta, caught, false_positives, gross_retained, passes`.
  - `c1.evaluate_rule(stream_a, stream_b, case_ids, y_true, baseline_caught, baseline_missed, delta, tau=BASELINE_TAU, min_catch=4) -> RuleResult`
  - `c1.robust_band(results, min_consecutive=3) -> list[RuleResult]` — the longest run of
    consecutive passing δ values, empty if none reaches `min_consecutive`.

  Task 8 renders these.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_c1_rule.py`:

```python
# tests/test_c1_rule.py
from ragregen import c1


def _streams(spec):
    """spec: {case_id: (sim, sim_coarse, semantic_ok)}"""
    a = {cid: {"p": {"sim": s, "sim_coarse": sc}} for cid, (s, sc, _) in spec.items()}
    b = {cid: {"ok": ok, "degenerate": False} for cid, (_, _, ok) in spec.items()}
    return a, b


def test_baseline_split_separates_caught_from_missed_with_delta_disabled():
    spec = {
        "gross": (0.10, 0.05, True),   # below tau -> caught by MISSING
        "fine":  (0.40, 0.55, True),   # above tau -> missed at delta=0
        "good":  (0.80, 0.20, True),   # a true pass
    }
    a, b = _streams(spec)
    caught, missed = c1.baseline_split(a, b, ["gross", "fine", "good"],
                                       [True, True, False])
    assert caught == ["gross"]
    assert missed == ["fine"]


def test_rule_passes_when_enough_misses_are_caught_without_false_positives():
    spec = {
        "gross": (0.10, 0.05, True),
        "fine1": (0.40, 0.55, True),
        "fine2": (0.41, 0.56, True),
        "good":  (0.80, 0.20, True),
    }
    ids = ["gross", "fine1", "fine2", "good"]
    y = [True, True, True, False]
    a, b = _streams(spec)
    caught, missed = c1.baseline_split(a, b, ids, y)
    r = c1.evaluate_rule(a, b, ids, y, caught, missed, delta=0.05, min_catch=2)
    assert r.caught == ["fine1", "fine2"]
    assert r.false_positives == []
    assert r.gross_retained is True
    assert r.passes is True


def test_rule_fails_when_a_true_pass_is_flagged():
    spec = {
        "fine1": (0.40, 0.55, True),
        "good":  (0.50, 0.52, True),   # margin -0.02, caught at delta=0.05
    }
    ids = ["fine1", "good"]
    y = [True, False]
    a, b = _streams(spec)
    caught, missed = c1.baseline_split(a, b, ids, y)
    r = c1.evaluate_rule(a, b, ids, y, caught, missed, delta=0.05, min_catch=1)
    assert r.false_positives == ["good"]
    assert r.passes is False, "precision must be 1.000"


def test_robust_band_requires_consecutive_passes():
    mk = lambda d, p: c1.RuleResult(delta=d, caught=[], false_positives=[],
                                    gross_retained=True, passes=p)
    results = [mk(0.00, True), mk(0.01, False), mk(0.02, True),
               mk(0.03, True), mk(0.04, True), mk(0.05, False)]
    band = c1.robust_band(results, min_consecutive=3)
    assert [r.delta for r in band] == [0.02, 0.03, 0.04]


def test_robust_band_is_empty_when_no_run_is_long_enough():
    mk = lambda d, p: c1.RuleResult(delta=d, caught=[], false_positives=[],
                                    gross_retained=True, passes=p)
    results = [mk(0.00, True), mk(0.01, True), mk(0.02, False)]
    assert c1.robust_band(results, min_consecutive=3) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_c1_rule.py -v`
Expected: FAIL — `AttributeError: module 'ragregen.c1' has no attribute 'baseline_split'`

- [ ] **Step 3: Implement the baseline split**

Append to `ragregen/c1.py`:

```python
def baseline_split(stream_a: dict, stream_b: dict, case_ids: list[str],
                   y_true: list[bool], tau: float = BASELINE_TAU
                   ) -> tuple[list[str], list[str]]:
    """Which true failures the fused arm catches with delta disabled.

    This is the pre-registered baseline the decision rule is graded against.
    It is recomputed rather than taken from the published C1 numbers, which
    were reported at tau* = 0.02 and may not have the same membership here
    (fine-grained spec §5).
    """
    caught, missed = [], []
    for cid, is_fail in zip(case_ids, y_true):
        if not is_fail:
            continue
        if fused_fails(stream_a[cid], stream_b[cid], tau, 0.0):
            caught.append(cid)
        else:
            missed.append(cid)
    return caught, missed
```

- [ ] **Step 4: Implement the rule and the robustness band**

```python
@dataclass(frozen=True)
class RuleResult:
    delta: float
    caught: list[str]
    false_positives: list[str]
    gross_retained: bool
    passes: bool


def evaluate_rule(stream_a: dict, stream_b: dict, case_ids: list[str],
                  y_true: list[bool], baseline_caught: list[str],
                  baseline_missed: list[str], delta: float,
                  tau: float = BASELINE_TAU, min_catch: int = 4) -> RuleResult:
    """The three pre-registered criteria at one delta (fine-grained spec §5)."""
    fails = {cid: fused_fails(stream_a[cid], stream_b[cid], tau, delta)
             for cid in case_ids}

    caught = [cid for cid in baseline_missed if fails[cid]]
    false_positives = [cid for cid, is_fail in zip(case_ids, y_true)
                       if not is_fail and fails[cid]]
    gross_retained = all(fails[cid] for cid in baseline_caught)

    passes = (len(caught) >= min_catch
              and not false_positives
              and gross_retained)
    return RuleResult(delta=delta, caught=caught,
                      false_positives=false_positives,
                      gross_retained=gross_retained, passes=passes)


def robust_band(results: list[RuleResult],
                min_consecutive: int = 3) -> list[RuleResult]:
    """The longest run of consecutive passing deltas, if it is long enough.

    A result surviving at exactly one grid point is the tau* mistake wearing a
    different letter, so a lone pass is reported as no band at all.
    """
    best: list[RuleResult] = []
    run: list[RuleResult] = []
    for r in results:
        run = run + [r] if r.passes else []
        if len(run) > len(best):
            best = run
    return best if len(best) >= min_consecutive else []
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_c1_rule.py -v`
Expected: PASS.

- [ ] **Step 6: Mutation-check the precision guard**

Temporarily drop `and not false_positives` from `passes`. Run:

Run: `$PY -m pytest tests/test_c1_rule.py::test_rule_fails_when_a_true_pass_is_flagged -v`
Expected: FAIL. Paste the real output, then revert.

Repeat with `min_consecutive` ignored in `robust_band` (return `best` unconditionally) and
confirm `test_robust_band_is_empty_when_no_run_is_long_enough` fails.

- [ ] **Step 7: Run the full suite**

Run: `$PY -m pytest`
Expected: 221 passed, 9 deselected.

- [ ] **Step 8: Commit**

```bash
git add ragregen/c1.py tests/test_c1_rule.py
git commit -m "feat: pre-registered fine-grained decision rule and robustness band"
```

---

## Task 8: Report the mechanism split and the rule verdict

**Files:**
- Modify: `ragregen/c1.py` (add `render_finegrained`), `scripts/c1_report.py:31-80` (`main`)
- Test: `tests/test_c1_render.py`

**Interfaces:**
- Consumes: `sweep_delta`, `baseline_split`, `evaluate_rule`, `robust_band` (Tasks 6-7).
- Produces: `c1.render_finegrained(stream_a, stream_b, case_ids, y_true, column, tau=BASELINE_TAU) -> tuple[str, dict]`
  returning the markdown section and the JSON artifact fragment. `c1_report.py` appends the
  section to `c1.md` and the fragment under `artifact[column]["finegrained"]`.

The existing τ report and `render_report` are left untouched — the published C1 numbers stay
reproducible, and this is added alongside.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_c1_render.py`:

```python
def test_finegrained_section_names_the_verdict_and_the_missed_cases():
    stream_a = {
        "gross": {"p": {"sim": 0.10, "sim_coarse": 0.05}},
        "fine1": {"p": {"sim": 0.40, "sim_coarse": 0.55}},
        "fine2": {"p": {"sim": 0.41, "sim_coarse": 0.56}},
        "good":  {"p": {"sim": 0.80, "sim_coarse": 0.20}},
    }
    stream_b = {c: {"ok": True, "degenerate": False} for c in stream_a}
    ids = ["gross", "fine1", "fine2", "good"]
    y = [True, True, True, False]

    md, art = c1.render_finegrained(stream_a, stream_b, ids, y,
                                    column="verdict_identity")
    assert "fine1" in md and "fine2" in md
    assert "gross" in md
    assert art["baseline_caught"] == ["gross"]
    assert art["baseline_missed"] == ["fine1", "fine2"]
    assert "tau" in art and art["tau"] == c1.BASELINE_TAU


def test_finegrained_section_reports_no_band_when_the_rule_never_holds():
    stream_a = {"fine1": {"p": {"sim": 0.40, "sim_coarse": 0.55}},
                "good":  {"p": {"sim": 0.50, "sim_coarse": 0.52}}}
    stream_b = {c: {"ok": True, "degenerate": False} for c in stream_a}
    md, art = c1.render_finegrained(stream_a, stream_b, ["fine1", "good"],
                                    [True, False], column="verdict_identity")
    assert art["band"] == []
    assert "NOT MET" in md


def test_delta_table_prints_every_0_02_step_without_float_gaps():
    """Float modulo on this grid silently drops 6 of the 21 rows."""
    stream_a = {"c0": {"p": {"sim": 0.40, "sim_coarse": 0.55}}}
    stream_b = {"c0": {"ok": True, "degenerate": False}}
    md, _ = c1.render_finegrained(stream_a, stream_b, ["c0"], [True],
                                  column="verdict_identity")
    body = md.split("### Fused arm across delta")[1]
    rows = [ln for ln in body.splitlines()
            if ln.startswith("|") and "---" not in ln and "delta" not in ln]
    assert len(rows) == 21, f"expected 21 delta rows, got {len(rows)}"
    assert "| -0.2 |" in body and "| 0.2 |" in body
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_c1_render.py -k finegrained -v`
Expected: FAIL — `AttributeError: module 'ragregen.c1' has no attribute 'render_finegrained'`

- [ ] **Step 3: Implement the renderer**

Append to `ragregen/c1.py`:

```python
def render_finegrained(stream_a: dict, stream_b: dict, case_ids: list[str],
                       y_true: list[bool], column: str,
                       tau: float = BASELINE_TAU) -> tuple[str, dict]:
    """The fine-grained section: baseline, mechanism split, rule verdict."""
    caught0, missed0 = baseline_split(stream_a, stream_b, case_ids, y_true, tau)
    results = [evaluate_rule(stream_a, stream_b, case_ids, y_true,
                             caught0, missed0, delta=d, tau=tau)
               for d in DELTA_GRID]
    band = robust_band(results)
    sweep = sweep_delta(stream_a, stream_b, case_ids, y_true, tau=tau)

    met = bool(band)
    chosen = band[len(band) // 2] if met else None

    L = [f"## Fine-grained verification — `{column}`", "",
         f"tau pinned at **{tau}** (not fitted). delta swept over "
         f"{len(DELTA_GRID)} values from {DELTA_GRID[0]} to {DELTA_GRID[-1]}.", "",
         f"**Baseline at tau = {tau}, delta disabled:** "
         f"{len(caught0)} caught, {len(missed0)} missed.", "",
         f"- caught: {', '.join(caught0) if caught0 else '(none)'}",
         f"- missed: {', '.join(missed0) if missed0 else '(none)'}", ""]

    if met:
        L += [f"### RULE MET across delta {band[0].delta} to {band[-1].delta} "
              f"({len(band)} consecutive values)", "",
              f"At delta = {chosen.delta}: **{len(chosen.caught)} of "
              f"{len(missed0)}** previously-missed cases now caught "
              f"({', '.join(chosen.caught) if chosen.caught else 'none'}), "
              f"**{len(chosen.false_positives)} false positives**, "
              f"gross-category catches retained: "
              f"**{'yes' if chosen.gross_retained else 'NO'}**.", ""]
    else:
        best = max(results, key=lambda r: len(r.caught))
        L += ["### RULE NOT MET", "",
              "No band of 3 or more consecutive delta values satisfies all "
              "three criteria. The best single delta was "
              f"{best.delta}, catching {len(best.caught)} of {len(missed0)} "
              f"with {len(best.false_positives)} false positives. "
              "Per spec §5 the conclusion is that phrase-level contrast is "
              "insufficient, not that delta needs more tuning.", ""]

    L += ["### Fused arm across delta", "", "| delta | prec | recall | bal-acc | MCC |",
          "|---|---|---|---|---|"]
    for r in sweep:
        # Every 0.02, via integer arithmetic. Float modulo on this grid drops
        # -0.2, -0.14, -0.1 and three others to representation error, silently
        # printing 15 rows where 21 are meant -- verified, not hypothetical.
        if round(r["delta"] * 1000) % 20 != 0:
            continue
        m = r["arms"]["fused"]
        L.append(f"| {r['delta']} | {m.precision:.3f} | {m.recall:.3f} | "
                 f"{m.balanced_accuracy:.3f} | {m.mcc:+.3f} |")

    artifact = {
        "tau": tau,
        "baseline_caught": caught0,
        "baseline_missed": missed0,
        "rule_met": met,
        "band": [r.delta for r in band],
        "chosen_delta": chosen.delta if met else None,
        "caught_at_chosen": chosen.caught if met else [],
        "false_positives_at_chosen": chosen.false_positives if met else [],
    }
    return "\n".join(L), artifact
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_c1_render.py -v`
Expected: PASS.

- [ ] **Step 5: Wire it into `c1_report.py`**

Inside `main`'s `for column in COLUMNS:` loop, after the existing `sections.append(...)`:

```python
        fg_md, fg_art = c1.render_finegrained(stream_a, stream_b, case_ids,
                                              y_true, column=column)
        sections.append(fg_md)
```

and inside the `artifact[column] = {...}` dict, add the key:

```python
            "finegrained": fg_art,
```

`fg_art` must be computed before the dict literal; move the `render_finegrained` call above it
if the ordering does not work out.

- [ ] **Step 6: Run the full suite**

Run: `$PY -m pytest`
Expected: 224 passed, 9 deselected.

- [ ] **Step 7: Commit**

```bash
git add ragregen/c1.py scripts/c1_report.py tests/test_c1_render.py
git commit -m "feat: report the fine-grained mechanism split and rule verdict"
```

---

## Task 9: Coarse terms for the 22 cases, and the RUNBOOK

**Files:**
- Modify: `configs/dataset.yaml` (22 cases), `configs/dataset.example.yaml`, `docs/RUNBOOK.md`
- Test: none (data and docs; Task 2's loader tests already cover the validation)

**Interfaces:**
- Consumes: the `coarse:` key required by Task 2.
- Produces: a `dataset.yaml` that loads cleanly and a documented operator contract.

- [ ] **Step 1: Add `coarse:` to every case in `configs/dataset.yaml`**

Insert a `coarse:` line directly below each `concept:` line, using exactly these values. They
are authored from the concept names alone, per the spec's integrity rule — do **not** adjust
any of them to match a failure you have seen in the labels.

| case id | concept | coarse |
|---|---|---|
| african_grey_parrot | African grey parrot | parrot |
| amur_leopard | Amur leopard | leopard |
| anas_platyrhynchos | Anas platyrhynchos | duck |
| axolotl | axolotl | salamander |
| boston_bull | Boston bull | dog |
| durian | durian | fruit |
| love_in_a_mist | love-in-a-mist | flower |
| monkey_puzzle_tree | monkey puzzle tree | tree |
| zalophus_californianus | Zalophus californianus | sea lion |
| bonsai_tree | bonsai tree | tree |
| cardinal_bird | cardinal bird | bird |
| hedgehog | hedgehog | small mammal |
| cactus | cactus | plant |
| flamingo | flamingo | bird |
| fox | fox | canine |
| golden_retriever | golden retriever | dog |
| panda | panda | bear |
| polar_bear | polar bear | bear |
| red_ferrari | red Ferrari | sports car |
| stack_of_books | stack of books | books |
| sushi | sushi | food |
| violin | violin | string instrument |

- [ ] **Step 2: Verify the config loads**

Run: `$PY -c "from ragregen import config; ds = config.load_dataset(); print(len(ds.cases), 'cases'); print([c.coarse for c in ds.cases])"`
Expected: `22 cases` followed by the 22 coarse terms. Any `ValueError` names the offending case — fix and re-run.

- [ ] **Step 3: Update `configs/dataset.example.yaml`**

Add `coarse:` to the example case with a comment stating the contract:

```yaml
    concept: "African grey parrot"
    # The concept's superordinate category, used for the fine-grained
    # contrastive check: Stream A fails the draft when the crop looks more
    # like the coarse term than the fine one. Author it from the concept
    # name alone -- never from a draft you have already looked at, which
    # would fit the test to its own answer key.
    coarse: "parrot"
```

- [ ] **Step 4: Document it in `docs/RUNBOOK.md` §3.1**

Add to the dataset-config section, matching the surrounding prose style:

```markdown
**`coarse`** (required). The concept's superordinate category — `parrot` for
`African grey parrot`, `dog` for `Boston bull`. Stream A scores the detected
crop against both terms and fails the draft when the coarse term wins by more
than the configured margin. This is what lets the verifier catch "a leopard
that is merely a leopard" rather than only gross category errors.

Two rules:

- Author it from the concept name, before looking at any draft. Choosing a
  coarse term because a draft happens to resemble it fits the test to its own
  answer key.
- It must differ from `concept`. Identical terms make the margin zero for
  every case and silently disable the check; `validate` rejects this.

For basic-level concepts with no tighter category above them (`durian`,
`sushi`, `violin`), pick the nearest honest superordinate (`fruit`, `food`,
`string instrument`). The contrastive check carries little signal for these,
and that is expected — the absolute threshold remains their mechanism.
```

- [ ] **Step 5: Run the full suite**

Run: `$PY -m pytest`
Expected: 224 passed, 9 deselected.

- [ ] **Step 6: Commit**

```bash
git add configs/dataset.yaml configs/dataset.example.yaml docs/RUNBOOK.md
git commit -m "feat: coarse superordinate terms for all 22 cases"
```

---

## Task 10: Run it and report the finding

**Files:**
- Create: `docs/findings/2026-07-27-finegrained-result.md`
- Outputs (gitignored): `outputs/screen_20260726_233320/stream_a.json`, `c1.md`, `c1.json`

**Interfaces:**
- Consumes: everything above.
- Produces: the measured verdict that unblocks or redirects the repair loop.

**This is the only task that touches the GPU.** Check free VRAM first — Plan 1 recorded another
process holding ~10.7 GB of the 24.5 GB card.

- [ ] **Step 1: Confirm the suite is green before spending GPU time**

Run: `$PY -m pytest`
Expected: 224 passed, 9 deselected.

- [ ] **Step 2: Re-run Stream A with `--force`**

The cached `stream_a.json` predates `sim_coarse` and cannot be patched offline.

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
./scripts/run.sh score-a --labels outputs/screen_20260726_233320/labels.csv --force
```

Expected: ~4 minutes, 22 cases scored, no errors. Stream B is **not** re-run — its cached
`stream_b.json` is unchanged by this plan.

- [ ] **Step 3: Verify both similarities are present**

Run:

```bash
$PY -c "
import json
d = json.load(open('outputs/screen_20260726_233320/stream_a.json'))
n = sum(1 for c in d.values() for s in c.values() if s.get('sim_coarse') is not None)
print(len(d), 'cases;', n, 'concepts with a coarse similarity')
"
```

Expected: `22 cases; 22 concepts with a coarse similarity` — one per case, since only the
target concept carries a coarse term. If the second number is 0, the `coarse` join in Task 5
is not reaching the verifier.

- [ ] **Step 4: Render the report**

```bash
./scripts/run.sh c1 --run outputs/screen_20260726_233320
```

Expected: `c1.md` and `c1.json` written, now containing a "Fine-grained verification" section
per verdict column.

- [ ] **Step 5: Read the verdict and record the baseline honestly**

From `c1.json`, read `finegrained.baseline_caught` and `finegrained.baseline_missed` for
`verdict_identity`. Compare the missed set against the published C1 split
(`african_grey_parrot`, `boston_bull`, `bonsai_tree`, `cardinal_bird`, `hedgehog`,
`golden_retriever`, `polar_bear`).

**If they differ, say so plainly in the finding.** The τ = 0.25 baseline is what the rule is
graded against; the τ\* = 0.02 split is the published reference. Do not quietly adopt whichever
is more favourable.

- [ ] **Step 6: Write the finding**

Create `docs/findings/2026-07-27-finegrained-result.md` following the structure of
`docs/findings/2026-07-27-c1-result.md`. It must state:

1. The τ = 0.25 baseline split, and whether it matches the published τ\* = 0.02 split.
2. Whether the rule was **MET** or **NOT MET**, with the δ band if met.
3. Which of the previously-missed cases are now caught, **by name**.
4. Any false positives, by name — with particular attention to `cactus`, `stack_of_books` and
   `sushi`, the "right concept, wrong aesthetics" controls where precision breaks first.
5. Whether `hedgehog` was caught. The spec predicts it will not be: it is basic-level, and its
   failure ("spikes too sparse") is a rendering defect with no categorical contrast available.
   A catch there is worth examining for the wrong reason.
6. The recommendation for the repair loop: proceed, or move to image-prototype scoring as a
   separate design.

Keep the caveats section: n=22, τ pinned but δ still selected in-sample, no held-out split.

- [ ] **Step 7: Commit**

```bash
git add docs/findings/2026-07-27-finegrained-result.md
git commit -m "docs: fine-grained verifier result and the rule verdict"
```

`outputs/` is gitignored, so `c1.md` and `c1.json` stay with the run rather than in git.

---

## Self-Review

**Spec coverage.**

| spec section | task |
|---|---|
| §2 the fix: `FINE_MISMATCH`, margin, state precedence | 3 |
| §2 distinct state rather than MISSING | 3, 4 |
| §3 `coarse:` config field, authored blind, `coarse != concept` | 2, 9 |
| §4 one GPU sweep, Stream B reused, `sim_coarse` persisted | 5, 10 |
| §5 τ pinned at 0.25 | 6 |
| §5 baseline wrinkle recomputed at pinned τ | 7 (`baseline_split`), 10 step 5 |
| §5 ≥4/7 caught, precision 1.000, gross retained | 7 (`evaluate_rule`) |
| §5 robustness across ≥3 consecutive δ | 7 (`robust_band`) |
| §5 δ grid −0.20…+0.20 step 0.005 | 6 (`DELTA_GRID`) |
| §5 "if not met, this is a different design, not a tuning pass" | 8 (rendered), 10 step 6 |
| §6 exclusions | not implemented, by design |
| §7 testing rules, mutation evidence | every task's mutation step |
| §8 deliverable: re-run report + rule statement | 8, 10 |

**Placeholder scan.** No TBDs. Every code step carries real code. Three steps deliberately
instruct the implementer to check a real signature rather than trust this plan
(Task 4 Step 1 `SemanticVerdict`, Task 5 Step 1 test-file imports, Task 8 Step 5 dict ordering)
— that is the anti-vacuous-test rule applied to the plan itself, not a placeholder.

**Type consistency.** `state_at(sim, sim_coarse, tau, delta)` is used with that argument order in
Tasks 6, 7 and 8. `FAILING_STATES` is defined twice on purpose — `fusion.FAILING_STATES` (Task 4)
for the live path, `c1.FAILING_STATES` (Task 6) for the offline path — and Task 6's docstring
states they must mirror each other. `RuleResult` fields (`delta`, `caught`, `false_positives`,
`gross_retained`, `passes`) match between Task 7's definition and Task 8's renderer.

**Known gap, accepted.** The test counts quoted at each task's full-suite step (188 → 223) are
projections from the verified 188-passing baseline, assuming every test written here lands as one
item (the parametrized δ-validation test in Task 3 counts as two). If the real count differs, the
deliverable is a green suite, not a matching number.
