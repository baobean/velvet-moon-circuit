# Hybrid Verifier Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an unbiased 24-case smoke validation of the frozen Qwen semantic-plus-reranker policy, then promote it to the 92-case experiment only if every frozen gate passes.

**Architecture:** Keep policy decisions and statistics in a pure CPU module, use the existing model-scoring scripts to create raw artifacts, and add a bounded multi-attempt open-loop mode solely to generate a reusable candidate bank. Dataset construction is deterministic from local Wikimedia assets; DINO remains an answer key and never enters selection.

**Tech Stack:** Python 3.11, pytest, PyYAML, PIL, existing FLUX Kontext/Qwen3-VL/Qwen2.5-VL/DINO pipeline, Bash GPU waiter.

**Spec:** `docs/superpowers/specs/2026-08-18-hybrid-verifier-validation-design.md`

## Global Constraints

- Freeze reranker route threshold at `0.30078125`, strict `<` comparison.
- Freeze repair text margin at `0.02`, inclusive `>=` comparison.
- Require strictly positive reference-relevance improvement.
- Tie-break by text relevance, reference relevance, then lower attempt number.
- Never expose held-out references, DINO, or human labels to routing or selection.
- Do not modify the passing inpainting implementation or its prompt during this experiment.
- Use runtime Asia/Ho_Chi_Minh timestamps; do not embed copied run dates.
- Preserve every pre-existing dirty-worktree change.
- Because the shared tree is already dirty across files this plan touches, use explicit diff checkpoints instead of commits; committing those files would capture unrelated existing changes.

---

### Task 1: Deterministic holdout dataset emitter

**Files:**

- Create: `scripts/make_verifier_holdout.py`
- Create: `tests/test_make_verifier_holdout.py`
- Create at runtime: `configs/dataset_verifier_holdout.yaml`
- Create at runtime: `configs/dataset_verifier_holdout_rare.yaml`
- Create at runtime: `configs/dataset_verifier_holdout_control.yaml`

**Interfaces:**

- Consumes: sibling asset roots `../rag-edit/data/images` and `../rag-edit/data/scenes`.
- Produces: `build_entries(rare_root: Path, control_root: Path) -> tuple[list[dict], list[dict]]` and `emit_dataset(name: str, entries: list[dict]) -> str`.

- [ ] **Step 1: Write failing tests for identity, cohort, reference count, and disk validation**

```python
def test_build_entries_emits_twelve_rare_and_twelve_controls(asset_tree):
    rare, controls = build_entries(asset_tree / "images", asset_tree / "scenes")
    assert len(rare) == len(controls) == 12
    assert {row["kind"] for row in rare} == {"target"}
    assert {row["kind"] for row in controls} == {"control"}

def test_rare_cases_have_three_edit_refs_and_one_heldout(asset_tree):
    rare, _ = build_entries(asset_tree / "images", asset_tree / "scenes")
    assert all(len(row["gt_refs"]) == 4 for row in rare)

def test_controls_have_two_edit_refs_and_one_heldout(asset_tree):
    _, controls = build_entries(asset_tree / "images", asset_tree / "scenes")
    assert all(len(row["gt_refs"]) == 3 for row in controls)
```

- [ ] **Step 2: Run the focused tests and confirm the missing-module failure**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_make_verifier_holdout.py -q`

Expected: FAIL because `scripts.make_verifier_holdout` does not exist.

- [ ] **Step 3: Implement the fixed manifest and YAML emitter**

Define the twelve target ids from spec section 3.1 and these twelve controls:

```python
CONTROL_CASES = (
    ("apple", "apple_fruit", "fruit"),
    ("barn", "barn_building", "building"),
    ("church", "church_building", "building"),
    ("daisy", "daisy_flower", "flower"),
    ("deer", "deer", "animal"),
    ("donkey", "donkey", "animal"),
    ("labrador_retriever", "labrador_retriever", "dog"),
    ("lizard", "lizard_on_a_rock", "reptile"),
    ("pine_tree", "pine_tree", "tree"),
    ("pizza", "pizza", "food"),
    ("stone_cottage", "stone_cottage", "building"),
    ("tiger", "tiger", "big cat"),
)
```

Load rare display names, captions, and categories from
`../rag-edit/configs/concepts.json`; use explicit groundable coarse terms in
the manifest. Sort every reference path by filename, fail unless counts are
exactly four/three, emit full/rare/control YAMLs, and round-trip all three
through `config.load_dataset` before returning success.

- [ ] **Step 4: Run the focused tests**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_make_verifier_holdout.py tests/test_config.py -q`

Expected: PASS.

- [ ] **Step 5: Generate and validate the actual configs**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python scripts/make_verifier_holdout.py`

Run: `./scripts/run.sh validate --dataset configs/dataset_verifier_holdout.yaml`

Expected: 24 cases and no validation errors.

- [ ] **Step 6: Inspect the scoped diff checkpoint**

Run: `git diff -- scripts/make_verifier_holdout.py tests/test_make_verifier_holdout.py configs/dataset_verifier_holdout.yaml configs/dataset_verifier_holdout_rare.yaml configs/dataset_verifier_holdout_control.yaml`

Expected: only deterministic dataset-emitter changes.

---

### Task 2: Pure frozen hybrid policy

**Files:**

- Create: `ragregen/hybrid_verifier.py`
- Create: `tests/test_hybrid_verifier.py`

**Interfaces:**

- Produces: `route_draft(semantic_ok: bool, reference_relevance: float) -> bool`.
- Produces: `select_repair(scores: Mapping[str, Mapping[str, float]]) -> str`.
- Produces: `evaluate(labels: Mapping[str, str], semantic: Mapping[str, dict], reranker: dict, dino: dict, cohorts: Mapping[str, str]) -> dict`.
- Consumes score records shaped as `{"draft": {"text_relevance": float, "reference_relevance": float}, "attempt_1": ...}`.

- [ ] **Step 1: Write failing boundary and tie-break tests**

```python
def test_route_is_union_and_threshold_equality_passes():
    assert route_draft(False, 0.9)
    assert route_draft(True, 0.3007)
    assert not route_draft(True, 0.30078125)

def test_selector_requires_both_improvements_and_inclusive_text_margin():
    scores = {
        "draft": {"text_relevance": 0.40, "reference_relevance": 0.30},
        "attempt_1": {"text_relevance": 0.42, "reference_relevance": 0.31},
        "attempt_2": {"text_relevance": 0.60, "reference_relevance": 0.29},
    }
    assert select_repair(scores) == "attempt_1"

def test_selector_falls_back_to_draft_and_breaks_ties_stably():
    assert select_repair({"draft": {"text_relevance": 0.5,
                                     "reference_relevance": 0.5}}) == "draft"
```

- [ ] **Step 2: Run the focused tests and confirm the missing-module failure**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_hybrid_verifier.py -q`

Expected: FAIL because `ragregen.hybrid_verifier` does not exist.

- [ ] **Step 3: Implement constants, routing, stable selection, confusion metrics, and gates**

Use immutable module constants:

```python
REFERENCE_THRESHOLD = 0.30078125
TEXT_MARGIN = 0.02
HARM_MARGIN = -0.02
MIN_RECALL = 0.733
MAX_FALSE_POSITIVES = 1
MIN_SIGN_ACCURACY = 0.75
MAX_HARMFUL = 1
```

Reject missing/non-finite scores with a named `ValueError`. Return per-case
decisions and every gate input so the report is auditable.

- [ ] **Step 4: Run policy tests and existing verifier tests**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_hybrid_verifier.py tests/test_verifier_eval.py tests/test_reranker_smoke.py -q`

Expected: PASS.

- [ ] **Step 5: Inspect the scoped diff checkpoint**

Run: `git diff -- ragregen/hybrid_verifier.py tests/test_hybrid_verifier.py`

Expected: a model-free policy module and deterministic unit tests only.

---

### Task 3: Resumable multi-attempt candidate-bank mode

**Files:**

- Modify: `scripts/run_pipeline.py`
- Modify: `tests/test_run_pipeline.py`

**Interfaces:**

- Adds CLI option `--open-loop-attempts N`, integer default `1`.
- Produces all `attempt_1.png` through `attempt_N.png` while retaining
  `attempt_1` as the declared no-verifier output.
- Existing `--verifier none` without the new option remains byte-for-byte
  behaviorally compatible.

- [ ] **Step 1: Write failing parser and scheduling tests**

```python
def test_open_loop_attempts_defaults_to_one():
    assert _parse_args(["--verifier", "none"]).open_loop_attempts == 1

def test_open_loop_attempts_rejects_zero_and_fused_mode():
    with pytest.raises(SystemExit):
        _parse_args(["--verifier", "none", "--open-loop-attempts", "0"])
    with pytest.raises(SystemExit):
        _parse_args(["--verifier", "fused", "--open-loop-attempts", "2"])

def test_candidate_bank_resolves_attempt_one_only_after_all_rounds(tmp_path):
    # Queue remains pending after attempt 1, then selection.json names
    # attempt_1 after the final requested round.
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_run_pipeline.py -k 'open_loop or candidate_bank' -q`

Expected: FAIL because the option and deferred resolver do not exist.

- [ ] **Step 3: Implement bounded deferred open-loop resolution**

Parse a positive integer. For `verifier=none`, set rounds to
`min(open_loop_attempts, retry_budget, max_available_refs)`, leave completed
cases pending between rounds, and resolve a case to `attempt_1` immediately
before a round for which it has no reference. After the final requested round,
resolve every remaining case to `attempt_1` and write:

```json
{
  "best": "attempt_1",
  "mode": "open_loop",
  "verifier_used": false,
  "candidate_attempts": ["attempt_1", "attempt_2"]
}
```

Resume must skip completed `regen@N` cells and finish missing later rounds.

- [ ] **Step 4: Run focused and scheduler regression tests**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_run_pipeline.py tests/test_schedule.py tests/test_schedule_crash.py -q`

Expected: PASS, including existing one-attempt behavior.

- [ ] **Step 5: Inspect the overlapping-file diff carefully**

Run: `git diff -- scripts/run_pipeline.py tests/test_run_pipeline.py`

Expected: only parser, open-loop round-count, deferred resolution, and tests;
all earlier CUDA-cleanup and inpainting changes remain intact.

---

### Task 4: Holdout report and arm materialization

**Files:**

- Create: `scripts/hybrid_verifier_report.py`
- Create: `tests/test_hybrid_verifier_report.py`
- Modify: `scripts/run.sh`

**Interfaces:**

- Consumes: dataset YAML, `labels.csv`, `stream_b.json`, Qwen reranker score
  JSON, pairwise DINO JSON, screen run, and candidate run.
- Produces: `hybrid_validation.json`, `hybrid_validation.md`, per-case CSV,
  `decisions.json`, and arm images under `arms/{draft,no_verifier,hybrid}/`.

- [ ] **Step 1: Write failing CLI tests with synthetic artifacts**

```python
def test_report_materializes_three_arms_and_frozen_decisions(tmp_path):
    rc = main(["--dataset", str(dataset), "--labels", str(labels),
               "--semantic", str(semantic), "--reranker", str(reranker),
               "--dino", str(dino), "--screen-run", str(screen),
               "--candidate-run", str(candidates), "--output-dir", str(out)])
    assert rc == 0
    assert json.loads((out / "decisions.json").read_text())["c1"]["hybrid"] == "attempt_1"
    assert (out / "arms" / "draft" / "c1.png").is_file()
```

- [ ] **Step 2: Run the test and confirm the missing-script failure**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_hybrid_verifier_report.py -q`

Expected: FAIL because `scripts.hybrid_verifier_report` does not exist.

- [ ] **Step 3: Implement strict artifact joining and Markdown rendering**

Fail on missing labeled cases, inconsistent case ids, absent draft scores, or
attempt decisions whose image is absent. Copy images rather than symlinking so
the report remains valid if a source run moves. Include frozen constants,
confusion tables, selector table, cohort DINO deltas, gates, and per-case
decisions.

- [ ] **Step 4: Add the command to `scripts/run.sh`**

Add:

```bash
hybrid-report) exec "$PY" scripts/hybrid_verifier_report.py "$@" ;;
```

and list it in the help text.

- [ ] **Step 5: Run report and command-dispatch tests**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_hybrid_verifier_report.py tests/test_report.py -q`

Expected: PASS.

- [ ] **Step 6: Inspect the scoped diff checkpoint**

Run: `git diff -- scripts/hybrid_verifier_report.py tests/test_hybrid_verifier_report.py scripts/run.sh`

Expected: report-only changes and one command dispatch.

---

### Task 5: CPU verification before GPU booking

**Files:**

- Verify all files changed by Tasks 1–4.

**Interfaces:**

- Consumes: completed implementation.
- Produces: a trustworthy CPU-green checkpoint before expensive model work.

- [ ] **Step 1: Run focused tests together**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_make_verifier_holdout.py tests/test_hybrid_verifier.py tests/test_hybrid_verifier_report.py tests/test_run_pipeline.py -q`

Expected: PASS.

- [ ] **Step 2: Run the full non-GPU suite**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest -q -m 'not gpu'`

Expected: all tests pass; only established skips/deselections remain.

- [ ] **Step 3: Check repository state without altering unrelated changes**

Run: `git status --short`

Expected: pre-existing dirty files plus the explicitly planned additions and
narrow modifications; no model caches or generated outputs tracked by git.

---

### Task 6: Generate, label, and score the fresh drafts

**Files:**

- Runtime artifacts: `outputs/screen_verifier_holdout_<TIMESTAMP>/`
- Runtime artifacts: `outputs/hybrid_validation_<TIMESTAMP>/draft_contact_sheet.png`

**Interfaces:**

- Consumes: `configs/dataset_verifier_holdout.yaml`.
- Produces: locked `labels.csv`, fresh drafts, semantic scores, and a contact
  sheet created before reranker scores are inspected.

- [ ] **Step 1: Confirm an eligible GPU and record its physical index**

Run: `nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader`

Expected: one card has at least 18 GiB free and remains stable for three
checks. Use its index in `gpu_wait.sh`; do not assume GPU 0.

- [ ] **Step 2: Generate 24 fresh drafts**

Run:

```bash
./scripts/gpu_wait.sh --gpu 2 --need 18 --stable 3 --interval 60 -- \
  ./scripts/run.sh screen --dataset configs/dataset_verifier_holdout.yaml
```

Expected: a successful screen run with 24 draft images and 24 blank label rows.
If GPU 2 is not the free card, substitute the index observed in Step 1.

- [ ] **Step 3: Create and inspect the contact sheet, then lock labels**

Use the generated screen-run path. Inspect all drafts without opening semantic
or reranker scores. Fill both `verdict` and `verdict_identity` with lowercase
`pass`, `fail`, or `exclude`; include a reason for every exclusion or ambiguous
identity. Reopen the CSV through `c1.load_labels` to verify all 24 rows parse.

- [ ] **Step 4: Score the existing semantic VLM after labels are locked**

Run:

```bash
./scripts/gpu_wait.sh --gpu 2 --need 17 --stable 3 --interval 60 -- \
  ./scripts/run.sh score-b --labels outputs/screen_verifier_holdout_<TIMESTAMP>/labels.csv
```

Expected: `stream_b.json` has exactly every non-excluded labeled case and raw
responses are preserved.

---

### Task 7: Generate the reusable candidate bank

**Files:**

- Runtime artifacts: `outputs/holdout_candidates_<TIMESTAMP>/`

**Interfaces:**

- Consumes: locked screen run and the full holdout dataset.
- Produces: three rare attempts and two control attempts per case when all
  references ground successfully.

- [ ] **Step 1: Generate the variable-depth candidate bank**

Run:

```bash
./scripts/gpu_wait.sh --gpu 2 --need 18 --stable 3 --interval 60 -- \
  ./scripts/run.sh pipeline --tag holdout_candidates --arm oracle \
    --mechanism inpaint --verifier none --open-loop-attempts 3 \
    --dataset configs/dataset_verifier_holdout.yaml \
    --screen-run outputs/screen_verifier_holdout_<TIMESTAMP>
```

Expected: each nonfailed target has `attempt_1.png` through `attempt_3.png`;
each nonfailed control has `attempt_1.png` and `attempt_2.png`, then resolves
cleanly before round 3 because only two edit references exist. Resume the
identical run directory after a transient CUDA failure.

- [ ] **Step 2: Inspect masks, prepared references, and attempts**

Generate contact sheets containing draft, mask, prepared references, raw
attempts, and blended attempts. Record reference-grounding rejections and
exclude only cases meeting spec section 3.3.

---

### Task 8: Reranker, DINO answer key, and frozen decision report

**Files:**

- Runtime artifacts under `outputs/hybrid_validation_<TIMESTAMP>/`.

**Interfaces:**

- Consumes: locked labels, semantic scores, and the single candidate-bank run.
- Produces: raw Qwen reranker scores, pairwise DINO scores, frozen decisions,
  materialized arms, and promotion decision.

- [ ] **Step 1: Score draft and candidates with Qwen3-VL-Reranker-2B**

Run `scripts/qwen_reranker_score.py` with the holdout dataset, locked screen
run, candidate-bank run, and an output path inside the validation directory.

Expected: every non-excluded case has a draft score and every existing attempt
has both text and reference scores.

- [ ] **Step 2: Compute held-out pairwise DINO scores**

Run `scripts/pairwise_dino.py` with the same dataset and candidate-bank run.

Expected: held-out references appear only in the DINO artifact metadata and
were never passed to Qwen or FLUX.

- [ ] **Step 3: Render the frozen report**

Run `./scripts/run.sh hybrid-report` with all artifact paths and a newly
timestamped output directory.

Expected: JSON, Markdown, CSV, three arm directories, and an overall
`PASS|FAIL|INCONCLUSIVE` decision determined only by spec section 6.

- [ ] **Step 4: Visually inspect every selected hybrid output**

Compare draft, reference, no-verifier output, hybrid output, and mask. Record
paste-through, pose/layout replacement, boundary artifacts, and identity
failures per case. Any material paste-through/layout failure blocks promotion.

- [ ] **Step 5: Make the promotion decision without retuning**

If every detector, selector, end-to-end, no-harm, and visual gate passes,
continue to Task 9. Otherwise stop, publish the failure report, and retain the
no-verifier inpainting result as the supported component.

---

### Task 9: Conditional production integration and 92-case rerun

**Files:**

- Modify after promotion only: `scripts/run_pipeline.py`
- Modify after promotion only: `tests/test_run_pipeline.py`
- Modify after promotion only: `scripts/report.py`
- Runtime artifacts: three paired 92-case arms.

**Interfaces:**

- Consumes: a passing frozen holdout report.
- Produces: production `--verifier hybrid` behavior identical to
  `ragregen.hybrid_verifier`, followed by the paired 92-case evaluation.

- [ ] **Step 1: Write failing scheduler integration tests**

Test retrieval-before-routing, semantic/reranker union routing, conservative
attempt eligibility, draft fallback, stable ties, resume, and raw score
persistence. Use fixture score providers; no GPU in unit tests.

- [ ] **Step 2: Implement the minimal production adapter**

Add `hybrid` to verifier choices. Persist Qwen reranker scores per attempt and
delegate all decisions to `ragregen.hybrid_verifier`; do not duplicate policy
constants in the scheduler.

- [ ] **Step 3: Run focused and full CPU tests**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_hybrid_verifier.py tests/test_run_pipeline.py tests/test_report.py -q`

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest -q -m 'not gpu'`

Expected: PASS.

- [ ] **Step 4: Launch three paired 92-case arms on the same physical GPU**

Use `configs/dataset_common.yaml`, identical seed/config, and the frozen 57k
LAION index:

1. existing drafts as baseline;
2. `--arm full --verifier none` as one-shot no-verifier repair;
3. `--arm full --verifier hybrid` as the frozen method.

Each command runs through `gpu_wait.sh` with the actually free GPU index and
an explicit resumable output path.

- [ ] **Step 5: Produce the final stratified report**

Report bridge-target, bridge-control, and common cohorts separately; include
retrieval misses, verifier routing, repair selection, held-out DINO,
preservation, confidence intervals, and per-case visual grids. State whether
the method improves rare generation and whether it preserves common concepts.
