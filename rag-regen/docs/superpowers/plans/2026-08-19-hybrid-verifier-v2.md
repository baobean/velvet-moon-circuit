# Hybrid Verifier V2 Development Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit and audit a conservative hybrid-verifier-v2 policy on exposed development artifacts, then produce an immutable readiness decision without GPU work or production integration.

**Architecture:** Add a pure manifest/provenance module and a pure v2 policy-search module. A CPU-only CLI joins existing semantic, reranker, DINO, queue, and label artifacts, performs leave-one-concept-out policy fitting, and emits a complete development report plus a frozen policy only when every readiness gate passes.

**Tech Stack:** Python 3.11, pytest, CSV/JSON, hashlib, existing `ragregen.c1`, `ragregen.metrics`, and cached v1 artifacts.

**Spec:** `docs/superpowers/specs/2026-08-19-hybrid-verifier-v2-design.md`

## Global Constraints

- Call all exposed 2026-08-18/19 artifacts `development`, never `validation` or `test`.
- Preserve `identity_truth`; mask/reference failure changes only `selector_eligible` and `eligibility_reason`.
- Routing family is semantic failure OR the conjunction of low text and low reference relevance.
- Selection requires inclusive text and reference margins; ties are text, reference, then lower attempt number.
- Use DINO only to fit/evaluate development policy, never as a live policy input.
- All fitting is leave-one-concept-out; held-out-fold predictions are aggregated without refitting.
- Phase A is CPU-only: no model inference, no pipeline run, no production adapter, no 92-case run.
- Writers refuse to overwrite immutable manifests and frozen policy files.
- Preserve every pre-existing dirty-worktree change. Use scoped diffs; do not commit overlapping dirty files.

---

### Task 1: Immutable identity and eligibility manifest

**Files:**

- Create: `ragregen/eval_manifest.py`
- Create: `scripts/freeze_eval_manifest.py`
- Create: `tests/test_eval_manifest.py`
- Modify: `scripts/run.sh`

**Interfaces:**

- Produces `sha256_file(path: Path) -> str`.
- Produces `build_manifest(dataset, identity_rows, screen_run, candidate_run, *, protocol_status, source_paths) -> dict`.
- Produces one row per dataset case with `identity_truth`, `selector_eligible`, `eligibility_reason`, cohort, and source hashes.

- [ ] **Step 1: Write failing tests for truth/eligibility separation and immutability**

```python
def test_mask_failure_preserves_identity_truth(tmp_path):
    manifest = build_manifest(dataset, {"rare": "FAIL"}, screen, candidates,
                              protocol_status="reconstructed_after_score",
                              source_paths={})
    row = manifest["cases"]["rare"]
    assert row["identity_truth"] == "FAIL"
    assert row["selector_eligible"] is False
    assert row["eligibility_reason"] == "mask_not_grounded"

def test_cli_refuses_to_overwrite_manifest(tmp_path):
    out = tmp_path / "manifest.json"
    out.write_text("sentinel")
    with pytest.raises(FileExistsError):
        freeze_main(valid_args + ["--out", str(out)])
    assert out.read_text() == "sentinel"
```

- [ ] **Step 2: Run tests and confirm the missing-module failure**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_eval_manifest.py -q`

Expected: FAIL because `ragregen.eval_manifest` does not exist.

- [ ] **Step 3: Implement deterministic manifest construction**

Use queue status plus files, in this order:

```python
if state["stages"].get("mask") == "failed":
    eligible, reason = False, "mask_not_grounded"
elif not (case_dir / "mask.png").is_file():
    eligible, reason = False, "corrupt_artifact"
elif not list(case_dir.glob("attempt_*.png")):
    eligible, reason = False, "no_prepared_reference"
else:
    eligible, reason = True, "eligible"
```

Validate dataset/identity IDs exactly, allow only `PASS`, `FAIL`, or `UNJUDGEABLE`, hash dataset, identity source, semantic JSON, reranker JSON, DINO JSON, queue, drafts, masks, and attempts, and sort case IDs and paths before JSON serialization.

- [ ] **Step 4: Implement the CLI and `run.sh` dispatch**

CLI arguments must be explicit: `--dataset`, `--identity-source`, `--identity-format {labels_csv,reranker_truth}`, `--screen-run`, `--candidate-run`, `--semantic`, `--reranker`, `--dino`, `--protocol-status {frozen_before_score,reconstructed_after_score}`, and `--out`. `reranker_truth` reads each case's top-level `truth` field; it never reads the current CSV.

Add `freeze-eval-manifest)` to `scripts/run.sh`; refuse overwrite before reading model artifacts.

- [ ] **Step 5: Run focused tests**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_eval_manifest.py tests/test_config.py tests/test_report.py -q`

Expected: PASS.

- [ ] **Step 6: Inspect the scoped diff**

Run: `git diff -- ragregen/eval_manifest.py scripts/freeze_eval_manifest.py tests/test_eval_manifest.py scripts/run.sh`

Expected: manifest/provenance logic and one command dispatch only.

---

### Task 2: Pure v2 policy and deterministic fitters

**Files:**

- Create: `ragregen/hybrid_verifier_v2.py`
- Create: `tests/test_hybrid_verifier_v2.py`

**Interfaces:**

- Produces frozen `Policy(text_threshold, reference_threshold, text_margin, reference_margin)`.
- Produces `route_draft(semantic_ok, text_relevance, reference_relevance, policy) -> bool`.
- Produces `select_repair(scores, policy) -> str`.
- Produces `fit_detector(rows) -> DetectorFit` and `fit_selector(rows) -> SelectorFit`.

- [ ] **Step 1: Write failing boundary and tie tests**

```python
def test_route_uses_guarded_conjunction_after_semantic_pass():
    p = Policy(0.50, 0.25, 0.02, 0.01)
    assert route_draft(False, 0.9, 0.9, p)
    assert route_draft(True, 0.49, 0.24, p)
    assert not route_draft(True, 0.49, 0.25, p)
    assert not route_draft(True, 0.50, 0.24, p)

def test_selector_uses_inclusive_margins_and_stable_ties():
    p = Policy(0.5, 0.25, 0.02, 0.01)
    scores = {"draft": score(.40, .30),
              "attempt_2": score(.42, .31),
              "attempt_1": score(.42, .31)}
    assert select_repair(scores, p) == "attempt_1"
```

- [ ] **Step 2: Run tests and confirm the missing-module failure**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_hybrid_verifier_v2.py -q`

Expected: FAIL because `ragregen.hybrid_verifier_v2` does not exist.

- [ ] **Step 3: Implement validation and pure decisions**

Reject missing/non-finite values. Use strict `<` routing thresholds, inclusive `>=` margins, and stable candidate ordering `(text, reference, -attempt_number)`.

- [ ] **Step 4: Write failing deterministic-fit tests**

Use synthetic rows where reference-only causes a control false positive but text-and-reference conjunction separates it. Assert shuffled input produces the same fit. Add a selector fixture where a no-op margin has high raw accuracy but zero DINO gain; assert the positive-gain feasible fit wins.

- [ ] **Step 5: Implement threshold grids and lexicographic objectives**

Detector candidates are value midpoints plus one sentinel below/above the observed range. Feasible fits satisfy recall `>=0.733` and false-positive rate `<=0.0834`; choose highest MCC, lower FP, higher recall, then smaller routed population.

Selector margin candidates are observed nonnegative text/reference deltas plus `0.0`. Feasible fits satisfy sign accuracy `>=0.750`, harmful `<=1`, and mean selected DINO delta `>0`; choose higher mean delta, exact-best rate, text margin, reference margin, then the smaller numeric tuple.

- [ ] **Step 6: Run policy tests and v1 regressions**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_hybrid_verifier_v2.py tests/test_hybrid_verifier.py tests/test_reranker_smoke.py -q`

Expected: PASS; v1 remains unchanged.

- [ ] **Step 7: Inspect the scoped diff**

Run: `git diff -- ragregen/hybrid_verifier_v2.py tests/test_hybrid_verifier_v2.py`

Expected: pure CPU policy/search code only.

---

### Task 3: Leave-one-concept-out development evaluator

**Files:**

- Create: `ragregen/hybrid_v2_develop.py`
- Create: `tests/test_hybrid_v2_develop.py`

**Interfaces:**

- Consumes immutable manifest, semantic decisions, reranker scores, and pairwise DINO.
- Produces `develop(manifest, semantic, reranker, dino) -> dict` with fold parameters, held-out decisions, detector/selector confusion, Wilson intervals, cohort deltas, and readiness gates.

- [ ] **Step 1: Write a failing fold-isolation test**

```python
def test_each_prediction_uses_fit_without_heldout_case():
    result = develop(manifest, semantic, reranker, dino)
    assert len(result["folds"]) == len(manifest["cases"])
    for fold in result["folds"]:
        assert fold["case_id"] not in fold["fit_case_ids"]
```

- [ ] **Step 2: Write failing denominator tests**

Assert mask-failed cases remain in detector confusion but do not appear in selector comparisons; assert the report records both detector and selector denominators and the mechanical reason.

- [ ] **Step 3: Run tests and confirm failure**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_hybrid_v2_develop.py -q`

Expected: FAIL because the module does not exist.

- [ ] **Step 4: Implement fold fitting and aggregation**

For each case ID, fit on every other case, apply to only the held-out case, and persist the fitted policy and training IDs. Never fit once on all rows and relabel that result cross-validation.

Use `c1.confusion`, `c1.wilson_ci`, and `metrics.paired_bootstrap_ci`. Add pairwise TP/FP/TN/FN where positive means “attempt improves held-out DINO.” Report semantic-only, reference-only, v1 union, and v2 guarded policies over identical detector rows.

- [ ] **Step 5: Implement readiness gates**

Detector uses all judgeable identities. Selector/end-to-end use only eligible cases. Gates are conjunctive and include exact reasons for failure or inconclusive strata.

- [ ] **Step 6: Run focused tests**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_hybrid_v2_develop.py tests/test_verifier_eval.py tests/test_hybrid_verifier_v2.py -q`

Expected: PASS.

---

### Task 4: CPU-only development CLI and immutable outputs

**Files:**

- Create: `scripts/hybrid_v2_develop.py`
- Create: `tests/test_hybrid_v2_develop_cli.py`
- Modify: `scripts/run.sh`

**Interfaces:**

- Produces `development.json`, `development.md`, `folds.csv`, and, only on readiness PASS, `frozen_policy.json` plus `frozen_policy.sha256`.

- [ ] **Step 1: Write failing CLI artifact tests**

Test PASS and FAIL fixtures. FAIL must not create `frozen_policy.json`. Both outcomes must create reports containing `development`, protocol status, hashes, denominators, Wilson intervals, baseline tables, and gate reasons.

- [ ] **Step 2: Run tests and confirm the missing-script failure**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_hybrid_v2_develop_cli.py -q`

Expected: FAIL because `scripts.hybrid_v2_develop` does not exist.

- [ ] **Step 3: Implement strict CLI joins and renderers**

Arguments: `--manifest`, `--semantic`, `--reranker`, `--dino`, and `--output-dir`. Refuse an existing nonempty output directory. Verify every input hash against the manifest before evaluation. Exit `0` for a completed PASS or FAIL study; artifact corruption raises and exits nonzero. On readiness PASS only, refit detector thresholds on all 24 identity rows and selector margins on all 22 eligible rows, serialize that full-development `Policy`, and hash the exact serialized bytes.

- [ ] **Step 4: Add `hybrid-v2-develop` dispatch to `run.sh`**

List it as CPU-only in help text.

- [ ] **Step 5: Run CLI and report regressions**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_hybrid_v2_develop_cli.py tests/test_report.py tests/test_hybrid_verifier_report.py -q`

Expected: PASS.

---

### Task 5: Reconstruct the exposed development manifest and run Phase A

**Files:**

- Runtime: `outputs/hybrid_v2_development/manifest.json`
- Runtime: `outputs/hybrid_v2_development/report/development.{json,md}`

**Interfaces:**

- Consumes `outputs/screen_20260818_172307`, `outputs/holdout_candidates_20260818_182938`, and `outputs/hybrid_validation_20260819_085952`.
- Produces the sole Phase A readiness decision.

- [ ] **Step 1: Reconstruct identity truth without rewriting the source CSV**

Use `qwen_reranker.json`'s frozen `truth` field for all 24 cases, including `azawakh` and `bergamasco_shepherd` as `FAIL`. Record `protocol_status=reconstructed_after_score`. Do not edit `labels.csv`.

- [ ] **Step 2: Freeze the manifest**

Run:

```bash
./scripts/run.sh freeze-eval-manifest \
  --dataset configs/dataset_verifier_holdout.yaml \
  --identity-source outputs/hybrid_validation_20260819_085952/qwen_reranker.json \
  --identity-format reranker_truth \
  --screen-run outputs/screen_20260818_172307 \
  --candidate-run outputs/holdout_candidates_20260818_182938 \
  --semantic outputs/screen_20260818_172307/stream_b.json \
  --reranker outputs/hybrid_validation_20260819_085952/qwen_reranker.json \
  --dino outputs/hybrid_validation_20260819_085952/pairwise_dino.json \
  --protocol-status reconstructed_after_score \
  --out outputs/hybrid_v2_development/manifest.json
```

Expected: 24 detector rows, 22 selector-eligible rows, two `mask_not_grounded` reasons, and exact input hashes.

- [ ] **Step 3: Run the CPU-only development study**

Run:

```bash
./scripts/run.sh hybrid-v2-develop \
  --manifest outputs/hybrid_v2_development/manifest.json \
  --semantic outputs/screen_20260818_172307/stream_b.json \
  --reranker outputs/hybrid_validation_20260819_085952/qwen_reranker.json \
  --dino outputs/hybrid_validation_20260819_085952/pairwise_dino.json \
  --output-dir outputs/hybrid_v2_development/report
```

Expected: no CUDA/model initialization; complete fold and baseline tables.

- [ ] **Step 4: Apply the stop condition**

If readiness is `FAIL` or `INCONCLUSIVE`, stop and write a short finding recommending a new-signal design. Do not create a holdout plan.

If readiness is `PASS`, fit one final policy on all 24 detector rows and all 22 selector-eligible rows, verify `frozen_policy.json` and its SHA-256, then write the separate Phase B fresh-holdout design and plan required by spec section 7. Do not generate the holdout or integrate production in this plan.

- [ ] **Step 5: Run the full non-GPU suite**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest -q -m 'not gpu'`

Expected: all tests pass with only established skips/deselections.

- [ ] **Step 6: Record repository state**

Run: `git status --short`

Expected: preserved pre-existing dirty files plus only the planned Phase A modules/tests/docs; outputs remain ignored.
