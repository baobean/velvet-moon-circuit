# Retrieved-Reference Detector — Phase B Holdout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Confirm the frozen detector-only repair policy on an independent, frozen-before-score 24+24 holdout, producing one immutable readiness decision without any threshold fitting.

**Architecture:** Pure modules hold every decision (policy freezing/hashing, denominator/backfill logic, frozen-policy application, three-arm assembly, metric aggregation and gates); thin CLIs orchestrate the GPU/model steps by reusing those pure modules plus already-built scripts (`retrieved_reference_score.py`, `score_b.py`, `run_pipeline.py` open-loop). Detector scoring runs and gates **before** any generation; generation and end-to-end evaluation run only on a detector PASS.

**Tech Stack:** Python 3.11, pytest, CSV/JSON, hashlib, existing `ragregen.{config,c1,metrics,retrieve,eval_manifest,detector_develop,retrieved_reference,hybrid_verifier_v2}`, cached Qwen2.5-VL / Qwen3-VL-Reranker-2B / SigLIP / DINO / CLIP / SigLIP-eval models, the disposable reranker venv (sentence-transformers 5.4.0).

**Spec:** `docs/superpowers/specs/2026-08-19-retrieved-detector-phase-b-design.md`

## Global Constraints

- Frozen rule, verbatim: `route = semantic_failure OR (draft_text_relevance < 0.5 AND draft_retrieved_reference_relevance < 0.25048828125)`; routed→`attempt_1` on success, else draft flagged `generation_failed=true`.
- Cohort denominators are fixed at **exactly 24 judgeable rare FAIL** and **exactly 24 judgeable control PASS**; never shrink a denominator.
- **No** threshold fitting, LOCO, exclusions, or replacements after any verifier score is visible. Backfill from the replacement reserve happens **only at freeze time**.
- Detector gates: complete uncontaminated coverage, failure recall **≥ 18/24**, false positives **≤ 2/24**. Stop before generation on failure/INCONCLUSIVE.
- End-to-end gates (arm 3 vs draft): rare mean cropped-DINO delta **> 0**; harmful (< −0.02) **≤ 1 across all 48**; control non-inferiority = paired bootstrap CI lower bound of arm-3 − draft over **all 24 controls ≥ −0.02**; outside-mask preservation **exactly 1.0**; visual PASS = no reference-photo paste-through, no material background/layout replacement.
- Wilson intervals are **descriptive only** — no confidence-bound gate. Thin-PASS trigger is the discrete boundary alone: recall exactly 18/24 **or** FP exactly 2/24.
- Any post-freeze preparation/generation failure makes Phase B **INCONCLUSIVE** before end-to-end evaluation.
- `protocol_status: frozen_before_score`. Every manifest/result hashes source artifacts, model **weight files**, prompts, full env manifest + venv wheel hashes, retrieval/index hashes, and a **clean code snapshot** (not `git HEAD` — the worktree is dirty). Writers refuse to overwrite.
- A PASS authorizes **integration testing only** — not an unqualified production claim, not the 92-case run.
- **Preserve the dirty worktree.** Do not commit over unrelated dirty files: each task ends with a **scoped `git diff` review** of only its own new files (mirroring the Phase A working style), not a commit. Outputs go under `outputs/` (git-ignored).
- CPU tests use `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest`; the reranker step uses the disposable venv at `…/scratchpad/rerank_venv/bin/python`.

---

### Task 1: Frozen policy manifest + hashing

**Files:**
- Create: `ragregen/detector_policy.py`
- Create: `scripts/freeze_detector_policy.py`
- Create: `tests/test_detector_policy.py`
- Modify: `scripts/run.sh` (add `freeze-detector-policy` dispatch)

**Interfaces:**
- Consumes: `ragregen.eval_manifest.sha256_file`.
- Produces:
  - `build_policy_manifest(*, thresholds, semantic, reranker, retrieval, generator, env_manifest, code_snapshot_sha256, degenerate_behaviour) -> dict` — assembles the complete frozen policy (rule text + both branches, model ids **and weight-file SHA-256s**, prompts, full env manifest + wheel hashes, retrieval/index hashes, clean-code-snapshot hash, degenerate-input rule), `protocol_status="frozen_before_score"`.
  - `policy_sha256(manifest: dict) -> str` — SHA-256 over `json.dumps(manifest, sort_keys=True)`.
  - `snapshot_code_sha256(paths: list[Path]) -> str` — deterministic tree hash over the exact executed source files.
  - `DEGENERATE_RULE: dict` — the frozen constant: `{"semantic_parse_failure": "fail_closed_route", "missing_or_nonfinite_score": "coverage_failure_inconclusive"}`.

- [ ] **Step 1: Write the failing test for rule + threshold capture**

```python
# tests/test_detector_policy.py
from ragregen.detector_policy import build_policy_manifest, policy_sha256, DEGENERATE_RULE

def _manifest(**over):
    base = dict(
        thresholds={"text": 0.5, "retrieved_reference": 0.25048828125},
        semantic={"model_id": "qwen2.5-vl", "weight_sha256": "s1", "template": "T", "decoding": {"max_new_tokens": 256}},
        reranker={"model_id": "qwen3-vl-reranker-2b", "weight_sha256": "r1", "prompt": "P", "max_image_side": 448},
        retrieval={"encoder": "siglip_so400m_384", "encoder_weight_sha256": "e1",
                   "index_sha256": "i1", "corpus": "laion100k",
                   "query_builder": "ragregen.retrieve.reference_query", "k": 5,
                   "ref_prep": "crop_to_mask+thumbnail448"},
        generator={"model_id": "flux-kontext", "weight_sha256": "g1"},
        env_manifest={"torch": "2.6.0", "sentence-transformers": "5.4.0"},
        code_snapshot_sha256="c1",
        degenerate_behaviour=DEGENERATE_RULE)
    base.update(over)
    return build_policy_manifest(**base)

def test_manifest_captures_full_rule_and_semantic_branch():
    m = _manifest()
    assert m["protocol_status"] == "frozen_before_score"
    assert "semantic_failure OR" in m["rule"]
    assert m["thresholds"] == {"text": 0.5, "retrieved_reference": 0.25048828125}
    assert m["degenerate_behaviour"]["semantic_parse_failure"] == "fail_closed_route"
    assert m["semantic"]["weight_sha256"] == "s1"
    assert m["generator"]["weight_sha256"] == "g1"

def test_policy_sha256_is_order_independent():
    a = _manifest(); b = _manifest()
    assert policy_sha256(a) == policy_sha256(b)
```

- [ ] **Step 2: Run to verify it fails**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_detector_policy.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragregen.detector_policy'`.

- [ ] **Step 3: Implement `ragregen/detector_policy.py`**

```python
"""Complete, hashable frozen policy manifest for the Phase B holdout."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from ragregen.eval_manifest import sha256_file

RULE = ("route = semantic_failure OR (draft_text_relevance < 0.5 AND "
        "draft_retrieved_reference_relevance < 0.25048828125); "
        "routed -> attempt_1 on success else draft(generation_failed=true); "
        "not routed -> draft")

DEGENERATE_RULE = {
    "semantic_parse_failure": "fail_closed_route",
    "missing_or_nonfinite_score": "coverage_failure_inconclusive",
}

def snapshot_code_sha256(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x)):
        h.update(str(p).encode()); h.update(b"\0"); h.update(Path(p).read_bytes()); h.update(b"\0")
    return h.hexdigest()

def build_policy_manifest(*, thresholds, semantic, reranker, retrieval, generator,
                          env_manifest, code_snapshot_sha256, degenerate_behaviour) -> dict:
    return {
        "schema": 1, "protocol_status": "frozen_before_score", "rule": RULE,
        "thresholds": thresholds, "semantic": semantic, "reranker": reranker,
        "retrieval": retrieval, "generator": generator, "env_manifest": env_manifest,
        "code_snapshot_sha256": code_snapshot_sha256,
        "degenerate_behaviour": degenerate_behaviour,
    }

def policy_sha256(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
```

- [ ] **Step 4: Run to verify pass**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_detector_policy.py -q`
Expected: PASS.

- [ ] **Step 5: Implement `scripts/freeze_detector_policy.py`**

CLI args: `--thresholds-json`, `--semantic-weights DIR`, `--reranker-weights DIR`, `--encoder-weights DIR`, `--generator-weights DIR`, `--index PATH`, `--env-freeze FILE` (a `pip freeze` capture), `--code-paths GLOB...`, `--out`. Refuse overwrite of `--out`. Hash each weight directory's `*.safetensors` via `sha256_file` (sorted, combined), call `build_policy_manifest`, write `{"manifest": m, "policy_sha256": policy_sha256(m)}` with `json.dumps(..., indent=2, sort_keys=True)`.

- [ ] **Step 6: Add `freeze-detector-policy` dispatch to `scripts/run.sh`**

Insert after the `retrieved-detector-develop` line:
```bash
  freeze-detector-policy) exec "$PY" scripts/freeze_detector_policy.py "$@" ;;
```
and a help line: `  freeze-detector-policy  hash the complete frozen policy (no GPU, seconds)`.

- [ ] **Step 7: Scoped diff review**

Run: `git diff -- ragregen/detector_policy.py scripts/freeze_detector_policy.py tests/test_detector_policy.py scripts/run.sh`
Expected: policy manifest/hashing logic and one command dispatch only.

---

### Task 2: Phase B holdout freeze (denominators, reserves, contamination backfill)

**Files:**
- Create: `ragregen/phase_b_manifest.py`
- Create: `scripts/freeze_phase_b_holdout.py`
- Create: `tests/test_phase_b_manifest.py`
- Modify: `scripts/run.sh`

**Interfaces:**
- Consumes: `ragregen.eval_manifest.sha256_file`, `ragregen.retrieved_reference.contamination`.
- Produces:
  - `select_cohort(candidates, *, cohort, needed=24) -> dict` where `candidates` is an ordered list of `{case_id, identity_truth, eligible, contaminated}`; returns `{"cases": [24 judgeable ids], "backfilled": [...], "status": "complete"|"inconclusive"}`. Picks the first `needed` judgeable+eligible+uncontaminated cases in declared order; records any skipped case as a backfill event; `inconclusive` if fewer than `needed` remain.
  - `build_holdout_manifest(rare_selection, control_selection, *, dino_reserves, references, sources) -> dict` — immutable manifest with `protocol_status="frozen_before_score"`, exactly-24/24 cohorts, separated `dino_reserves`, `references`, source hashes.

- [ ] **Step 1: Write the failing test for fixed denominators + backfill**

```python
# tests/test_phase_b_manifest.py
import pytest
from ragregen.phase_b_manifest import select_cohort

def _cand(cid, truth, eligible=True, contaminated=False):
    return {"case_id": cid, "identity_truth": truth, "eligible": eligible, "contaminated": contaminated}

def test_backfill_preserves_24_and_records_substitution():
    cands = [_cand(f"r{i}", "FAIL") for i in range(24)]
    cands[3]["contaminated"] = True           # must be skipped
    cands.append(_cand("spare", "FAIL"))       # replacement reserve
    sel = select_cohort(cands, cohort="rare", needed=24)
    assert len(sel["cases"]) == 24
    assert "r3" not in sel["cases"] and "spare" in sel["cases"]
    assert sel["backfilled"] == ["r3"] and sel["status"] == "complete"

def test_exhausted_reserve_is_inconclusive_not_shrunk():
    cands = [_cand(f"r{i}", "FAIL") for i in range(23)]  # only 23 judgeable
    sel = select_cohort(cands, cohort="rare", needed=24)
    assert sel["status"] == "inconclusive"
    assert len(sel["cases"]) < 24

def test_non_judgeable_and_ineligible_are_skipped():
    cands = [_cand(f"r{i}", "FAIL") for i in range(24)]
    cands[5]["identity_truth"] = "UNJUDGEABLE"
    cands[6]["eligible"] = False
    cands += [_cand("s1", "FAIL"), _cand("s2", "FAIL")]
    sel = select_cohort(cands, cohort="rare", needed=24)
    assert len(sel["cases"]) == 24
    assert {"r5", "r6"}.isdisjoint(sel["cases"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_phase_b_manifest.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `select_cohort` and `build_holdout_manifest`**

```python
# ragregen/phase_b_manifest.py (select_cohort shown; build_holdout_manifest mirrors eval_manifest)
def select_cohort(candidates, *, cohort, needed=24):
    want = {"rare": "FAIL", "control": "PASS"}[cohort]
    chosen, backfilled = [], []
    for c in candidates:
        if len(chosen) == needed:
            break
        judgeable = c["identity_truth"] == want
        ok = judgeable and c["eligible"] and not c["contaminated"]
        if ok:
            chosen.append(c["case_id"])
        elif judgeable:            # a would-be case skipped -> backfill event
            backfilled.append(c["case_id"])
    status = "complete" if len(chosen) == needed else "inconclusive"
    return {"cases": chosen, "backfilled": backfilled, "status": status}
```

`build_holdout_manifest` writes exactly-24/24 cohorts, the separate `dino_reserves` map, the `references` map (retrieved top-1 per case), source hashes via `sha256_file`, and `protocol_status="frozen_before_score"`; it raises if either cohort is not exactly 24.

- [ ] **Step 4: Run to verify pass**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_phase_b_manifest.py -q`
Expected: PASS.

- [ ] **Step 5: Implement `scripts/freeze_phase_b_holdout.py` + run.sh dispatch**

CLI joins hand-labels, eligibility (queue + files), retrieved references (from a `retrieval.json` produced by `retrieved-ref-score --retrieval-only` on the holdout candidates), and gt-ref contamination via `retrieved_reference.contamination`; calls `select_cohort` per cohort; refuses to proceed unless both are `complete`; writes the immutable holdout manifest. Add `freeze-phase-b-holdout)` dispatch + help line to `scripts/run.sh`.

- [ ] **Step 6: Scoped diff review**

Run: `git diff -- ragregen/phase_b_manifest.py scripts/freeze_phase_b_holdout.py tests/test_phase_b_manifest.py scripts/run.sh`
Expected: cohort/backfill/manifest logic and one dispatch only.

---

### Task 3: Frozen-policy application + detector gate (pure)

**Files:**
- Create: `ragregen/detector_apply.py`
- Create: `tests/test_detector_apply.py`

**Interfaces:**
- Consumes: `ragregen.hybrid_verifier_v2.{Policy, route_draft}`.
- Produces:
  - `apply_frozen(rows, thresholds) -> dict` — applies the frozen rule (NO fitting) to `rows` (`{case_id, truth_fail, semantic_ok, text_relevance, retrieved_reference_relevance, cohort}`); returns per-case predictions + confusion split by cohort.
  - `detector_gate(applied, *, coverage_ok) -> dict` — gates: coverage_ok AND rare recall ≥ 18/24 AND control FP ≤ 2/24; returns `{"readiness": "PASS"|"FAIL"|"INCONCLUSIVE", "gates": [...], "recall_k": (k,24), "fp_k": (k,24)}`. `readiness="INCONCLUSIVE"` iff not `coverage_ok`.
  - `degenerate_coverage(rows) -> bool` — False if any row has a missing/non-finite `text_relevance`/`retrieved_reference_relevance` (per `DEGENERATE_RULE`).

- [ ] **Step 1: Write failing tests for no-fit application + discrete gates**

```python
# tests/test_detector_apply.py
from ragregen.detector_apply import apply_frozen, detector_gate, degenerate_coverage
T = {"text": 0.5, "retrieved_reference": 0.25048828125}

def _row(cid, tf, sem_ok, text, ref, cohort):
    return {"case_id": cid, "truth_fail": tf, "semantic_ok": sem_ok,
            "text_relevance": text, "retrieved_reference_relevance": ref, "cohort": cohort}

def test_applies_frozen_rule_without_fitting():
    rows = [_row("f", True, True, 0.10, 0.10, "rare"),   # conjunction routes
            _row("p", False, True, 0.90, 0.90, "control")]  # clean control
    a = apply_frozen(rows, T)
    assert a["cases"]["f"]["routed"] is True
    assert a["cases"]["p"]["routed"] is False

def test_gate_passes_at_recall_18_and_fp_2():
    rows = [_row(f"f{i}", True, i < 18, 0.9, 0.9, "rare") for i in range(24)]  # 18 semantic-caught
    rows += [_row(f"p{i}", False, True, 0.10 if i < 2 else 0.9, 0.10 if i < 2 else 0.9, "control")
             for i in range(24)]  # 2 controls routed -> 2 FP
    g = detector_gate(apply_frozen(rows, T), coverage_ok=True)
    assert g["recall_k"] == (18, 24) and g["fp_k"] == (2, 24)
    assert g["readiness"] == "PASS"

def test_missing_score_is_inconclusive():
    rows = [_row("f", True, True, float("nan"), 0.1, "rare")]
    assert degenerate_coverage(rows) is False
    g = detector_gate(apply_frozen([_row("f", True, True, 0.1, 0.1, "rare")], T), coverage_ok=False)
    assert g["readiness"] == "INCONCLUSIVE"
```

- [ ] **Step 2: Run to verify it fails**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_detector_apply.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `apply_frozen`, `detector_gate`, `degenerate_coverage`**

```python
# ragregen/detector_apply.py
import math
from ragregen import c1
from ragregen.hybrid_verifier_v2 import Policy, route_draft

def degenerate_coverage(rows):
    for r in rows:
        for k in ("text_relevance", "retrieved_reference_relevance"):
            v = r.get(k)
            if v is None or not math.isfinite(float(v)):
                return False
    return True

def apply_frozen(rows, thresholds):
    p = Policy(thresholds["text"], thresholds["retrieved_reference"], 0.0, 0.0)
    cases, by = {}, {"rare": ([], []), "control": ([], [])}
    for r in rows:
        routed = route_draft(r["semantic_ok"], r["text_relevance"],
                             r["retrieved_reference_relevance"], p)
        cases[r["case_id"]] = {"routed": bool(routed), "truth_fail": r["truth_fail"], "cohort": r["cohort"]}
        yt, yp = by[r["cohort"]]; yt.append(bool(r["truth_fail"])); yp.append(bool(routed))
    conf = {c: c1.confusion(*by[c]) for c in by}
    return {"cases": cases, "confusion": conf}

def detector_gate(applied, *, coverage_ok):
    rare, control = applied["confusion"]["rare"], applied["confusion"]["control"]
    recall_k = (rare.tp, rare.tp + rare.fn); fp_k = (control.fp, control.tp + control.fp + control.tn + control.fn)
    recall_ok = rare.tp >= 18
    fp_ok = control.fp <= 2
    gates = [("uncontaminated_coverage", coverage_ok), ("detector_recall_ge_18", recall_ok),
             ("detector_fp_le_2", fp_ok)]
    if not coverage_ok:
        readiness = "INCONCLUSIVE"
    elif recall_ok and fp_ok:
        readiness = "PASS"
    else:
        readiness = "FAIL"
    return {"readiness": readiness, "gates": [{"name": n, "passed": p} for n, p in gates],
            "recall_k": recall_k, "fp_k": fp_k}
```

- [ ] **Step 4: Run to verify pass**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_detector_apply.py -q`
Expected: PASS.

- [ ] **Step 5: Scoped diff review**

Run: `git diff -- ragregen/detector_apply.py tests/test_detector_apply.py`
Expected: pure application/gate logic only.

---

### Task 4: Detector-first scoring CLI (orchestration, stops before generation)

**Files:**
- Create: `scripts/phase_b_detector.py`
- Create: `tests/test_phase_b_detector_cli.py`
- Modify: `scripts/run.sh`

**Interfaces:**
- Consumes: `ragregen.detector_apply.{apply_frozen, detector_gate, degenerate_coverage}`, the holdout manifest (Task 2), the frozen policy manifest (Task 1), a holdout `retrieved_reranker.json` (from `retrieved-ref-score` on holdout drafts), and a holdout `stream_b.json` (from `score-b`).
- Produces: `outputs/phase_b/detector/{detector.json,detector.md}`; on detector PASS writes `detector_gate: PASS` and a `proceed_to_generation: true` flag; on FAIL/INCONCLUSIVE writes the report and **no** proceed flag. Exit 0 for a completed study; corruption raises nonzero. `main(argv=None, *, apply_fn=..., gate_fn=...)` for injectable tests.

- [ ] **Step 1: Write failing CLI tests (inject fakes; no GPU)**

```python
# tests/test_phase_b_detector_cli.py
import json
from scripts.phase_b_detector import main

def _inputs(tmp_path, coverage_ok=True):
    # write a minimal frozen policy, holdout manifest, retrieved reranker, semantic json with matching hashes
    ...  # build tmp files; return argv list, out dir
    return argv, out

def test_pass_writes_proceed_flag(tmp_path):
    argv, out = _inputs(tmp_path)
    assert main(argv, apply_fn=lambda rows, t: {"confusion": {"rare": _c(20,0,0,4), "control": _c(0,1,23,0)}, "cases": {}},
                gate_fn=None) == 0
    rep = json.loads((out / "detector.json").read_text())
    assert rep["readiness"] == "PASS" and rep["proceed_to_generation"] is True

def test_fail_writes_no_proceed_flag(tmp_path):
    argv, out = _inputs(tmp_path)
    main(argv, apply_fn=lambda rows, t: {"confusion": {"rare": _c(10,0,0,14), "control": _c(0,5,19,0)}, "cases": {}}, gate_fn=None)
    rep = json.loads((out / "detector.json").read_text())
    assert rep["readiness"] == "FAIL" and rep.get("proceed_to_generation") is not True
```

(`_c(tp,fp,tn,fn)` returns a `c1.Confusion`.)

- [ ] **Step 2: Run to verify it fails**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_phase_b_detector_cli.py -q`
Expected: FAIL — script missing.

- [ ] **Step 3: Implement `scripts/phase_b_detector.py`**

Verify every input hash against the holdout + policy manifests; build detector rows (truth/cohort from holdout manifest, `semantic_ok` from stream_b, `text_relevance`/`retrieved_reference_relevance` from the holdout retrieved reranker draft scores); `coverage_ok = degenerate_coverage(rows) and (no contamination) and (48 judgeable rows present)`; call `apply_frozen` then `detector_gate`; render `detector.md` (per-cohort confusion, recall_k, fp_k, gate table); set `proceed_to_generation` only when `readiness=="PASS"`. Refuse non-empty output dir.

- [ ] **Step 4: Run to verify pass; add `phase-b-detector` dispatch to run.sh**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_phase_b_detector_cli.py -q`
Expected: PASS. Then add `phase-b-detector)` dispatch + help line to `scripts/run.sh`.

- [ ] **Step 5: Scoped diff review**

Run: `git diff -- scripts/phase_b_detector.py tests/test_phase_b_detector_cli.py scripts/run.sh`
Expected: strict-join orchestration + gate reporting only.

---

### Task 5: Three-arm assembly with flagged fallback (pure)

**Files:**
- Create: `ragregen/phase_b_arms.py`
- Create: `tests/test_phase_b_arms.py`

**Interfaces:**
- Produces:
  - `assemble_arms(cases) -> dict` where each `case` is `{case_id, cohort, routed, attempt_ok, draft_path, attempt_path}`. Returns per-case outputs for the three arms with an explicit `generation_failed` flag; **raises `PhaseBInconclusive`** if any routed case has `attempt_ok is False` (post-freeze failure → INCONCLUSIVE).
  - Arm rule: arm1=draft always; arm2 (route-all)=attempt if `attempt_ok` else draft(`generation_failed=true`); arm3 (frozen-detector)=attempt if `routed and attempt_ok`, draft otherwise, with `generation_failed=true` when `routed and not attempt_ok`.
  - `class PhaseBInconclusive(Exception)`.

- [ ] **Step 1: Write failing tests for the single frozen fallback rule**

```python
# tests/test_phase_b_arms.py
import pytest
from ragregen.phase_b_arms import assemble_arms, PhaseBInconclusive

def _case(cid, routed, attempt_ok=True):
    return {"case_id": cid, "cohort": "rare", "routed": routed, "attempt_ok": attempt_ok,
            "draft_path": f"/d/{cid}.png", "attempt_path": f"/a/{cid}.png"}

def test_every_arm_has_one_output_and_flags_fallback():
    out = assemble_arms([_case("a", routed=True), _case("b", routed=False)])
    assert out["a"]["arm3"]["path"] == "/a/a.png" and out["a"]["arm3"]["generation_failed"] is False
    assert out["b"]["arm3"]["path"] == "/d/b.png"        # not routed -> draft, ordinary
    assert out["b"]["arm3"]["generation_failed"] is False

def test_routed_failure_makes_study_inconclusive():
    with pytest.raises(PhaseBInconclusive):
        assemble_arms([_case("a", routed=True, attempt_ok=False)])

def test_no_denominator_shrink_all_48_present():
    cases = [_case(f"c{i}", routed=(i % 2 == 0)) for i in range(48)]
    out = assemble_arms(cases)
    assert len(out) == 48 and all("arm1" in v and "arm2" in v and "arm3" in v for v in out.values())
```

- [ ] **Step 2: Run to verify it fails**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_phase_b_arms.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `assemble_arms`**

```python
# ragregen/phase_b_arms.py
class PhaseBInconclusive(Exception): ...

def _out(path, failed): return {"path": path, "generation_failed": failed}

def assemble_arms(cases):
    result = {}
    for c in cases:
        if c["routed"] and not c["attempt_ok"]:
            raise PhaseBInconclusive(f"{c['case_id']}: routed attempt failed post-freeze")
        d, a, ok = c["draft_path"], c["attempt_path"], c["attempt_ok"]
        arm2 = _out(a, False) if ok else _out(d, True)
        arm3 = _out(a, False) if (c["routed"] and ok) else _out(d, False)
        result[c["case_id"]] = {"cohort": c["cohort"], "arm1": _out(d, False), "arm2": arm2, "arm3": arm3}
    return result
```

- [ ] **Step 4: Run to verify pass**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_phase_b_arms.py -q`
Expected: PASS.

- [ ] **Step 5: Scoped diff review**

Run: `git diff -- ragregen/phase_b_arms.py tests/test_phase_b_arms.py`
Expected: pure arm-assembly logic only.

---

### Task 6: Conditional generation CLI (frozen pipeline, detector-PASS only)

**Files:**
- Create: `scripts/phase_b_generate.py`
- Create: `tests/test_phase_b_generate_cli.py`
- Modify: `scripts/run.sh`

**Interfaces:**
- Consumes: `ragregen.phase_b_arms.{assemble_arms, PhaseBInconclusive}`, the Task-4 `detector.json` (must have `proceed_to_generation: true`), the holdout manifest, and the frozen generation stack (via `run_pipeline.py` open-loop: `--verifier none --open-loop-attempts 1`).
- Produces: `outputs/phase_b/generation/`: one deterministic `attempt_1` per case (48), an `arms.json` mapping each case to its arm1/arm2/arm3 output path + `generation_failed`, and a `generation.md` enumerating any prep/gen failures. On any routed-case failure, writes an INCONCLUSIVE marker and exits 0 without arm metrics. `main(argv=None, *, generate_fn=...)` injectable.

- [ ] **Step 1: Write failing CLI tests (inject fake generator; no GPU)**

```python
# tests/test_phase_b_generate_cli.py
import json
from scripts.phase_b_generate import main

def test_refuses_without_detector_pass(tmp_path):
    # detector.json with proceed_to_generation false -> raises/refuses
    ...

def test_inconclusive_on_routed_failure(tmp_path):
    argv, out = _setup(tmp_path, routed_fail="c0")
    main(argv, generate_fn=_fake_gen_with_failure)
    assert json.loads((out / "arms.json").read_text())["status"] == "inconclusive"

def test_success_writes_48_arm_rows(tmp_path):
    argv, out = _setup(tmp_path)
    main(argv, generate_fn=_fake_gen_ok)
    arms = json.loads((out / "arms.json").read_text())
    assert len(arms["cases"]) == 48
```

- [ ] **Step 2: Run to verify it fails**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_phase_b_generate_cli.py -q`
Expected: FAIL — script missing.

- [ ] **Step 3: Implement `scripts/phase_b_generate.py`**

Refuse unless `detector.json["proceed_to_generation"] is True`. Verify the generation stack hashes against the frozen policy manifest (Task 1). `generate_fn` runs the frozen open-loop pipeline over the 48 holdout drafts, returning `{case_id: {attempt_ok, attempt_path}}`; build `assemble_arms` inputs from holdout `routed` (Task 4) + generation results; on `PhaseBInconclusive`, write `arms.json` with `status="inconclusive"` and enumerate failures; else write `status="complete"` with 48 arm rows. Refuse non-empty output dir. Add `phase-b-generate)` dispatch (GPU) + help line to `scripts/run.sh`.

- [ ] **Step 4: Run to verify pass**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_phase_b_generate_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Scoped diff review**

Run: `git diff -- scripts/phase_b_generate.py tests/test_phase_b_generate_cli.py scripts/run.sh`
Expected: generation orchestration + arm assembly wiring only.

---

### Task 7: End-to-end evaluation, gates, thin-guard, readiness (pure)

**Files:**
- Create: `ragregen/phase_b_eval.py`
- Create: `tests/test_phase_b_eval.py`

**Interfaces:**
- Consumes: `ragregen.metrics.{dino_identity, prompt_alignment, preservation, paired_bootstrap_ci}`, `ragregen.c1.wilson_ci`.
- Produces:
  - `aggregate(metric_rows) -> dict` — per-arm × per-cohort tables: cropped DINO, whole CLIP, whole SigLIP, preservation, paired bootstrap CI vs draft, improved/unchanged/worsened counts, exact denominators, enumerated `generation_failed` cases.
  - `end_to_end_gates(agg, visual_review) -> dict` — rare arm3 mean cropped-DINO > 0; harmful (< −0.02) ≤ 1 across all 48; control non-inferiority = bootstrap CI lower bound (arm3 − draft over all 24 controls) ≥ −0.02; preservation exactly 1.0 for every generated output; visual PASS (no paste-through / no material bg-layout replacement).
  - `readiness(detector_gate, e2e_gates) -> dict` — conjunctive; `thin` flag iff recall exactly 18/24 or FP exactly 2/24.

- [ ] **Step 1: Write failing tests for the exact gate contract**

```python
# tests/test_phase_b_eval.py
from ragregen.phase_b_eval import end_to_end_gates, readiness

def _agg(rare_mean, harmful_48, ctrl_ci_lo, preservation_min):
    return {"rare": {"arm3": {"mean_cropped_dino_delta": rare_mean}},
            "harmful_count_48": harmful_48,
            "control": {"arm3_minus_draft_ci": [ctrl_ci_lo, 0.05]},
            "preservation_min": preservation_min}

def test_gates_pass_on_clean_agg():
    g = end_to_end_gates(_agg(0.15, 1, -0.01, 1.0), visual_review={"pass": True})
    assert all(x["passed"] for x in g["gates"])

def test_preservation_below_one_fails():
    g = end_to_end_gates(_agg(0.15, 0, -0.0, 0.999), visual_review={"pass": True})
    assert any(x["name"] == "preservation_exact_1" and x["passed"] is False for x in g["gates"])

def test_control_ci_lower_below_margin_fails():
    g = end_to_end_gates(_agg(0.15, 0, -0.05, 1.0), visual_review={"pass": True})
    assert any(x["name"] == "control_non_inferiority" and x["passed"] is False for x in g["gates"])

def test_thin_flag_on_discrete_boundary():
    r = readiness({"readiness": "PASS", "recall_k": (18, 24), "fp_k": (1, 24)},
                  {"readiness": "PASS", "gates": []})
    assert r["readiness"] == "PASS" and r["thin"] is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_phase_b_eval.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `aggregate`, `end_to_end_gates`, `readiness`**

`end_to_end_gates` builds gates: `rare_mean_dino_positive` (`rare.arm3.mean_cropped_dino_delta > 0`), `harmful_le_1` (`harmful_count_48 <= 1`), `control_non_inferiority` (`control.arm3_minus_draft_ci[0] >= -0.02`), `preservation_exact_1` (`preservation_min == 1.0`), `visual_review` (`visual_review["pass"]`). `readiness` is conjunctive over detector + e2e; sets `thin=True` iff `recall_k == (18,24) or fp_k[0] == 2`.

- [ ] **Step 4: Run to verify pass**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_phase_b_eval.py -q`
Expected: PASS.

- [ ] **Step 5: Scoped diff review**

Run: `git diff -- ragregen/phase_b_eval.py tests/test_phase_b_eval.py`
Expected: pure aggregation/gate/readiness logic only.

---

### Task 8: Evaluation CLI + full-suite verification

**Files:**
- Create: `scripts/phase_b_evaluate.py`
- Create: `tests/test_phase_b_evaluate_cli.py`
- Modify: `scripts/run.sh`

**Interfaces:**
- Consumes: `ragregen.phase_b_eval.*`, the Task-6 `arms.json`, the holdout manifest (cohorts, dino_reserves), and the eval encoders (`ragregen.metrics.EvalEncoders`).
- Produces: `outputs/phase_b/report/{development.json,development.md,per_arm.csv}` and, only on a non-thin readiness PASS, `phase_b_result.json` + `.sha256`. Renders every arm×cohort table, gate reasons, thin flag, and the "integration-testing-only" scope statement. `main(argv=None, *, score_fn=...)` injectable (fake per-image metrics in tests).

- [ ] **Step 1: Write failing CLI tests (inject fake scorer; no GPU)**

```python
# tests/test_phase_b_evaluate_cli.py
import json
from scripts.phase_b_evaluate import main

def test_pass_writes_result_and_scope_statement(tmp_path):
    argv, out = _setup(tmp_path, clean=True)
    main(argv, score_fn=_fake_clean_scores)
    rep = json.loads((out / "development.json").read_text())
    assert rep["readiness"] == "PASS"
    assert "integration testing only" in (out / "development.md").read_text().lower()

def test_inconclusive_arms_short_circuits(tmp_path):
    argv, out = _setup(tmp_path, arms_status="inconclusive")
    main(argv, score_fn=_fake_clean_scores)
    assert json.loads((out / "development.json").read_text())["readiness"] == "INCONCLUSIVE"
    assert not (out / "phase_b_result.json").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_phase_b_evaluate_cli.py -q`
Expected: FAIL — script missing.

- [ ] **Step 3: Implement `scripts/phase_b_evaluate.py` + run.sh dispatch**

If `arms.json["status"] != "complete"`, write `readiness="INCONCLUSIVE"` and stop (no result freeze). Else `score_fn` computes per-image cropped DINO / whole CLIP / whole SigLIP / preservation for every arm output (using `EvalEncoders` + the held-out DINO reserve from the manifest); `aggregate` → `end_to_end_gates` (with a required `--visual-review` JSON) → `readiness` (folding in the Task-4 detector gate). On non-thin PASS only, serialize `phase_b_result.json` and hash it. Add `phase-b-evaluate)` dispatch (GPU) + help line to `scripts/run.sh`.

- [ ] **Step 4: Run to verify pass**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest tests/test_phase_b_evaluate_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Full non-GPU suite**

Run: `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python -m pytest -q -m 'not gpu'`
Expected: all pass with only the established skips/deselections.

- [ ] **Step 6: Scoped diff review + repo state**

Run: `git diff -- scripts/phase_b_evaluate.py tests/test_phase_b_evaluate_cli.py scripts/run.sh` then `git status --short`
Expected: evaluation orchestration only; preserved pre-existing dirty files plus only the planned Phase B modules/tests; outputs ignored.

---

## Runtime sequence (operator, after all tasks reviewed — not part of coding)

1. `retrieved-ref-score --retrieval-only` over holdout candidates → holdout `retrieval.json`.
2. `freeze-phase-b-holdout` → immutable 24+24 manifest (`frozen_before_score`); **stop if INCONCLUSIVE**.
3. `freeze-detector-policy` → complete frozen policy + `policy_sha256`.
4. `score-b` (semantic) + `retrieved-ref-score` (reranker, disposable venv, parity-gated) over the 48 holdout drafts.
5. `phase-b-detector` → detector gate. **Stop before generation unless PASS.**
6. `phase-b-generate` → 48 deterministic `attempt_1` + three arms; **INCONCLUSIVE on any routed-case failure.**
7. `phase-b-evaluate` (+ exhaustive visual review) → readiness. Non-thin PASS authorizes **integration testing only**; a thin PASS requires another untouched confirmation before the 92-case run.
