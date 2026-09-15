# Fast Eval Driver — Design

**Status:** Design (approved 2026-09-03; user waived the written-spec review gate — proceed straight to plan + implementation)
**Workspace:** `ndbao_hbngoc/kg_test`
**Motivates:** the Phase 2 full-corpus benchmark (`docs/superpowers/specs/2026-08-30-graft-phase2-design.md` §6). The validation/de-bias sprint (`reports/sprint-report.md`) reversed the Phase 1 result on n=8, but n=8 is not statistically conclusive and the run took 27h — ~98% of it cold model-loading, not compute. This driver removes the load tax so a statistically real n (30–66 species) is feasible.

---

## 1. Problem

`graft/run_eval.py` runs one subprocess per (species × method × name_mode × ip_scale) cell for generation, and a **separate** subprocess per cell for metrics. Measured on the 176-cell sprint matrix:

- A 50-step SDXL generation is **~8 s** of compute; loading SDXL from the NAS is **~2.5 min**.
- Per cell: b0 ~6.5 min, `ours` ~10 min — **~98% cold model-loading, ~2% work.**
- Total model loads across the matrix ≈ 176 × (1 SDXL + 4 metric models) ≈ **~880 loads.**

The load tax, not the science, is the cost. Subprocess-per-cell isolation exists because sustained load/unload churn of **different** models in one long-lived process segfaults intermittently (see `graft/env.py`, `graft/refine.py` docstrings; `test_verify_gpu::test_verify_returns_report` still reproduces it). The fix must cut loads **without** reintroducing that churn.

## 2. Goals / non-goals

**Goals**
1. Cut model loads from ~880 to ~14 for the sprint matrix (KG ~8 + SDXL 1 + embedders ~5), turning a 27h run into ~1–2h.
2. Stay strictly inside the "one model family resident per process, no cross-model churn" safety envelope that the passing gpu tests validate.
3. Produce `results.json` / `analysis.json` / `results.md` **identical in shape** to the slow path, reusing `graft.analysis.build_analysis` and `graft.run_eval.summarize`.
4. Fold the sprint's findings into the defaults so the full run isn't over-specified (fixed ip=0.6, drop `ours_notree`, neutral default, k=2 seeds).
5. Be cross-checkable against the slow path (same seeds → same images → same metrics) so it can be trusted for the full corpus.

**Non-goals**
- No change to `run_eval.py` / `worker_baseline.py` / `worker_metrics.py` — the slow path stays as the trusted reference.
- No persistent cross-model resident server (the segfault trap).
- No multi-GPU / distributed execution.
- No `part_sim_threshold` recalibration (a Phase 2 experiment; noted in `reports/sprint-report.md` §7).
- No new metrics; the row schema is exactly today's `{dino, siglip2, clip_i, clip_t, attribute_accuracy}` plus the existing tags.

## 3. Architecture — four phases

Each GPU phase is a subprocess holding **one** model family for its whole life. Phases run sequentially; images and score tables land on disk between them, so nothing but PNGs is held across a phase boundary.

```
Phase 1  build KGs          worker_build_kg (EXISTING, reused)   ~8 loads
Phase 2  generate all       worker_generate_many.py  (SDXL)      1 load
Phase 3  score all          worker_score_many.py     (1/embedder) ~4-5 loads
Phase 4  aggregate + write  run_eval_fast.py (pure Python)       0 GPU
```

### 3.1 Manifest (pure, unit-tested)

`run_eval_fast` expands the CLI selection into a flat **cell manifest**: a list of
`{species, method, name_mode, ip_scale, seed}` plus, per species, its `build_refs` / `heldout_refs`
(from the existing adaptive `split_refs`) and `kg_path`. Rules (mirroring the slow path):
- swept methods `{ours, ours_notree, b1}` get every `ip_scale`; `b0`/`b2` get one cell per name_mode (ip-invariant, tagged `ip_scale=None`).
- each cell expands to `seeds` seed-rows (default 2: seed 0, seed 1).
- a species that cannot form a build+heldout split is skipped (existing guard).
- a stale/missing `ref_embeddings` KG triggers a rebuild in Phase 1 (existing `_kg_is_usable` logic, lifted into a shared helper).

The manifest is the single source of truth every phase consumes. Manifest expansion, seed fan-out, and the b0/b2 dedup are pure and unit-tested with no GPU.

### 3.2 Phase 2 — `worker_generate_many.py` (SDXL resident)

One subprocess. Loads `Models(cfg)` and touches **only** the SDXL/IP-Adapter generator for its whole life. Reads the manifest; for each seed-row not already on disk:
- build the per-cell `cfg` via `dataclasses.replace(cfg, neutralize_name=(name_mode=="neutral"), ip_scale=ip_scale)` — but note `ip_scale` is applied by calling `models.generator.set_scale(ip_scale)` between generations (cheap; no reload), NOT by rebuilding the pipe.
- compose the prompt (`compose_prompt` for ours/ours_notree, `b0`/`b1`/`b2` heads via the baseline prompt logic — factored into a shared pure `prompt_for(method, kg, concept, cfg)` so generation and the slow path agree).
- pick the exemplar (medoid for ours/ours_notree/b1; none for b0/b2).
- generate at `seed`, save to `outputs/<species>/gen_fast/<method>_<name_mode>_ip<ip>_seed<seed>.png`.

Requires a small `Models.generator.set_scale(x)` accessor (or reuse `pipe.set_ip_adapter_scale`) so ip can change without reconstructing the pipe. Resumable: existing PNG ⇒ skip. OOM on a cell ⇒ log + skip that seed-row (the row is later reported missing), never abort the process.

### 3.3 Phase 3 — `worker_score_many.py` (one embedder per invocation)

Invoked once per embedder: `worker_score_many.py <embedder> <manifest> <images_dir> <out_table>`. Each invocation loads exactly one model (`dino` | `siglip` | `clip` | `vlm` | `detector`), forward-passes every generated image the manifest references, and writes a JSON table keyed by image id. Per embedder it computes only what that model can:
- `dino`, `siglip`: held-out fidelity vs `heldout_refs`, and (if `use_part_tree`) part-crop sims for verify.
- `clip`: `clip_i` (held-out) + `clip_t` (vs the real species name — diagnostic, name kept deliberately).
- `vlm`: attribute checklist over `kg.attribute_texts()` ⇒ both verify `attr_pass` and metric `attribute_accuracy` (same computation).
- `detector`: part boxes on generated images, only when `use_part_tree` (default off per the sprint) — otherwise this embedder is skipped entirely.

Five one-model processes, ~5 loads total, zero cross-model churn. Resumable: an out_table that already covers every manifest image ⇒ skip that embedder.

### 3.4 Phase 4 — aggregate (pure Python, no GPU)

`run_eval_fast` reads the per-embedder tables and, for each cell, **picks the best seed** by the verify score (`0.5·mean_part_sim + 0.5·attr_pass`, or `attr_pass` alone when `use_part_tree=False`). The winning seed's `{dino, siglip2, clip_i, clip_t, attribute_accuracy}` becomes the cell's row, tagged `{method, species, name_mode, ip_scale, n_heldout}` exactly as today. Then it calls the **existing** `summarize` and `analysis.build_analysis`, and writes `results.json` (with `expected_cells` / `missing_cells`), `analysis.json`, `results.md`. Aggregation and best-of-k are pure and unit-tested.

## 4. Safety

- Every GPU process is single-model-family for life. Phase 2 = SDXL only (`test_generate_gpu` passes). Phase 3 = one embedder per process (`test_verify_gpu::test_loaders_smoke` passes for the loaders). We never run the in-process generate+verify churn that segfaults.
- `env.setup()` (gc.disable + CUDA alloc conf) unchanged; called first in each worker's `main`.
- Any single seed-row / image failure logs and continues; a phase never aborts on one bad cell. Missing rows surface in `results.json['missing_cells']`, and a `--resume` re-run refills them.

## 5. Validation (the trust gate)

Same seeds ⇒ same images ⇒ same metrics. Acceptance test: run slow `run_eval` and `run_eval_fast` on 1–2 species with identical config, and assert every per-cell metric agrees within a small float tolerance (only the best-of-k seed pick may differ, and it is deterministic given identical images). Until that cross-check passes, the fast path is not used for the full corpus. Pure pieces (manifest, aggregation, resume-skip) get Protocol-fake unit tests; the GPU workers get `@pytest.mark.gpu` smoke tests.

## 6. CLI / defaults

`python -m graft.run_eval_fast --root <treevill> [--species A,B,... | --n-species N --select-by unique]`
with matrix-trim defaults that encode the sprint's findings:

| flag | default | note |
|---|---|---|
| `--methods` | `ours,b0,b1,b2` | `ours_notree` opt-in (degenerate until the gate is recalibrated) |
| `--ip-scales` | `0.6` | best from the sprint; sweep only on request |
| `--name-modes` | `neutral` | `named` is an opt-in control |
| `--seeds` | `2` | matches slow-path `1 + n_refine` |
| `--use-part-tree` | off | sprint default; drops GroundingDINO from Phase 3 |
| `--resume` | off | skip cells whose PNG / score rows exist |
| `--config` | `configs/pipeline.yaml` | |

The full-corpus default run is thus ~4 cells/species (vs 22), on top of the ~14-load structure.

## 7. Feasibility

Every model, the subprocess pattern, and `analysis.build_analysis` / `summarize` are proven. The new code is: a pure manifest expander, two single-model GPU workers whose bodies are lifts of existing `generate_lever_a` / `worker_metrics` logic, a pure aggregator, and a thin orchestrator. No new weights, no new dependency. Expected wall-clock for the sprint matrix: **~1–2h** (from 27h); the trimmed full corpus (66 species × ~4 cells): an overnight run.

## 8. Self-review

- **Placeholders:** none. Every phase, flag, and file is named.
- **Consistency:** row schema, tags, and the `build_analysis`/`summarize` reuse match `run_eval.py` exactly; manifest rules mirror `evaluate`'s slow-path rules (swept-vs-unswept, adaptive split, stale-KG rebuild).
- **Scope:** one implementation plan — 4 new files (`run_eval_fast.py`, `worker_generate_many.py`, `worker_score_many.py`, plus a shared `eval_common.py` for the manifest / prompt_for / KG-usable helpers) + a small `Models.generator.set_scale` accessor + tests. Slow path untouched.
- **Ambiguity:** best-of-k tie-break is "highest verify score, lowest seed on tie" — made explicit in Phase 4.
