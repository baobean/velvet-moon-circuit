# Stage-A′ — Per-Part Inpaint Transfer Under a Real Deficit (design)

**Date:** 2026-09-07
**Status:** design, approved for planning
**Supersedes-for-execution:** the Stage-A recovery test in
`2026-09-05-relational-mmkg-parttype-transfer-design.md` (that spec's mechanism and hub
construction are kept; only the *test* is rebuilt). Reads on top of the Stage-A result report
`reports/2026-09-06-stage-a-result.md`.

## 0. Why this exists

Stage A's make-or-break test (`+Hub-k − +RawNN-k`) came back **NULL, but with a fatal caveat: the
starvation manipulation never produced a fidelity deficit** (recovery curve flat, build=1 actually
highest). With no deficit to recover, the recovery design could not test the hypothesis, and a
second confound — whole-image IP-Adapter conditioning on borrowed *part* crops diluted identity, so
borrowing anything slightly *hurt* — muddied the arm comparison.

This spec rebuilds the test so the verdict is trustworthy. Two changes, both agreed:

1. **A guaranteed, measurable deficit** — mask a specific part region out of a real held-out image
   (an information hole) and require the pipeline to restore it. The hole *is* the deficit.
2. **Per-part transfer via inpainting** — restore only the masked part region, conditioned on
   borrowed part-crops, instead of whole-image conditioning. This is also exactly the downstream
   rag-regen "repair the wrong region" operation, so the test double-serves the direction.

If, under a **working** deficit and **per-part** transfer, `+Hub ≈ +RawNN`, that is the decisive
negative and we report it. This re-run exists to make that verdict trustworthy — not to rescue the
idea.

## 1. Experimental unit

One **restoration cell**: `(concept C, part-type P, build_P, arm, draw)`.

Per `(C, P, target-image)` the geometry is computed **once and cached**, reused by every arm /
build_P / draw so the only thing that ever varies within a cell is the **selector**:

- Choose a **held-out** image of C (not in the build set) where `GroundingDinoDetector` localizes P
  with confidence ≥ threshold → **box**.
- `SamSegmenter(box)` → **SAM mask** (tight to the part).
- **Inpaint region = the box** (rectangular-ish; SDXL-inpaint behaves better and avoids a tight-mask
  seam). **Scoring region = the SAM mask** crop inside the box (evaluation stays tight to the actual
  part, not surrounding background).
- The box and SAM mask for a target image are **identical across all arms and build_P** in that cell.

## 2. The deficit and the starvation axis

- **Deficit (guaranteed):** the box region is masked out of the target image before inpainting →
  no part information remains there. Isolated-with-`build_P=0` must hallucinate P; the borrowing
  arms supply it from crops.
- **Starvation axis:** `build_P ∈ {0, 1, 2, 4}` — the number of C's **own** P-crops the inpaint may
  condition on. Drives the recovery curve. `build_P=0` is total own-evidence starvation.
  `build_P` is **clamped to the number of own P-crops available** in C's build set for P (a concept
  with 3 own P-crops has no distinct `build_P=4` cell); clamped levels are logged and de-duplicated.
- **Set disjointness (no leak):** the **own P-crops used for conditioning** come from C's **build**
  split; the **held-out P-crops used for scoring** come from C's **held-out** split (target image
  excluded). These two sets are disjoint by the standing `split_refs` split, so a conditioning crop
  is never also a scoring reference.

## 3. Arms and the conditioning-budget invariant

Let, per `(C, P)` cell,
**`k_eff = min(k, #borrowable P-crops in C's hub after excluding C's own crops and the target image)`**,
with `k = 4` (Stage-A P4). The **same `k_eff`** is used by all three borrowing arms in the cell; if
`k_eff < k` it is clamped for all of them, and if `k_eff = 0` the cell is **skipped**. (This is the
Stage-A Task-3/Task-8 cross-task invariant, enforced from the start.)

| Arm | Selector (which crops condition the inpaint) | Count at `build_P=b` |
|-----|-----------------------------------------------|:--------------------:|
| **Isolated** (reference, *not* quantity-matched) | own P crops only | `b` |
| **+Random-k** | own + `k_eff` random off-concept P crops | `b + k_eff` |
| **+RawNN-k** | own + `k_eff` nearest-neighbour P crops (all concepts, no hub) | `b + k_eff` |
| **+Hub-k** (ours) | own + `k_eff` sibling P crops via the PartType hub | `b + k_eff` |

Within any cell and any `build_P`, the three borrowing arms carry an **identical crop count**
(`b + k_eff`) and differ **only** in the ranking function → **`+Hub − +RawNN` is a pure selector
effect.** Isolated deliberately carries fewer (`b`); it answers "does borrowing help at all?", never
the make-or-break.

**Explicit budget at the make-or-break levels:**

| Arm | `build_P = 0` | `build_P = 1` |
|-----|:---:|:---:|
| Isolated | 0 | 1 |
| +Random-k | `k_eff` | `1 + k_eff` |
| +RawNN-k | `k_eff` | `1 + k_eff` |
| +Hub-k | `k_eff` | `1 + k_eff` |

**Guards:**
- **Contamination:** borrowed crops for all three +k arms are drawn from the *cross-concept*,
  *part-restricted* pool with C's own crops **and** the target image excluded (SHA-256 / id guard,
  as in the sprint / Stage-A).
- **Part-restricted pool:** all arms borrow only P-labelled crops (the Stage-A final-review bug was
  an off-part cross-part pool; here the pool is part-restricted by construction).
- `k_eff` is **logged per cell** so analysis can confirm equal realized counts.

## 4. Metric

**Primary: DINOv3 cosine** of the restored SAM-region crop vs C's **other held-out P crops**
(excluding the target). **Secondary: SigLIP2, CLIP-I.** Name-free, same metric family as the sprint /
Stage-A, so numbers are directly comparable. Semantic ("is it the right *kind* of P for *this*
concept"); honest — if hubs make C's part look like a sibling's, it only scores well if the siblings'
P truly match C's P.

## 5. Consumer path (infra)

Additive, reuses Stage-A assets; no rewrite of the graph or crop store.

1. **`SdxlIpGenerator.inpaint_multi(...)`** — new method on the existing generator in `models.py`.
   Lazily builds an `AutoPipelineForInpainting` SDXL pipeline with `load_ip_adapter` (same
   repo/weight/scale as the text2img path). Signature mirrors `generate_multi`: base image, mask,
   list of conditioning crops → per-image CLIP encode + **weighted sum inside `torch.no_grad()`**
   (carries the Stage-A VRAM-leak fix). `generate_multi` (text2img) stays untouched — Stage-A
   reference path.
2. **`graft/parts.py`** — thin wrappers: `locate_part(image, part_label) → box`
   (`GroundingDinoDetector`), `box_to_mask(image, box) → mask` (`SamSegmenter`), `crop_box`.
3. **`graft/restore_select.py`** — the four selectors behind one interface
   `select(concept, part, k_eff, pool) → crop_ids`: `isolated`, `random`, `rawnn` (cosine k-NN over
   the part-restricted cross-concept pool), `hub` (siblings via `graph.json` PartType hub). Enforces
   `k_eff` + contamination exclusion in one place; every arm shares this code except the ranking fn.
4. **`graft/worker_restore_cell.py`** — GPU, subprocess-per-cell (standing architecture). One
   `(concept, part, build_P, arm, draw)` cell: build/cache box+mask, gather conditioning crops via
   the selector, `inpaint_multi(box mask)`, crop the SAM region, score with dino/siglip/clip vs
   held-out P crops → write one row. **One shared consumer**; arm swaps only the selector.
5. **`graft/run_restore.py`** — resumable driver behind `scripts/gpu_queue.sh`, resident two-phase
   (inpaint-all → unload → score-all) to avoid per-cell model reloads.
6. **`graft/restore_analysis.py`** — recovery curves + `hub_vs_rawnn` paired-delta (reused shape from
   `starvation_analysis.py`).

**GPU/VRAM:** inpaint pipeline + IP-Adapter + 3 embedders will not co-reside in 24 GB (Stage-A
lesson) → **two-phase**; `torch.no_grad()` + `empty_cache` per gen.

**Reused as-is:** `outputs/mmkg/graph.json` (12 hubs), the Stage-A part-crop store, the 8
held-out≥4 concepts, `metrics.image_fidelity`, `models.{dino,siglip,clip,detector,segmenter,
generator}`.

## 6. Smoke gate (infra-only — MUST pass before the full matrix)

Run **1 concept** end-to-end (all arms × all build_P × all draws). This is an **infrastructure gate,
not experimental evidence**: **do not launch the full matrix unless it passes; if it fails, fix the
GPU path — never interpret a smoke failure as a result about the hypothesis.**

Explicit pass/fail checks:
1. **Valid SAM scoring masks:** every target image yields a non-empty SAM mask with area within a
   sane fraction of the box (not empty, not the whole image).
2. **Expected `k_eff` counts:** the three borrowing arms carry an identical realized crop count
   (`build_P + k_eff`) in every cell; `k_eff` matches the hub's borrowable count; `k_eff = 0` cells
   are skipped, not silently run with fewer refs.
3. **Contamination exclusion:** no borrowed crop id belongs to the target concept or the target
   image (assert on the logged crop ids).
4. **Actual inpainting happened:** the restored box region differs from the masked input (not a
   pass-through), and inpainting is confined to the box (pixels outside the box unchanged).
5. **Comparable outputs across arms:** all four arms produce a scored row for each `(build_P, draw)`;
   scores are finite and in range; same target/mask used across arms (assert cached geometry identity).
6. **No OOM / no VRAM leak:** peak VRAM stays within budget across the two phases and does not grow
   per generation (watch for the Stage-A ~0.5 GB/gen leak signature).

Manual eyeball: inspect a handful of restored SAM crops for obvious seams / wrong-part fills.

## 7. Run matrix

`(concept, part, build_P, arm, draw)`:
- **concepts:** 8 held-out≥4 (reused).
- **parts:** per concept, part-types with a **borrowable hub** in `graph.json` (~31/32 (concept,part)
  pairs; ~3–4 parts/concept, ≈30 (concept,part) cells).
- **build_P:** {0, 1, 2, 4}. **arms:** {isolated, random, rawnn, hub}. **draw:** 3 (a fixed seed
  fixes the target held-out image + own-crop subset; **shared across arms** so only the selector
  differs).

≈ **1,440 cells**, resident two-phase, resumable overnight via `gpu_queue.sh` (est. ~6–9 h).
Artifacts under `outputs/mmkg/restore/` (rows + `recovery.json` + `hub_vs_rawnn.json`, restored
images under `_imgs/`).

## 8. Analysis

1. **Recovery curves:** mean restored-region DINOv3 fidelity vs `build_P`, one line per arm.
   Expected-if-real signature: Isolated rises with `build_P`; +Hub sits **above** Isolated at low
   `build_P` and the gap **closes** as own evidence arrives.
2. **Make-or-break:** paired `+Hub − +RawNN` per `(concept, part)` at `build_P ∈ {0, 1}`, DINOv3
   primary (+ SigLIP2/CLIP-I); mean + win-rate + Wilcoxon.
3. **Borrowing-helps:** paired `+Hub − +Isolated` and `+RawNN − +Isolated` at `build_P = 0` — does
   per-part inpaint reverse Stage-A's "borrowing hurts"?
4. **Reliability stratification (sprint §0 lesson):** stratify every delta by the concept's held-out
   P-crop count; report the reliable subset as headline, never the diluted flat mean. `k_eff` logged
   per cell to confirm equal realized counts.

## 9. Honesty commitment

If `+Hub ≈ +RawNN` under this working deficit and per-part transfer, that is the **decisive
negative** — report it straight, do not advance to a big MMKG or the rag-regen swap. If the deficit
*still* fails to instantiate (recovery curve flat even with a masked hole), report that as an
infra/measurement finding, not as evidence about the hypothesis.

## 10. Scope & boundaries

**Keep / freeze (reuse, do not touch):** `outputs/mmkg/graph.json` (12 hubs; P1–P3 construction
params frozen), the Stage-A part-crop store, the 8 held-out≥4 concepts, `metrics.image_fidelity`,
the subprocess / resident-two-phase GPU architecture, and the `generate_multi` text2img path
(Stage-A reference, untouched).

**Deferred:** Stage B (AttributeValue hubs + attribute-prior transfer — the *separate* channel of
confound #3, tested only if exemplar transfer earns it here), the `+OracleHub-k` arm, and the
rag-regen A/B swap (still gated on this result).

**Out of scope:** re-inducing hubs / re-clustering, retraining anything, FLUX-Kontext.

**Env / git:** `kg_test` has no working git repo (`.git` is a stub) → **no git init, work in-place,
plain-markdown SDD ledger + briefs/reviews** (same as Stage A). The plan's per-task `git commit`
steps are skipped.

## 11. Open items / risks

- **Deficit may still not bite** even with a masked hole (if IP-Adapter inpaint is robust enough to
  reconstruct P from context alone). §9 covers reporting this honestly; the smoke gate's "actual
  inpainting" check surfaces the pass-through failure mode early.
- **GroundingDINO part localization** may fail on some `(concept, part)` targets → those cells are
  dropped (logged), reducing n for that part.
- **n = 8 concepts** remains modest; reliability stratification (§8.4) is the mitigation, not a full
  fix.

## 12. Parameters to pin in the plan (config-backed, no drift)

All thresholds are config-backed (new `GraftConfig` fields, passed explicitly — the Stage-A P10
lesson) and locked in the plan, not hard-coded in workers:

- `k = 4` (borrowed budget); `build_P ∈ {0, 1, 2, 4}`; `draws = 3`.
- GroundingDINO part-detection confidence threshold (target-image acceptance).
- SAM-mask area sanity bounds (fraction-of-box min/max for the smoke gate check §6.1).
- Inpaint denoise params (steps, strength, guidance) — one setting shared by **all** arms.
- IP-Adapter scale — reuse the Stage-A/sprint default (`ip_scale = 0.6`) unless the smoke inspection
  warrants a small inpaint-specific sweep (deferred; not in the primary matrix).
- Seed scheme mapping `draw → (target image, own-crop subset)`, shared across arms within a cell.
