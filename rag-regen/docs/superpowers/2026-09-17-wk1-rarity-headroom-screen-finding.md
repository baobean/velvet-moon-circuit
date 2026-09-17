# Wk1 rarity/headroom screen — CUB dropped, PlantCLEF cleared

**Date:** 2026-09-17
**Gate (spec §8):** ~30 species/dataset, text-only FLUX draft → does it *fail* (DINO +
eyeball) **AND** is single-image SigLIP retrieval *weak*? Both halves must hold, or the
dataset is dead on arrival (the prior MMKG-as-picker trap: if retrieval alone already
nails identity, there's nothing left for any mechanism — MMKG graph or part-composition —
to add over it).

## Result

| | CUB-200-2011 | PlantCLEF2024 |
|---|---|---|
| Species sampled | 30 (random, seed=0) | 30 (long-tail: 6-20 images/species, seed=0/1) |
| Retrieval: leave-one-out SigLIP top-1 acc. | **94.0%** (141/150) | **83.2%** (124/149) |
| FLUX-failure: mean object-cropped DINO cosine | **0.234** | **0.083** |
| Verdict | **FAIL — drop** | **PASS — proceed** |

Interactive reports (draft images, per-species scores): [CUB](https://claude.ai/artifact/6EvsNDHEkMZRtkiYzoqQmB), [PlantCLEF](https://claude.ai/artifact/NFaY5vX3NSZaiM81MH6smH).
Raw data: `outputs/headroom_cub_wk1_screen_20260917_115409.json`,
`outputs/headroom_plantclef_wk1_screen_20260917_133000.json`,
`outputs/fidelity_cub_wk1_screen_cropped.json`, `outputs/fidelity_plantclef_wk1_screen_cropped.json`
(all gitignored — rerun the two screen scripts to regenerate).

**CUB: dropped, exactly as predicted.** This confirms the design doc's own prior ("CUB is
the droppable one, PlantCLEF is load-bearing") with a number instead of an assumption.
Retrieval alone already solves species ID at near-ceiling accuracy on this sample — the
`single_medoid` baseline (which itself picks its reference via the same class of
retrieval) would already look nearly perfect on identity, leaving no headroom for
part-structured conditioning to show a gain over it. No further GPU time goes into CUB:
no real MMKG store build (Task 6), no wk2 pilot species from it.

**PlantCLEF: cleared, both halves.** Retrieval is meaningfully softer than CUB's
(83.2% vs 94.0%), and — more informative than the aggregate — a real confusable
pocket exists: 6 of 30 sampled species scored 1/5–3/5 on leave-one-out ID (`Agrostis ×
murbeckii`, `Sisymbrium polyceratium`, `Narcissus viridiflorus`, `Teucrium turredanum`,
`Rostraria litorea`, `Marsilea batardae`). **The wk2 pilot's ~15-species list should be
drawn from this confusable pocket**, not a fresh random sample — that's where genuine
headroom lives, and a random draw would dilute the signal with easier species. FLUX
failure is unambiguous and uniform: every one of the 30 drafts scored low (mean 0.083,
max 0.266), and the highest scorer was hand-verified to be a genuine miss (see below) —
no case looks like a plausible depiction of its real species.

## A real metric-reliability finding, not just a screen result

v1 of the fidelity script scored **whole, uncropped images**. A user-caught anomaly —
Common Tern scoring 0.516, one of CUB's *highest* — led to two rounds of investigation:

1. **First hypothesis (background confound), tested and rejected.** Rewrote
   `scripts/draft_fidelity_score.py` to ground and crop to the coarse-term object (draft
   *and* every reference) before scoring, matching the crop discipline `ragregen/metrics.py`
   already documents elsewhere in this codebase ("a rare parrot in a wide scene scores
   against its own background unless the crop is taken"). Common Tern's score barely
   moved: **0.507**, still object-cropped. The crops were verified by hand to be genuinely
   tight, background-free bird-only images on both sides — cropping was not the fix.
2. **Root cause, confirmed by inspection.** FLUX drew a plain white/grey gull (stout
   orange bill, orange legs, no cap, no forked tail); the real Common Tern has a black
   cap, thin dark-tipped red bill, red legs, and a forked tail — genuinely different
   birds. DINOv3, even on clean crops, rates them 0.507-similar: it is tracking coarse
   "pale seabird, similar pose/size/color" structure, not the specific diagnostic field
   marks that separate closely related species. This is a real sensitivity limit of the
   encoder for fine-grained categories, not a framing bug.

**Why this matters beyond the screen:** the real pipeline's evaluation (Tasks 4/5/7,
`gain_verdict`) uses DINO safely — as a **paired, within-case delta**
(`partgraph_dino − single_medoid_dino` for the *same* species against the *same*
held-out refs), never as an absolute cross-species score. This wk1 screen used DINO the
risky way: absolute scores compared *across different species* to rank a batch — exactly
the usage mode where the insensitivity above bites. Implication for reading these
numbers and any future absolute-DINO triage:
- **Low scores stay trustworthy** — hard to be fooled into a *low* score by accident, so
  every case flagged as a clear FLUX-fail here is a real fail.
- **High/mid scores need an eyeball check before being called a pass** — a numeric cutoff
  alone is not sufficient, which is exactly why the spec says "DINO + eyeball," not
  "DINO alone."
- PlantCLEF's numbers were spot-checked against this exact failure mode (the top scorer,
  0.266, was hand-verified — see the report) and hold up; nothing in the PlantCLEF
  sample showed CUB's false-positive pattern.

## What's committed

- `scripts/retrieval_headroom_screen.py` — SigLIP leave-one-out top-1 species ID, the
  headroom half.
- `scripts/draft_fidelity_score.py` (v2) — object-cropped DINO cosine, the FLUX-failure
  half. v1 (whole-image) is superseded; do not revert to it.
- `configs/partgraph_cub_wk1_screen.yaml`, `configs/partgraph_plantclef_wk1_screen.yaml`
  — the two 30-species samples.

## Next step

PlantCLEF has cleared wk1 — Task 6's PlantCLEF half (`build_plantclef.py`) can now
proceed per the plan (`docs/superpowers/plans/2026-09-17-mmkg-partgraph-phase0.md`).
When picking the real ~15-species pilot list for the wk2 gate (Task 7 Step 1), draw from
the 6-species confusable pocket identified above first, filling out to ~15 with the next
weakest-retrieval species from the full 30-species sample or a fresh draw from the same
6–20-image long-tail band.
