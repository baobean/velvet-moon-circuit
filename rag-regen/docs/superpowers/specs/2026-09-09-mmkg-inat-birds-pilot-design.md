# MMKG attribute-verifier — iNaturalist birds discrimination PILOT (design)

**Date:** 2026-09-09
**Status:** design, pre-registered. Author sign-off pending.
**Context:** The LAION/ImageNet-holdout run (`docs/superpowers/2026-09-09-mmkg-stage1-finding.md`) was
corpus-confounded — MMKG targets built from noisy name-retrieval over arbitrary objects, ~60% abstain, no
fair test of the hypothesis. This pilot re-tests the **core hypothesis** on the right substrate: a
fine-grained species domain with an authoritative taxonomy and dense clean images.

> **Hypothesis.** An MMKG of per-species **visual attributes** (built name-blind from clean images)
> discriminates a bird from its **same-family siblings** — i.e. it carries fine-grained identity signal that
> a coarse "is this the concept" check lacks.

This is a **pilot**: prove the pipeline and get an early signal on ~25 species before scaling. The bars below
are pre-registered and not changed after seeing results. Ground truth = **iNaturalist's authoritative species
labels** (the VLM is the detector, the labels are truth — the guard against the Stage-B VLM-vs-VLM
circularity).

## 1. Substrate (frozen)

- **Dataset:** iNaturalist 2021 (FGVC8), `val` split — **10 images/species**, full Linnaean taxonomy per
  category (`class`, `order`, `family`, `genus`, species). Public (AWS Open Data
  `s3://ml-inat-competition-datasets/2021/`); research-only license, **no image redistribution** (internal use,
  we publish only aggregate numbers). Bird subset confirmed: 1486 species, 108 multi-member families.
- **Storage:** images under the bulk area (primary NAS full) — `/mmlabworkspace_new/Students/tuanld/
  soict-2026-data/inat2021_birds/` (or a path the user names); metadata/derived files in the repo.

## 2. Species selection (frozen, deterministic)

From `val` bird categories (`class == "Aves"`), keep families with **≥3 species** (so same-family hard
negatives exist within the pilot). Sort those families by (species-count ascending, then family name); take
families in that order, and within each take species sorted by iNat `category id` ascending, accumulating until
**≥24 species across the fewest whole families that reach 24** (whole families only — never a partial family,
so every kept species has ≥2 in-set siblings). Record the exact selected set in the run artifact. (Ascending
family size favours small, tight families — the hardest, cleanest discrimination.)

## 3. Build / test split (frozen)

Each species has 10 `val` images, sorted by image id. **First 7 = build** (attribute consensus), **last 3 =
test** (held-out). The code asserts build ∩ test = ∅. Target attributes are read from build images only; every
verdict is scored on test images only.

## 4. Attribute ontology (frozen — 6 bird slots)

Name-blind VLM read, `do_sample=False`, one JSON object per image with exactly these keys:
`primary_plumage_color`, `secondary_plumage_color`, `plumage_pattern`, `bill_shape`, `bill_color`,
`distinctive_markings`. Read prompt frozen (adapted from the Stage-1 READ_PROMPT: "Describe ONLY the single
bird in this image. Do not name the species. Return a JSON object with exactly these keys: … Each value a
short phrase, or \"not visible\" if you cannot tell. Output only the JSON."). Consensus, NOT_VISIBLE, and
`norm` are reused verbatim from `ragregen.mmkg.schema` (target attribute = slot with `V_vis ≥ 3` build reads
and a strict-plurality value ≥ `max(2, ceil(0.5·V_vis))`; a species is **decidable** iff ≥2 target attributes).

## 5. Discrimination protocol (frozen)

Reuse `ragregen.mmkg.verifier.case_verdict` verbatim (present/contradicted/missing; the blessed exact-match
short-circuit; FAIL iff ≥1 present contradiction; ABSTAIN if undecidable or nothing present). For each **test
image t of species S in family F**, form pairs against candidate species' target profiles:
- **SAME pair:** `case_verdict(profile[S], read(t))` — ground-truth label **SAME** (should PASS).
- **SIBLING pairs:** for each other in-set species S′ in family F (cap at the **3** nearest by iNat
  category-id distance, for cost), `case_verdict(profile[S′], read(t))` — ground-truth label **DIFFERENT**
  (should FAIL).
Each test image is read **once** (name-blind); the same read is scored against S and its siblings.

## 6. Metric and pre-registered gate

Over all (t, candidate) pairs, treating verdict FAIL as "predicted DIFFERENT", PASS/ABSTAIN as "predicted
SAME":
- **sibling-recall** = P(FAIL | label DIFFERENT) — correctly rejecting a same-family sibling.
- **true-FP** = P(FAIL | label SAME) — wrongly rejecting the bird's own species.
- **abstain rate** (reported; abstains count as predicted-SAME, i.e. a miss on SIBLING pairs — conservative).
- **PASS bar (pilot, all required):** `sibling_recall − true_FP ≥ 0.20`, `sibling_recall ≥ 0.40`, and decidable
  species ≥ 18 of the ~24 (else under-powered → not passed). **Baseline (reported, not gating):** SigLIP
  cosine of t to each candidate species' build-image centroid — does nearest-centroid pick the true species
  over siblings? — so we can see whether the attribute verifier adds over plain embedding similarity.
- **Decision:** PASS → scale to the full-val bird run (pre-registered separately). FAIL → the attribute-verifier
  signal is absent even on the ideal substrate; wind down with that conclusion (much stronger than the
  confounded LAION null).

## 7. Anti-shopping / honesty

Every knob (slots, selection rule, split, consensus, verdict rule, gate) is frozen here before any image is
read. Ground truth is iNat labels, never a VLM self-label and never a DINO/embedding signal (SigLIP is a
reported baseline, not the truth). Single pre-registered analysis; abstains disclosed, not dropped.

## 8. Scope, reuse, non-goals

- **Reuses verbatim:** `ragregen.mmkg.{schema, verifier, evaluate}`; the read/judge pattern from
  `ragregen.mmkg.vlm_read` (with the bird READ_PROMPT); `ragregen.vlm.QwenVLM`; `ragregen.encoders` for the
  SigLIP baseline.
- **New (pilot):** an iNat `val` corpus loader (parse categories → Aves → selection §2; locate build/test
  image paths), the bird ontology read prompt (§4), and a small discrimination harness (§5–6) + `main`.
- **Non-goals:** the full-val bird run (gated on this pilot), rag-regen draft-failure detection (a separate
  downstream question — this pilot tests whether the signal exists at all), train_mini (50/species) density,
  any Role-1 conditioning use.

## 9. Feasibility / cost

~24 species × 7 build + 3 test = ~240 image reads + a few hundred judge calls — comparable to the Stage-1 run
(~1 GPU-hour). Download: stream `val.tar.gz` (8.4 GB) and extract only the selected bird category dirs (~a few
hundred images kept), or the individual category dirs if a lighter path exists. One VLM load; offline scoring.
