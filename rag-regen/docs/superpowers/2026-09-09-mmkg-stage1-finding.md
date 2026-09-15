# MMKG Stage-1 finding — attribute verifier + organized retrieval

**Date:** 2026-09-09
**Decision:** **WIND DOWN on this corpus** — but see the correction below: this was **not a fair test of
the hypothesis.** The pre-registered feasibility gate is not met and the Role-2 verifier gate is NEGATIVE
on the LAION-100k / ImageNet-class holdout used here.

> **Correction (2026-09-09, post-run).** This test was run on the wrong substrate. Two design errors:
> (1) the verifier's *target attributes* were built from the same noisy LAION-100k name-retrieval slice
> (a mis-application of the Role-1 "reference parity" rule to Role 2 — a verifier's targets should be
> *correct*, from a clean source, not parity-constrained), and (2) the domain was arbitrary ImageNet
> objects with 6 generic slots, where fine-grained attribute verification is ill-defined. The intended
> substrate is a **fine-grained species dataset with an authoritative taxonomy** — **FewMedical-XJAU**
> (`kg_test/reports/2026-09-07-fewmedical-xjau-scope.md`) per the user, or Treevill. So the ~60% abstain
> rate and the NEGATIVE gate are a verdict on *MMKG-from-LAION-on-arbitrary-objects*, **not** on the
> fine-grained-attribute-verifier hypothesis, which remains untested pending the right corpus.
**Spec (frozen):** `docs/superpowers/specs/2026-09-09-mmkg-verifier-retrieval-stage1-design.md`
**Run:** `outputs/mmkg_stage1/result.json` (+ 48 per-case traces). One VLM load, offline scoring; ground
truth = the **human** Phase-B holdout labels (`outputs/phase_b/labels_holdout48.csv`, `verdict_identity`).

## Headline

| pre-registered gate | required | observed | met? |
|---|---|---|---|
| **Feasibility (Step 3):** rare-cohort concepts decidable (≥2 reliable attributes) | ≥ 18/24 | **9/24** | ✗ |
| **Role 2 recall** (rare identity-FAIL caught) | ≥ 0.75 | **0.208** (5/24) | ✗ |
| **Role 2 FP rate** (controls wrongly routed) | ≤ 0.083 | **0.167** (4/24) | ✗ |

All three fail. Per the frozen protocol (under-powered feasibility → not passed, never widened post-hoc;
single pre-registered analysis), the pilot **winds down**.

## What happened, and why (the interesting part)

The data limit is **not** slice size. K=40 re-retrieval from the same LAION-100k index returned full
slices (median 39, mean 37.2 images/concept; **zero** undersized). Yet **29 of 48 concepts are
undecidable** — the name-blind VLM reads over those ~39 images do not reach a stable per-slot consensus
(`V_vis ≥ 3` AND plurality `≥ max(2, ceil(0.5·V_vis))`) for ≥2 of the 6 attribute slots.

The cause is **retrieval heterogeneity**, not scarcity: text→image retrieval of a rare concept name over a
100k web corpus returns visually inconsistent images (the corpus's own `retrieve.py` documents this —
"axolotl retrieved crochet toys"), so the attribute reads disagree and no consensus target forms. The
MMKG's attribute store is only as good as the slice it is built from, and name-retrieval over LAION-100k
is too noisy to anchor a fine-grained attribute on 60% of these concepts.

This is the same shape as GRAFT's negatives: **the bottleneck is retrieval quality, not the graph
mechanism.**

## Role 2 — the verifier (score-only, no gating this phase)

Confusion over all 48 (ABSTAIN → PASS, per §6.5): tp=5, fp=4, tn=20, fn=19, **n_abstain=29**.
- **Rare (n=24):** recall **0.208** (5 caught, 19 missed — 15 by abstaining, 4 by wrong PASS).
- **Control (n=24):** FP rate **0.167** (4 false positives, 14 abstain, 20 correct PASS).
- **Cross-tab vs the semantic branch (on rare FAILs):** both 3, **MMKG-only-catch 2**, **semantic-only-catch
  9**, neither 10. The MMKG verifier catches *fewer* unique identity failures than the coarse semantic
  verifier it was meant to complement — no positive signal, and on the decidable subset it is worse than
  the baseline, not better.

**Interpretation (not over-claimed):** with 29/48 abstains driven by the retrieval-consensus limit, this
is not a decisive refutation that fine-grained attribute checking *cannot* help identity detection — it is
a clean absence of positive signal, plus a hard data/retrieval limit, on this corpus. Consistent with the
detector line's own conclusion that the problem "needs a genuinely different signal," an MMKG attribute
verifier built on noisy name-retrieval is not that signal here.

## Role 1 — organized retrieval (measurable, non-trivial; structured-coverage inert)

Over 48 concepts: mean pool **Jaccard 0.109**, **selected-ref-identical 0.0%** — the flat arm (5 stored
LAION hits, top-1) and the MMKG-organized arm (40 re-retrieved, deduped, contamination-guarded slice,
medoid) **never pick the same reference**. So Role 1 is a genuinely distinct intervention, not a no-op
(the testability guard the design required is satisfied).

**Honest limitation (per review):** the *structured-coverage* dimension did not materialize —
`mmkg_unit_size` = 1.0 and `mmkg_part_types` = 0.0 for every concept, because these ImageNet concepts carry
no part decomposition, so the coverage-aware selection collapses to a single medoid. Role 1's Phase-1
result is therefore **pool/reference divergence only**, not multi-instance structured coverage; and since
our results predict parity-not-gain on fidelity, this divergence is not claimed as an improvement.

## Honesty invariants (held)

- **Ground truth = human labels only.** The verdict derives solely from `case_verdict(targets, draft_read,
  judge_fn)` (all VLM-derived); DINO/semantic/reranker appear only in the trace `scores_reused` block and
  the descriptive cross-tab — never in the gate. (Verified in the whole-branch review.)
- **No VLM-vs-VLM ground-truth loop:** the detector is a VLM; the truth is human. **No DINO-delta-sign**
  anywhere in the decision.
- **Reference parity + contamination guard** applied to every built slice (SHA-256 vs the case's
  ground-truth references).
- **Missing/occluded attributes abstain, never contradict** — the control-FP guard; controls still drew 4
  FPs, all from confident cross-slot contradictions, not from missing reads.

## Consequence

- **Do not** proceed to Phase 2 (wiring the verifier as a regeneration gate) or a full benchmark — there is
  no positive signal and the feasibility gate fails.
- **Standing conclusion reinforced:** across GRAFT (visual repair, rare-attribute transfer) and now this
  MMKG-in-rag-regen pilot, a relational/attribute MMKG gives **no advantage over retrieval**; the leverage
  is in **retrieval quality**. Here the MMKG's own attribute consensus is defeated by noisy name-retrieval —
  a retrieval-quality problem, not a graph problem.
- **If ever revisited:** it needs (a) a cleaner per-concept image source than LAION-100k name-retrieval (so
  a stable attribute consensus can form), and (b) a non-VLM-circular attribute ground truth — i.e. a
  different dataset, not a re-run here. Low priority given this result.

## Reproducibility

`ragregen/mmkg/` (schema, vlm_read, verifier, retrieval_arms, evaluate, trace, build, run_stage1); 22 unit
tests pass; whole-branch reviewed (a Critical case_id-join bug and the FP-gate boundary were caught and
fixed pre-run). Every threshold frozen in the spec before the 48 cases were scored. Run artifact
`outputs/mmkg_stage1/result.json`; per-case traces under `outputs/mmkg_stage1/trace/`.
