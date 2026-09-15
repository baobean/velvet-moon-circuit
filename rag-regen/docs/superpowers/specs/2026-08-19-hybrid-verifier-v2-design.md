# Hybrid Verifier V2 Development Design

**Date:** 2026-08-19

**Status:** Approved direction; implementation delegated

**Predecessor:** `docs/superpowers/specs/2026-08-18-hybrid-verifier-validation-design.md`

## 1. Objective

Develop a conservative verifier-v2 policy using only already-exposed development artifacts. Do not integrate it into the production scheduler and do not create or inspect a new holdout until the policy passes a CPU-only, leave-one-concept-out readiness gate.

The v1 holdout is no longer an independent holdout. It may be used for v2 development, but all reports must call it `development`, never `validation` or `test`.

## 2. Evidence from v1

The frozen v1 policy failed for three independent reasons:

- reranker-assisted routing produced `5/12` false positives, above the maximum of `1`;
- selector sign accuracy was `0.6226`, below `0.7500`;
- visual review found unnecessary replacement of a correct Labrador and several weak rare-concept selections.

The inpainting mechanism remains useful: the selected rare cohort had mean cropped-DINO delta `+0.2313`, while no selected result crossed the harmful threshold. V2 therefore changes routing, selection, and evaluation discipline—not the passing inpainting implementation.

## 3. Protocol correction

Identity truth and pipeline eligibility are separate immutable facts:

- `identity_truth`: `PASS`, `FAIL`, or `UNJUDGEABLE`, assigned from the draft alone;
- `selector_eligible`: whether a valid mask and at least one prepared edit reference/candidate exist;
- `eligibility_reason`: a mechanical reason such as `eligible`, `mask_not_grounded`, `no_prepared_reference`, or `corrupt_artifact`.

An invalid mask never rewrites `identity_truth`. All identity-labeled cases remain in detector metrics. Only selector and end-to-end metrics use `selector_eligible=true`, with excluded denominators and reasons reported explicitly.

Every manifest and policy artifact is immutable: writers refuse to overwrite, record SHA-256 hashes for source files, and preserve source paths, timestamps, git status, model IDs, and constants. The existing 24-case data receives `protocol_status: reconstructed_after_score`; future holdouts must receive `protocol_status: frozen_before_score`.

## 4. Recommended policy family

### 4.1 Routing

Replace the permissive union with guarded confirmation:

```text
route = semantic_failure OR
        (draft_text_relevance < text_threshold AND
         draft_reference_relevance < reference_threshold)
```

The conjunction is the smallest change consistent with the observed failure: semantic decisions had zero false positives, while low reference relevance alone caused five. On the exposed 22 evaluable cases, the conjunction has feasible in-sample settings with full recall and zero false positives; that observation motivates the family but does not freeze thresholds.

Thresholds are selected only by deterministic leave-one-concept-out development evaluation. Semantic-only, reference-only, and v1-union policies remain reported baselines.

### 4.2 Repair selection

Retain the interpretable two-margin family:

```text
eligible = attempt_text - draft_text >= text_margin AND
           attempt_reference - draft_reference >= reference_margin
```

V2 permits a nonzero inclusive `reference_margin`; it remains conservative and deterministic. Among eligible attempts, select by text relevance, then reference relevance, then lower attempt number. If none is eligible, retain the draft.

Candidate text/reference margins are selected on development data by leave-one-concept-out evaluation. A setting is feasible only when sign accuracy is at least `0.750`, harmful selections are at most `1`, and mean selected cropped-DINO delta is strictly positive. Among feasible settings, choose the highest mean DINO delta, then exact-DINO-best rate, then the larger text margin, larger reference margin, and finally lexicographically smaller numeric tuple. This prevents a no-op policy from winning merely through class imbalance.

## 5. Development evaluation

Use the exposed 24-case run: all 24 identity rows participate in detector folds and the 22 mechanically eligible rows participate in selector/end-to-end folds. Do not combine it with older studies whose reranker/candidate artifacts are not schema-identical.

For each held-out concept fold:

1. fit detector thresholds and selector margins on all other concepts;
2. apply the fitted policy to the held-out concept without refitting;
3. aggregate only held-out predictions;
4. report every fold's fitted parameters and decision.

Development readiness gates retain the v1 bar so failure cannot be hidden by changing targets:

- detector failure recall `>= 0.733`;
- detector false positives `<= 1` on the 12-control exposed cohort, or false-positive rate `<= 0.0834` when populations are combined;
- selector sign accuracy `>= 0.750`;
- harmful selections below `-0.02`: at most `1`;
- selected rare mean cropped-DINO delta strictly above zero;
- no selected common case with material identity/layout damage in exhaustive review.

Report Wilson intervals, confusion matrices, pairwise confusion/precision/recall, all attempted settings, and exact denominators. Passing is a readiness gate for a new holdout—not evidence of generalization.

## 6. Scope boundaries

Phase A is CPU-only and may read existing JSON/CSV/PNG artifacts. It must not:

- rerun FLUX, Qwen, DINO, CLIP, or SigLIP;
- modify v1 artifacts or labels;
- change the inpainting prompt, mask, seed, scheduler, or production verifier choices;
- integrate `--verifier hybrid-v2`;
- launch the 92-case experiment;
- claim the fitted policy passed an independent holdout.

If no setting passes leave-one-concept-out readiness, stop. The next design must add a genuinely new signal, such as candidate semantic confirmation; it must not keep widening threshold searches.

## 7. Phase B boundary

Only after Phase A passes, write a separate design and plan for an independently curated holdout. That plan must include:

- unused concepts and references not inspected during v2 fitting;
- at least 24 rare targets and 24 common controls;
- immutable identity and eligibility manifests frozen before verifier/DINO scoring;
- a predeclared maximum of 2 false positives among 24 controls;
- full three-arm cropped DINO, whole-image CLIP, whole-image SigLIP, outside-mask preservation, Wilson/bootstrap intervals, provenance, and exhaustive visual review;
- the same rule that production integration and the 92-case run occur only after every gate passes.

## 8. Alternatives rejected for Phase A

- **Retune only the reference threshold:** rejected because it repeats the v1 single-signal failure and overfits an exposed holdout.
- **Semantic-only routing:** retained as a baseline, not the primary family, because it had zero false positives but only `0.600` recall.
- **Train a learned classifier or add another model immediately:** deferred until the two existing reranker signals fail cross-validated readiness; additional model complexity is not yet justified.
- **Complete the missing v1 CLIP/SigLIP report first:** useful for archival completeness, but it cannot change the failed promotion decision and should not consume the next experiment budget.
