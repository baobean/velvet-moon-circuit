# Frozen Hybrid Verifier Validation Design

**Date:** 2026-08-18

**Status:** Approved for implementation

**Inputs:**

- `outputs/patch_no_verifier_20260817_120359`
- `outputs/verifier_smoke_20260817_000703`
- `../rag-edit/configs/concepts.json`
- `../rag-edit/data/images/`
- `../rag-edit/data/scenes/`

## 1. Decision being tested

The repaired FLUX Kontext inpainting implementation passed its six-case
mechanism pilot: all six cases improved in held-out cropped DINO, with mean
delta `+0.182528` and no observed layout damage. The remaining uncertainty is
whether an automatic verifier can decide when to repair and which repair to
keep without giving away that gain or damaging already-correct images.

This experiment freezes one verifier policy selected on the existing 22-case
development set and evaluates it on concepts that did not influence that
selection. No threshold or decision rule may change after holdout scores are
viewed.

## 2. Frozen inpainting mechanism

The experiment uses the implementation that produced
`patch_no_verifier_20260817_120359` without modification:

- FLUX Kontext inpainting, not pixel stitching;
- grounded reference crop preparation;
- rejection of references in which the fine concept cannot be grounded;
- edit only inside the prepared mask, followed by the existing boundary blend;
- every attempt starts from the original draft;
- current model, seed, step count, guidance, mask dilation, and prompt builder;
- no CLIP/T5 prompt split experiment during this validation.

Changing any of these creates a new inpainting method and requires a new
mechanism pilot before verifier results are comparable.

## 3. Validation data

### 3.1 Rare target cases

Use the twelve concepts below from the sibling `rag-edit` corpus. They have
not appeared in the current rag-regen controlled dataset or verifier smoke
artifacts. Each has four curated Wikimedia images. The first three are edit
references and the fourth is a held-out DINO answer key.

1. Lagotto Romagnolo
2. Xoloitzcuintli
3. Azawakh
4. Bergamasco Shepherd
5. khachapuri
6. salak
7. cherimoya
8. Hakka tulou
9. stave church
10. trullo
11. okapi
12. saiga antelope

These are unseen to the current FLUX/verifier experiment, but not a claim of a
new public benchmark: they already exist in a sibling project. A later
publication-grade result should use another independently curated holdout.

### 3.2 Common controls

Use twelve common concepts with three local Wikimedia images each. The first
two are edit references and the third is held out for scoring:

1. apple
2. barn
3. church
4. daisy
5. deer
6. donkey
7. Labrador Retriever
8. lizard
9. pine tree
10. pizza
11. stone cottage
12. tiger

Controls test false repair and no-harm behavior. They are not assumed to pass:
fresh FLUX drafts receive the same identity labeling as rare targets.

### 3.3 Drafts and labels

- Generate one fresh FLUX draft per case with the frozen pipeline seed and
  settings.
- Create a contact sheet containing only case id, prompt, and draft.
- Assign `PASS`, `FAIL`, or `EXCLUDE` identity labels before verifier scores
  are opened.
- `PASS` means the exact named concept is visually present; generic-category
  resemblance is insufficient for fine-grained targets.
- `EXCLUDE` is allowed only for an unjudgeable image, invalid mask, or corrupt
  reference, with a written reason.
- Persist labels in the existing `labels.csv` schema.

## 4. Frozen hybrid policy

### 4.1 Draft routing

Retrieve or inject the edit reference before the reranker decision. Route a
draft to repair when either condition is true:

1. the existing semantic Qwen VLM returns failure; or
2. Qwen3-VL-Reranker-2B reference relevance is below `0.30078125`.

The reranker model, prompt, image crop, maximum side `448`, sigmoid activation,
and bfloat16 inference remain exactly those in
`scripts/qwen_reranker_score.py`.

The threshold `0.30078125` is the full-development-set optimum computed before
the holdout run. Equality passes: only scores strictly below the threshold
route through the reranker condition.

### 4.2 Repair selection

For every routed case, score the original draft and each available repair
attempt with both Qwen reranker queries. A repair is eligible only when:

- `attempt.text_relevance - draft.text_relevance >= 0.02`; and
- `attempt.reference_relevance - draft.reference_relevance > 0.0`.

Select the eligible attempt with the highest text relevance. Break an exact
text-score tie by higher reference relevance, then lower attempt number. If no
attempt is eligible, retain the draft.

No DINO, CLIP, SigLIP, human label, or held-out reference may enter routing or
selection.

## 5. Candidate generation and counterfactual arms

Generate candidates once so every policy sees identical images:

- rare cases: up to three attempts, one per edit reference;
- common cases: up to two attempts, one per edit reference;
- every attempt starts from the original draft;
- failures are recorded per attempt and never silently replaced.

Materialize three offline arms:

1. **draft:** original FLUX draft;
2. **no verifier:** attempt 1 when available, otherwise draft;
3. **frozen hybrid:** draft routing and repair selection from section 4.

This makes the verifier comparison deterministic and avoids paying for
different random generations in different arms.

## 6. Metrics and frozen promotion gates

### 6.1 Draft detector

Against locked human identity labels, report TP, FP, TN, FN, recall,
specificity, balanced accuracy, MCC, and Wilson intervals for:

- existing semantic VLM;
- reranker threshold alone;
- frozen union.

The union passes when:

- failure recall is at least `0.733`;
- false positives are at most `1`.

If either PASS or FAIL has zero labeled cases, detector validation is
inconclusive, not passing.

### 6.2 Repair selector

For every scored draft-attempt pair, compare the reranker's predicted
direction with held-out cropped DINO direction. Report sign accuracy, exact
DINO-best selection rate, mean selected DINO delta, and number of harmful
selections.

The selector passes when:

- sign accuracy is at least `0.750`;
- at most one selected result has DINO delta below `-0.02`.

### 6.3 End-to-end image outcome

For each arm and cohort, report cropped held-out DINO, whole-image CLIP,
whole-image SigLIP, preservation outside the mask, per-case delta, bootstrap
95% confidence interval, and improved/unchanged/worsened counts.

Promotion additionally requires:

- frozen-hybrid mean cropped DINO delta above zero on rare targets;
- no more than one hybrid output below `-0.02` DINO across all cases;
- common-control DINO delta confidence-interval lower bound at least `-0.02`;
- visual inspection finds no reference-photo paste-through or material
  background/layout replacement.

All gates are conjunctive. A failure remains a reported result; thresholds are
not retuned on this holdout.

## 7. Integration and 92-case experiment

Do not alter production scheduling until the holdout passes. First implement
the policy as a pure offline evaluator and test it against synthetic score
records. If the holdout passes:

1. integrate the same frozen policy into the scheduler;
2. run focused tests and the full non-GPU suite;
3. rerun the 92-case dataset with identical seeds and the same 57k-image
   LAION retrieval database using:
   - draft baseline;
   - full retrieval plus inpainting without verifier;
   - full retrieval plus frozen hybrid verifier;
4. report bridge-target, bridge-control, and common cohorts separately;
5. expose retrieval failures separately from verifier and inpainting failures.

The 92-case run is not launched when the holdout fails. In that event, retain
the successful no-verifier inpainting result and report the verifier as the
remaining unsolved component.

## 8. Artifacts, timestamps, and reproducibility

- New outputs use a runtime Asia/Ho_Chi_Minh timestamp beginning with
  `20260818` when run today; no copied date is embedded in scripts.
- The validation output contains the dataset snapshot, labels, raw VLM
  responses, raw reranker scores, routing decisions, selection decisions,
  per-case metrics, contact sheets, JSON report, and Markdown report.
- The report records source run ids, git status/commit, model ids, GPU, package
  versions, seeds, and every frozen constant.
- Resume reuses completed artifacts and never regenerates successful cases.
- Existing dirty-worktree changes are preserved; implementation is additive
  or narrowly scoped to the new policy interfaces.

## 9. Alternatives rejected for this stage

- **Immediately rerun the 92 cases:** expensive and does not provide an
  independent verifier test because the controlled cases influenced policy.
- **Tune again on the existing 22 cases:** would improve development numbers
  without supporting generalization.
- **Curate a new 20–30 concept public benchmark first:** scientifically
  stronger, but slower than using the already-local unused concepts for this
  promotion smoke test. It remains the next step after a successful smoke.
