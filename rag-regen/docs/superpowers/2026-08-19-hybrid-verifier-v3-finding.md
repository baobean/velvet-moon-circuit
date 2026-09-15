# Hybrid verifier v3 — candidate-semantic finding

**Date:** 2026-08-19
**Decision:** readiness **FAIL**. The pre-specified new signal does **not** work.
Do not freeze a policy, do not write a Phase B holdout plan, do not integrate.

## What was tried

v3 replaced v2's failed reference-relevance margin with the *pre-specified* new
signal: run the plain semantic verifier (Qwen2.5-VL) on each existing repair
attempt and require it to affirm the concept before the attempt is
selector-eligible (Option A). ("Pre-specified", not "pre-registered": candidate
semantic confirmation was named in the v2 finding before any v3 inference, and
timestamps support that ordering, but the exact v3 policy lived only in untracked
source, not a frozen design document.) Detector unchanged from v2. One GPU pass scored the
53 existing attempt images; everything else CPU. No image generation, no
DINO/reranker reruns, `labels.csv` untouched.

- verdicts: `outputs/hybrid_v2_development/candidate_semantic.json`
  (`sha256 ee7ddc15…`), 53 attempts, 0 degenerate.
- report: `outputs/hybrid_v2_development/report_v3/development.{json,md}`
  (json `sha256 3b84446a…`).

## Result

| metric | v1 | v2 | **v3** | bar |
|---|--:|--:|--:|--:|
| selector sign accuracy | 0.6226 | 0.6038 | **0.4906** | ≥ 0.750 |
| harmful selections | — | 0 | 1 | ≤ 1 |
| rare mean DINO delta | — | +0.229 | +0.223 | > 0 |
| detector fp (v2 routing) | 5 | 0 | 0 | ≤ 1 |

The semantic gate made the selector **worse**, not better: sign accuracy fell to
0.49, below both v2 (0.60) and the trivial always-"better" base rate (0.736; see
next section).

## Why it failed — the signal does not track selection quality

The VLM affirmed the concept on **42/53** attempts. Cross-tabulated against the
sign of the DINO identity delta:

| | DINO better | DINO worse |
|---|--:|--:|
| **VLM = ok** | 29 | **13** |
| **VLM = no** | **10** | 1 |

The right yardstick is the base rate, not chance: **39/53** attempts improve DINO,
so a trivial always-"better" rule scores **73.6%**. The VLM's "concept present"
verdict agrees with the DINO-delta sign only **56.6%** of the time — *below* that
baseline — with **balanced accuracy 40.8%** and **MCC −0.20** (mildly
anti-correlated). As a binary eligibility gate it carries no usable discrimination
about which attempt improves identity.

**Careful reading of the mechanism.** Do not over-claim fine-identity blindness
from this table alone: **12 of the 13** VLM-ok/DINO-worse cases are *common
controls*, where the named concept may genuinely still be present despite a small
DINO decrease (a preservation wobble, not a concept error); among rare attempts
DINO says 27/29 improve, leaving only **two** rare negatives — too few to isolate
the mechanism. The supported conclusion is narrower and still decisive: **plain
semantic presence is badly mismatched with the DINO-delta sign and lacks useful
candidate discrimination on this dataset.** Fine-identity leniency may contribute
but is not established here.

## Recommendation

Candidate semantic confirmation, as a plain yes/no, is **eliminated** as the
selector signal. Three independent selector signals — v1 reference margin, v2
guarded margins, v3 semantic gate — have now failed the same 0.75 bar. **Stop the
selector series.**

The evidence-backed next move is a **detector-only, single-attempt policy** tested
on a genuinely fresh holdout. The frozen behaviour would be:

```
route = semantic_failure OR (draft_text < 0.5 AND draft_reference < 0.21728515625)
if routed and mask/reference preparation succeeds:  emit deterministic attempt_1
else:                                                retain draft
```

No candidate scoring, best-of-N, VLM confirmation, or learned selector — so
**selector sign accuracy is removed entirely**, because there is no selector. The
exposed development evidence is encouraging: detector LOCO finds 12/12 failures
with 0/12 false positives, and deterministic attempt_1 on the ten repairable rare
cases gives mean DINO delta **+0.207** (bootstrap 95% CI [+0.070, +0.370]), 9/10
improved, 0 harmful below −0.02.

Note this is a *new* production behaviour, not today's `--verifier none`: the
current one-shot arm (`--verifier none` ⇒ `open_loop_attempts == 1`, pinned by
`tests/test_run_pipeline.py`) routes every screened draft and emits attempt_1 with
**no detector gate**. The proposed policy adds the v2 detector as the gate.

**Material caveat — the detector win may be oracle-dependent.** Every
`reference_relevance` score above was computed with `reference_source =
first_edit_gt_ref` (a curated ground-truth reference), while production retrieves
its reference from the LAION corpus. The routing conjunction is what lifts detector
recall from semantic-only's 0.667 to 1.000 (four extra catches) and holds false
positives at zero — and that conjunction depends entirely on the oracle
`reference_relevance`; the `0.21728515625` threshold has **not** been validated on
the retrieved-reference distribution. So the sequence below inserts a
retrieved-reference development check *before* any generation-GPU Phase B:

1. **Retrieved-reference Phase A (no image generation).** Reuse the exposed 24
   drafts and the existing LAION index (`data/laion100k/index.faiss`, present):
   retrieve deterministically, re-score each draft's `reference_relevance` against
   its top prepared retrieved reference, and repeat leave-one-concept-out detector
   fitting (re-fitting the threshold, not reusing 0.217). If recall/FP gates fail,
   **stop — the detector win was oracle-dependent.** If they pass, freeze that
   retrieved-reference detector and proceed.
2. Only then design the fresh 24+24 Phase B holdout and spend generation GPU.

An oracle-only alternative — go straight to Phase B with curated references —
would be scientifically valid but must limit its claim to "detector-only repair
with curated references" and cannot authorise production integration or the
retrieved-reference 92-case run.

Do **not** pursue a reference-aware VLM judge instead: that arm already showed only
5/15 = **0.333** failure recall on the earlier 22-case study
(`outputs/screen_20260726_233320/verifier_ref_oracle_verdict_identity.md`). If
best-of-N selection later becomes a hard product requirement, first build a
balanced, human-labelled candidate challenge set with genuine fine-identity hard
negatives — do not use the DINO-delta sign alone as the judge's ground truth.

Recommended sequence: (1) correct this v3 archival report [done], (2) run the
retrieved-reference Phase A check above (no generation), (3) only if it passes,
write the separate detector-only one-shot fresh-holdout design + plan (≥24 rare +
24 controls, gates pre-registered before scoring) and run one clean holdout. Do not
integrate or launch the 92-case run until that holdout passes. If the GPU budget
is not worth it, stop here and report the selector as a negative result.

No frozen policy, no Phase B plan, no production change was produced (readiness ≠
PASS). Full non-GPU suite green (679 passed). See
`docs/superpowers/2026-08-19-hybrid-verifier-v2-finding.md` for the v2 detector
win this builds on.
