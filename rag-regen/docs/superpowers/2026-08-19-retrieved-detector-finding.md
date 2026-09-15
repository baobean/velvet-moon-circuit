# Retrieved-reference detector — Phase A finding

**Date:** 2026-08-19
**Decision:** readiness **PASS** (thin). The detector survives the reference-
distribution shift and clears every pre-registered gate, so the win is **not**
purely oracle-dependent. This authorises writing a fresh 24+24 Phase B holdout
design — **not** production integration and **not** the 92-case generation run.

## What was tested

The v2/v3 detector's `reference_relevance` was scored against `first_edit_gt_ref`
(a curated oracle reference), while production retrieves from LAION. This study
re-scored only the draft's `reference_relevance` against the **retrieved** top-1
LAION reference, preserved `text_relevance` bit-for-bit (it is text-only, so
reference-independent), and repeated leave-one-concept-out detector fitting
(re-fitting the threshold, never reusing the oracle's). No image generation, no
attempts, no DINO, no selector, no label edits.

## Comparability and integrity (all verified)

- **Oracle parity: max abs diff 0.000000.** Re-scoring the oracle references in
  the same residency reproduced `qwen_reranker.json` bit-for-bit, so retrieved and
  oracle scores are directly comparable. `text_relevance` copied unchanged.
- **Contamination: 0/24.** No retrieved image is byte-identical to any ground-
  truth reference (exact-SHA-256 check); coverage complete.
- **Environment:** disposable venv (kontext + `--system-site-packages`), only
  `sentence-transformers==5.4.0`, `scikit-learn==1.6.1`, `joblib==1.5.3`,
  `threadpoolctl==3.6.0` installed `--no-deps` (wheel hashes in
  `venv_provenance.json`); Torch/Transformers/NumPy/SciPy inherited from kontext,
  unchanged; ABI smoke test passed before any model load.
- retrieved artifact `sha256 68d0b03f…`; frozen policy `sha256 d28234…`.

## Result

| policy | recall | fp | fp_rate | mcc |
|---|--:|--:|--:|--:|
| **retrieved_guarded** | **0.917** (11/12) | **1** | **0.083** | +0.833 |
| oracle_v2 | 1.000 | 0 | 0.000 | +1.000 |
| semantic_only | 0.667 | 0 | 0.000 | +0.707 |
| retrieved_reference_only | 0.917 | 3 | 0.250 | +0.676 |

All four gates pass: coverage (24/24, 0 contaminated), all 24 folds feasible,
recall 0.917 ≥ 0.733, false positives 1 ≤ 1 (and fp_rate 0.0833 ≤ 0.0834). Frozen
policy (all 24 rows): `text < 0.5 AND reference < 0.25049` (the reference
threshold rose from the oracle's 0.217 because retrieved references score lower).

## Honest reading — a *thin* pass

The detector degrades meaningfully from the oracle: **−1 recall** and **+1 false
positive**, MCC 1.000 → 0.833. The two flips are concrete:

- **Missed FAIL: `stave_church`** (rare) — retrieved reference scored high enough
  that the conjunction did not fire, and semantic also missed it.
- **False positive: `labrador_retriever`** (control) — its retrieved reference
  scored low enough to route a correct draft. (Notably the same case v1's selector
  wrongly rewrote.)

At n=12 per stratum this lands exactly on the FP bar (1/12 = 0.0833 vs 0.0834) and
the recall Wilson 95% CI is wide, [0.646, 0.985] — its lower bound is *below* the
0.733 bar. The `retrieved_reference_only` baseline (3 FP, fp_rate 0.25) shows the
retrieved signal alone is not trustworthy; it is the **conjunction with semantic
failure** that holds FP to 1. This is a real but fragile pass, not a clean one.

## Recommendation

Proceed to a **fresh 24+24 Phase B holdout design** (the only thing this PASS
authorises). That design must, at minimum:

- freeze identity + eligibility manifests and the frozen detector policy
  (`text<0.5 AND retrieved_reference<0.25049`, plus semantic failure) *before*
  scoring;
- retrieve references from the same corpus for unseen concepts and controls;
- predeclare detector recall ≥ 0.733 and **≤ 2 false positives among 24 controls**
  (double the headroom this thin n=12 pass had);
- keep the deterministic single-attempt (attempt_1) repair, no selector;
- hold production integration and the 92-case run until every gate passes on that
  holdout.

Full non-GPU suite green (700 passed). Builds on the oracle detector win in
`docs/superpowers/2026-08-19-hybrid-verifier-v2-finding.md`; supersedes the
oracle-dependence caveat in `docs/superpowers/2026-08-19-hybrid-verifier-v3-finding.md`
(the caveat was real; the detector cleared it here, thinly).
