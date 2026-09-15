# Hybrid verifier v2 — Phase A development finding

**Date:** 2026-08-19
**Decision:** readiness **FAIL**. Stop. Do not freeze a policy, do not write a
fresh-holdout (Phase B) plan, do not integrate `--verifier hybrid-v2`, do not run
the 92-case experiment.

## What was run (CPU-only)

Leave-one-concept-out development study over the exposed 24-case run, using the
frozen identity/eligibility manifest. No model inference; DINO was used only to
fit and grade the development policy, never as a live input.

- manifest: `outputs/hybrid_v2_development/manifest.json`
  (`sha256 387b89b0edbd380c808453e4e66731469590e33b98b593bbb1371df5698bd20d`)
  — 24 detector rows, 22 selector-eligible, 2 `mask_not_grounded`
  (`azawakh`, `bergamasco_shepherd`), `protocol_status=reconstructed_after_score`.
- report: `outputs/hybrid_v2_development/report/development.{json,md}`
  (json `sha256 1119b9915a12e52e51bd635afcd925763e7c147f9425a89d594eec343bead93d`).

The protocol correction held: the two invalid-mask cases stay in the detector
population as identity `FAIL` while being selector-ineligible. Identity truth was
read from the reranker artifact's frozen `truth` field; `labels.csv` was not
edited.

## Result

| gate | value | bar | verdict |
|---|---|---|---|
| detector failure recall | 1.000 (Wilson95 [0.757, 1.000]) | ≥ 0.733 | PASS |
| detector false positives | 0 / fp_rate 0.000 | ≤ 1 or ≤ 0.0834 | PASS |
| selector sign accuracy | **0.6038** | ≥ 0.750 | **FAIL** |
| selector harmful (< −0.02) | 0 | ≤ 1 | PASS |
| rare (bridge) mean DINO delta | +0.229 (95% CI [+0.087, +0.387]) | > 0 | PASS |

Baselines over identical detector rows: semantic-only recall 0.667 / 0 FP;
reference-only and v1-union recall 1.000 / **5 FP** (fp_rate 0.417).

## Interpretation

**The routing change worked.** Replacing v1's permissive union with the guarded
conjunction `semantic_failure OR (text < t AND reference < r)` drove control
false positives from 5 to **0** while keeping full failure recall, cross-validated.
The detector is no longer the blocker.

**The selector is the blocker, and it is the same blocker as v1.** Cross-validated
sign accuracy is 0.6038 — no better than v1's 0.6226, and still far below 0.750.
The rare cohort's mean DINO delta is genuinely positive (+0.229) and no selection
is harmful, so the inpainting mechanism remains useful; but the two-margin
reranker selector cannot reliably tell an identity-improving attempt from a
worsening one. The exact-DINO-best rate is only 0.500 — coin-flip agreement with
the oracle.

This is exactly the failure mode spec §6 anticipated: widening threshold searches
over the two existing reranker signals (text relevance, reference relevance) will
not clear the selector gate.

## Recommendation for the next design

Per spec §6, the next design must add a **genuinely new signal** to the selector,
not another threshold sweep. The pre-registered candidate is **candidate semantic
confirmation** — re-running the semantic verifier on each repair attempt and
requiring it to affirm the concept before the attempt is eligible — so selection
is gated on an independent identity judgement rather than on reranker relevance
margins alone. Only after such a signal clears leave-one-concept-out readiness
should a fresh-holdout (Phase B) plan be written.

## Boundaries honoured

CPU-only; no FLUX/Qwen/DINO/CLIP/SigLIP reruns; v1 artifacts and `labels.csv`
unchanged; no production adapter; no frozen policy written (readiness ≠ PASS); no
Phase B holdout plan authored (readiness ≠ PASS); no 92-case run.
