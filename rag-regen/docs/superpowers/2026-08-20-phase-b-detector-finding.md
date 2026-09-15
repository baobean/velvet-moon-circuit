# Retrieved-reference detector — Phase B holdout finding

**Date:** 2026-08-20
**Decision:** readiness **FAIL**. The frozen detector-only policy does not
clear either predeclared gate on a fresh, independent, never-before-seen
24+24 holdout. **Stopped before generation** per protocol (design section 5)
— no `phase-b-generate`, no visual review, no end-to-end evaluation.

## What was tested

The exact policy Phase A froze thinly-passed
(`docs/superpowers/2026-08-19-retrieved-detector-finding.md`), applied
**verbatim, no fitting**, to a brand-new 24 rare-FAIL + 24 control-PASS
holdout drawn entirely from ImageNet classes never touched by any v1/v2/v3,
Phase A, or hybrid-verifier work (72 rare + 41 control candidates screened,
hand-labelled, mechanically checked for eligibility/contamination, then
frozen per `docs/superpowers/specs/2026-08-19-retrieved-detector-phase-b-design.md`
section 3).

```
route = semantic_failure OR (draft_text_relevance < 0.5
        AND draft_retrieved_reference_relevance < 0.25048828125)
```

## Comparability and integrity (all verified)

- **Oracle parity: max abs diff 0.000000.** Retrieved and oracle reranker
  scores are directly comparable.
- **Contamination: 0/48.** No retrieved reference is byte-identical to any
  ground-truth reference.
- **Coverage: complete, no degenerate scores.** All 48 rows scored cleanly
  (`uncontaminated_coverage` gate: PASS).
- **Environment:** disposable venv, `sentence-transformers==5.4.0`,
  `scikit-learn==1.6.1`, `joblib==1.5.3`, `threadpoolctl==3.6.0`, same
  pinned versions as Phase A.
- Holdout manifest `sha256 0a113383b269…`; frozen policy `sha256
  a398d8dc49b8…`.

## Result

| metric | value | gate | passed |
|---|--:|---|:--:|
| recall | **12/24 = 0.500** | ≥ 18/24 (0.75) | **NO** |
| false positives | **4/24 = 0.167** | ≤ 2/24 (0.083) | **NO** |
| pooled MCC | 0.354 | — | — |
| pooled accuracy | 0.667 | — | — |

Wilson 95% CIs (descriptive only, per protocol — not a gate):
recall [0.314, 0.686] — the **entire interval sits below the 0.75 floor**,
not just the point estimate; FP rate [0.067, 0.359] — point estimate is
double the 0.083 ceiling.

## Why it failed — the conjunction adds nothing on this holdout

Breaking down routing by which branch fired:

- **All 12 true positives (100%) were caught by `semantic_failure` alone.**
  The `text < 0.5 AND retrieved_reference < 0.25049` conjunction never
  independently caught a single genuine rare-FAIL case the semantic branch
  hadn't already caught.
- **The 12 missed rare-FAIL cases all have `semantic_ok = True`** (the VLM
  judge said the draft matches its prompt) **and `text_relevance` well
  above 0.5** (range 0.484–0.742) — so the conjunction structurally cannot
  fire for them regardless of how low `retrieved_reference_relevance` gets.
  `text_relevance` is not discriminating fine-grained identity failure on
  this holdout at all.
- **3 of 4 false positives came from `semantic_failure`** firing on
  hand-labelled identity-PASS controls (`broccoli`, `leopard`, `seashore`)
  — the judge template checks general prompt/premise adherence, not
  identity specifically, so this is a different failure mode leaking into
  the identity gate.
- Only 1 false positive (`water_bottle`) came from the conjunction itself.

Net: on this independent holdout, the detector's behaviour reduces almost
entirely to the semantic verifier alone, which tops out at 50% recall here
with a 3/24 false-positive cost riding along — the retrieved-reference
signal is not adding the discriminating power Phase A's thin pass hoped it
would.

## Honest reading

Phase A's own finding called its pass "real but fragile" and flagged that
the recall Wilson lower bound (0.646) sat below the gate at n=12. This
Phase B holdout — 2× the sample, fully independent, frozen before any score
was visible — is exactly the confirmation that fragility warned about: it
does not replicate. This is not a marginal/thin miss; both gates fail by a
wide margin, and the whole recall CI sits under the floor.

## Recommendation

**Do not build on the retrieved-reference detector.** This is now three
converging signals against it: the original hybrid verifier holdout FAILed
(`docs/superpowers/2026-08-18-hybrid-verifier-validation`-era findings),
Phase A's own pass was already flagged fragile, and this independent Phase
B holdout fails decisively. Per the design's scope boundaries (section 10):
no integration, no 92-case run. Recommend reverting to `--verifier none`
(the existing, validated baseline) and not re-attempting a fourth holdout
on this same policy shape without a genuinely different signal — Phase A's
own note on the v2 selector applies here too: this needs a new signal, not
more threshold tuning.

Full non-GPU suite green (726 passed) at the time of this run. Fixed two
real bugs discovered while executing this holdout (not part of the
detector's own result): a missing `Retriever`/`HFEncoder.free()` causing a
VRAM leak into the next pipeline stage, and a reranker model-id
slug-vs-repo-id mismatch in the detector CLI's own hash verification.
