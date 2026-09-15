# Weekly Results Report — August 12–20, 2026

## Executive result

- We made the pipeline auditable and safer: verifier evidence is now persisted, failed attempts can no longer silently replace drafts without evidence, runs can be re-verified without regenerating images, and GPU/model lifecycle bugs were fixed.
- The inpainting mechanism works when correctly triggered. In the 92-case open-loop baseline, the target/bridge cohort improved by **+0.066 cropped DINO** with a 95% CI of **[+0.008, +0.132]**, while outside-mask preservation stayed at **1.000**.
- The verifier is still not production-ready. Three selector variants failed, and the retrieved-reference detector failed decisively on a fresh 48-case holdout: **50.0% recall** and **4/24 false positives**.
- Current decision: keep **`--verifier none`** as the validated baseline. Do not integrate the hybrid/retrieved detector and do not launch a verifier-driven 92-case run until a genuinely different identity signal passes an independent holdout.

## What we completed

### August 12–13 — Pipeline correctness and persistence

We fixed the loss of verifier evidence between scoring and scheduling. The earlier pipeline reduced 202 grounded verdicts to binary `1.0/0.0` values; this caused 3 of 19 reported repairs to select a failing attempt over a failing draft.

Delivered results:

- Added one canonical verifier serializer that preserves states, similarity scores, prototype scores, candidates, boxes, and confidence values.
- Represented missing evidence as `null` instead of a fabricated perfect score; scoreless failed attempts no longer displace the draft.
- Persisted raw VLM replies, issues, grounded scores, retrieval hits, mask-selection method, and candidate counts.
- Added re-verification of completed runs without running FLUX or changing the source run.
- Made prototype scoring and prototype-guided masking reachable through separate, default-off flags.
- Split `healthy` from `unrepaired`, preserved failed-case status during finalization, prevented resolved cases from being rewritten on resume, and made old run artifacts backward-compatible.
- Added missing re-verification guards and prototype-arm VRAM preflight checks.

Result: future runs retain enough evidence for threshold analysis, failure diagnosis, and reproducible re-selection. See the [persistence design](specs/2026-08-12-verifier-persistence-design.md) and [implementation plan](plans/2026-08-12-verifier-persistence.md).

### August 18–19 — Hybrid verifier validation

We built a frozen holdout workflow, reusable candidate bank, deterministic policy evaluation, provenance manifests, and gated reporting. The experiments then isolated the verifier's actual failure point.

| Experiment | Main result | Decision |
|---|---|---|
| Hybrid v1 | Detector produced **5/12 control false positives**; selector sign accuracy was **0.6226** versus the **0.750** gate. | FAIL |
| Hybrid v2 | Routing improved to **1.000 recall and 0 false positives**, but selector sign accuracy was only **0.6038**. Rare mean DINO delta remained positive at **+0.229**. | FAIL |
| Hybrid v3 | Candidate semantic confirmation reduced selector sign accuracy further to **0.4906**; its VLM decision had **40.8% balanced accuracy** and **MCC −0.20** against DINO direction. | FAIL; stop selector series |
| Retrieved-reference Phase A | **11/12 recall**, **1/12 false positives**, MCC **+0.833**. It passed every preregistered gate, but exactly at the FP boundary and with a wide recall interval. | Thin PASS; authorize fresh holdout only |

Problems resolved:

- The inpainting mechanism was separated from the routing/selection policy: repair quality was positive when the correct cases were edited, while the verifier policy was unsafe.
- Guarded routing removed v1's five development false positives without losing development recall.
- We established that repeated threshold searches and plain candidate-semantic confirmation do not solve best-of-N selection.
- We verified that the Phase A detector gain was not purely caused by using curated oracle references, although the retrieved-reference result was fragile.

Evidence: [v2 finding](2026-08-19-hybrid-verifier-v2-finding.md), [v3 finding](2026-08-19-hybrid-verifier-v3-finding.md), and [retrieved-reference Phase A finding](2026-08-19-retrieved-detector-finding.md).

### August 20 — Independent holdout and baseline results

#### Retrieved-reference detector: decisive failure

The frozen Phase A policy was applied without refitting to a new holdout containing 24 rare failures and 24 correct controls.

| Metric | Result | Required gate |
|---|---:|---:|
| Failure recall | **12/24 = 0.500** | ≥ 18/24 = 0.750 |
| False positives | **4/24 = 0.167** | ≤ 2/24 = 0.083 |
| Pooled MCC | **0.354** | descriptive |
| Coverage/contamination | **48/48, 0 contaminated** | complete and clean |

The experiment stopped before generation as preregistered. All 12 true positives came from the semantic branch; the text-plus-retrieved-reference branch caught **zero additional failures**. Three of four false positives also came from the general semantic branch. See the [Phase B finding](2026-08-20-phase-b-detector-finding.md).

#### Retrieved prototype verifier: preregistered rule not met

On the 22 labelled C1 cases, the best prototype threshold caught all six previously missed identity failures but introduced **three new false positives**. No valid band of three consecutive thresholds satisfied the preregistered rule.

Result: retrieved image-side prototype contrast is insufficient; further tuning of the same score is not justified.

#### No-verifier 92-case baseline

The open-loop inpainting baseline completed 89 of 92 cases; three cases were unmaskable. Every maskable case received exactly one deterministic edit.

| Cohort | Mean cropped-DINO delta | 95% CI | Result |
|---|---:|---:|---|
| Target/bridge, n=11 | **+0.066** | **[+0.008, +0.132]** | Positive improvement |
| Control/bridge, n=11 | **+0.026** | **[−0.009, +0.070]** | No harm |
| Control/common, n=67 | **+0.000** | **[−0.021, +0.022]** | Inconclusive; lower bound misses the −0.02 margin by 0.001 |

Outside-mask preservation was **1.000** in every reported cohort. This confirms that the repair mechanism can improve rare targets, but it does not provide safe case selection.

Two operational bugs found during the holdout were also fixed: missing model cleanup that leaked VRAM into the next stage, and a reranker model-ID mismatch during policy-hash verification. The non-GPU suite reached **726 passing tests** at the Phase B checkpoint.

## New problem

The remaining blocker is **generalizable fine-grained identity detection**.

- The semantic verifier checks broad prompt adherence, so it misses half of the true identity failures and incorrectly routes some valid controls.
- Text relevance is high for most missed identity failures, so combining it with retrieved-reference relevance cannot trigger on them.
- Retrieved-reference relevance added no independent true positives on the fresh holdout.
- Candidate reference margins, guarded margins, semantic confirmation, phrase-level contrast, and prototype contrast have all failed their promotion criteria.

The 92-case baseline shows the consequence: editing can improve rare targets, but without a reliable detector the system must either edit all cases and risk unnecessary changes or miss many cases that need repair.

## Focus going forward

1. Stop tuning the current semantic/text/reference threshold family; the independent holdout shows the problem is the signal, not the cutoff.
2. Keep `--verifier none` as the baseline and do not promote the hybrid or retrieved detector.
3. If verifier work continues, introduce a genuinely identity-specific signal and validate it on a balanced, human-labelled development set before freezing another independent holdout.
4. Treat best-of-N selection as a separate problem; it needs a balanced candidate-quality challenge set rather than DINO direction alone as ground truth.

