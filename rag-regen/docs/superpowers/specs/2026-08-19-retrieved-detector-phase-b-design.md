# Retrieved-Reference Detector — Phase B Holdout Design

**Date:** 2026-08-19
**Status:** Design for review. No implementation plan, code, data, or GPU runs
until approved.
**Predecessors:** `docs/superpowers/2026-08-19-retrieved-detector-finding.md`
(Phase A thin PASS), `docs/superpowers/specs/2026-08-19-hybrid-verifier-v2-design.md`.

## 1. Objective

Confirm, on an **independent, untouched** holdout, the deployable detector-only
repair policy whose development-set behaviour Phase A validated (thinly). Phase B
is a frozen-before-score holdout: the policy is applied exactly as frozen, with no
fitting of any kind. A PASS authorises **integration testing only** — not an
unqualified production claim and not the 92-case generation run.

## 2. The frozen policy under test

The complete routing rule, frozen in full:

```text
route = semantic_failure
        OR (draft_text_relevance < 0.5
            AND draft_retrieved_reference_relevance < 0.25048828125)

if not routed:                       retain the draft
elif attempt_1 preparation+generation succeeds:  emit deterministic attempt_1
else:                                retain the draft as an explicitly flagged
                                     operational fallback (generation_failed=true)
```

No retries, no best-of-N, no selector. Every arm always yields exactly one output
per case: `attempt_1` on success, or the draft on failure — but a failure-fallback
draft carries `generation_failed=true` and is never counted as an ordinary
preserved draft.

**Policy manifest (must hash all of the following).** Phase A's
`frozen_detector_policy.json` records only the two thresholds; that is
insufficient. Phase B freezes a policy manifest with a single SHA-256 over the
serialized whole, capturing:

- the complete rule text above, including the `semantic_failure` branch and both
  thresholds (`text 0.5`, `retrieved_reference 0.25048828125`);
- **semantic verifier**: model id **and the SHA-256 of the exact weight/snapshot
  files loaded**, judge template/prompt, decoding constants;
- **reranker**: model id (`qwen3-vl-reranker-2b`) **and the SHA-256 of the exact
  weight/snapshot files**, prompt, `max_image_side`;
- **retrieval config**: encoder (`siglip_so400m_384`) **and its weight-file
  SHA-256**, index path + index SHA-256, corpus name, query builder
  (`ragregen.retrieve.reference_query`), `k`, and the reference-prep transform
  (crop-to-mask, thumbnail to 448);
- **full environment manifest**: the resolved version of *every* inherited
  package (not just the pinned four) — e.g. `pip freeze` of the executing
  interpreter — plus the wheel hashes of the venv-installed set
  (sentence-transformers 5.4.0, scikit-learn 1.6.1, joblib 1.5.3,
  threadpoolctl 3.6.0);
- **clean code snapshot**: a content hash (tree/tarball SHA-256) over the *exact
  source files executed*, not merely `git HEAD`. The worktree is dirty and
  untracked here, so the current commit does not identify the executed code; the
  snapshot must be taken from the actual files run and hashed;
- **degenerate-input behaviour, frozen**: define deterministically before scoring —
  a semantic parse-failure / degenerate reply fails **closed** (counts as
  `semantic_failure`, routes), matching the shipped verifier; a missing or
  non-finite `text_relevance` or `retrieved_reference_relevance`, or an
  abstention with no usable score, is a **coverage failure** (the case cannot be
  scored) and makes the study INCONCLUSIVE — it is never silently defaulted;
- `protocol_status: frozen_before_score`.

## 3. Holdout construction (frozen before any scoring)

- **Final frozen cohort: exactly 24 judgeable rare FAIL cases and exactly 24
  judgeable control PASS cases** — all entirely unseen (concepts, drafts,
  references, reserves not inspected during any v1/v2/v3 or Phase A work). The two
  denominators are fixed at 24 and 24; a case is never simply "dropped" in a way
  that reduces either.
- Two **distinct** reserves, both frozen up front and never conflated:
  - a **held-out DINO reference reserve** per case — references reserved solely for
    cropped-DINO identity scoring, never used for editing or routing;
  - a **replacement-case reserve** — additional pre-labelled, pre-retrieved unseen
    cases per cohort, drawn from *only* to backfill a case that is ineligible or
    contaminated, so the final cohort still contains exactly 24 judgeable FAIL and
    24 judgeable PASS.
- Freeze, before a single verifier or DINO score is computed:
  - identity **labels** (`PASS`/`FAIL`/`UNJUDGEABLE`) hand-assigned from the draft
    alone; only judgeable `FAIL` (rare) and `PASS` (control) cases enter the two
    denominators;
  - **eligibility** (valid mask + prepared reference) and its mechanical reason;
  - **references**: the deterministically retrieved top-1 LAION reference per case;
  - **contamination handling**: exact-SHA-256 check of every retrieved reference
    against that case's ground-truth references. A contaminated or ineligible case
    is **backfilled from the replacement reserve at freeze time** to preserve the
    24+24 denominators; the substitution is recorded. If the replacement reserve
    for a cohort is exhausted before reaching 24 judgeable cases, the study is
    **INCONCLUSIVE** (it never proceeds with a reduced denominator). Contamination
    or ineligibility discovered *after* scores are visible cannot be repaired and
    also makes the study INCONCLUSIVE.
- All of the above are written to immutable manifests with source hashes and
  `protocol_status: frozen_before_score`; writers refuse to overwrite.

## 4. Protocol invariants

After any verifier score is visible, there is **no** leave-one-out, **no**
threshold fitting, **no** exclusions, and **no** replacements. The frozen policy
(§2) is applied as-is to the frozen holdout (§3). Any deviation voids the study.

## 5. Detector-first gate (scored before generation)

Run detector scoring first — semantic verdict on each draft, plus the retrieved
`reference_relevance` (text_relevance is reference-independent). Apply the frozen
rule. Require **all** of:

- **complete, uncontaminated coverage** of all 48 judgeable cases;
- **failure recall ≥ 18/24** (0.75) on the 24 rare targets;
- **false positives ≤ 2/24** on the 24 controls.

**Stop before generation if any fails.**

**Statistical framing (stated accurately).** `2/24 = 8.33%` is the *same*
false-positive-rate ceiling as Phase A's `1/12`; the larger control stratum does
**not** loosen the cap and does **not** "double the headroom" — its only benefit
is a tighter estimate of that same rate. Wilson intervals and confusion matrices
are reported **descriptively**. No confidence-bound gate is predeclared here; if
one is wanted, it must be predeclared explicitly and would replace, not
supplement, the point-estimate gate above.

## 6. Conditional generation and three offline arms

Only on a detector PASS: generate **exactly one deterministic `attempt_1`** from
each original draft (rare and control alike). The generation stack is **frozen and
hashed** into the policy manifest before this step:

- **generator revision** (the Kontext/FLUX model id + weight-file SHA-256);
- **edit prompt** template;
- **seed mapping** (the exact per-case seed derivation);
- **scheduler parameters** (steps, guidance, scheduler id);
- **inference precision** (dtype);
- **mask/reference preparation** (mask source + `mask_dilate_px`, reference
  `ref_prep`/crop, resolution);
- **compositor** (the exact stitch that composes the edited region back onto the
  draft).

From these same images, materialize three offline arms:

1. **draft** — the original draft, unrepaired (baseline);
2. **route-all attempt_1** — every draft replaced by its `attempt_1` (the current
   `--verifier none` one-shot behaviour; upper bound on repair activity);
3. **frozen-detector output** — `attempt_1` where the frozen rule routed, draft
   otherwise (the proposed system).

No retries, no selector, no second attempt.

**Attempt / preparation failure handling (single frozen rule).** Every arm always
retains exactly one output per case: `attempt_1` on success, and the draft as an
**explicitly flagged operational fallback** (`generation_failed=true`) on failure.
That flag is recorded and reported; a fallback draft is never presented as an
ordinary preserved draft, so a suppressed repair cannot masquerade as a preserved
one, and no arm's denominator ever shrinks. Failures are never retried or replaced
(backfill happens only at freeze time, §3, never after scores are visible).

This flagged-fallback rule is the deployable behaviour. For the **Phase B study
specifically**, however, all three arms and the fixed 24+24 denominators are
required, so **any preparation or generation failure after freezing makes Phase B
INCONCLUSIVE** — the study stops and is reported rather than computing end-to-end
metrics over a fallback-contaminated set. This keeps the deployable fallback
semantics while preventing a failure from improving the metrics through a
zero-delta draft row.

## 7. End-to-end gates (preserved from prior specs)

**Reporting contract (every arm × every cohort).** For arms 1–3, split by rare and
control cohort, report: cropped held-out DINO, whole-image CLIP, whole-image
SigLIP, outside-mask preservation, the paired bootstrap CI versus draft, and the
improved / unchanged / worsened counts. Exact denominators and every
generation/prep failure (§6) are shown.

**Gates (conjunctive), evaluated on arm 3 (frozen-detector output) versus arm 1
(draft):**

- **rare identity**: rare-cohort **mean cropped-DINO delta > 0**;
- **harmful**: at most **one** arm-3 output below −0.02 cropped-DINO delta,
  counted **across all 48 cases** (not rare only);
- **control non-inferiority**: the **paired bootstrap CI lower bound of arm-3
  minus draft cropped-DINO, over all 24 controls** (not routed controls only),
  must be **≥ −0.02**;
- **preservation (exact rule)**: outside-mask preservation must be **exactly
  1.0** for every generated output — pixels outside the mask are bit-identical by
  construction, so any value < 1.0 is a compositor defect and fails the study
  (`ragregen.metrics.preservation` docstring);
- **exhaustive visual review** of every arm-3 output that differs from its draft:
  a visual **PASS means no reference-photo paste-through and no material
  background/layout replacement**; any such artifact fails.

## 8. Readiness and what a PASS authorizes

Gates are conjunctive: detector gate (§5) AND end-to-end gates (§7), with exact
denominators and reasons reported. Outcomes:

- **FAIL / INCONCLUSIVE** → stop; report; no integration.
- **PASS** → authorises **integration testing only** — wiring the frozen policy
  into the pipeline behind a flag and exercising it — **not** an unqualified
  production claim and **not** the 92-case run.
- **Thin PASS guard**: if Phase B lands **exactly on a predeclared discrete
  boundary** — recall exactly 18/24, or false positives exactly 2/24 — label it
  **thin** and require one further untouched confirmation holdout before the
  92-case run. (No Wilson-bound condition enters this guard: §5 keeps Wilson
  intervals descriptive, and a confidence-bound promotion gate is not predeclared;
  the thin trigger is the discrete count boundary alone.)

## 9. Provenance and immutability

Every manifest and result records SHA-256 hashes of all source artifacts, model
ids **and weight-file hashes**, prompts, the **full environment manifest** +
venv wheel hashes, retrieval/index hashes, the **clean code snapshot hash** (not
merely `git HEAD`, which the dirty worktree does not pin), timestamps, and
`protocol_status`. Writers refuse to overwrite frozen files. The exposed 24-case
development data and its artifacts are never mixed into this holdout.

## 10. Scope boundaries

Phase B must **not**: reuse any concept/reference/reserve seen in development or
Phase A; fit or tune any threshold; apply LOCO; exclude or replace cases after
scores are visible; run retries, best-of-N, or any selector; integrate into
production beyond flag-gated integration testing; or launch the 92-case run. Those
remain gated on a clean (non-thin) Phase B PASS and a subsequent explicit
decision.
