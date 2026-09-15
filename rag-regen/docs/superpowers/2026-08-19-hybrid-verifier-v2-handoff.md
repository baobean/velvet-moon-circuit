# Start here: Hybrid Verifier V2

## Objective

Execute the CPU-only Phase A plan. Do not run GPU jobs, create a fresh holdout, modify production scheduling, or launch the 92-case experiment.

## Required reading

1. `docs/superpowers/specs/2026-08-19-hybrid-verifier-v2-design.md`
2. `docs/superpowers/plans/2026-08-19-hybrid-verifier-v2.md`
3. `outputs/hybrid_validation_20260819_085952/audit_addendum.md`
4. `outputs/hybrid_validation_20260819_085952/visual_audit_codex.md`
5. `.superpowers/sdd/2026-08-18-hybrid-verifier-validation/progress.md`

## Execution instruction for the next agent

Use `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Preserve the dirty worktree and use scoped diff checkpoints; the predecessor plan explicitly avoided commits because touched files already contain unrelated changes.

Start with Task 1. Use TDD exactly as written. Phase A must remain CPU-only and must label the exposed run as development. The key protocol invariant is:

```text
identity_truth is immutable
mask/reference status controls selector_eligible only
```

For reconstruction, use the Qwen artifact's stored truth values so `azawakh` and `bergamasco_shepherd` remain detector `FAIL` cases while being selector-ineligible. Never edit the source label CSV.

At Task 5, obey the stop condition:

- readiness FAIL/INCONCLUSIVE: publish the development finding and stop;
- readiness PASS: freeze the policy hash, then author a separate fresh-holdout spec/plan; still do not integrate production or run 92 cases.

## Known source artifacts

- screen/semantic: `outputs/screen_20260818_172307`
- candidate bank: `outputs/holdout_candidates_20260818_182938`
- reranker/DINO/v1 report: `outputs/hybrid_validation_20260819_085952`
- dataset: `configs/dataset_verifier_holdout.yaml`

## Baseline facts to reproduce

- current CSV: 12 PASS, 10 FAIL, 2 EXCLUDE;
- reconstructed identity truth: 12 PASS, 12 FAIL;
- selector eligibility: 22 eligible, 2 mask failures;
- candidate attempts: 53;
- v1 union: TP 10, FP 5, TN 7, FN 0 on the post-hoc 22-case report;
- v1 selector sign accuracy: `0.6226415094`;
- no direct held-out-reference leakage was found.

## Definition of a safe handoff completion

The next agent has completed this handoff only when all five plan tasks are reviewed, the full non-GPU suite is green, and either:

- a failure finding explains why a new signal is required; or
- an immutable cross-validated policy plus SHA-256 exists and a separate Phase B plan is ready for human review.

