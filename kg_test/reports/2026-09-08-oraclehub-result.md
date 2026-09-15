# OracleHub result — does an orthogonal taxonomy relation beat NN retrieval? NO (decisive)

**Date:** 2026-09-08
**Run:** `outputs/mmkg/oraclehub/` — 76 Gate-A cells, 25 concepts, 10 families, build_P=0, 654 inpaints
(arms random/rawnn/oraclehub, matched k_eff, 30 steps), config `configs/oraclehub.yaml`.
**Precondition (passed):** `reports/2026-09-07-oraclehub-precondition.md` — siblings are embedding-distant
(87% NN-missed). **This is the test that precondition earned.**

> **Hypothesis:** a relation orthogonal to SigLIP2 (taxonomy) can identify references that are
> taxonomically relevant but embedding-distant, improving restoration where RawNN fails.

## Verdict: REFUTED — taxonomy transfer is no better than random and worse than NN

| contrast (DINOv3, cell-paired) | overall | **far (rank>12)** | mid (5–12) |
|---|---|---|---|
| **oraclehub − rawnn** | **−0.064** (win 34%, p=4e-4) | **−0.065** (win 34%, p=0.011) | −0.063 (p=0.009) |
| oraclehub − random | −0.002 (win 45%, p=0.43) | +0.0004 (p=0.53) | −0.007 (p=0.82) |
| rawnn − random | **+0.062** (win 63%, p=2e-4) | +0.065 (p=0.012) | +0.056 (p=0.004) |

Mean restored-region fidelity by stratum: **rawnn 0.270/0.251 > random 0.205/0.195 ≈ oraclehub 0.205/0.188.**

Three facts, together decisive:
1. **The metric has power and the deficit is real:** RawNN beats random by **+0.062 (p=2e-4)** — borrowing
   a *visually* good crop genuinely restores fidelity. So a null for OracleHub is not insensitivity.
2. **OracleHub ≈ random** (−0.002, p=0.43), and on the make-or-break **far** stratum it is *exactly*
   random (+0.0004, p=0.53). A taxonomically-correct but embedding-distant crop carries **no more
   restoration value than a random crop.**
3. **OracleHub is significantly *worse* than RawNN** (−0.064, p=4e-4), including on the far stratum.
   Conditioning on a family-correct-but-visually-different crop actively pulls the restoration away
   from the query's own appearance.

## Why (robustness breakdown)

- **More siblings → worse:** k_eff=1 gives −0.048; k_eff=4 (dense Fabaceae) gives **−0.101**. Averaging
  several family-correct-but-visually-off crops hurts more, not less — the opposite of a real transfer
  signal.
- **The only families where OracleHub doesn't lose are the ones where taxonomy and appearance
  coincide:** Arecaceae (+0.23, Palm↔Khejur — palms look like palms) and Lauraceae (+0.02,
  Avocado↔Camphor). Every family where taxonomy *diverges* from visual similarity — the entire point
  of the test — is negative (Myrtaceae −0.16, Rubiaceae −0.16, Fabaceae −0.10, …). OracleHub only ties
  NN exactly where it would have picked the same crop anyway.

## Interpretation

The restoration/repair task rewards **visual** fidelity, and IP-Adapter conditioning transfers **visual
appearance**. A relation orthogonal to the embedding is, by construction, orthogonal to visual
appearance — so it provides no useful visual prior and behaves like random. The "deeper relevance" of
taxonomy is real but **does not translate into visual restoration fidelity**; where it does help
(Arecaceae) it is because taxonomy and appearance happen to align, i.e. exactly where it adds nothing
over NN. This is a clean, high-powered refutation, not an infra artifact (the precondition confirmed
the structure exists; the metric confirmed it has power).

## Scope of the claim

This refutes relational/taxonomy transfer **for the visual part-restoration (repair) objective** — the
stated MMKG→rag-regen use case. It does not test whether taxonomy helps a *non-visual* objective
(attribute/semantic correctness, rare-attribute recall, diversity); those are different metrics and out
of scope here. For repair fidelity specifically, the evidence is decisive.

## Reproducibility

`graft/oraclehub_run.py` (build_P=0 resident), `graft/oraclehub_analysis.py`, `graft/taxonomy.py`,
`select_borrowed(kind="oraclehub")`. Held-out store `outputs/mmkg_oh/` (k_build=0). Tests: 125 non-GPU
pass incl. `tests/test_oraclehub.py`. Process core-teardown after writing all 654 rows + result JSON.
