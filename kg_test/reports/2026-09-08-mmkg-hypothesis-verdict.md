# MMKG relational-transfer hypothesis — final verdict

**Date:** 2026-09-08
**Question:** does a relational MMKG (shared PartType/taxonomy hubs) let knowledge-transfer selection
beat plain nearest-neighbour retrieval for the rare-concept **repair** use case (the MMKG→rag-regen
thesis)? **Answer: NO — abandon the hub-beats-retrieval hypothesis for visual repair.** Retrieval is
what works; relational structure adds nothing over it and an orthogonal relation actively hurts.

## The evidence chain (each step forced the next)

1. **Stage-A (2026-09-06):** hub vs rawnn null, but the starvation deficit never bit (flat recovery) →
   null-with-caveat, not decisive.
2. **Stage-A′ inpaint (2026-09-07):** rebuilt with a real mask+inpaint deficit. Deficit bit
   (borrow−isolated +0.12, p≈1e-6) but hub−rawnn = +0.003 (p=0.54) → **decisive negative for
   SigLIP2 hubs.**
3. **Audit:** the null hid structure (corr(Jaccard,|Δ|)=−0.69); hub selections differ from rawnn yet
   don't help → need to decouple filtering from re-ranking.
4. **HubNN (2026-09-07):** hub-gated candidates, query-ranked. HubNN−RawNN ≈ 0 (18/24 exact ties);
   HubNN−Hub ≈ 0 (zero-mean noise). → **both components of a SigLIP2-induced hub are redundant with
   NN**, as they must be — the hub *is* an embedding neighbourhood.
5. **OracleHub precondition (2026-09-07):** the only way the hypothesis could still be true is a hub
   built on a signal *orthogonal* to SigLIP2. Expanded to a 25-concept / 10-family taxonomy; confirmed
   taxonomic siblings are embedding-distant (87% NN-missed) → the structure to test it exists.
6. **OracleHub result (2026-09-08):** under a working, powered deficit (rawnn−random +0.062, p=2e-4),
   **oraclehub ≈ random** (−0.002, p=0.43) and **< rawnn** (−0.064, p=4e-4), *worse* on the far
   stratum. More siblings → worse; the only non-losing families are those where taxonomy coincides with
   appearance. → **an orthogonal relation does not help visual restoration; it hurts.**

## What this establishes

Refuted from **both** directions:
- an **embedding-induced** hub can only re-rank within the NN neighbourhood → redundant with NN (HubNN);
- an **orthogonal** (taxonomy) hub reaches embedding-distant crops, but those carry no visual-restoration
  value → no better than random, worse than NN (OracleHub).

The mechanism is simple and now well-supported: the repair task rewards **visual** fidelity, IP-Adapter
conditioning transfers **visual appearance**, so the useful signal is visual (embedding) similarity —
i.e. exactly what plain retrieval optimises. A graph helps only insofar as it reproduces NN (then it is
redundant) or departs from it (then it is worse).

## Recommendation

- **Abandon** the "relational MMKG beats retrieval" claim as a contribution for the visual repair use
  case. Do **not** build a large relational MMKG or swap rag-regen's raw-image DB for a hub/taxonomy-DB
  expecting a repair-fidelity gain — there is no evidence for one and evidence against.
- **Keep** the one robust, strongly-supported finding: **borrowing a good retrieved reference helps a
  lot** (borrow−isolated +0.12; rawnn−random +0.062). A flat own-part / cross-concept **retrieval**
  index is the defensible, cheaper design. If rag-regen's repair is revisited, it is a
  *retrieval-quality* problem, not a graph problem.
- **Out of scope / not refuted:** relational or attribute structure for *non-visual* objectives
  (attribute/semantic correctness, rare-attribute recall, diversity). If the project wants to keep an
  MMKG angle, that is the only remaining place to look — and it needs a different metric than visual
  fidelity. **Stage-B VLM-attribute hubs** would test that; **FewMedical-XJAU**
  (`reports/2026-09-07-fewmedical-xjau-scope.md`) is a candidate external dataset with authoritative
  taxonomy if a visual-repair replication were ever wanted, but the present result makes that low
  priority.

## Honesty note

Every negative here was taken only after (a) confirming the deficit bit and (b) confirming the metric
had power, per the standing commitment. The precondition/power checks are what make these decisive
refutations rather than infra artifacts. Reports: `2026-09-07-stage-a-prime-result.md`,
`-audit.md`, `-hubnn.md`, `2026-09-07-oraclehub-precondition.md`, `2026-09-08-oraclehub-result.md`.
