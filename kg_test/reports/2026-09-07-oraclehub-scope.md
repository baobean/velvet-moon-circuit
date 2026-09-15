# OracleHub — scope, orthogonality argument, and a feasibility BLOCKER

**Date:** 2026-09-07
**Purpose:** define the deferred `oraclehub` arm — the only remaining test of whether a hub built on a
signal **orthogonal to SigLIP2** can beat plain NN, after HubNN showed every SigLIP2-induced hub
mechanism is redundant with NN (`reports/2026-09-07-stage-a-prime-hubnn.md`).
**Headline:** as specified (taxonomy oracle) OracleHub is **not runnable on the current 8-concept set**
— the precondition fails empirically. This doc says exactly why, and what would make it runnable.

## 1. The relational signal and why it is orthogonal to SigLIP2

**Signal:** biological taxonomy. Each concept's source *species* maps (via an external oracle —
a fixed species→genus→family→order table, e.g. GBIF/Catalogue of Life) to a lineage. A taxonomic hub
groups same-part crops whose species share a taxon (family, say), **regardless of visual appearance.**

**Why genuinely orthogonal:** the oracle never sees a pixel. Membership is decided by the species name
→ lineage, so two crops can be:
- taxonomic **siblings but embedding-distant** (e.g. juvenile vs mature bark of two con-familial
  species) — the case NN *misses* and the hypothesis needs, or
- embedding-**close but taxonomically unrelated** (convergent morphology across families) — the case
  NN *wrongly* borrows.

This is the property HDBSCAN hubs (and therefore HubNN) can never have: they are defined *in* the
SigLIP2 metric, so they cannot encode a relation the metric doesn't already express.

## 2. The hypothesis and how OracleHub would test it

Original claim ([[mmkg-to-ragregen-direction]]): hubs enable transfer "a flat baseline *provably
can't*" — reaching an **embedding-distant but relationally relevant** crop. Test design:

1. Arms, same shared consumer / k_eff / geometry as the existing matrix: add `oraclehub` =
   borrow from same-part crops of **taxonomic siblings** (exclude own concept), ranked by query
   similarity within the sibling set.
2. **Stratify by sibling embedding-distance.** For each cell compute how embedding-distant the
   taxonomic siblings are from the query. The decisive contrast is `oraclehub − rawnn` **on the
   high-distance stratum** — cells where the relationally-correct crop is one NN would never rank
   top-k. A win *there* is the graph contribution NN cannot make; a win only on the low-distance
   stratum is just NN by another name.
3. Honesty rule carries over: if `oraclehub ≈ rawnn` even on the high-distance stratum under the
   working deficit, the orthogonal-relation hypothesis is refuted too.

## 3. FEASIBILITY BLOCKER (measured, CPU-only)

Two empirical facts kill the experiment on the current 8 concepts:

**(a) Taxonomy is almost all singletons.** Best-effort lineage of the 8:

| concept | family / order |
|---|---|
| Avocado | **Lauraceae / Laurales** |
| Camphor Tree | **Lauraceae / Laurales** |
| Ashok | Fabaceae / Fabales |
| Bamboo | Poaceae / Poales |
| Egyptian lotus | Nymphaeaceae / Nymphaeales |
| Hijol | Lecythidaceae / Ericales |
| Nageshore | Calophyllaceae / Malpighiales |
| Ashore | *unverified* |

Only **{Avocado, Camphor}** share a family. Every other concept is a singleton at family *and* order.
An OracleHub would have exactly **one** non-trivial hub and 6–7 singletons → nothing to borrow for
almost every cell. No experiment.

**(b) Even the one sibling pair is not embedding-distant** — so the precondition in §2 fails where we
*can* test it. For Avocado↔Camphor (true Lauraceae siblings), mean same-part cosine and the best
NN non-sibling neighbour of a Camphor query:

| part | sibling cos (Av,Cam) | best other-concept cos | is the sibling the NN? |
|---|---|---|---|
| bark | 0.833 | 0.833 (Avocado) | yes |
| branching | 0.834 | 0.843 (Ashok) | no, but within 0.009 |
| cone_or_flower | 0.721 | 0.778 (Bamboo) | no |
| leaf | 0.814 | 0.823 (Ashok) | no, but within 0.009 |

The taxonomic sibling is roughly as embedding-close as NN's best pick — taxonomy would pull a crop NN
**already nearly selects**, not a far-but-relevant one. There is no high-distance stratum to win on.

## 4. What would make OracleHub a real test

Do **not** run it on the current set — it would produce a null that means "no taxonomic structure
here", not "the hypothesis is false" (exactly the infra-vs-hypothesis confusion the spec forbids).
To run it meaningfully:

1. **Expand the concept set for taxonomic density.** The build pool has 30+ species
   (`logs/species_ranking.txt`). Choose ≥2–3 species per family for several families (e.g. multiple
   Lauraceae, multiple Fabaceae, multiple monocots), so families have real membership and per-cell
   siblings exist.
2. **Pre-register the precondition as a CPU gate (cheap, run before any GPU):** for the chosen set,
   compute the §3(b) table and keep only families where siblings are **embedding-distant** from the
   query yet taxonomically correct (a non-empty high-distance stratum). If the gate is empty, stop —
   the set still can't test the hypothesis.
3. Only then build the taxonomy oracle table, add the `oraclehub` selector (mirrors `hubnn` but the
   candidate set = taxonomic siblings, not the SigLIP2 hub), and run the deficit matrix **restricted
   to the qualifying families**, reporting `oraclehub − rawnn` stratified by sibling distance.
4. **Alternative orthogonal signal if taxonomy stays too sparse:** VLM-extracted discrete botanical
   attributes (leaf compound/simple, phyllotaxy, bark texture class) — this is the deferred **Stage-B
   AttributeValue hub**, a *different* orthogonal relation with denser structure. It may be the more
   testable route to the same hypothesis and is worth pricing against the concept-set expansion.

## 5. Cost & decision

- §3 findings are **already done** (CPU). The concept-set expansion (step 1–2) is a new build:
  part-crop extraction + graph induction over the added species, then the CPU gate — hours of GPU for
  crop/embperception, no new modelling.
- Recommendation for the **experiment plan** (not an abandon/advance verdict): before committing that
  build, decide taxonomy-expansion vs Stage-B-attributes as the orthogonal signal, since both test the
  same hypothesis and Stage-B may have denser structure on the *existing* concepts.

**No abandon/advance recommendation is made** until we have an OracleHub (or Stage-B) result from a set
that actually satisfies the §2 precondition — per the standing instruction to wait for both HubNN and
OracleHub.
