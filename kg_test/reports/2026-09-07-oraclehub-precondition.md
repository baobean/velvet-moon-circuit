# OracleHub precondition analysis — expanded taxonomy set (CPU-only)

**Date:** 2026-09-07
**Follows:** `reports/2026-09-07-oraclehub-scope.md` (which found the 8-concept set could not test the
hypothesis). **Method:** CPU-only, over the existing per-part SigLIP2 exemplars in
`outputs/<concept>/kg.json` (65 concepts already embedded — no new build). **No GPU.**

> **Hypothesis under test:** OracleHub is only a meaningful test if it introduces relational
> information **not already recoverable from SigLIP2 similarity** — i.e. it must be able to identify
> references that are *taxonomically relevant but embedding-distant*, which RawNN cannot reach.

The precondition asks exactly one thing: **does that structure exist in the data?** If taxonomic
siblings are already embedding-near (as they were for the only sibling pair in the 8-concept set),
OracleHub is redundant with RawNN by construction and must not be run.

## Expanded set & taxonomy density

65 concepts carry per-part SigLIP2 exemplars; 36 are confidently mapped to a family (scientific name
recorded in `scratchpad/tax_precond.py` for audit). **10 families have ≥2 members:**

| family | n | members |
|---|--:|---|
| Fabaceae | 8 | Ashok, Akashmoni, Karanja, Sisso, Golden Shower Tree, Koinar, Piliostigma, Holudkrishnachura |
| Lauraceae | 2 | Avocado, Camphor Tree |
| Combretaceae | 2 | Bahera, Haritaki |
| Moraceae | 2 | Jack Fruit, Chaplash |
| Myrtaceae | 2 | Guava, Baro bottle brush |
| Anacardiaceae | 2 | Mango, Marking Nut tree |
| Calophyllaceae | 2 | Nageshore, Mastwood |
| Lecythidaceae | 2 | Cannonball Tree, Hijol |
| Arecaceae | 2 | Palm, Khejur |
| Rubiaceae | 2 | Haldu, Crown Gardenia |

(Taxonomy is best-effort from common names → scientific names; the family assignments should be
verified against dataset provenance before publication, but the structural conclusion is robust to a
few reassignments.)

## Precondition gate & result

For each query cell (mapped concept c, part p) with ≥1 same-family sibling, rank **all** other-concept
part-p exemplars (the full 65-concept pool = realistic NN candidates) by cosine to the query, and find
the best sibling's rank.

- **Gate A** = NN top-K (K=4) **misses every sibling** (best sibling rank > 4) → RawNN would never
  borrow a taxonomically-correct crop.
- **Gate B** (strong) = best sibling rank > 12 → the sibling is genuinely deep in the ranking.

| metric | value |
|---|---|
| cells with a taxonomic sibling for the part | **89** |
| **pass Gate A** (NN top-4 misses all siblings) | **77 / 89 = 87%** |
| pass Gate B (best sibling rank > 12) | 51 / 89 = 57% |
| cells where NN top-4 has **zero** siblings (all borrows off-family) | 77 / 89 = 87% |
| mean best-sibling rank | **20.3** of ~59 candidates |
| distinct concepts / families passing Gate A | 25 / 10 |

Best-sibling-rank distribution: rank 1 (NN top) 5 · rank 2–4 7 · rank 5–12 26 · **rank >12 51**.

**Verdict: PRECONDITION SATISFIED (strongly).** Unlike the 8-concept set, taxonomic siblings here are
systematically embedding-distant — NN borrows visually-similar but taxonomically-unrelated crops in
87% of cells. The relation OracleHub encodes is genuinely *not recoverable from SigLIP2 similarity*.
Concrete distant cases: Haritaki/branching (sibling at rank 61/64), Baro-bottle-brush/bark (60/64),
Chaplash/leaf (55/59), Hijol/bark (20/64), Nageshore/bark (26/64).

### Important caveat (what the precondition does *not* establish)

It proves the *structure* exists (distant, NN-missed siblings), **not** that those siblings are
*good conditioning references*. A sibling may be embedding-distant because it is genuinely visually
different, which could make it a worse restore reference despite taxonomic relevance. Whether
taxonomy's "deeper" relevance actually improves restoration is exactly what the GPU experiment tests.
The precondition only earns the right to run it.

## Experiment design — OracleHub vs RawNN

Selector implemented: `graft/transfer.py::select_borrowed(kind="oraclehub", …)` — restrict candidates
to the query's taxonomic family (labels = family ids, the oracle signal), rank by query similarity,
2/concept cap. Structurally identical to `hubnn` but with taxonomy labels instead of HDBSCAN labels,
so the ONLY thing that changes vs the SigLIP2 arms is the candidate-group definition. Unit tests:
`tests/test_transfer_unit.py::test_oraclehub_uses_taxonomy_labels_not_embedding` (+6 others, all pass).

**Arms** (shared Stage-A′ consumer, matched k_eff, identical mask/geometry per cell):
`isolated` · `rawnn` · `oraclehub`. Make-or-break contrast = **oraclehub − rawnn**.

**Stratification (required, per instruction 6):** report `oraclehub − rawnn` split by the sibling's NN
rank — `near` (rank ≤ 4), `mid` (5–12), `far` (>12). A win concentrated in **far** = "taxonomy adds
value exactly where the embedding misses the relation" (the hypothesis). A flat null across strata =
generic null. A win only in `near` = NN by another name.

### Two ways to run it — a real tradeoff

| | **A. Reuse Stage-A′ geometry (no new build)** | **B. Build geometry for qualifying concepts** |
|---|---|---|
| query concepts | the 5 with held-out stores: Ashok, Avocado, Camphor, Hijol, Nageshore | the 25 Gate-A concepts |
| eligible cells | **12** pass Gate A (of 18 with a sibling) | **77** |
| k_eff | **1 for Hijol/Nageshore/Avocado/Camphor; 5–7 for Ashok** | richer (Fabaceae dense) |
| new GPU build | **none** — reuse held-out boxes/masks; borrow sibling exemplar crops (already on disk) | held-out part store for ~25 concepts (`worker_build_heldout_parts`) |
| inpaint cost | ~12 cells × 4 levels × 3 draws × 2 arms ≈ **288 inpaints (~1–1.5 h)** | several × larger + the build |
| power | **low** (n=12, mostly k_eff=1) — a null would be suggestive, not decisive | high — decisive |
| the interesting cells | **present**: Hijol & Nageshore siblings at rank 12–28 (8 cells) | present, many more |

**Neither reuses the 1176-cell scores** (OracleHub selections are new), so both need fresh inpaints;
option A is small and cannot flip from any existing arm. Option A's Hijol/Nageshore cells are the
cleanest single-crop test (does a rank-12–28 taxonomic crop beat NN's off-family top pick?), but n=12
/ k_eff=1 means a null there should not be over-read.

## Recommendation & decision needed

Precondition is met, so we should proceed — but the A-vs-B tradeoff (fast/weak vs build/decisive) is a
resource call for you. My recommendation: **run A now** as a fast directional read (it can't produce a
false *positive*, and a clear win on the Hijol/Nageshore `far` cells would already be informative),
**and in parallel scope B** so that if A is encouraging or ambiguous we build the high-power version.
Per the standing instruction, **no abandon/advance verdict** until we have an OracleHub result. Stage-B
VLM-attribute hubs remain the fallback only if we later want a *denser* orthogonal signal — taxonomy
density is now sufficient, so it is not needed to proceed.

---

## ADDENDUM — Build-B feasibility BLOCKER (data), 2026-09-07

Chose Build B (high-power, ~25 concepts). Checked unique-image availability first: **it does not
exist.** The Treevill `rawdata2` is ~99% duplicates — after content-hash dedup, species have only
~4–6 unique images each. With frozen `k_build_refs=5`, held-out = unique−5, so **only 5 species have
≥3 held-out images** (needed: 1 to mask+restore, ≥1 to score against): Ashok(15 uniq), Nageshore(11),
Avocado(10), Hijol(10), Palm(9). The other 20 Gate-A concepts have exactly **1** held-out image → no
scoring crop → cannot be query concepts. **The precondition (which counts exemplars) passes at 77
cells, but scorable restoration cells are capped at ~13 across 5 query concepts** — essentially the
geometry-reuse option + Palm, k_eff=1 for 12/13. The 8 Hijol/Nageshore + Palm/bark "far" cells
(sibling NN-rank 12–52) remain the cleanest single-crop test, but overall power is low.

**Partial rescue:** lowering the *held-out* build to `k_build=2` (keeping the k_build=5 exemplar borrow
pool unchanged, so Gate-A ranks stay valid; no cross-concept contamination since a query never borrows
from itself) makes species with ≥5 unique images viable, rescuing ~7 more Gate-A queries (Guava,
Bahera, Haritaki, Jack Fruit, Haldu, Mango, Piliostigma) → ~25–30 cells across ~7 families. Minor
protocol deviation (smaller own-build set; build-levels clamp lower). This is the max feasible power on
this dataset.
