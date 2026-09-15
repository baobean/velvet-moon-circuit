# Audit — is the Hub-vs-RawNN "decisive negative" a false negative?

**Date:** 2026-09-07
**Audited run:** `outputs/mmkg/restore/` (1176 cells), report `reports/2026-09-07-stage-a-prime-result.md`
**Method:** static trace of the selector path + empirical replication of both selectors over the
frozen `outputs/mmkg/graph.json` (147 part-instances, 12 hubs), joined against the 1176 per-cell
DINOv3 scores. **No GPU, no re-run of the experiment.**

## TL;DR judgment

The negative is **not a wiring bug and is correctly measured**, but it is **NOT a clean refutation of
the MMKG hypothesis** — it refutes only *this specific hub design*. Two findings make the aggregate
null misleading:

1. **The null hides a real, directional effect.** `corr(Jaccard, mean|Δ|) = −0.69`: the more Hub and
   RawNN actually diverge in what they pick, the *larger* the fidelity gap. A true no-effect would be
   ~0. The +0.003 aggregate is a **cancellation**, not an absence of signal.
2. **Where they diverge, Hub is worse — by construction, not by hypothesis.** On low-overlap cells
   (Jaccard<0.3) mean(Hub−RawNN) = **−0.025** (disjoint cells −0.10/−0.088/−0.062). Hub ranks by
   *hub-centroid* similarity, so it deliberately picks crops **less** similar to the query
   (mean query-cos 0.849 vs RawNN 0.869). A repair-toward-the-query task rewards query-similarity, so
   the graph's re-ranking is actively counterproductive here.

So "Hub ≈ RawNN" is really "**Hub ≈ RawNN on average, but meaningfully worse exactly where the graph
changes the answer.**" That is a fixable design/measurement issue, and the interesting version of the
hypothesis was never tested. **Recommendation: fix the experiment before abandoning MMKG.**

---

## 1. Exact code path for Hub vs RawNN

Both arms enter through one shared consumer (`graft/restore.py::select_refs` → `graft/transfer.py::
select_borrowed`); only the `kind` branch differs. k_eff, mask, geometry, part-pool, weights are
identical (verified §5).

**RawNN** (`transfer.py:12-15`):
```
pool = all part-instances with same part name, excluding own concept   # whole part pool
rank by  pool_embed · query_embed   (query = own[0].siglip2)
take top-k
```
**Hub** (`transfer.py:9-11` → `parttype_graph.py::centroid_prototype`):
```
candidates = members of the query's OWN hub, excluding own concept     # a subset of the pool
centroid   = mean(candidate embeds), L2-normed
rank by  candidate_embed · centroid                                    # NOT the query
cap 2 per concept, take top-k
```

Two structural differences: **(a) candidate pool** — Hub restricts to its induced cluster, RawNN uses
the whole part; **(b) ranking reference** — Hub uses the cluster centroid, RawNN uses the query.
There is **no fallback / override** that makes them equal (checked: no `except → rawnn`, no empty-hub
→ NN path). When the hub is non-empty they are genuinely different functions.

## 2. Candidate overlap (empirical, 25 (concept,part) cells with a hub)

| metric | value |
|---|---|
| mean Jaccard(Hub, RawNN) | **0.38** (median 0.33) |
| mean overlap count | 1.84 of k_eff≈3.6 |
| cells with identical selection | 2 / 25 |
| cells with **fully disjoint** selection | 4 / 25 |
| **RawNN picks landing inside the query's OWN hub** | **89 / 100 = 89%** |

Interpretation: the arms are **not** degenerate — they pick different crops in 23/25 cells. **But**
89% of RawNN's picks fall inside the query's own hub, because the hubs were induced by HDBSCAN *on the
same SigLIP2 space RawNN ranks in*. So hub-membership ≈ embedding neighbourhood; the two arms differ
mostly in *which members of the same coherent cluster* they choose and *how they rank them*, not in
reaching a different region of the space.

## 3. Hub signal — what relational information actually changes the ranking?

The only signal the hub adds over embedding-NN is: **(i)** a hard restriction to the induced cluster,
and **(ii)** re-ranking by cluster-centroid instead of query. Both are *derived from the same
embedding metric*. **There is no relation orthogonal to SigLIP2 distance in this graph** — no
taxonomy, no attribute, no co-occurrence edge. Therefore the hub **cannot** select an
embedding-distant-but-relationally-relevant crop, which was the intended hypothesis
([[mmkg-to-ragregen-direction]]: "transfer a flat baseline *provably can't*"). The one arm that would
inject an independent relation — **OracleHub / taxonomy** — was deferred out of this run. The
experiment thus tests only the *weakest* form of the hypothesis.

## 4. Candidate-pool construction

- Hub and RawNN operate over **different** pools (hub-subset vs whole part), so pool identity is not
  the collapse cause. ✔ no bug.
- The hub pool is **not** so broad that ranking collapses to RawNN by breadth — the largest hubs are
  branching 29/40 and leaf 25/38, but bark/cone have 4 hubs of size 4-11. The collapse is instead
  from **shared metric** (§2/§3), not shared pool.
- Filtering: the `max_per_concept=2` cap is Hub-only and *reduces* overlap with RawNN (removes extra
  same-concept picks RawNN keeps) — it does not delete hub-specific candidates. No dedup/capping
  removes the graph's distinctive picks. ✔ no bug.

## 5. Experiment validity

| check | result |
|---|---|
| k_eff parity Hub vs RawNN | **0/25 mismatches** — borrowed count identical ✔ |
| same mask / geometry / part-pool / weights across arms | yes (single `build_restore_inputs`) ✔ |
| draws give independent selector samples | **NO** — hub/rawnn borrowed sets are byte-identical across all 3 draws (seed unused by these arms); draws only re-randomize own-crop starvation + target image. Effective independent unit = the **25 cells**, which is what the make-or-break n used. Not inflated, but the "×3 draws" adds no selector-comparison power. |
| independence of the 25 pairs | **weak** — 47/72 distinct borrowed crops are reused across >1 cell (max reuse 8), so cells share conditioning material; Wilcoxon independence is mildly violated. Does not change the verdict, but the p-values are optimistic. |

No validity bug that would manufacture a null. Matching is, if anything, *too* clean — it guarantees
the two arms differ only in selection, which is exactly why the −0.69 correlation is meaningful.

## 6. 3–5 concrete cells where the selections differ

Full crop ids; `qsHub/qsRaw` = mean query-cosine of each arm's picks; Δ = mean(Hub−RawNN) DINOv3.

| cell | J | Δ | note |
|---|---|---|---|
| **Ashore / branching** | 0.00 | **−0.104** | hub picks Avocado×2+Camphor+Nageshore (centroid-typical); rawnn picks Ashok+Hijol+Camphor+Avocado (query-near). Hub qsim 0.73 vs 0.84. |
| **Hijol / branching** | 0.00 | **−0.088** | disjoint; hub qsim 0.74 vs 0.84; hub materially worse. |
| **Hijol / leaf** | 0.00 | **−0.062** | disjoint; hub 0.73 vs 0.82. |
| **Ashok / branching** | 0.00 | +0.024 | disjoint but hub slightly better — the one disjoint cell favouring hub. |
| **Camphor / cone_or_flower** | 0.40 | +0.113 | hub qsim 0.843 > rawnn 0.836 (rare case hub is *more* query-near) → hub wins big. |

Pattern: **Δ tracks the qsim gap.** When the hub's centroid re-ranking pulls picks *away* from the
query (qsHub < qsRaw), Hub loses; in the rare cell where it doesn't, Hub wins. The graph is not adding
value — it is perturbing the query-similarity ranking, usually downward.

## 7. Bugs / design issues found

- **No implementation bug** in the selector wiring, k_eff matching, pool construction, or scoring.
- **Design issue #1 (confound): candidate-filtering and re-ranking are entangled.** Hub changes *both*
  the candidate set (hub-gated) *and* the ranking metric (centroid, not query). We cannot tell whether
  hub *filtering* helps, because it is bundled with centroid *re-ranking* that hurts. Needs a
  **hub-gated-NN** arm: hub candidates, ranked by the query (not centroid).
- **Design issue #2 (underpowered hypothesis): the graph carries no signal orthogonal to the
  baseline's metric.** Hubs are induced from the same SigLIP2 space RawNN ranks in, so the graph can
  only *re-rank within* the NN neighbourhood, never reach outside it. The hypothesis "hubs enable
  transfer NN can't" is untestable without an independent relation (taxonomy/attribute) — the deferred
  **OracleHub** arm.
- **Measurement issue #3: task–metric mismatch for the centroid selector.** The task is
  *reconstruct the query's own part*; centroid-prototypicality is the wrong ranking objective for that
  and is what drives the negative. This tells us centroid-ranking is bad for *repair*; it says little
  about hubs for *generation of a novel/typical instance*.

## 8. Is the negative trustworthy? Could it be a false negative?

**Trustworthy as:** a correct refutation of *centroid-prototype selection over embedding-induced hubs
for a query-reconstruction task.* That mechanism does not beat NN and slightly hurts. Real, measured,
reproduced.

**A (partial) false negative for:** the broader "relational MMKG transfer" claim. The −0.69
Jaccard/|Δ| correlation proves selection *does* move the metric; the experiment simply (a) bundled a
helpful-or-neutral hub filter with a harmful centroid re-rank, and (b) never gave the graph a signal
NN doesn't already have. Under those two limits a null (or slight loss) is the *expected* outcome
**regardless of whether relational transfer has value** — so the result cannot discriminate the
hypothesis it was meant to test.

## 9. Recommendation — fix the experiment, do not abandon MMKG yet

Cheap changes, same 1176-cell harness, before any verdict:

1. **Add a `hubnn` arm** = hub-gated candidates ranked by the **query** (decouples filtering from
   centroid re-rank). If `hubnn ≈ rawnn` → hub *filtering* is worthless. If `hubnn > rawnn > hub` →
   the filter helps and centroid re-ranking was the whole problem (a one-line selector fix).
2. **Run the deferred `oraclehub` arm** (taxonomy relation) — the only arm that injects a signal
   orthogonal to SigLIP2. This is the actual test of "transfer NN can't do."
3. **Report stratified by the Jaccard of the day**, not just pooled: the pooled mean is
   structurally ~0 whenever most cells overlap.
4. Only if `hubnn` and `oraclehub` **both** ≈ rawnn under the working deficit is the negative
   decisive. Then abandon with confidence.

Cost: arms 1 and 3 are CPU-only re-analysis of a new selector on the *existing* geometry plus one
extra inpaint arm; well under the original run. Do this before the rag-regen swap decision.
