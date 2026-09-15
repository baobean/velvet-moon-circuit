# HubNN arm — decoupling hub candidate-filtering from centroid re-ranking

**Date:** 2026-09-07
**Follows:** `reports/2026-09-07-stage-a-prime-audit.md` (which found the Hub−RawNN null bundled two
effects). **Method:** new `hubnn` selector (`graft/transfer.py`), CPU-only re-analysis reusing the
existing 1176-cell DINOv3 scores. **No GPU, no re-inpaint.**

## What HubNN is

`hubnn` = **hub candidate filter** (restrict to the query's own PartType hub, same 2/concept cap as
`hub`) **+ query ranking** (rank those candidates by similarity to the query, *not* the hub centroid).
It sits exactly between the two existing arms:

| arm | candidate pool | ranking signal |
|---|---|---|
| RawNN | whole part pool | query similarity |
| **HubNN (new)** | **query's hub only** | **query similarity** |
| Hub | query's hub only | hub-centroid similarity |

So **HubNN − RawNN** isolates the value of *hub-gated candidate filtering*, and **Hub − HubNN**
isolates the *centroid re-ranking* effect. Together they decompose the original Hub−RawNN null.

## Score reuse is exact (why this is CPU-only)

The inpaint conditioning is an **order-independent weighted sum** of the borrowed crops' IP-Adapter
embeds (`models.py::inpaint_multi`, `avg = Σ wᵢ·embᵢ`), `seed=0` fixed, weights depend only on the
own/borrowed *counts* (identical per cell across arms), and base image / mask / prompt are identical.
Therefore **two arms that select the same borrowed crop *set* produce a byte-identical inpaint and
DINOv3 score.** I reuse the already-computed score wherever HubNN's set matches an arm that ran.

## Coverage

| HubNN's set equals… | cells |
|---|---|
| RawNN's set exactly | **17 / 25** |
| Hub's set exactly | 7 / 25 |
| neither (NOVEL, needs GPU) | **1 / 25** (Ashok/leaf) |

24 of 25 cells are scorable with zero GPU. The one novel cell (12 inpaints) cannot flip a 24-cell
null and is not worth a GPU trip on its own.

## Result 1 — the hub *filter* is worthless over plain NN

**HubNN − RawNN:** mean Δ = **+0.0037**, win 2/24, **18 exact ties**, Wilcoxon p = 0.46.
mean Jaccard(HubNN, RawNN) = **0.864**.

The hub gate reproduces RawNN almost exactly. Reason (from the audit): the hubs are HDBSCAN clusters
in the *same SigLIP2 space RawNN ranks in*, and 89% of RawNN's picks already fall inside the query's
hub — so restricting to the hub removes only the ~11% out-of-hub picks, and in 17/25 cells that did
not even change the top-k. **Hub-gated candidate generation adds no fidelity over embedding NN.**

## Result 2 — the centroid re-rank is zero-mean noise

**HubNN − Hub:** mean Δ = **+0.0024**, win 6/24, Wilcoxon p = 0.49 — null in aggregate, but the
per-cell `ΔvsHub` swings are large and two-signed:

| cell | HubNN(=RawNN) − Hub | direction |
|---|---|---|
| Ashore / branching | **+0.104** | query-rank beats centroid |
| Hijol / branching | +0.088 | query-rank beats centroid |
| Hijol / leaf | +0.062 | query-rank beats centroid |
| Avocado / leaf | **−0.052** | centroid beats query-rank |
| Camphor / branching | −0.058 | centroid beats query-rank |

Query-ranking (HubNN) beats centroid-ranking (Hub) on some cells and loses on others; it averages to
≈0. So the centroid re-rank is **high-variance noise, not a systematic harm** — my audit's
"centroid ranking is what hurts" was too strong. The honest statement: *neither* ranking signal
systematically wins for this restoration task.

## Query-similarity of picks

mean qsim — HubNN **0.877** > RawNN 0.869 > Hub 0.849. HubNN is (marginally) the most query-aligned
because it query-ranks within the tightest neighbourhood. Yet its fidelity is indistinguishable from
RawNN — i.e. *within* an already-coherent hub, squeezing out a bit more query-similarity does not move
restoration fidelity. This is consistent with the audit's "within-hub crops are near-interchangeable
as conditioning."

## What this settles, and what it does NOT

**Settled:** the Hub−RawNN null is not an artifact of a bad ranking choice masking a good filter.
*Both* the hub filter (Result 1) and the centroid re-rank (Result 2) are null vs plain NN. Every hub
mechanism built purely from SigLIP2 clustering is redundant with SigLIP2 NN — as it must be, since the
hub *is* a SigLIP2 neighbourhood.

**Not settled:** whether a hub built on a signal **orthogonal to SigLIP2** (taxonomy / attributes)
could reach an embedding-distant-but-relevant crop that NN cannot. HubNN actually *strengthens* the
case that this is the only remaining open question — the redundancy is now demonstrated from two
angles, so the deferred **OracleHub** arm (an oracle relation the embedding can't see) is the decisive
next test. Scope: `reports/2026-09-07-oraclehub-scope.md`.

**No abandon/advance recommendation is made here** — that waits on OracleHub, per the standing
instruction.

## Reproducibility

Selector: `graft/transfer.py::select_borrowed(kind="hubnn", …)` + `_capped_topk`. Tests:
`tests/test_transfer_unit.py::test_hubnn_is_hub_gated_but_query_ranked`,
`::test_hubnn_respects_max_per_concept`. Full non-GPU suite: 120 pass. Analysis is a pure re-read of
`outputs/mmkg/graph.json` + `outputs/mmkg/restore/*.json`.
