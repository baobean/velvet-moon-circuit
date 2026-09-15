# Stage-A′ result — per-part inpaint-restore: does PartType-hub selection beat raw-NN retrieval?

**Date:** 2026-09-07
**Spec:** `docs/superpowers/specs/2026-09-07-relational-mmkg-stage-a-prime-inpaint-design.md`
**Plan:** `docs/superpowers/plans/2026-09-07-relational-mmkg-stage-a-prime-inpaint.md`
**Run:** `configs/restore_full.yaml` — 1176 cells, 30 inpaint steps, 3 draws, build-levels {0,1,2,4}, 8 concepts.
**Outputs:** `outputs/mmkg/restore/{recovery,hub_vs_rawnn,stratified}.json` + 1176 per-cell rows + `_imgs/`.

## Verdict: DECISIVE NEGATIVE

Under a **working, sensitive deficit**, PartType-hub selection is statistically indistinguishable
from plain embedding nearest-neighbour retrieval for choosing which crops to borrow. The graph's
hub structure buys nothing over raw-NN. Per spec §9 this is the decisive negative — **do not advance
to a large MMKG build or the rag-regen DB swap on the strength of the hub hypothesis.**

## The make-or-break test (`hub_vs_rawnn.json`)

Paired **+Hub − +RawNN** delta in DINOv3 restored-region fidelity, matched at the same k_eff per cell:

| build_P | mean Δ (Hub−RawNN) | win rate | n | Wilcoxon p |
|--------:|-------------------:|---------:|--:|-----------:|
| 0 | **+0.0033** | 0.40 | 25 | 0.54 |
| 1 | **−0.0031** | 0.48 | 25 | 0.84 |

Both indistinguishable from zero, in both directions. Hub selection neither helps nor hurts relative
to raw nearest-neighbour.

## The deficit is real and the metric has power (`stratified.json → borrowing_helps`)

This is what makes the null decisive rather than an infra artifact. Borrowing conditioning crops
(either arm) massively out-restores the no-borrow Isolated reference:

| contrast | mean Δ | win rate | n | Wilcoxon p |
|---|---:|---:|--:|---:|
| Hub − Isolated | **+0.121** | 0.88 | 25 | 4.2e-6 |
| RawNN − Isolated | **+0.117** | 0.92 | 25 | 2.6e-6 |

Masking + inpaint-restoring a part region creates a genuine fidelity hole, and borrowed crops fill it
(+0.12 DINOv3, p≈1e-6). The measurement is sensitive to *whether* you borrow — it is simply blind to
*hub-vs-raw-NN* as the selection rule, because the two select near-identical crops.

## Recovery curve (`recovery.json`, DINOv3 by build-level)

| arm | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| isolated | 0.077 | 0.247 | 0.223 | 0.143 | 0.226 |
| random | 0.139 | 0.184 | 0.190 | 0.151 | 0.210 |
| rawnn | 0.195 | 0.222 | 0.215 | 0.190 | 0.221 |
| hub | 0.198 | 0.219 | 0.210 | 0.190 | 0.224 |

The curve is **not flat** — Isolated climbs from 0.077 (no own-part crops) to ~0.23 once its own part
is represented, confirming the deficit responds to conditioning. hub and rawnn track each other within
noise at every level.

## Reliability stratification (`stratified.json → hub_vs_rawnn`)

| held-P-crop stratum | mean Δ (Hub−RawNN) | win rate | n | p |
|---|---:|---:|--:|---:|
| held5+ | +0.006 | 0.47 | 19 | 0.18 |
| held3–4 | −0.058 | 0.00 | 4 | 0.25 |
| held2 | +0.098 | 0.50 | 2 | 1.0 |

No stratum shows hub winning; the largest, most reliable stratum (held5+, n=19) is a clean null. The
two small strata (n=4, n=2) are underpowered and point in opposite directions.

## Interpretation

The hypothesis was that PartType hubs let knowledge-transfer selection beat naive retrieval. It does
not. With `select_borrowed` restricted to the same part-pool at matched k_eff, the hub-aware selector
returns essentially the same crops as raw embedding NN — so the two arms are the same experiment. The
value in this pipeline lives entirely in the *decision to borrow own-part references at all* (+0.12),
which is retrieval, not graph structure.

## Consequence (per spec §9 honesty commitment)

- **Stop** the hub-selection thesis here. Do not scale to a big relational MMKG or swap rag-regen's
  raw-image DB for a hub-DB expecting a selection-quality gain — there is no evidence for one.
- The **borrow-vs-isolated** effect is strong and worth keeping: a flat retrieval index of own-part
  crops is a defensible, cheaper design than a PartType-hub graph for the restore/repair use case.
- Infra is validated (smoke gate passed, composite localization exact, k_eff matched per cell), so
  this null is about the hypothesis, not the measurement.

## Provenance / caveats

- 30 inpaint steps (not 50): the comparison is a *relative* Hub-vs-RawNN paired delta, unbiased by
  step count — both arms share identical geometry, mask, k_eff, and step budget per cell.
- Legacy SDXL inpaint + composite paste-back (cached-only constraint; no dedicated inpaint checkpoint
  downloaded). Outside-box pixels exactly preserved (smoke OUTSIDE d=0.00); only the box is filled.
- The run's Python process core-dumped on CUDA teardown *after* logging `DONE 1176 rows` and flushing
  all three result JSONs; every output is complete and intact.
