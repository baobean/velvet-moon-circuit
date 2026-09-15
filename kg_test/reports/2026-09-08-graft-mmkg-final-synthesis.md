# GRAFT / relational-MMKG — end-to-end synthesis and final verdict

**Date:** 2026-09-08
**Scope:** the whole GRAFT relational-MMKG line, from the origin thesis to the last gate. This document
consolidates seven experiments (Stage-A → Stage-A′ → audit → HubNN → OracleHub precondition → OracleHub →
Stage-B Gate 0) into one narrative and one verdict. It supersedes nothing — every number here is sourced
from the per-experiment reports listed in §8 — but it is the single artifact that ties them together.

---

## Verdict (one paragraph)

A relational MMKG — a graph of shared PartType / taxonomy hubs over concept part-crops — provides **no
advantage over plain nearest-neighbour retrieval** for the rare-concept use case, and an orthogonal
relation actively *hurts* the visual objective. This is established from both directions (an
embedding-induced hub is redundant with NN; an orthogonal taxonomy hub reaches distant crops that carry no
restoration value) and across both objectives it could plausibly serve (visual part-repair, decisive;
rare-attribute transfer, under-powered but with zero positive signal). The one robust, strongly-supported
finding that survives is orthogonal to the graph: **borrowing a good *retrieved* reference helps a lot**
(+0.12 DINOv3 over not borrowing; +0.062 over a random reference). The value lives in **retrieval quality,
not graph structure.** The original MMKG→rag-regen thesis (swap the raw-image DB for a hub/taxonomy-DB) is
refuted; a flat retrieval index is the defensible, cheaper design.

## Why GRAFT existed (the thesis under test)

The premise ([[mmkg-to-ragregen-direction]]) was that rag-regen's raw-image retrieval DB could be replaced
by a **multimodal knowledge graph** whose relational structure (shared part hubs, taxonomy) would let
knowledge-transfer selection pick conditioning references a flat retrieval baseline *provably cannot* — in
particular, references that are **relevant but embedding-distant**. If true, the MMKG-DB is a genuine
contribution; if the graph only ever reproduces NN, it is redundant. Every experiment below was the forced
next step in discriminating those two worlds.

---

## The evidence chain (each step forced the next)

| # | experiment | make-or-break contrast | result | reading |
|---|---|---|---|---|
| 1 | **Stage-A** (09-06) | Hub−RawNN, build=1 | **+0.0053**, win 3/8; noise across levels | null **with caveat** — deficit never bit |
| 2 | **Stage-A′ inpaint** (09-07) | Hub−RawNN, real mask+inpaint deficit | **+0.0033** (p=0.54) / **−0.0031** (p=0.84) | **decisive negative for SigLIP2 hubs** |
| 3 | **Audit** (09-07) | is the null a false negative? | corr(Jaccard,\|Δ\|) = **−0.69** | null = cancellation; must decouple + test orthogonal relation |
| 4 | **HubNN** (09-07, CPU) | HubNN−RawNN / HubNN−Hub | **+0.0037** (18 ties) / **+0.0024** (p≈0.49) | **both hub components redundant with NN** |
| 5 | **OracleHub precond.** (09-07) | are taxonomy siblings NN-reachable? | **87% NN-missed** | the orthogonal structure to test the thesis *exists* |
| 6 | **OracleHub** (09-08) | oraclehub−rawnn, far stratum | **−0.064** (p=4e-4); ≈ random | **orthogonal relation has no visual value; it hurts** |
| 7 | **Stage-B Gate 0** (09-08) | taxonomy rare-attr recall − retrieval | **−0.105**, n_rare=19, p=0.5 | last non-visual angle **winds down** |

### Step 1 — Stage-A: the manipulation didn't bite (null with a caveat)
384-cell starvation/recovery, 12 PartType hubs. Hub−RawNN was **+0.0053** at build=1 (win 3/8) and stayed
noise-around-zero across build levels. But the recovery curve was **nearly flat** (isolated 0.317 at
build=1, the highest point) — reducing the build set from 5 images to 1 barely moved DINOv3 fidelity, so
there was no fidelity *deficit* for transfer to recover. Whole-image conditioning on a borrowed part-crop
also diluted identity (Hub−Isolated negative at every level). Honest read: an **under-powered null**, not a
refutation. Two confounds had to be fixed first — instantiate a real deficit, and edit the *specific* part
(inpaint) instead of whole-image conditioning.

### Step 2 — Stage-A′: with a real deficit, the SigLIP2-hub null is decisive
Rebuilt as per-part mask + inpaint restore, 1176 cells. The deficit now bit hard and the metric had power:
**borrowing beats not-borrowing by +0.12** (Hub−Isolated +0.121, p=4.2e-6; RawNN−Isolated +0.117,
p=2.6e-6). Under that working deficit, **Hub−RawNN was +0.0033 (p=0.54) at build_P=0 and −0.0031 (p=0.84)
at build_P=1** — indistinguishable from zero in both directions. This is the decisive negative *for the
embedding-induced hub*: the graph's hub structure buys nothing over raw NN, because at matched k_eff it
selects near-identical crops.

### Step 3 — Audit: the null hid structure, so it wasn't yet a refutation of the *idea*
A no-GPU audit found the aggregate null was a **cancellation**, not an absence of signal:
`corr(Jaccard, mean|Δ|) = −0.69` — the more Hub and RawNN diverged in what they picked, the larger the
fidelity gap. Where they diverged (low Jaccard), **Hub was worse (−0.025)**, because it ranked by
hub-centroid similarity and thus picked crops *less* query-similar (mean qsim 0.849 vs RawNN 0.869) — the
wrong objective for a repair-toward-the-query task. Crucially, the audit identified that **the graph
carried no signal orthogonal to SigLIP2** (hubs were HDBSCAN clusters in the same space RawNN ranks in), so
the interesting form of the hypothesis had not been tested. Recommendation: run two cheap arms (HubNN,
OracleHub) *before* abandoning.

### Step 4 — HubNN: both mechanisms of an embedding hub are redundant with NN
A CPU-only re-analysis (exact score reuse — order-independent IP-Adapter sum, fixed seed) decomposed the
hub into filter + re-rank. **HubNN−RawNN = +0.0037** (win 2/24, **18 exact ties**, p=0.46, Jaccard 0.864):
the hub *filter* adds nothing (89% of RawNN's picks already fall inside the query's hub). **HubNN−Hub =
+0.0024** (p=0.49): the centroid *re-rank* is zero-mean, two-signed noise — not a systematic harm. Settled:
every hub mechanism built purely from SigLIP2 clustering is redundant with SigLIP2 NN, *as it must be — the
hub is a SigLIP2 neighbourhood.* Not settled: a hub built on a signal **orthogonal** to SigLIP2.

### Step 5 — OracleHub precondition: the orthogonal structure exists
Expanded to a 25-concept / 10-family taxonomy and confirmed taxonomic siblings are genuinely
embedding-distant — **87% are missed by NN**. So a taxonomy hub *can* reach crops NN cannot; the decisive
test is now runnable rather than vacuous.

### Step 6 — OracleHub: the orthogonal relation has no visual value, and hurts
Under a working, powered deficit (**rawnn−random +0.062, p=2e-4**): **oraclehub ≈ random** (−0.002, p=0.43;
on the make-or-break **far** stratum *exactly* random, +0.0004, p=0.53) and **significantly worse than
rawnn** (−0.064, p=4e-4). More siblings made it worse, not better (k_eff=1 −0.048 → k_eff=4 −0.101). The
only families where OracleHub didn't lose are the ones where taxonomy and appearance *coincide* (Arecaceae
+0.23, Lauraceae +0.02) — i.e. exactly where it adds nothing over NN. A taxonomically-correct but
visually-different crop carries no restoration value: the repair task rewards visual fidelity, IP-Adapter
transfers visual appearance, and an orthogonal relation is by construction orthogonal to that.

### Step 7 — Stage-B Gate 0: the last non-visual angle winds down
The one contribution the visual results did not touch was **attribute** transfer: does taxonomy predict a
held-out concept's *rare* attributes better than retrieval? Pre-registered, frozen protocol
(Qwen2.5-VL held-out labels + frozen bidirectional judge, 25 held-out concepts, 5 part attributes).
Result: **rare-attribute recall MMKG 0.579 < retrieval 0.684 (Δ −0.105)**, **n_rare = 19 (< 30)**, McNemar
**p = 0.50**; taxonomy uniquely recovered a rare attribute on **0 of 19** cases. Common-case recall (sanity)
0.804 vs 0.875 confirms the pipeline works and retrieval edges it there too. All three pre-registered
conditions fail. Consistent with OracleHub: a sibling's value is a *family-typical* value, which for a
specific target's *rare* attribute is systematically the wrong one. **WIND_DOWN**; Gate 1 (generation
pilot) not run.

---

## What this establishes

The MMKG-beats-retrieval claim is refuted from **both** directions and across **both** objectives:

- **Embedding-induced hub → redundant with NN** (Stage-A′, HubNN): it can only re-rank within the NN
  neighbourhood, so it reproduces NN (then it is redundant) or departs from it (then it is worse).
- **Orthogonal (taxonomy) hub → no visual value, and hurts** (OracleHub): it reaches embedding-distant
  crops, but those carry no visual-restoration information (≈ random, < NN).
- **Orthogonal relation for the non-visual (attribute) objective → no positive signal** (Stage-B Gate 0):
  a family-typical value is the wrong prior for a target's *rare* attribute.

The mechanism is simple and well-supported: the repair task rewards **visual** fidelity, IP-Adapter
conditioning transfers **visual appearance**, so the useful signal is visual (embedding) similarity — which
is exactly what plain retrieval optimises. A graph helps only insofar as it reproduces NN, in which case it
is redundant.

## The one positive finding to keep

Orthogonal to the graph question, and strongly supported: **borrowing a good *retrieved* reference helps a
lot.**

- Borrow − isolated: **+0.12** DINOv3 (p≈1e-6, Stage-A′).
- Retrieved-NN − random reference: **+0.062** (p=2e-4, OracleHub).

The leverage is in *retrieval quality* (which reference you borrow), not in graph structure. A flat
retrieval index of own-part / cross-concept crops is the defensible, cheaper design.

---

## Implications for rag-regen (the original goal, redirected)

- **Do not** build a large relational MMKG or swap rag-regen's raw-image DB for a hub/taxonomy-DB expecting
  a repair-fidelity or rare-attribute gain — there is no evidence for one and clear evidence against.
- **Do** treat rag-regen's repair path as a **retrieval-quality** problem: a flat own-part / cross-concept
  retrieval index, with effort spent on *what makes a retrieved reference good*, is where the +0.12 lives.
  This is the forward direction (the "then #2" of this line).

## Limitations and honesty notes

- Every negative was taken only after (a) confirming the deficit bit and (b) confirming the metric had
  power — the precondition/power checks are what make Steps 2 and 6 decisive rather than infra artifacts.
- **Stage-B is under-powered, not a decisive "worse" claim.** n_rare came in at 19, not the feasibility
  proxy's 42, because 20 of 25 concepts' held-out labels rest on a *single* disjoint image (accepted in the
  spec) and single-image VLM reads collapse to "not visible"/common values more often. Per pre-registration
  the observed n governs the decision; the honest statement is "no hint of the required gain," not
  "significantly worse."
- **VLM circularity (Stage-B):** the held-out label and the equivalence judge share one VLM. Residual
  VLM-dependent bias is a limitation, not assumed absent; it is mitigated (bidirectional, paired, both arms
  judged identically) but not eliminated, so it adds noise rather than a directional advantage.
- The Treevill image budget is the binding constraint for any attribute claim.

## If ever revisited

The only honest way to revive the MMKG attribute angle is a **genuinely different, non-VLM-circular
attribute ground truth** with a **denser per-concept image budget** than Treevill provides — i.e. a new
dataset (candidate: the scoped **FewMedical-XJAU**, `reports/2026-09-07-fewmedical-xjau-scope.md`), not a
re-run here. The present results make this **low priority**.

---

## Source reports (§8)

| step | report |
|---|---|
| Stage-A | `reports/2026-09-06-stage-a-result.md` |
| Stage-A′ inpaint | `reports/2026-09-07-stage-a-prime-result.md` |
| Audit | `reports/2026-09-07-stage-a-prime-audit.md` |
| HubNN | `reports/2026-09-07-stage-a-prime-hubnn.md` |
| OracleHub precondition | `reports/2026-09-07-oraclehub-precondition.md` |
| OracleHub result | `reports/2026-09-08-oraclehub-result.md` |
| Visual-repair verdict | `reports/2026-09-08-mmkg-hypothesis-verdict.md` |
| Stage-B Gate 0 | `reports/2026-09-08-stageb-gate0-result.md` |
| FewMedical-XJAU scope (if revisited) | `reports/2026-09-07-fewmedical-xjau-scope.md` |

Specs (frozen, pre-registered) under `docs/superpowers/specs/`; plans under `docs/superpowers/plans/`.
Code: `graft/transfer.py` (`select_borrowed`), `graft/oraclehub_run.py`, `graft/oraclehub_analysis.py`,
`graft/taxonomy.py`, `graft/stageb_gate0.py`, `graft/stageb_vlm.py`. Non-GPU test suite: 136 pass.
