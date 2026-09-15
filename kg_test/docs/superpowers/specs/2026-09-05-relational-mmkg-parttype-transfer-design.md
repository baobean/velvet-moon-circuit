# Relational MMKG — shared-hub knowledge transfer (design)

**Date:** 2026-09-05
**Status:** design, **LOCKED for Stage A** (2026-09-05)
**Supersedes framing of:** the isolated per-concept `ConceptKG` as "the MMKG" (it is retained as the
**baseline representation**, not the graph).
**Related:**
- GRAFT validation/de-bias sprint report (`reports/sprint-report.md`), esp. §0 reliability-stratified
  re-analysis (the modest full-corpus effect this design is meant to lift) and §5.4 (the degenerate
  part-tree ablation — the "decorative structure" failure this design must not repeat).
- Current data model: `graft/schema.py` (`ConceptKG`, `PartNode`, `AttributeNode`), build in
  `graft/kg_build.py`, part crops via GroundingDINO, medoid via `graft/selection.py`.
- Strategic direction memo (paused, downstream): `../../../rag-regen/docs/superpowers/specs/2026-09-05-mmkg-db-repair-reference-stage1-design.md`.

---

## 1. Motivation

Two facts from the sprint define the problem:

1. **The effect is real but modest** (§0): GRAFT beats flat single-reference retrieval (B1) by ~+0.08
   DINOv3 where fidelity is measurable, but the flat full-corpus mean is small and diluted by
   data-poor concepts.
2. **The current "graph" is decorative** (§5.4): each `ConceptKG` is an isolated per-concept record,
   flattened to a text bag + a medoid at consumption. Its part-tree ablated to zero (`ours ==
   ours_notree`). It provides **no shared knowledge and no graph reasoning**.

The strongest way to lift a modest result *and* make the "knowledge graph" claim real is the same
move: **let data-poor concepts inherit visual evidence from other concepts that share the same part
type, through shared typed hub entities.** The graph's schema does the work, and it does it exactly
where the flat baseline is weakest.

`ConceptKG` remains valuable as the **initial multimodal concept representation** — the per-concept
substrate the graph is built over. This design adds the relational layer above it.

## 2. Hypothesis / core contribution

Stated as a hypothesis about **shared entities enabling transfer**, deliberately *not* as a claim that
graph structure per se creates the benefit:

> **Shared PartType entities enable useful knowledge transfer from data-rich concepts to data-poor
> concepts** — improving rare-concept fidelity where an isolated per-concept representation cannot.

**The locked question the Stage-A implementation must answer:**

> *Does shared PartType-based transfer improve fidelity for data-poor concepts (a) beyond isolated
> per-concept representations, AND (b) beyond simply adding more reference images?*

Both "beyond" bars are load-bearing. (a) rules out that the per-concept substrate already sufficed;
(b) rules out that any gain is just exemplar quantity. A "yes" to both is the basis for Stage B and a
richer MMKG; a "no" to (b) in particular means the construct has collapsed into plain retrieval (see
§7.2 and the honest null in §10).

## 3. Governing design principle (do not repeat the decorative-graph failure)

**Every edge type MUST have an inference-time consumer that traverses it and changes the generated
output.** The part-tree ablated to zero because its topology was never traversed. Here, transfer is
*defined* as graph traversal through hubs, so the ablation "graph-transfer vs isolated-schema" is
non-degenerate by construction. Any relation we cannot show is traversed and consumed does not go in.

## 4. Graph schema

### 4.1 Entities (nodes) — all multimodal where marked (M)

| Entity | Description | Multimodal payload |
|---|---|---|
| **Concept** | A species (today's `ConceptKG`). | (M) reference bank `ref_paths`/`ref_embeddings`, medoid. |
| **PartInstance** | *This* concept's instance of a part (≈ today's `PartNode`). | (M) `exemplar_crop`, part embedding. |
| **PartType** | A **shared hub** many concepts' parts belong to (e.g. "lanceolate compound leaf"). | (M) exemplars aggregated across member instances, centroid embedding, optional VLM label. |
| **AttributeValue** *(Stage B)* | A canonicalized shared value ("serrated margin", "peeling bark"). | (M, optional) visual exemplars of the property. |
| **Taxon** *(oracle only)* | Genus/family. | — |

### 4.2 Typed relations (edges)

```
Concept      ─has_part→       PartInstance
PartInstance ─instance_of→    PartType          ← the transfer highway
PartInstance ─has_attribute→  AttributeValue    (Stage B; Stage A keeps attrs on the instance)
PartType     ─typically_has→  AttributeValue    (Stage B; aggregated visual prior)
Concept      ─is_a→           Taxon             (oracle only, never in the method)
```

This is a **multipartite, typed, multimodal graph**: concepts are connected *through shared hub
entities*, not to each other directly, and the predicates are semantic (`instance_of`, `has_part`,
`typically_has`), not similarity. That is what distinguishes it from a k-NN proximity graph.

## 5. Construction (visual/name-free backbone; VLM/taxonomy secondary)

1. Build each per-concept `ConceptKG` as today (baseline substrate).
2. Collect all **PartInstances of a given part category** (leaf, bark, branching, flower — the audit
   showed leaf/bark/branching detect at ~100%) across **all** concepts.
3. **Cluster** their part-crop embeddings → **PartType hubs**. This is the primary, **name-free**
   construction (visual). Algorithm/granularity is an open planning item (§10).
4. Set each PartInstance `instance_of` its cluster.
5. **(Hybrid) VLM blind-describes** each cluster's exemplars → a semantic type label. Used for
   typing/interpretability and canonicalization, **not** as the clustering source of truth.
6. **Stage B:** canonicalize `AttributeNode` values into shared `AttributeValue` nodes (string
   normalization + optional embedding merge of synonyms); compute each PartType's `typically_has`
   priors by aggregating attributes over its member instances.
7. **Oracle:** attach `Taxon` edges from external botanical taxonomy — **ablation upper-bound only**.

### 5.1 Validating the induced hubs (the central construct-validity risk)

If a PartType is *only* an embedding cluster, the whole construct risks collapsing into "visual
clustering + retrieval" with no meaningful KG. We therefore commit up front to **validating that the
induced hubs are coherent and useful**, and treat that validation as a gate, not an afterthought:

- **Membership source of truth = visual, full stop.** A PartType's membership is determined **entirely
  by SigLIP2 embeddings + HDBSCAN** (P1). **No LLM/VLM is involved in forming, splitting, merging, or
  assigning hub membership.** The VLM runs **only afterward**, on an *already-formed* hub, to attach a
  human-readable **label** — and that label is **non-functional in Stage A**: it is never read by the
  transfer consumer (§6), never affects generation, and never feeds back into clustering. It is
  interpretability/diagnostic metadata only. So even a wrong label cannot change any result. The locked
  labeling setup is **P10**.
- **Coherence (intrinsic).** Report per-hub coherence (e.g. intra-hub vs inter-hub embedding
  similarity, silhouette) and **purity against the taxonomy oracle** (do members tend to share
  genus/family?). Incoherent hubs are flagged and **coherence-gated out of transfer** (§6).
- **Usefulness (extrinsic, the one that matters).** Coherence alone is not the claim — a hub must help
  *transfer*. The §7 controls are what establish this: a hub earns its name only if transferring from
  it beats quantity-matched non-hub references (§7.2). Intrinsic coherence without extrinsic usefulness
  is reported as a negative.

## 6. Transfer mechanism (the consumer)

For each (Concept, part) that is **data-poor** for that part (own evidence thin — few unique crops,
missing detection, or only generic attributes; concrete trigger τ is a planning item):

1. **Traverse:** `PartInstance ─instance_of→ PartType ←instance_of─ {sibling PartInstances}`.
2. **Rank** siblings by hub-centroid proximity / edge weight / hub coherence.
3. **Inherit (Stage A):** select `k` borrowed crops from the hub (§11 P3) and condition IP-Adapter on a
   **weighted average of image embeddings** over {own crops + borrowed crops} (§11 P4–P5, P8),
   replacing single-medoid conditioning. The borrowed crops are the hub's cross-concept prototype, not
   the concept's own nearest neighbours — that is what separates this from `+RawNN-k`.
4. **Inherit (Stage B):** fill only **missing or generic** attributes from `PartType.typically_has`,
   tagged `source:"inherited"` for provenance.
5. **Guard (identity preservation):** own crops carry **2× the per-crop weight** of borrowed crops and
   the borrowed **aggregate fraction is capped at 0.7** (≥0.3 own-identity anchor, §11 P8), so the
   aggregate lets borrowed dominate exactly when the concept is starved yet never fully detaches
   identity toward the type prototype. In the *experiment* transfer is forced on at every starvation
   level to trace the recovery curve; the deployment trigger τ (§11 P7) is a separate policy.

Downstream generation/repair is **unchanged** — only the concept's evidence changed. This keeps the
causal attribution clean and reuses the entire validated pipeline.

## 7. Evaluation — reliable by construction

### 7.1 Primary: controlled starvation/recovery
On the **8 concepts with held-out ≥ 4** (the pilot set: Bamboo, Ashok, Egyptian lotus, Nageshore,
Avocado, Camphor Tree, Hijol, Ashore — median held-out 5.5, all with build = 5), artificially
**starve the build set** to sizes **{1, 2, 3, 5}** (5 = un-starved ceiling), with **3 seed-controlled
keep-draws** per level to average which-image variance, holding each concept's large held-out set fixed
for reliable measurement. At every level, run all §7.2 conditions with **transfer forced on**, so we
trace the full recovery curve without the circularity of τ-gating (τ is a deployment policy, not part
of the experiment). **Predicted signature:** at build = 1 (max data-poverty) `+Hub-k` **recovers**
toward the build = 5 ceiling and beats `+RawNN-k`; at build ≥ 3 the curves converge and transfer does
**not hurt** (identity guard holds). This sidesteps the single-held-out-photo noise that muddied the
full-corpus mean.

### 7.2 Conditions — isolating the graph from mere exemplar quantity

The locked question (§2) has two "beyond" bars, so the conditions are built to answer both. Every
condition that adds references uses the **same added-reference budget k** (quantity-matched), so only
the *selection* differs:

| condition | added references | isolates |
|---|---|---|
| **Isolated** (`ConceptKG`, no added refs) | none | the per-concept substrate — answers bar (a) |
| **+Random-k** | k part-crops drawn at random from *other* concepts | pure quantity, wrong selection (floor) |
| **+RawNN-k** | k nearest part-crops by raw embedding to the concept's own part, **no hub structure** | plain k-NN **retrieval** — the graph-isolating control |
| **+Hub-k** *(ours)* | k crops via the PartType hub (coherence-gated, from the hub's prototype/aggregate) | shared-hub selection |
| **+OracleHub-k** | k crops via taxonomy-defined hubs | upper bound |

The comparisons that decide the thesis:

- **+Hub-k > Isolated** → transfer helps at all, and (per §7.1) the gain **concentrates on data-poor
  concepts**; the graph-vs-isolated ablation is **non-degenerate** (the §5.4 test that failed).
- **+Hub-k > +Random-k** → *selection* matters, not just quantity (bar b, weak form).
- **+Hub-k > +RawNN-k** → **the hub adds value beyond nearest-neighbour retrieval** (bar b, strong
  form). *This is the make-or-break comparison.* If +Hub-k ≈ +RawNN-k, the construct has collapsed into
  retrieval and there is no "graph" contribution — reported honestly as the null in §10.
- **+Hub-k vs +OracleHub-k** → how much of the oracle ceiling the name-free visual hubs recover.

*(Stage B adds one comparison: Stage A vs Stage A+AttributeValue — the marginal value of the attribute
layer — run only after Stage A clears §10.)*

**Selection-only invariant (enforced in code).** `+Hub-k`, `+RawNN-k`, and `+Random-k` MUST share **one
consumer code path** — identical borrowed-crop count (`k`, P4), identical weighting/cap (P8), identical
IP-Adapter conditioning (P5). The *only* input that varies between them is the **selection function**
that returns the `k` crop paths (hub-centroid vs raw-NN vs random). This is a correctness constraint: if
the consumer differed, the comparison would confound selection with the consumer and could not isolate
the hub mechanism. The plan must structure the selectors as interchangeable functions behind a single
`transfer(own_crops, selector) → conditioning` call, with a test asserting the three arms differ only in
the returned crop set.

What must the hub provide to beat +RawNN-k (i.e. to *not* be retrieval)? Cross-concept **aggregation**
(a denser, cleaner prototype than one concept's local neighbourhood), **coherence-gating** (no transfer
from incoherent hubs), and **stability** when the concept's own crop is atypical. If none of these
register, that is itself the finding.

### 7.3 External validity & discipline
Naturally-thin concepts and vs-B1, read with the sprint §0 discipline: **per-concept paired deltas,
stratified by held-out count, plus a held-out-weighted mean** (`scripts/stratify_heldout.py` template).
Fidelity metrics unchanged (DINOv3/SigLIP2/CLIP-I) + attribute accuracy. Report the hub-quality
diagnostics (coherence, purity-vs-taxonomy) defined in §5.1 alongside these, so intrinsic hub quality
can be read against extrinsic transfer usefulness.

### 7.4 Parity / contamination guard
Borrowed exemplars must **never** include the target concept's own held-out images (that would leak the
test set through a hub). Extend the sprint's exemplar-parity + SHA guard to inherited crops.

## 8. Staging

- **Stage A — PartType hubs + exemplar transfer.** Entities: Concept, PartInstance, PartType. Edges:
  `has_part`, `instance_of`. Transfer: exemplar inheritance. Eval: §7.1 starvation + graph-vs-isolated
  ablation. **Gate:** does hub exemplar transfer recover starved fidelity? If no → the thesis fails
  cleanly; stop before Stage B.
- **Stage B — add AttributeValue hubs.** Entities: +AttributeValue. Edges: +`has_attribute`,
  +`typically_has`. Transfer: +attribute-prior inheritance. Eval: re-run §7 + the Stage-A-vs-B ablation.

## 9. Scope

**In:** graph construction (PartType hubs; Stage B AttributeValue), guarded hub transfer, and the §7
evaluation, on Treevill in `kg_test`, extending (not replacing) `ConceptKG`.

**Out / deferred:**
- Open-world customization (build a concept's graph attachment from user refs at inference).
- Relational / nearest-neighbour retrieval from a large multi-concept DB.
- rag-regen integration — the paused Stage-1 A/B spec is the **downstream application**, resumed only
  after the graph proves itself here.
- Multi-hop reasoning beyond concept → part → type → sibling.
- Taxon edges in the method (oracle-only).

## 10. Success criteria & honest failure modes

- **Success (both bars of the §2 question must clear):** hub transfer recovers a substantial fraction
  of starvation-induced fidelity loss on starved concepts, with the gain **concentrated on data-poor
  concepts**, AND **+Hub-k > Isolated** (beyond the per-concept substrate, bar a) AND **+Hub-k >
  +RawNN-k** (beyond mere retrieval / more references, bar b). Ideally the name-free visual hubs also
  recover most of the taxonomy-oracle gain. Only then does Stage B (AttributeValue) proceed.
- **Interpretable nulls (each a clean, publishable negative that stops before Stage B):**
  - *No transfer:* hub transfer does not recover starved fidelity, or is not concentrated on data-poor
    concepts → the shared-hub thesis is wrong.
  - *Collapse into retrieval:* +Hub-k ≈ +RawNN-k → the hubs add nothing beyond k-NN; the construct is
    retrieval, not a knowledge graph. This is the failure mode the whole controls design exists to
    detect, and we report it rather than dress retrieval up as a graph.
- **Guardrails against a false positive:** the identity guard (§6, step 5), the contamination/parity
  guard (§7.4), the **quantity-matched controls** (§7.2), and the reliability-stratified readout
  (§7.3).

## 11. Locked Stage-A parameters

Each default is fixed for reproducibility; the rationale is Treevill-specific.

**P1 — PartType clustering & granularity.** Per part category, pool **all** part-crops from **every
build reference** of **every concept** (not just medoid crops), embed with **SigLIP2**, L2-normalize,
and cluster with **HDBSCAN** (`min_cluster_size=3`, `min_samples=1`, euclidean ≈ cosine on normalized
vectors). Granularity is **data-driven** (no fixed `K_hub`). *Why:* we don't know the true number of
part-types a priori, so a density method that picks its own granularity and marks outliers as noise is
right; SigLIP2 (**not** the DINOv3 eval metric) avoids clustering in the metric's own space; pooling
all build crops (not medoids) gives enough points to cluster and richer hub prototypes.

**P2 — Hub validity & coherence.** A hub is a valid transfer source only if it (i) contains
PartInstances from **≥ 2 distinct concepts** and (ii) has **mean member-to-centroid cosine ≥ 0.50**.
HDBSCAN-noise crops and single-concept clusters are excluded as sources. *Why:* a "shared" type must
span concepts to be a transfer highway; the sprint audit put matching-part sims ≈ 0.72 median, so 0.50
is a permissive floor that rejects grab-bag hubs while keeping real visual types.

**P3 — Hub prototype selection.** From a valid hub, take the `k` crops **nearest the hub centroid**,
with **≤ 2 crops per contributing concept**. *Why:* centroid-nearest = the type's canonical appearance,
robust to the query concept's own crop being atypical/starved (the data-poor regime); the per-concept
cap forces the **cross-concept aggregation** that distinguishes `+Hub-k` from `+RawNN-k` (which can
return `k` near-duplicates from one neighbour).

**P4 — Added-reference budget.** **k = 4**, identical across `+Random-k` / `+RawNN-k` / `+Hub-k` /
`+OracleHub-k`. *Why:* it lifts a starved 1-crop concept to ~5 effective references — the natural
`k_build = 5` data-rich ceiling — so "recovery" targets the real reference count; matched `k` makes the
controls exactly quantity-controlled.

**P5 — Consumer.** Condition IP-Adapter on a **weighted average of image embeddings** over {own crops +
k borrowed}, replacing single-medoid conditioning; generation is otherwise unchanged. `+RawNN-k` query
= the concept's own part crop, pool = all other concepts' crops of that part category **excluding
held-out** (§7.4); `+Random-k` samples that same pool uniformly. *Why:* an embedding average makes `k`
a real, exactly-matched budget and keeps the rest of the validated pipeline intact.

**P6 — Starvation protocol.** 8 concepts (held-out ≥ 4); build levels **{1, 2, 3, 5}**; **3
seed-controlled keep-draws** per level; held-out fixed; transfer forced on at every level (§7.1). *Why:*
reliable per-concept measurement, a full 1→5 recovery curve, which-image noise averaged, no τ
circularity.

**P7 — Data-poor trigger τ (deployment policy, not the experiment).** Fire transfer for a (concept,
part) when it has **< 3** own reliable crops for that part. *Why:* ≥ 3 crops gives a reasonably stable
medoid; below that, 1–2 possibly-atypical crops dominate and transfer helps. The experiment forces
transfer on regardless, precisely to verify τ's boundary (help at build 1–2, no harm at 3+).

**P8 — Borrowed weighting & cap.** Per-crop weights **own : borrowed = 2 : 1**, normalized, with the
borrowed **aggregate fraction capped at 0.7**. So build=1 + k=4 → own ≈ 0.33 (borrowed carries the
starved case), build=3 + k=4 → own ≈ 0.6 (own dominates as data returns), and the cap keeps ≥ 0.3
identity anchor always. *Why:* each real observation is worth twice a borrowed one (identity anchored to
evidence), while the aggregate shifts correctly with data availability; the cap prevents homogenization
toward the type prototype.

**Persistence (P9).** Extend the on-disk graph as a **separate store** (`PartInstance` / `PartType`
records + edges) that references existing `ConceptKG` JSON by concept id, rather than bloating
`ConceptKG`; the per-concept files stay the untouched baseline substrate.

**P10 — Hub labeling (interpretability-only; does NOT touch membership).** After hubs are formed by
SigLIP2+HDBSCAN, label each with **Qwen2.5-VL-7B-Instruct** (`GraftConfig.vlm_id`, the same VLM as
`kg_build`), via `models.vlm.describe(...)`, with **deterministic decoding passed explicitly** — not
inherited from model/method defaults. The two decoding parameters are **first-class, locked
`GraftConfig` fields** so they are serialized into every run's `cfg.yaml`:
`hub_label_do_sample = False` and `hub_label_max_new_tokens = 512`. The labeling call must pass these
explicitly (`models.vlm.describe(crops, instruction, do_sample=cfg.hub_label_do_sample,
max_new_tokens=cfg.hub_label_max_new_tokens)`) so a change to any default cannot silently alter the
setup. Labeling operates on the hub's **top-4 centroid-nearest member crops** (P3 prototype,
unweighted), with a **blind, name-free instruction** (no concept/species names):
`"These are cropped photos of the same plant part from different plants. In 3–8 words, name the part
type by its visible shape/texture/arrangement. Reply with only the phrase."` The returned phrase is
stored as `PartTypeRec.label` and used **only** for reporting/inspection. Labeling is a **separate,
skippable step** run after graph construction; skipping it leaves `label=""` and changes no result.

*Deferred to Stage B (not needed to answer the locked question):* `AttributeValue` canonicalization
(string-normalize vs embedding-merge) and `typically_has` aggregation.
