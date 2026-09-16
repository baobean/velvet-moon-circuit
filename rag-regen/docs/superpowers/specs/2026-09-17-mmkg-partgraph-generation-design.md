# MMKG PartGraph-RAG — part-structured multimodal-KG conditioning for rare-concept generation

**Date:** 2026-09-17
**Status:** design approved (brainstorm 2026-09-16/17); not yet planned
**Target venue:** CVPR 2027 (submission ~Nov 2026 → ~7 weeks from approval)
**Working name:** PartGraph-RAG (placeholder — rename before write-up)
**Supersedes in intent:** the MMKG-as-reference-picker / MMKG-as-attribute-gate lines,
all of which returned null (see §11).

---

## 1. The claim

Text-to-image diffusion fails on rare, fine-grained concepts. Two families of fixes
exist, each blind in one axis:

- **Image-RAG** (IP-Adapter, ImageRAG, Re-Imagen, and this repo's own single-medoid
  repair) conditions on **one global reference image** — *structurally* blind: a single
  photo cannot convey which *parts* define the identity.
- **Text-KG guidance** (RAVEL, [2412.09614](https://arxiv.org/abs/2412.09614)) injects
  **compositional/relational context from a *textual* KG** — *visually* blind: it never
  grounds the fine part-level appearance.

> **Central claim.** Conditioning rare-concept generation on a **multimodal knowledge
> graph** — decomposing the concept into its parts and typed relations, grounding each
> part with a retrieved visual exemplar, and composing them via **relation-structured
> reference conditioning** — beats both single-image RAG and text-only KG guidance,
> because fine-grained identity lives in **part-level visual detail** that neither a
> single reference image nor a textual graph can carry.

The paper is a **mechanism** paper (the conditioning module is the star), proven on a
**focused eval** (PlantCLEF rare + CUB-200 standard) with a baseline matrix that
isolates "multimodal + structured" from "more pixels" and from "text structure".

### Why this is not a re-run of the null program

Every prior MMKG arm collapsed the graph to a **single medoid image** or a bag of
per-attribute flags (`mmkg_unit_size = 1.0`, `mmkg_part_types = 0.0` in every Stage-1
concept — the structure was inert). This design's contribution is precisely to make the
**graph structure drive generation**, on a substrate where that structure exists and is
dataset-grounded (§4). The prior nulls become the paper's **single-image baseline**.

## 2. Scope

**In.** (a) An enriched, dataset-grounded MMKG (§3); (b) a part-structured conditioning
mechanism over an existing FLUX repair engine (§5); (c) a baseline matrix + ablation
ladder (§6); (d) automated + VLM-judge + human eval (§7); (e) fail-fast gates (§8).

**Out.** Identity inference from the draft (the failed detector line — never revived).
A learned/trained fusion adapter as the *primary* method (kept as an optional arm only,
§5). Any accuracy claim from the parked `mmkg_store` artifact (treated as prior null).

## 3. The MMKG (built from dataset metadata, not fabricated)

The prior graph had ~2 relation types (taxonomy, part-of) and was never consumed. For
the graph to earn the name "MMKG" and to *drive* generation, it carries four relation
families, **each grounded in dataset metadata**:

| relation | source | used by |
|---|---|---|
| `part-of` (leaf/flower/fruit/bark/branch; 15 CUB parts) | PlantCLEF per-image content-type labels; CUB part annotations | part decomposition (§5) |
| `has-attribute` (per-part colour/shape/texture) | CUB 312 attributes; PlantCLEF via VLM read | conditioning + text-KG baseline |
| `spatial/structural arrangement` | dataset part geometry / canonical layout | composed-canvas placement (§5) |
| `sibling-contrast` (family/genus hard negatives) | taxonomy hubs (already built) | hard-negative-aware selection |

Storage stays as the existing FAISS + JSON-sidecar form (a valid MMKG representation;
graph-DB form is not required — cf. VisualSem, [survey 2202.05786](https://arxiv.org/html/2202.05786v2)).
Cross-modal alignment via SigLIP, encoder locked and asserted on load.

## 4. Datasets

The theme is **biodiversity / fine-grained species generally** — not birds-and-plants
specifically. The two starting substrates are chosen because they carry **native part
structure** (so the MMKG is built from labels, not hand-annotation), but if the §8 screen
kills one, **swap in another biodiversity domain rather than abandon the direction**.

- **PlantCLEF (primary, rare/hard).** Subset ~200–500 long-tail species with ≥N
  part-typed images/species. Content-type labels (leaf/flower/fruit/stem-trunk/branch/
  entire) give `part-of` for free; high-res close-ups fix the reference-quality problem
  that killed Treevill. Long-tail → genuine base-model failure.
- **CUB-200-2011 (comparability).** 200 birds, 15 part locations + 312 attributes.
  Standard fine-grained benchmark → instant reviewer familiarity. **Droppable fallback**
  if the 7-week timeline slips; PlantCLEF is the load-bearing result.

**Biodiversity fallbacks (if a primary fails the screen).** Any fine-grained taxon with
rarity + recoverable part structure: iNaturalist long-tail across taxa (insects,
reptiles, amphibians, fungi), Danish Fungi (DF20), fine-grained insect/moth sets, or
marine (FathomNet). Native part labels are ideal; where absent, `part-of` can be
VLM-derived (weaker but still structured). The bar is the same for any substrate: it must
pass §8 (base model fails on it **and** single-image retrieval is weak).

Each dataset passes the §8 rarity/headroom screen before any mechanism work.

## 5. Mechanism (Approach A — composed structured conditioning)

```
draft = text-only FLUX(prompt)                      # the single-image-RAG-free floor
sub   = mmkg.subgraph(concept)                      # parts + typed relations
crops = { p: nearest_crops(sub.part[p]) for p in sub.parts }   # 1 exemplar / part
canvas= compose(crops, sub.spatial_relations)       # relation-placed reference canvas
out   = flux_reference_guided(draft, canvas)        # ONE pass — no edit-chaining
```

- **One pass over a composed reference**, not N chained part-edits → respects this
  repo's "never chain edits" rule (chaining compounds full-frame VAE-decode drift).
- Reuses the working masked-repair engine (`ragregen/regen.py`; +0.066 cropped-DINO,
  preservation 1.000 when triggered). The generator internals are **unchanged**; the
  novelty is the MMKG→canvas conditioning upstream of it.
- **Ablation arms folded here, not competing methods:**
  - **B (multi-image fusion):** graph-weighted IP-Adapter-style embedding fusion —
    optional arm; a trained adapter is *out of scope* as the primary (timeline risk).
  - **C (text-KG / RAVEL-style):** structured *textual* graph description only — this
    is the "structure but visually blind" baseline, not a method.

**Open technical risk (gated wk2):** whether FLUX.1-Kontext honours a *composed
multi-part* reference canvas. The pilot tests this before any scale-up.

## 6. Baselines · ablations · metrics

**Baseline matrix (isolates the contribution):**

```
text-only FLUX                                   (floor)
 → single-image RAG: IP-Adapter | ImageRAG | single-medoid   (more pixels, no structure)
 → text-KG: RAVEL-style structured prompt         (structure, no vision)
 → OURS: MMKG composed structured conditioning
```

**Ablation ladder (the isolation argument — each rung must add a CI-clean gain):**
`single-medoid → +parts → +relational arrangement → +multi-instance`.

**Metrics.** Cropped-DINO identity (primary; output crop ↔ held-out GT refs), CLIP +
SigLIP (paper comparability), three-zone preservation (anti-gaming guard), held-out VLM
judge (never used in the loop), **human preference study** (labeler available).
**Encoder hygiene:** retrieve and score with different checkpoints (existing discipline).

## 7. Evaluation protocol

Pre-registered in this repo's culture: margins and decision rules frozen **before**
scoring; no post-hoc rescue; report `best` and `last-attempt` columns; contamination
guard (SHA-256 output-crop vs GT refs) on every case; per-case trace of subgraph, chosen
crops, canvas, and every score. Human study: paired A/B preference, ours vs each
baseline, on a held-out species split, labeler-blind to arm.

## 8. Fail-fast gates (the runway protection)

- **Wk 1 — dataset viability.** Build PlantCLEF+CUB MMKG from metadata. **Rarity/
  headroom screen**: ~30 species/dataset, text-only FLUX draft → does it *fail*
  (DINO + eyeball) AND is single-image SigLIP retrieval *weak*? A dataset that FLUX
  already renders well, or where retrieval is near-ceiling, is **dead on arrival**
  (the prior trap) → drop it.
- **Wk 2 — GO/NO-GO (the whole bet).** ~15-species pilot: does composed part-structure
  beat single-medoid **and** single-image RAG by a large, CI-clean cropped-DINO margin
  + a small VLM-judge run? Also validates FLUX-Kontext multi-part conditioning (§5 risk).
  **NO → pivot the mechanism now (5 weeks left), do not chase the deadline with a dead
  mechanism.**
- **Wk 3–5** full runs, both datasets, full matrix + ablations.
- **Wk 6** human study + rebuttal-proofing. **Wk 7** write-up.

## 9. Risks

- **FLUX multi-part conditioning may not work** — gated wk2; fallback is region-wise
  disjoint-mask repair (still one edit per disjoint region, not chaining).
- **PlantCLEF rarity weaker than hoped** — gated wk1; CUB or a rarer subset as fallback.
- **Two datasets in 7 weeks** — CUB is the droppable one; PlantCLEF is load-bearing.
- **"It's just retrieval with a taxonomy"** (the fatal reviewer line) — defended by the
  ablation ladder showing parts+relations add gain *over* single-image retrieval.
- **GPU instability** — runs must be resumable and machine-portable (see the handoff
  doc's infra section: hosts .200/.202/.245, GPUs die often).

## 10. Positioning / related work to beat

RAVEL (text-KG, visually blind), ImageRAG / IP-Adapter / Re-Imagen (single image),
[VAT-KG 2506.21556](https://arxiv.org/html/2506.21556) (2025 MMKG-for-RAG — closest
substrate/related work), [AspectMMKG 2308.04992](https://arxiv.org/pdf/2308.04992)
(aspect/part-aware entities). Targeted survey (wk1) confirms no prior *multimodal*-KG →
diffusion part-conditioning method before freezing the claim.

## 11. What this replaces (prior nulls — do not re-run)

Triple-confirmed across GRAFT (visual repair, rare-attribute transfer), the iNat-birds
attribute-verifier pilot, and the medoid-fidelity pilot: **MMKG-as-reference-picker and
MMKG-as-attribute-gate give no advantage over plain retrieval on single-object identity.**
That is the *boundary*, not a refutation of MMKG-for-generation. This design tests the
untested core (structure driving generation), not the settled corner (pick one image).
