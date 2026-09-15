# GRAFT — Validation & De-bias Sprint (pre-benchmark)

**Status:** Design
**Date:** 2026-08-30
**Workspace:** `ndbao_hbngoc/kg_test`
**Gates:** the full Treevill benchmark in `2026-08-30-graft-phase2-design.md` §6 (that run should not
start until this sprint's outputs are in).
**Reconciles with:** Phase 2 design §5.1 (attribute specificity), §5.2 (conditioning sweep),
§5.3 (exemplar parity) — see §7 below.

---

## 1. Why this sprint exists

Phase 1 (see `reports/phase1-report.md`) produced an honest but untrustworthy pilot: GRAFT beats
the no-grounding (B0) and LLM-recall (B2) baselines by 7–12× on DINOv3 fidelity, but does **not**
beat flat single-reference retrieval (B1), on n=11 cases across 3 species. Before spending the
full-benchmark GPU budget settling the GRAFT-vs-B1 question, four issues make the current numbers
unreliable enough that the full run would inherit the same confounds:

1. **Sample-selection bias.** The eval requires a species to have **≥6 unique images** (fixed
   `k_build_refs=5` build split + a non-empty held-out split), even though generation only ever
   consumes **one** reference image. This silently restricts the benchmark to image-rich species.
2. **Concept-name confound.** The rare species *name* is injected into the generation prompt, the
   B1 retrieval query, the reranker, *and* the M1 attribute-reading VLM. For a rare concept the
   name is at best a coarse prior and at worst a wrong-meaning one (e.g. "Avocado" → the fruit).
   This makes it impossible to attribute a good result to the reference images (the thesis) rather
   than to SDXL's text encoder — and it will not transfer to the intended future use case
   (a user supplies references for a novel concept the model has no correct name for).
3. **Cross-method prompt asymmetry.** B1 and GRAFT use the *same* IP-Adapter + one reference image,
   but B1's prompt is `"According to this image, generate a photo of a {concept}"` while GRAFT's is
   `"a photo of a {concept}, {attrs}"` — GRAFT never tells the model to follow its reference. This
   confounds exactly the comparison Phase 2 wants to settle.
4. **Unvalidated part-tree.** The MMKG's per-part image crops (leaf/bark/cone-or-flower/branching)
   are produced by GroundingDINO on low-quality Treevill photos. Within the committed scope (no
   regional-attention surgery, no part-compositing) these crops **never condition the generator** —
   they are used only to gate the M3 reseed loop. Two latent correctness bugs (below) plus poor
   image quality mean this gate may be scoring noise.

This sprint fixes 1–3, audits and decides 4, and hands the Phase 2 full run a pipeline whose
result can be trusted and whose framing matches the actual research claim.

## 2. Research framing this sprint enforces

The claim under test is: **the pipeline generates fine-grained, breed-correct images of a rare
concept from its reference images** (customizable in future to any user-supplied concept).

Two consequences the design must honor:

- **Identity comes from the references, not the name.** Fine-grained breed correctness is *proven*
  by fidelity against held-out real photos of that exact species — not by the prompt token. For a
  genuinely rare concept the name cannot supply fine-grained detail (the model never learned the
  breed); relying on it only inflates scores for semi-known species in a way the customizable
  future cannot reproduce.
- **Retrieval here is exemplar *selection* within one concept's own photos, not concept retrieval
  across a corpus.** Each eval case's candidate pool is a single species' deduped images, so
  removing the name cannot cause a wrong-species retrieval; it only changes *which* of that
  species' photos is chosen.

## 3. Goals / non-goals

**Goals**
1. Remove the sample-selection floor down to the unavoidable minimum (§4).
2. Fully neutralize the concept name across generation, selection, and attribute-building, with a
   named secondary arm that *quantifies* the name's contribution (§5).
3. Give B1 and GRAFT identical, name-free exemplar selection (parity + asymmetry fix) (§6).
4. Audit the MMKG part-tree and decide by ablation whether to prune it; fix the two correctness
   bugs found regardless (§8).
5. Pull an early `ip_scale` mini-sweep forward, since neutralization makes conditioning strength
   more load-bearing (§9).

**Non-goals** (unchanged from Phase 1/2, restated to bound scope)
- No FLUX backbone work — that is Phase 2 §4; this sprint is SDXL-only.
- No full Treevill run — this sprint runs on a small validation subset only (§10).
- No regional-attention surgery, part-compositing, fine-tuning/LoRA/textual-inversion.
- No anchor→delta editing (already cut in Phase 2 §3).

## 4. Adaptive dataset split (removes the ≥6-image gate)

`dataset.split_refs` currently does `build = shuffled[:k_build]`, `heldout = shuffled[k_build:]`
with a fixed `k_build=5`, so `run_eval` drops any species with fewer than 6 unique images.

**Change:** compute an adaptive build size per species:

```
k_build = min(k_build_cap, n_unique - n_heldout_min)      # n_heldout_min = 1
```

with `k_build_cap = 5` (a config field, replaces the old fixed `k_build_refs`).

- **≥6 unique:** unchanged (build 5, rest held out) — stays comparable to Phase 1.
- **2–5 unique:** now included (build `n−1`, hold out 1) instead of dropped.
- **1 unique:** still excluded — there is no honest held-out image; `run_eval` keeps its existing
  "no held-out → skip species" guard as the only remaining floor.

**Reporting caveat (surfaced, not hidden):** small species yield a 1–2 image held-out estimate
(higher variance). The eval must record **per-species held-out count** alongside every score, and
the primary GRAFT-vs-B1 read is the **paired, per-species** analysis (Phase 2 §6.3), where each
species is its own sample and unequal held-out sizes do not bias the pairing.

**Deferred (noted, not committed):** a `k_build ∈ {1,3,5}` sweep would directly test the "only one
reference is really needed" intuition. Left out of this sprint's committed matrix to bound GPU
cost; revisit if budget is comfortable.

## 5. Full name neutralization + named arm

Replace the concept name with a fixed neutral token **`"plant"`** (Treevill spans trees, shrubs,
and aloe; `"plant"` is the safe superset) at **every** point the raw name currently leaks in:

| Site | Now | After neutralization |
|---|---|---|
| Generation prompt (all methods) | `"a photo of a {concept}[, {attrs}]"` | `"a photo of a plant[, {attrs}]"` |
| B1 retrieval query | `siglip.embed_text([concept])` | removed — selection becomes name-free (§6) |
| Reranker exemplar rank | `reranker.rank(concept, refs)` | removed — selection becomes name-free (§6) |
| M1 attribute VLM (`kg_build`) | `"...a single plant/tree species: {concept}. Describe ONLY what is visible..."` | name stripped: `"...a single plant species shown in these reference photographs. Describe ONLY what is visible..."` |
| CLIP-T metric | vs `"a photo of a {concept}"` | demoted to a **diagnostic**: "drift toward the name's common meaning" (low is good for wrong-meaning names); no longer a headline metric |

Concept signal per method after neutralization becomes a clean gradient — B0: none (true no-info
floor); B2: name-recalled attribute strings (mechanism intact); B1: the reference image; GRAFT:
reference image + **genuinely ref-derived** attributes (name no longer leaks into M1).

**Named arm (secondary run).** Re-run the matrix with the real name restored in the generation
prompt template (only), on the validation subset, to produce the paired **named − neutralized**
delta per method — i.e. how much the species name contributes, net of the references. Generation
is the expensive stage, so the named arm runs on the subset (not scaled) in this sprint.

**Config:** a `neutralize_name: bool` flag (default `true`) drives both the prompt token and the
M1 instruction so the named arm is a config toggle, not a code fork.

**Also dropped from M1:** the anchor/delta VLM pass (`ANCHOR_INSTRUCTION`) — it injects the name
too, is already slated for removal in Phase 2 §3.1, and feeds nothing consumed in this sprint's
scope. `ConceptKG.anchor`/`delta` stay as optional default-empty fields so Phase-1 `kg.json` files
still deserialize.

## 6. Name-free exemplar selection (parity + asymmetry fix)

With the name gone there is no text query for selection. Replace text-keyed retrieval/reranking
with **medoid selection**: pick the reference whose embedding is closest to the mean of that
species' reference embeddings (the most representative real photo). `kg_build` already computes the
reference mean for `concept_embeddings`, so the medoid is cheap to add.

- **B1 and GRAFT draw the *same* medoid exemplar** → exemplar parity (Phase 2 §5.3 satisfied): the
  only remaining difference between them is the graph structure.
- **Prompt scaffolds unified:** B1 = `"a photo of a plant"` + medoid ref; GRAFT = `"a photo of a
  plant, {attrs}"` + medoid ref. The `"According to this image"` asymmetry is removed.
- **Part crops now come from the medoid**, not the arbitrary `ref_images[0]` — this fixes
  correctness bug (a) in §8 as a side effect.
- **Config:** `exemplar_selection: "medoid" | "text"` (default `"medoid"`); `"text"` preserves the
  Phase-1 behavior for an optional ImageRAG-faithful B1 arm if wanted later.

## 7. Reconciliation with the Phase 2 design doc

This sprint supersedes or subsumes three Phase 2 §5 levers; the Phase 2 doc gets short pointers to
here so they are not implemented twice:

- **§5.1 attribute specificity** → still wanted, and *strengthened*: with the name stripped from
  M1 (§5), the VLM must produce discriminative attributes from the images alone. The `is_generic`
  filter and one-time re-ask from Phase 2 §5.1 are pulled into this sprint's M1 change.
- **§5.2 conditioning sweep** → the `ip_scale` mini-sweep is pulled early here (§9).
- **§5.3 exemplar parity** → satisfied by medoid selection (§6).

The Phase 2 doc's §6 (full benchmark) and §4 (FLUX backbone) remain Phase 2 work and consume this
sprint's validated config.

## 8. MMKG audit + prune ablation

### 8.1 Correctness bugs to fix regardless of the prune decision

- **(a) Part exemplars from an arbitrary reference.** `kg_build` crops every part from
  `ref_images[0]`, but exemplar ranking happens later. Fix: crop parts from the **medoid** (§6).
- **(b) Crop-vs-whole-image inconsistency.** When GroundingDINO finds no box, `kg_build` embeds the
  whole reference and `verify` embeds the whole generated image, so a part's "similarity" can be
  crop↔crop, whole↔whole, or crop↔whole depending on detection luck. Fix: make the comparison
  well-defined — a part with **no reliable box in the reference is not scored** (excluded from the
  gate) rather than silently falling back to a whole-image embedding; log the exclusion.

### 8.2 Audit (instrumentation, ~5 species, generation-light)

Measure and record, per part, on the validation subset:
- GroundingDINO **hit-rate** (fraction of references / generated images where a box is found);
- the **part-similarity distribution** (are scores clustered near the `part_sim_threshold=0.5`
  gate, i.e. is the gate doing anything?);
- how often the §8.1(b) fallback would have fired.

This is cheap (detection + embedding, no full refine budget) and either exposes the tree as noise
(→ prune with evidence) or clears it.

### 8.3 Prune ablation (the decision)

On the validation subset, run GRAFT in two configurations against B1:

| Arm | Part-tree in verify | Refine gate |
|---|---|---|
| GRAFT-full | yes | part-sim + attribute checklist |
| GRAFT-pruned | no | attribute checklist only |
| B1 | — | — (single shot) |

`use_part_tree: bool` (default `true` for the sprint — matches current behavior; the §8.3 outcome
sets its permanent value) toggles the part-crop branch in `verify`.

**Decision rule:** if GRAFT-pruned ≥ GRAFT-full on the primary fidelity metric (paired, per-species)
within noise, **prune permanently** — drop the part-crop branches from M1/M3, keep the flat
attribute list (which still feeds the prompt and the attribute checklist). This removes a
GroundingDINO dependency and directly addresses the "the cut-outs confuse things" concern with a
number rather than an assumption.

## 9. Early `ip_scale` mini-sweep

Neutralization moves *all* fine-grained identity onto the reference image, so conditioning strength
is now load-bearing. Sweep `ip_scale ∈ {0.4, 0.6, 0.8}` on the validation subset for GRAFT (and,
for parity, B1). Report per setting; the best per-metric setting is the one handed to the Phase 2
full run. (This is Phase 2 §5.2, executed early and small.)

## 10. Execution plan

**Validation subset.** Select by **deduped unique-image count** (not alphabetical): the top ~6–8
species by unique images that satisfy the adaptive split, spanning a range of unique-image counts
(so the small-species held-out-variance behavior is observed too). This replaces Phase 1's
alphabetical selection (Phase 2 §6.1 hardening, pulled early).

**Order (each stage gates the next):**
1. Land the code changes: adaptive split (§4), neutralization flag + token + M1 instruction (§5),
   medoid selection (§6), the two correctness fixes (§8.1), the `use_part_tree` and
   `neutralize_name`/`exemplar_selection`/`ip_scale` config knobs.
2. Run the **audit** (§8.2) on the subset.
3. Run the sprint matrix on the subset — methods `{B0, B1, B2, GRAFT-full, GRAFT-pruned}` ×
   `{neutralized, named}` × `ip_scale ∈ {0.4,0.6,0.8}` (GRAFT/B1 for the sweep; B0/B2 once) —
   staged and resumable on the shared 4090 via the existing `gpu_queue.sh` + per-case subprocess
   pattern.
4. Read out: prune decision (§8.3), best `ip_scale` (§9), name-contribution delta (§5), and a
   trustworthy GRAFT-vs-B1 paired result on the subset.
5. Hand the resulting validated config to Phase 2 §6.

**Execution model unchanged:** every GPU-touching phase runs in its own disposable subprocess
(the Phase-1 architecture — kept verbatim). The Phase 2 §6.1 refine-crash-tolerance fix (retry the
next seed on a worker crash the way OOM is already tolerated) is pulled in here so the subset run
does not lose cases.

## 11. Metrics & how we decide

- **Primary:** DINOv3 / SigLIP2 / CLIP-I fidelity vs held-out real photos (unchanged); reported as
  **paired per-species** GRAFT-vs-B1 deltas + win-rate.
- **Secondary:** attribute accuracy (VLM checklist).
- **Diagnostic:** CLIP-T-vs-name (drift toward common meaning); GroundingDINO hit-rates;
  part-similarity distribution; per-species held-out count.
- **Readouts this sprint must produce:** (1) prune yes/no with evidence; (2) best `ip_scale`;
  (3) named − neutralized delta per method; (4) a GRAFT-vs-B1 paired result free of the name and
  exemplar-selection confounds.

## 12. Feasibility

All models and the subprocess-isolated, resumable execution are proven in Phase 1. Every change
here is a prompt/instruction edit, a selection-rule swap (medoid), a dataset-split formula, a
verify-gate toggle, or added instrumentation — no new checkpoints, no new heavy dependency.

## 13. Open items

- **Neutral token** fixed at `"plant"`; revisit only if the audit shows SDXL treats it as a strong
  shape prior that biases all methods equally-but-materially.
- **`part_sim_threshold` calibration** — the audit's part-sim distribution (§8.2) may show the 0.5
  gate is mis-set; recalibrate before reading the prune ablation if so.
- **Named-arm scope** — subset only in this sprint; whether the full Phase 2 run also carries a
  named arm is a Phase 2 budget decision, not settled here.
