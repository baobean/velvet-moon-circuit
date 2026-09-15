# MMKG-DB as repair reference — Stage 1 (in-rag-regen A/B) design

**Date:** 2026-09-05
**Status:** design, pending review
**Related:**
- GRAFT validation/de-bias sprint report (`kg_test/reports/sprint-report.md`, esp. §0 reliability-stratified re-analysis).
- rag-regen retrieved-reference phase-B design (`docs/superpowers/specs/2026-08-19-retrieved-detector-phase-b-design.md`).
- LAION common-concept benchmark (`docs/superpowers/specs/2026-08-05-laion-common-concept-benchmark-design.md`).

---

## 1. Purpose and the staged plan

The long-term goal is to replace rag-regen's raw-image retrieval DB with a **multimodal
knowledge graph (MMKG) DB**, so that repair conditions on a *structured, multi-image, visually
verified* concept representation instead of a single retrieved photo — and, ultimately, to make
the editor **customizable** (a user supplies references at inference and we edit toward them).

That end goal decomposes into two stages with different unproven risks:

- **Stage 1 (this spec) — closed-world DB replacement.** The target concept is *known* per case
  (it drove the draft prompt), so the MMKG is fetched by concept key, not by nearest-neighbour.
  This isolates the one question that must hold before anything else is worth building:
  **does an MMKG reference repair better than a single retrieved reference, in rag-regen's own
  pipeline, on rag-regen's own concepts, changing only the reference source?**
- **Stage 2 (out of scope here) — open-world customization.** User-supplied references, MMKG
  built on the fly, no fixed-DB retrieval. Deferred until Stage 1 shows the primitive pays off.

The two stages share one repair core; only *where the MMKG comes from* differs. Stage 1 does not
commit us to a general-vs-per-domain DB: the per-domain question is a **retrieval** question, and
Stage 1 has no cross-concept retrieval (keyed lookup), so it is deferred with Stage 2.

## 2. The claim to establish

> Replacing rag-regen's single retrieved reference with an MMKG (SigLIP2 medoid + visually
> verified attributes) improves repair fidelity — measured in rag-regen's existing harness, on
> its existing concepts, with only the reference source changed.

This is the GRAFT sprint's `ours > B1` result (DINOv3 ~+0.08 where measurable, reference-driven
not name-driven) transplanted from standalone generation into rag-regen's `draft → detect/mask →
retrieve → repair` pipeline. Confirming it here is what de-risks "build a big MMKG-DB."

## 3. Design: change exactly one thing

Everything rag-regen already does is **frozen**: draft generation, GroundingDINO detect +
mask, the FLUX-Kontext repair mechanism, the eval case set, and the fidelity metric. The **only**
independent variable is the reference that conditions repair.

```
   draft ─▶ detect + mask ─▶ [ REFERENCE SOURCE ] ─▶ FLUX-Kontext repair ─▶ score
                                     │
   arm A (baseline): current single retrieved image (text→image FAISS)
   arm B (ablation): MMKG medoid image only, no attribute text
   arm C (full):     MMKG medoid image + verified attribute text
```

**Three arms, on purpose.** The A→B→C decomposition localizes any effect and makes a null result
interpretable, exactly as the sprint's B1/ours arms did:

- **B − A** isolates *"a medoid over many concept images"* vs *"one retrieved image."*
- **C − B** isolates *"verified attributes add discriminative signal"* on top of the medoid.

**Retrieval is a keyed lookup, not NN.** The concept is known per case, so arm B/C fetch that
concept's MMKG by id/name. We deliberately do **not** embed the masked region and retrieve the
nearest concept: the masked region is the very thing the draft rendered wrong, so embedding-NN
would risk retrieving the wrong concept and confirming the draft's error. NN retrieval is a
Stage-2 (open-world) concern.

### 3.1 Integration surface (to be pinned down in the plan)

- **Reference production.** rag-regen builds the repair reference through its retrieval path
  (`ragregen/retrieve.py`, `ragregen/retrieved_reference.py`). Arms B/C add a parallel
  *MMKG reference source* that returns (medoid image path, attribute text) for a concept key.
- **MMKG build.** Reuse the GRAFT M1 build from `kg_test/graft/` (name-neutralized attribute
  VLM, SigLIP2 medoid selection) to produce one MMKG per eval concept. The build is a batch
  pre-step, not part of the per-case loop.
- **Repair conditioning.** The FLUX-Kontext repair already consumes a context image; arm C
  additionally injects the attribute text into the repair prompt. The exact conditioning API and
  where attribute text concatenates into the prompt are confirmed during planning.
- **Metric.** Reuse rag-regen's existing repair-fidelity scoring so the arms plug into the current
  readout; do not invent a metric.

## 4. The correctness constraint that decides validity: reference parity

The MMKG for concept *X* MUST be built from the **same corpus rag-regen retrieves from** (X's
LAION slice), **not** from a curated ground-truth reference set. Otherwise arm C wins by getting
privileged images, and the result does not transfer to the real goal (a DB built from web images).
This is the direct analog of the sprint's medoid/exemplar-parity fix.

- The existing **SHA-256 contamination guard** (`retrieved_reference.contamination`) is extended
  to the MMKG arm: any MMKG image byte-identical to a case's ground-truth reference is detected,
  never silently used.
- Arms A, B, C draw from the **same candidate pool** so the comparison isolates *structure*
  (medoid + attributes) rather than accidentally comparing different source images.

## 5. Metric and analysis — carry the reliability lesson forward

The sprint's full-corpus mean was diluted by species scored against a single held-out photo
(§0 of the sprint report). Stage 1 adopts that lesson as a first-class analysis rule:

- **Per-concept paired deltas** (C−A and B−A), never only a pooled mean.
- **Stratify by measurement reliability** — the number of eval/held-out images per concept — and
  report the flat mean, the reliability-stratified subsets, and a held-out-count-weighted mean, so
  a diluted average cannot mislead. (Method: `kg_test/scripts/stratify_heldout.py` is the template.)
- **Re-run, never splice.** Reuse the frozen doc-5 / LAION eval cases and re-run the arms; do not
  hand-edit provenance or reference scores into existing artifacts.

## 6. Scope

**In scope:**
- Build one MMKG per existing rag-regen eval concept, from that concept's retrieval corpus slice.
- Add the MMKG reference source and wire arms B and C alongside the existing arm A.
- Run the existing repair eval across the three arms; produce the stratified paired readout.

**Out of scope (deferred):**
- Stage 2 open-world / user-supplied-reference build and NN concept retrieval.
- Any production-scale or general cross-domain MMKG-DB build.
- A backbone-agnostic generator abstraction (SDXL/FLUX interface unification).
- Changes to draft, detector, mask, or the repair mechanism itself.

## 7. Success criteria and honest failure modes

- **Success:** arm C beats arm A on repair fidelity with a positive per-concept paired delta that
  holds (does not vanish) on the reliability-stratified subsets; arm B locates whether the gain is
  from the medoid, the attributes, or both.
- **Interpretable null:** if C ≈ A on well-measured concepts, the MMKG structure does not pay off
  for repair in this domain — a real finding that stops the big-DB build before it is funded.
- **Guardrails against a false positive:** reference parity + contamination guard (§4), and the
  stratified readout (§5), prevent both a privileged-images win and a dilution-masked/­outlier-driven
  verdict.

## 8. Open items for planning

- Confirm the exact FLUX-Kontext conditioning entry point and how attribute text joins the repair
  prompt (arm C).
- Confirm the repair-fidelity metric field(s) to report and the eval-image count per concept that
  defines the reliability strata.
- Confirm reuse boundary between `kg_test/graft` M1 build code and rag-regen (import vs. vendored
  batch script) given the two repos are separate.
