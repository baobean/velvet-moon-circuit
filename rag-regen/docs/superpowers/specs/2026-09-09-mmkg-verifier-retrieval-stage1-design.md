# MMKG in rag-regen — Stage 1: attribute verifier + organized retrieval (design)

**Date:** 2026-09-09
**Status:** design, pending review. Author sign-off pending.
**Repos:** rag-regen (this repo) + reuse from `kg_test/graft` (separate repo).

**Supersedes the claim of** `docs/superpowers/specs/2026-09-05-mmkg-db-repair-reference-stage1-design.md`.
That paused spec's headline — *an MMKG reference conditions repair to **better fidelity** than a single
retrieved image* — is now **refuted** by the GRAFT line
(`kg_test/reports/2026-09-08-graft-mmkg-final-synthesis.md`): a relational MMKG gives no fidelity advantage
over NN retrieval, and an orthogonal relation hurts. This spec **keeps that spec's infrastructure**
(closed-world by-key build, reference parity, SHA-256 contamination guard, per-concept reliability-stratified
analysis, re-run-not-splice) but **changes the claim** to two roles that our results do *not* touch and that
rag-regen actually needs.

---

## 0. Why this, and why now

rag-regen's verifier line is a documented dead end. v1/v2/v3 selectors all failed the 0.75 sign-accuracy bar
(v3's semantic gate scored 0.49, MCC −0.20 — anti-correlated); the retrieved-reference detector failed a fresh
48-case holdout at **recall 0.50 / FP 0.167**; the prototype verifier added FPs. Production baseline is
`--verifier none` (edit every maskable case). The Aug-20 weekly report names the open blocker exactly:

> **"The remaining blocker is generalizable fine-grained identity detection."** The semantic verifier checks
> broad prompt adherence, so it misses half the true identity failures and false-positives on valid controls.

An MMKG that stores a concept's **fine-grained visual attributes** is a candidate signal for precisely that
blocker — it checks *whether the specific attributes are right*, not merely *whether the concept is present*.
This spec tests that, honestly, on rag-regen's own frozen holdout, under the methodological guardrails the
prior failures earned.

**Two guardrails, non-negotiable (both learned the hard way):**
1. **No DINO-delta-sign as ground truth.** Using it as the selector yardstick is what made v3 look
   anti-correlated; the v3 finding forbids it twice.
2. **No VLM-vs-VLM ground-truth loop.** GRAFT Stage-B (`kg_test/reports/2026-09-08-stageb-gate0-result.md`)
   showed a VLM-label vs VLM-judge protocol is circular. Here the **detector is a VLM; the ground truth is the
   human holdout labels.** That asymmetry is what makes the evaluation valid.

## 1. Two roles, one graph, one trace

One MMKG per eval concept feeds two consumers, and **one comprehensive per-case trace** feeds all evaluation:

- **Role 1 — organized retrieval (measurable intervention, §5).** A distinct candidate-formation policy
  (MMKG-organized) run *alongside* the current flat FAISS retrieval on the same cases. NN still ranks. We log
  both arms' candidate pools, NN scores, and selected references, so the two are directly comparable and Role 1
  cannot silently collapse into "the same retrieval, unchanged."
- **Role 2 — attribute verifier (score-only, §6).** Reads the concept's attributes off each logged image and
  emits a **frozen, pre-registered case-level identity verdict**, scored against the human holdout labels as a
  fine-grained identity **detector**. It does **not** gate regeneration in Phase 1.

**Execution strategy (the speedup).** Build the integrated path once; run **one batch**; write everything each
component needs into the trace; then evaluate every component **offline** over the trace — no per-component
re-run, no per-component GPU. Phase 1 needs **no new image generation**: Role 2 reads the holdout's existing
drafts, Role 1 compares retrieval offline. The only new compute is (a) the one-time MMKG build over the 48
concepts' LAION slices and (b) **one** VLM attribute-read pass over the logged images.

**Discipline preserved.** Logging everything is not licence to pick a metric after seeing outputs. Every
component's metrics and gates below are **frozen in this spec, before any of the 48 cases is scored.** The
trace is written first; analysis code is run against frozen gates second.

## 2. Frozen shared definitions

- **Eval set:** the frozen Phase-B holdout, 48 cases = 24 rare identity-FAIL + 24 control identity-PASS,
  human-labelled. Sources on disk: `outputs/phase_b/holdout.json`, `outputs/phase_b/labels_holdout48.csv`,
  `outputs/phase_b/retrieval.json`, `outputs/phase_b/retrieved_reranker.json`, `outputs/phase_b/dino_reserves.json`.
  The label is **case-level identity FAIL/PASS on the draft** — the exact ground truth the detector was scored on.
- **Reference parity (correctness constraint).** Each concept's MMKG is built **only** from that concept's own
  LAION retrieval slice — the images rag-regen itself retrieves — never a curated ground-truth set. Otherwise
  the verifier wins on privileged images and nothing transfers. The existing SHA-256 contamination guard
  (`retrieved_reference.contamination`) is applied: any MMKG image byte-identical to a case's ground-truth
  reference is detected and excluded, never silently used.
- **`norm(v)` = `str(v).strip().lower()`. NOT_VISIBLE = `{"not visible","none","n/a",""}`.**
- **VLM:** Qwen2.5-VL (rag-regen's `ragregen.vlm` / reuse `kg_test/graft` read path), `do_sample=False`,
  deterministic decoding, name-blind prompts (the concept name is never shown to the VLM, so it cannot pattern-
  match a label instead of reading the image).
- **Judge (frozen, from Stage-B):** text-only equivalence judge, `do_sample=False`, evaluated in **both orders**;
  two values **match** iff both orders answer "yes"; otherwise **not-match**. Prompt frozen in §6.4.

## 3. The trace (the central artifact)

`outputs/mmkg_stage1/trace/<case_id>.json`, one record per holdout case, written by the single batch. It MUST be
complete enough that Roles 1 and 2 are each a pure offline pass — under-logging forces a re-run, which defeats
the strategy. Fields:

```
case_id, concept, coarse_type, cohort (rare|control), human_label (FAIL|PASS)

draft: { image_path, prompt }

retrieval:
  flat:  { query, candidate_pool: [{path, nn_score, rank}], selected_ref: {path, nn_score} }   # current FAISS
  mmkg:  { query, candidate_pool: [{path, nn_score, rank, part_type?, attr_tags}],
           selected_ref | selected_set: [...], selection_rule }                                  # Role 1 arm
  pool_overlap: { jaccard, n_flat, n_mmkg, selected_ref_identical: bool }                        # precomputed convenience

mmkg_concept:
  built_from_slice: [image_path...], slice_size, contamination_excluded: [path...]
  target_attributes: { slot: { value, support, visible_count, is_target: bool } }                # §4 consensus
  n_target_attributes

attribute_reads:                                                                                 # §6 VLM read pass
  draft: { slot: read_value }                                                                     # name-blind read of the draft
  # (attempts[] left as an optional list for Phase 2; empty in Phase 1)

verifier:                                                                                         # §6 offline, from reads+targets
  per_attribute: { slot: { target, read, present: bool, match: bool, contradicted: bool } }
  n_present, n_contradicted, decidable: bool, verdict: FAIL|PASS|ABSTAIN

scores_reused: { reranker_text_relevance, reranker_reference_relevance, semantic_ok, dino_reserve }  # from frozen holdout, for cross-tab only
provenance: { sha256 of each reused source, vlm_id, build_config_hash }
```

## 4. The MMKG build (Role 1 data + concept-general attributes)

A batch pre-step, one MMKG per holdout concept, from that concept's LAION slice (reuse the GRAFT M1 build
path: image embedding, dedup, name-blind VLM attribute read).

**Concept-general attribute schema (frozen).** GRAFT's ontology is tree-specific (leaf/bark/branching); the 48
holdout concepts are diverse ImageNet classes, so we use a small **fixed set of generic visual attribute slots**
the VLM fills for *any* main subject. The slot list is frozen here and not changed after seeing results:

> **SLOTS = { `primary_color`, `secondary_color`, `pattern_or_markings`, `surface_texture`,
> `overall_shape_or_form`, `distinctive_feature` }**

Name-blind, per slice image: the VLM returns `{slot: value}` for the visible main subject (`do_sample=False`).

**Target-attribute consensus (frozen).** For concept C and slot S, over the slice images where S is non-missing
(`norm(value) ∉ NOT_VISIBLE`), let `V_vis(S)` = that count. S is a **target attribute** of C iff:
- `V_vis(S) ≥ 3`, **and**
- a strict-plurality normalized value has count `≥ max(2, ceil(0.5 · V_vis(S)))`.

The **target value** is that plurality value. A concept is **decidable** iff it has `n_target_attributes ≥ 2`;
concepts with fewer are recorded and their cases ABSTAIN (§6.3). All qualifying slots are used (≤ 6).

## 5. Role 1 — the measurable retrieval intervention

**The intervention must be a genuinely different candidate-formation policy, or it is untestable.** The two arms
run on the **same cases, same query, same encoder**, differing only in the candidate pool and the selection unit:

- **Flat arm (baseline, unchanged):** `retrieve.reference_query(concept, coarse)` → text→image FAISS over the
  whole LAION index → top-k hits → `selected_ref` = top-1. This is current production behavior verbatim.
- **MMKG-organized arm:** the candidate pool is the concept's **MMKG part/attribute-tagged instance store** —
  the build's deduped, quality-filtered, part-typed instances derived from the *same* slice (parity §2), **not**
  the raw FAISS hit list. Within that pool, NN ranks by the same query embedding; selection is a
  **coverage-aware rule** (frozen): pick the per-slot medoid set spanning the concept's target attribute slots,
  and report its NN-top-1 as the comparable `selected_ref`. So the arms differ in **(a) candidate pool** (verified
  structured instances vs raw hits) and **(b) selection unit** (attribute-spanning set vs single top-1).

**What is logged (trace §3):** both pools with per-item NN scores and ranks, both selected references, pool
Jaccard, and whether the selected reference is identical across arms.

**Frozen offline metrics (Role 1):**
- **Non-triviality (the testability check):** `selected_ref_identical` rate and mean pool Jaccard across the 48
  cases. If the arms select the same reference on nearly every case, Role 1 is a no-op — reported as such
  (an honest negative), not hidden.
- **Structural coverage:** (i) reference-unit size — flat is always 1 image; MMKG is a set of N instances;
  (ii) distinct part-types represented in the MMKG unit where the concept is part-structured (else 0/NA);
  (iii) whether the flat arm's top-1 image is present in the MMKG pool at all. Together these quantify what
  organized retrieval *is* relative to flat top-1 (a medoid-over-many, optionally part-typed, structured
  reference vs a single hit) — the thing flat retrieval structurally cannot express.
- **Parity guardrail (stated, deferred):** whether the divergent reference changes repair **fidelity** requires
  generation and is **out of Phase-1 offline scope**; our GRAFT result predicts parity, not gain. If run later,
  it is a batch extension (both arms generated), never a Phase-1 gate. Role 1's Phase-1 claim is **structure and
  divergence, not fidelity.**

## 6. Role 2 — the case-level identity verdict (fully formalized, frozen)

All rules below are frozen **before** any of the 48 cases is scored. Ground truth = the **human** holdout labels.

**6.1 Which attributes count.** Only C's **target attributes** (§4): slots with a reliable slice consensus. A
case's verdict considers only these; slots that are not targets for C are ignored (no consensus target to check
against).

**6.2 Per-attribute state.** For each target attribute (slot S, target value t), read value r = name-blind VLM
read of the image under test at slot S (Phase 1: the draft). Then:
- **missing/occluded:** `norm(r) ∈ NOT_VISIBLE` → the attribute is not readable in this image. **Abstains** — never
  counts as a contradiction (this directly counters the control-FP mode that broke the semantic verifier).
- **present:** `norm(r) ∉ NOT_VISIBLE`. It is **consistent** iff `norm(t) == norm(r)` (trivial exact match — skips a VLM call) **OR** `judge_match(t, r)` (§6.4, bidirectional) returns match; otherwise **contradicted** (a confident fine-grained identity error on this attribute). *(Amendment 2026-09-09, per implementation ruling: the exact-string short-circuit is a no-op under a real judge — which answers "yes" for identical strings — and is blessed as an optimization; the judge remains the sole authority for non-identical strings.)*

**6.3 Case verdict (frozen rule).**
- `n_present` = # target attributes present; `n_contradicted` = # present-and-contradicted.
- **ABSTAIN** iff the concept is undecidable (`n_target_attributes < 2`, §4) **or** `n_present = 0` (no target
  attribute is readable in the image — the verifier has nothing to check).
- Else **FAIL** iff `n_contradicted ≥ 1`; otherwise **PASS**.
- The threshold `τ_c = 1` (any confident contradiction ⇒ identity failure) is frozen. Rationale: a fine-grained
  identity failure is, by definition, at least one attribute rendered wrong; requiring one contradiction maximizes
  recall of true failures while the "present-only, conservative judge, missing-abstains" chain holds FPs down.

**6.4 Uncertainty handling (frozen).** The judge is **bidirectional and conservative**: `match` iff both
`(t,r)` and `(r,t)` orders answer "yes"; any other outcome (one-yes, both-no, unparseable) ⇒ **not-match ⇒
contradicted** *only when the attribute is present*. Judge prompt, frozen verbatim:
`You are comparing two descriptions of an object's {slot}. A: "{x}". B: "{y}". Do A and B describe essentially
the same {slot}? Answer with only 'yes' or 'no'.` Parse: strip/lower, `startswith("yes")`.

**6.5 Too-few-attributes / abstain accounting (frozen).** ABSTAIN means the MMKG cannot form a verdict. For the
**primary** metric it is mapped to **PASS** (an abstaining detector does not route — the honest deployment
behavior), computed over **all 48** cases. Abstain counts (and a decidable-only secondary readout) are reported
alongside but are **not** the headline, so hard cases are never silently dropped from the denominator.

**6.6 Metric and gates (frozen, = the detector's own bars).** Confusion matrix of verdict (FAIL/PASS, abstain→PASS)
vs human label over the 48 cases:
- **failure recall** = TP / (TP+FN) over the 24 FAIL cases; **PASS bar ≥ 0.75 (≥ 18/24).**
- **false-positive rate** = FP / (FP+TN) over the 24 PASS controls; **PASS bar ≤ 0.083 (≤ 2/24).**
- Report MCC and accuracy (descriptive). **Cross-tab** the MMKG verdict against the semantic verifier's verdict
  on the same cases, to show whether it catches the identity failures the semantic branch missed (the 12 misses
  that capped the detector at 0.50) — this is the mechanism claim, but the gates above are the decision.
- **Decision:** both gates pass ⇒ the first signal to clear the blocker ⇒ proceed to Phase 2 (wire as a gate,
  fresh holdout, pre-registered). Either fails ⇒ honest negative, reported like its predecessors; no integration.

## 7. The one batch + offline evaluation

1. **Build (batch, one-time):** per concept, assemble the LAION slice (reuse `outputs/phase_b/retrieval.json`),
   apply the contamination guard, run the name-blind slot read over the slice, compute target attributes (§4).
2. **Attribute-read pass (batch, one VLM pass):** name-blind slot read of each case's **draft** image.
3. **Retrieval log (offline/CPU):** run both Role-1 arms; write pools, NN scores, selected refs.
4. **Write the trace** (§3) — all of the above, with provenance SHA-256s.
5. **Offline evaluation (no GPU, frozen gates):** Role 1 metrics (§5), Role 2 verdict + recall/FP (§6),
   **per-concept and stratified by `n_target_attributes` and slice size**, flat and stratified means both
   reported. **Re-run, never splice** — reuse the frozen holdout artifacts read-only; never hand-edit scores in.

## 8. Scope, reuse, non-goals

**Reuses:** frozen Phase-B holdout + human labels + retrieval + reranker + DINO reserves (`outputs/phase_b/`);
rag-regen `retrieve.py`, `vlm.py`, `retrieved_reference.contamination`, the persistence/serializer layer;
`kg_test/graft` build + `stageb_vlm` read/judge path. **No new generation.**

**Non-goals (deferred):**
- **Phase 2:** wiring the verifier as a regeneration gate; a fresh, independent, pre-registered holdout before any
  integration (Phase 1 is score-only).
- **Stage 2 (open-world):** on-the-fly MMKG from user references, NN concept retrieval. The §3–§4 schema is built
  to accept on-the-fly population later, but only closed-world (by-key, concept known) is built now.
- **Role 1 fidelity comparison** (needs generation) — §5 guardrail.
- Any production-scale cross-domain MMKG-DB build; changes to draft/detector/mask/repair mechanism.

## 9. Success criteria and honest failure modes

- **Role 2 success:** recall ≥ 0.75 **and** FP ≤ 0.083 on the 48 cases, with the cross-tab showing failures the
  semantic branch missed. This is the first signal to clear rag-regen's named blocker.
- **Role 2 interpretable negative:** if it misses the bar, it joins v1/v2/v3/detector as a clean, cheap negative
  (no generation spent) — fine-grained VLM attribute checking does not solve identity detection on this data.
- **Role 1 finding:** divergence + coverage quantify whether organized retrieval is a non-trivial, distinct
  policy (per point 1 of review) — with fidelity parity, not gain, the honest expectation.
- **Guards against a false positive:** reference parity + contamination guard (§2); human labels as ground truth
  (not DINO-sign, not a VLM self-loop); every threshold frozen here (§4–§6); reliability-stratified readout (§7).

## 10. Open items for planning

- Confirm the exact `ragregen.vlm` entry point for the name-blind slot read and that decoding matches Stage-B.
- Confirm the LAION slice per concept is recoverable from `outputs/phase_b/retrieval.json` at sufficient size
  (`V_vis ≥ 3` feasibility per concept) — if many concepts are undecidable, surface it before the VLM pass, like
  the GRAFT feasibility pre-check.
- Confirm the reuse boundary between `kg_test/graft` build code and rag-regen (import vs. vendored batch script),
  given the two repos are separate.
- Pin the Role-1 MMKG candidate-pool construction against the actual build output fields (part-tags may be coarse
  for non-part-structured concepts; if a concept has no meaningful part decomposition, the MMKG pool = deduped
  quality-filtered slice, and coverage is measured over attribute slots rather than parts).
