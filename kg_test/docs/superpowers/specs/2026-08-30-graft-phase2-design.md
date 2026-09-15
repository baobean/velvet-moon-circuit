# GRAFT Phase 2 — Full benchmark, second backbone, and the B1 gap

**Status:** Design (Phase 2)
**Date:** 2026-08-30
**Workspace:** `ndbao_hbngoc/kg_test`
**Supersedes:** the "Deferred to Phase 2" section (§7) of `2026-08-25-graft-rare-concept-generation-design.md`

---

## 1. Where Phase 1 left us

Phase 1 is complete (see `kg_test/reports/phase1-report.md`). GRAFT — a training-free
method that builds a visually-grounded multimodal knowledge graph (MMKG) from a handful of
real reference photos and steers SDXL+IP-Adapter generation with part-level verify-refine —
was evaluated against three baselines on a small Treevill pilot (11 cases, 3 species).

**Headline result:**

| Method | DINOv3 | SigLIP2 | CLIP-I | Attr. acc. | CLIP-T |
|---|---|---|---|---|---|
| GRAFT (ours) | 0.226 | 0.710 | 0.633 | 0.611 | 0.203 |
| B0 — vanilla | 0.018 | 0.539 | 0.351 | 0.222 | 0.246 |
| **B1 — flat retrieval** | **0.278** | **0.733** | **0.665** | **0.778** | 0.211 |
| B2 — RAVEL-style repro | 0.031 | 0.560 | 0.427 | 0.333 | 0.216 |

GRAFT clearly beats B0 (no grounding) and B2 (LLM-recalled symbolic grounding — our
RAVEL/Context Canvas reproduction), by 7–12× on DINOv3 fidelity. **It does not yet beat B1**,
flat single-reference retrieval, on any fidelity metric. The pilot is too small (n=11) to know
whether that gap is real or noise.

## 2. Goals of Phase 2

1. **Settle the B1 comparison.** Make *"does a visually-grounded KG (part decomposition +
   attribute schema + verify-refine) actually beat flat single-reference retrieval, at scale?"*
   the driving question, and answer it with a full benchmark plus fixes aimed at the gap.
2. **Add a second generation backbone.** Evaluate GRAFT and all baselines on **FLUX** as well
   as SDXL, so the finding is not backbone-specific.
3. **Run the full, duplication-aware Treevill benchmark** — every species with enough unique
   images, not a hand-picked sample.

## 3. Scope: what is explicitly removed from the old Phase 2

The original spec §7 deferred three things to Phase 2. One is now **cut**, because it
replicates the mechanism GRAFT was designed to break away from:

- **CUT — Lever B: FLUX-Kontext anchor→delta editing.** Generating a *common anchor* concept
  and then correcting it toward the target with a *symbolic text delta* is, in mechanism, the
  same prompt/text-driven correction RAVEL/Context Canvas already does. It does not advance
  GRAFT's wedge (below-the-prompt, reference-anchored steering), so it is removed rather than
  built.
- **CUT — in-loop targeted Kontext region edits** of failing parts. This was the delta-editing
  lever applied inside the refine loop; it goes with Lever B. M3 keeps its Phase-1 re-seed
  strategy.
- **KEPT — full Treevill benchmark** (§5 below).
- **KEPT and expanded — FLUX** as a second backbone, but via reference-conditioned generation,
  **not** anchor→delta editing (§4 below).

### 3.1 Consequences for M1 and the graph

The `anchor` and `delta` fields existed in the MMKG *only* to feed Lever B. With Lever B cut:

- M1 **stops running its anchor/delta VLM pass** (saves one Qwen-VL call per concept — material
  at full-benchmark scale).
- `ConceptKG.anchor` / `ConceptKG.delta` remain in the schema as **optional, default-empty**
  fields so existing Phase-1 `kg.json` files still deserialize; they are simply no longer
  populated, documented, or consumed.

### 3.2 Consequence for the research framing (Phase-1 spec edit)

The Phase-1 spec §2 listed "anchor→delta editing" among GRAFT's contributions. That is no
longer a contribution. The Phase-1 spec is edited so the wedge reads:

> GRAFT's contribution: the MMKG is built from real reference images (attributes are visually
> verified, not recalled), and it steers generation and closes the loop **below the prompt** —
> via reference conditioning and **part-level grounded verify-refine**.

and its §7 is updated to point here.

## 4. Backbone-agnostic Lever A

Phase 1's M2 was SDXL-specific (IP-Adapter). Phase 2 makes reference-conditioned generation
**backbone-agnostic** behind one interface, and adds FLUX as a second implementation.

```
                         ┌───────────────────────────────┐
   KG prompt  ──────────▶│  Generator.generate(           │
   reranked real exemplar│    prompt, ref_image, seed,... )│──▶ image
                         └───────────────────────────────┘
                              ▲                      ▲
                              │                      │
                    SdxlIpGenerator          FluxKontextGenerator
                    (IP-Adapter, ip_scale)   (FLUX.1-Kontext nf4,
                                              real exemplar = context image)
```

- **`SdxlIpGenerator`** — unchanged from Phase 1: SDXL + `ip-adapter-plus_sdxl_vit-h`,
  conditioning strength `ip_scale`.
- **`FluxKontextGenerator`** (new) — `FluxKontextPipeline`, nf4-quantized, reusing the proven
  loader idioms from `rag-regen/ragregen/draft.py` / `ImageRAG/generators/kontext_gen.py`. It
  feeds the **reranked real exemplar** as the Kontext context image alongside the KG-structured
  prompt. Conditioning strength is Kontext's guidance/strength knobs (the FLUX analog of
  `ip_scale`). This is reference-conditioned *generation*, not anchor→delta editing.
- **Only FLUX.1-Kontext-dev is cached** in the workspace (no base FLUX.1-dev, no FLUX
  IP-Adapter); it is the workspace's native FLUX reference-conditioning path, so no new large
  checkpoint download is required.
- **Single-GPU discipline (unchanged):** SDXL (~fits) and FLUX-nf4 (~12 GB) plus the VLM
  (~16 GB) do not co-fit on the 24 GB RTX 4090. Backbones and modules load sequentially and are
  released between stages; every GPU-touching phase runs in its own disposable subprocess (the
  Phase-1 execution model — kept verbatim).

## 5. Closing the B1 gap

Three concrete levers, all traceable to the Phase-1 report's own diagnosis (§7):

### 5.1 Attribute specificity

Phase 1 noted some species' schemas came back generic ("tree / full / green"), which gives the
KG prompt nothing discriminative to add over B1's single reference.

- M1's VLM instruction is revised to demand **discriminative, non-generic** per-part attributes
  (explicitly: color/shape/texture/count details that distinguish *this* species from a generic
  tree).
- A pure, unit-tested `is_generic(value) -> bool` filter drops or flags low-information values
  (a small stop-list plus a length/specificity heuristic).
- If a concept's schema comes back with too many generic values, M1 **re-asks once** with a
  sharper instruction before accepting the schema.

### 5.2 Conditioning-strength tuning

B1 wins in part by conditioning hard on one strong reference. GRAFT must condition at least as
strongly, with the KG attributes adding *on top* rather than diluting the image signal.

- **SDXL:** `ip_scale` sweep (e.g. {0.4, 0.6, 0.8}) as a first-class ablation axis.
- **FLUX:** Kontext guidance/strength sweep over a comparable range.
- The sweep is reported, and the best setting per backbone is used for the headline comparison.

### 5.3 Exemplar parity

GRAFT and B1 must draw their reference from the **same reranked candidate pool**, so the
GRAFT-vs-B1 comparison isolates *structure* (part decomposition + attribute schema +
verify-refine) rather than accidentally comparing two different reference photos. This is a
correctness constraint on the eval harness, verified in the driver.

## 6. Benchmark

### 6.1 Hardening (before scaling up — cheap fixes from report §7)

- **Species selection by unique-image count.** Replace alphabetical selection with a ranking by
  deduped unique-image count, so a requested "N species" run spends its slots on species the
  dataset can actually support (build split + non-empty held-out split).
- **Refine loop tolerates a worker crash** the same way it already tolerates a GPU-memory error
  — retry the next seed instead of failing the whole case. Closes the one case lost in Phase 1.

### 6.2 The run

- **Matrix:** `{SDXL, FLUX} × {GRAFT, B0, B1, B2}` over **all** dedup-qualified Treevill
  species (those with more unique images than `k_build_refs`, leaving a non-empty held-out
  split). Baselines run per backbone too (B0 = text-only on that backbone; B1 =
  retrieved-single-ref conditioning on that backbone; B2 = LLM-recalled attributes, text-only).
- **Staged and resumable.** Because the shared 4090 is contended, the run is staged — **SDXL
  matrix first** (the proven path, where the B1 gap lives), then the **FLUX matrix** on the same
  species set — and is resumable at case granularity via the existing `gpu_queue.sh` +
  per-case subprocess pattern, so a stall or crash never re-does completed cases.

### 6.3 Metrics

- **Fidelity (primary), unchanged:** DINOv3, SigLIP2, CLIP-I mean cosine similarity of the
  generated image to the held-out real references.
- **Attribute accuracy** (Qwen2.5-VL checklist) and **CLIP-T** (secondary), unchanged.
- **Paired GRAFT-vs-B1 analysis (new, central to the thesis):** per-species paired deltas,
  **win-rate** (fraction of species where GRAFT ≥ B1 on the primary metric), mean delta with a
  simple confidence interval / paired test. Reported per backbone.
- **FID / diversity (new, if sample budget allows):** added once the core matrix is in, as a
  distributional check; skipped without prejudice if the GPU budget is tight.

## 7. Feasibility (verified, not aspirational)

- SDXL + IP-Adapter path: proven in Phase 1 (already implemented and run end-to-end).
- FLUX.1-Kontext nf4 on the single 24 GB RTX 4090: proven in-workspace
  (`rag-regen/ragregen/draft.py`, `ImageRAG/generators/kontext_gen.py`); the checkpoint
  (`black-forest-labs/FLUX.1-Kontext-dev`) is already cached under `ndbao_hbngoc/.cache`.
- All other models (Qwen2.5-VL-7B, Qwen3-VL-Reranker-2B, SigLIP2, DINOv3, CLIP, GroundingDINO,
  SAM, IP-Adapter-SDXL): cached, used unchanged from Phase 1.
- Subprocess-isolated, resumable execution: proven in Phase 1 (`graft/worker_*.py`,
  `scripts/gpu_queue.sh`, `run_remaining_phase1.sh`).

## 8. Non-goals (unchanged from Phase 1, plus one)

- **No** anchor→delta / symbolic-text editing lever (newly cut — §3).
- **No** cross-attention / regional-attention surgery on SDXL or FLUX's MMDiT.
- **No** fine-tuning, LoRA, textual inversion, or DreamBooth.
- **No** part-compositing "Frankenstein + harmonize" as a core path.
- **No** reliance on the LLM already knowing the concept — the refs are the ground truth.

## 9. Open items

- **Kontext conditioning knobs:** confirm the exact guidance/strength range for
  `FluxKontextPipeline` that gives reference conditioning comparable to SDXL's `ip_scale`
  before locking the sweep grid (§5.2).
- **GPU budget for the full matrix:** 8 cells × N qualified species is large on a contended
  card; §6.2 staging + resumability is the mitigation, but the FLUX stage and FID/diversity
  (§6.3) are the first things to trim if the budget runs short.
- **`is_generic` tuning:** the stop-list/heuristic (§5.1) will need one calibration pass on real
  Treevill schemas so it drops genuinely generic values without discarding legitimately simple
  ones.
