# GRAFT — Graph-grounded, Reference-Anchored, Fidelity-Tested rare-concept generation

**Status:** Design (Phase 1 approved)
**Date:** 2026-08-25
**Workspace:** `ndbao_hbngoc/kg_test`

---

## 1. Problem

Text-to-image diffusion models (SDXL, FLUX) fail on **rare concepts** — species,
objects, and culturally specific subjects underrepresented in training data. A
prompt like *"a monkey puzzle tree"* yields a generic conifer because the model
never learned the concept's distinguishing structure. We want a **training-free**
method that fixes this by injecting a **multimodal knowledge graph (MMKG)** built
from a handful of real reference images.

## 2. Prior art and our wedge

**RAVEL / Context Canvas** (arXiv 2412.09614 — same paper, v1 "Context Canvas",
v2 "RAVEL") is the closest work: a training-free, graph-based RAG for T2I. Its
three load-bearing assumptions:

1. The KG is **LLM-hallucinated symbolic** text (name, traits, relations extracted
   from an LLM's parametric memory) — explicitly *not* multimodal, working "in the
   absence of visual priors".
2. It assumes the **LLM actually knows** the rare concept's attributes.
3. Both its steering and its self-correction (SRD) operate **at the prompt level**
   — it rewrites/expands the prompt from global alignment feedback.

**ImageRAG** (in-workspace) retrieves a single, unstructured reference image and
conditions generation on it — no graph structure, no part decomposition, no anchor.

**GRAFT's contribution.** The MMKG is **built from real reference images** so
attributes are *visually verified, not recalled* (breaks assumptions 1–2), and it
steers generation and closes the loop **below the prompt** — via reference
conditioning and **part-level** grounded verify-refine (breaks assumption 3).

> **Phase-2 revision (2026-08-30):** an earlier draft also listed *anchor→delta
> editing* as a contribution. That lever was cut in Phase 2 — generating a common
> anchor and correcting it with a symbolic text delta replicates RAVEL/Context
> Canvas's own prompt-level mechanism and does not advance the wedge. See
> `specs/2026-08-30-graft-phase2-design.md` §3.

## 3. Architecture (Phase 1)

Three modules. All models are already cached locally; nothing is trained.

```
 real refs (Treevill)                                 held-out real refs
        │                                                     │
        ▼                                                     ▼
 ┌──────────────────┐   MMKG   ┌───────────────────┐   ┌───────────────┐
 │ M1 Build MMKG    │────────▶ │ M2 Generate       │──▶│ M3 Verify-    │
 │ (visually        │          │ (Lever A:         │   │ Refine        │
 │  grounded)       │          │  SDXL+IP-Adapter) │◀──│ (part-level)  │
 └──────────────────┘          └───────────────────┘   └───────────────┘
                                        │  final image
                                        ▼
                                  ┌───────────────┐
                                  │ Eval harness  │
                                  └───────────────┘
```

### M1 — Visually-grounded MMKG (offline, per concept)

**Input:** K real reference images of a rare concept (Treevill species).
**Process:**
- **Qwen2.5-VL-7B** reads the refs and fills a fixed structured schema —
  `{global: {overall_form, canopy/silhouette, color_palette}, parts: {leaf,
  bark, cone_or_flower, branching}}`. Each value is read *from the image*.
- **GroundingDINO-base + SAM-ViT-huge** localize and crop each named part in the
  refs → one exemplar crop per part node.
- **SigLIP2** and **DINOv3-ViTL** embed the whole-concept ref and each part crop →
  visual embedding attached to every node.
- **Anchor edge:** Qwen2.5-VL names the nearest *common, generatable* concept
  (e.g. monkey-puzzle → "conifer/pine") and a symbolic **delta** describing what
  differs. (Anchor is stored in Phase 1; the delta-editing lever that consumes it
  is Phase 2, but the field is populated now so the graph is complete.)
- **Qwen3-VL-Reranker-2B** ranks candidate exemplar crops / refs for each node so
  the "best" visual evidence is chosen (its native reranking role).

**Output:** a per-concept MMKG serialized to JSON + an embeddings sidecar. Nodes:
`concept`, `part:*`, `attribute:*`, each `{symbolic_value, visual_embedding_ref,
exemplar_crop_path}`. Edges: `has-part`, `has-attribute`, `is-a/anchor`, `delta`.

### M2 — KG-steered generation, Lever A (SDXL + IP-Adapter)

- Compose a **KG-structured prompt**: base concept + ordered global/part attribute
  clauses drawn from M1's symbolic values.
- Select the KG's best exemplar image(s) (via the reranker) and condition SDXL
  through **IP-Adapter** (`ip-adapter-plus_sdxl_vit-h`, already cached; ImageRAG
  uses this exact path).
- Configurable `ip_scale`; N seeds per prompt.

*(Lever B — FLUX-Kontext anchor→delta editing — is designed but deferred to Phase 2.)*

### M3 — Multimodal part-grounded verify-refine

- For each KG part, **Qwen2.5-VL (+ GroundingDINO)** locate it in the generated
  image and score it against the part's exemplar crop using **DINOv3/SigLIP2**
  cosine similarity **and** a VLM attribute checklist derived from the KG.
- Parts below threshold → **re-seed** (Phase 1 default) with an attribute-boosted
  prompt / adjusted `ip_scale`. Loop ≤ N_refine (default 2).
- *(In-loop targeted Kontext region-edit of a failing part is Phase 2.)*

## 4. Evaluation

**Data:** small Treevill subset — ~10–15 rare species. Per species, hold out a
disjoint set of real reference images for scoring (build MMKG on the rest).

**Metrics (common, off-the-shelf):**
- **Concept fidelity (primary):** generated vs held-out real refs — DINOv3 and
  SigLIP2 image-embedding cosine similarity; CLIP-I. This is the metric that
  actually measures "did we render the rare concept".
- **KG-attribute accuracy:** Qwen2.5-VL TIFA-style checklist — fraction of the
  concept's KG attributes present in the generated image.
- **Prompt alignment (secondary):** CLIP-T. Reported but known-weak for rare concepts.

**Baselines:**
- **B0** vanilla SDXL prompt ("a photo of a <concept>").
- **B1** ImageRAG flat retrieval (one unstructured reference via IP-Adapter).
- **B2** RAVEL-style: symbolic-KG-expanded prompt, *no* visual grounding, prompt-only
  (our reproduction — the key ablation isolating the "visually grounded + reference"
  contribution).
- **Ours** GRAFT Lever A (+ verify-refine).

**Ablations:** visual-grounding on/off; anchor field on/off (Phase 2 uses it);
verify-refine on/off; `ip_scale` sweep.

## 5. Feasibility (verified, not aspirational)

- SDXL + IP-Adapter + SigLIP retrieval: **proven in-workspace** (ImageRAG
  `imageRAG_SDXL.py`, `retrieval.py`).
- FLUX.1-Kontext on the single 24 GB RTX 4090 via **nf4**: **proven in-workspace**
  (rag-regen `draft.py:load_kontext_t2i`). Relevant to Phase 2.
- GroundingDINO-base, SAM-ViT-huge, DINOv3-ViTL, SigLIP2, Qwen2.5-VL-7B,
  Qwen3-VL-Reranker-2B, IP-Adapter-SDXL: **all cached** under `ndbao_hbngoc/.cache`.
- Env: `kontext` conda env (torch 2.6 + cu124, diffusers 0.39, transformers 5.14.1,
  bitsandbytes) sees the GPU.
- **Single-GPU discipline:** M1/M3 (VLM + detection + embeddings) and M2 (SDXL)
  load sequentially, not concurrently; models are released between stages.

## 6. Non-goals / explicitly rejected ("no sci-fi")

- **No** token-level cross-attention / regional-attention surgery on FLUX's MMDiT
  (brittle, not reliably training-free) — we use IP-Adapter + Kontext editing.
- **No** fine-tuning, LoRA, textual inversion, or DreamBooth (violates training-free).
- **No** part-compositing "Frankenstein + harmonize" as a core path (optional
  ablation at most).
- **No** reliance on the LLM already knowing the concept — the refs are the ground truth.

## 7. Deferred to Phase 2

> **Revised 2026-08-30 — see `specs/2026-08-30-graft-phase2-design.md`.** The
> Phase-2 scope changed after the Phase-1 results:
>
> - **CUT — Lever B (FLUX-Kontext anchor→delta editing)** and the in-loop targeted
>   region edits that went with it: they replicate RAVEL/Context Canvas's
>   prompt-level correction mechanism. M1's `anchor`/`delta` fields are no longer
>   populated (kept optional in the schema for backward compatibility only).
> - **KEPT — full benchmark** over the complete duplication-aware Treevill split;
>   FID/diversity if budget allows.
> - **KEPT and expanded — FLUX as a second backbone**, via reference-conditioned
>   generation (FLUX.1-Kontext fed the real exemplar), *not* anchor→delta editing.
> - **Added — closing the GRAFT-vs-B1 gap** (attribute specificity,
>   conditioning-strength tuning, exemplar parity) as the phase's central thesis.

## 8. Open items

- **Treevill acquisition:** user provides `~/.kaggle/kaggle.json`; then pull
  `shuvokumarbasak4004/treevill-n-b-g-unique-and-rare-raw-dataset` → `kg_test/data/`.
  Inspect its folder structure (per-species dirs assumed) before finalizing the loader.
- Confirm GroundingDINO reliably localizes botanical "parts" (leaf/bark/cone); if a
  part is not localizable, M1 degrades gracefully to whole-image embedding for that node.
