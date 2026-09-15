# rag-regen — dual-stream verification and retrieval-augmented regeneration

**Date:** 2026-07-25
**Status:** design approved, not yet implemented
**Supersedes in scope:** `../rag-edit/` (SDXL + IP-Adapter; steps 2–4 never ran)

---

## 1. The claim

ImageRAG ([2502.09411](https://arxiv.org/abs/2502.09411)) uses one GPT-4o call to decide whether a
generated image matches its prompt, name the missing concepts, and write a retrieval caption. That
call is opaque, proprietary, and emits a hard label with nothing to calibrate.

**This project decomposes that single judge into two streams — one grounded, one semantic — fuses
them, and repairs the image with FLUX.1-Kontext instead of a global IP-Adapter.**

Two claims follow, and they are separable:

> **C1 (verifier).** A grounded stream (GroundingDINO + a region-text scorer) fused with a semantic
> stream (open-weights VLM) detects prompt-image mismatch at least as well as a single VLM judge,
> and — unlike it — produces continuous per-concept scores that can be calibrated.

> **C2 (repair).** Given a correct reference, masked reference-guided regeneration renders a rare
> concept that the base generator cannot, *without* the global-conditioning penalty that made
> ImageRAG score below its own baseline.

C1 is measurable without any diffusion. C2 needs the generator. They fail independently.

### Why C2 is worth re-testing

Every evaluation run in `../ImageRAG/results/` has RAG losing to its no-RAG baseline:

| run | CLIP no_rag → imagerag | SigLIP no_rag → imagerag |
|---|---|---|
| `eval_methods_20260704_130505` | 0.3392 → 0.2291 (**−0.110**) | 0.1919 → 0.0564 (**−0.136**) |
| `eval_methods_20260704_211625` | 0.3875 → 0.2676 (**−0.120**) | 0.2687 → 0.1544 (**−0.114**) |
| `eval_methods_20260707_233423` | 0.2996 → 0.2548 (**−0.045**) | — |

Two causes, and the design addresses both:

1. **Global IP-Adapter at scale 0.5 drags the whole canvas.** Fixed by confining the edit — the
   change is spatially bounded by a mask, and unmasked pixels are composited back.
2. **The metric could not see the improvement.** The paper reports **CLIP, SigLIP, and DINO**. The
   fork implements only the first two — `grep -c dino ../ImageRAG/scripts/eval_methods.py` → `0`.
   The dropped metric is precisely the one that measures concept identity. Restored here.

---

## 2. Architecture

```
prompt ──> [0] draft: FluxKontextPipeline (text-only)
              │
              ▼
        ┌──────────────────────────────────┐
        │ [1+2] VERIFY  (dual stream)       │
        │  A grounded: DINO → crop scorer   │──┐
        │    → per-concept continuous score │  │ fuse
        │  B semantic: Qwen3-VL → JSON      │──┘
        └──────────────────────────────────┘
              │ pass → done
              │ fail → target_concept
              ▼
        [3] query.formulate → validate_query()
              ▼
        [4] retrieve: SigLIP + FAISS IndexFlatIP → hits[1..N]
              ▼
        ┌──── for attempt in 1..N ─────────────────┐
        │ [5] mask_reference(hits[attempt])        │
        │ [6] regen(draft, mask_draft, ref)        │
        │     re-verify → break if pass            │
        └──────────────────────────────────────────┘
```

### Why the streams are cut this way

The original sketch had step 1 (does it match?) and step 2 (what is wrong?) as parallel streams.
They are not independent: within any single verifier, the verdict is a *derivative* of the
diagnosis — if a concept is flagged, the verdict is already fail. `verdict = (nothing flagged)`,
one computation with two outputs.

The useful axis is **grounded vs. semantic**:

| | Stream A (grounded) | Stream B (semantic) |
|---|---|---|
| models | GroundingDINO + FG-CLIP/SigLIP | Qwen3-VL |
| output | per-concept continuous score | `{verdict, issues[]}` |
| verdict | `min(score) < τ` | model-emitted |
| diagnosis | `argmin` | `issues[0]` |
| sees | object presence, fine-grained identity | counts, relations, attribute binding |
| blind to | anything unboxable | calibration — it is a hard label |

This buys three things the original cut did not: one VLM call per image instead of two (matters —
Qwen-7B at ~16 GB and FLUX-nf4 at ~12 GB do not co-fit on a 24 GB card); a tunable τ that yields a
precision/recall curve for the verifier; and a **measurable disagreement rate** between streams,
which is evidence that the decomposition buys something. Under the original cut, disagreement
between step 1 and step 2 was an inconsistency to paper over.

### Fusion rule

```
fail  if  A fires  OR  B fires
target = A.argmin  if A fired, else B.issues[0]
```

**Abstention is not absence.** When GroundingDINO returns no box for a phrase, that means either
the concept is missing *or* the detector failed. Stream A must return `ABSTAIN`, not `MISSING`, and
fusion defers to Stream B. Getting this wrong makes every relational or counting prompt read as a
false failure.

### Modules

| module | interface | provenance |
|---|---|---|
| `ragregen/concepts.py` | `parse(prompt) -> [Concept{phrase, kind}]` | new |
| `ragregen/verify/grounded.py` | `score(img, concepts) -> {phrase: {box, conf, sim, state}}` | DINO wiring from `../rag-edit/ragedit/mask.py` |
| `ragregen/verify/semantic.py` | `judge(img, prompt) -> {verdict, issues[]}` | adapt `../rag-edit/ragedit/evaluate.py` |
| `ragregen/verify/fusion.py` | `fuse(a, b) -> Verdict{ok, target, evidence}` | new, ~40 lines |
| `ragregen/query.py` | `formulate(prompt, verdict) -> str` | `evaluate.py` + its `validate_query()` |
| `ragregen/retrieve.py` | `search(query, k) -> [Hit]` | `../ImageRAG/retrieval.py`, `../rag-edit/ragedit/db.py` |
| `ragregen/mask.py` | `mask_draft(img, concept)`, `mask_reference(ref, concept)` | **`../ImageRAG/scripts/finegrained_seg.py`** |
| `ragregen/regen.py` | `inpaint(...)` \| `stitch(...)` | **`../ImageRAG/scripts/kontext_engine.py`** |
| `ragregen/schedule.py` | stage-batched work queue | new |
| `ragregen/metrics.py` | CLIP, SigLIP, DINO-identity, preservation | `eval_methods.py` + restore DINO |
| `ragregen/validate.py` | preflight on operator-supplied data | new |
| `ragregen/trace.py` | per-case reasoning chain | `../rag-edit/ragedit/trace.py` |

**`mask.py` has two callers with opposite intent.** `mask_draft` finds the region to destroy;
`mask_reference` finds the object to keep. Same DINO → boxes → classifier → SAM chain, different
post-processing (dilate + feather vs. tight crop + alpha). One implementation, two thin wrappers —
so the measured 0.917 accuracy transfers to both.

### Measured basis for steps 4 and 5

Neither is speculative. From `../ImageRAG/results/`:

**Retrieval** (`hard_query_benchmark_20260722_102957`, 72 images / 22 categories):

| method | CAT P@3 | INST R@1 | INST MRR |
|---|---|---|---|
| clip_b32 | 0.955 | 0.667 | 0.819 |
| **siglip** | **1.000** | **0.833** | **0.910** |
| openclip_l14 | 0.939 | 0.833 | 0.910 |

**Masking** (`finegrained_seg_20260722_110705`, 12 breed targets):

| method | accuracy |
|---|---|
| GroundedSAM baseline (breed prompt) | 0.667 |
| detect-then-classify — prototype | **0.917** |
| detect-then-classify — siglip | **0.917** |

---

## 3. The loop

```python
draft = kontext_t2i(prompt)
v0 = verify(draft, prompt)
if v0.ok: return draft

q      = formulate(prompt, v0)          # + validate_query()
hits   = retrieve(q, k=N)
mask_d = mask_draft(draft, v0.target)   # computed ONCE

best = (draft, v0.score)                # the draft is itself a candidate
for i in 1..N:
    ref = mask_reference(hits[i], v0.target)
    out = regen(draft, mask_d, ref)     # ALWAYS from draft, never from out
    vi  = verify(out, prompt)
    if vi.score > best.score: best = (out, vi.score)
    if vi.ok: break
return best
```

Three rules, each preventing a specific failure:

1. **Never chain edits.** Each attempt regenerates from the original draft. Every pass through
   `FluxKontextInpaintPipeline` does a full-frame VAE decode (`kontext_engine.py:21-27`), so
   chaining compounds degradation across the whole canvas — attempt 3 would be visibly mushier than
   attempt 1 regardless of reference quality, and that would be misread as retrieval degrading.
2. **`best` is seeded with the draft.** This makes "RAG scores worse than no-RAG" structurally
   impossible per case — the exact failure in all four `eval_methods` runs.
3. **Report the selection rule's effect.** Rule 2 is mildly self-favouring, so the results table
   carries both a `best` column and a plain `last-attempt` column. The gap between them is the
   value of the retry budget, and hiding it would overstate the method.

`k` and `N` are the same number: the retrieval depth *is* the retry budget.

---

## 4. Scheduling

Measured from `../ImageRAG/results/kontext_example_20260721_224056/run.log`:

- FLUX pipeline load: **5 min 21 s** (line 191)
- One generation, 28 steps, nf4: **~2 min**
- Qwen-7B ≈ 16 GB + FLUX-nf4 ≈ 12 GB > 24 GB card → **they do not co-fit**

A naive per-case loop alternates VLM → FLUX → VLM → FLUX. At N=3 that is 3 FLUX loads per case:

| | per case | 30 cases |
|---|---|---|
| FLUX loading | 16 min | **~8 h** |
| actual generation | 6 min | 3 h |

**73% of wall-clock would be spent loading weights.**

**Stage-batching:** advance *all* cases through one stage before switching models.

```
[load VLM]   verify all drafts                [unload]
(cpu only)   formulate + retrieve + mask
[load FLUX]  attempt-1 for all failing cases  [unload]
[load VLM]   re-verify all                    [unload]
[load FLUX]  attempt-2 for survivors          [unload]   ...
```

At N=3: **4 VLM loads + 3 FLUX loads total, independent of case count.** FLUX loading drops from
~8 h to ~16 min.

The consequence is that `schedule.py` is a work queue with per-case state persisted to disk —
`{stage, attempt, draft, mask, hits, best_so_far}` in JSON — where each stage is a short-lived
process that sweeps every eligible case and writes back.

That is not pure overhead. `../rag-edit`'s last run died at `ragedit/mask.py:49` with
`torch.OutOfMemoryError` because five other processes held the card. Short-lived stage processes
over persisted state mean a contended GPU costs one retried stage, not the whole run, and a crash
resumes where it stopped.

---

## 5. Metrics

Aligned to the paper, which reports **CLIP, SigLIP, and DINO** plus a human study (46 participants,
767 comparisons) over ImageNet, iNaturalist (first 1000 classes), CUB, and Aircraft.

| metric | model | in the loop? | role |
|---|---|---|---|
| CLIP | open-CLIP ViT-L/14 | no | paper comparability |
| SigLIP | SigLIP-**base** (≠ retrieval's SO400M-384) | no | paper comparability |
| **DINO** | DINOv3-L; output crop ↔ ground-truth refs | no | **the actual claim (C2)** |
| Preservation | three-zone, `kontext_engine.py:477` | no | anti-gaming guard |
| Verifier pass-rate | Stream A + B | **yes** | development signal only — never a headline |

**Encoder hygiene.** The paper retrieves with OpenAI CLIP and evaluates with open-CLIP —
deliberately different checkpoints. This design keeps the same discipline: retrieval uses
SigLIP-SO400M-384 (or FG-CLIP, pending §6 Stage 1), evaluation uses a different checkpoint, and the
split is stated in the results table. Without it, scoring outputs with the model that chose the
reference is self-marking.

**Resolution pinning.** `FluxKontextInpaintPipeline` output size ≠ input size
(`kontext_engine.py:16-19`). All metrics compute at one pinned resolution, or resampling artifacts
appear as real deltas.

**Human study.** 46 participants / 767 comparisons is likely out of reach. Substitute a held-out
VLM judge — one never used inside the loop — and state the deviation explicitly rather than
implying parity.

---

## 6. Experiment matrix

### Stage 1 — encoder bake-off (no FLUX; runs on a contended GPU)

FG-CLIP vs. SigLIP, evaluated **separately in each of three slots**. FG-CLIP is trained for
region-text alignment, so it is expected strongest where it scores *crops* and least differentiated
as a whole-image retriever. A single blended "winner" would hide that.

| slot | task | harness | SigLIP baseline |
|---|---|---|---|
| Stream A crop scorer | phrase ↔ DINO crop | new (small) | — |
| Step 5 box selector | class name ↔ candidate crop | `finegrained_seg.py` — add `FGCLIPClassifier` to `heads` (L177) | 0.917 |
| Step 4 retriever | query ↔ corpus | `hard_query_benchmark.py` — add `encode_fgclip` + one `elif` (L109-114) | R@1 0.833 |

Both harnesses are already generic over the encoder: `finegrained_seg.py` iterates `heads` for
scoring (L246) and table generation (L263-269); `hard_query_benchmark.py` dispatches `encode_*`
functions returning `(img_mat, enc_text)`.

### Stage 2 — main table

Arms `no_rag` / `oracle` / `full` (the A→B→C decomposition from `../rag-edit/README.md`), N=3.
`A→B` is the contribution; `B→C` is retrieval error.

Worst case per case: 1 draft + 1 oracle + 3 full = 5 generations per mechanism. Both mechanisms
(`inpaint+composite`, `stitch`) = 9 generations ≈ **18 min/case**; 30 cases ≈ **9 h** plus ~40 min
of stage-batched loading.

**Run a 6-case pilot first to pick one mechanism, then the full matrix with the winner** — ~5 h
instead of ~10, and the pilot is itself reportable as the mechanism ablation.

On the two mechanisms: `inpaint` and `inpaint+composite` are **the same diffusion pass** —
`composite_back` is a pixel-space post-op. And per `kontext_engine.py:21-27`, the pipeline preserves
unmasked regions by *latent* blending at 1/8 resolution followed by full-frame VAE decode, so
unmasked pixels are guaranteed to drift. Compositing is therefore not an optional polish step;
running without it is a **diagnostic** that measures latent-blend drift, not a competitive arm.

### Stage 3 — verifier ablation (no FLUX)

`grounded-only` / `semantic-only` / `fused`. This is claim C1. It re-scores images Stage 2 already
produced, so it costs no generation. The same run yields the stream-disagreement analysis.

---

## 7. Operator interface

A second person runs the experiments and **does not modify code**. They supply two things: a test
dataset and a retrieval database.

- `configs/dataset.yaml`, `configs/retrieval_db.yaml` and `configs/pipeline.yaml` are the only files
  they edit — paths, field names and numeric knobs (retry budget `N`, threshold `τ`, steps, seed,
  encoder choice). No Python.
- `scripts/build_index.py` turns an image folder into embeddings + a FAISS index.
- `ragregen/validate.py` runs **before any GPU work**: images readable, concept labels present,
  ground-truth references present (both the `oracle` arm and the DINO metric require them), index
  dimensionality matching the configured encoder. It fails with actionable messages rather than
  dying nine hours into a run.
- One entry point: `run.sh {validate|screen|bakeoff|build-index|pilot|full|report}`.

Operator-facing documentation lives in `docs/RUNBOOK.md`.

---

## 8. Environment

Verified 2026-07-25 on this server.

**One conda env runs everything: `kontext` (Python 3.11.15).**

| component | status in `kontext` |
|---|---|
| diffusers 0.39.0 — `FluxKontextPipeline`, `FluxKontextInpaintPipeline` | present |
| transformers 5.14.1 — GroundingDino, SAM, Siglip, Qwen2.5-VL, **Qwen3-VL** (dense + MoE) | present |
| bitsandbytes (nf4 quantization) | present |
| `faiss`, `open_clip` | **missing — `pip install faiss-cpu open_clip_torch`** |

**`PYTHONNOUSERSITE=1` is not needed here.** `~/.local/lib/python3.10/` shadows Python **3.10**
environments only, which is why `../rag-edit` requires it (README guardrail #1) — that project runs
in `ImageRAG_qwen` (Python 3.10.13, transformers 4.44.2, diffusers 0.31.0, no Kontext, no Qwen3-VL).
`kontext` is Python 3.11.15 and cannot be shadowed; `transformers` resolves from the env's own
`site-packages`. The guardrail does not propagate to this project.

Weights resolve from `../.cache`. Already present: FLUX.1-Kontext-dev, SAM-ViT-H,
GroundingDINO base+tiny, SigLIP-SO400M-384, DINOv3-L, Qwen2.5-VL-7B.
**Not present: FG-CLIP, Qwen3-VL.**

---

## 9. Risks

**R1 — The rarity premise is unverified, and the risk has grown.**
`../rag-edit/README.md` lists under Known Risks that nobody has shown the generator actually fails
on durian / axolotl / Amur leopard; steps 2–4 never ran, so it was never settled. That README
assumed **SDXL**. This project uses **FLUX.1-Kontext**, which is substantially stronger on long-tail
concepts. If Kontext renders these correctly, the verifier passes everything and there is nothing to
repair. *This is the largest scientific risk and it costs ~1 hour to test.* Build step 0 is the gate.

**R2 — Disk. 270 GB free on a 14 TB volume at 99% use, shared with other people's work.**
500k originals ≈ 75 GB. Storing 384px re-encoded copies cuts that to ~15–25 GB and loses nothing,
since both encoders resize to 384 anyway. Decide before the upload. Failure mode otherwise is
ENOSPC partway through an embedding run — and collateral damage to other users.

**R3 — Stream A abstention** (see §2). `ABSTAIN ≠ MISSING`.

**R4 — Kontext output size ≠ input size.** Pin metric resolution (see §5).

**R5 — Missing weights.** FG-CLIP and Qwen3-VL (~16 GB) must be downloaded; interacts with R2.

**R6 — Retrieval index scale.** At 500k, SigLIP-SO400M embeddings are 500k × 1152 × 4 B ≈ 2.3 GB
fp32. `IndexFlatIP` brute force remains tractable, so **exact search is retained** and retrieval
accuracy is never confounded by index approximation. Do not reach for IVF/HNSW without a measured
reason.

**R7 — Human study infeasible at paper scale** (see §5).

---

## 10. Build order

| # | step | cost | why here |
|---|---|---|---|
| **0** | **Screen the premise** — draft ~30 candidate concepts with Kontext, hand-label pass/fail | ~1 h | **Gate.** If <30% fail, the concept list must get rarer and everything downstream shifts. |
| 1 | Env: `pip install faiss-cpu open_clip_torch`; pull FG-CLIP + Qwen3-VL | ~30 min | unblocks all |
| 2 | Encoder bake-off (Stage 1) | ~1 h, no FLUX | picks the encoder for three slots |
| 3 | `verify/` — grounded + semantic + fusion | — | validated against step 0's labels |
| 4 | `mask.py` — two wrappers over `finegrained_seg.py` | — | 0.917 transfers |
| 5 | `regen.py` — inpaint exists in `kontext_engine.py`; add `stitch` | — | |
| 6 | `schedule.py` — persisted work queue | — | |
| 7 | `metrics.py` — restore DINO | — | |
| 8 | `validate.py` + `build_index.py` + `run.sh` | — | operator interface |
| 9 | 6-case pilot → mechanism choice → full run | ~5 h | |

Step 0 pays twice: hand-labelling those 30 drafts gives both the premise screen and the
ground-truth set for validating the verifier in step 3.

---

## 11. Logging

Inherited from `../rag-edit`, which got this right: every run opens its own
`outputs/<tag>_<TIMESTAMP>/` and tees the console into it, so an ablation cannot overwrite the
baseline it exists to be compared against. `outputs/<tag>_latest` symlinks the most recent.

```
outputs/full_20260801_093000/
  run.log        full console output (raw VLM replies, retrieval scores)
  run.json       argv, args, GPU, library versions, status, result means
  summary.md     the arm table
  per_case.csv   per-case scores
  queue.json     scheduler state — resumable
  <case>/        draft.png mask.png attempt_{1..N}.png best.png trace.json
```

`trace.json` is the per-case reasoning chain: every raw VLM output, every Stream A per-concept
score, every retrieval hit with its similarity. When a RAG pipeline underperforms its own baseline,
the cause is almost never the algorithm — it is one silently-garbage intermediate that nothing
logged. `../ImageRAG`'s rare-concept run lost partly because Qwen's retrieval caption collapsed to
the literal string `"A"` and nothing caught it; `validate_query()` is the guard, and the trace is
how you find the next one.
