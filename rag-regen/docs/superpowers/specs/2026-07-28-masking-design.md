# Masking — locating the region to repaint and the object to copy

**Date:** 2026-07-28
**Parent spec:** `docs/superpowers/specs/2026-07-25-rag-regen-design.md` (§2 Modules, §3 The loop, §10 step 4)
**Port sources:** `../rag-edit/ragedit/mask.py`, `../ImageRAG/scripts/finegrained_seg.py`
**Depends on:** `ragregen/models.py` (`DinoDetector`), `ragregen/verify/prototype.py` (`PrototypeBank`)

**Doc 1 of 4** in the regeneration half: **masking** → regeneration → orchestration → metrics.

---

## 1. The claim

Given an image and a phrase, produce a pixel mask of the right object — the region of the draft to
destroy, or the object in a reference to copy from. Both callers get the same thing, and the
difference between them lives downstream.

## 2. What already exists

This module is mostly wiring between parts that are written and measured.

| piece | status | where |
|---|---|---|
| GroundingDINO box proposals | **done** | `ragregen/models.py:50-83`, `DinoDetector.all_boxes` |
| SAM box → mask | to port, ~15 lines | `../rag-edit/ragedit/mask.py:74-87` |
| Prototype classifier | **done** (plan 4) | `ragregen/verify/prototype.py`, `PrototypeBank` |
| Dilation | to port, ~8 lines | `../rag-edit/ragedit/mask.py:103-111` |
| Reference crop / matte | **done, and stays in doc 2** | `../ImageRAG/scripts/kontext_engine.py:415-467` |
| GPU eviction | to port | `../rag-edit/ragedit/mask.py:115-131` |

`DinoDetector.all_boxes` already returns `[(box, score), ...]` sorted by score descending, capped at
8, at `box_threshold = text_threshold = 0.25`. Stream A uses it unchanged. Masking reuses it rather
than re-porting the DINO wiring the parent spec's provenance table anticipated.

**`facebook/sam-vit-huge` is already in `.cache/`.** No download. It ships natively in
`transformers` (`SamModel`) — pure PyTorch, no CUDA extension build.

## 3. Mechanism

Staged exactly as the 0.917 harness stages it (`finegrained_seg.py:191-207`):

```
DINO(phrase)            -> up to 8 candidate boxes, sorted by confidence
SAM(each candidate)     -> one binary mask per candidate
select_box(candidates)  -> the winner
post-process            -> dilate (draft side only)
```

SAM runs on **every** candidate before selection, not only on the winner. That is what the measured
harness does, and it means the mask and the selection come from one candidate set rather than two
passes that can disagree. Cost is 8 SAM forwards on an already-loaded model.

> **Correction, 2026-07-28.** This is wasteful and should change. Selection scores the *rectangular
> crop*, so it never reads a candidate's mask — only the winner's is used. The harness segmented
> every candidate because its own evaluation needed per-candidate masks; we have no such need.
> Deferring SAM to the winner cuts masking cost 8× and, more importantly, lets the three models run
> in sequence rather than co-resident:
>
> ```
> DINO -> candidates -> free  |  encoder -> scores -> free  |  SAM -> winner's mask -> free
> ```
>
> Prototype selection OOMed on the shared 4090 precisely because SigLIP, DINO and SAM were resident
> together (~9.4 GB). Sequencing them is doc 3's stage-batching applied inside one module.

## 4. Box selection, and why the draft side needs it too

`select_box` is one seam with two rules:

```python
@dataclass(frozen=True)
class Candidate:
    box: Box
    dino_conf: float
    mask: np.ndarray        # SAM's mask for this box
    crop: Image.Image       # the plain rectangular crop, for the classifier


def select_box(candidates: list[Candidate], *, prototypes=None,
               phrase=None) -> Candidate
    # no bank, or a single candidate -> highest DINO confidence
    # bank and >1 candidate         -> highest prototype similarity on the box CROP
```

The classifier scores the **plain rectangular crop**, not the SAM-masked object — again matching
the harness (`finegrained_seg.py:202`).

**The draft side needs this, and the reason is not the one the harness measured.** The 0.917 result
came from disambiguating two breeds deliberately composited into one frame; no prompt in
`configs/dataset.yaml` poses that. The real exposure is **whole versus part**:

| case | `concept` | `coarse` | what grounding the coarse term returns |
|---|---|---|---|
| `stack_of_books` | stack of books | `books` | plausibly one box per visible book |
| `sushi` | sushi | `food` | plausibly one box per piece |

Highest-confidence on `books` selects a single book. Repainting one book out of a stack is not a
repair. A prototype built from photographs of book *stacks* scores the whole-stack crop above any
single-book crop, which is precisely the discriminative structure plan 4 built and validated.

> **Correction, 2026-07-28.** The mechanism is right; both examples were wrong. Measured on the real
> photographs (`findings/2026-07-28-masking-result.md`):
>
> - **`stack_of_books` does not fail.** Grounding `books` returns 8 candidates as predicted, but the
>   highest-confidence one *is* the whole stack. Confidence needs no help here.
> - **`sushi` fails differently** — SAM segments the plate for both the coarse and the fine phrase,
>   so no selection rule can fix it (§10 M2).
> - **`boston_bull` is the real case.** Grounding `dog` returns the head and the whole dog;
>   confidence takes the head (mask fraction 0.052 versus 0.183 for the whole animal).
>
> **The scoring phrase is the concept, not the coarse term.** Scoring `dog`-ness cannot separate a
> head from a whole dog — both crops are dog. "Which is the Boston bull" can. `select_box` therefore
> takes the phrase to *score*, which `mask_draft` sets independently of the phrase it grounds. The
> fine prototype is also the only one that exists: `gt_refs` covers all 22 concepts, while coarse
> prototypes need `coarse_refs`, which no dataset carries.
>
> **Prototypes must be built from cropped references.** See §10 M5.

**Which phrase grounds which side:**

- **`mask_draft` grounds the coarse term.** The draft is by definition the image that got the fine
  concept wrong, so the fine phrase may ground poorly or not at all, and how well it grounds would
  vary with how wrong the draft is — making mask geometry a function of draft quality. The coarse
  term is present whether the draft succeeded or failed. This is also what the harness does: stage 1
  uses `generic` (`finegrained_seg.py:192`).
- **`mask_reference` grounds the fine concept.** A reference is by construction a photograph of the
  real thing, so the fine phrase is the accurate one, and clutter (a second animal, a handler, an
  enclosure) is exactly what the classifier is for.

## 5. Interfaces

```python
@dataclass(frozen=True)
class MaskResult:
    mask: np.ndarray        # float HxW in [0, 1]
    box: Box                # the selected candidate
    score: float            # that box's DINO confidence
    n_candidates: int       # how many DINO returned
    selected_by: str        # "confidence" | "prototype" -- recorded, never inferred


class Masker:
    """Owns SAM; takes a DinoDetector. Never loads at import."""
    def __init__(self, detector, device="cuda", sam_id=SAM_ID): ...
    def segment(self, image, phrase, *, prototypes=None) -> MaskResult | None
    def to_cpu(self) -> None
    def free(self) -> None


def mask_draft(image, coarse, masker, *, prototypes=None,
               dilate_px=12) -> MaskResult | None
def mask_reference(image, concept, masker, *,
                   prototypes=None) -> MaskResult | None
```

**Mask dtype.** SAM returns a boolean array. `MaskResult.mask` carries it as float in `[0, 1]` so it
satisfies both of `prepare_reference`'s uses — the `mask > 0.5` bbox derivation and the
`np.clip(mask, 0, 1)` matte arithmetic — without a second conversion at the call site. Dilation
operates on the uint8 view and converts back.

**Dilation is a draft-side post-op only.** A pixel-tight mask leaves a halo of the original object's
edge, which the inpainter happily reconstructs — reintroducing the thing we asked it to replace
(`../rag-edit/ragedit/mask.py:104-107`). Dilating the *reference* mask would matte background into
the cutout, so `mask_reference` never dilates. Feathering happens later still, at composite time
(`kontext_engine.py:473`, doc 2).

**`dilate_px` becomes an operator knob** in `configs/pipeline.yaml` as `mask_dilate_px: 12`,
alongside the existing numeric knobs. No code edit to change it.

### Why the reference crop is not here

`kontext_engine.prepare_reference` (L415-467) already derives a padded bbox from a mask, optionally
mattes onto neutral grey, and crops — and its `crop | crop_matte | none` modes are left to an
ablation rather than asserted. Its masker contract (L429-431) is:

```
masker: callable(image, keyword) -> (float mask HxW, info)
```

`mask_reference` satisfies that contract directly. Re-implementing the crop inside `mask.py` would
fork a function that already exists and has an experiment planned against it.

## 6. Prototype sources per arm

Parent spec §3 orders the loop `formulate → retrieve → mask_draft`, so retrieved hits are available
before either mask is cut.

| arm | prototype source | leakage guard |
|---|---|---|
| `oracle` | `gt_refs` from `dataset.yaml` | the case's own draft never enters its own prototype |
| `full` | the retrieved hits | never reads `gt_refs` — separate constructor, as in plan 4 |

Both reuse `prototype.bank_from_gt_refs` / `bank_from_retrieval` unchanged.

**Consequence to state, not bury:** on the `full` arm this makes mask geometry depend on retrieval
quality, so a full-arm failure may be a masking failure wearing a generation failure's clothes. The
oracle/full split is what separates them, and `selected_by` plus `n_candidates` in the trace is what
makes the separation checkable per case.

**This is a weaker circularity than the verifier's.** Using `gt_refs` inside Stream A would
contaminate the *judgment* of whether the pipeline worked — that is why DINOv3 is barred there.
Here it only chooses which region to repaint; the identity metric still scores the repainted output.
On the oracle arm, perfect references are the arm's declared premise.

## 7. Failure paths

**`None`, never an empty mask.** A concept that does not ground means "do not repair this case" —
the same distinction the verifier draws between ABSTAIN and MISSING. An all-zeros mask would
silently repaint nothing and read downstream as a successful attempt.

| condition | result |
|---|---|
| DINO returns no candidate | `None`; the case is recorded as unrepairable and keeps its draft |
| SAM returns an empty mask for the winner | `None`, with the box recorded in the trace |
| A bank is supplied but lacks the phrase | falls back to confidence, `selected_by="confidence"` |

`PrototypeBank.get` already returns `None` for an unknown phrase rather than raising, so the last
row needs no new machinery.

## 8. GPU lifecycle

`to_cpu` and `free` are part of the interface, not an afterthought. `../rag-edit`'s last run died at
`ragedit/mask.py:49` with `torch.OutOfMemoryError` because other processes held the card, and
SAM-ViT-H allocates roughly 1 GB of windowed-attention activations at 1024 px on top of 2.4 GB of
weights. Doc 3's stage-batching requires masking to be fully evictable before FLUX loads —
Qwen-7B (~16 GB) and FLUX-nf4 (~12 GB) already do not co-fit on a 24 GB card.

Masking for every case finishes before any inpainting starts. Callers precompute masks, then `free()`.

## 9. Out of scope

- Reference cropping and matting (doc 2 — already written in `kontext_engine.py`).
- Feathering and compositing (doc 2).
- The retry loop, stage-batching and persisted state (doc 3).
- Any metric over masks. Mask quality is judged by the pipeline's output, not scored directly.
- Multi-concept repair. `Verdict.target` names one concept; masking serves that one.

## 10. Risks

**M1 — Whole-versus-part is asserted, not measured.** ~~The claim that a stack-of-books prototype
outranks a single-book crop is reasoning from plan 4's structure, not a number.~~ **Measured
2026-07-28: real, but on `boston_bull`, not `stack_of_books`.** Confidence selects the dog's head;
object-cropped prototype selection selects the whole animal. Resolved, and the diagnostic
(`n_candidates` + `selected_by`) is what showed it.

**M2 — The coarse term may ground the wrong thing entirely. CONFIRMED, and worse than written.**
`sushi` carries `coarse: "food"`, and grounding it boxes the whole plate — but so does the *fine*
phrase `sushi`. SAM then segments the plate surface, leaving every piece of food unmasked. This is
not a vocabulary problem fixable by editing `coarse:`; SAM prefers the large coherent plate region
over a scatter of small pieces. At least one of 22 cases will therefore produce a meaningless
repair, and nothing in this module can fix it.

**M5 — Prototypes built from whole photographs measure size, not identity.** `gt_refs` are
full-frame stock images, so a prototype built from them encodes framing and background — "a
photograph of a durian" rather than "a durian". Crops that look like photographs then outscore tight
object crops whatever they contain. Measured: whole-photo prototypes selected the *backdrop* on
`durian` and the saguaro *plus every desert shrub* on `cactus`, and in 3 of 4 multi-candidate cases
picked a strictly larger box, never a smaller one.

Fixed by `crop_to_object`: each reference is cropped to its object before embedding, with the same
padding `prepare_reference` uses, so both sides of the comparison frame alike. Object-cropped
prototypes are correct on 4 of 4 — they keep the `boston_bull` fix and revert `cactus` to exactly
the confidence box. **Whole-photo prototypes must never be the default**; `--whole-photo-protos`
keeps them available only as the diagnostic that reproduces the bias.

**M3 — 0.917 does not transfer unexamined.** It was measured on CLIP ViT-B/32 prototypes over
composited breed pairs. Here the encoder is SigLIP-SO400M-384 and the task is whole-versus-part.
The structure transfers; the number does not, and should not be quoted for this module.

**M4 — SAM on 8 candidates is 8 forwards per image.** Acceptable on a loaded model, but it makes
masking's cost scale with detector noise rather than with scene complexity.

## 11. Testing

Same standard as the existing suite: dependency injection throughout, fakes for every model, no
module-level loading, no GPU. The 260 existing tests stay green.

- `select_box` — confidence rule with no bank; prototype rule with a bank and several candidates;
  confidence rule when only one candidate survives even with a bank supplied; fallback when the bank
  lacks the phrase, asserting `selected_by`.
- A whole-versus-part fixture: two candidates, the larger scoring lower on DINO confidence and higher
  against the prototype. Pins that the prototype rule wins — this is the `stack_of_books` scenario in
  miniature.
- `mask_draft` dilates and `mask_reference` does not, verified by mask area, not by inspecting calls.
- `None` on every failure path in §7, with a fake detector returning no boxes and a fake SAM
  returning an empty mask.
- `mask_reference` satisfies `prepare_reference`'s `(mask, info)` contract, asserted by calling the
  real `prepare_reference` with a fake masker built from `mask_reference`'s output.
- `free()` drops the model references, asserted against a fake that records eviction.
- Grounding phrase per wrapper: `mask_draft` is called with `coarse`, `mask_reference` with
  `concept`. A guard test, so a later refactor cannot silently swap them.

GPU paths are verified by one 1–2 case smoke run at the end of doc 2, when there is an output to
look at, not by unit tests.
