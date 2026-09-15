# Regeneration — repainting the masked region from a reference

**Date:** 2026-07-28
**Parent spec:** `docs/superpowers/specs/2026-07-25-rag-regen-design.md` (§2 Modules, §3 The loop, §6 Stage 2, §10 step 5)
**Depends on:** `ragregen/mask.py` (doc 1)
**Port source:** `../ImageRAG/scripts/kontext_engine.py`
**Prior finding:** `docs/findings/2026-07-28-masking-result.md`

**Doc 2 of 4** in the regeneration half: masking → **regeneration** → orchestration → metrics.

---

## 1. The claim

Given a draft, a mask, and a reference image of the real concept, produce a repaired image in which
only the masked region has changed. Two mechanisms, compared on one dataset:

| mechanism | how the region is filled | diffusion? |
|---|---|---|
| **inpaint** | FLUX.1-Kontext repaints the region conditioned on the reference | yes |
| **stitch** | the reference's segmented object is pasted into the region | no |

The pilot picks one (parent spec §6). Until it runs, neither is the default.

## 2. What already exists

`inpaint` is a port. `stitch` is new code — it appears three times in the parent spec, is defined
nowhere, and exists in neither reference project. ImageRAG's `PIPELINE_PLAN.md:60` names the
intent — a tight object mask is "what would be pasted in" — and stops there.

| piece | status | where |
|---|---|---|
| Pipeline load, nf4 quantisation, offload policy | **done** | `kontext_engine.py:533-586` |
| `validate_inputs` — binarise, largest CC, dilate, blur | **done** | `kontext_engine.py:331-385` |
| `prepare_reference` — crop / matte | **done** | `kontext_engine.py:415-467` |
| `edit()` — the single conditioned call | **done** | `kontext_engine.py:592-660` |
| `feather_alpha` / `composite_back` | **done** | `kontext_engine.py:473-518` |
| `preflight` — refuse rather than thrash | **done** | `kontext_engine.py:180-255` |
| **`stitch`** | **new** | — |

Porting means moving these into `ragregen/regen.py` under this project's rules — dependency
injection, no module-level loading, no network at import — not rewriting them. The arithmetic in
`composite_back` is load-bearing and is reproduced exactly.

## 3. Two integration hazards the port must fix

**Double dilation.** `validate_inputs` dilates by `cfg.dilate_px` (default 12), and `mask_draft`
already dilated by `mask_dilate_px` (default 12). Chained, the mask grows ~24px and the repaint
region silently doubles its slack.

**Dilation is owned by `mask_draft`.** It is the operator knob documented in `pipeline.yaml`, it is
tested, and the mask is already the shape the inpainter should see. The port therefore constructs
`KontextConfig(dilate_px=0)` and a test pins that a mask entering `regen` comes out the same size it
went in.

**`largest_cc` drops components.** `validate_inputs` keeps only the largest connected component by
default. For `sushi` — many separate pieces — that discards most of the mask. Given the finding that
`sushi` masks the plate anyway, this is not the case that saves it, but the behaviour must be
explicit: the port sets `largest_cc=False` and lets doc 1's mask stand as authored. A mask with many
components is a fact about the concept, not noise to clean up.

## 4. `stitch`

```
cutout   = reference RGBA, alpha = mask_reference's silhouette
target   = draft mask's bounding box
scale    = fit-inside(cutout.size, target.size), aspect preserved
place    = centred on the draft mask's centroid
paste    = feather_alpha over the cutout's own silhouette
residual = draft_mask AND NOT pasted_alpha  ->  cv2.inpaint (TELEA)
```

**Fit-inside, not fill.** The object must never be cropped by its own hole.

**One compositing rule for both mechanisms.** `stitch` reuses `feather_alpha`, so the preservation
metric measures the same three zones — outside bit-identical, feathered band, interior replaced —
whichever mechanism produced the image. Two blending rules would make the mechanisms
incomparable on exactly the metric meant to compare them.

**The residual is the hard part.** `mask_draft` is a dilated silhouette of the *wrong* object. The
reference cutout has a different shape, so pasting leaves a crescent of mask where the original
object's pixels still show — fragments of a generic leopard around a pasted Amur leopard, which
reads as a generation artifact rather than as what it is.

Filled with `cv2.inpaint` (TELEA) from the surrounding background. Classical, milliseconds, and
`cv2` is already a dependency (`mask.py` dilates with it). It keeps `stitch` genuinely
diffusion-free, which is the entire point of having it as an arm. **It will look soft where it
fills**, and that is reported, not hidden.

**No colour matching.** Lighting and white-balance transfer is a research problem of its own and
doing it badly is worse than not doing it. This is the most likely reason `stitch` loses the
ablation — see R3.

## 5. Interfaces

```python
@dataclass(frozen=True)
class RegenResult:
    image: Image.Image          # composited, at the ORIGINAL resolution
    raw: Image.Image | None     # pre-composite model output; None for stitch
    alpha: np.ndarray           # the three-zone alpha actually applied
    mechanism: str              # "inpaint" | "stitch" -- recorded, never inferred
    meta: dict                  # seed, steps, sizes, seconds, peak_vram_gb


class Inpainter:
    """Wraps FluxKontextInpaintPipeline. Injected; never loads at import."""
    def __init__(self, cfg: KontextConfig | None = None): ...
    def load(self) -> "Inpainter"
    def regen(self, draft, mask, reference, prompt, *, seed=None,
              use_reference=True) -> RegenResult
    def free(self) -> None


def cutout_from(reference, mask_result) -> Image.Image     # RGBA
def stitch(draft, mask, cutout, *, feather_px=6) -> RegenResult
```

`cutout_from` turns a reference image plus doc 1's `MaskResult` into an RGBA image whose alpha is
the object's silhouette, cropped to its bounding box. It is the only new thing `stitch` needs from
the masking side, and keeping it separate means `stitch` itself takes pixels rather than a
`MaskResult` — so it can be tested on synthetic arrays with no masker in sight.

`KontextConfig` ports across unchanged apart from the two defaults §3 pins (`dilate_px=0`,
`largest_cc=False`). It is the record of the whole tunable state and doc 3's trace will persist it.

`stitch` is a free function: it holds no model and needs no lifecycle. That asymmetry is the honest
shape — one mechanism owns 12 GB of weights and the other owns none.

**`use_reference=False` is the built-in ablation**, running the identical call with
`image_reference=None`. It isolates what the reference contributes from what the mask and prompt
contribute, and it costs one extra generation per case.

**`padding_mask_crop` stays `None` whenever a reference is passed** — the pipeline preprocesses the
reference using the *source's* crop region, cropping the reference to an unrelated rectangle
(`kontext_engine.py:9-13`). The port keeps this as a raised `ValueError`, not a comment.

## 6. The end-to-end script

`scripts/repair_case.py` — one case, oracle arm, no scheduler:

```
draft (or a gt_ref stand-in) -> mask_draft -> mask_reference on a gt_ref
   -> {inpaint | stitch} -> composite -> write before/after/alpha PNGs
```

This is the first point in the project where a repaired image exists to look at. It deliberately
does **not** use retrieval, a queue, or the VLM: the oracle arm needs none of them, and every one
of them is a separate failure surface between the input and the output.

**Drafts do not exist yet.** `outputs/` is empty. Until a `screen` run produces drafts, the script
accepts any image as the draft — including a `gt_ref`, which repairs a correct image and should
therefore change little. That is a weak but real end-to-end check, and it is honest about being one.

## 7. GPU reality

| | |
|---|---|
| FLUX-nf4 resident | ~12 GB |
| `preflight` refuses below | 12 GB (`MIN_VRAM_GB`) |
| free on the shared 4090 at time of writing | ~10 GB, 14 GB held by two other researchers |
| pipeline load | 5 min 21 s |
| one generation, 28 steps, nf4 | ~2 min |

**The inpaint half may be unrunnable until the card frees up**, and `preflight` will say so rather
than thrash. `stitch` is pure array work and runs on CPU today. The test plan (§10) is therefore
built so that everything except the diffusion call is verifiable without a GPU — not as a
convenience, but because otherwise doc 2 cannot be validated at all on a contended machine.

## 8. Failure paths

| condition | result |
|---|---|
| `mask_draft` returned `None` | no repair attempted; the draft stands. Not an error |
| `mask_reference` returned `None` | `inpaint` proceeds with the uncropped reference; `cutout_from` returns `None` and the caller skips `stitch` for that case rather than calling it |
| mask entirely black or white | `ValueError` from `validate_inputs`, naming which |
| mask covers >50% of the frame | proceed, with a note — overwhelmingly the signature of an inverted mask |
| VRAM below `MIN_VRAM_GB` | `preflight` refuses before loading |

`stitch` requiring a reference cutout while `inpaint` does not is a real asymmetry, not an
oversight: pasting needs a segmented object, conditioning does not.

## 9. Out of scope

- Retrieval, query formulation, the retry loop, stage-batching — doc 3.
- All metrics: CLIP, SigLIP, DINO identity, preservation — doc 4. §5's `alpha` is what doc 4 will
  measure preservation against.
- Colour and lighting transfer for `stitch` (R3).
- The `sushi` plate failure. It is a masking result, unfixable here, and its consequence is simply
  that one case's repair is meaningless.
- Prompt construction. Kontext is instruction-driven and `validate_inputs` rejects an empty prompt;
  the instruction text is doc 3's, alongside query formulation.

## 10. Risks

**R1 — `stitch` is unvalidated by anything.** `inpaint` is a port of code with a working example
suite behind it. `stitch` has never been run. Its four decisions — fit-inside, centroid placement,
feathered silhouette, classical residual fill — are all reasoned, none measured.

**R2 — The residual fill is visible.** `cv2.inpaint` diffuses surrounding colour into the crescent.
On a busy background it will smear. The affected area is a function of how much the reference
object's silhouette differs from the draft object's, which is exactly the fine-grained cases where
the two differ most.

**R3 — `stitch` will probably lose, for a reason unrelated to retrieval.** A pasted object carries
its source lighting, white balance and perspective. If it loses, that is not evidence that
reference-guided repair fails — only that naive compositing does. Reporting it as a mechanism
ablation without that caveat would overclaim.

**R4 — Double-dilation and `largest_cc` are silent when wrong.** Both produce a plausible image with
a subtly wrong region. §3 pins both, and the tests in §10 exist because neither would be caught by
looking at output.

**R5 — Kontext's output resolution is not the input's.** `composite_back` resizes with LANCZOS
before blending. Every metric must be computed at one pinned resolution or resampling shows up as a
real delta (parent spec §5).

## 11. Testing

Same standard as the rest of the suite: dependency injection, fakes for every model, no GPU in the
default run. The 286 existing tests stay green.

**Without a GPU:**

- `stitch` end to end against a synthetic draft, mask and cutout: the pasted object lands inside the
  mask bbox, aspect ratio is preserved, and pixels outside the mask are **bit-identical** to the
  draft.
- Fit-inside geometry: a cutout wider than its target and one taller than its target, both scaled to
  fit without cropping.
- The residual is filled — no pixel inside the draft mask retains its original value after a paste
  whose cutout is strictly smaller than the mask.
- `feather_alpha` produces the three zones, and `stitch` and the inpaint composite path apply the
  *same* function — asserted by comparing alphas, so the mechanisms stay comparable.
- Double-dilation guard: a mask through `regen`'s input preparation comes out with the same area it
  went in, pinning `dilate_px=0`.
- `largest_cc=False`: a two-component mask survives with both components.
- `padding_mask_crop` with a reference raises, naming the pipeline behaviour.
- `RegenResult.mechanism` records which path ran; a fake inpainter cannot report `"stitch"`.
- Failure paths in §8, each returning what the table says rather than raising.

**Requiring a GPU** (`@pytest.mark.gpu`, deselected by default): one `Inpainter.regen` call on a
small synthetic image, asserting the output is the original's size and that pixels outside the mask
are unchanged. This is the only test that needs FLUX, and it is the first thing to run when the card
frees up.

**GPU smoke:** `scripts/repair_case.py` on one case per mechanism, outputs inspected by eye. That is
the point of doc 2 — the first repaired image anyone can look at.
