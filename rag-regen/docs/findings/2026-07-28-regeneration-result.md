# Regeneration end to end — the mechanism was fine, the mask was the wrong part of the dog

**Date:** 2026-07-28
**Spec:** `docs/superpowers/specs/2026-07-28-regeneration-design.md`
**Plan:** `docs/superpowers/plans/2026-07-28-regeneration.md`
**Runs:** `outputs/repair/boston_bull_stitch/` (confidence selection),
`outputs/repair_proto/boston_bull_stitch/` (object-cropped prototypes)
**Code:** `ragregen/composite.py`, `ragregen/regen.py`, `scripts/repair_case.py`

---

## 1. What was run

`scripts/repair_case.py` on `boston_bull`, the oracle arm, both selection rules. The stitch
mechanism only — see §4 for why the inpaint path is unverified.

**This is not a repair measurement.** No drafts exist yet, so `--draft` defaulted to the case's
first `gt_ref`: the pipeline was asked to repair an image that was already correct. It exercises
the wiring end to end and nothing more. Every claim below is "does this mechanism do what it says",
not "does repair improve anything".

`outputs/` is gitignored, so the tables here are the durable record.

## 2. Confidence selection masks the head, and the repair is meaningless

| | mask box | box size | cutout placed | scale | `selected_by` |
|---|---|---|---|---|---|
| confidence | `[363, 61, 489, 219]` | 126×158 | `(369, 65)` | 0.374 | `confidence` |
| object-cropped prototypes | `[55, 61, 489, 436]` | 434×375 | `(163, 62)` | 0.885 | `prototype` |

Both runs grounded the same phrase — `dog`, the coarse term — and DINO returned the same two
candidates. Only the tiebreak differs. Under confidence the head wins, so `stitch` scaled a whole
Boston bull down to 0.374 and pasted it into the head's bounding box: a small complete dog standing
on the neck of the original animal, whose body is untouched because it lies outside the mask.

This is doc 1's measurement reproduced through a second mechanism
(`findings/2026-07-28-masking-result.md` §4: *"grounding `dog` returns the head and the whole
animal, and confidence takes the head"*). It is worth recording separately because the consequence
is invisible in doc 1's tables — a box that is merely *smaller than ideal* becomes a visibly absurd
image once something is pasted into it.

**`--prototypes` is therefore not optional on multi-candidate cases.** The flag now exists on
`repair_case.py` and the RUNBOOK row says so.

### The scoring phrase is what does the work

`mask_draft` was passed `score_phrase="Boston bull"` in *both* runs. It changed nothing in the first
one: `_prototype_scores` returns `[]` when `prototypes is None`, and `select_box` falls through to
`max(..., key=dino_conf)`. The fine phrase only becomes load-bearing once there is a bank to score
against — which is exactly the asymmetry `select_box`'s docstring describes, confirmed at the
mechanism level.

## 3. Stitch does what it claims, and looks like what R3 predicted

With the correct mask: the reference's dog lands in the masked region at 0.885 scale, fit-inside so
the object is never cropped by its own hole, centred on the mask centroid.

The visible defect is the residual — the mask is a dilated dog silhouette, the cutout is a bbox, and
the difference is filled classically. It streaks. This is **R3, predicted in the design**, and it is
the reason the inpaint arm exists rather than a defect to fix in `stitch`: a diffusion-free
mechanism has nothing to synthesise the gap with.

Pixels outside the mask are bit-identical to the draft, as the composite tests assert with `==`.

## 4. The inpaint path is unverified

Another researcher's job took the card mid-session (15.2 GB, peaking at 23.1 of 24.5 GB used).
FLUX-nf4 needs 12 GB. The guard refused, which is correct behaviour:

```
[repair] only 2.3 GB free; FLUX-nf4 needs 12.0 GB. Another job is holding the
card -- wait rather than thrash. `nvidia-smi` shows who.
```

The guard was **not** lowered to make it fit. `Inpainter.load()` is therefore exercised only up to
`check_vram`; the diffusion call itself has never run on real weights. Everything below it is
covered on CPU against an injected pipeline, which is what the `composite.py` / `regen.py` split
was for — but "the seam is tested" is not "the model produces a repair", and nobody should read
this document as claiming otherwise.

To close it, when the card frees:

```bash
./scripts/run.sh repair --case boston_bull --mechanism inpaint --prototypes
```

## 5. Two operational notes

**The masker needs a CPU fallback on a contended card.** The first prototype run died in
`DinoDetector`'s `.to(cuda)`, not in the diffusion path — grounding is small but it is not free, and
`--device cpu` is what let the run complete at all. A whole-case CPU run (DINO + SAM + SigLIP over
one draft and three references) takes a few minutes.

**`--proto-device` defaults to `cpu` deliberately.** The bank makes SigLIP resident beside DINO and
SAM, which is what OOMed the shared 4090 in doc 1 (`findings/2026-07-28-masking-result.md` §6).
Both prototype runs there were completed on CPU for the same reason.

One run exited 139 (SIGSEGV) during CPU model loading and did not reproduce in three subsequent
attempts. Recorded because an unreproduced segfault is worth knowing about, not because there is
anything actionable yet.

## 6. Status

324 tests pass, 11 `@pytest.mark.gpu` deselected. Tasks 1–9 of the regeneration plan are complete.
The one claim the doc cannot make is the one that needs the GPU: that FLUX.1-Kontext, given a
reference, repairs anything.
