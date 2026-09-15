# Masking on the real dataset — box selection, and a size bias that looked like identity

**Date:** 2026-07-28
**Spec:** `docs/superpowers/specs/2026-07-28-masking-design.md`
**Runs:** `outputs/smoke_mask_all/` (22 cases, confidence), `outputs/smoke_mask_proto2/` and
`outputs/smoke_mask_cropproto/` (4 cases, prototype selection)
**Code:** `ragregen/mask.py`, `ragregen/verify/prototype.py`, `scripts/smoke_mask.py`

---

## 1. What was run

`mask_draft` and `mask_reference` over the `gt_ref` photographs of all 22 cases in
`configs/dataset.yaml`, then the four multi-candidate cases again under two prototype
configurations.

**These are reference photographs, not drafts.** No drafts exist — `outputs/` is empty and the
labelled run the earlier findings cite is gone from disk. A `gt_ref` is the concept rendered
correctly, so `mask_draft`'s coarse grounding was exercised on an easier image than it will face in
the pipeline. Every number here is "does this ground at all", not a repair-quality measurement.

`outputs/` is gitignored, so the tables below are the durable record.

## 2. All 22 cases ground

No case failed to produce a mask from either wrapper. On 18 of 22 the draft and reference boxes
agree within a few pixels — the coarse and fine phrases locate the same object, which is the
assumption `mask_draft` rests on.

Four cases returned more than one candidate: `boston_bull` (2), `durian` (3),
`zalophus_californianus` (2 on the reference side), `cactus` (8), `stack_of_books` (8). Selection
only matters for these; for the other 18 every rule agrees by construction.

## 3. `sushi` — SAM segments the plate, for both phrases

| wrapper | phrase | candidates | mask fraction |
|---|---|---|---|
| draft | `food` | 1 | 0.208 |
| reference | `sushi` | 1 | 0.078 |

Both phrases box the whole plate; SAM then returns the plate surface and rim, leaving every piece of
food unmasked. Repainting that mask repaints the dish.

**This is not a vocabulary problem.** The spec's risk M2 predicted the *coarse* term would ground
badly and implied editing `coarse:` would fix it. The fine phrase fails identically, so the cause is
SAM preferring one large coherent region over a scatter of small pieces. Nothing in the masking
module addresses it, and at least one of 22 cases will therefore produce a meaningless repair.

## 4. Box selection: three rules on four cases

| case | confidence | whole-photo prototype | object-cropped prototype |
|---|---|---|---|
| `boston_bull` | head only, 0.052 ✗ | whole dog, 0.255 ✓ | whole dog, 0.255 ✓ |
| `cactus` | saguaro, 0.103 ✓ | saguaro + every shrub, 0.254 ✗ | saguaro, 0.103 ✓ |
| `durian` | opened flesh segment, 0.154 | the black backdrop, 0.530 ✗ | the spiky husk, 0.219 ✓ |
| `stack_of_books` | whole stack, 0.362 ✓ | whole stack ✓ | whole stack ✓ |
| **correct** | **3 / 4** | **2 / 4** | **4 / 4** |

### The spec's motivating example was wrong

§4 argued the classifier was needed because grounding `books` returns one box per book. It does
return 8 candidates — but the highest-confidence one *is* the whole stack. `stack_of_books` never
needed help.

`boston_bull` is the real case: grounding `dog` returns the head and the whole animal, and
confidence takes the head. The mechanism was justified by the wrong example.

### The scoring phrase must differ from the grounded phrase

The first prototype run reported `selected_by = confidence` on every case. The bank held 22 phrases
— exactly the 22 `concept` values and no coarse terms, because `bank_from_gt_refs` sources coarse
prototypes from `dataset.coarse_refs`, which no dataset carries. `mask_draft` grounded `dog`, asked
the bank to score `dog`, got `None`, and fell back silently.

The fix is not to author `coarse_refs`. Scoring `dog`-ness cannot separate a head from a whole dog —
both crops are dog. "Which is the Boston bull" can. `select_box` now takes the phrase to **score**,
which `mask_draft` sets to the concept while still grounding the coarse term.

## 5. The size bias, and why the first win was suspect

Whole-photo prototypes fixed `boston_bull` but broke `cactus` and `durian`. In 3 of the 4 cases they
selected a **strictly larger box**; in the fourth they tied. They never selected a smaller one.

That is not what identity discrimination looks like. `gt_refs` are full-frame stock photographs, so
a prototype built from them encodes framing, background and composition — *"a photograph of a
durian"* rather than *"a durian"*. Crops resembling full photographs score higher regardless of
content, which is a size preference wearing identity's clothes. It is the same species of confound
plan 4's finding named, where box selection favoured one side before the margin was taken.

Under that reading the `boston_bull` "win" was unsafe: the whole-dog crop is more photo-like than
the head crop, so the bias would produce the right answer there for the wrong reason.

**`crop_to_object` resolves it.** Each reference is cropped to its object before embedding, using the
same padding arithmetic as `kontext_engine.prepare_reference`, so prototype and candidate are framed
alike. Object-cropped prototypes are correct on 4 of 4, and the two diagnostics that matter are:

- **`cactus` reverts to exactly the confidence box** `(171, 139, 277, 604)`. Where confidence was
  already right, the cropped prototype agrees rather than inventing a larger region.
- **`boston_bull` keeps the fix.** So the win was not purely the size bias — there is identity
  signal underneath it.

A third consequence: on `durian` the draft mask `(194, 393, 1401, 1599)` and the reference mask
`(189, 391, 1413, 1601)` now select the *same part* of the object. Under confidence they picked
different parts — flesh versus husk. Since the pipeline pastes the reference cutout into the draft's
masked region, that disagreement would have copied a husk into a hole shaped like flesh.

## 6. Co-residency, and a design consequence

Prototype selection OOMed on the shared 4090 even for a single case: it makes SigLIP resident
*alongside* DINO and SAM (~9.4 GB in-process, against ~14 GB held by two other researchers).
Confidence selection never does. Both prototype runs above were completed on CPU.

This exposes a waste in the spec. §3 has SAM segmenting every candidate "matching the measured
harness" — but selection scores the **rectangular crop** and never reads a candidate's mask. Only
the winner's is used. The harness segmented all of them because its own evaluation needed them; we
have no such need. Deferring SAM to the winner cuts masking cost 8× and lets the three models run in
sequence rather than together:

```
DINO -> candidates -> free  |  encoder -> scores -> free  |  SAM -> winner's mask -> free
```

That is doc 3's stage-batching applied inside one module, and it is now the recommended shape.

## 7. Dilation is scale-dependent

`dilate_px = 12` is absolute, and source images range from 541×567 to 3024×3022. Mask inflation
ranges from 1.07× (`bonsai_tree`) to 2.67× (`sushi`) as a result — the slack the inpainter receives
varies ~5× on nothing but the file's pixel count.

Probably moot in production: real drafts come out of Kontext at a pinned ~1 MP and `mask_reference`
never dilates. Recorded so it is pinned deliberately rather than by accident.

## 8. What this does and does not establish

**Established.** All 22 phrases ground. `sushi` masks the container rather than the contents, for
both phrases. Whole-photo prototypes carry a size bias that produces wrong masks. Object-cropped
prototypes are correct on all four cases where selection matters.

**Not established.** n = 4 informative cases, chosen because they were the multi-candidate ones —
the other 18 are single-candidate, where every rule agrees by construction. Four cases cannot
separate "object-cropped prototypes are correct" from "object-cropped prototypes were correct
here". Nothing has been measured on an actual FLUX draft, which is the image the pipeline will face
and is harder than a stock photograph. No repair has been generated, so no mask has been judged by
its effect on an output.

**The cheap diagnostic still worth running** is whether prototype score correlates with crop area
across all candidates. If it does even after cropping, the residual mechanism is still partly size,
and that belongs in the record as a measured caveat rather than an inference from four cases.

---

## 9. Reproduced, 2026-07-29

**Run:** `outputs/smoke_mask_reverify/` — the same four cases plus
`zalophus_californianus`, object-cropped prototypes, CPU.

Every number §4 and §5 quote came back identical, boxes included:

| case | wrapper | cands | `selected_by` | box | mask frac | §4 said |
|---|---|---|---|---|---|---|
| `boston_bull` | draft | 2 | prototype | `(64, 70, 477, 429)` | 0.255 | whole dog, 0.255 ✓ |
| `cactus` | draft | 8 | prototype | `(171, 139, 277, 604)` | 0.103 | saguaro, 0.103 ✓ |
| `durian` | draft | 3 | prototype | `(194, 393, 1401, 1599)` | 0.219 | spiky husk, 0.219 ✓ |
| `stack_of_books` | draft | 8 | prototype | `(112, 95, 379, 470)` | 0.362 | whole stack, 0.362 ✓ |

Both §5 diagnostics hold exactly. `cactus` returns the confidence box
`(171, 139, 277, 604)` to the digit. `durian`'s draft and reference boxes —
`(194, 393, 1401, 1599)` and `(189, 391, 1413, 1601)` — are unchanged, so the two wrappers still
select the same part of the object.

**One case §4 did not cover.** `zalophus_californianus` is multi-candidate only on the reference
side (2 candidates; the draft grounds 1, so selection cannot matter there). The prototype picked
`(5, 59, 359, 338)` against the draft's `(4, 60, 360, 337)` — the same animal, within a pixel. A
fifth case where prototype selection agrees with confidence rather than overriding it, which is the
behaviour §5 wanted from `cactus`.

**This does not widen §8.** It is the same four informative cases run twice; a reproduction, not new
evidence. n is still 4, still stock photographs, still no FLUX draft and no repair judged by its
output. What it does establish is that the rule is deterministic and the numbers in §4 were not a
one-off — worth knowing before doc 3 runs it unattended over 22 cases.
