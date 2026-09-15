# Evaluation design — doc 4 of 4

**Date:** 2026-07-30
**Prior docs:** `2026-07-28-masking-design.md` (1), `2026-07-28-regeneration-design.md` (2),
`2026-07-28-orchestration-design.md` (3)
**Prior findings:** `docs/findings/2026-07-29-orchestration-result.md`
**Produces:** `ragregen/metrics.py`, `scripts/report.py`, `run.sh report`

---

## 1. What this doc is for

Docs 1–3 built a pipeline that repairs cases. Nothing yet says whether the repairs are any good.
This doc turns a finished run into a number that survives scrutiny.

The claim under test, from `RUNBOOK.md` §5.1: **`no_rag → oracle` is the contribution.** If
reference-guided repair does not beat the un-retrieved draft, retrieval quality is irrelevant and
improving the database will not help.

**Non-goals.** The `full` arm is out of scope — it needs a retrieval corpus, which `RUNBOOK.md` line 7
makes the operator's input, not this project's deliverable. The arm axis is built so `full` drops in
without redesign. The held-out VLM judge that design §5 offers as a human-study substitute is also
out of scope for v1: DINO carries the claim, and the report states the deviation rather than implying
parity with the paper's 46-participant study.

## 2. The leakage problem, and what we do about it

The oracle arm regenerates *using* `gt_refs`. DINO scores the output *against* `gt_refs`. Scoring an
edit with the image it was given is self-marking, and the project already applies exactly this
reasoning elsewhere — `RUNBOOK.md` line 247 forbids quoting `--proto-arm ceiling` as a verifier
result "because `gt_refs` are also the DINO metric's target."

Design §5 guards the *encoder* version of this (retrieval encoder ≠ eval encoder) but not the image
version. This doc closes it.

**The split.** For each case, the last `h` references are held out: the oracle arm may edit with
`gt_refs[:-h]`, and DINO scores against `gt_refs[-h:]`. `h` is `eval.heldout_refs`, default 1.

The dataset has 3 references for 17 cases, 4 for four, 5 for one, so `h=1` always leaves at least two
references to edit with and exactly one to score against on the thin cases. That single held-out
image is noisier than an average over three, but it is unbiased, and unbiased-and-noisy is reportable
where biased-and-tight is not.

**Both numbers are reported.** The held-out DINO is the result. The all-refs DINO is printed beside
it, labelled a ceiling, never quoted alone. The gap between them prices the leakage, which is
information rather than embarrassment.

**Round clamping is required, not optional.** `run_pipeline.py:249` does
`reference = Image.open(refs[attempt - 1])`, and `_regen_one` raises if `cutout_<attempt>.png` is
absent. Shrinking the edit pool to 2 while `retry_budget` stays 3 would make attempt 3 raise
`ValueError`, and `run_stage` would mark a perfectly good case `failed`. The round count must clamp
to `len(edit refs)`. Getting this wrong turns a hygiene fix into fabricated failures, so it is pinned
by a test.

## 3. What gets measured

| metric | model | inputs | role |
|---|---|---|---|
| **DINO** | DINOv3-L | output crop ↔ held-out refs | **the headline** |
| CLIP | open-CLIP ViT-L/14 | prompt ↔ output | paper comparability |
| SigLIP | SigLIP-**base** | prompt ↔ output | paper comparability, secondary |
| Preservation | none (pixels) | output vs draft, three zones | anti-gaming guard |

Read per `RUNBOOK.md` §5.2: DINO is the claim, CLIP/SigLIP are context, and a flat or slightly
negative CLIP delta beside a clear DINO gain is a success. Preservation is always read next to DINO —
a method that improves identity by repainting the canvas has not solved the problem. Verifier
pass-rate is not in this table and never becomes a headline; the pipeline optimises against it.

**Encoder hygiene.** `configs/pipeline.yaml` sets `retriever` and `crop_scorer` to
`siglip_so400m_384`. The eval encoders above are different checkpoints, and config load **fails** if
an eval encoder equals either — self-marking should be impossible to configure, not merely
discouraged.

**Resolution pinning.** Every metric computes at the draft's size. The compositor already guarantees
the output is the draft's size with pixels outside the mask bit-identical, so this costs nothing and
removes resampling artifacts that would otherwise read as real deltas. Mismatched sizes are an error,
not a resize.

**The DINO crop.** Identity is a property of the object, not the canvas. `crop_to_mask` takes the
mask bounding box with 15% padding, on both the output and each held-out reference, so a small
animal in a large scene is not averaged into its background. This is doc 2's finding
(`2026-07-27-finegrained-result.md` §3a) applied to the metric.

## 4. Denominators — the field you must not use

`docs/findings/2026-07-29-orchestration-result.md` §2 records that `status: "unrepaired"` is written
by two paths meaning opposite things: `_resolve` (the draft passed, no repair was needed) and
`_finalise` (the draft failed and nothing beat it). A repair rate computed from `status` silently
mixes healthy cases into the failure count.

The report therefore partitions on `scores.json`'s **draft verdict**, never on `status`:

| bucket | definition |
|---|---|
| healthy | `scores["draft"][0]` is true — never needed repair |
| needed repair | `scores["draft"][0]` is false |
| repaired | needed repair **and** `best != "draft"` |

**Repair rate = repaired / needed repair.** Healthy cases appear in the report as their own line and
are excluded from that denominator.

**Healthy cases have no mask, and that constrains what can be measured for them.** `mask` is an
attempt-free stage that runs only for cases entering repair, so a case resolved at round 0 has just
`streams.json` and `scores.json` on disk — verified on `african_grey_parrot` in
`findings/2026-07-29-orchestration-result.md` §3. There is no bounding box to crop to.

Therefore:

- **Cropped DINO is computed only for cases that entered repair.** Those are the only cases where
  `no_rag → oracle` means anything anyway: a healthy case has no oracle output, its `best` is the
  draft, and no attempt was ever rendered.
- **Healthy cases get whole-image DINO**, in a separate column marked as such, never averaged into
  the cropped figure. It answers the one question worth asking of them — whether the verifier is
  passing drafts that do not look like the concept — without pretending it is the same measurement.

Mixing the two into one mean would compare a tight object crop against a whole scene and read the
difference as identity.

## 5. Stratification — targets and controls

`configs/dataset.yaml` is 11 fine-grained targets (`african_grey_parrot` … `cardinal_bird`) and 11
controls (`hedgehog` … `violin`) that FLUX should already render correctly. Its own header explains
why: "If the controls fail too, the drafts are broken rather than the concepts rare — that is what
makes the fail rate interpretable."

That distinction currently exists only in a YAML comment. This doc promotes it to data: `Case` gains
`kind: "target" | "control"`, defaulting to `"target"`, and `dataset.yaml` labels its 11 controls.
Every table in the report is broken down by `kind`.

The expected shape, if the method works: targets improve on DINO from `no_rag` to `oracle`; controls
start high and move little. Controls improving as much as targets means the metric is measuring
"an edit happened," not identity.

## 6. Architecture

Two units, split on the same principle that made doc 3's scheduler testable: the part that needs
models is separated from the part that needs judgement.

**`ragregen/metrics.py`** — pure functions plus eval encoders. Ported from
`ImageRAG/scripts/kontext_metrics.py`, which already has `crop_to_mask`, three-zone `preservation`,
and an `_HFEncoder` base, into this project's `encoders.py` / `config.py` idioms.

- `crop_to_mask(image, mask, pad_frac=0.15) -> Image`
- `preservation(draft, output, alpha) -> float` — the three zones from `feather_alpha`: outside
  (`alpha == 0`) must be bit-identical, band, inside
- `dino_identity(output_crop, ref_crops, encoder) -> float`
- `prompt_alignment(image, prompt, encoder) -> float` — serves both CLIP and SigLIP
- `EvalEncoders` — a small holder that loads the three eval encoders and frees them, mirroring
  `stage_with_model`'s discipline

Preservation and `crop_to_mask` take arrays and are tested on synthetic images with no model loaded.

**`scripts/report.py`** — walks a run directory and a screen run, computes the table, writes
`report.json` (machine) and `report.md` (human). Model residency is one load of the eval encoders for
the whole run, not one per case. Missing cases are skipped with a recorded reason rather than
crashing the report; a run that died halfway must still be readable.

`configs/pipeline.yaml` gains:

```yaml
eval:
  heldout_refs: 1
  dino: facebook/dinov3-vitl16-pretrain-lvd1689m
  clip: laion/CLIP-ViT-L-14-laion2B-s32B-b82K
  siglip: google/siglip-base-patch16-384
```

`run.sh` gains a `report` stage. It is CPU-cheap apart from the encoder loads.

## 7. Data flow

```
screen_latest/<case>/draft.png ──┐
                                 ├─→ no_rag arm  (the draft, unrepaired)
pipeline_<TS>/<case>/            │
  attempt_N.png, mask.png,       ├─→ oracle arm  (best-of selection)
  scores.json  ──────────────────┘
dataset.yaml gt_refs[-h:] ───────→ DINO target (held out, never edited with)
```

Per case: read `scores.json` for the draft verdict and `best`; load the draft and the selected
output; crop both to the mask; score. `best` is the pipeline's selection and this doc does not
second-guess it — selection uses the verifier only, and re-selecting on DINO would be selecting on
the thing being measured.

## 8. Error handling

- **A missing case directory** is recorded as `skipped: no output` and excluded from every mean. A
  partial run reports on what it has.
- **A missing `mask.png`** is expected for healthy cases, not an error — it routes the case to the
  whole-image column per §4. It *is* an error for a case whose draft verdict failed, because that
  case should have run `mask`.
- **A case with fewer than `h+1` references** fails config validation, not the run. `validate` gains
  this check, because discovering it after an hour of encoding is the failure mode this project
  keeps designing against.
- **Size mismatch** between draft and output raises. It means the compositor changed behaviour, and
  silently resizing would hide that behind a plausible number.
- **An eval encoder equal to `retriever` or `crop_scorer`** fails config load, per §3.

## 9. Testing

Everything except the encoder loads is CPU-testable, and the GPU on this machine is routinely held by
someone else (`findings/2026-07-28-regeneration-result.md` §4).

- `preservation` on synthetic three-zone images: an untouched output scores 1.0; a repainted canvas
  scores near 0; changes confined inside the mask leave the outside zone exactly 1.0.
- `crop_to_mask` on a known box, including a mask touching the image edge, where padding must clamp.
- Hold-out slicing: `h=1` on a 3-ref case yields 2 edit refs and 1 scoring ref; the scoring ref is
  never in the edit list. Pinned for 3-, 4-, and 5-ref cases.
- Round clamping: a 2-ref edit pool with `retry_budget: 3` runs 2 rounds and marks nothing `failed`.
  This is the test that stops the hygiene fix from fabricating failures.
- Denominators: a fixture with one healthy case, one repaired, one unrepairable produces repair rate
  1/2, not 1/3 or 2/3 — the two wrong answers `status` would give.
- A healthy case with no `mask.png` reports a whole-image DINO and does not appear in the cropped
  mean; a failed-draft case with no `mask.png` raises.
- Encoder-collision config raises.
- The report renders from a fixture run directory with no model loaded.

The 358 existing tests stay green.

## 10. What this will not prove

- Nothing about the `full` arm, so nothing about retrieval quality end to end.
- `inpaint` has still never run under the pipeline; if the measured run uses `stitch`, the report
  says so in its header, because a stitched rectangle and a diffused edit are not the same method.
- No human study. A held-out VLM judge is the design's intended substitute and is deferred; the
  report states this rather than implying parity.
- Verifier pass-rate remains a development signal and appears nowhere near a headline.
