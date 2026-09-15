# LAION common-concept robustness benchmark — doc 5

**Date:** 2026-08-05
**Prior docs:** `2026-07-28-masking-design.md` (1), `2026-07-28-regeneration-design.md` (2),
`2026-07-28-orchestration-design.md` (3), `2026-07-30-evaluation-design.md` (4)
**Prior findings:** `docs/findings/2026-07-29-orchestration-result.md`
**Produces:** `scripts/fetch_corpus.py`, `scripts/make_cases.py`, `scripts/supervise.sh`,
edits to `ragregen/config.py`, `ragregen/schedule.py`, `ragregen/validate.py`, `scripts/report.py`

---

## 1. What this doc is for

Doc 4 measured the `oracle` arm and named its own boundary: the `full` arm "needs a retrieval corpus,
which `RUNBOOK.md` line 7 makes the operator's input, not this project's deliverable." That corpus
never arrived, so `run_pipeline.py:139`'s `--arm full` branch — `_load_retriever`, `_retrieve_one`,
and the index wiring beneath them — has never once executed against real data.

This doc supplies the corpus and runs that arm, on a question the project has not asked before.

Docs 1–4 are about **rare** concepts: can the pipeline repair what FLUX gets wrong? The complementary
risk is the opposite one, and it is the failure that sank the work this project forked from. Every
`eval_methods` run in `../ImageRAG/results/` has RAG scoring **below** its own no-RAG baseline (spec
§1). A repair system that helps on the long tail but damages everything else is not deployable.

So: **does the full retrieve-and-repair pipeline leave common concepts alone?**

## 2. The claim

> **C3.** On common concepts, running the full retrieve-and-repair pipeline does not degrade the
> image relative to leaving it alone.

Measured as a **paired within-case comparison**: `DINO(best) − DINO(draft)`, one delta per case.
Paired because each case is its own control — same prompt, same held-out references, same mask, same
crop — so between-case variance in concept difficulty cancels rather than swamping the effect.

Three readings, fixed in advance so the result cannot be reinterpreted after it is seen:

| outcome | reading |
|---|---|
| CI lower bound ≥ **−0.02** | no-harm holds |
| CI upper bound < 0 | the pipeline damages common concepts — ImageRAG's documented failure reproducing here, and a publishable negative |
| CI spans −0.02 | inconclusive at this n; reported as such, with the per-case scatter |

**The non-inferiority margin is −0.02 DINO cosine, fixed here before any data exists.** Doc 4's
measured cropped DINO was 0.626, so 0.02 is ~3% of the operating point — below it, a drop is not
distinguishable from crop and reference noise; above it, a reviewer would rightly call it a
regression. Choosing this margin after seeing the deltas would make the test meaningless.

**Improved / unchanged / worsened** are defined on `best`, not on a delta threshold: *unchanged* is
`best == "draft"` (delta exactly 0, the pipeline declined to change anything), and the remaining
cases split by the sign of the delta. This keeps the counts independent of the margin above.

**Preservation is read beside every delta**, per spec §5: identity bought by repainting the canvas is
not a repair.

Note the guarantee this does *not* inherit. `schedule.py:175 select_best` seeds `best` with the
draft, so the pipeline cannot lose **on the verifier's score** (RUNBOOK §1). DINO is not the
selection criterion — doc 4 §7 is explicit that re-selecting on DINO would be selecting on the thing
being measured. So DINO can fall even though the verifier score cannot. That gap is exactly where
C3 lives, and it is why C3 is a real question rather than a tautology.

## 3. The two datasets and their roles

They never touch. The separation is the experiment.

| | source | role | seen by the pipeline? |
|---|---|---|---|
| **LAION** | `laion2B-en-aesthetic` metadata → `img2dataset` → 384px | the retrieval corpus the system searches | **yes** — this is the input |
| **ImageNet** | `evanarlian/imagenet_1k_resized_256`, val split | class names → prompts; 3 val images per class → `gt_refs` | **never** — answer key only |

```
ImageNet class name ──> prompt ──> FLUX ──> draft
                                             │
                                   verifier: "not really a fox"
                                             │
                                             ▼
LAION 100k ───────────> search "fox" ──> reference photo
                                             │
                                             ▼
                                   repair the draft ──> output
                                             │
ImageNet val images (invisible to everything above)
       └───────────────> DINO score ────────┘
```

ImageNet asks the question, LAION is what the system may consult, ImageNet marks the answer.

`validate.py:88` already enforces the separation, raising `gt_ref_in_corpus` with the message *"The
'full' arm would retrieve its own answer key."* The `full` arm is unusually clean here: it edits with
LAION images and is graded on ImageNet images, disjoint by construction — different downloads,
different directories. It still uses doc 4's held-out split for its scoring refs, so the numbers stay
directly comparable when `oracle` is eventually run.

**`evanarlian/imagenet_1k_resized_256` over `ILSVRC/imagenet-1k`** because it is ungated, ships actual
image bytes in parquet (no URL attrition), and covers all 1000 classes. The official repo is
`gated: auto` and needs a token that has accepted the terms — a manual step outside the code. The
mirror's 256px is a real but bounded cost: DINOv3 resizes to 224 internally, and more importantly the
headline is a *paired within-case* comparison where both sides use the same refs, so resolution
shifts both equally and cancels. It would only bite across cohorts, which §5 forbids.

## 4. Corpus size: 100k, and why that is the right shape

**100,000 images**, composed as a random slice plus a caption-keyword-matched slice per concept.
The two components do different jobs and are sized for different reasons.

**The seeded slice guarantees the answer exists.** Retrieval at `k = retry_budget = 3` needs at least
three usable photographs per concept. In purely random web images a given ImageNet class appears at
roughly 1-in-10⁴ to 1-in-10⁵, so a random 100k yields single-digit images for a common concept and
near-zero for a rare one. Random alone is not enough; the seeded slice is what makes retrieval
possible at all.

**The random bulk is distractor mass, and that is its whole purpose.** A corpus containing only the
92 seeded concepts would make retrieval trivially correct and the benchmark would measure nothing.
The bulk exists to make the search a real test.

**Per-concept counts are published.** `corpus_manifest.json` records how many caption-matched images
landed for each concept, and the report carries the number per case. A bad result is then
attributable to retrieval failure versus corpus sparsity, instead of guessed at.

### Why 100k also minimises contamination

LAION and ImageNet both scrape the web, so a near-duplicate of a held-out reference could exist
inside the corpus. Path-based `gt_ref_in_corpus` compares directories, not pixels, and cannot catch
it. The estimate, per reference image:

```
P(contaminated) = P(a near-duplicate exists in LAION) × P(we draw that duplicate)
                ≈        f  (uncertain, ~0.1–0.3)     ×   ~1e-4 (seeded slice)
                ≈ 1e-5
```

Across ~280 reference images that is order 1-in-a-few-hundred that even one is contaminated, and one
bad reference in 280 moves no mean. The dominant uncertainty is `f`, the ImageNet↔LAION duplicate
rate — studied but not precisely known — so this is stated as an order of magnitude, not a decimal.

**Corpus size trades against itself.** A larger corpus improves retrieval density but raises
contamination proportionally: at 1M the odds rise tenfold. Spec §9 R6 sized `IndexFlatIP` for 500k
and 250–500k would be a defensible richer operating point, so **corpus size is a config knob**, not a
hard-coded constant. 100k is the chosen setting: cheap to build, search is already non-trivial, and
contamination is lowest.

**Disk.** 384px re-encode ≈ 3–5 GB at 100k; index 100k × 1152 × 4 B ≈ 460 MB; ImageNet val ≈ 1.5 GB.
Against 467 GB free at 97% on a shared 14 TB volume — spec §9 R2's exact concern, comfortably inside
it.

## 5. The case set

**92 cases = 70 new common concepts + 22 bridge cases.**

`configs/dataset.yaml`'s existing 22 (11 `target`, 11 `control`) are carried forward unchanged, with
their existing full-resolution `gt_refs` and their existing drafts in `outputs/screen_latest/`. They
cost only repair time and preserve comparability with docs 1–4.

**"Common" is measured, not asserted.** ImageNet-1k class names are ranked by caption frequency in
the downloaded LAION slice; the top 70 not already among the 22 are taken. This makes commonness a
property of the corpus rather than an opinion, it is reproducible from a seed, and it is honest about
what the benchmark tests: concepts the retrieval database actually knows.

**Prompts** follow the house style of `dataset.yaml` (`"a fox trotting through tall grass at dusk"`)
via a small set of hand-written scene frames applied deterministically to class names. No model in
the loop, and comparable in shape to `controlled_dataset_v2`.

**Cohorts never merge.** `Case` gains `cohort: "bridge" | "common"`. The report groups by
`(kind, cohort)` and never averages a 256px-ref case with a full-resolution-ref case into one DINO
figure — different reference resolutions, different concept rarity.

**No human labelling of the 70 new drafts.** The paired delta needs no ground-truth verdict, and
doc 4 §4 already buckets on the verifier's draft verdict rather than on human labels.
`screen_premise.py` still emits its `labels.csv`; for the new cohort it is left unlabelled. Stated as
a limitation in §10 rather than hidden.

## 6. What gets reported

"Robust on common concepts" has two failure modes and therefore two numbers.

**A — the false-alarm rate.** Of the 70 common cases, what fraction did the verifier route into
repair at all? A common concept sent to repair burns ~6 min of GPU and exposes a good image to
damage, even when `best` ends up being the draft. Needs no FLUX; falls out of the dry-run.

**B — the paired delta**, `DINO(best) − DINO(draft)`, over the cases that entered repair. Reported as
mean, paired-bootstrap 95% CI, and the improved / unchanged / worsened counts.

Bootstrap rather than a t-test: n will be well under 100 and the deltas are likely skewed, so no
normality assumption is made. The improved/worsened counts are the honest backstop when the CI
straddles zero.

**Healthy cases are excluded from the delta mean** and reported separately as delta-zero-by-
construction. Averaging them in would dilute a real effect toward zero and flatter the method — the
same reasoning doc 4 §4 applies to repair-rate denominators.

Preservation sits beside the delta in every row. Per-concept corpus density sits in the per-case CSV.
Verifier pass-rate stays absent from the headline, per doc 4 §3.

## 7. Architecture

Five units, ordered by dependency. Three new, two edits.

**`scripts/fetch_corpus.py`** — LAION acquisition, in three separable subcommands so a failed
download resumes without re-selecting:

- `select` — LAION parquet metadata → a URL parquet, seeded. Over-requests, because dead links are
  the norm.
- `download` — shells out to `img2dataset`, re-encoding to 384px.
- `manifest` — walks what actually landed, counts caption matches per concept, writes
  `corpus_manifest.json`.

No GPU. The split matters: selection is deterministic and cheap, download is slow and flaky, and
conflating them means a network failure discards a reproducible sample.

**`scripts/make_cases.py`** — reads the ImageNet val parquets and `corpus_manifest.json`; ranks the
1000 class names by corpus caption frequency; takes the top 70 not among the 22; extracts 3 val
images per class to `data/gt_refs/<class>/`; applies scene-frame templates; writes
`configs/dataset_common.yaml` with the 22 merged in. Deterministic from a seed. No GPU, no model.

**`ragregen/config.py`** — `Case` gains `cohort`, validated against a constant exactly as `kind`
already is at line 125.

**`scripts/report.py`** — the substantive edit. `score_case` gains `dino_cropped_draft`, computed
with **the same mask and the same held-out refs** as the output; that identity is what makes the
comparison paired and it is pinned by a test. `summarise` groups by `(kind, cohort)`. New columns:
`dino_delta`, its paired-bootstrap 95% CI, improved/unchanged/worsened counts.

**Unchanged: `build_index.py`, `screen_premise.py`, `run_pipeline.py`.** `build_index.py` already
walks `images_root` and writes `IndexFlatIP`; `screen_premise.py` already takes `--dataset`;
`run_pipeline.py:139` already branches to the retrieval path on `--arm full`. That branch has never
executed, which is what §8's dry-run is for.

## 8. The run plan

Staged so that the expensive step is the last thing attempted, not the first.

| stage | what | cost | GPU |
|---|---|---|---|
| 0 | `fetch_corpus` select + download + manifest | ~2–4 h wall, network-bound | none |
| 1 | `make_cases` | minutes | none |
| 2 | `build-index` over 100k | ~20 min | yes, encoder only |
| 3 | `screen` — drafts for the 70 new cases | ~2.3 h | yes, FLUX |
| 4 | **dry-run**: `--arm full --mechanism stitch`, 92 cases | ~2 h | yes, Qwen + DINO/SAM; **no FLUX** |
| 5 | **the run**: `--arm full --mechanism inpaint`, 92 cases | ~15 h | yes, everything |
| 6 | `report` | minutes + encoder loads | yes, encoders only |

Stage 4 is the whole point of the staging. `stitch` pastes the reference cutout in pixel space
(`regen.py`), so it needs no FLUX weights while still exercising retrieval, index wiring, masking,
scoring and the report end to end — roughly 200 lines of never-executed code, against real data, for
2 hours instead of 15. It also yields metric **A** (§6) on its own.

Stage 5's ~15 h: 92 cases × up to 3 attempts × ~2 min, plus stage-batched model loads that do not
scale with case count (spec §4).

**`oracle` is deferred to a documented stage 7** (~12 h). It prices retrieval error — the `B→C` gap
of spec §6 — and is worth running only if stage 5's result is ambiguous and the cause needs
attributing between search and repair. The arm axis already exists in `run_pipeline.py`, so this
needs no redesign.

## 9. Surviving a 15-hour run on a shared card

The card is shared and stage-batching **must** free VRAM between stages: spec §4 established that
Qwen-7B (~16 GB) and FLUX-nf4 (~12 GB) do not co-fit on 24 GB, and `env.reclaim_gpu()` exists to hand
memory back. Holding the card is therefore not an option — any ballast that reserved memory would
starve the pipeline's own next stage. The answer is to yield cleanly and re-acquire, not to squat.

Most of that is already built:

- `schedule.py:134 is_fatal()` separates "the card died" from "this case is bad", so an OOM leaves the
  case **unmarked** rather than failed — "the next 29 would fail the same way."
- `schedule.py:162 StageAborted` saves `queue.json`, exits 2, and prints the resume command. At most
  the one case in flight is lost.
- `scripts/gpu_wait.sh` polls `nvidia-smi` and launches only after ≥17 GB has been free for 3
  consecutive polls — its header already names the hazard: *"a neighbour's job dips between
  iterations."*

What is missing is that `gpu_wait.sh` guards only the **initial launch**. Nothing re-acquires the
card after an abort, so a 3 a.m. eviction sits idle until a human notices. Over 15 hours that is the
dominant failure mode — dead time, not data loss. Three additions close it:

**`scripts/supervise.sh`** — loops `gpu_wait.sh -- run.sh pipeline --resume <dir>`. Exit 2 waits and
relaunches; exit 0 stops; anything else stops and says why. Bounded retries and a total deadline, so
a permanently-held card does not spin forever.

**VRAM preflight** in `schedule.py` — before `stage_with_model` loads anything, compare free VRAM
against the stage's need; if short, raise `StageAborted` deliberately. Today the failure is a real
`OutOfMemoryError` partway through loading weights: `is_fatal()` catches it, but yielding on purpose
beats being evicted mid-allocation.

**Heartbeat** — one timestamped line per completed case into `run.log`, so `tail -f` distinguishes
"waiting for the card" from "hung."

## 10. Error handling

- **A concept with fewer than `k` caption-matched images** is a `validate.py` **warning**, not an
  error. The run proceeds and the report records the count: "the corpus had nothing here" is a
  finding, not a crash.
- **`img2dataset` attrition** is expected. `select` over-requests; `manifest` reports what actually
  landed. A thinner-than-intended corpus is visible rather than silent.
- **A missing case directory** stays doc 4 §8's behaviour: recorded as skipped, excluded from every
  mean.
- **A missing `mask.png`** stays doc 4 §8's behaviour: expected for healthy cases, an error for a
  case whose draft verdict failed.
- **Draft and output of different sizes** raises, per doc 4 §8.
- **VRAM preflight failure** raises `StageAborted`, never a bare exception, so `run_stage`'s existing
  policy leaves the case unmarked for resume.

## 11. Testing

Everything except the encoder loads is CPU-testable — doc 4 §9's discipline, on a card
`findings/2026-07-28-regeneration-result.md` §4 says is routinely held by someone else.

- **Paired identity:** draft and output are scored with the same mask and the same held-out refs.
  Pinned, because a silent divergence here makes the headline meaningless while still producing a
  plausible number.
- **Cohorts never merge:** a fixture with `bridge` and `common` cases produces separate rows, never
  one mean.
- **Bootstrap CI** on a fixture with known deltas returns a known interval, seeded.
- **Healthy exclusion:** a fixture of one healthy, one improved and one worsened case yields a delta
  mean over 2 cases, not 3.
- **Supervisor:** exit 2 relaunches, exit 0 stops, exit 1 stops. Driven with a fake command, no GPU.
- **Preflight** raises `StageAborted` rather than a bare exception.
- **`make_cases.py` determinism:** same seed, same 70 classes, same prompts.
- **`fetch_corpus select` determinism:** same seed, same URL list.
- **Corpus-density warning** fires below `k` and does not raise.
- The existing suite (399 test functions across 34 files) stays green.

## 12. Risks

**R1 — ImageNet↔LAION near-duplicates.** Path-based `gt_ref_in_corpus` compares directories, not
pixels. Estimated order 1-in-a-few-hundred across all refs (§4). Mitigation: an optional
embedding-similarity spot-check of held-out refs against the index, **reported, not enforced** —
a threshold that silently dropped cases would be worse than a number with a caveat.

**R2 — `img2dataset` attrition** yields a thinner corpus than requested. Mitigated by over-requesting
in `select` and reporting the truth in `manifest`.

**R3 — 15 h on a contended card.** §9 is the mitigation. Residual: the multi-resume path has never run
this long, only across a single kill (`findings/2026-07-29-orchestration-result.md`).

**R4 — the verifier may pass nearly every common concept**, leaving too few repaired cases for a
usable CI. Detected in the ~2 h dry-run, before the 15 h run is committed. If it happens, **that is
the robustness result** — the pipeline correctly leaves common concepts alone — and it is written
here in advance rather than revised into later.

**R5 — `inpaint` has never run under the pipeline** (doc 4 §10). The dry-run exercises everything
except that one call, so stage 5 is the first exposure. A stitched rectangle and a diffused edit are
not the same method, and the report header states which produced it.

**R6 — 256px references** for the common cohort. Bounded by the paired design (§3) and by §5's rule
that cohorts never merge into one mean.

## 13. What this will not prove

- **Nothing about rare concepts.** The 22 bridge cases are context, not the population under test.
- **Nothing about retrieval error.** That is the `oracle` arm, deferred to stage 7.
- **No human study**, and no human labels on the 70 new drafts. Doc 4's held-out VLM judge remains
  deferred; the report states the deviation rather than implying parity with the paper's
  46-participant study.
- **Verifier pass-rate** remains a development signal and appears nowhere near a headline.
- **Nothing about corpora other than 100k LAION.** Size is a config knob and 250–500k is a defensible
  richer setting, but only the configured 100k is measured here.
