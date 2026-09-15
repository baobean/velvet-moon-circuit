# rag-regen — Operator Runbook

> **Status: every stage in `scripts/run.sh` is implemented and runnable** — run
> `./scripts/run.sh --help` for the current list.
> Nothing below requires you to edit Python.

**Your job:** supply a test dataset and a retrieval database, run the stages, read the report.
**Not your job:** modifying code. If something can only be fixed by editing a `.py` file, that is a
bug in this runbook or in the pipeline — report it, do not patch around it.

---

## 1. What the method does

A text-to-image model is asked for a prompt containing a **rare or fine-grained concept** — an Amur
leopard, a monkey puzzle tree, a Boston bull. It usually produces something plausible but wrong: a
generic leopard, a generic conifer. The pipeline detects that, finds a real photograph of the actual
concept, and regenerates just the wrong region using that photograph as a reference.

```
prompt
  │
  ├─ 0. DRAFT ........... FLUX.1-Kontext generates a first image from the prompt alone
  │
  ├─ 1. VERIFY .......... two independent judges decide if the draft matches the prompt
  │      Stream A (grounded)  GroundingDINO finds each concept, a region-text model
  │                           scores how well the crop matches the phrase → a number
  │      Stream B (semantic)  a VLM (Qwen3-VL) reads the image and reports what is wrong
  │      Fusion               fails if either stream fails; names the concept to fix
  │
  │      passes → done, the draft is the output
  │      fails  → continue
  │
  ├─ 2. RETRIEVE ........ SigLIP searches YOUR retrieval database for the
  │                       concept name itself, returning the top N matches.
  │                       (A VLM-written search caption is designed but not
  │                       built -- orchestration design §9. The concept name
  │                       is what is searched today.)
  │
  ├─ 3. MASK ............ two masks are cut:
  │                       · on the draft — the region to replace
  │                       · on the reference — the object to copy from
  │                       both via GroundingDINO → SAM, with a classifier picking the right
  │                       box when several similar objects are present
  │
  └─ 4. REGENERATE ...... FLUX.1-Kontext repaints the masked region using the reference,
                          then re-verifies. If it still fails, it retries with the next
                          retrieved reference, up to N attempts.
```

**Two guarantees worth knowing when you read results:**

- Each retry regenerates from the **original draft**, never from the previous attempt. Chained
  edits would degrade the whole image.
- The original draft is always kept as a candidate. The pipeline returns the **best** of
  {draft, attempt 1 … attempt N}, so the method cannot score worse than the no-retrieval baseline
  on a given case. The report shows both `best` and `last-attempt` columns so you can see how much
  that rule contributed.

---

## 2. Server setup

Everything runs on this machine. You share it with other people — read §2.3.

### 2.1 Environment

**Use the `kontext` conda environment. Do not use `ImageRAG_qwen`.**

```bash
CONDA=/mnt/mmlab2024nas/ldtuan/miniconda3/envs
PY=$CONDA/kontext/bin/python
$PY -V          # expect: Python 3.11.15
```

| | `kontext` ✅ | `ImageRAG_qwen` ❌ |
|---|---|---|
| Python | 3.11.15 | 3.10.13 |
| diffusers | 0.39.0 — has FLUX Kontext | 0.31.0 — **no Kontext** |
| transformers | 5.14.1 — has Qwen3-VL | 4.44.2 — **no Qwen3-VL** |
| `~/.local` shadowing | immune | affected — needs `PYTHONNOUSERSITE=1` |

`~/.local/lib/python3.10/` contains an old `transformers` that silently shadows Python 3.10
environments. `kontext` is 3.11, so it is immune, and you do **not** need `PYTHONNOUSERSITE=1` for
this project. If you see `transformers 4.44.2` reported anywhere, you are in the wrong environment.

### 2.2 One-time installs and downloads

```bash
export HF_HOME=/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.cache

$PY -m pip install pytest faiss-cpu open_clip_torch pandas   # already installed
```

Already in `$HF_HOME` — do not re-download:
FLUX.1-Kontext-dev · SAM-ViT-H · GroundingDINO (base + tiny) · SigLIP-SO400M-384 · DINOv3-L ·
Qwen2.5-VL-7B

Must be downloaded (see §6 disk warning first): **Qwen3-VL** (~16 GB).

**FG-CLIP does not run in this environment.** Its remote code targets transformers ~4.12 and
fails to build under the 5.14.1 pinned here, so it can never be the live `crop_scorer` or
`retriever` even if it wins the bake-off. It remains a bake-off-only candidate, run from a
separate conda env via `./scripts/run.sh bakeoff --fgclip-python <that-env>/bin/python`.
Without that flag its arm is reported as SKIPPED, not as a loss.

Verify the environment before anything else:

```bash
./scripts/run.sh validate
```

### 2.3 GPU etiquette — important

**There is one GPU: a single RTX 4090, 24 GB, shared with other researchers.**

```bash
nvidia-smi        # always check before starting a long run
```

The models do not co-fit: Qwen-7B is ~16 GB and FLUX-nf4 is ~12 GB. The pipeline therefore runs
**stage-batched** — it loads the VLM, processes every case, unloads, loads FLUX, processes every
case, unloads. This is deliberate. Loading FLUX takes **5 min 21 s**; a naive per-case loop would
spend 73% of its wall-clock loading weights (~8 h of pure loading for 30 cases).

Consequences for you:

- **Runs are resumable.** State lives in `outputs/<tag>_<TS>/queue.json`. If a stage dies because
  someone else grabbed the card, re-run the same command — it picks up where it stopped. Do not
  start over.
- A previous project (`../rag-edit`) died with `torch.OutOfMemoryError` because five other
  processes held the card. That is expected on a shared machine; it is why resume exists.
- Long runs are best started when `nvidia-smi` is quiet.

---

## 3. Plugging in your data

You edit two YAML files. Nothing else.

### 3.1 Test dataset — `configs/dataset.yaml`

The prompts to generate and verify, with the ground truth needed for scoring.

```yaml
name: my_test_set
images_root: /path/to/your/gt_reference_images

cases:
  - id: amur_leopard_01
    prompt: "an Amur leopard walking across a snowy ridge"
    concept: "Amur leopard"          # the rare concept to check and repair
    coarse: "leopard"                # the concept's superordinate category
    gt_refs:                          # ground-truth photos of the real concept
      - amur_leopard/ref_01.jpg       # REQUIRED — used by the DINO metric and
      - amur_leopard/ref_02.jpg       # by the `oracle` arm
```

**`gt_refs` is required, not optional.** Two things depend on it: the **DINO identity metric**
(which measures whether the output actually looks like the real concept — the central claim) and the
**`oracle` arm** (which injects the correct reference directly, bypassing retrieval, to separate
"retrieval failed" from "the generator could not use a good reference"). Without `gt_refs` both are
silently unmeasurable.

Keep `gt_refs` **out of** the retrieval database. If the ground truth is retrievable, the `full` arm
is scoring against its own answer key.

**`coarse`** (required). The concept's superordinate category — `parrot` for
`African grey parrot`, `dog` for `Boston bull`. Stream A scores the detected
crop against both terms and fails the draft when the fine term's lead over the
coarse term falls below the configured margin δ (so a negative δ requires the
coarse term to actually win). This is what lets the verifier catch "a leopard
that is merely a leopard" rather than only gross category errors.

Two rules:

- Author it from the concept name, before looking at any draft. Choosing a
  coarse term because a draft happens to resemble it fits the test to its own
  answer key.
- It must differ from `concept`. Identical terms make the margin zero for
  every case and silently disable the check; `validate` rejects this.

For basic-level concepts with no tighter category above them (`durian`,
`sushi`, `violin`), pick the nearest honest superordinate (`fruit`, `food`,
`string instrument`). The contrastive check carries little signal for these,
and that is expected — the absolute threshold remains their mechanism.

**`coarse_refs`** (optional, top level). Reference images per coarse term, used
by the prototype verifier. Keyed by the term itself, so terms several cases
share — `dog`, `tree`, `bird`, `bear` — are authored once:

```yaml
coarse_refs:
  parrot: [parrot_generic_0.jpg, parrot_generic_1.jpg, parrot_generic_2.jpg]
  dog:    [dog_generic_0.jpg, dog_generic_1.jpg, dog_generic_2.jpg]
```

Three or more per term, and **spread across the category** — three photographs
of the same parrot make a second fine-grained prototype, not a superordinate
one, and the contrast stops measuring what it claims. `validate` checks the
count; nothing can check the spread.

### 3.2 Retrieval database — `configs/retrieval_db.yaml`

```yaml
name: my_retrieval_db
images_root: /path/to/retrieval/corpus     # a folder of images, any nesting
index_path: data/rag_db/index.faiss        # written by build-index
encoder: siglip_so400m_384                 # must match what build-index used
captions: null                             # optional CSV: path,caption
```

Then build the index (one-time per corpus, GPU, no FLUX):

```bash
./scripts/run.sh build-index
```

Scale guidance: at 500k images the index is 500k × 1152 × 4 B ≈ **2.3 GB** and stays an exact
brute-force `IndexFlatIP` — no approximation, so retrieval quality is never confounded by the index.
Embedding 500k images takes roughly an hour on the 4090.

### 3.3 Preflight

```bash
./scripts/run.sh validate
```

Checks, **before any GPU work**: every image opens; every case has a `concept` and non-empty
`gt_refs`; `gt_refs` do not appear in the retrieval corpus; the FAISS index dimensionality matches
the configured encoder; disk headroom is sufficient. Fix everything it reports before running
anything long — the alternative is discovering it nine hours in.

### 3.4 The common-concept benchmark (doc 5)

Answers: does the pipeline damage images of concepts FLUX already gets right?

Two datasets, two jobs, and they never touch. **LAION is the library the
system searches.** **ImageNet asks the questions and marks the answers** — the
pipeline never sees an ImageNet image. `validate` refuses to run if they
overlap.

```bash
./scripts/run.sh fetch-corpus select        # ~5 min, seeded and reproducible
./scripts/run.sh fetch-corpus download      # ~2-4h, network-bound, resumable
./scripts/run.sh fetch-corpus manifest      # ~2 min, reports what landed
./scripts/run.sh make-cases                 # ~10 min, writes dataset_common.yaml
./scripts/run.sh validate --dataset configs/dataset_common.yaml
./scripts/run.sh build-index                # ~20 min GPU
./scripts/run.sh screen --dataset configs/dataset_common.yaml   # ~2.3h GPU
```

Then the dry-run, which costs 2 hours and de-risks 15:

```bash
./scripts/run.sh supervise -- ./scripts/run.sh pipeline \
    --dataset configs/dataset_common.yaml --arm full --mechanism stitch \
    --tag doc5dry
```

`stitch` pastes the reference in pixel space and needs no FLUX weights, so
this exercises retrieval, masking, scoring and the report end to end — code
that has never executed — without booking the card for a full day. Read its
report before continuing. The `stitch` dry-run is also where to **visually
inspect the retrieved references for the highest-ranked `common` cases**:
caption matching has no sense disambiguation, so a class name whose dominant
English sense is not the ImageNet one retrieves the wrong object entirely, and
that becomes visible here for 2 hours of card time instead of 15.

Then the real run:

```bash
mkdir -p outputs/doc5_$(date +%Y%m%d_%H%M%S)   # pin it, then reuse the path
./scripts/run.sh supervise -- ./scripts/run.sh pipeline \
    --dataset configs/dataset_common.yaml --arm full --mechanism inpaint \
    --resume outputs/doc5_<TS>
./scripts/run.sh report --run outputs/doc5_<TS> \
    --dataset configs/dataset_common.yaml
```

`supervise` re-acquires the card and resumes after an eviction. Exit 2 means
the card was taken and the queue is checkpointed; any other code is a real
failure and stops the loop.

**Pass `--resume <dir>`, not `--tag`, to anything wrapped in `supervise`.**
This is not a style preference. `supervise` retries by re-running the
*identical* command, and `pipeline` without `--resume` calls `trace.open_run`
every time — so each eviction mints a fresh `outputs/<tag>_<TS>/` and restarts
the queue from zero. On a contended card that discards hours of work per
eviction and a long run may never finish. Create the directory once, pass it
as `--resume`, and every retry continues the same queue. `scripts/campaign.sh`
does this for you.

**Or just run the whole campaign:** `./scripts/campaign.sh` executes every
stage above in order, records a marker per completed stage under
`logs/campaign/`, and skips what is already done when re-run — so a crash, a
reboot, or an interrupted session resumes rather than restarts.
`./scripts/campaign.sh status` prints where it got to.

**`supervise` needs `scripts/gpu_wait.sh`, which is not tracked in this
repo.** It fails its preflight if the file is missing or not executable. The
contract is small enough to write a five-line stand-in: it is invoked as
`gpu_wait.sh --need GB --log FILE -- <cmd> [args...]`, it should block until
at least `GB` gigabytes of VRAM are free, then exec the command and **pass the
child's exit code through unchanged** — `supervise`'s whole retry rule is
`rc == 2`, so a wrapper that swallows or rewrites the code breaks it. Pass
`--no-gpu-wait` to run the command directly without any GPU acquisition.

**If `supervise` reports a non-2 exit, check for a segfault before debugging
anything.** A crash in a native extension is not an eviction, so the loop
correctly stops rather than retrying — but the cure is just to re-run the
identical command: the queue is checkpointed per case, so at most one case's
work is lost. Do not add 139 to the retry set; blanket-retrying a crash is
exactly what the exit-2-only rule exists to prevent.

**Reading the result.** The delta column is the claim: `DINO(best) −
DINO(draft)`, paired within each case. `no_harm` means the CI's lower bound
clears −0.02, a margin fixed before any data existed. It is a non-inferiority
verdict, not a claim of zero change — an interval that sits entirely below
zero but stays above the margin still reads `no_harm`; that means the case
set measurably got worse, just by less than the tolerance allows, not that it
was untouched. A `harm` verdict is a publishable negative, not a bug to fix.

---

## 4. Running

```bash
./scripts/run.sh <stage> [--tag NAME]
```

| stage | GPU | time | what it does |
|---|---|---|---|
| `validate` | no | seconds | preflight on your configs — **always run first** |
| `build-index` | yes, light | ~1 h / 500k | embeds your corpus into a FAISS index |
| `screen` | yes | ~1 h | **gate.** Drafts every case, you hand-label pass/fail |
| `bakeoff` | yes, light | ~1 h | FG-CLIP vs SigLIP in two roles; no FLUX |
| `score-a` | yes | ~5 min | Stream A over the labelled drafts |
| `score-a --proto-arm ceiling` | yes | ~5 min | Stream A with prototypes from `gt_refs`. **A diagnostic ceiling — never quote it as a verifier result**, because `gt_refs` are also the DINO metric's target. |
| `score-a --proto-arm retrieved` | yes | ~5 min | Stream A with prototypes from your corpus. This is the reportable configuration; needs `build-index` first. |
| `score-b` | yes | ~10 min | Stream B over the labelled drafts |
| `repair` | yes | ~1 min (stitch) / ~8 min (inpaint) | Repairs one case on the oracle arm and writes before/after PNGs. The smoke test for the regeneration half; not part of the main table. Add `--prototypes` where the coarse term grounds more than one candidate — without it `boston_bull` masks the head and the repair pastes a whole dog onto a neck. |
| `pipeline` | yes | hours | The full loop over every case: verify → retrieve → mask → regenerate → re-verify, up to N attempts. **Resumable** — if it dies because someone took the card, re-run with `--resume outputs/pipeline_<TS>`. Never re-drafts; it reads `outputs/screen_latest/`. |
| `pipeline --proto-arm {ceiling,retrieved}` | yes | hours | **Default off.** Also builds a prototype bank and records image-side scores in `streams.json`. **Records only** — the live verdict logic in `ragregen/verify/grounded.py` is unchanged, so this does not change any pass/fail outcome. |
| `pipeline --mask-prototypes` | yes | hours | **Default off.** Uses the prototype bank to pick the mask box instead of the default classifier. **This changes mask geometry, so it changes results.** Requires `--proto-arm` to be something other than `none`; errors out otherwise. |
| `reverify` | yes | minutes | Re-grades a finished run's images (masks and attempts already on disk) with the current verify code, into a **new** run directory — regenerates nothing and never writes to the source run. Needs `--run outputs/<tag>_<TS>`; `--dataset` defaults to `configs/dataset_common.yaml`, override it if reverifying a different dataset. Accepts `--proto-arm`. |
| `report` | yes, light | ~5 min | Metrics over a finished pipeline run: DINO identity against **held-out** references, preservation, CLIP/SigLIP. Writes `report.md` and `report.json` into the run directory. Needs `--run outputs/pipeline_<TS>`. |
| `c1` | no | seconds | fuses the streams, sweeps τ and δ, writes `c1.md` |

Run them in that order. `screen` is a gate, not a formality:

- **`screen`** answers whether the premise holds. If FLUX.1-Kontext already renders your concepts
  correctly, the verifier passes everything and there is nothing for the pipeline to repair. If
  fewer than ~30% of drafts fail, **stop and pick rarer concepts** — scoring the rest would produce
  a null result for a boring reason. This risk is real: the predecessor project assumed SDXL, and
  FLUX is much stronger on long-tail concepts.

Everything lands in `outputs/<stage>_<TIMESTAMP>/`, with `outputs/<stage>_latest` symlinked to the
most recent. Runs never overwrite each other — an ablation cannot clobber the baseline it exists to
be compared against. Use `--tag` to name a sweep.

---

## 5. Reading the results

`outputs/<tag>_<TS>/summary.md` carries the main table; `per_case.csv` has per-case numbers.

### 5.1 The arms

| arm | what it does | what it tells you |
|---|---|---|
| `no_rag` | the draft, no retrieval | the baseline |
| `oracle` | regenerate using the **ground-truth** reference | **the core claim** — can reference-guided repair work at all? |
| `full` | regenerate using the **retrieved** reference | the end-to-end system |

Read them as two gaps. **`no_rag` → `oracle` is the contribution.** **`oracle` → `full` is
retrieval error.** If `oracle` does not beat `no_rag`, retrieval quality is irrelevant — the repair
mechanism itself is not working, and improving the database will not help.

### 5.2 The metrics

| metric | range | measures | read it as |
|---|---|---|---|
| **DINO** | 0–1, ↑ | output crop vs. your `gt_refs`, via DINOv3 | **the headline.** Does the output actually look like the real concept? |
| CLIP | ~0–0.4, ↑ | prompt ↔ image alignment (open-CLIP) | comparability with the ImageRAG paper |
| SigLIP | ↑ | prompt ↔ image alignment | same; secondary |
| Preservation | 0–1, ↑ | how much of the unmasked region survived | **guard.** High DINO + low preservation = it wrecked the image |
| Verifier pass-rate | 0–1, ↑ | how many cases the verifier passes | development signal **only** |

**The oracle arm's references are split.** `eval.heldout_refs` (default 1) reserves the last
reference of every case for DINO. The repair never sees it. Without the split, DINO would score
each output against an image the repair was handed — the same objection §4 makes about
`--proto-arm ceiling`.

Three cautions:

1. **DINO is the claim; CLIP and SigLIP are context.** CLIP measures prompt alignment, which cannot
   distinguish an Amur leopard from a generic leopard — that blindness is the entire premise of the
   project. The predecessor's runs showed RAG *losing* on CLIP (−0.11) partly for this reason. A
   flat or slightly negative CLIP delta alongside a clear DINO gain is a **success**, not a failure.
2. **Never quote verifier pass-rate as a result.** The pipeline optimises against the verifier, so
   improvement there is partly guaranteed. It is for debugging.
3. **Always read Preservation next to DINO.** A method that improves identity by repainting the
   whole canvas is not solving the problem.

### 5.3 When a case looks wrong

Open `outputs/<tag>_<TS>/<case_id>/trace.json`. It records every raw VLM reply, every Stream A
per-concept score, and the retrieved file paths for each round (not their similarity scores — see
`ragregen/trace.py`). Check in this order:

1. **The concept name.** Retrieval today searches `case.concept` verbatim (§1) — there is no
   VLM-written query to inspect; that step is designed but not built (orchestration design §9).
   The predecessor pipeline once had its VLM-written query collapse to the literal string `"A"`,
   so SigLIP retrieved a cactus for a sea-lion prompt and the generator faithfully painted a
   cactus — that specific failure mode does not exist here, but a mistyped `concept` or one whose
   dominant sense is wrong (§3.4) is the equivalent silent failure: check it first.
2. **The retrieved images.** Wrong reference in → wrong concept out, always.
3. **The masks** (`mask.png`, and the reference crop). A mask on the wrong animal produces a
   confident, well-executed edit in the wrong place.
4. **The raw VLM replies.** A reply containing `!!!!` means the vision tower produced NaNs.

---

## 6. ⚠️ Disk

```
192.168.6.133:/volume1/mmlab/mmlab2024   14T   14T  270G  99% /mnt/mmlab2024nas
```

**The shared volume is 99% full — about 270 GB free, on a filesystem other people are actively
writing to.** Filling it does not just break your run.

Before uploading a large retrieval corpus:

- 500k original images ≈ **75 GB**. 500k images re-encoded at 384px ≈ **15–25 GB**, and **nothing is
  lost** — every encoder in this pipeline resizes to 384 anyway. Re-encode before uploading.
- Check `df -h /mnt/mmlab2024nas` first, and again before `build-index`.
- Budget for Qwen3-VL (~16 GB), and for FG-CLIP plus its separate conda env if you intend to run
  that bake-off arm.

---

## 7. Files

Everything lives under `/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/rag-regen/`.

### You edit these

| path | what |
|---|---|
| `configs/dataset.yaml` | your test cases: prompts, concepts, ground-truth refs |
| `configs/retrieval_db.yaml` | your retrieval corpus and index location |
| `configs/pipeline.yaml` | knobs: retry budget N, threshold τ, steps, seed, encoder choice |

### You run these

| path | what |
|---|---|
| `scripts/run.sh` | the only entry point — run `./scripts/run.sh --help` for the current stage list |
| `scripts/build_index.py` | corpus → embeddings → FAISS (invoked by `run.sh build-index`) |
| `scripts/screen_premise.py` | drafts every case for hand-labelling (the gate) |
| `scripts/bakeoff.py` | FG-CLIP vs SigLIP across two roles (retriever, crop scorer) |
| `scripts/run_pipeline.py` | the stage-batched scheduler driver |
| `scripts/reverify.py` | re-grades a finished run's images into a new run directory, no regeneration (invoked by `run.sh reverify`) |
| `scripts/report.py` | rebuilds `summary.md` from a finished run |

### You read these

| path | what |
|---|---|
| `outputs/<tag>_<TS>/summary.md` | the arm table |
| `outputs/<tag>_<TS>/per_case.csv` | per-case metric values |
| `outputs/<tag>_<TS>/run.json` | argv, args, GPU, library versions, status |
| `outputs/<tag>_<TS>/run.log` | full console output |
| `outputs/<tag>_<TS>/queue.json` | scheduler state — why resume works |
| `outputs/<tag>_<TS>/<case>/` | `draft.png` `mask.png` `attempt_*.png` `best.png` `trace.json` |

### You do not touch these

| path | what |
|---|---|
| `ragregen/concepts.py` | prompt → list of concept phrases |
| `ragregen/verify/grounded.py` | Stream A — DINO + region-text scoring |
| `ragregen/verify/semantic.py` | Stream B — Qwen3-VL judge |
| `ragregen/verify/fusion.py` | combines both streams into one verdict |
| `ragregen/retrieve.py` | SigLIP + FAISS search — searches `case.concept` verbatim; `ragregen/query.py` (a VLM-written search caption) is designed but not built, orchestration design §9 |
| `ragregen/mask.py` | draft mask and reference mask (DINO → SAM → classifier) |
| `ragregen/regen.py` | FLUX.1-Kontext regeneration |
| `ragregen/schedule.py` | stage-batched work queue and resume |
| `ragregen/metrics.py` | CLIP, SigLIP, DINO, preservation |
| `ragregen/validate.py` | the preflight checks |
| `ragregen/trace.py` | per-case reasoning log |
| `ragregen/env.py` | HF cache paths, GPU reclaim |

### Related projects — reference only, do not modify

| path | what |
|---|---|
| `../ImageRAG/` | the ImageRAG fork; `scripts/kontext_engine.py` and `scripts/finegrained_seg.py` are the basis for `regen.py` and `mask.py` |
| `../rag-edit/` | predecessor (SDXL + IP-Adapter). Its `README.md` documents failures worth not repeating |
| `../.cache/` | shared model weights (`HF_HOME`) |

---

## 8. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `transformers 4.44.2` reported | wrong conda env | use `$CONDA/kontext/bin/python` |
| `AttributeError: ... Qwen3VL...` | wrong conda env | same |
| `no attribute FluxKontextInpaintPipeline` | diffusers < 0.39 → wrong env | same |
| `torch.OutOfMemoryError` | someone else has the card | `nvidia-smi`, wait, re-run the same command (it resumes) |
| Run died mid-stage | shared GPU, or ENOSPC | re-run the identical command; check `df -h` |
| VLM output contains `!!!!` | vision tower NaN | report it — a resolution cap in the code is missing |
| Retrieval returns nonsense | mistyped or wrong-sense `concept` in `configs/dataset.yaml` (§3.4, §5.3) | fix the concept name, or if it looks correct, report it; do not edit code |
| `validate` complains about `gt_refs` | missing ground truth | add them — DINO metric and `oracle` arm both need them |
| Everything passes the verifier | concepts are not rare enough for FLUX | pick rarer concepts (see `screen`, §4) |
| Disk full | see §6 | re-encode corpus at 384px |

Anything that would require editing a `.py` file to fix is a bug — report it rather than patching.
