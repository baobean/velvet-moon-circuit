# rag-regen — C1 verifier validation

**Date:** 2026-07-27
**Status:** design approved, not yet implemented
**Parent spec:** `docs/superpowers/specs/2026-07-25-rag-regen-design.md`
**Depends on:** Plan 1 (verifier, retrieval, operator interface) — complete

---

## 1. What this validates

> **C1 (verifier).** A grounded stream (GroundingDINO + a region-text scorer) fused with a semantic
> stream (open-weights VLM) detects prompt-image mismatch at least as well as a single VLM judge,
> and — unlike it — produces continuous per-concept scores that can be calibrated.

Plan 1 built the entire verifier — Stream A, Stream B, fusion — but never ran it against labelled
data, because no labels existed. The premise gate has now produced 22 hand-labelled drafts. The
parent spec §10 notes that step 0 "pays twice": the labels are both the premise screen and the
ground-truth set for validating the verifier. **This is the second payment.**

Two things follow, and both are cheap:

1. C1 is measurable with **zero diffusion**. Stream A and Stream B re-score images that already
   exist. No FLUX, no retrieval, no masking, no regeneration.
2. `configs/pipeline.yaml` currently ships `tau: 0.25`, a number with **no empirical basis**. The
   verifier is about to be embedded in a ~9-hour experiment loop. Calibrating it first is what stops
   every downstream number from being noise.

### What this is explicitly not

No `query.py`, `mask.py`, `regen.py`, `schedule.py`, or `metrics.py`. No Stage 2 matrix. Claim C2
needs the generator and fails independently of C1 (parent spec §1); mixing them would make this the
repair plan wearing a disguise. Those are the next plan.

---

## 2. Inputs and ground truth

Input is a screen run's `labels.csv` — currently `outputs/screen_20260726_233320/`, 22 cases,
11 rare concepts and 11 controls.

**Two ground truths, reported side by side.** The operator's labels mixed two standards, and the
distinction is load-bearing rather than pedantic:

| column | definition | current counts |
|---|---|---|
| `verdict_identity` | **primary.** Is it the right thing? | 15 fail / 7 pass |
| `verdict` | **secondary.** Any visible defect, as originally labelled | 18 fail / 4 pass |

On three controls the operator stated the concept *was* correct and failed it on appearance —
`cactus` ("is a cactus; placement weird"), `stack_of_books` ("is a stack of books; book design not
correct"), `sushi` ("**it is a sushi**; meat looks weird"). Three more were left undecided on the
same fine-detail grounds (`cardinal_bird`, `golden_retriever`, `hedgehog`).

This matters because **Stream A structurally cannot see aesthetics.** It scores crop-versus-phrase
similarity. Grading it on "the meat looks weird" measures nothing. It matters again downstream:
reference-guided regeneration repairs identity, and the DINO metric that judges the final claim
measures identity — so a verifier tuned to the as-labelled standard would fire on cases the method
cannot repair.

**Operator ruling, 2026-07-27.** Three cases the operator left unsure — `cardinal_bird` (leg
colour), `golden_retriever` (eyebrows, fur), `hedgehog` (sparse spines, long legs, colour) — are
**identity failures**: on a fine-grained concept the fine details *are* the identity, which is the
same reason the African grey failed for lacking its eye patch. The three aesthetic controls
(`cactus`, `stack_of_books`, `sushi`) remain passes, because there the operator stated the concept
itself was correct.

Both columns are operator-editable in `labels.csv`. Rows blank in a column are excluded from that
column's metrics, and the exclusion count is printed. Blanks are never guessed at.

`scripts/screen_premise.py` is amended to emit `verdict_identity` alongside `verdict`.

---

## 3. Architecture

Three stages behind the existing `run.sh` — two on GPU, one pure CPU.

```
outputs/screen_<ts>/labels.csv
        │
   [score-a]  GroundingDINO + SigLIP crop scorer      GPU, ~5 min
        │     └─→ runs/<tag>/stream_a.json
        │           {case_id: {phrase: {sim, box, dino_conf, kind, abstained}}}
        │
   [score-b]  Qwen2.5-VL-7B                           GPU, ~10 min
        │     └─→ runs/<tag>/stream_b.json
        │           {case_id: {raw, ok, degenerate, issues[]}}
        │
   [c1]       no models                               CPU, seconds
              └─→ runs/<tag>/c1.md + c1.json
```

**The two GPU stages never co-reside.** `score-a` unloads before `score-b` loads. Qwen-7B (~16 GB)
plus the detector stack does not fit beside other users on a 24 GB card, and Plan 1 measured the
card at ~9 GB already occupied by other researchers. This is the parent spec's stage-batching
rationale (§4) applied to two stages, without building the general work queue — `schedule.py` is
designed for the 5-stage repair loop with retry budgets, and fitting it to a 2-stage no-retry
consumer would shape it around the wrong requirements.

### Why the scores are cached, not the verdicts

Calibrating τ is a sweep, not a run. Against a live verifier, 50 τ values cost 50 passes of
GroundingDINO plus SigLIP. Against cached similarities it is arithmetic.

Plan 1's `ConceptScore` already carries continuous `sim` alongside the thresholded `state`, and
`ABSTAIN` — no box found, or an unboxable `kind` — is τ-independent and recorded as `sim=None`. So
Stream A runs **once at any τ** and every threshold is recovered offline:

```python
state(τ) = "ABSTAIN" if sim is None else ("PRESENT" if sim >= τ else "MISSING")
```

The `>=` is deliberate and matches `grounded.py:83`, which Plan 1 pins with a dedicated boundary
test. `c1.py` must not silently diverge from it.

**No change to Plan 1's verifier is required.** Its dependency-injection discipline — every
model-using class receives its models as constructor arguments — is what makes this possible.

---

## 4. Components

| file | responsibility | status |
|---|---|---|
| `ragregen/vlm.py` | `QwenVLM`, implementing Plan 1's `VLM` protocol: `ask(image, prompt) -> str` | new |
| `ragregen/c1.py` | pure metrics: `state_at_tau`, `confusion`, `wilson_ci`, `sweep_tau`, `render_report` | new |
| `scripts/score_a.py` | Stream A sweep → `stream_a.json` | new |
| `scripts/score_b.py` | Stream B sweep → `stream_b.json` | new |
| `scripts/c1_report.py` | fuse, sweep, render | new |
| `scripts/run.sh` | dispatch `score-a`, `score-b`, `c1` | modify |
| `scripts/screen_premise.py` | emit `verdict_identity` | modify |

`ragregen/c1.py` contains no models and no I/O. Every metric is a pure function over cached scores,
which is what makes the arms comparison testable on CPU with fixtures — the same split that made
Plan 1's `fusion.py` and `bakeoff.render_summary` verifiable without a GPU.

---

## 5. Metrics

**Three arms.** `grounded-only` (fail iff any concept is MISSING at τ), `semantic-only`, `fused`
(Plan 1's `fuse`). `semantic-only` **is** the single-VLM-judge baseline C1 is defined against, so
the comparison falls out of the same run at no extra cost.

### The caveat that shapes the evaluation

`fuse()` fails if *either* stream fails — it is an OR. Therefore fused recall is mathematically
≥ both arms and fused precision ≤ both. **"Fusion catches more" is guaranteed and proves nothing.**

### The base rate makes F1 misleading too

The identity ground truth is **15 fail / 7 pass — a 68% failure base rate.** A degenerate verifier
that says FAIL to everything therefore scores:

| metric | always-FAIL verifier |
|---|---|
| accuracy | 0.68 |
| recall | 1.00 |
| precision | 0.68 |
| **F1** | **0.81** |
| **balanced accuracy** | **0.50** |
| **MCC** | **0.00** |

An F1 of 0.81 from a verifier that does nothing at all. So F1 cannot be the headline either.

**The headline metrics are balanced accuracy and MCC**, both of which sit at chance for the
degenerate case. `always-FAIL` and `always-PASS` are reported as explicit baseline rows in every
table, so no arm can be praised for clearing a bar that a constant beats.

### Reported per arm, per ground truth

Confusion matrix (positive = the verifier says FAIL), precision, recall, F1, balanced accuracy and
MCC, each with a **Wilson 95% interval** where it is a proportion — the normal approximation is
unreliable at n=22. Plus the two constant baselines above, and the `agreement` breakdown Plan 1
already emits (`BOTH_FAIL` / `GROUNDED_ONLY` / `SEMANTIC_ONLY` / `BOTH_PASS`), whose disagreement
rate is the parent spec's stated evidence (§2) that decomposing the single judge buys anything.

### Two rates that must be reported or the arms are misread

- **Degenerate rate** (Stream B). Plan 1 fails closed by design, so a broken VLM reads as a perfect
  failure detector. Non-zero degeneracy inflates semantic recall, and the report says so.
- **Abstain rate** (Stream A). If most concepts abstain, `grounded-only` is inert and `fused` is
  silently just Stream B wearing a second name.

### On τ, stated honestly

The sweep reports the full F1-versus-τ curve and the maximizing τ\*. But **τ\* is selected on the
same 22 cases it is scored on** — it is fitted to the evaluation sample, and at n=22 there is no
held-out split worth making. The report labels it "τ\* on this sample," never "the calibrated τ."

### On statistical power, stated up front

n=22 with 15 identity failures cannot settle C1. One case moves a proportion by 4.5%, and Wilson
intervals at this size span roughly ±20 points. Worse, only **7 negatives** carry the entire
specificity estimate, so precision and balanced accuracy are the least stable numbers in the report
— a single false positive moves specificity by 14 points. Unless fusion beats the single judge by a
wide margin, the arms will not separate — and the report must say that rather than presenting an
overlapping difference as a result.

This is deliberate. The harness is the deliverable; expanding the sample is then a re-run of the
same code, at ~45 s of GPU per additional draft plus the operator's labelling time.

---

## 6. Error handling

- Missing or unreadable draft → named error, non-zero exit. Never a silent skip.
- Degenerate VLM reply → recorded, counted, surfaced in the report. Never silently a FAIL.
- Every concept abstaining on a case → the case cannot fail `grounded-only`; counted in the abstain
  rate.
- Each stage skips cases already present in its JSON unless `--force`, so a contended GPU costs one
  case rather than the run.
- Outputs are JSON and Markdown. No disk-size risk on a volume at 71 GB free.

---

## 7. Testing

- `c1.py` is pure and gets fixture tests throughout: known confusion matrices, Wilson intervals
  against published values, and the `sim == τ` boundary — which must stay consistent with the `>=`
  Plan 1 already pins.
- The arms comparison is **mutation-checked**. Plan 1 shipped four tests that passed for the wrong
  reason; assertions claiming to pin behaviour are verified by breaking the behaviour.
- `QwenVLM`'s CPU fake is built from `inspect.signature` of the real Qwen API, **not** from
  assumption. Plan 1's `FakePipe` declared a signature the real `FluxKontextPipeline` did not have,
  producing a fully green suite that crashed on the first real case. This is now a standing rule.
- The scripts get exit-code contract tests, as `validate_cli` has.
- GPU tests are marked `@pytest.mark.gpu` and excluded by default.

---

## 8. Environment

Inherits Plan 1's constraints. Carried forward because each has already cost something:

- Interpreter `/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python` (3.11.15,
  transformers 5.14.1, diffusers 0.39.0).
- `HF_HOME=/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.cache` before any transformers import.
- **Stream B uses `Qwen/Qwen2.5-VL-7B-Instruct`, already cached.** Qwen3-VL is deliberately *not*
  downloaded: it stays unused so it can serve as the held-out judge the parent spec asks for (§5),
  and the volume has 71 GB free against a 79 GB cache.
- One shared RTX 4090, routinely ~9 GB occupied by others. Never hold two large models.
- **transformers 5.x changes return contracts.** `get_image_features`/`get_text_features` return
  `BaseModelOutputWithPooling`, not tensors, on both SigLIP and CLIP. Verify Qwen's generation API
  at the library source rather than from model cards or memory.
- Dependency injection is mandatory: no module-level model loading, no `from_pretrained` at import.

---

## 9. Risks

| # | risk | mitigation |
|---|---|---|
| R1 | n=22 cannot separate the arms | Stated in the report; Wilson intervals shown; harness built for cheap re-run |
| R2 | τ\* overfits the evaluation sample | Labelled as sample-fitted, not calibrated; curve reported, not just the argmax |
| R3 | Stream A abstains on most concepts, making C1 a Stream-B-only measurement | Abstain rate reported per arm; a high rate invalidates the grounded arm explicitly |
| R4 | Qwen2.5-VL degenerates on the JSON judge template | Degenerate rate reported; Plan 1 already fails closed and preserves `raw` for inspection |
| R5 | Contended GPU kills a stage | Per-case resume via the stage JSON; stages are short-lived |
| R6 | Fusion's OR structure misread as evidence for C1 | Balanced accuracy and MCC are the headline; recall gain is documented as an identity |
| R7 | 68% base rate flatters every arm — F1 0.81 is reachable by a constant | `always-FAIL` and `always-PASS` reported as baseline rows in every table |
| R8 | Only 7 negatives, so precision/specificity are very unstable | Flagged in the report next to those numbers, with Wilson intervals |

---

## 10. Success criteria

The plan is done when:

1. `./scripts/run.sh score-a && ./scripts/run.sh score-b && ./scripts/run.sh c1` produces `c1.md`
   from the 22 labelled drafts.
2. `c1.md` reports all three arms against both ground truths, with Wilson intervals, the
   agreement breakdown, the degenerate and abstain rates, and the `always-FAIL` / `always-PASS`
   baseline rows.
3. The F1-versus-τ curve and τ\* are reported, labelled as fitted to this sample.
4. The report states plainly whether n=22 separates the arms — including, if so, that it does not.
5. CPU suite green; GPU tests pass when the card is free.

Whether C1 *holds* is a finding, not a success criterion. A well-measured negative result is a
successful outcome of this plan.
