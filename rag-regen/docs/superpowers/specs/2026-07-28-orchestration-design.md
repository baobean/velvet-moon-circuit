# Orchestration — the round-based scheduler, and surviving a shared card

**Date:** 2026-07-28
**Parent spec:** `docs/superpowers/specs/2026-07-25-rag-regen-design.md` (§3 The loop)
**Depends on:** `ragregen/mask.py` (doc 1), `ragregen/regen.py` + `ragregen/composite.py` (doc 2),
`ragregen/verify/` and `ragregen/retrieve.py` (already built)
**Prior findings:** `docs/findings/2026-07-28-masking-result.md`,
`docs/findings/2026-07-28-regeneration-result.md`

**Doc 3 of 4** in the regeneration half: masking → regeneration → **orchestration** → metrics.

---

## 1. The claim

Docs 1 and 2 built the pieces that repair one case when a human tells them to. This doc runs them
over a whole dataset, unattended, on a GPU shared with other researchers — and picks up where it
stopped when one of those researchers takes the card.

The scheduler is the only component whose correctness is measured in *what survives a crash*.
Everything else in this project can be re-run; a run that loses eight hours of FLUX output because
it kept its state in memory cannot.

## 2. What already exists

| | |
|---|---|
| `ragregen/draft.py` | `Drafter`, text-only FLUX t2i. Used by `screen`, not by this doc. |
| `ragregen/mask.py` | `mask_draft`, `mask_reference`, `Masker` — doc 1 |
| `ragregen/regen.py` | `stitch`, `Inpainter`, `KontextConfig` — doc 2 |
| `ragregen/composite.py` | `cutout_from`, `composite_back` — doc 2 |
| `ragregen/retrieve.py` | `Retriever.search(phrase, k) -> [Hit]`, FAISS |
| `ragregen/verify/grounded.py` | `GroundedVerifier` — Stream A, DINO + crop scorer |
| `ragregen/verify/semantic.py` | `SemanticVerifier` — Stream B, Qwen3-VL |
| `ragregen/verify/fusion.py` | `fuse(grounded, semantic) -> Verdict(ok, score, agreement, …)` |
| `ragregen/trace.py` | `open_run(tag) -> RunDir`, `CaseTrace`, `RunDir.finish(status)` |
| `ragregen/env.py` | `reclaim_gpu()` |

**Nothing here needs changing.** Doc 3 adds `ragregen/schedule.py` and `scripts/run_pipeline.py` and
calls the above. If this doc finds itself editing `mask.py` or `regen.py`, the boundary was drawn
wrong.

## 3. What forces the design: models do not co-fit

One RTX 4090, 24 GB, shared (`RUNBOOK.md` §2.3). Residency, measured or documented:

| stage | models | ~VRAM | load cost |
|---|---|---|---|
| `grounded` | GroundingDINO + SigLIP | ~3 GB | seconds |
| `semantic` | Qwen3-VL | ~16 GB | minutes |
| `retrieve` | SigLIP + FAISS | ~2 GB | seconds |
| `mask` | GroundingDINO + SAM | ~9.4 GB | seconds |
| `regen` | FLUX-nf4 | ~12 GB | **5 min 21 s** |

FLUX and Qwen3-VL cannot be resident together, so verification and regeneration can never interleave
per case. A naive per-case loop reloads FLUX 30 times and spends 73% of wall-clock loading weights —
about 8 hours of pure loading for 30 cases. Stage-batching is not an optimisation, it is the only
affordable shape.

## 4. The round structure

Retries are what make this non-trivial: a retry needs a fresh verification, verification needs the
VLM, and the VLM needs FLUX evicted. So retries happen in **rounds**, each round costing one FLUX
load and one VLM load.

```
  screen's labelled drafts
        │
  ROUND 0   grounded@0 → semantic@0 → fuse
        │   passing cases are DONE (nothing to repair)
        │
  ONCE  retrieve      k = retry_budget references per failing case
  ONCE  mask          draft mask + all k reference cutouts, cached to disk
        │
  ROUND 1..N  regen@i → grounded@i → semantic@i → fuse
        │     a case that passes leaves the loop; the round stops early when
        │     no case is left unresolved
        │
  select  best of {draft, attempt 1 … attempt N}
```

**`retrieve` and `mask` run once, not per round.** This falls out of a guarantee the RUNBOOK already
makes: *each retry regenerates from the original draft, never from the previous attempt* (§1). The
draft never changes, so its mask never changes. And retrieval depth `k` is already the retry budget
(`pipeline.yaml`: *"N — also the retrieval depth k"*), so every reference a run could ever need is
known the moment retrieval finishes. Masking all of them in one DINO+SAM residency turns N model
loads into one.

At `retry_budget: 3` that is **1 mask load, 3 FLUX loads and 4 Qwen3-VL loads** — the draft's
verification plus one per attempt. Masking inside each round instead would make it 3 mask loads: 10
residencies rather than 8, for no gain, since every input to those masks was already known.

**Best-of selection is the verifier's job here, not the metrics'.** The first attempt whose `Verdict`
is `ok` wins and the loop stops. If no attempt passes, the candidate with the highest fused score
wins — with the draft always in the candidate set, which is what makes the method unable to score
worse than the no-retrieval baseline (`RUNBOOK.md` §1). The identity and preservation metrics are
doc 4's and are computed *after* selection, for reporting; selecting on them would be selecting on
the thing being measured.

## 5. Interfaces

```python
# ragregen/schedule.py -- no models, no torch, no I/O beyond queue.json

#: One entry per MODEL RESIDENCY, not per logical step -- "verify" is two
#: separate loads (DINO+SigLIP, then Qwen3-VL) and a run killed between them
#: must resume at the second, not redo both.
#: `retrieve` and `mask` are attempt-free (§4); the rest are keyed `<stage>@<n>`,
#: where attempt 0 is the draft and 1..N are the repairs.
STAGES = ("grounded", "semantic", "retrieve", "mask", "regen")

class StageAborted(Exception):
    """The card is gone. Checkpoint and stop; resume will retry these cases."""

def is_fatal(exc: BaseException) -> bool:
    """OOM and CUDA errors are fatal to the STAGE, not to the case."""

@dataclass
class CaseState:
    case_id: str
    stages: dict[str, str]          # "grounded@0" | "regen@2" -> done|failed
    attempt: int = 0
    status: str = "pending"         # pending|repaired|unrepaired|failed
    best: str | None = None         # "draft" | "attempt_2"
    error: str | None = None

class Queue:
    @classmethod
    def open(cls, path: Path, case_ids: list[str], *, retry_budget: int,
             screen_run: str) -> "Queue":
        """Load queue.json if it exists, else start one. Resume is this call."""

    def save(self) -> None:
        """Atomic: write .tmp, os.replace. A half-written queue is worse than none."""

    def pending(self, stage: str, *, attempt: int | None = None) -> list[str]
    def mark(self, case_id: str, stage: str, status: str, *,
             attempt: int | None = None, **fields) -> None
    def unresolved(self) -> list[str]
    def needs_round(self, attempt: int) -> bool

def run_stage(queue: Queue, stage: str, fn, *, attempt: int | None = None) -> None:
    """Apply fn to every case pending this stage, isolating per-case failures.

    The single place the §6 policy lives, so no stage handler can get it wrong.
    Checkpoints after each case -- a stage killed at case 20 of 30 keeps 19.
    """
```

`run_stage` takes the per-case work as a callable, so `schedule.py` never imports a model. The driver
owns residency:

```python
# scripts/run_pipeline.py -- owns loading, unloading and the stage order
inpainter = regen.Inpainter(cfg).load()
try:
    schedule.run_stage(queue, "regen", lambda case: _regen_one(case, inpainter),
                       attempt=i)
finally:
    inpainter.free()          # already calls env.reclaim_gpu()
```

## 6. Failure policy: split by cause

Two failure modes, and conflating them is how `../rag-edit` died.

| cause | scope | action |
|---|---|---|
| `torch.OutOfMemoryError`, CUDA errors | the **stage** | checkpoint, raise `StageAborted`, exit non-zero. Every remaining case would fail identically; burning 30 cases to discover that wastes an hour. |
| anything else | the **case** | record the traceback in that case's `CaseState.error` and its `CaseTrace`, mark the stage `failed`, continue to the next case. |

A concept that will not ground, or a corrupt reference, is a fact about one case. It must not block
the other 29, and it must be visible in the report rather than silently dropped.

**A `StageAborted` run exits non-zero and prints the resume command.** Re-running the identical
command is the recovery path; the RUNBOOK already promises this and §7 makes it true.

## 7. `queue.json` and resume

Lives at `outputs/pipeline_<TS>/queue.json`, beside the per-case directories `trace.py` already
creates.

```json
{
  "schema": 1,
  "tag": "pilot",
  "screen_run": "outputs/screen_20260726_233320",
  "retry_budget": 3,
  "cases": {
    "boston_bull": {
      "stages": {"grounded@0": "done", "semantic@0": "done",
                 "retrieve": "done", "mask": "done",
                 "regen@1": "done", "grounded@1": "done", "semantic@1": "done"},
      "attempt": 1, "status": "repaired", "best": "attempt_1", "error": null
    }
  }
}
```

**Resume is `Queue.open` finding an existing file.** `pending(stage)` returns only cases whose
`stages` lack a `done` entry, so a completed stage costs nothing on a resumed run — including the
model load, which the driver skips entirely when `pending()` is empty. That last point is the
difference between a resume that takes seconds and one that reloads FLUX to do nothing.

**Writes are atomic and after every case.** `.tmp` + `os.replace`. The failure being defended against
is a SIGKILL mid-write leaving unparseable JSON, which would strand the whole run — the exact
scenario resume exists for.

**`--tag` picks the run directory; resume needs the path.** A resumed run is
`./scripts/run.sh pipeline --resume outputs/pipeline_20260728_143000`. Without `--resume` a fresh
directory is created, because silently adopting the newest run is how someone appends to the wrong
experiment.

## 8. `scripts/run_pipeline.py`

A new `pipeline` stage in `run.sh`. **`screen`, `score-a`, `score-b` and `c1` are untouched** — they
are validated C1 tooling and doc 3 has no reason to destabilise them.

```
./scripts/run.sh pipeline [--tag NAME] [--resume DIR] [--arm oracle|full] [--limit N]
```

Drafts are **read from `outputs/screen_latest/`**, not generated. `screen` is the gate the RUNBOOK
insists on, its drafts are the ones that were hand-labelled, and re-drafting at a different seed
would mean the labels no longer describe the images being repaired. **The run errors out if no
`screen` run exists** rather than helpfully drafting something.

## 9. Out of scope

- **Query formulation** (`ragregen/query.py`) and the FLUX **instruction text**. Doc 2 §192 credits
  both to doc 3; they are prompt-wording problems, not scheduling problems, and bundling them would
  put two unrelated review surfaces in one plan. The `full` arm retrieves on `case.concept` until
  that doc exists, and the scheduler takes the query as a callable so nothing needs rewriting later.
- **Metrics and the report** — `ragregen/metrics.py`, `scripts/report.py`, and the arm *table*.
  Doc 4. `--arm` exists here because the scheduler must know where a reference comes from
  (`gt_refs` for `oracle`, the retriever for `full`); scoring the arms against each other is doc 4's.
- **Parallelism.** One GPU, one process. Nothing here is concurrent and nothing should be.
- **A daemon, or automatic retry of a `StageAborted` run.** The operator decides when the card is
  quiet; a scheduler that retries into a busy GPU is how you become the problem.

## 10. Risks

**R1 — The round loop is only affordable if early-stop works.** If most cases fail all N attempts,
the run costs the full 3 FLUX + 4 VLM loads regardless. That is the honest worst case and it is
hours. Mitigation is reporting per-round pass counts so an operator can kill a hopeless sweep, not
pretending the cost is smaller.

**R2 — Round 0 may pass nearly everything.** Then there is nothing to repair and the
scheduler is exercised on an empty set. This is the premise risk `screen` exists to catch
(`RUNBOOK.md` §4: *stop and pick rarer concepts*), and it is a finding about the dataset rather than
a defect in this doc — but doc 3 is where it becomes visible at scale.

**R3 — Resume correctness is hard to test honestly.** A test that calls `save()` then `open()` in
one process proves serialisation, not crash-safety. The suite must kill a real subprocess mid-stage
and resume it (§11), or the guarantee is decorative.

**R4 — Caching k reference cutouts costs disk.** `outputs/` already carries a disk warning
(`RUNBOOK.md` §6). N=3 RGBA cutouts per failing case is small beside the attempt PNGs, but it is not
zero and it is written before anyone knows whether attempt 1 will succeed.

## 11. Testing

Everything below runs on CPU with fake stage callables. `schedule.py` imports no models, which is
what makes that possible — the same split that made `composite.py` testable in doc 2.

- **Resume skips completed cells.** A queue with `regen@1: done` returns no cases from
  `pending("regen", attempt=1)`.
- **Resume skips the model load.** The driver's load callable is not invoked when `pending()` is
  empty — asserted with a spy, because this is the difference between a seconds-long resume and a
  five-minute one.
- **Crash mid-stage keeps prior cases.** Run a real subprocess over 5 cases, `SIGKILL` it during
  case 3, re-open the queue: cases 1–2 are `done`, and the file parses.
- **Atomic write.** A `save()` interrupted before `os.replace` leaves the previous queue intact.
- **OOM aborts the stage.** A callable raising `torch.OutOfMemoryError` on case 2 of 5 raises
  `StageAborted`, and cases 3–5 are never attempted.
- **Any other exception isolates.** A callable raising `ValueError` on case 2 of 5 leaves cases 3–5
  processed, case 2 `failed` with its traceback recorded.
- **Early stop.** With every case passing at attempt 1, `needs_round(2)` is False and round 2 never
  loads a model.
- **Retry regenerates from the draft.** The image handed to `regen` at attempt 2 is byte-identical to
  the one handed to attempt 1 — the RUNBOOK guarantee, pinned.
- **Best-of includes the draft.** When no attempt passes and the draft outscores all of them, the
  selected candidate is `"draft"`.
- **The mask stage runs once.** Across 3 rounds, the masking callable is invoked once per case.

**Requiring a GPU** (`@pytest.mark.gpu`): none. The scheduler is fully testable without one, and
that is deliberate — doc 2's finding was that the card is unavailable for days at a time
(`findings/2026-07-28-regeneration-result.md` §4).

**Smoke:** `./scripts/run.sh pipeline --limit 2 --arm oracle` once a `screen` run exists, killed
halfway and resumed, to prove §7 on real weights rather than fakes.
