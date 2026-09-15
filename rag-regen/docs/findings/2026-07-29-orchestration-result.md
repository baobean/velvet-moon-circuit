# Orchestration end to end — the rule was right, the wiring was missing, and a healthy case was billed as a repair

**Date:** 2026-07-29
**Spec:** `docs/superpowers/specs/2026-07-28-orchestration-design.md`
**Plan:** `docs/superpowers/plans/2026-07-29-orchestration.md`
**Runs:** `outputs/pipeline_20260729_145620/` (pre-fix, CPU),
`outputs/pipeline_fixed_20260729_185707/` (post-fix, CPU),
`outputs/smoke_resume_20260729_230416/` (GPU, killed and resumed)
**Code:** `ragregen/schedule.py`, `scripts/run_pipeline.py`

---

## 1. What was run

Three runs of `scripts/run_pipeline.py`, all `--arm oracle --mechanism stitch`. `stitch` needs no
FLUX weights, so every run exercises the whole state machine without the diffusion path — which is
the point: the scheduler is what is under test, not the repair.

| run | cases | device | wall clock |
|---|---|---|---|
| `pipeline_20260729_145620` | 1 | cpu | 14:56 → 18:42 |
| `pipeline_fixed_20260729_185707` | 1 | cpu | 18:57 → 19:43 |
| `smoke_resume_20260729_230416` | 2 | cuda | 23:04 → 23:49 (intended as a kill/resume test; ran to completion instead — §3) |

`outputs/` is gitignored, so the tables here are the durable record.

## 2. The round loop never stopped early, and it mislabelled the case

`african_grey_parrot`'s draft passed the verifier outright:

| candidate | verdict | score |
|---|---|---|
| `draft` | true | 1.0 |
| `attempt_1` | true | 1.0 |
| `attempt_2` | **false** | **0.0** |
| `attempt_3` | true | 1.0 |

The pipeline ran all three regeneration rounds anyway — 15:45 to 18:01, about 2h16m — on a case
that was already correct at 1.0. `attempt_1` bought nothing, and `attempt_2` actively regressed to a
failing verdict.

**The compute is the smaller half of the cost.** The run recorded the case as `status: "repaired"`,
`best: "attempt_1"`. A case that was never broken was credited as a repair, and the image reported
as the repair was a re-render no better than the draft it replaced. Doc 4 computes repair rate over
exactly this field; the bug would have inflated it silently, and nothing downstream would have
looked wrong.

**Cause.** `needs_round()` asks for cases still `pending`. Nothing moved a case off `pending` until
`_finalise`, which runs *after* the last round. The early-stop rule was correct and its caller was
correct — nothing connected them.

**Why the suite missed it.** The unit test for the rule set `status="repaired"` by hand. It tested
the rule in isolation and never asked whether anything in the system would ever set that field. The
three regression tests added with the fix go through `_resolve`, so they fail if the wiring is
removed again.

**Fix** (`e9bdb86`): `_resolve` fires the moment a verdict passes, setting `best` and moving the case
off `pending`. Re-run on the identical case: two stage cells instead of thirteen, `status:
"unrepaired"`, `best: "draft"`.

### `unrepaired` means two opposite things, and doc 4 cannot tell them apart

This one is read off the source, not off a run, so the botched smoke in §3 does not affect it. Two
different code paths both write `status: "unrepaired"`:

- `_resolve` — the draft **passed**, so the case stopped at round 0. It never needed repair.
- `_finalise` — `select_best` returned `draft`, i.e. the draft **failed** and no attempt beat it.
  Repair was tried and did not work.

`_finalise`'s rule is `"unrepaired" if best == "draft" else "repaired"`, which cannot consult whether
the draft passed. A healthy case and an unfixable case land on the same label.

Both appeared in the same 2-case run: `african_grey_parrot` at `draft [true, 1.0]` and
`amur_leopard` at `draft [false, 0.0]` with all three attempts also `[false, 0.0]`. Identical status,
opposite meanings.

**Doc 4 must not compute a repair rate from `status` alone.** The denominator it wants — cases that
actually needed repair — is `scores.json`'s `draft` verdict, not this field. Recorded here because
the natural reading of `unrepaired` is the wrong one, and doc 4 is the consumer.

## 3. Resume does the remaining work and only the remaining work

Run `resume2_20260729_235739`, 2 cases on the GPU, `SIGKILL`ed at the point where the two cases were
in deliberately different states:

| case | state at kill |
|---|---|
| `african_grey_parrot` | resolved — `unrepaired`, `best: draft`, 2 cells |
| `amur_leopard` | mid-flight — `grounded@0`, `semantic@0`, `retrieve` done; **`mask` pending** |

The resumed run completed normally (`[pipeline] done`, `status: ok`) and its `run.json` records
`--resume`, so the provenance of a resumed run is what it claims to be.

**The load arithmetic is the proof.** Counting completed weight loads:

| run | loads |
|---|---|
| uninterrupted 2-case run (`smoke_resume`, §1) | 22 |
| killed run, up to the kill | 6 |
| resume | 16 |

6 + 16 = 22. The resume paid for exactly the work that remained and reloaded nothing the first
process had already done. Had `stage_with_model`'s guard been absent, the resume would have loaded
the grounding and retrieval stacks again to discover they had nothing to do.

**Completed work was not redone**, confirmed by mtime rather than by log-reading:

- `african_grey_parrot/streams.json` — unchanged across the resume. The resolved case ran no VLM
  and no grounding.
- `amur_leopard/refs.json` — unchanged. `retrieve` was not re-run.

**The queue trusted the file over the arguments.** The resume command omitted `--limit`, which
defaults to 0 — all 22 dataset cases. The queue came back with the original 2. This is Task 1's
`test_save_then_open_round_trips` holding on real hardware rather than in `tmp_path`: a resumed run
that honoured a fresh `--limit` would silently drop cases the first run had already paid for.

### One wart: resolved cases are re-fused on resume

`african_grey_parrot/scores.json` *was* rewritten during the resume, though `streams.json` was not.
`_verify_round` re-fuses any case whose `streams.json` holds both streams, without consulting
`status`, so an already-resolved case has its verdict recomputed from cached streams. No model
loads, the inputs are identical, and the result is the value already there — but the write is
unnecessary, and a reader comparing timestamps will wonder why a finished case was touched.

### Getting the kill right

The first attempt at this test was botched and is worth recording so it is not repeated. The run was
launched as `cd <dir> && nohup ./scripts/run.sh ... &`, so `$!` named the **subshell**, not the
pipeline — `run.sh` was its child and `python` a grandchild. `kill -9 $!` killed a wrapper and
orphaned the pipeline, which ran to completion while a `--resume` process was started alongside it.
For eight minutes two pipelines shared one card and one `queue.json`, until the kernel OOM killer
reaped the second. Every measurement from that attempt was discarded.

It did establish one thing for free: **concurrent writers to one `queue.json` do not corrupt it.**
Two processes marked cells in the same file for eight minutes and it still parsed, every completed
cell intact — the atomic-write property from Task 1 holding under a condition no test covers.

Kill the process by its own PID:

```bash
kill -9 $(pgrep -f 'bin/python scripts/run_pipeline.py')      # NOT $!
```

## 4. Three operational notes

**`Ctrl-C` is not available to a backgrounded run.** The plan's Step 4 says to interrupt with
Ctrl-C. A job started with `&` from a non-interactive shell has `SIGINT` set to `SIG_IGN`, so
`kill -INT` does nothing — verified, the process ignored it and kept running. `SIGKILL` is the
right instrument, and the stronger test anyway: it is what design §R3 promises to survive, with no
cleanup, no `atexit`, no flush.

**`$!` is the wrong PID for this pipeline.** See §3. `run.sh` `exec`s python, so a *direct* launch
gives one PID — but wrapping it in `cd … && nohup … &` puts a subshell in front, and `$!` names the
subshell. Use `pgrep -f run_pipeline.py`.

**The run directory scaffolds a folder per dataset case, not per queued case.** A 2-case run with
`--limit` absent created all 22 directories, 20 of them empty, because `by_id` is built from the
whole dataset while the queue is built from the file. Cosmetic, inside gitignored `outputs/`, and
the queue itself was untouched — recorded so the next reader does not mistake it for the queue
having grown.

## 5. What this does not prove

- **The `inpaint` mechanism has still never run under the pipeline.** Every run here is `stitch`.
  The scheduler is mechanism-agnostic by construction, but that is an argument, not a measurement.
- **No full-dataset run exists.** The largest run was 2 cases of 22. Nothing here measures how the
  state machine behaves over a long run, which is the case it was built for.
- **The `full` arm is untested end to end.** All three runs are `oracle`.
- **No repair quality is claimed.** `african_grey_parrot` entered at 1.0, and `amur_leopard` failed
  at 0.0 on every candidate under a mechanism that pastes a rectangle. Repair measurement is doc 4's.

## 6. Status

358 tests pass, 11 `@pytest.mark.gpu` deselected. Tasks 1–10 of the orchestration plan are complete,
including Task 10's Step 4 kill-and-resume smoke (§3), plus the `_resolve` fix that smoke forced.
