# Verifier persistence — one serialiser, a score that can be absent, and a re-grade without the GPU drafting anything

**Date:** 2026-08-12
**Parent spec:** `docs/superpowers/specs/2026-07-25-rag-regen-design.md` (§2 Modules, §11 Logging)
**Supersedes the persistence in:** `scripts/run_pipeline.py:311 _stash`, `:333 _record_score`
**Prior findings:** `docs/findings/2026-07-27-c1-result.md`,
`docs/findings/2026-07-29-orchestration-result.md`
**Produces:** `ragregen/verify/record.py`, `scripts/reverify.py`, edits to
`ragregen/verify/fusion.py`, `ragregen/schedule.py`, `scripts/run_pipeline.py`, `scripts/score_a.py`,
`docs/RUNBOOK.md`

---

## 1. The claim

The doc-5 campaign ran a verifier whose numbers were destroyed between two stages of the same
process. This doc restores them, makes the fused score honest about its own absence, and re-grades
the 92 finished cases **without regenerating a single image**.

Nothing here changes what the verifier computes. It changes what survives the computation.

## 2. The root cause is a duplication, not a typo

Two implementations of "score Stream A and save the result", and they discard opposite halves:

| | persists | discards |
|---|---|---|
| `scripts/score_a.py:30 score_case` | every `sim`, `sim_coarse`, `sim_proto`, `sim_proto_coarse`, all 8 candidates, box, `dino_conf` | the verdict — deliberately, since its τ is arbitrary |
| `scripts/run_pipeline.py:375 _stash` | the verdict | **every number behind it** |

`score_case` lives in a script, so the package cannot import it, so the pipeline grew its own lossy
version. The fix is to give the serialiser one owner inside `ragregen/`, and to have the pipeline
persist **both** halves — evidence because the run must be re-gradeable, verdict because resume
depends on it.

### What the loss cost, measured on `outputs/doc5_20260810_183939`

`_record_score` rehydrates `ConceptScore(phrase, "concept", state)`; every other field defaults to
`None`. So in `fuse()`, `sims` is always empty and the score falls through to
`1.0 if semantic.ok else 0.0`. Four consequences:

1. **Selection ran on Stream B alone.** Every one of the 202 verdicts scored exactly 1.0 or 0.0.
   Three of the 19 `repaired` cases — `anas_platyrhynchos`, `tiger`, `guacamole` — selected a
   **failing** attempt (`[false, 1.0]`) over their draft (`[false, 0.0]`), because a failing
   candidate that Stream B happened to pass outranks everything on that axis.
2. **`Verdict.target` is arbitrary** — `min(failing, key=sim or 0.0)` over all-`None` sims returns
   whichever phrase iterated first.
3. **`kind` is `"concept"`**, which is not in `concepts.KINDS`.
4. **The run cannot be re-graded.** 202 verdicts of GPU time exist as 202 bare strings. No τ can be
   re-swept, no per-concept margin recovered. `score_a.py` writes exactly what would have been
   needed; the pipeline wrote none of it.

Grounded states across the campaign were **MISSING 135 / PRESENT 64 / FINE_MISMATCH 2 /
ABSTAIN 41**. Stream A failed 56% of the concepts it could see and nothing on disk says whether
those sat just under τ = 0.25 or at zero.

## 3. `trace.py` is dead code

`CaseTrace` — `vlm(stage, raw)` with a `garbage` flag, `grounded(scores)`, `hits()`, `save()` — is
constructed by nothing. `grep -rn CaseTrace scripts/ ragregen/` returns only its own definition.

Parent spec §11 wrote it for one purpose: *"when a RAG pipeline underperforms its own baseline, the
cause is almost never the algorithm — it is one silently-garbage intermediate that nothing logged."*
The campaign has one `degenerate: true` verdict (`anas_platyrhynchos`, draft) and no way to see what
Qwen actually said, because `_stash` persists `{ok, degenerate}` and drops `issues` and `raw`.

This is fixed by wiring tested code, not by writing new code.

## 4. Mechanism

### 4.1 One serialiser — `ragregen/verify/record.py`

```python
def to_dict(score: ConceptScore, *, with_state: bool) -> dict
def from_dict(d: dict | str) -> ConceptScore
```

`to_dict` emits every field of `ConceptScore` including `candidates`. `with_state` controls one key.

- **`score_a.py` passes `with_state=False`.** Its output stays byte-for-byte what it is today. This
  is not politeness: `c1.py` consumes `stream_a.json` and the pinned τ = 0.25 baseline must keep
  reproducing, which §7 pins by test.
- **`run_pipeline.py` passes `with_state=True`.** The scheduler resumes on verdicts.

`from_dict` accepts **either** a dict or a bare state string. Existing run directories on disk carry
the old shape, and `--resume` is the guarantee doc 3 exists to make; a resumed campaign must not die
on its own history. A bare string rehydrates to exactly today's lossy `ConceptScore`, which is
honest — that run really does hold no numbers.

### 4.2 A score that can be absent — `fusion.py`

```python
sims  = [s.sim for s in scores.values() if s.state != "ABSTAIN" and s.sim is not None]
score = float(min(sims)) if sims else None
```

`Verdict.score` becomes `float | None`. `None` means *Stream A produced no usable evidence* and is
written to `scores.json` as JSON `null`.

`schedule.select_best` skips `None` candidates in its fallback, so a candidate with no evidence can
never displace the draft:

```python
for label, _, score in attempts:
    if score is not None and score > best_score:
        best_label, best_score = label, score
```

The first-passing-attempt rule is untouched — it reads `ok`, never `score`. Rare in practice: 201 of
the campaign's 202 verdicts carry at least one non-ABSTAIN concept, so the fallback fires almost
never. That is the argument for `None` rather than a blend — the degenerate branch is an edge case
and should read as one, not as a perfect score.

**`ok` and `score` still come from different streams and can disagree.** A case can be `ok=False`
with a high `min(sim)`. That is the design (`ok` is the OR of both streams, `score` is Stream A's
weakest concept), it is what C1 measured, and changing it is a new mechanism needing its own
pre-registration. Out of scope here; recorded so the next reader does not mistake it for an
oversight.

### 4.3 The trace is written — `run_pipeline.py`

One `CaseTrace` per case per run, saved to `<case>/trace.json`:

- `trace.vlm(f"semantic@{attempt}", raw)` — the raw Qwen reply, with `garbage` flagged
- `trace.grounded(...)` — the per-concept scores at each attempt
- `trace.hits(...)` — the retrieval hits, at the one `retrieve` stage

Stream B's `_stash` payload also gains `issues` and `raw`, so `Verdict.target` becomes recoverable
from `semantic.issues[0].concept`.

### 4.4 Prototypes become reachable — two independent flags, both default off

| flag | effect | changes behaviour? |
|---|---|---|
| `--proto-arm {none,ceiling,retrieved}` | builds a `PrototypeBank` and passes it to `GroundedVerifier` | **no** — `grounded.py:87` records prototype scores and never acts on them, and that invariant stays |
| `--mask-prototypes` | passes the same bank to `mask_draft` as `prototypes=` | **yes** — mask geometry changes |

Both default off, so a resumed campaign behaves identically. They are separate flags because they
are separate claims: the first only persists numbers for offline grading, the second changes which
region gets repainted. `bank_from_retrieval` already builds fine **and** coarse prototypes from the
corpus and never reads `gt_refs` (`prototype.py:156-172`), so the `retrieved` arm needs no new data.

**The mask stage persists `selected_by` and `n_candidates` to `<case>/mask.json` unconditionally.**
Doc 1 §6 called this the diagnostic that makes a full-arm masking failure separable from a
generation failure; the pipeline never wrote it, so the campaign cannot answer which rule chose any
of its 46 masks.

### 4.5 `scripts/reverify.py` — the re-grade

```
reverify --run outputs/doc5_20260810_183939 --dataset configs/dataset_common.yaml
         [--proto-arm retrieved] [--tag reverify]
```

Walks a finished run, re-runs **only** the two verify stages over the images already on disk —
`screen_latest/<case>/draft.png` for attempt 0, `<case>/attempt_N.png` for the rest — and writes a
**new** run directory. Stage-batched exactly as the pipeline is: one grounded residency, one
semantic residency, over all 202 images.

**The source run is never written to.** Per the operator's standing rule of 2026-08-10 — when a
run's provenance is compromised, re-run cleanly rather than hand-edit metadata — `doc5_20260810_183939`
is evidence and stays untouched. A test asserts the source directory's mtimes are unchanged.

The new directory holds fresh `streams.json`, `scores.json` and a `queue.json` carrying the
re-selected `best` and `status` per case; it symlinks the pixel artifacts (`mask.png`,
`attempt_*.png`, `cutout_*.png`) and records `source_run` in `run.json`. That is exactly the shape
`scripts/report.py` already walks, so the report runs against it unchanged. Symlinks rather than
copies because the pixels are the one thing this run did not produce, and claiming authorship of
them is the provenance error we are correcting.

Cost, from the measured rates in `findings/2026-07-27-c1-result.md` (Stream A ~4 min / 22 images,
Stream B ~13 min / 22 at nf4): **~40 min + ~2 h** over 202 images. No FLUX, no drafting, no
generation.

### 4.6 The C1 prototype grading

`score_a.py --proto-arm retrieved --labels outputs/screen_20260726_233320/labels.csv --force`, then
`run.sh c1`. `c1.render_prototype` and the band machinery are already written and tested; this is a
run, not an implementation.

The 22 hand-labelled drafts survive with `labels.csv` fully populated (`verdict_identity` and
notes) — `findings/2026-07-28-masking-result.md` §1 recorded them as lost from disk and they are
not. **The pre-registration of 2026-07-28 §7 stands unchanged and is graded as written**: τ pinned
at 0.25, `delta_proto` over the 81-point grid, graded target the 6 cases the τ = 0.25 baseline
missed, RULE MET requiring ≥4 of 6 newly caught with no *new* false positives over a band of ≥3
consecutive δ, box selection by **detector confidence**.

`ceiling` is not run. It needs `coarse_refs` for 18 terms authored as a spread across each category
(§6 P2 makes that an operator responsibility code cannot check), and design §6 forbids quoting it as
a verifier result. It becomes the follow-up **only if `retrieved` fails**, where it separates "the
mechanism is dead" from "retrieval fed it bad references".

## 5. Also fixed, because they are in the blast radius

- **`status: "unrepaired"` means two opposite things** (`findings/2026-07-29-orchestration-result.md`
  §2). `_resolve` writes it when the draft *passed*; `_finalise` when repair was tried and failed.
  A third value `healthy` is added and `queue.json`'s `schema` bumps to 2. **`Queue.open` still
  reads a schema-1 file** and leaves its statuses as written — the doc-5 queue must stay resumable,
  which is the same back-compat rule §4.1 applies to `streams.json`. `report.py` already partitions
  on `scores.json`'s draft verdict and is unaffected — this is for whoever reads the queue.
- **Resolved cases are re-fused on resume** (§3 wart, same finding). `_verify_round` skips any case
  whose status is already resolved, so finished work stops being rewritten.
- **`configs/retrieval_db.yaml` mixes an absolute `images_root` with a relative `index_path`**
  (ledger MINOR 13), and both `validate_all` and `report.main` derive the manifest from
  `index_path.parent`, so both silently depend on CWD. `reverify.py` is a new entry point that will
  be invoked directly, so this one stops being theoretical: paths resolve against the repo root.
- **RUNBOOK §1 documents a step that does not exist.** `ragregen/query.py` was never built and
  `_retrieve_one` searches `case.concept` verbatim (deliberately deferred, doc 3 §9). The operator
  doc is corrected to say so.

## 6. Out of scope

- **Blending the two streams into one score** (§4.2). New mechanism, needs pre-registration.
- **`ragregen/query.py`.** Still deferred; only the documentation lie is fixed.
- **Re-running generation.** No FLUX. The corrected numbers come from re-verifying images that exist.
- **The `oracle` arm and the ceiling prototype arm.** Both deferred, both gated on results.
- **The remaining parked ledger minors** (11, 12, 14, 15). None is in this blast radius.

## 7. Testing

Dependency injection throughout, fakes for every model, no GPU in the default run. The **520
existing tests stay green**.

- **Round trip:** `from_dict(to_dict(s, with_state=True)) == s` for a score carrying candidates,
  prototypes and `None`s. Mutation-tested — dropping any field must fail it.
- **Back-compat:** `from_dict("MISSING")` yields today's lossy score, so an old `streams.json`
  resumes.
- **`score_a.py` output is unchanged**, asserted against a golden fixture — the τ = 0.25 C1 baseline
  must keep reproducing bit for bit. This is the regression that matters most.
- **`fuse` returns `None`** when every concept abstains, and `min(sims)` otherwise.
- **`select_best` skips `None`** — a `None`-scored failing attempt never displaces the draft. Built
  as the `anas_platyrhynchos` shape: draft `[false, 0.0]`, attempt `[false, None]`.
- **The trace carries raw VLM text** and flags a garbage reply.
- **Prototype flags default off**: a run with neither flag constructs no bank, and `mask_draft`
  receives `prototypes=None` — asserted with a spy, because "off by default" is what keeps a resumed
  campaign comparable.
- **`reverify` never writes the source run** — source mtimes unchanged, asserted on a fixture dir.
- **`reverify` re-selects** — a fixture where the old selection took a failing attempt yields the
  draft under the corrected rule.
- **Status:** a case resolved at round 0 is `healthy`; one that tried and failed is `unrepaired`.

GPU paths are verified by the re-verify run itself, not by unit tests.

## 8. Risks

**V1 — Stream B may not reproduce.** The re-verify re-runs Qwen at nf4 on possibly a different card.
Greedy decoding should be deterministic, but quantised inference across hardware is not guaranteed
to be. The finding must report the campaign's Stream B verdicts beside the re-verified ones and name
any case that moved, rather than silently adopting the new ones.

**V2 — the corrected selection will move the published numbers.** That is the point, and the
direction is not predictable: three cases lose a wrongly-promoted attempt, and every case gains a
real continuous score that may reorder the fallback. The doc-5 report's +0.047 target/bridge delta
may change. Stated in advance so the result cannot be reinterpreted afterwards.

**V3 — the prototype rule is graded in-sample on 6 cases**, δ selected on the same data, no held-out
split. Unchanged from the 2026-07-28 pre-registration and its `P1`/`P4`; the bar stays 4 of 6.

**V4 — masks were cut by confidence.** `--mask-prototypes` cannot retroactively fix the 46 masks the
campaign produced, because changing geometry means regenerating, which §6 excludes. The flag serves
future runs; the re-grade inherits the masks it has, and `boston_bull`-shaped selection errors stay
in the data.

**V5 — a contended card.** Two verify residencies, ~2.5 h. `gpu_wait.sh` and `supervise.sh` already
exist and `reverify` exits 2 on `StageAborted` like every other stage, so the supervisor drives it
unchanged.

## 9. Deliverable

1. The fixes above, on a branch cut after the 2026-08-10 working-tree fixes are committed.
2. A re-verified run directory and a corrected `report.md`.
3. The C1 prototype section, graded against the pre-registered rule.
4. `docs/findings/2026-08-12-verifier-persistence-result.md` — what the corrected selection did to
   the doc-5 numbers, whether the prototype rule was MET, and the Stream B reproduction check from
   V1. The existing report and findings are **not edited**; they stand as what was true when written.
