# Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the doc 1 + doc 2 machinery over a whole dataset unattended, and resume correctly when another researcher takes the GPU mid-run.

**Architecture:** `ragregen/schedule.py` is a pure state machine — no torch, no models, no I/O beyond `queue.json` — so every scheduling rule is testable on CPU in milliseconds. `scripts/run_pipeline.py` owns model residency and stage order, handing per-case callables to `schedule.run_stage`. The split is what makes "resume skips the model load" assertable with a spy instead of a GPU.

**Tech Stack:** Python 3.11, stdlib `json`/`os`/`dataclasses`, pytest. No new dependencies.

## Global Constraints

- `$PY=/mnt/mmlab2024nas/ldtuan/miniconda3/envs/kontext/bin/python`. Bare `python` is conda 3.13 and fails on faiss.
- **`schedule.py` imports no models and no torch.** `is_fatal` classifies exceptions by name and message, never by `isinstance`. This is what keeps the whole scheduler CPU-testable.
- **Writes are atomic**: `.tmp` + `os.replace`, after every case. A SIGKILL mid-write must not strand the run.
- **`retrieve` and `mask` are attempt-free.** They run once per case, never per round (design §4).
- **Stage keys are `<stage>@<attempt>`** for `grounded`, `semantic`, `regen`; bare stage names for `retrieve` and `mask`. Attempt 0 is the draft.
- **OOM aborts the stage; anything else isolates the case** (design §6). Never the reverse.
- **The draft is always a candidate** in best-of selection (`RUNBOOK.md` §1).
- `screen`, `score-a`, `score-b`, `c1` are untouched. They are validated C1 tooling.
- The 324 existing tests must stay green.
- `git add <exact paths>` — never `git add -A`. `outputs/` is gitignored.
- **GPU etiquette:** `nvidia-smi` before anything that loads FLUX or Qwen3-VL. Never lower `regen.MIN_VRAM_GB` to make a run fit.

---

### Task 1: `CaseState` and `Queue` round-trip

**Files:**
- Create: `ragregen/schedule.py`
- Test: `tests/test_schedule.py`

**Interfaces:**
- Consumes: nothing
- Produces: `CaseState`, `Queue.open(path, case_ids, *, retry_budget, screen_run) -> Queue`, `Queue.save() -> None`

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path

from ragregen.schedule import CaseState, Queue


def test_open_creates_a_queue_when_none_exists(tmp_path):
    q = Queue.open(tmp_path / "queue.json", ["a", "b"],
                   retry_budget=3, screen_run="outputs/screen_x")

    assert sorted(q.cases) == ["a", "b"]
    assert q.cases["a"] == CaseState(case_id="a")
    assert q.retry_budget == 3


def test_save_then_open_round_trips(tmp_path):
    path = tmp_path / "queue.json"
    q = Queue.open(path, ["a"], retry_budget=3, screen_run="outputs/screen_x")
    q.cases["a"].stages["grounded@0"] = "done"
    q.cases["a"].attempt = 2
    q.cases["a"].status = "repaired"
    q.cases["a"].best = "attempt_2"
    q.save()

    #: case_ids is deliberately WRONG on reopen. A resumed run must trust the
    #: file, not the argument, or a --limit on the resume would silently drop
    #: cases the first run had already finished.
    again = Queue.open(path, ["totally", "different"],
                       retry_budget=99, screen_run="outputs/other")

    assert again.cases["a"].stages == {"grounded@0": "done"}
    assert again.cases["a"].attempt == 2
    assert again.cases["a"].best == "attempt_2"
    assert again.retry_budget == 3
    assert again.screen_run == "outputs/screen_x"
    assert sorted(again.cases) == ["a"]


def test_saved_file_is_readable_json_with_a_schema(tmp_path):
    path = tmp_path / "queue.json"
    Queue.open(path, ["a"], retry_budget=3, screen_run="s").save()

    data = json.loads(path.read_text())
    assert data["schema"] == 1
    assert data["cases"]["a"]["status"] == "pending"
    #: case_id is the dict key; storing it twice invites the two to disagree.
    assert "case_id" not in data["cases"]["a"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_schedule.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragregen.schedule'`

- [ ] **Step 3: Implement**

Create `ragregen/schedule.py`:

```python
"""The scheduler's state, and nothing else.

No torch, no models, no I/O beyond queue.json. That constraint is load-bearing:
it is what lets every scheduling rule below be tested on CPU in milliseconds,
on a machine whose GPU is routinely held by someone else for days
(findings/2026-07-28-regeneration-result.md §4).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: One entry per MODEL RESIDENCY, not per logical step. "verify" is two
#: separate loads -- DINO+SigLIP, then Qwen3-VL -- and a run killed between
#: them must resume at the second rather than redo both.
STAGES = ("grounded", "semantic", "retrieve", "mask", "regen")

#: Stages that run once per case rather than once per attempt (design §4).
ATTEMPT_FREE = ("retrieve", "mask")

SCHEMA = 1


@dataclass
class CaseState:
    case_id: str
    stages: dict[str, str] = field(default_factory=dict)
    attempt: int = 0
    status: str = "pending"          # pending|repaired|unrepaired|failed
    best: str | None = None          # "draft" | "attempt_2"
    error: str | None = None


def stage_key(stage: str, attempt: int | None = None) -> str:
    if stage in ATTEMPT_FREE or attempt is None:
        return stage
    return f"{stage}@{attempt}"


class Queue:
    def __init__(self, path: Path, cases: dict[str, CaseState],
                 retry_budget: int, screen_run: str):
        self.path = Path(path)
        self.cases = cases
        self.retry_budget = retry_budget
        self.screen_run = screen_run

    @classmethod
    def open(cls, path: Path, case_ids: list[str], *, retry_budget: int,
             screen_run: str) -> "Queue":
        """Load queue.json if it exists, else start one. Resume is this call.

        An existing file wins over every argument. A resumed run that honoured
        a fresh --limit would drop cases the first run had already paid for.
        """
        path = Path(path)
        if path.is_file():
            data = json.loads(path.read_text())
            cases = {cid: CaseState(case_id=cid, **st)
                     for cid, st in data["cases"].items()}
            return cls(path, cases, int(data["retry_budget"]),
                       str(data["screen_run"]))
        return cls(path, {cid: CaseState(case_id=cid) for cid in case_ids},
                   retry_budget, screen_run)

    def save(self) -> None:
        """Atomic. A half-written queue is worse than no queue at all."""
        payload = {
            "schema": SCHEMA,
            "retry_budget": self.retry_budget,
            "screen_run": self.screen_run,
            "cases": {cid: {k: v for k, v in asdict(s).items()
                            if k != "case_id"}
                      for cid, s in self.cases.items()},
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        os.replace(tmp, self.path)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_schedule.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ragregen/schedule.py tests/test_schedule.py
git commit -m "feat: the scheduler's state, serialised atomically"
```

---

### Task 2: `pending` and `mark`

**Files:**
- Modify: `ragregen/schedule.py`
- Test: `tests/test_schedule.py`

**Interfaces:**
- Consumes: `Queue`, `CaseState`, `stage_key` (Task 1)
- Produces: `Queue.pending(stage, *, attempt=None) -> list[str]`, `Queue.mark(case_id, stage, status, *, attempt=None, **fields) -> None`

- [ ] **Step 1: Write the failing test**

```python
def test_pending_returns_cases_missing_this_stage(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a", "b"], retry_budget=3,
                   screen_run="s")
    q.mark("a", "grounded", "done", attempt=0)

    assert q.pending("grounded", attempt=0) == ["b"]
    #: A different attempt is a different cell.
    assert sorted(q.pending("grounded", attempt=1)) == ["a", "b"]


def test_pending_skips_cases_that_are_no_longer_pending(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a", "b"], retry_budget=3,
                   screen_run="s")
    q.mark("a", "grounded", "done", attempt=0, status="repaired")

    #: 'a' is finished. It must never be handed to another stage, whatever
    #: cells its stages dict is missing.
    assert q.pending("semantic", attempt=0) == ["b"]


def test_attempt_free_stages_ignore_the_attempt(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a"], retry_budget=3, screen_run="s")
    q.mark("a", "mask", "done")

    assert q.pending("mask", attempt=2) == []


def test_mark_persists_immediately(tmp_path):
    path = tmp_path / "q.json"
    q = Queue.open(path, ["a"], retry_budget=3, screen_run="s")
    q.mark("a", "regen", "done", attempt=1)

    #: Not q.save() -- mark checkpoints on its own, because a stage killed at
    #: case 20 of 30 must keep the first 19.
    assert Queue.open(path, [], retry_budget=3,
                      screen_run="s").cases["a"].stages == {"regen@1": "done"}


def test_mark_applies_extra_fields(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a"], retry_budget=3, screen_run="s")
    q.mark("a", "semantic", "failed", attempt=1, status="failed",
           error="boom", best="draft")

    s = q.cases["a"]
    assert (s.status, s.error, s.best) == ("failed", "boom", "draft")
    assert s.stages == {"semantic@1": "failed"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_schedule.py -v`
Expected: FAIL — `AttributeError: 'Queue' object has no attribute 'pending'`

- [ ] **Step 3: Implement**

Add to `class Queue` in `ragregen/schedule.py`:

```python
    def pending(self, stage: str, *, attempt: int | None = None) -> list[str]:
        """Cases still owed this stage, in insertion order.

        A case whose status has left "pending" is finished -- repaired, or
        failed -- and is never handed to another stage regardless of which
        cells its stages dict is missing.
        """
        key = stage_key(stage, attempt)
        return [cid for cid, s in self.cases.items()
                if s.status == "pending" and key not in s.stages]

    def mark(self, case_id: str, stage: str, status: str, *,
             attempt: int | None = None, **fields) -> None:
        """Record a stage outcome and checkpoint.

        Saves on every call rather than at stage boundaries: the failure being
        defended against is a SIGKILL partway through a 30-case stage.
        """
        state = self.cases[case_id]
        state.stages[stage_key(stage, attempt)] = status
        for key, value in fields.items():
            if not hasattr(state, key):
                raise AttributeError(f"CaseState has no field {key!r}")
            setattr(state, key, value)
        self.save()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_schedule.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add ragregen/schedule.py tests/test_schedule.py
git commit -m "feat: pending cells and marking, checkpointed per case"
```

---

### Task 3: Crash safety, proved with a real subprocess

**Files:**
- Test: `tests/test_schedule_crash.py`

**Interfaces:**
- Consumes: `Queue` (Tasks 1–2)
- Produces: no new API — this task only proves the existing one

- [ ] **Step 1: Write the failing test**

Create `tests/test_schedule_crash.py`:

```python
"""Crash safety, proved by killing a real process.

A test that calls save() then open() in one process proves serialisation and
nothing else. Design §R3: if the suite cannot survive a SIGKILL, the resume
guarantee is decorative.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

WRITER = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, {root!r})
    from ragregen.schedule import Queue

    q = Queue.open({path!r}, ["c0","c1","c2","c3","c4"], retry_budget=3,
                   screen_run="s")
    for i in range(5):
        q.mark(f"c{{i}}", "regen", "done", attempt=1)
        print(i, flush=True)
        time.sleep(0.4)
""")


def test_sigkill_midstage_keeps_completed_cases(tmp_path):
    root = str(Path(__file__).resolve().parent.parent)
    path = tmp_path / "q.json"
    script = tmp_path / "writer.py"
    script.write_text(WRITER.format(root=root, path=str(path)))

    proc = subprocess.Popen([sys.executable, str(script)],
                            stdout=subprocess.PIPE, text=True)
    #: Wait for case 2 to report done, then kill hard -- no cleanup, no
    #: atexit, no flush. This is what a stolen GPU looks like to the process.
    for _ in range(3):
        proc.stdout.readline()
    proc.kill()
    proc.wait(timeout=10)

    data = json.loads(path.read_text())          # must parse at all
    done = {c for c, s in data["cases"].items()
            if s["stages"].get("regen@1") == "done"}
    assert {"c0", "c1", "c2"} <= done
    assert "c4" not in done


def test_no_tmp_file_is_left_behind(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a"], retry_budget=3, screen_run="s")
    q.save()

    assert [p.name for p in tmp_path.iterdir()] == ["q.json"]
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_schedule_crash.py -v`
Expected: 2 passed — the atomic write from Task 1 already satisfies this. If either fails, `save()` is not atomic and Task 1 must be fixed before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_schedule_crash.py
git commit -m "test: prove resume survives SIGKILL, not just serialisation"
```

---

### Task 4: `is_fatal`, `StageAborted` and `run_stage`

**Files:**
- Modify: `ragregen/schedule.py`
- Test: `tests/test_schedule.py`

**Interfaces:**
- Consumes: `Queue` (Tasks 1–2)
- Produces: `StageAborted`, `is_fatal(exc) -> bool`, `run_stage(queue, stage, fn, *, attempt=None) -> None`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from ragregen.schedule import Queue, StageAborted, is_fatal, run_stage


class FakeOOM(Exception):
    """Stands in for torch.OutOfMemoryError, which we refuse to import."""


FakeOOM.__name__ = "OutOfMemoryError"


def test_is_fatal_recognises_oom_without_importing_torch():
    assert is_fatal(FakeOOM("CUDA out of memory. Tried to allocate 26.00 MiB"))
    assert is_fatal(RuntimeError("CUDA error: device-side assert triggered"))
    assert not is_fatal(ValueError("no box found for 'durian'"))


def test_run_stage_marks_every_case_done(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a", "b"], retry_budget=3,
                   screen_run="s")
    seen = []
    run_stage(q, "grounded", seen.append, attempt=0)

    assert seen == ["a", "b"]
    assert q.cases["a"].stages == {"grounded@0": "done"}


def test_oom_aborts_the_stage_and_spares_the_rest(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a", "b", "c"], retry_budget=3,
                   screen_run="s")
    seen = []

    def fn(cid):
        seen.append(cid)
        if cid == "b":
            raise FakeOOM("CUDA out of memory")

    with pytest.raises(StageAborted):
        run_stage(q, "regen", fn, attempt=1)

    #: 'c' is never attempted -- every remaining case would fail identically,
    #: and discovering that 28 more times costs an hour.
    assert seen == ["a", "b"]
    assert q.cases["c"].status == "pending"
    #: 'b' is NOT marked failed. The card was the problem, not the case, so
    #: resume must retry it.
    assert "regen@1" not in q.cases["b"].stages
    assert q.cases["b"].status == "pending"


def test_other_exceptions_isolate_the_case(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a", "b", "c"], retry_budget=3,
                   screen_run="s")
    seen = []

    def fn(cid):
        seen.append(cid)
        if cid == "b":
            raise ValueError("no box found")

    run_stage(q, "mask", fn)

    assert seen == ["a", "b", "c"]
    assert q.cases["b"].status == "failed"
    assert q.cases["b"].stages["mask"] == "failed"
    assert "no box found" in q.cases["b"].error
    assert q.cases["c"].stages["mask"] == "done"


def test_run_stage_skips_cells_already_done(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a", "b"], retry_budget=3,
                   screen_run="s")
    q.mark("a", "grounded", "done", attempt=0)
    seen = []
    run_stage(q, "grounded", seen.append, attempt=0)

    assert seen == ["b"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_schedule.py -v`
Expected: FAIL — `ImportError: cannot import name 'StageAborted'`

- [ ] **Step 3: Implement**

Add to `ragregen/schedule.py`:

```python
import traceback


class StageAborted(RuntimeError):
    """The card is gone. Checkpoint and stop; resume will retry these cases."""


#: Matched by NAME and MESSAGE, never by isinstance. Importing torch here would
#: cost ~3s and a CUDA context in every unit test, and would make the scheduler
#: untestable on a machine without the library.
_FATAL_NAMES = ("OutOfMemoryError", "CudaError", "CUDAOutOfMemoryError")
_FATAL_TEXT = ("out of memory", "cuda error", "cuda out of memory",
               "no cuda-capable device")


def is_fatal(exc: BaseException) -> bool:
    """True when the exception means the CARD failed, not the case.

    An OOM tells you nothing about the case that hit it -- the next 29 would
    fail the same way. A ValueError from a concept that will not ground is a
    fact about that one case.
    """
    if type(exc).__name__ in _FATAL_NAMES:
        return True
    text = str(exc).lower()
    return any(t in text for t in _FATAL_TEXT)


def run_stage(queue: Queue, stage: str, fn, *,
              attempt: int | None = None) -> None:
    """Apply fn to every case pending this stage, isolating per-case failures.

    The single place the failure policy lives, so no stage handler can get it
    wrong by forgetting a try block.
    """
    for case_id in queue.pending(stage, attempt=attempt):
        try:
            fn(case_id)
        except BaseException as exc:
            if is_fatal(exc):
                # Deliberately NOT marked failed: the case never got a fair
                # attempt, and resume must pick it up untouched.
                queue.save()
                raise StageAborted(
                    f"{stage} aborted on '{case_id}': {exc}. "
                    f"The card is held by another job -- `nvidia-smi` shows "
                    f"who. Re-run the identical command to resume."
                ) from exc
            queue.mark(case_id, stage, "failed", attempt=attempt,
                       status="failed",
                       error="".join(traceback.format_exception_only(
                           type(exc), exc)).strip())
            continue
        queue.mark(case_id, stage, "done", attempt=attempt)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_schedule.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add ragregen/schedule.py tests/test_schedule.py
git commit -m "feat: an OOM kills the stage, a bad case kills only itself"
```

---

### Task 5: Rounds and best-of selection

**Files:**
- Modify: `ragregen/schedule.py`
- Test: `tests/test_schedule.py`

**Interfaces:**
- Consumes: `Queue` (Tasks 1–2)
- Produces: `Queue.unresolved() -> list[str]`, `Queue.needs_round(attempt) -> bool`, `select_best(draft_score, attempts) -> str`

- [ ] **Step 1: Write the failing test**

```python
from ragregen.schedule import select_best


def test_needs_round_is_false_when_nothing_is_unresolved(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a"], retry_budget=3, screen_run="s")
    q.mark("a", "semantic", "done", attempt=1, status="repaired")

    assert q.unresolved() == []
    assert not q.needs_round(2)


def test_needs_round_respects_the_budget(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a"], retry_budget=3, screen_run="s")

    assert q.needs_round(3)
    #: Attempt 4 is past the budget however many cases still fail.
    assert not q.needs_round(4)


def test_first_passing_attempt_wins():
    #: (label, ok, score) in attempt order.
    assert select_best(0.9, [("attempt_1", False, 0.95),
                             ("attempt_2", True, 0.10)]) == "attempt_2"


def test_the_draft_wins_when_no_attempt_passes():
    #: This is the guarantee that the method cannot score worse than the
    #: no-retrieval baseline (RUNBOOK §1).
    assert select_best(0.80, [("attempt_1", False, 0.10),
                              ("attempt_2", False, 0.79)]) == "draft"


def test_highest_scoring_attempt_wins_when_none_pass():
    assert select_best(0.20, [("attempt_1", False, 0.30),
                              ("attempt_2", False, 0.55)]) == "attempt_2"


def test_no_attempts_at_all_selects_the_draft():
    assert select_best(0.42, []) == "draft"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_schedule.py -v`
Expected: FAIL — `ImportError: cannot import name 'select_best'`

- [ ] **Step 3: Implement**

Add to `class Queue`:

```python
    def unresolved(self) -> list[str]:
        return [cid for cid, s in self.cases.items() if s.status == "pending"]

    def needs_round(self, attempt: int) -> bool:
        """Whether round `attempt` has any work. Checked BEFORE loading FLUX."""
        return attempt <= self.retry_budget and bool(self.unresolved())
```

And at module level:

```python
def select_best(draft_score: float, attempts) -> str:
    """The candidate to report, from {draft, attempt_1 .. attempt_N}.

    `attempts` is (label, ok, score) in attempt order.

    The first PASSING attempt wins outright -- a later attempt scoring higher
    while still failing is not better, it is differently wrong. When nothing
    passes it falls back to the highest score, with the draft always in the
    running. That last clause is the RUNBOOK §1 guarantee: the method cannot
    score worse than the no-retrieval baseline on any case.

    Selection uses the VERIFIER's score, never the identity or preservation
    metrics -- those are doc 4's, and selecting on them would be selecting on
    the thing being measured.
    """
    for label, ok, _ in attempts:
        if ok:
            return label

    best_label, best_score = "draft", draft_score
    for label, _, score in attempts:
        if score > best_score:
            best_label, best_score = label, score
    return best_label
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_schedule.py -v`
Expected: 19 passed

- [ ] **Step 5: Run the suite**

Run: `$PY -m pytest -q`
Expected: 345 passed, 11 deselected

- [ ] **Step 6: Commit**

```bash
git add ragregen/schedule.py tests/test_schedule.py
git commit -m "feat: rounds stop early, and the draft is always a candidate"
```

---

### Task 6: The driver skips loads it does not need

**Files:**
- Create: `scripts/run_pipeline.py`
- Test: `tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: everything in `ragregen/schedule.py`
- Produces: `run_pipeline.stage_with_model(queue, stage, load, work, *, attempt=None) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_pipeline.py`:

```python
"""The one driver rule worth testing without a GPU.

Everything else in run_pipeline.py is model wiring. This is the rule that
decides whether a resumed run takes seconds or reloads FLUX for 5m21s to
discover it has nothing to do.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen.schedule import Queue          # noqa: E402
from scripts.run_pipeline import stage_with_model  # noqa: E402


def test_the_model_is_not_loaded_when_no_case_needs_it(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a"], retry_budget=3, screen_run="s")
    q.mark("a", "regen", "done", attempt=1)
    loads = []

    stage_with_model(q, "regen", lambda: loads.append("loaded"),
                     lambda model, cid: None, attempt=1)

    assert loads == []


def test_the_model_is_loaded_once_for_many_cases(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a", "b", "c"], retry_budget=3,
                   screen_run="s")
    loads, worked = [], []

    stage_with_model(q, "regen", lambda: loads.append("m") or "MODEL",
                     lambda model, cid: worked.append((model, cid)),
                     attempt=1)

    assert loads == ["m"]
    assert worked == [("MODEL", "a"), ("MODEL", "b"), ("MODEL", "c")]


def test_the_model_is_freed_even_when_the_stage_aborts(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a"], retry_budget=3, screen_run="s")
    freed = []

    class Model:
        def free(self):
            freed.append(True)

    def work(model, cid):
        raise RuntimeError("CUDA out of memory")

    try:
        stage_with_model(q, "regen", Model, work, attempt=1)
    except Exception:
        pass

    assert freed == [True]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_run_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_pipeline'`

- [ ] **Step 3: Implement**

Create `scripts/run_pipeline.py` with the helper first (the CLI arrives in Task 7):

```python
#!/usr/bin/env python
"""The stage-batched scheduler driver.

Owns model residency and stage order. Every scheduling RULE lives in
ragregen/schedule.py, which imports no torch -- this file is the only place
that knows FLUX exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen import schedule  # noqa: E402


def stage_with_model(queue, stage: str, load, work, *,
                     attempt: int | None = None) -> None:
    """Load a model only if some case needs it, run the stage, always free it.

    The guard is the difference between a resumed run that takes seconds and
    one that spends 5m21s loading FLUX to discover every case is done.
    """
    if not queue.pending(stage, attempt=attempt):
        return

    model = load()
    try:
        schedule.run_stage(queue, stage, lambda cid: work(model, cid),
                           attempt=attempt)
    finally:
        free = getattr(model, "free", None)
        if callable(free):
            free()
```

Add `tests/__init__.py` compatibility by ensuring `scripts/__init__.py` exists (it already does — `scripts/__init__.py` is an empty file in the repo).

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_run_pipeline.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/run_pipeline.py tests/test_run_pipeline.py
git commit -m "feat: never load a model no case is waiting for"
```

---

### Task 7: The CLI and the stage order

**Files:**
- Modify: `scripts/run_pipeline.py`
- Test: `tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: `stage_with_model` (Task 6), `config.load_dataset`, `config.load_pipeline`, `trace.open_run`, `mask.mask_draft`, `mask.mask_reference`, `composite.cutout_from`, `regen.stitch`, `regen.Inpainter`, `regen.KontextConfig`, `retrieve.Retriever`, `verify.grounded.GroundedVerifier`, `verify.semantic.SemanticVerifier`, `verify.fusion.fuse`
- Produces: `main() -> int`, `resolve_screen_run(path) -> Path`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from scripts.run_pipeline import resolve_screen_run


def test_a_missing_screen_run_is_an_error_not_a_redraft(tmp_path):
    #: The pipeline must NEVER helpfully draft its own images: those would be
    #: a different seed from the set that was hand-labelled, so the labels
    #: would no longer describe the images being repaired.
    with pytest.raises(FileNotFoundError, match="screen"):
        resolve_screen_run(tmp_path / "nope")


def test_the_screen_run_must_contain_drafts(tmp_path):
    empty = tmp_path / "screen_latest"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no drafts"):
        resolve_screen_run(empty)


def test_a_screen_run_with_drafts_resolves(tmp_path):
    run = tmp_path / "screen_latest"
    (run / "boston_bull").mkdir(parents=True)
    (run / "boston_bull" / "draft.png").write_bytes(b"x")

    assert resolve_screen_run(run) == run
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_run_pipeline.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_screen_run'`

- [ ] **Step 3: Implement**

Add to `scripts/run_pipeline.py`:

```python
import argparse
import json

from ragregen import composite, config, env, mask, models, regen, trace  # noqa: E402
from ragregen.verify import fusion, grounded, semantic  # noqa: E402


def resolve_screen_run(path: Path) -> Path:
    """The drafts to repair. Never generated here.

    `screen` is the gate (RUNBOOK §4) and its drafts are the ones that were
    hand-labelled. Drafting fresh ones at a different seed would mean the
    labels no longer describe the images being repaired -- so a missing screen
    run is an error, not a prompt to be helpful.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(
            f"no screen run at {path}. Run `./scripts/run.sh screen` first -- "
            f"it is the gate everything downstream reads.")
    if not list(path.glob("*/draft.png")):
        raise FileNotFoundError(
            f"{path} has no drafts (expected <case>/draft.png).")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pipeline")
    ap.add_argument("--resume", type=Path, default=None,
                    help="an existing outputs/pipeline_<TS>/ to continue")
    ap.add_argument("--arm", choices=("oracle", "full"), default="oracle")
    ap.add_argument("--mechanism", choices=regen.MECHANISMS, default="inpaint")
    ap.add_argument("--screen-run", type=Path,
                    default=Path("outputs/screen_latest"))
    ap.add_argument("--limit", type=int, default=0, help="0 = every case")
    ap.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET_PATH)
    ap.add_argument("--pipeline", type=Path, default=config.DEFAULT_PIPELINE_PATH)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds = config.load_dataset(args.dataset)
    pipe_cfg = config.load_pipeline(args.pipeline)
    screen_run = resolve_screen_run(args.screen_run)

    cases = list(ds.cases)[:args.limit] if args.limit else list(ds.cases)
    by_id = {c.id: c for c in cases}

    if args.resume:
        run_dir = trace.RunDir(path=Path(args.resume), argv=sys.argv, args=vars(args))
    else:
        run_dir = trace.open_run(args.tag, argv=sys.argv, args=vars(args))

    queue = schedule.Queue.open(
        run_dir.path / "queue.json", [c.id for c in cases],
        retry_budget=pipe_cfg.retry_budget, screen_run=str(screen_run))

    print(f"[pipeline] {len(queue.cases)} cases -> {run_dir.path}  "
          f"arm={args.arm} mechanism={args.mechanism}", flush=True)

    try:
        _run(queue, run_dir, by_id, pipe_cfg, screen_run, args)
    except schedule.StageAborted as exc:
        print(f"[pipeline] {exc}")
        run_dir.finish("aborted", {"unresolved": queue.unresolved()})
        print(f"[pipeline] resume with: ./scripts/run.sh pipeline "
              f"--resume {run_dir.path}")
        return 2

    run_dir.finish("ok", {
        "repaired": [c for c, s in queue.cases.items() if s.status == "repaired"],
        "unrepaired": [c for c, s in queue.cases.items() if s.status == "unrepaired"],
        "failed": [c for c, s in queue.cases.items() if s.status == "failed"],
    })
    print(f"[pipeline] done -> {run_dir.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Then `_run`, which is the stage order from design §4:

```python
def _run(queue, run_dir, by_id, pipe_cfg, screen_run, args) -> None:
    from PIL import Image

    def draft_of(cid):
        return Image.open(screen_run / cid / "draft.png").convert("RGB")

    # --- ROUND 0: verify the draft -------------------------------------
    _verify_round(queue, run_dir, by_id, pipe_cfg, draft_of, attempt=0,
                  device=args.device, image_of=draft_of)

    # --- ONCE: retrieve, then mask (design §4) -------------------------
    if args.arm == "oracle":
        for cid in queue.pending("retrieve"):
            refs = [str(p) for p in by_id[cid].gt_refs]
            (run_dir.case_dir(cid) / "refs.json").write_text(json.dumps(refs))
            queue.mark(cid, "retrieve", "done")
    else:
        stage_with_model(
            queue, "retrieve",
            lambda: _load_retriever(pipe_cfg, args.device),
            lambda r, cid: _retrieve_one(r, run_dir, by_id[cid],
                                         pipe_cfg.retry_budget))

    stage_with_model(queue, "mask",
                     lambda: _load_masker(args.device),
                     lambda m, cid: _mask_one(m, run_dir, by_id[cid],
                                              draft_of(cid), pipe_cfg))

    # --- ROUNDS 1..N ---------------------------------------------------
    for attempt in range(1, pipe_cfg.retry_budget + 1):
        if not queue.needs_round(attempt):
            break
        stage_with_model(
            queue, "regen",
            lambda: _load_regen(args, pipe_cfg),
            lambda eng, cid: _regen_one(eng, run_dir, by_id[cid],
                                        draft_of(cid), attempt, args),
            attempt=attempt)
        _verify_round(queue, run_dir, by_id, pipe_cfg, draft_of,
                      attempt=attempt, device=args.device,
                      image_of=lambda cid: Image.open(
                          run_dir.case_dir(cid) / f"attempt_{attempt}.png"
                      ).convert("RGB"))

    _finalise(queue, run_dir)
```

`_run` references eight helpers that do not exist yet. **Task 8 writes them.** Until then this
module will not run end to end — which is fine, because the tests in this task only exercise
`resolve_screen_run`, and Step 5 confirms the import of `run_pipeline` still succeeds.

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_run_pipeline.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the suite**

Run: `$PY -m pytest -q`
Expected: 351 passed, 11 deselected

- [ ] **Step 6: Commit**

```bash
git add scripts/run_pipeline.py tests/test_run_pipeline.py
git commit -m "feat: the pipeline driver, stage-batched over rounds"
```

---

### Task 8: The eight stage helpers

**Files:**
- Modify: `scripts/run_pipeline.py`
- Test: `tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: `stage_with_model` (Task 6), `main`/`_run` (Task 7)
- Produces: `_load_masker`, `_mask_one`, `_load_retriever`, `_retrieve_one`, `_load_regen`, `_regen_one`, `_verify_round`, `_finalise`

- [ ] **Step 1: Write the failing test**

```python
from ragregen.schedule import Queue
from scripts.run_pipeline import _finalise


class _Run:
    """Minimal RunDir stand-in: only case_dir is used by _finalise."""

    def __init__(self, root):
        self.path = root

    def case_dir(self, cid):
        d = self.path / cid
        d.mkdir(parents=True, exist_ok=True)
        return d


def test_finalise_marks_a_winning_attempt_repaired(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a"], retry_budget=3, screen_run="s")
    run = _Run(tmp_path)
    (run.case_dir("a") / "scores.json").write_text(
        '{"draft": [false, 0.1], "attempt_1": [true, 0.9]}')

    _finalise(q, run)

    assert q.cases["a"].best == "attempt_1"
    assert q.cases["a"].status == "repaired"


def test_finalise_marks_a_winning_draft_unrepaired(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a"], retry_budget=3, screen_run="s")
    run = _Run(tmp_path)
    (run.case_dir("a") / "scores.json").write_text(
        '{"draft": [false, 0.8], "attempt_1": [false, 0.2]}')

    _finalise(q, run)

    assert q.cases["a"].best == "draft"
    #: Not "failed" -- the pipeline ran correctly and produced nothing better.
    #: That is a result, not an error, and the arm table needs the difference.
    assert q.cases["a"].status == "unrepaired"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_run_pipeline.py -v`
Expected: FAIL — `ImportError: cannot import name '_finalise'`

- [ ] **Step 3: Implement**

Add to `scripts/run_pipeline.py`. Each helper mirrors the equivalent block in
`scripts/repair_case.py` (masking, regen) or `scripts/score_a.py` / `score_b.py` (verification):

```python
def _load_masker(device: str):
    return mask.Masker(models.DinoDetector(device=device),
                       sam=models.SamSegmenter(device=device))


def _mask_one(masker, run_dir, case, draft, pipe_cfg) -> None:
    """The draft's mask, and a cutout per reference. Once per case, not per round.

    Every reference a run could need is already known -- retrieval depth is the
    retry budget -- so masking them all in one DINO+SAM residency turns N model
    loads into one (design §4).
    """
    d = run_dir.case_dir(case.id)
    drafted = mask.mask_draft(draft, case.coarse, masker,
                              score_phrase=case.concept,
                              dilate_px=pipe_cfg.mask_dilate_px)
    if drafted is None:
        raise ValueError(f"'{case.coarse}' did not ground in the draft")

    from PIL import Image
    Image.fromarray((drafted.mask > 0.5).astype("uint8") * 255, "L").save(
        d / "mask.png")

    refs = [Path(p) for p in json.loads((d / "refs.json").read_text())]
    for i, ref_path in enumerate(refs, 1):
        reference = Image.open(ref_path).convert("RGB")
        cut = composite.cutout_from(
            reference, mask.mask_reference(reference, case.concept, masker))
        if cut is not None:
            cut.save(d / f"cutout_{i}.png")


def _load_retriever(pipe_cfg, device: str):
    from ragregen import encoders, retrieve

    db = config.load_retrieval_db()
    return retrieve.Retriever.from_index(
        db.index_path, encoders.build_encoder(pipe_cfg.retriever, device=device))


def _retrieve_one(retriever, run_dir, case, k: int) -> None:
    hits = retriever.search(case.concept, k)
    (run_dir.case_dir(case.id) / "refs.json").write_text(
        json.dumps([str(h.path) for h in hits]))


def _load_regen(args, pipe_cfg):
    if args.mechanism == "stitch":
        return _Stitcher()                      # no weights, no card
    cfg = regen.KontextConfig(device=args.device, steps=pipe_cfg.steps,
                              seed=pipe_cfg.seed)
    return regen.Inpainter(cfg).load()


class _Stitcher:
    """Gives the diffusion-free mechanism the same load/free shape as Inpainter."""

    def free(self) -> None:
        env.reclaim_gpu()


def _regen_one(engine, run_dir, case, draft, attempt: int, args) -> None:
    """Attempt `attempt`, always from the ORIGINAL draft.

    `draft` is re-opened by the caller every round. Chaining edits -- feeding
    attempt N-1 into attempt N -- would degrade the whole image, which is the
    guarantee RUNBOOK §1 makes to anyone reading the results.
    """
    from PIL import Image

    d = run_dir.case_dir(case.id)
    mask_pil = Image.open(d / "mask.png").convert("L")
    cutout_path = d / f"cutout_{attempt}.png"
    if not cutout_path.is_file():
        raise ValueError(f"no reference cutout for attempt {attempt}")

    if args.mechanism == "stitch":
        result = regen.stitch(draft, mask_pil,
                              Image.open(cutout_path).convert("RGBA"),
                              feather_px=regen.KontextConfig().feather_px)
    else:
        refs = json.loads((d / "refs.json").read_text())
        reference = Image.open(refs[attempt - 1]).convert("RGB")
        prompt = f"replace the {case.coarse} with the {case.concept}"
        result = engine.regen(draft, mask_pil, reference, prompt)

    result.image.save(d / f"attempt_{attempt}.png")


def _verify_round(queue, run_dir, by_id, pipe_cfg, draft_of, *, attempt: int,
                  device: str, image_of) -> None:
    """Stream A then Stream B, as TWO residencies.

    Never one call: DINO+SigLIP and Qwen3-VL are separate loads, and a run
    killed between them must resume at the second rather than redo both.
    """
    from ragregen import concepts, encoders

    def load_grounded():
        return grounded.GroundedVerifier(
            models.DinoDetector(device=device),
            models.build_crop_scorer(pipe_cfg.crop_scorer, device=device),
            tau=pipe_cfg.tau)

    def score_grounded(verifier, cid):
        case = by_id[cid]
        #: parse takes the target and coarse term explicitly -- the same call
        #: score_a.py:41 makes. Passing only the prompt silently yields no
        #: fine-grained contrast.
        parsed = concepts.parse(case.prompt, target=case.concept,
                                coarse=case.coarse)
        scores = verifier.score(image_of(cid), parsed)
        _stash(run_dir, cid, attempt, "grounded", {k: v.state
                                                   for k, v in scores.items()})

    stage_with_model(queue, "grounded", load_grounded, score_grounded,
                     attempt=attempt)

    def load_semantic():
        from ragregen import vlm

        #: QwenVLM, not Qwen3VL -- the class in ragregen/vlm.py:30, wired the
        #: same way score_b.py:71 wires it.
        return semantic.SemanticVerifier(vlm.QwenVLM(device=device))

    def judge(verifier, cid):
        case = by_id[cid]
        verdict = verifier.judge(image_of(cid), case.prompt)
        _stash(run_dir, cid, attempt, "semantic",
               {"ok": verdict.ok, "degenerate": verdict.degenerate})

    stage_with_model(queue, "semantic", load_semantic, judge, attempt=attempt)


def _stash(run_dir, case_id: str, attempt: int, stream: str, payload) -> None:
    """Append one stream's result for one attempt to the case's scores file."""
    path = run_dir.case_dir(case_id) / "streams.json"
    data = json.loads(path.read_text()) if path.is_file() else {}
    data.setdefault(str(attempt), {})[stream] = payload
    path.write_text(json.dumps(data, indent=2, default=str))


def _finalise(queue, run_dir) -> None:
    """Pick the candidate to report, per case.

    Reads scores.json -- {label: [ok, score]} -- which the fuse step wrote.
    """
    for case_id, state in queue.cases.items():
        path = run_dir.case_dir(case_id) / "scores.json"
        if not path.is_file():
            continue
        scored = json.loads(path.read_text())
        draft_ok, draft_score = scored.get("draft", [False, 0.0])
        attempts = [(label, bool(v[0]), float(v[1]))
                    for label, v in sorted(scored.items())
                    if label != "draft"]
        best = schedule.select_best(float(draft_score), attempts)
        queue.cases[case_id].best = best
        queue.cases[case_id].status = ("unrepaired" if best == "draft"
                                       else "repaired")
    queue.save()
```

The `fuse` call that writes `scores.json` belongs at the end of `_verify_round`, after both
residencies have run:

```python
    for cid in list(by_id):
        streams = run_dir.case_dir(cid) / "streams.json"
        if not streams.is_file():
            continue
        both = json.loads(streams.read_text()).get(str(attempt), {})
        if "grounded" not in both or "semantic" not in both:
            continue                      # one stream failed; not fusable
        label = "draft" if attempt == 0 else f"attempt_{attempt}"
        _record_score(run_dir, cid, label, both)
```

with `_record_score` fusing the two streams into the one number `_finalise` selects on:

```python
def _record_score(run_dir, case_id: str, label: str, both: dict) -> None:
    """Fuse Stream A and Stream B into {label: [ok, score]} in scores.json."""
    from ragregen.verify.grounded import ConceptScore

    scores = {phrase: ConceptScore(phrase, "concept", state)
              for phrase, state in both["grounded"].items()}
    verdict = fusion.fuse(scores, semantic.SemanticVerdict(
        ok=both["semantic"]["ok"], raw="",
        degenerate=both["semantic"]["degenerate"]))

    path = run_dir.case_dir(case_id) / "scores.json"
    data = json.loads(path.read_text()) if path.is_file() else {}
    data[label] = [bool(verdict.ok), float(verdict.score)]
    path.write_text(json.dumps(data, indent=2, default=str))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$PY -m pytest tests/test_run_pipeline.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/run_pipeline.py tests/test_run_pipeline.py
git commit -m "feat: the stage helpers, masking once and regenerating from the draft"
```

---

### Task 9: The two round-structure guarantees

**Files:**
- Test: `tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: `stage_with_model` (Task 6), `Queue` (Tasks 1–2)
- Produces: no new API — these pin behaviour the spec §11 requires

- [ ] **Step 1: Write the failing test**

```python
def test_the_mask_stage_runs_once_across_every_round(tmp_path):
    """Design §4. Masking per round turns 1 model load into N for no gain."""
    q = Queue.open(tmp_path / "q.json", ["a"], retry_budget=3, screen_run="s")
    masked, loads = [], []

    for attempt in range(1, 4):
        #: mask is attempt-free, so the SAME cell is consulted every round.
        stage_with_model(q, "mask", lambda: loads.append("m") or "M",
                         lambda m, cid: masked.append((cid, attempt)))

    assert masked == [("a", 1)]
    assert loads == ["m"]


def test_every_attempt_regenerates_from_the_same_draft(tmp_path):
    """RUNBOOK §1: never from the previous attempt. Chained edits degrade."""
    q = Queue.open(tmp_path / "q.json", ["a"], retry_budget=3, screen_run="s")
    handed = []

    #: Stands in for run_pipeline's draft_of(cid), which re-opens the file
    #: from the screen run every round rather than reading attempt_{n-1}.
    def draft_of(cid):
        return f"DRAFT::{cid}"

    for attempt in range(1, 4):
        stage_with_model(q, "regen", lambda: "ENGINE",
                         lambda eng, cid: handed.append(draft_of(cid)),
                         attempt=attempt)

    assert handed == ["DRAFT::a", "DRAFT::a", "DRAFT::a"]
    assert len(set(handed)) == 1
```

- [ ] **Step 2: Run the test to verify it fails or passes**

Run: `$PY -m pytest tests/test_run_pipeline.py -v -k "once_across or same_draft"`
Expected: PASS — both are guaranteed by `stage_key`'s attempt-free handling (Task 1) and by
`_run` re-deriving the draft each round (Task 7). If either FAILS, the corresponding earlier task
is wrong and must be fixed rather than the test relaxed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_run_pipeline.py
git commit -m "test: pin masking-once and regenerate-from-draft"
```

---

### Task 10: Wire it in and document it

**Files:**
- Modify: `scripts/run.sh`, `docs/RUNBOOK.md` §4
- Test: none — exercised by running it

**Interfaces:**
- Consumes: `scripts/run_pipeline.py` (Tasks 6–8)
- Produces: no new API

- [ ] **Step 1: Add the stage to `run.sh`**

In the usage block, after the `score-b` line:

```
  pipeline     THE LOOP: repair every case    (GPU, hours; resumable)
```

In the `case` statement, after the `score-b` line:

```bash
  pipeline)    exec "$PY" scripts/run_pipeline.py "$@" ;;
```

- [ ] **Step 2: Document it**

In `docs/RUNBOOK.md` §4, add after the `repair` row:

```markdown
| `pipeline` | yes | hours | The full loop over every case: verify → retrieve → mask → regenerate → re-verify, up to N attempts. **Resumable** — if it dies because someone took the card, re-run with `--resume outputs/pipeline_<TS>`. Never re-drafts; it reads `outputs/screen_latest/`. |
```

- [ ] **Step 3: Check the card before running anything**

Run: `nvidia-smi`
Expected: ≥12 GB free. If not, stop — `regen`'s guard will refuse anyway, and that refusal is correct behaviour.

- [ ] **Step 4: Smoke the resume path on two cases**

Run: `./scripts/run.sh pipeline --limit 2 --arm oracle --mechanism stitch`

`stitch` needs no FLUX weights, so this exercises the whole state machine in about a minute. Kill it with Ctrl-C partway, then re-run with `--resume outputs/pipeline_<TS>` and confirm from the log that completed cases are skipped and no model is reloaded for an empty stage.

- [ ] **Step 5: Run the complete suite**

Run: `$PY -m pytest -q`
Expected: all green, `@pytest.mark.gpu` deselected.

- [ ] **Step 6: Commit**

```bash
git add scripts/run.sh docs/RUNBOOK.md
git commit -m "feat: pipeline is a run.sh stage, and resumable by documented command"
```

---

## Notes for the executor

- **`schedule.py` must never import torch.** If a test needs `torch.OutOfMemoryError`, fake it by name as Task 4 does. The moment torch appears in that module the whole scheduler becomes untestable on a machine whose GPU is busy — which, per `findings/2026-07-28-regeneration-result.md` §4, is most of the time.
- **Do not mark a case failed on OOM.** It never got a fair attempt. Task 4 pins this and it is easy to "fix" wrongly.
- **`retrieve` and `mask` are attempt-free on purpose.** Masking inside the round loop turns 1 model load into N for no gain (design §4). If a test seems to want per-attempt masking, the test is wrong.
- **Check `nvidia-smi` before Task 10 Step 4.** Use `--mechanism stitch` for the smoke: it needs no diffusion weights and proves the same state machine.
- **`sushi` masks the plate, not the food** (`findings/2026-07-28-masking-result.md` §3). Expected, not a bug.
- Selection uses the verifier's score only. The identity and preservation metrics are doc 4's; selecting on them would be selecting on the thing being measured.
