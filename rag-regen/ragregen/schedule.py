"""The scheduler's state, and nothing else.

No torch, no models, no I/O beyond queue.json. That constraint is load-bearing:
it is what lets every scheduling rule below be tested on CPU in milliseconds,
on a machine whose GPU is routinely held by someone else for days
(findings/2026-07-28-regeneration-result.md §4).
"""
from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: One entry per MODEL RESIDENCY, not per logical step. "verify" is two
#: separate loads -- DINO+SigLIP, then Qwen3-VL -- and a run killed between
#: them must resume at the second rather than redo both.
STAGES = ("grounded", "semantic", "retrieve", "mask", "regen")

#: Stages that run once per case rather than once per attempt (design §4).
ATTEMPT_FREE = ("retrieve", "mask")

SCHEMA = 2


@dataclass
class CaseState:
    case_id: str
    stages: dict[str, str] = field(default_factory=dict)
    attempt: int = 0
    status: str = "pending"          # pending|healthy|repaired|unrepaired|failed
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

    def pending(self, stage: str, *, attempt: int | None = None) -> list[str]:
        """Cases still owed this stage, in insertion order.

        A case whose status has left "pending" is finished -- repaired, or
        failed -- and is never handed to another stage regardless of which
        cells its stages dict is missing.
        """
        key = stage_key(stage, attempt)
        return [cid for cid, s in self.cases.items()
                if s.status == "pending" and key not in s.stages]

    def mark(self, case_id: str, stage: str, outcome: str, *,
             attempt: int | None = None, **fields) -> None:
        """Record a stage outcome and checkpoint.

        `outcome` is the STAGE cell's value (done|failed). The CASE's status
        is a different thing and travels in **fields -- naming this parameter
        `status` made `mark(..., status="repaired")` collide with the
        positional and raise TypeError, so the two names stay distinct.

        Saves on every call rather than at stage boundaries: the failure being
        defended against is a SIGKILL partway through a 30-case stage.
        """
        state = self.cases[case_id]
        state.stages[stage_key(stage, attempt)] = outcome
        for key, value in fields.items():
            if not hasattr(state, key):
                raise AttributeError(f"CaseState has no field {key!r}")
            setattr(state, key, value)
        self.save()

    def unresolved(self) -> list[str]:
        return [cid for cid, s in self.cases.items() if s.status == "pending"]

    def needs_round(self, attempt: int) -> bool:
        """Whether round `attempt` has any work. Checked BEFORE loading FLUX."""
        return attempt <= self.retry_budget and bool(self.unresolved())


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
    suffix = "" if attempt is None else f" attempt {attempt}"
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
            #: A failing case gets its own line too. Returning here without
            #: one would make a stage where EVERY case fails look identical
            #: to a hang -- defeating the heartbeat exactly when the operator
            #: most needs it. The exception type is named so `tail -f` shows
            #: whether it is one bad concept or the same fault 30 times.
            print(f"[{time.strftime('%H:%M:%S')}] {stage}{suffix}: {case_id} "
                  f"FAILED ({type(exc).__name__})", flush=True)
            continue
        queue.mark(case_id, stage, "done", attempt=attempt)
        #: One line per completed case. On a 15-hour run this is how `tail -f`
        #: distinguishes "waiting for the card" from "hung" (doc 5 §9).
        print(f"[{time.strftime('%H:%M:%S')}] {stage}{suffix}: {case_id} done",
              flush=True)


def select_best(draft_score: float | None, attempts) -> str:
    """The candidate to report, from {draft, attempt_1 .. attempt_N}.

    `attempts` is (label, ok, score) in attempt order; score may be None,
    meaning Stream A produced no evidence for that candidate. A candidate with
    no evidence can never displace the draft -- it is not better, it is
    unmeasured.

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
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_label, best_score = label, score
    return best_label


def select_winner(draft_ok: bool, draft_score: float | None, attempts) -> str:
    """The candidate to report, for a population select_best alone cannot see.

    The ONE home for "which candidate wins" -- shared by reverify.reselect,
    report._best_label and run_pipeline._finalise_one, so the three cannot
    independently drift the way `unrepaired` once meant two things
    (findings/2026-07-29-orchestration-result.md §2).

    `draft_ok` short-circuits: a genuinely PASSING draft wins outright, even
    against an attempt that also passed and scores higher. In a live
    pipeline run this is a no-op -- `_resolve` stops the round loop the
    moment a passing candidate is found, so `draft_ok=True` always implies
    `attempts` is empty there, and select_best(draft_score, []) already
    returns "draft". But `reverify` re-grades pre-existing attempt images
    against a draft that a corrected scoring may now pass for the first
    time, and that later attempt was only generated because the ORIGINAL
    (buggy) scoring said the draft had failed. Under corrected scoring it
    never should have been generated at all, so it cannot be allowed to
    outrank a draft that, correctly scored, was fine.
    """
    if draft_ok:
        return "draft"
    return select_best(draft_score, attempts)
