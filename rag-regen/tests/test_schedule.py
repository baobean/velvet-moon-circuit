"""The scheduler's state machine.

Every test here runs on CPU in milliseconds, because schedule.py imports no
torch and no models. That is the whole point of the split: the card is held by
someone else most of the time (findings/2026-07-28-regeneration-result.md §4),
and the scheduling rules still have to be verifiable.
"""
import json
import re

import pytest

from ragregen import schedule
from ragregen.schedule import (CaseState, Queue, StageAborted, is_fatal,
                               run_stage, select_best, select_winner)


class FakeOOM(Exception):
    """Stands in for torch.OutOfMemoryError, which we refuse to import."""


FakeOOM.__name__ = "OutOfMemoryError"


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
    #: SCHEMA bumped to 2 for the healthy/unrepaired split -- Queue.open still
    #: reads schema-1 files unchanged, pinned by
    #: test_a_schema_1_queue_still_opens.
    assert data["schema"] == 2
    assert data["cases"]["a"]["status"] == "pending"
    #: case_id is the dict key; storing it twice invites the two to disagree.
    assert "case_id" not in data["cases"]["a"]


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


def test_the_heartbeat_names_every_completed_case(capsys):
    #: A stand-in rather than Queue.open, so the test needs no temp directory;
    #: run_stage only calls pending(), mark() and save().
    class _Q:
        def __init__(self):
            self.done = []

        def pending(self, stage, attempt=None):
            return ["a", "b"]

        def mark(self, cid, stage, state, attempt=None, **kw):
            self.done.append(cid)

        def save(self):
            pass

    queue = _Q()
    schedule.run_stage(queue, "verify", lambda cid: None)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln]
    #: No case failed here, so every line must be a done-line.
    assert len(lines) == 2
    #: `tail -f` on a 15-hour run must distinguish "waiting for the card"
    #: from "hung" -- that only works if each case gets its OWN line. A
    #: substring check ("verify" in out and "a" in out and "b" in out) would
    #: pass just as happily on one batched summary line like
    #: f"{stage}: {', '.join(done)} done", which destroys the property.
    #: Pinning the cadence means matching one done-line per case id, robust
    #: to the "[HH:MM:SS]" timestamp prefix.
    done_lines = [ln for ln in lines if re.search(r"verify: \w+ done$", ln)]
    assert len(done_lines) == 2
    assert any(ln.endswith("verify: a done") for ln in done_lines)
    assert any(ln.endswith("verify: b done") for ln in done_lines)


def test_the_heartbeat_names_a_failing_case_too(capsys):
    #: The `except` branch marked the case failed and `continue`d BEFORE
    #: reaching the heartbeat print, so a stage where every case fails looked
    #: identical to a hang -- defeating the feature's stated purpose (`tail
    #: -f` distinguishing "waiting for the card" from "hung") exactly when
    #: the operator most needs it.
    class _Q:
        def pending(self, stage, attempt=None):
            return ["a", "b"]

        def mark(self, cid, stage, state, attempt=None, **kw):
            pass

        def save(self):
            pass

    def _fn(cid):
        if cid == "b":
            raise ValueError("this concept will not ground")

    schedule.run_stage(_Q(), "grounded", _fn, attempt=0)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln]
    #: One line per case, whichever way the case went.
    assert len(lines) == 2
    done = [ln for ln in lines if re.search(r"grounded attempt 0: \w+ done$",
                                            ln)]
    assert [ln for ln in done if ln.endswith("a done")]
    failed = [ln for ln in lines
              if re.search(r"grounded attempt 0: \w+ FAILED \(\w+\)$", ln)]
    assert len(failed) == 1
    #: The case id and the exception type, so `tail -f` shows whether it is
    #: one bad concept or the same fault 30 times.
    assert failed[0].endswith("grounded attempt 0: b FAILED (ValueError)")


def test_a_scoreless_failing_attempt_never_displaces_the_draft():
    """The anas_platyrhynchos shape: the draft failed, the attempt failed, and
    the attempt carries no Stream A evidence at all."""
    assert select_best(0.0, [("attempt_1", False, None)]) == "draft"


def test_a_higher_scoring_failing_attempt_still_wins_when_it_has_evidence():
    assert select_best(0.1, [("attempt_1", False, 0.4)]) == "attempt_1"


def test_a_none_draft_score_does_not_crash_selection():
    assert select_best(None, [("attempt_1", False, 0.4)]) == "attempt_1"
    assert select_best(None, [("attempt_1", False, None)]) == "draft"


def test_select_winner_short_circuits_a_genuinely_passing_draft():
    """CRITICAL 1: reverify's population -- pre-existing attempt images
    scored against a draft that may now pass -- can have draft_ok=True
    alongside a passing attempt. A live pipeline run never reaches this
    shape (_resolve stops the round loop the moment a passing candidate is
    found), but reverify's corrected re-grade is exactly the first caller
    that can. select_winner is the ONE place this rule lives, shared by
    reverify.reselect, report._best_label and run_pipeline._finalise_one.
    """
    assert select_winner(True, 0.9, [("attempt_1", True, 0.4)]) == "draft"


def test_select_winner_is_a_no_op_when_no_attempts_exist():
    """A live pipeline run's own invariant: draft_ok=True implies no
    attempts were ever generated. select_winner must behave exactly like
    plain select_best there -- this is the no-op proof for existing runs."""
    for draft_ok, draft_score in ((True, 0.9), (False, 0.1)):
        assert (select_winner(draft_ok, draft_score, [])
                == select_best(draft_score, []))


def test_select_winner_falls_through_to_select_best_when_the_draft_failed():
    assert (select_winner(False, 0.1, [("attempt_1", True, 0.4)])
            == select_best(0.1, [("attempt_1", True, 0.4)])
            == "attempt_1")


def test_a_schema_1_queue_still_opens(tmp_path):
    """The doc-5 queue must stay resumable across the schema bump."""
    p = tmp_path / "queue.json"
    p.write_text(json.dumps({
        "schema": 1, "retry_budget": 3, "screen_run": "outputs/screen_x",
        "cases": {"fox": {"stages": {"grounded@0": "done"}, "attempt": 0,
                          "status": "unrepaired", "best": "draft",
                          "error": None}}}))
    q = Queue.open(p, ["fox"], retry_budget=3, screen_run="ignored")
    assert q.cases["fox"].status == "unrepaired"
    assert q.cases["fox"].best == "draft"
