# tests/test_reverify.py
import json
import sys
from pathlib import Path

# scripts/ is a package (scripts/__init__.py) but reverify.py itself is
# written to be imported bare (it does the same trick internally to reach
# run_pipeline) -- so the test has to put scripts/ on sys.path, the same way
# every other scripts/*.py's sibling-module imports work when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import reverify


def _finished_run(tmp_path):
    src = tmp_path / "doc5_src"
    for cid in ("fox", "duck"):
        d = src / cid
        d.mkdir(parents=True)
        (d / "mask.png").write_bytes(b"PNG")
        (d / "attempt_1.png").write_bytes(b"PNG")
        (d / "scores.json").write_text(json.dumps({"draft": [False, 0.0],
                                                   "attempt_1": [False, 1.0]}))
    (src / "queue.json").write_text(json.dumps({
        "schema": 1, "retry_budget": 3, "screen_run": "outputs/screen_x",
        "cases": {"fox": {"stages": {}, "attempt": 1, "status": "repaired",
                          "best": "attempt_1", "error": None},
                  "duck": {"stages": {}, "attempt": 1, "status": "repaired",
                           "best": "attempt_1", "error": None}}}))
    return src


def test_the_source_run_is_never_written(tmp_path):
    src = _finished_run(tmp_path)
    before = {p: p.stat().st_mtime_ns for p in src.rglob("*") if p.is_file()}
    reverify.plan_targets(src)
    after = {p: p.stat().st_mtime_ns for p in src.rglob("*") if p.is_file()}
    assert before == after


def test_targets_are_the_draft_and_every_attempt(tmp_path):
    src = _finished_run(tmp_path)
    targets = reverify.plan_targets(src)
    assert targets["fox"] == [0, 1]


def test_pixels_are_symlinked_not_copied(tmp_path):
    src = _finished_run(tmp_path)
    dst = tmp_path / "out"
    dst.mkdir()
    reverify.link_pixels(src, _RunDirStub(dst), ["fox"])
    linked = dst / "fox" / "attempt_1.png"
    assert linked.is_symlink()
    assert linked.resolve() == (src / "fox" / "attempt_1.png").resolve()


def test_reselection_drops_a_promoted_failing_attempt(tmp_path):
    """anas_platyrhynchos: draft [false, 0.0], attempt_1 [false, 1.0]."""
    scored = {"draft": [False, 0.0], "attempt_1": [False, None]}
    assert reverify.reselect(scored) == "draft"


def test_reselection_keeps_a_genuine_pass():
    scored = {"draft": [False, 0.0], "attempt_1": [True, 0.4]}
    assert reverify.reselect(scored) == "attempt_1"


def test_reselect_agrees_with_the_report_on_a_passing_draft_and_attempt():
    """CRITICAL 1: reverify.reselect and report's own selection must never
    be able to disagree about which image a case's metrics describe. A
    draft that passed under corrected scoring must win outright even when
    a leftover attempt image also happens to score higher -- that attempt
    was only generated because the ORIGINAL (buggy) scoring said the draft
    had failed.
    """
    from scripts.report import _best_label

    scored = {"draft": [True, 0.9], "attempt_1": [True, 0.4]}
    assert reverify.reselect(scored) == _best_label(scored) == "draft"


def test_source_run_survives_into_the_final_run_json(tmp_path, monkeypatch):
    """The exact bug the draft's main() would have shipped: it wrote
    run.json with source_run BEFORE calling run_dir.finish(), and
    RunDir.finish rewrites run.json wholesale (trace.py:64-72), silently
    erasing it. source_run must still be there after main() returns.

    _verify_round and model loading are stubbed out -- this test is about
    run.json plumbing, not verification, and must run with no GPU."""
    src = _finished_run(tmp_path)
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(json.dumps({
        "name": "t",
        "images_root": str(tmp_path),
        "cases": [
            {"id": "fox", "prompt": "a fox in a den", "concept": "fox",
             "coarse": "animal", "gt_refs": ["x.png"]},
            {"id": "duck", "prompt": "a duck on a pond", "concept": "duck",
             "coarse": "animal", "gt_refs": ["y.png"]},
        ],
    }))

    monkeypatch.setattr(reverify.run_pipeline, "_verify_round",
                        lambda *a, **k: None)
    #: Redirects trace.open_run's default root under tmp_path, so this test
    #: never touches the real outputs/ directory.
    monkeypatch.setattr(reverify.trace.env, "PROJECT_ROOT", tmp_path)

    rc = reverify.main(["--run", str(src), "--dataset", str(dataset)])
    assert rc == 0

    runs = [p for p in (tmp_path / "outputs").iterdir()
           if p.is_dir() and not p.is_symlink()
           and p.name.startswith("reverify_")]
    assert len(runs) == 1
    run_json = json.loads((runs[0] / "run.json").read_text())
    assert run_json["results"]["source_run"] == str(src.resolve())
    assert run_json["status"] == "ok"


def test_finalize_loop_computes_best_and_status_end_to_end(tmp_path, monkeypatch):
    """Wires reselect() + _status_for + the queue mutation together inside
    main(), against a REAL Queue and a REAL queue.json on disk -- reselect's
    own unit tests never exercise this wiring, and a swapped argument, a
    wrong dict key, or a dropped queue.save() would pass every other test
    in this file silently.

    The fake _verify_round below writes scores.json directly instead of
    running real DINO/SigLIP/Qwen -- this test is about the finalize loop,
    not verification, and must run with no GPU.
    """
    src = tmp_path / "doc5_src"
    case_ids = ("draft_passes", "attempt_beats_draft", "anas_platyrhynchos")
    for cid in case_ids:
        (src / cid).mkdir(parents=True)
        (src / cid / "attempt_1.png").write_bytes(b"PNG")

    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(json.dumps({
        "name": "t",
        "images_root": str(tmp_path),
        "cases": [{"id": cid, "prompt": f"a {cid}", "concept": cid,
                   "coarse": "animal", "gt_refs": ["x.png"]}
                  for cid in case_ids],
    }))

    #: {case_id: {label: [ok, score]}} -- the three shapes doc 5's fix
    #: exists for.
    scores_by_case = {
        "draft_passes": {"draft": [True, 0.9]},
        "attempt_beats_draft": {"draft": [False, 0.1],
                                "attempt_1": [True, 0.4]},
        "anas_platyrhynchos": {"draft": [False, 0.0],
                               "attempt_1": [False, None]},
    }

    def fake_verify_round(queue, run_dir, by_id, pipe_cfg, *, attempt,
                          device, image_of, bank=None, embedder=None):
        label = "draft" if attempt == 0 else f"attempt_{attempt}"
        for cid in by_id:
            entry = scores_by_case[cid].get(label)
            if entry is None:
                continue
            path = run_dir.case_dir(cid) / "scores.json"
            data = json.loads(path.read_text()) if path.is_file() else {}
            data[label] = entry
            path.write_text(json.dumps(data))

    monkeypatch.setattr(reverify.run_pipeline, "_verify_round",
                        fake_verify_round)
    #: Redirects trace.open_run's default root under tmp_path, so this test
    #: never touches the real outputs/ directory.
    monkeypatch.setattr(reverify.trace.env, "PROJECT_ROOT", tmp_path)

    rc = reverify.main(["--run", str(src), "--dataset", str(dataset)])
    assert rc == 0

    runs = [p for p in (tmp_path / "outputs").iterdir()
           if p.is_dir() and not p.is_symlink()
           and p.name.startswith("reverify_")]
    assert len(runs) == 1
    #: Read queue.json back off disk -- not the in-memory Queue -- so a
    #: dropped queue.save() would fail this test too.
    cases = json.loads((runs[0] / "queue.json").read_text())["cases"]

    assert cases["draft_passes"]["best"] == "draft"
    assert cases["draft_passes"]["status"] == "healthy"

    assert cases["attempt_beats_draft"]["best"] == "attempt_1"
    assert cases["attempt_beats_draft"]["status"] == "repaired"

    assert cases["anas_platyrhynchos"]["best"] == "draft"
    assert cases["anas_platyrhynchos"]["status"] == "unrepaired"


def _dataset_for(tmp_path, case_ids):
    """A minimal dataset.yaml (written as JSON, which YAML parses fine)
    covering exactly `case_ids` -- the shape every reverify test below
    needs, factored out once several tests started duplicating it."""
    path = tmp_path / "dataset.yaml"
    path.write_text(json.dumps({
        "name": "t",
        "images_root": str(tmp_path),
        "cases": [{"id": cid, "prompt": f"a {cid}", "concept": cid,
                   "coarse": "animal", "gt_refs": ["x.png"]}
                  for cid in case_ids],
    }))
    return path


def _reverify_run_dirs(tmp_path):
    return [p for p in (tmp_path / "outputs").iterdir()
           if p.is_dir() and not p.is_symlink()
           and p.name.startswith("reverify_")]


def test_a_short_case_never_reaches_a_round_it_has_no_image_for(
        tmp_path, monkeypatch):
    """CRITICAL 2(a): plan_targets already knows fox has only attempt_1 while
    duck has attempt_1 AND attempt_2. Before the fix, the round loop asked
    EVERY case for every attempt up to the campaign's max -- image_of(fox, 2)
    would open a non-existent attempt_2.png, run_stage would catch the
    FileNotFoundError and mark fox `failed` with a stray traceback, and the
    dead round would still pay a full DINO+SigLIP+Qwen-7B load even though
    fox had nothing left to verify.
    """
    src = tmp_path / "doc5_src"
    for cid, n_attempts in (("fox", 1), ("duck", 2)):
        d = src / cid
        d.mkdir(parents=True)
        for a in range(1, n_attempts + 1):
            (d / f"attempt_{a}.png").write_bytes(b"PNG")

    dataset = _dataset_for(tmp_path, ("fox", "duck"))
    seen_pending = {}

    def fake_verify_round(queue, run_dir, by_id, pipe_cfg, *, attempt,
                          device, image_of, bank=None, embedder=None):
        seen_pending[attempt] = sorted(queue.pending("grounded", attempt=attempt))

    monkeypatch.setattr(reverify.run_pipeline, "_verify_round",
                        fake_verify_round)
    monkeypatch.setattr(reverify.trace.env, "PROJECT_ROOT", tmp_path)

    rc = reverify.main(["--run", str(src), "--dataset", str(dataset)])
    assert rc == 0

    assert seen_pending[1] == ["duck", "fox"]
    #: fox has no attempt_2.png -- it must never be pending at attempt 2.
    assert seen_pending[2] == ["duck"]


def test_skip_out_of_round_pre_marks_cases_missing_this_attempt(tmp_path):
    """Unit-level: the mechanism _skip_out_of_round uses to keep a short
    case out of a round it has no image for -- pre-marking the per-round
    stage keys the same way Queue.pending already understands, no new
    code path."""
    q = reverify.schedule.Queue.open(tmp_path / "q.json", ["fox", "duck"],
                                     retry_budget=2, screen_run="s")
    targets = {"fox": [0, 1], "duck": [0, 1, 2]}

    reverify._skip_out_of_round(q, targets, attempt=2)

    assert q.pending("grounded", attempt=2) == ["duck"]
    assert q.pending("semantic", attempt=2) == ["duck"]
    #: fox is skipped for THIS round only -- its status must stay untouched,
    #: so a case whose targets have a gap could still be picked up later.
    assert q.cases["fox"].status == "pending"


def test_a_card_abort_finishes_the_run_and_prints_a_resume_command(
        tmp_path, monkeypatch, capsys):
    """CRITICAL 2(b): reverify.main had no try/except at all, so a
    schedule.StageAborted (routine -- 'someone else has the card') killed the
    run with a traceback and never called run_dir.finish, leaving a run
    directory with no run.json."""
    src = _finished_run(tmp_path)
    dataset = _dataset_for(tmp_path, ("fox", "duck"))

    def _raise(*a, **k):
        raise reverify.schedule.StageAborted(
            "grounded aborted on 'fox': CUDA out of memory")

    monkeypatch.setattr(reverify.run_pipeline, "_verify_round", _raise)
    monkeypatch.setattr(reverify.trace.env, "PROJECT_ROOT", tmp_path)

    rc = reverify.main(["--run", str(src), "--dataset", str(dataset)])
    assert rc == 2

    runs = _reverify_run_dirs(tmp_path)
    assert len(runs) == 1
    run_json = json.loads((runs[0] / "run.json").read_text())
    assert run_json["status"] == "aborted"

    out = capsys.readouterr().out
    assert "resume" in out.lower()
    assert "--resume" in out
    assert str(runs[0]) in out


def test_resume_reuses_an_existing_run_dir_and_its_queue(tmp_path, monkeypatch):
    """CRITICAL 2(c): trace.open_run always mints a fresh timestamped
    directory, so a killed reverify used to start over from nothing. --resume
    must construct a trace.RunDir over the EXISTING directory so
    schedule.Queue.open finds its queue.json and picks up where it left off."""
    src = _finished_run(tmp_path)
    dataset = _dataset_for(tmp_path, ("fox", "duck"))
    monkeypatch.setattr(reverify.trace.env, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(reverify.run_pipeline, "_verify_round",
                        lambda *a, **k: None)

    rc1 = reverify.main(["--run", str(src), "--dataset", str(dataset)])
    assert rc1 == 0
    runs = _reverify_run_dirs(tmp_path)
    assert len(runs) == 1
    run_dir_path = runs[0]

    #: Pre-seed the queue as if grounded@0 already ran for fox, to prove
    #: --resume picks the EXISTING queue.json back up rather than Queue.open
    #: starting fresh over the same directory.
    queue_path = run_dir_path / "queue.json"
    data = json.loads(queue_path.read_text())
    data["cases"]["fox"]["stages"]["grounded@0"] = "done"
    queue_path.write_text(json.dumps(data))

    rc2 = reverify.main(["--run", str(src), "--dataset", str(dataset),
                        "--resume", str(run_dir_path)])
    assert rc2 == 0

    #: Still exactly one reverify_* directory -- --resume must not mint a
    #: fresh timestamped one.
    assert _reverify_run_dirs(tmp_path) == [run_dir_path]
    resumed = json.loads(queue_path.read_text())
    assert resumed["cases"]["fox"]["stages"]["grounded@0"] == "done"


class _RunDirStub:
    def __init__(self, path):
        self.path = path

    def case_dir(self, cid):
        d = self.path / cid
        d.mkdir(parents=True, exist_ok=True)
        return d
