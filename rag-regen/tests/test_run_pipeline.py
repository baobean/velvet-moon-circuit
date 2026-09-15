"""The driver rules worth testing without a GPU.

Everything else in run_pipeline.py is model wiring. These are the rules that
decide whether a resumed run takes seconds or reloads FLUX for 5m21s to
discover it has nothing to do.
"""
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen.config import Case             # noqa: E402
from ragregen.schedule import CaseState, Queue  # noqa: E402
from scripts.run_pipeline import (_finalise, _resolve, _resume_command,  # noqa: E402
                                  resolve_screen_run, stage_with_model)
import scripts.run_pipeline as rp            # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_vram_check(monkeypatch):
    #: stage_with_model's preflight reads the ACTUAL card via
    #: env.free_vram_gb(), and this card is shared -- a neighbour routinely
    #: holds it (findings/2026-07-28-regeneration-result.md §4). Every test
    #: in this file that calls stage_with_model without an opinion about VRAM
    #: must not have its outcome depend on what that neighbour is doing right
    #: now. None is the documented "unknown" value, and every caller already
    #: has to treat unknown as permission to proceed, so this is the honest
    #: default rather than a bypass. Tests that DO have an opinion (below)
    #: override it with their own monkeypatch.setattr, which wins.
    monkeypatch.setattr(rp.env, "free_vram_gb", lambda i=0: None)


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


def test_cleanup_error_does_not_mask_the_resumable_stage_abort(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a"], retry_budget=3,
                   screen_run="s")

    class Model:
        def free(self):
            raise RuntimeError("CUDA context is poisoned")

    def work(model, cid):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(rp.schedule.StageAborted, match="out of memory"):
        stage_with_model(q, "regen", Model, work, attempt=1)


def test_a_missing_screen_run_is_an_error_not_a_redraft(tmp_path):
    #: The pipeline must NEVER helpfully draft its own images: those would be
    #: a different seed from the set that was hand-labelled, so the labels
    #: would no longer describe the images being repaired.
    with pytest.raises(FileNotFoundError, match="screen"):
        resolve_screen_run(tmp_path / "nope")


def test_resume_command_replays_every_original_flag(monkeypatch):
    #: A bare `--resume PATH` silently reverts every OTHER flag to its
    #: argparse default -- that is exactly how `--verifier none` came back
    #: as `--verifier fused` on resume and marked every case's queue status
    #: "failed" on a KeyError. The printed hint must be copy-paste safe.
    monkeypatch.setattr(sys, "argv", [
        "run_pipeline.py", "--dataset", "configs/dataset_phase_b.yaml",
        "--screen-run", "outputs/screen_20260819", "--verifier", "none",
        "--open-loop-attempts", "1", "--arm", "full", "--mechanism", "inpaint"])
    cmd = _resume_command(Path("/out/pipeline_20260820"))
    assert "--verifier none" in cmd
    assert "--dataset configs/dataset_phase_b.yaml" in cmd
    assert "--resume /out/pipeline_20260820" in cmd


def test_resume_command_does_not_duplicate_a_prior_resume_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_pipeline.py", "--verifier", "none",
        "--resume", "/out/pipeline_old"])
    cmd = _resume_command(Path("/out/pipeline_new"))
    assert cmd.count("--resume") == 1
    assert "--resume /out/pipeline_new" in cmd
    assert "/out/pipeline_old" not in cmd


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


class _Run:
    """Minimal RunDir stand-in: only case_dir is used by _finalise."""

    def __init__(self, root):
        self.path = root

    def case_dir(self, cid):
        d = self.path / cid
        d.mkdir(parents=True, exist_ok=True)
        return d


@pytest.fixture
def fake_run_dir(tmp_path):
    """Same shape as `_Run` above -- reused rather than duplicated."""
    return _Run(tmp_path)


def _blank_image():
    return Image.new("RGB", (8, 8), "white")


def run_pipeline_case_stub():
    return Case(id="fox_case", prompt="a fox in grass", concept="fox",
               coarse="canine", gt_refs=[], kind="target", cohort="bridge")


def _pipe_cfg_stub():
    class _Cfg:
        mask_dilate_px = 12
        retry_budget = 3
        tau = 0.25
        crop_scorer = "siglip_so400m_384"
        #: Real PipelineConfig fields, added for the _build_bank/_run wiring
        #: tests below -- not needed by the earlier mask tests, but a second
        #: near-identical stub class would be the duplication this fixture
        #: exists to avoid.
        retriever = "siglip_so400m_384"
        eval_heldout_refs = 1
    return _Cfg()


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


def test_finalise_marks_a_passing_draft_healthy(tmp_path):
    """Scope extension: _finalise_one is the SECOND site that conflated
    'draft never needed repair' with 'draft failed and nothing beat it' --
    both wrote 'unrepaired'. _finalise runs unconditionally over every case
    at the end of the run, including ones _resolve already ended early, so
    this bug silently overwrote every early-stopped healthy case too."""
    q = Queue.open(tmp_path / "q.json", ["a"], retry_budget=3, screen_run="s")
    run = _Run(tmp_path)
    (run.case_dir("a") / "scores.json").write_text('{"draft": [true, 0.9]}')

    _finalise(q, run)

    assert q.cases["a"].best == "draft"
    assert q.cases["a"].status == "healthy"


def test_the_mask_stage_runs_once_across_every_round(tmp_path):
    """Design §4. Masking per round turns 1 model load into N for no gain."""
    q = Queue.open(tmp_path / "q.json", ["a"], retry_budget=3, screen_run="s")
    masked, loads = [], []

    for attempt in range(1, 4):
        #: mask is attempt-free, so the SAME cell is consulted every round.
        stage_with_model(q, "mask", lambda: loads.append("m") or "M",
                         lambda m, cid, n=attempt: masked.append((cid, n)))

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


def test_a_passing_attempt_ends_the_case_immediately(tmp_path):
    """Regression. The smoke run executed regen@1, regen@2 AND regen@3 for a
    case whose attempt_1 scored [true, 1.0] -- 15:45 to 18:01 -- because
    nothing moved the case off `pending` until _finalise, which runs after the
    last round. needs_round() was correct; nothing was calling through to it.
    """
    q = Queue.open(tmp_path / "q.json", ["a"], retry_budget=3, screen_run="s")
    assert q.needs_round(2)

    _resolve(q, "a", "attempt_1")

    assert q.cases["a"].status == "repaired"
    assert q.cases["a"].best == "attempt_1"
    assert not q.needs_round(2)


def test_a_passing_draft_also_ends_the_case(tmp_path):
    """Design §4: round 0's passing cases are DONE -- there is nothing to
    repair, so they must not enter the regen loop at all."""
    q = Queue.open(tmp_path / "q.json", ["a"], retry_budget=3, screen_run="s")

    _resolve(q, "a", "draft")

    assert q.cases["a"].best == "draft"
    #: Nothing was repaired; the draft was already right. "healthy", not
    #: "unrepaired" -- that word means the opposite: a draft that failed and
    #: could not be fixed (findings/2026-07-29-orchestration-result.md §2).
    assert q.cases["a"].status == "healthy"
    assert not q.needs_round(1)


def test_resolving_one_case_does_not_stop_the_others(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a", "b"], retry_budget=3,
                   screen_run="s")

    _resolve(q, "a", "attempt_1")

    assert q.unresolved() == ["b"]
    assert q.needs_round(2)


def _queue_stub(ids):
    q = Queue(Path("/dev/null"), {i: CaseState(case_id=i) for i in ids}, 3, "x")
    q.save = lambda: None
    return q


def test_a_draft_that_passed_is_healthy_not_unrepaired():
    """Two code paths wrote 'unrepaired' meaning opposite things
    (findings/2026-07-29-orchestration-result.md §2)."""
    q = _queue_stub(["fox"])
    _resolve(q, "fox", "draft")
    assert q.cases["fox"].status == "healthy"


def test_a_repair_that_worked_is_repaired():
    q = _queue_stub(["fox"])
    _resolve(q, "fox", "attempt_2")
    assert q.cases["fox"].status == "repaired"


def test_a_resolved_case_is_not_re_fused(fake_run_dir, monkeypatch):
    """A resolved case had its scores.json rewritten on every resume."""
    calls = []
    monkeypatch.setattr(rp, "_record_score",
                        lambda *a, **k: calls.append(a) or True)
    q = _queue_stub(["fox"])
    q.cases["fox"].status = "healthy"
    rp._fuse_pending(q, fake_run_dir, {"fox": None}, attempt=1)
    assert calls == []


def test_finish_results_counts_every_terminal_status():
    """A healthy case must not silently drop out of run.json's summary --
    the same conflation this task fixes elsewhere, in a third spot."""
    q = _queue_stub(["a", "b", "c", "d"])
    q.cases["a"].status = "healthy"
    q.cases["b"].status = "repaired"
    q.cases["c"].status = "unrepaired"
    q.cases["d"].status = "failed"
    assert rp._finish_results(q) == {
        "healthy": ["a"], "repaired": ["b"],
        "unrepaired": ["c"], "failed": ["d"],
    }


def test_rounds_clamp_to_the_edit_pool(tmp_path):
    """A 2-ref edit pool with retry_budget 3 must run 2 rounds, not 3.

    run_pipeline.py reads refs[attempt - 1] and _regen_one raises when the
    cutout is missing, so an unclamped budget marks a perfectly good case
    `failed` -- a hygiene fix that fabricates failures.
    """
    from scripts.run_pipeline import rounds_for
    assert rounds_for(retry_budget=3, n_edit_refs=2) == 2
    assert rounds_for(retry_budget=3, n_edit_refs=3) == 3
    assert rounds_for(retry_budget=2, n_edit_refs=5) == 2


def test_a_case_out_of_references_is_retired_not_failed(tmp_path):
    """A short edit pool must end the case, not manufacture a failure.

    This is the whole risk of the hold-out: _regen_one raises when
    cutout_<attempt>.png is absent, and run_stage turns that into
    `status: failed` -- a dataset with fewer references would read as a method
    that could not repair it.
    """
    import json

    from ragregen.schedule import Queue
    from scripts.run_pipeline import _retire_exhausted

    class _RunDir:
        def __init__(self, root):
            self.root = root

        def case_dir(self, cid):
            d = self.root / cid
            d.mkdir(parents=True, exist_ok=True)
            return d

    rd = _RunDir(tmp_path)
    q = Queue.open(tmp_path / "q.json", ["short"], retry_budget=3,
                   screen_run="s")
    (rd.case_dir("short") / "refs.json").write_text(json.dumps(["a", "b"]))
    (rd.case_dir("short") / "scores.json").write_text(
        json.dumps({"draft": [False, 0.0], "attempt_1": [False, 0.0],
                    "attempt_2": [False, 0.0]}))

    _retire_exhausted(q, rd, attempt=3)

    assert q.cases["short"].status != "failed"
    assert q.cases["short"].status == "unrepaired"
    assert q.cases["short"].best == "draft"


def test_a_short_card_aborts_before_any_weights_load(monkeypatch):
    import scripts.run_pipeline as rp

    loaded = []

    class _Q:
        saved = False

        def pending(self, stage, attempt=None):
            return ["a"]

        #: Load-bearing, not a no-op: stage_with_model calls queue.save()
        #: before raising StageAborted, so the abort leaves queue.json
        #: written and resume is correct (doc 5 §9). The test asserts this
        #: actually happened rather than merely tolerating the attribute.
        def save(self):
            self.saved = True

    q = _Q()
    monkeypatch.setattr(rp.env, "free_vram_gb", lambda i=0: 2.0)
    #: The whole point: `load` must never be called. An OOM partway through
    #: loading 16 GB of Qwen leaves the allocator worse off than a clean yield.
    with pytest.raises(rp.schedule.StageAborted, match="VRAM"):
        rp.stage_with_model(q, "verify", lambda: loaded.append(1),
                            lambda m, cid: None, need_gb=17.0)
    assert loaded == []
    assert q.saved is True


def test_an_unknown_card_is_treated_as_permission_to_proceed(monkeypatch):
    #: free_vram_gb returns None on a CPU-only machine. Blocking there would
    #: make every CPU test abort.
    import scripts.run_pipeline as rp

    ran = []

    class _Q:
        def pending(self, stage, attempt=None):
            return ["a"]

        def mark(self, *a, **k):
            pass

        def save(self):
            pass

    monkeypatch.setattr(rp.env, "free_vram_gb", lambda i=0: None)
    rp.stage_with_model(_Q(), "verify", lambda: "model",
                        lambda m, cid: ran.append(cid), need_gb=17.0)
    assert ran == ["a"]


def test_verify_round_pads_the_budget_for_a_live_embedder(fake_run_dir,
                                                           monkeypatch):
    """IMPORTANT 4: --proto-arm's embedder sits GPU-resident for the whole
    run, outside stage_with_model's load/free discipline -- it is genuinely
    used on EVERY grounded-stage call, every round (grounded.py:115-120), so
    it cannot just be freed once the bank is built. The preflight must count
    its footprint on top of each stage's own budget, or it under-counts what
    the card actually needs to hold.
    """
    calls = []
    monkeypatch.setattr(
        rp, "stage_with_model",
        lambda *a, attempt=None, need_gb=None: calls.append(need_gb))

    rp._verify_round(_queue_stub([]), fake_run_dir, {}, _pipe_cfg_stub(),
                     attempt=0, device="cpu", image_of=lambda cid: None,
                     bank=object(), embedder=object())

    assert calls == [rp.STAGE_VRAM_GB["grounded"] + rp.PROTO_EMBEDDER_VRAM_GB,
                     rp.STAGE_VRAM_GB["semantic"] + rp.PROTO_EMBEDDER_VRAM_GB]


def test_verify_round_uses_the_plain_budget_with_no_embedder(fake_run_dir,
                                                              monkeypatch):
    calls = []
    monkeypatch.setattr(
        rp, "stage_with_model",
        lambda *a, attempt=None, need_gb=None: calls.append(need_gb))

    rp._verify_round(_queue_stub([]), fake_run_dir, {}, _pipe_cfg_stub(),
                     attempt=0, device="cpu", image_of=lambda cid: None)

    assert calls == [rp.STAGE_VRAM_GB["grounded"], rp.STAGE_VRAM_GB["semantic"]]


def test_run_pads_the_mask_and_regen_budgets_for_a_live_embedder(
        fake_run_dir, monkeypatch, tmp_path):
    """The same embedder stays resident through `mask` and `regen` too --
    both run AFTER _build_bank, in the same process, for the rest of the
    run. Their preflight budget must include it as well."""
    screen_run = tmp_path / "screen"
    (screen_run / "fox_case").mkdir(parents=True)
    _blank_image().save(screen_run / "fox_case" / "draft.png")

    sentinel_bank, sentinel_embedder = object(), object()
    monkeypatch.setattr(
        rp, "_build_bank",
        lambda arm, ds, pipe_cfg, device: (sentinel_bank, sentinel_embedder))
    monkeypatch.setattr(rp, "_verify_round", lambda *a, **kw: None)
    monkeypatch.setattr(rp, "_mask_one", lambda *a, **kw: None)
    monkeypatch.setattr(rp, "_load_masker", lambda device: "MASKER")

    need_gbs = {}
    real_stage_with_model = rp.stage_with_model

    def spy(queue, stage, load, work, *, attempt=None, need_gb=None):
        need_gbs[stage] = need_gb
        return real_stage_with_model(queue, stage, load, work,
                                     attempt=attempt, need_gb=need_gb)

    monkeypatch.setattr(rp, "stage_with_model", spy)

    case = Case(id="fox_case", prompt="a fox in grass", concept="fox",
               coarse="canine", gt_refs=[Path("r1.png"), Path("r2.png")],
               kind="target", cohort="bridge")
    q = Queue.open(fake_run_dir.path / "q.json", [case.id], retry_budget=0,
                  screen_run=str(screen_run))

    import types
    args = types.SimpleNamespace(arm="oracle", device="cpu",
                                 proto_arm="retrieved", mask_prototypes=False,
                                 mechanism="inpaint")
    pipe_cfg = _pipe_cfg_stub()
    #: 0 -- keeps the regen round loop from running, so this test needs no
    #: fake for _load_regen/_regen_one and only asserts `mask`'s budget.
    pipe_cfg.retry_budget = 0

    rp._run(q, fake_run_dir, {case.id: case}, pipe_cfg, screen_run, args,
           object())

    assert need_gbs["mask"] == rp.STAGE_VRAM_GB["mask"] + rp.PROTO_EMBEDDER_VRAM_GB


def test_every_stage_that_runs_has_a_vram_budget():
    #: A STAGE_VRAM_GB key that names no real stage silently disables the
    #: preflight for it -- `want = STAGE_VRAM_GB.get(stage)` just returns
    #: None and the whole check is skipped, no error, no warning. That is
    #: exactly how "verify" (never a real stage name) sat as dead code while
    #: "semantic" and "grounded" -- the two heaviest loads, Qwen-7B and
    #: GroundingDINO -- ran with no preflight at all. This test makes that
    #: drift loud instead of silent.
    assert set(rp.STAGE_VRAM_GB) == set(rp.schedule.STAGES)


def test_grounded_stash_persists_similarities_not_just_states(fake_run_dir):
    """202 verdicts of GPU time were persisted as 202 bare strings."""
    from ragregen.verify.grounded import ConceptScore

    scores = {"fox": ConceptScore("fox", "subject", "MISSING", sim=0.12,
                                  sim_coarse=0.03)}
    rp._stash_grounded(fake_run_dir, "fox_case", 0, scores)
    data = json.loads(
        (fake_run_dir.case_dir("fox_case") / "streams.json").read_text())
    assert data["0"]["grounded"]["fox"]["sim"] == 0.12
    assert data["0"]["grounded"]["fox"]["state"] == "MISSING"


def test_record_score_reads_an_old_bare_string_streams_file(fake_run_dir):
    """A doc-5 run dir must stay resumable."""
    both = {"grounded": {"fox": "MISSING"},
            "semantic": {"ok": True, "degenerate": False}}
    ok = rp._record_score(fake_run_dir, "fox_case", "draft", both)
    assert ok is False
    scored = json.loads(
        (fake_run_dir.case_dir("fox_case") / "scores.json").read_text())
    assert scored["draft"] == [False, None]


def test_record_score_uses_the_real_similarity_when_it_is_there(fake_run_dir):
    both = {"grounded": {"fox": {"phrase": "fox", "kind": "subject",
                                 "state": "MISSING", "sim": 0.12,
                                 "candidates": []}},
            "semantic": {"ok": True, "degenerate": False}}
    rp._record_score(fake_run_dir, "fox_case", "draft", both)
    scored = json.loads(
        (fake_run_dir.case_dir("fox_case") / "scores.json").read_text())
    assert scored["draft"] == [False, 0.12]


def test_finalise_one_never_overwrites_a_failed_status(tmp_path):
    """IMPORTANT 5: a case that failed at `mask` or `regen` can still carry
    a round-0 scores.json (the draft's own verdict). _finalise_one used to
    set `status` unconditionally for any case with a scores.json, so a
    failed case got relabelled unrepaired/repaired and its `error` string
    was stranded -- undercounting run.json's `failed` list, which
    _finish_results' docstring promises cannot happen.
    """
    q = Queue.open(tmp_path / "q.json", ["a"], retry_budget=3, screen_run="s")
    run = _Run(tmp_path)
    (run.case_dir("a") / "scores.json").write_text('{"draft": [false, 0.8]}')
    q.mark("a", "mask", "failed", status="failed",
          error="ValueError: 'ship' did not ground in the draft")

    rp._finalise_one(q, run, "a")

    assert q.cases["a"].status == "failed"
    assert q.cases["a"].error == "ValueError: 'ship' did not ground in the draft"
    #: best is untouched too -- _finalise_one never reached the assignment.
    assert q.cases["a"].best is None


def test_open_loop_selects_only_a_completed_attempt_without_scores(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a", "b"], retry_budget=3,
                   screen_run="s")
    run = _Run(tmp_path)
    q.mark("a", "regen", "done", attempt=1)
    q.mark("b", "regen", "failed", attempt=1, status="failed",
           error="generation failed")

    rp._resolve_open_loop(q, run, 1)

    assert q.cases["a"].best == "attempt_1"
    assert q.cases["a"].status == "repaired"
    assert not (run.case_dir("a") / "scores.json").exists()
    selection = json.loads((run.case_dir("a") / "selection.json").read_text())
    assert selection == {"best": "attempt_1", "mode": "open_loop",
                         "verifier_used": False}
    assert q.cases["b"].status == "failed"


def test_finalise_one_survives_a_score_with_no_grounded_evidence(fake_run_dir):
    """Scope extension: _record_score can now write [ok, null] to
    scores.json, and _finalise_one's coercion (`float(draft_score)`,
    `float(v[1])`) used to crash on that. select_best already accepts
    float | None; the fix is to stop coercing it away.
    """
    q = Queue.open(fake_run_dir.path / "q.json", ["a"], retry_budget=3,
                   screen_run="s")
    (fake_run_dir.case_dir("a") / "scores.json").write_text(
        json.dumps({"draft": [False, None], "attempt_1": [False, None]}))

    rp._finalise_one(q, fake_run_dir, "a")

    assert q.cases["a"].best == "draft"
    assert q.cases["a"].status == "unrepaired"


def test_trace_json_load_then_append_round_trip_survives_two_writes(
        fake_run_dir):
    """rp._trace reads trace.json's `vlm`/`grounded`/`retrieval` keys back
    into CaseTrace.vlm_calls/grounded_scores/retrieval_hits, and
    CaseTrace.save writes them back out under `vlm`/`grounded`/`retrieval`
    -- the same three names, spelled independently in two files
    (run_pipeline._trace's dict.get calls and trace.CaseTrace.save's dict
    literal). Nothing else pins that the pairing survives an actual
    load-modify-save-load round trip rather than just each half in
    isolation.
    """
    path = fake_run_dir.case_dir("fox_case") / "trace.json"

    first = rp._trace(fake_run_dir, "fox_case")
    first.vlm("semantic@0", "first reply")
    first.save(path)

    second = rp._trace(fake_run_dir, "fox_case")
    second.vlm("semantic@1", "second reply")
    second.save(path)

    data = json.loads(path.read_text())
    assert [c["raw"] for c in data["vlm"]] == ["first reply", "second reply"]


def test_the_raw_vlm_reply_reaches_the_trace(fake_run_dir):
    rp._trace_vlm(fake_run_dir, "fox_case", 0,
                  '{"verdict": "FAIL", "issues": []}')
    data = json.loads(
        (fake_run_dir.case_dir("fox_case") / "trace.json").read_text())
    assert data["vlm"][0]["raw"] == '{"verdict": "FAIL", "issues": []}'
    assert data["vlm"][0]["stage"] == "semantic@0"


def test_a_garbage_reply_is_flagged_in_the_trace(fake_run_dir):
    """One doc-5 verdict was degenerate and nothing recorded what Qwen said."""
    from ragregen.trace import GARBAGE_SIGNATURE
    rp._trace_vlm(fake_run_dir, "duck", 0, GARBAGE_SIGNATURE)
    data = json.loads((fake_run_dir.case_dir("duck") / "trace.json").read_text())
    assert data["vlm"][0]["garbage"] is True


def test_semantic_stash_keeps_issues_and_raw(fake_run_dir):
    from ragregen.verify.semantic import Issue, SemanticVerdict
    v = SemanticVerdict(ok=False, raw="{...}", degenerate=False,
                        issues=[Issue(concept="fox", problem="wrong ears")])
    rp._stash_semantic(fake_run_dir, "fox_case", 0, v)
    data = json.loads(
        (fake_run_dir.case_dir("fox_case") / "streams.json").read_text())
    assert data["0"]["semantic"]["issues"] == [
        {"concept": "fox", "problem": "wrong ears"}]
    assert data["0"]["semantic"]["raw"] == "{...}"


def test_both_prototype_flags_default_off(monkeypatch):
    """A resumed doc-5 run must behave exactly as it did."""
    args = rp._parse_args(["--arm", "oracle"])
    assert args.proto_arm == "none"
    assert args.mask_prototypes is False


def test_mask_draft_gets_no_bank_when_the_flag_is_off(fake_run_dir, monkeypatch):
    seen = {}

    def _fake_mask_draft(image, coarse, masker, **kw):
        seen.update(kw)
        return None

    monkeypatch.setattr(rp.mask, "mask_draft", _fake_mask_draft)
    case = run_pipeline_case_stub()
    with pytest.raises(ValueError):
        rp._mask_one(object(), fake_run_dir, case, _blank_image(),
                     _pipe_cfg_stub(), prototypes=None)
    assert seen["prototypes"] is None


def test_mask_json_records_how_the_box_was_chosen(fake_run_dir, monkeypatch):
    """doc 1 §6 called selected_by the diagnostic that makes a masking failure
    separable from a generation failure. The pipeline never wrote it."""
    from ragregen.mask import MaskResult
    import numpy as np

    result = MaskResult(mask=np.ones((4, 4), dtype=float), box=(0, 0, 4, 4),
                        score=0.9, n_candidates=3, selected_by="confidence")
    monkeypatch.setattr(rp.mask, "mask_draft", lambda *a, **k: result)
    #: refs.json is "[]", so the reference-cutout loop below never runs --
    #: mask_reference is never called and needs no fake.
    case = run_pipeline_case_stub()
    (fake_run_dir.case_dir(case.id) / "refs.json").write_text("[]")
    rp._mask_one(object(), fake_run_dir, case, _blank_image(),
                _pipe_cfg_stub(), prototypes=None)
    meta = json.loads((fake_run_dir.case_dir(case.id) / "mask.json").read_text())
    assert meta == {"selected_by": "confidence", "n_candidates": 3,
                    "box": [0, 0, 4, 4], "score": 0.9}


def test_mask_stage_prepares_the_reference_for_inpainting(
        fake_run_dir, monkeypatch, tmp_path):
    import numpy as np
    from ragregen.mask import MaskResult

    draft_found = MaskResult(mask=np.ones((8, 8), dtype=float),
                             box=(0, 0, 8, 8), score=0.9,
                             n_candidates=1, selected_by="confidence")
    ref_arr = np.zeros((20, 20), dtype=float)
    ref_arr[5:15, 5:15] = 1.0
    ref_found = MaskResult(mask=ref_arr, box=(5, 5, 15, 15), score=0.8,
                           n_candidates=1, selected_by="confidence")
    monkeypatch.setattr(rp.mask, "mask_draft", lambda *a, **k: draft_found)
    monkeypatch.setattr(rp.mask, "mask_reference", lambda *a, **k: ref_found)

    ref_path = tmp_path / "ref.png"
    Image.new("RGB", (20, 20), "green").save(ref_path)
    case = run_pipeline_case_stub()
    d = fake_run_dir.case_dir(case.id)
    (d / "refs.json").write_text(json.dumps([str(ref_path)]))

    rp._mask_one(object(), fake_run_dir, case, _blank_image(),
                 _pipe_cfg_stub(), ref_prep="crop")

    prepared = Image.open(d / "reference_1.png")
    assert prepared.size == (18, 18), "object bbox plus the documented padding"
    records = json.loads((d / "references.json").read_text())
    assert records["1"]["applied"] == "crop"
    assert records["1"]["cutout_available"] is True


def test_mask_stage_rejects_ungrounded_refs_and_compacts_attempts(
        fake_run_dir, monkeypatch, tmp_path):
    import numpy as np
    from ragregen.mask import MaskResult

    draft_found = MaskResult(mask=np.ones((8, 8), dtype=float),
                             box=(0, 0, 8, 8), score=0.9,
                             n_candidates=1, selected_by="confidence")
    ref_arr = np.zeros((20, 20), dtype=float)
    ref_arr[5:15, 5:15] = 1.0
    ref_found = MaskResult(mask=ref_arr, box=(5, 5, 15, 15), score=0.8,
                           n_candidates=1, selected_by="confidence")
    monkeypatch.setattr(rp.mask, "mask_draft", lambda *a, **k: draft_found)
    found = iter([None, ref_found])
    monkeypatch.setattr(rp.mask, "mask_reference",
                        lambda *a, **k: next(found))

    bad, good = tmp_path / "bad.png", tmp_path / "good.png"
    Image.new("RGB", (20, 20), "red").save(bad)
    Image.new("RGB", (20, 20), "green").save(good)
    case = run_pipeline_case_stub()
    d = fake_run_dir.case_dir(case.id)
    (d / "refs.json").write_text(json.dumps([str(bad), str(good)]))

    rp._mask_one(object(), fake_run_dir, case, _blank_image(),
                 _pipe_cfg_stub(), ref_prep="crop")

    assert json.loads((d / "refs.json").read_text()) == [str(good)]
    assert (d / "reference_1.png").is_file()
    assert not (d / "reference_2.png").exists()
    records = json.loads((d / "references.json").read_text())
    assert records["1"]["source"] == str(good)
    assert records["_rejected"] == [{
        "source": str(bad),
        "reason": f"{case.concept!r} not grounded in reference",
    }]


def test_retrieve_one_records_context_query_scores_and_ranks(fake_run_dir):
    from ragregen.retrieve import Hit

    case = run_pipeline_case_stub()

    class Retriever:
        seen = None

        def search(self, query, k):
            self.seen = (query, k)
            return [Hit(Path("good.jpg"), 0.72, 0)]

    retriever = Retriever()
    rp._retrieve_one(retriever, fake_run_dir, case, 3)

    assert retriever.seen == (
        f"a real photograph of {case.concept}, a type of {case.coarse}, as the main subject",
        3)
    data = json.loads(
        (fake_run_dir.case_dir(case.id) / "retrieval.json").read_text())
    assert data["hits"] == [{"path": "good.jpg", "score": 0.72, "rank": 0}]


def test_inpainting_uses_prepared_reference_without_requiring_a_cutout(
        fake_run_dir):
    import numpy as np
    from types import SimpleNamespace
    from ragregen.regen import RegenResult

    case = run_pipeline_case_stub()
    d = fake_run_dir.case_dir(case.id)
    Image.new("L", (8, 8), 0).save(d / "mask.png")
    Image.new("RGB", (3, 5), "green").save(d / "reference_1.png")
    calls = []

    class Engine:
        def regen(self, draft, mask, reference, prompt):
            calls.append((reference.size, prompt))
            return RegenResult(
                image=Image.new("RGB", (8, 8), "blue"),
                raw=Image.new("RGB", (16, 16), "red"),
                alpha=np.zeros((8, 8), dtype=np.float32),
                mechanism="inpaint", meta={"seed": 0})

    rp._regen_one(Engine(), fake_run_dir, case, _blank_image(), 1,
                  SimpleNamespace(mechanism="inpaint"))

    assert calls[0][0] == (3, 5)
    assert "do not copy" in calls[0][1].lower()
    assert (d / "attempt_1.png").is_file()
    assert (d / "raw_attempt_1.png").is_file()
    assert (d / "alpha_1.png").is_file()
    assert json.loads((d / "regen_1.json").read_text())["meta"]["seed"] == 0


def test_mask_prototypes_without_an_arm_is_rejected(capsys):
    """`mask_bank = bank if args.mask_prototypes else None` would otherwise
    silently no-op: --mask-prototypes alone leaves `bank` at None (proto_arm
    defaults to "none"), so masking is unaffected and nothing says so. Run
    provenance is load-bearing evidence here -- fail before any model loads.
    """
    with pytest.raises(SystemExit) as exc:
        rp._parse_args(["--mask-prototypes"])
    assert exc.value.code == 2
    assert "requires --proto-arm" in capsys.readouterr().err


def test_mask_prototypes_with_an_arm_parses_cleanly():
    args = rp._parse_args(["--mask-prototypes", "--proto-arm", "retrieved"])
    assert args.mask_prototypes is True
    assert args.proto_arm == "retrieved"


def test_no_verifier_is_an_explicit_cli_arm():
    args = rp._parse_args(["--verifier", "none"])
    assert args.verifier == "none"


def test_open_loop_attempts_default_preserves_the_one_shot_arm():
    args = rp._parse_args(["--verifier", "none"])
    assert args.open_loop_attempts == 1


def test_open_loop_attempts_rejects_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        rp._parse_args(["--verifier", "none", "--open-loop-attempts", "0"])
    assert exc.value.code == 2
    assert "positive" in capsys.readouterr().err


def test_multiple_open_loop_attempts_require_no_verifier(capsys):
    with pytest.raises(SystemExit) as exc:
        rp._parse_args(["--verifier", "fused", "--open-loop-attempts", "2"])
    assert exc.value.code == 2
    assert "requires --verifier none" in capsys.readouterr().err


def test_candidate_bank_resolves_attempt_one_after_all_rounds(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a"], retry_budget=3,
                   screen_run="s")
    run = _Run(tmp_path)
    q.mark("a", "regen", "done", attempt=1)
    q.mark("a", "regen", "done", attempt=2)

    rp._resolve_candidate_bank(q, run)

    assert q.cases["a"].best == "attempt_1"
    assert q.cases["a"].status == "repaired"
    selection = json.loads((run.case_dir("a") / "selection.json").read_text())
    assert selection == {
        "best": "attempt_1", "mode": "open_loop",
        "verifier_used": False,
        "candidate_attempts": ["attempt_1", "attempt_2"],
    }


def test_candidate_bank_finishes_a_short_reference_case_before_next_round(
        tmp_path):
    q = Queue.open(tmp_path / "q.json", ["short", "long"], retry_budget=3,
                   screen_run="s")
    run = _Run(tmp_path)
    for case_id, refs in {"short": ["r1"], "long": ["r1", "r2"]}.items():
        (run.case_dir(case_id) / "refs.json").write_text(json.dumps(refs))
        q.mark(case_id, "regen", "done", attempt=1)

    rp._retire_open_loop_exhausted(q, run, attempt=2)

    assert q.cases["short"].best == "attempt_1"
    assert q.cases["short"].status == "repaired"
    assert q.cases["long"].status == "pending"


def test_multi_attempt_open_loop_never_invokes_a_verifier(
        fake_run_dir, monkeypatch, tmp_path):
    """Regression: the post-regen ``else`` used to bind to both
    ``verifier != none`` AND ``open_loop_attempts != 1``. Candidate-bank
    runs therefore loaded grounded + semantic after every cached attempt,
    despite run.json correctly recording verifier=none.
    """
    screen_run = tmp_path / "screen"
    (screen_run / "fox_case").mkdir(parents=True)
    _blank_image().save(screen_run / "fox_case" / "draft.png")
    refs = []
    for index in range(3):
        path = tmp_path / f"ref_{index}.png"
        _blank_image().save(path)
        refs.append(path)
    case = Case(id="fox_case", prompt="a fox in grass", concept="fox",
                coarse="canine", gt_refs=refs, kind="target",
                cohort="bridge")
    q = Queue.open(fake_run_dir.path / "q.json", [case.id], retry_budget=3,
                   screen_run=str(screen_run))

    monkeypatch.setattr(rp, "_build_bank", lambda *a: (None, None))
    monkeypatch.setattr(rp, "_load_masker", lambda device: "MASKER")
    monkeypatch.setattr(rp, "_mask_one", lambda *a, **kw: None)
    monkeypatch.setattr(rp, "_load_regen", lambda *a: "ENGINE")
    monkeypatch.setattr(rp, "_regen_one", lambda *a, **kw: None)

    def verifier_must_not_run(*args, **kwargs):
        raise AssertionError("open-loop candidate bank invoked verifier")

    monkeypatch.setattr(rp, "_verify_round", verifier_must_not_run)

    import types
    args = types.SimpleNamespace(
        arm="oracle", device="cpu", proto_arm="none",
        mask_prototypes=False, mechanism="inpaint", verifier="none",
        open_loop_attempts=2, ref_prep="crop")

    rp._run(q, fake_run_dir, {case.id: case}, _pipe_cfg_stub(), screen_run,
            args, object())

    assert q.cases[case.id].status == "repaired"
    assert q.cases[case.id].best == "attempt_1"
    assert q.cases[case.id].stages["regen@2"] == "done"
    assert not any(key.startswith(("grounded", "semantic"))
                   for key in q.cases[case.id].stages)


def test_build_bank_none_constructs_nothing(monkeypatch):
    """The default arm must not import or call any model constructor."""
    from ragregen import encoders

    calls = []
    monkeypatch.setattr(encoders, "build_encoder",
                        lambda *a, **k: calls.append("encoder"))
    monkeypatch.setattr(rp, "_load_retriever",
                        lambda *a, **k: calls.append("retriever"))

    bank, embedder = rp._build_bank("none", object(), _pipe_cfg_stub(), "cpu")

    assert (bank, embedder) == (None, None)
    assert calls == []


def test_build_bank_ceiling_uses_gt_refs(monkeypatch):
    """Ceiling arm: encoders.build_encoder feeds prototype.bank_from_gt_refs."""
    from ragregen import encoders
    from ragregen.verify import prototype

    fake_embedder = object()
    calls = {}

    def _fake_bank_from_gt_refs(ds, emb):
        calls["bank_from_gt_refs"] = (ds, emb)
        return "CEILING_BANK"

    monkeypatch.setattr(encoders, "build_encoder",
                        lambda name, device: fake_embedder)
    monkeypatch.setattr(prototype, "bank_from_gt_refs", _fake_bank_from_gt_refs)

    ds = object()
    bank, embedder = rp._build_bank("ceiling", ds, _pipe_cfg_stub(), "cpu")

    assert bank == "CEILING_BANK"
    assert embedder is fake_embedder
    assert calls["bank_from_gt_refs"] == (ds, fake_embedder)


def test_build_bank_retrieved_uses_the_retriever_and_its_encoder(monkeypatch):
    """Retrieved arm: goes through _load_retriever, and the bank is built
    from THAT retriever's own encoder -- never a second, possibly different,
    embedder."""
    from ragregen.verify import prototype

    fake_embedder = object()

    class _FakeRetriever:
        encoder = fake_embedder

    calls = {}

    def _fake_load_retriever(pipe_cfg, device):
        calls["load_retriever"] = (pipe_cfg, device)
        return _FakeRetriever()

    def _fake_bank_from_retrieval(ds, retriever, emb):
        calls["bank_from_retrieval"] = (ds, retriever, emb)
        return "RETRIEVED_BANK"

    monkeypatch.setattr(rp, "_load_retriever", _fake_load_retriever)
    monkeypatch.setattr(prototype, "bank_from_retrieval",
                        _fake_bank_from_retrieval)

    ds = object()
    cfg = _pipe_cfg_stub()
    bank, embedder = rp._build_bank("retrieved", ds, cfg, "cuda")

    assert bank == "RETRIEVED_BANK"
    assert embedder is fake_embedder
    assert calls["load_retriever"] == (cfg, "cuda")
    ret_ds, ret_retriever, ret_emb = calls["bank_from_retrieval"]
    assert ret_ds is ds
    assert isinstance(ret_retriever, _FakeRetriever)
    assert ret_emb is fake_embedder


@pytest.mark.parametrize("mask_prototypes", [False, True])
def test_run_only_uses_the_bank_for_masking_when_the_flag_is_set(
        fake_run_dir, monkeypatch, tmp_path, mask_prototypes):
    """--proto-arm records: GroundedVerifier gets the bank once an arm is
    requested, whether or not --mask-prototypes is set. --mask-prototypes is
    the one that changes results: mask_draft only gets the bank when the flag
    is set. Nothing else in this suite pins that distinction.
    """
    screen_run = tmp_path / "screen"
    (screen_run / "fox_case").mkdir(parents=True)
    _blank_image().save(screen_run / "fox_case" / "draft.png")

    sentinel_bank, sentinel_embedder = object(), object()
    monkeypatch.setattr(
        rp, "_build_bank",
        lambda arm, ds, pipe_cfg, device: (sentinel_bank, sentinel_embedder))
    #: A string, not a real Masker -- stage_with_model only needs something
    #: with no callable .free, and _mask_one is faked below so it is never
    #: asked to segment anything.
    monkeypatch.setattr(rp, "_load_masker", lambda device: "MASKER")

    verify_calls, mask_calls = [], []
    monkeypatch.setattr(rp, "_verify_round",
                        lambda *a, **kw: verify_calls.append(kw))
    monkeypatch.setattr(rp, "_mask_one",
                        lambda *a, **kw: mask_calls.append(kw))

    case = Case(id="fox_case", prompt="a fox in grass", concept="fox",
               coarse="canine", gt_refs=[Path("r1.png"), Path("r2.png")],
               kind="target", cohort="bridge")
    q = Queue.open(fake_run_dir.path / "q.json", [case.id], retry_budget=3,
                   screen_run=str(screen_run))

    #: SimpleNamespace, not a class with `mask_prototypes = mask_prototypes`
    #: -- a class body assigning a name equal to a captured closure variable
    #: shadows it for the whole body (NameError, not the outer value).
    import types
    args = types.SimpleNamespace(arm="oracle", device="cpu",
                                 proto_arm="retrieved",
                                 mask_prototypes=mask_prototypes,
                                 mechanism="inpaint")

    pipe_cfg = _pipe_cfg_stub()
    #: 0, not the stub's default 3 -- keeps rounds_for() at 0 regardless of
    #: the edit pool, so the regen/round-1 loop never runs and this test
    #: never needs to fake _load_regen or _regen_one.
    pipe_cfg.retry_budget = 0

    rp._run(q, fake_run_dir, {case.id: case}, pipe_cfg, screen_run, args,
           object())

    assert verify_calls[0]["bank"] is sentinel_bank
    assert verify_calls[0]["embedder"] is sentinel_embedder
    classifier = mask_calls[0]["prototypes"]
    if mask_prototypes:
        from ragregen.verify.prototype import CropClassifier

        assert isinstance(classifier, CropClassifier)
        assert classifier.bank is sentinel_bank
        assert classifier.embedder is sentinel_embedder
    else:
        assert classifier is None
