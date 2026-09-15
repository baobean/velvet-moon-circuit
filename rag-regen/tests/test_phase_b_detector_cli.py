import json

from ragregen.c1 import Confusion
from scripts.phase_b_detector import main

THRESHOLDS = {"text": 0.5, "retrieved_reference": 0.25048828125}
#: The policy freezes the full HF repo id; the scoring artifact records the
#: short slug ("qwen3-vl-reranker-2b") -- see main()'s normalised comparison.
RERANKER = {"model_id": "Qwen/Qwen3-VL-Reranker-2B", "prompt": "P"}


def _c(tp, fp, tn, fn):
    return Confusion(tp=tp, fp=fp, tn=tn, fn=fn)


def _inputs(tmp_path, coverage_ok=True):
    rare = [f"f{i}" for i in range(24)]
    control = [f"p{i}" for i in range(24)]
    references = {cid: {"path": f"/ref/{cid}.png", "image_sha256": f"sha-{cid}",
                        "contaminated": False}
                 for cid in rare + control}

    holdout = {
        "schema": 1, "protocol_status": "frozen_before_score",
        "cohorts": {"rare": {"cases": rare, "backfilled": []},
                   "control": {"cases": control, "backfilled": []}},
        "dino_reserves": {}, "references": references, "sources": {},
    }
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(json.dumps(holdout))

    policy = {
        "manifest": {"protocol_status": "frozen_before_score",
                    "thresholds": THRESHOLDS, "reranker": RERANKER},
        "policy_sha256": "irrelevant-for-this-test",
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))

    text = 0.9 if coverage_ok else float("nan")
    reranker_cases = {
        cid: {"scores": {"draft": {"text_relevance": text, "reference_relevance": 0.9}},
             "retrieval": {"image_sha256": references[cid]["image_sha256"]},
             "contaminated": False}
        for cid in rare + control
    }
    # The real artifact records the short slug, not the full repo id.
    retrieved = {"model": "qwen3-vl-reranker-2b", "prompt": RERANKER["prompt"],
                "cases": reranker_cases}
    retrieved_path = tmp_path / "retrieved_reranker.json"
    retrieved_path.write_text(json.dumps(retrieved))

    stream_b = {cid: {"ok": True, "degenerate": False} for cid in rare + control}
    stream_b_path = tmp_path / "stream_b.json"
    stream_b_path.write_text(json.dumps(stream_b))

    out = tmp_path / "detector"
    argv = ["--holdout-manifest", str(holdout_path),
           "--policy-manifest", str(policy_path),
           "--retrieved-reranker", str(retrieved_path),
           "--stream-b", str(stream_b_path),
           "--out-dir", str(out)]
    return argv, out


def test_pass_writes_proceed_flag(tmp_path):
    argv, out = _inputs(tmp_path)
    assert main(argv, apply_fn=lambda rows, t: {"confusion": {"rare": _c(20, 0, 0, 4), "control": _c(0, 1, 23, 0)}, "cases": {}},
                gate_fn=None) == 0
    rep = json.loads((out / "detector.json").read_text())
    assert rep["readiness"] == "PASS" and rep["proceed_to_generation"] is True
    assert "PASS" in (out / "detector.md").read_text()


def test_fail_writes_no_proceed_flag(tmp_path):
    argv, out = _inputs(tmp_path)
    main(argv, apply_fn=lambda rows, t: {"confusion": {"rare": _c(10, 0, 0, 14), "control": _c(0, 5, 19, 0)}, "cases": {}}, gate_fn=None)
    rep = json.loads((out / "detector.json").read_text())
    assert rep["readiness"] == "FAIL" and rep.get("proceed_to_generation") is not True
