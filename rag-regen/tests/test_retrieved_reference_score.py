"""Orchestration of the retrieved-reference scoring script.

The GPU residencies are injected as fakes; these tests pin the recovery
contract: retrieval.json is an immutable artifact that is *reused* (never
refused) so a reranker/parity failure can be resumed, while the final
retrieved_reranker.json is the only overwrite-guarded output. Oracle parity gates
the comparison.
"""
import json

import pytest

from scripts import retrieved_reference_score as rrs


def _oracle(tmp_path):
    oracle = {"schema": 1, "reference_source": "first_edit_gt_ref", "cases": {
        "a": {"truth": "FAIL", "concept": "a", "coarse": "x",
              "scores": {"draft": {"text_relevance": 0.30,
                                   "reference_relevance": 0.08}}},
        "b": {"truth": "PASS", "concept": "b", "coarse": "x",
              "scores": {"draft": {"text_relevance": 0.80,
                                   "reference_relevance": 0.79}}}}}
    p = tmp_path / "oracle.json"
    p.write_text(json.dumps(oracle))
    return p


def _retrieval():
    return {"provenance": {"index_sha256": "idx"},
            "cases": {
                "a": {"query": "q-a", "top": {"path": "/laion/a.jpg",
                      "score": 0.1, "rank": 0}, "image_sha256": "ha",
                      "contaminated": False},
                "b": {"query": "q-b", "top": {"path": "/laion/b.jpg",
                      "score": 0.1, "rank": 0}, "image_sha256": "hb",
                      "contaminated": False}}}


def _args(tmp_path, oracle_p, **extra):
    out = tmp_path / "retrieved_reranker.json"
    retrieval_out = tmp_path / "retrieval.json"
    base = ["--dataset", "x", "--screen-run", "x", "--candidate-run", "x",
            "--oracle-reranker", str(oracle_p), "--out", str(out),
            "--retrieval-out", str(retrieval_out)]
    for k, v in extra.items():
        base += [f"--{k}", str(v)] if v is not True else [f"--{k}"]
    return base, out, retrieval_out


def _good_rescore(retrieved):
    # oracle re-score reproduces the stored oracle numbers -> parity holds
    def fn(args, retrieval):
        results = {
            "a": {"reference_relevance": 0.20, "oracle_reference_relevance": 0.08},
            "b": {"reference_relevance": 0.77, "oracle_reference_relevance": 0.79}}
        return results, {"sentence_transformers": "5.4.0", "torch": "x"}
    return fn


def test_fresh_run_writes_retrieval_then_output(tmp_path):
    oracle_p = _oracle(tmp_path)
    argv, out, retrieval_out = _args(tmp_path, oracle_p)
    rrs.main(argv, retrieve_fn=lambda a: _retrieval(),
             rescore_fn=_good_rescore(_retrieval()))
    assert retrieval_out.is_file()
    art = json.loads(out.read_text())
    assert art["package_versions"]["sentence_transformers"] == "5.4.0"
    assert art["oracle_parity"]["max_abs_diff"] <= art["oracle_parity"]["atol"]
    assert art["cases"]["a"]["scores"]["draft"]["reference_relevance"] == 0.20
    assert art["cases"]["a"]["scores"]["draft"]["text_relevance"] == 0.30


def test_resume_reuses_existing_retrieval_without_reretrieving(tmp_path):
    oracle_p = _oracle(tmp_path)
    argv, out, retrieval_out = _args(tmp_path, oracle_p)
    retrieval_out.write_text(json.dumps(_retrieval()))

    def boom(args):
        raise AssertionError("retrieval must not run when retrieval.json exists")

    rrs.main(argv, retrieve_fn=boom, rescore_fn=_good_rescore(_retrieval()))
    assert out.is_file()


def test_retrieval_only_stops_before_reranker(tmp_path):
    oracle_p = _oracle(tmp_path)
    argv, out, retrieval_out = _args(tmp_path, oracle_p, **{"retrieval-only": True})

    def boom(args, retrieval):
        raise AssertionError("reranker must not run in retrieval-only mode")

    rrs.main(argv, retrieve_fn=lambda a: _retrieval(), rescore_fn=boom)
    assert retrieval_out.is_file()
    assert not out.exists()


def test_refuses_to_overwrite_final_output(tmp_path):
    oracle_p = _oracle(tmp_path)
    argv, out, retrieval_out = _args(tmp_path, oracle_p)
    out.write_text("sentinel")
    with pytest.raises(FileExistsError):
        rrs.main(argv, retrieve_fn=lambda a: _retrieval(),
                 rescore_fn=_good_rescore(_retrieval()))
    assert out.read_text() == "sentinel"


def test_parity_failure_stops_without_writing_output(tmp_path):
    oracle_p = _oracle(tmp_path)
    argv, out, retrieval_out = _args(tmp_path, oracle_p)

    def bad(args, retrieval):
        results = {
            "a": {"reference_relevance": 0.20, "oracle_reference_relevance": 0.50},
            "b": {"reference_relevance": 0.77, "oracle_reference_relevance": 0.79}}
        return results, {"sentence_transformers": "5.4.0"}

    with pytest.raises(SystemExit):
        rrs.main(argv, retrieve_fn=lambda a: _retrieval(), rescore_fn=bad)
    assert not out.exists()
