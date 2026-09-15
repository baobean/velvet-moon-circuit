"""Task 1: immutable identity/eligibility manifest.

The load-bearing protocol invariant under test: an invalid mask changes only
`selector_eligible`/`eligibility_reason`, never `identity_truth`.
"""
import json

import pytest

from ragregen import config
from ragregen.eval_manifest import build_manifest, sha256_file


def _case(cid, cohort="bridge", kind="target"):
    return config.Case(
        id=cid, prompt=f"a {cid}", concept=cid.replace("_", " ").title(),
        coarse="thing", kind=kind, cohort=cohort, gt_refs=["/dev/null"])


def _dataset(*cases):
    return config.DatasetConfig(
        name="t", images_root="/", cases=list(cases), coarse_refs={})


def _queue(candidate_run, states):
    (candidate_run / "queue.json").write_text(
        json.dumps({"schema": 1, "cases": states}))


def test_mask_failure_preserves_identity_truth(tmp_path):
    screen = tmp_path / "screen"
    screen.mkdir()
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    _queue(candidates, {"rare": {"stages": {"retrieve": "done",
                                            "mask": "failed"},
                                 "status": "failed"}})

    manifest = build_manifest(
        _dataset(_case("rare")), {"rare": "FAIL"}, screen, candidates,
        protocol_status="reconstructed_after_score", source_paths={})

    row = manifest["cases"]["rare"]
    assert row["identity_truth"] == "FAIL"
    assert row["selector_eligible"] is False
    assert row["eligibility_reason"] == "mask_not_grounded"


def test_eligible_case_has_mask_and_at_least_one_attempt(tmp_path):
    screen = tmp_path / "screen"
    screen.mkdir()
    candidates = tmp_path / "candidates"
    case_dir = candidates / "good"
    case_dir.mkdir(parents=True)
    (case_dir / "mask.png").write_bytes(b"m")
    (case_dir / "attempt_1.png").write_bytes(b"a")
    _queue(candidates, {"good": {"stages": {"mask": "done"},
                                 "status": "repaired"}})

    manifest = build_manifest(
        _dataset(_case("good")), {"good": "PASS"}, screen, candidates,
        protocol_status="reconstructed_after_score", source_paths={})

    row = manifest["cases"]["good"]
    assert row["selector_eligible"] is True
    assert row["eligibility_reason"] == "eligible"


def test_missing_attempts_is_no_prepared_reference(tmp_path):
    screen = tmp_path / "screen"
    screen.mkdir()
    candidates = tmp_path / "candidates"
    case_dir = candidates / "bare"
    case_dir.mkdir(parents=True)
    (case_dir / "mask.png").write_bytes(b"m")
    _queue(candidates, {"bare": {"stages": {"mask": "done"},
                                 "status": "repaired"}})

    manifest = build_manifest(
        _dataset(_case("bare")), {"bare": "FAIL"}, screen, candidates,
        protocol_status="reconstructed_after_score", source_paths={})

    row = manifest["cases"]["bare"]
    assert row["selector_eligible"] is False
    assert row["eligibility_reason"] == "no_prepared_reference"


def test_missing_mask_png_despite_done_stage_is_corrupt(tmp_path):
    screen = tmp_path / "screen"
    screen.mkdir()
    candidates = tmp_path / "candidates"
    (candidates / "gone").mkdir(parents=True)
    _queue(candidates, {"gone": {"stages": {"mask": "done"},
                                 "status": "repaired"}})

    manifest = build_manifest(
        _dataset(_case("gone")), {"gone": "PASS"}, screen, candidates,
        protocol_status="reconstructed_after_score", source_paths={})

    row = manifest["cases"]["gone"]
    assert row["selector_eligible"] is False
    assert row["eligibility_reason"] == "corrupt_artifact"


def test_counts_split_detector_from_selector(tmp_path):
    screen = tmp_path / "screen"
    screen.mkdir()
    candidates = tmp_path / "candidates"
    good = candidates / "good"
    good.mkdir(parents=True)
    (good / "mask.png").write_bytes(b"m")
    (good / "attempt_1.png").write_bytes(b"a")
    (candidates / "bad").mkdir()
    _queue(candidates, {
        "good": {"stages": {"mask": "done"}, "status": "repaired"},
        "bad": {"stages": {"mask": "failed"}, "status": "failed"}})

    manifest = build_manifest(
        _dataset(_case("good"), _case("bad")),
        {"good": "PASS", "bad": "FAIL"}, screen, candidates,
        protocol_status="reconstructed_after_score", source_paths={})

    assert manifest["counts"]["detector_rows"] == 2
    assert manifest["counts"]["selector_eligible"] == 1
    assert manifest["counts"]["excluded"] == 1


def test_identity_ids_must_match_dataset_exactly(tmp_path):
    screen = tmp_path / "screen"
    screen.mkdir()
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    _queue(candidates, {"a": {"stages": {"mask": "failed"}}})

    with pytest.raises(ValueError):
        build_manifest(
            _dataset(_case("a")), {"a": "FAIL", "b": "PASS"}, screen,
            candidates, protocol_status="reconstructed_after_score",
            source_paths={})


def test_only_pass_fail_unjudgeable_allowed(tmp_path):
    screen = tmp_path / "screen"
    screen.mkdir()
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    _queue(candidates, {"a": {"stages": {"mask": "failed"}}})

    with pytest.raises(ValueError):
        build_manifest(
            _dataset(_case("a")), {"a": "EXCLUDE"}, screen, candidates,
            protocol_status="reconstructed_after_score", source_paths={})


def test_source_paths_are_hashed(tmp_path):
    screen = tmp_path / "screen"
    screen.mkdir()
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    _queue(candidates, {"a": {"stages": {"mask": "failed"}}})
    src = tmp_path / "reranker.json"
    src.write_text('{"x": 1}')

    manifest = build_manifest(
        _dataset(_case("a")), {"a": "FAIL"}, screen, candidates,
        protocol_status="reconstructed_after_score",
        source_paths={"reranker": src})

    assert manifest["sources"]["reranker"]["sha256"] == sha256_file(src)
    assert manifest["sources"]["reranker"]["path"] == str(src)


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello world")
    assert sha256_file(p) == hashlib.sha256(b"hello world").hexdigest()


def test_cli_refuses_to_overwrite_manifest(tmp_path):
    from scripts.freeze_eval_manifest import main as freeze_main

    out = tmp_path / "manifest.json"
    out.write_text("sentinel")
    # Point every path arg at a placeholder; refusal must happen before any of
    # them is read, so their contents never matter.
    placeholder = tmp_path / "placeholder"
    placeholder.write_text("{}")
    args = [
        "--dataset", str(placeholder),
        "--identity-source", str(placeholder),
        "--identity-format", "reranker_truth",
        "--screen-run", str(tmp_path),
        "--candidate-run", str(tmp_path),
        "--semantic", str(placeholder),
        "--reranker", str(placeholder),
        "--dino", str(placeholder),
        "--protocol-status", "reconstructed_after_score",
        "--out", str(out),
    ]
    with pytest.raises(FileExistsError):
        freeze_main(args)
    assert out.read_text() == "sentinel"
