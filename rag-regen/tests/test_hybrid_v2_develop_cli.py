"""Task 4: CPU-only development CLI and immutable outputs.

The CLI joins already-exposed artifacts, verifies their hashes against the frozen
manifest, and emits a complete development report. A frozen policy is written
only when the numeric readiness gates all pass; a FAIL study must never leave one
behind.
"""
import json

import pytest

from ragregen import eval_manifest
from scripts.hybrid_v2_develop import main as develop_main


def _score(text, reference):
    return {"text_relevance": text, "reference_relevance": reference}


def _write_artifacts(tmp_path, reranker, semantic, dino, manifest_cases,
                     protocol="reconstructed_after_score"):
    semantic_p = tmp_path / "semantic.json"
    reranker_p = tmp_path / "reranker.json"
    dino_p = tmp_path / "dino.json"
    semantic_p.write_text(json.dumps(semantic))
    reranker_p.write_text(json.dumps(reranker))
    dino_p.write_text(json.dumps(dino))
    manifest = {
        "schema": 1,
        "protocol_status": protocol,
        "dataset": "fixture",
        "counts": {
            "detector_rows": len(manifest_cases),
            "selector_eligible": sum(1 for c in manifest_cases.values()
                                     if c["selector_eligible"]),
            "excluded": sum(1 for c in manifest_cases.values()
                            if not c["selector_eligible"]),
        },
        "sources": {
            "semantic": {"path": str(semantic_p),
                         "sha256": eval_manifest.sha256_file(semantic_p)},
            "reranker": {"path": str(reranker_p),
                         "sha256": eval_manifest.sha256_file(reranker_p)},
            "dino": {"path": str(dino_p),
                     "sha256": eval_manifest.sha256_file(dino_p)},
        },
        "cases": manifest_cases,
    }
    manifest_p = tmp_path / "manifest.json"
    manifest_p.write_text(json.dumps(manifest))
    return manifest_p, semantic_p, reranker_p, dino_p


def _clean_fixture():
    """Cleanly separable: v2 passes every numeric gate under leave-one-out."""
    reranker = {"cases": {}}
    semantic = {}
    dino = {"cases": {}}
    cases = {}
    for i in range(4):  # rare FAILs
        cid = f"rare_{i}"
        reranker["cases"][cid] = {"truth": "FAIL", "scores": {
            "draft": _score(0.10, 0.10),
            "attempt_1": _score(0.60, 0.50),
            "attempt_2": _score(0.11, 0.09)}}
        semantic[cid] = {"ok": False}
        dino["cases"][cid] = {"draft": 0.0, "attempt_1": 0.20,
                              "attempt_2": -0.05}
        cases[cid] = {"identity_truth": "FAIL", "selector_eligible": True,
                      "eligibility_reason": "eligible", "cohort": "bridge"}
    for i in range(4):  # common PASS controls
        cid = f"ctrl_{i}"
        reranker["cases"][cid] = {"truth": "PASS", "scores": {
            "draft": _score(0.80, 0.80),
            "attempt_1": _score(0.82, 0.81)}}
        semantic[cid] = {"ok": True}
        dino["cases"][cid] = {"draft": 0.0, "attempt_1": 0.03}
        cases[cid] = {"identity_truth": "PASS", "selector_eligible": True,
                      "eligibility_reason": "eligible", "cohort": "common"}
    return reranker, semantic, dino, cases


def _failing_fixture():
    """Selector disagrees with DINO: sign accuracy and harmful gates fail."""
    reranker, semantic, dino, cases = _clean_fixture()
    for i in range(4):  # relevance says repair, DINO says it hurt
        dino["cases"][f"rare_{i}"] = {"draft": 0.0, "attempt_1": -0.10,
                                      "attempt_2": 0.05}
    return reranker, semantic, dino, cases


def test_pass_study_writes_frozen_policy_and_hash(tmp_path):
    reranker, semantic, dino, cases = _clean_fixture()
    manifest_p, sem_p, rer_p, dino_p = _write_artifacts(
        tmp_path, reranker, semantic, dino, cases)
    out = tmp_path / "report"

    rc = develop_main(["--manifest", str(manifest_p), "--semantic", str(sem_p),
                       "--reranker", str(rer_p), "--dino", str(dino_p),
                       "--output-dir", str(out)])

    assert rc == 0
    report = json.loads((out / "development.json").read_text())
    assert report["readiness"] == "PASS"
    frozen = out / "frozen_policy.json"
    assert frozen.is_file()
    import hashlib
    assert (out / "frozen_policy.sha256").read_text().split()[0] == \
        hashlib.sha256(frozen.read_bytes()).hexdigest()


def test_fail_study_writes_no_frozen_policy(tmp_path):
    reranker, semantic, dino, cases = _failing_fixture()
    manifest_p, sem_p, rer_p, dino_p = _write_artifacts(
        tmp_path, reranker, semantic, dino, cases)
    out = tmp_path / "report"

    rc = develop_main(["--manifest", str(manifest_p), "--semantic", str(sem_p),
                       "--reranker", str(rer_p), "--dino", str(dino_p),
                       "--output-dir", str(out)])

    assert rc == 0
    report = json.loads((out / "development.json").read_text())
    assert report["readiness"] == "FAIL"
    assert not (out / "frozen_policy.json").exists()


def test_report_contains_required_sections(tmp_path):
    reranker, semantic, dino, cases = _failing_fixture()
    manifest_p, sem_p, rer_p, dino_p = _write_artifacts(
        tmp_path, reranker, semantic, dino, cases)
    out = tmp_path / "report"
    develop_main(["--manifest", str(manifest_p), "--semantic", str(sem_p),
                  "--reranker", str(rer_p), "--dino", str(dino_p),
                  "--output-dir", str(out)])

    md = (out / "development.md").read_text()
    assert "development" in md.lower()
    assert "reconstructed_after_score" in md
    for token in ("semantic_only", "reference_only", "v1_union"):
        assert token in md
    assert "wilson" in md.lower()
    assert "denominator" in md.lower() or "eligible" in md.lower()
    assert (out / "folds.csv").is_file()


def test_hash_mismatch_raises(tmp_path):
    reranker, semantic, dino, cases = _clean_fixture()
    manifest_p, sem_p, rer_p, dino_p = _write_artifacts(
        tmp_path, reranker, semantic, dino, cases)
    sem_p.write_text(json.dumps({"tampered": True}))  # break the hash
    out = tmp_path / "report"

    with pytest.raises(ValueError):
        develop_main(["--manifest", str(manifest_p), "--semantic", str(sem_p),
                      "--reranker", str(rer_p), "--dino", str(dino_p),
                      "--output-dir", str(out)])


def test_refuses_nonempty_output_dir(tmp_path):
    reranker, semantic, dino, cases = _clean_fixture()
    manifest_p, sem_p, rer_p, dino_p = _write_artifacts(
        tmp_path, reranker, semantic, dino, cases)
    out = tmp_path / "report"
    out.mkdir()
    (out / "stale.txt").write_text("old")

    with pytest.raises(FileExistsError):
        develop_main(["--manifest", str(manifest_p), "--semantic", str(sem_p),
                      "--reranker", str(rer_p), "--dino", str(dino_p),
                      "--output-dir", str(out)])
