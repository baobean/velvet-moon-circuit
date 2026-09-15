"""CLI: retrieved-reference detector-only development study."""
import json

import pytest

from ragregen import eval_manifest
from scripts.retrieved_detector_develop import main as develop_main


def _sc(text, ref):
    return {"scores": {"draft": {"text_relevance": text,
                                 "reference_relevance": ref}}}


def _fixtures(tmp_path, *, collapse=0, contaminate=None):
    manifest_cases, semantic = {}, {}
    oracle = {"schema": 1, "reference_source": "first_edit_gt_ref", "cases": {}}
    retrieved = {"schema": 1, "reference_source": "retrieved_laion", "cases": {}}
    for i in range(6):  # rare FAILs
        cid = f"f{i}"
        manifest_cases[cid] = {"identity_truth": "FAIL",
                               "selector_eligible": True,
                               "eligibility_reason": "eligible",
                               "cohort": "bridge"}
        # Half are semantic-missed, so the reference conjunction is load-bearing
        # for recall -- the fitter cannot just disable it to avoid FPs.
        semantic_catches = i < 3
        semantic[cid] = {"ok": not semantic_catches}
        oracle["cases"][cid] = {"truth": "FAIL", "concept": cid, "coarse": "x",
                                **_sc(0.10 + 0.01 * i, 0.05)}
        retrieved["cases"][cid] = {"truth": "FAIL", **_sc(0.10 + 0.01 * i, 0.06),
                                   "contaminated": False,
                                   "retrieval": {"path": f"/laion/{cid}.jpg"}}
    for i in range(6):  # common PASS controls
        cid = f"p{i}"
        manifest_cases[cid] = {"identity_truth": "PASS",
                               "selector_eligible": True,
                               "eligibility_reason": "eligible",
                               "cohort": "common"}
        semantic[cid] = {"ok": True}
        oracle["cases"][cid] = {"truth": "PASS", "concept": cid, "coarse": "x",
                                **_sc(0.80, 0.80)}
        # collapse: first `collapse` controls score low on retrieved ref -> FP
        low = i < collapse
        retrieved["cases"][cid] = {
            "truth": "PASS",
            **_sc(0.10 if low else 0.80, 0.02 if low else 0.80),
            "contaminated": (cid == contaminate),
            "retrieval": {"path": f"/laion/{cid}.jpg"}}

    semantic_p = tmp_path / "semantic.json"
    oracle_p = tmp_path / "oracle.json"
    retrieved_p = tmp_path / "retrieved.json"
    semantic_p.write_text(json.dumps(semantic))
    oracle_p.write_text(json.dumps(oracle))
    retrieved_p.write_text(json.dumps(retrieved))
    manifest = {"schema": 1, "protocol_status": "reconstructed_after_score",
                "cases": manifest_cases,
                "sources": {
                    "semantic": {"path": str(semantic_p),
                                 "sha256": eval_manifest.sha256_file(semantic_p)},
                    "reranker": {"path": str(oracle_p),
                                 "sha256": eval_manifest.sha256_file(oracle_p)}}}
    manifest_p = tmp_path / "manifest.json"
    manifest_p.write_text(json.dumps(manifest))
    return manifest_p, semantic_p, oracle_p, retrieved_p


def _run(tmp_path, out, mp, sp, op, rp):
    return develop_main(["--manifest", str(mp), "--semantic", str(sp),
                         "--oracle-reranker", str(op),
                         "--retrieved-reranker", str(rp),
                         "--output-dir", str(out)])


def test_pass_freezes_detector_policy_with_hash(tmp_path):
    mp, sp, op, rp = _fixtures(tmp_path)
    out = tmp_path / "report"
    assert _run(tmp_path, out, mp, sp, op, rp) == 0
    report = json.loads((out / "development.json").read_text())
    assert report["readiness"] == "PASS"
    policy = out / "frozen_detector_policy.json"
    assert policy.is_file()
    import hashlib
    assert (out / "frozen_detector_policy.sha256").read_text().split()[0] == \
        hashlib.sha256(policy.read_bytes()).hexdigest()


def test_oracle_dependence_fails_without_freezing(tmp_path):
    mp, sp, op, rp = _fixtures(tmp_path, collapse=3)
    out = tmp_path / "report"
    assert _run(tmp_path, out, mp, sp, op, rp) == 0
    report = json.loads((out / "development.json").read_text())
    assert report["readiness"] == "FAIL"
    assert not (out / "frozen_detector_policy.json").exists()
    # the oracle baseline should still look clean, isolating the reference shift
    assert report["detector"]["baselines"]["oracle_v2"]["fp"] == 0


def test_contamination_is_inconclusive_no_freeze(tmp_path):
    mp, sp, op, rp = _fixtures(tmp_path, contaminate="p0")
    out = tmp_path / "report"
    assert _run(tmp_path, out, mp, sp, op, rp) == 0
    report = json.loads((out / "development.json").read_text())
    assert report["readiness"] == "INCONCLUSIVE"
    assert "p0" in report["contaminated"]
    assert not (out / "frozen_detector_policy.json").exists()


def test_report_has_baselines_and_folds_csv(tmp_path):
    mp, sp, op, rp = _fixtures(tmp_path)
    out = tmp_path / "report"
    _run(tmp_path, out, mp, sp, op, rp)
    md = (out / "development.md").read_text().lower()
    for token in ("oracle_v2", "semantic_only", "retrieved_reference_only",
                  "wilson"):
        assert token in md
    assert (out / "folds.csv").is_file()


def test_refuses_nonempty_output_dir(tmp_path):
    mp, sp, op, rp = _fixtures(tmp_path)
    out = tmp_path / "report"
    out.mkdir()
    (out / "stale").write_text("x")
    with pytest.raises(FileExistsError):
        _run(tmp_path, out, mp, sp, op, rp)


def test_hash_mismatch_raises(tmp_path):
    mp, sp, op, rp = _fixtures(tmp_path)
    sp.write_text(json.dumps({"tampered": True}))
    out = tmp_path / "report"
    with pytest.raises(ValueError):
        _run(tmp_path, out, mp, sp, op, rp)
