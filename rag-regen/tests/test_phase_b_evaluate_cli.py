import json

from scripts.phase_b_evaluate import main

RARE = [f"f{i}" for i in range(24)]
CONTROL = [f"p{i}" for i in range(24)]
ALL_IDS = RARE + CONTROL


def _arm_out(cid):
    return {"path": f"/o/{cid}.png", "generation_failed": False}


def _setup(tmp_path, *, clean=True, arms_status="complete"):
    if arms_status == "complete":
        arms = {"status": "complete", "cases": {
            cid: {"cohort": "rare" if cid in RARE else "control",
                 "arm1": _arm_out(cid), "arm2": _arm_out(cid), "arm3": _arm_out(cid)}
            for cid in ALL_IDS}}
    else:
        arms = {"status": arms_status, "reason": "routed attempt failed"}
    arms_path = tmp_path / "arms.json"
    arms_path.write_text(json.dumps(arms))

    holdout = {
        "schema": 1, "protocol_status": "frozen_before_score",
        "cohorts": {"rare": {"cases": RARE, "backfilled": []},
                   "control": {"cases": CONTROL, "backfilled": []}},
        "dino_reserves": {cid: [f"/reserve/{cid}.png"] for cid in ALL_IDS},
        "references": {}, "sources": {},
    }
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(json.dumps(holdout))

    # Non-thin: recall and FP both strictly clear of the discrete boundary.
    detector = {"readiness": "PASS", "recall_k": [20, 24], "fp_k": [1, 24], "gates": []}
    detector_path = tmp_path / "detector.json"
    detector_path.write_text(json.dumps(detector))

    visual_review = {"pass": clean}
    visual_review_path = tmp_path / "visual_review.json"
    visual_review_path.write_text(json.dumps(visual_review))

    out = tmp_path / "report"
    argv = ["--arms", str(arms_path), "--holdout-manifest", str(holdout_path),
           "--detector-report", str(detector_path),
           "--visual-review", str(visual_review_path), "--out-dir", str(out)]
    return argv, out


def _fake_clean_scores(arms_cases, holdout):
    rows = []
    for cid, case in arms_cases.items():
        cohort = case["cohort"]
        for arm in ("arm1", "arm2", "arm3"):
            delta = 0.0 if arm == "arm1" else (0.1 if cohort == "rare" else 0.0)
            rows.append({"case_id": cid, "cohort": cohort, "arm": arm,
                        "generation_failed": case[arm]["generation_failed"],
                        "cropped_dino_delta": delta,
                        "whole_clip": 0.8, "whole_siglip": 0.8, "preservation": 1.0})
    return rows


def test_pass_writes_result_and_scope_statement(tmp_path):
    argv, out = _setup(tmp_path, clean=True)
    main(argv, score_fn=_fake_clean_scores)
    rep = json.loads((out / "development.json").read_text())
    assert rep["readiness"] == "PASS"
    assert "integration testing only" in (out / "development.md").read_text().lower()
    assert (out / "phase_b_result.json").exists()
    assert (out / "phase_b_result.json.sha256").exists()


def test_inconclusive_arms_short_circuits(tmp_path):
    argv, out = _setup(tmp_path, arms_status="inconclusive")
    main(argv, score_fn=_fake_clean_scores)
    assert json.loads((out / "development.json").read_text())["readiness"] == "INCONCLUSIVE"
    assert not (out / "phase_b_result.json").exists()
