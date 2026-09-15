import json

import pytest

from scripts.phase_b_generate import main

RARE = [f"f{i}" for i in range(24)]
CONTROL = [f"p{i}" for i in range(24)]
ALL_IDS = RARE + CONTROL


def _setup(tmp_path, *, detector_pass=True, routed_fail=None):
    holdout = {
        "schema": 1, "protocol_status": "frozen_before_score",
        "cohorts": {"rare": {"cases": RARE, "backfilled": []},
                   "control": {"cases": CONTROL, "backfilled": []}},
        "dino_reserves": {}, "references": {}, "sources": {},
    }
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(json.dumps(holdout))

    # A handful of cases routed, so a routed-failure test has something to fail.
    routed_ids = {ALL_IDS[0], ALL_IDS[1]}
    if routed_fail is not None:
        routed_ids.add(routed_fail)
    detector = {
        "readiness": "PASS" if detector_pass else "FAIL",
        "proceed_to_generation": detector_pass,
        "cases": {cid: {"routed": cid in routed_ids, "truth_fail": cid in RARE,
                       "cohort": "rare" if cid in RARE else "control"}
                 for cid in ALL_IDS},
    }
    detector_path = tmp_path / "detector.json"
    detector_path.write_text(json.dumps(detector))

    screen_run = tmp_path / "screen"
    out = tmp_path / "generation"
    argv = ["--holdout-manifest", str(holdout_path),
           "--detector-report", str(detector_path),
           "--screen-run", str(screen_run),
           "--out-dir", str(out)]
    return argv, out


def _fake_gen_ok(case_ids, screen_run, out_dir):
    return {cid: {"attempt_ok": True, "attempt_path": f"/a/{cid}.png"} for cid in case_ids}


def _fake_gen_with_failure(routed_fail):
    def _gen(case_ids, screen_run, out_dir):
        return {cid: {"attempt_ok": cid != routed_fail, "attempt_path": f"/a/{cid}.png"}
               for cid in case_ids}
    return _gen


def test_refuses_without_detector_pass(tmp_path):
    argv, out = _setup(tmp_path, detector_pass=False)
    with pytest.raises(Exception):
        main(argv, generate_fn=_fake_gen_ok)
    assert not out.exists()


def test_inconclusive_on_routed_failure(tmp_path):
    argv, out = _setup(tmp_path, routed_fail="f0")
    main(argv, generate_fn=_fake_gen_with_failure("f0"))
    assert json.loads((out / "arms.json").read_text())["status"] == "inconclusive"


def test_success_writes_48_arm_rows(tmp_path):
    argv, out = _setup(tmp_path)
    main(argv, generate_fn=_fake_gen_ok)
    arms = json.loads((out / "arms.json").read_text())
    assert len(arms["cases"]) == 48
    assert arms["status"] == "complete"
