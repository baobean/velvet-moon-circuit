from ragregen.phase_b_manifest import select_cohort


def _cand(cid, truth, eligible=True, contaminated=False):
    return {"case_id": cid, "identity_truth": truth, "eligible": eligible, "contaminated": contaminated}


def test_backfill_preserves_24_and_records_substitution():
    cands = [_cand(f"r{i}", "FAIL") for i in range(24)]
    cands[3]["contaminated"] = True           # must be skipped
    cands.append(_cand("spare", "FAIL"))       # replacement reserve
    sel = select_cohort(cands, cohort="rare", needed=24)
    assert len(sel["cases"]) == 24
    assert "r3" not in sel["cases"] and "spare" in sel["cases"]
    assert sel["backfilled"] == ["r3"] and sel["status"] == "complete"


def test_exhausted_reserve_is_inconclusive_not_shrunk():
    cands = [_cand(f"r{i}", "FAIL") for i in range(23)]  # only 23 judgeable
    sel = select_cohort(cands, cohort="rare", needed=24)
    assert sel["status"] == "inconclusive"
    assert len(sel["cases"]) < 24


def test_non_judgeable_and_ineligible_are_skipped():
    cands = [_cand(f"r{i}", "FAIL") for i in range(24)]
    cands[5]["identity_truth"] = "UNJUDGEABLE"
    cands[6]["eligible"] = False
    cands += [_cand("s1", "FAIL"), _cand("s2", "FAIL")]
    sel = select_cohort(cands, cohort="rare", needed=24)
    assert len(sel["cases"]) == 24
    assert {"r5", "r6"}.isdisjoint(sel["cases"])
