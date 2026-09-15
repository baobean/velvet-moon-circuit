import pytest

from ragregen.phase_b_arms import PhaseBInconclusive, assemble_arms


def _case(cid, routed, attempt_ok=True):
    return {"case_id": cid, "cohort": "rare", "routed": routed, "attempt_ok": attempt_ok,
            "draft_path": f"/d/{cid}.png", "attempt_path": f"/a/{cid}.png"}


def test_every_arm_has_one_output_and_flags_fallback():
    out = assemble_arms([_case("a", routed=True), _case("b", routed=False)])
    assert out["a"]["arm3"]["path"] == "/a/a.png" and out["a"]["arm3"]["generation_failed"] is False
    assert out["b"]["arm3"]["path"] == "/d/b.png"        # not routed -> draft, ordinary
    assert out["b"]["arm3"]["generation_failed"] is False


def test_routed_failure_makes_study_inconclusive():
    with pytest.raises(PhaseBInconclusive):
        assemble_arms([_case("a", routed=True, attempt_ok=False)])


def test_no_denominator_shrink_all_48_present():
    cases = [_case(f"c{i}", routed=(i % 2 == 0)) for i in range(48)]
    out = assemble_arms(cases)
    assert len(out) == 48 and all("arm1" in v and "arm2" in v and "arm3" in v for v in out.values())
