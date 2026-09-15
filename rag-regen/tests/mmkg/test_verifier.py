from ragregen.mmkg import verifier as vf

def _targets(**kv):  # helper: mark given slots as targets
    return {s: {"value": v, "is_target": True} for s, v in kv.items()}

def test_fail_on_one_contradiction():
    targets = _targets(primary_color="red", surface_texture="rough")
    read = {"primary_color": "red", "surface_texture": "smooth"}
    judge = lambda x, y, slot: x == y                      # exact-match stand-in
    out = vf.case_verdict(targets, read, judge)
    assert out["decidable"] and out["n_present"] == 2 and out["n_contradicted"] == 1
    assert out["verdict"] == "FAIL"

def test_pass_when_all_present_consistent():
    targets = _targets(primary_color="red", surface_texture="rough")
    read = {"primary_color": "red", "surface_texture": "rough"}
    assert vf.case_verdict(targets, read, lambda x,y,s: x==y)["verdict"] == "PASS"

def test_abstain_when_undecidable_or_nothing_present():
    one = _targets(primary_color="red")                    # 1 target -> undecidable
    assert vf.case_verdict(one, {"primary_color":"red"}, lambda x,y,s:x==y)["verdict"] == "ABSTAIN"
    two = _targets(primary_color="red", surface_texture="rough")
    read = {"primary_color": "not visible", "surface_texture": "none"}   # nothing present
    v = vf.case_verdict(two, read, lambda x,y,s:x==y)
    assert v["n_present"] == 0 and v["verdict"] == "ABSTAIN"

def test_missing_never_contradicts():
    targets = _targets(primary_color="red", surface_texture="rough")
    read = {"primary_color": "red", "surface_texture": "not visible"}
    calls = []
    def judge(x, y, slot):
        calls.append((x, y, slot)); return True
    v = vf.case_verdict(targets, read, judge)
    assert v["n_present"] == 1 and v["n_contradicted"] == 0 and v["verdict"] == "PASS"
    # the missing attribute must never be handed to the judge:
    assert all(c[1] != "not visible" for c in calls)
    assert not any(c[0] == "rough" for c in calls)

def test_judge_fn_decides_differing_present_pair():
    targets = _targets(primary_color="crimson", surface_texture="rough")
    # differing strings -> short-circuit does NOT apply -> judge is the authority
    read = {"primary_color": "scarlet", "surface_texture": "rough"}
    seen = []
    def judge_yes(x, y, slot): seen.append((x, y, slot)); return True
    v = vf.case_verdict(targets, read, judge_yes)
    assert ("crimson", "scarlet", "primary_color") in seen   # judge WAS called on the differing pair
    assert v["n_contradicted"] == 0 and v["verdict"] == "PASS"
    # and when the judge says mismatch on that differing pair -> contradicted -> FAIL
    v2 = vf.case_verdict(targets, read, lambda x, y, slot: False)
    assert v2["n_contradicted"] == 1 and v2["verdict"] == "FAIL"
