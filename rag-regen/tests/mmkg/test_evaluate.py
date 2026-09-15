import math
from ragregen.mmkg import evaluate as ev

def _mk(n_fail_caught, n_fail, n_fp, n_pass):
    cases = []
    for i in range(n_fail):
        cases.append({"verdict": "FAIL" if i < n_fail_caught else "PASS", "label": "fail"})
    for i in range(n_pass):
        cases.append({"verdict": "FAIL" if i < n_fp else "PASS", "label": "pass"})
    return cases

def test_confusion_and_gate_pass():
    conf = ev.confusion(_mk(20, 24, 1, 24))            # recall 20/24=.833, fp 1/24=.042
    assert conf["tp"] == 20 and conf["fn"] == 4 and conf["fp"] == 1 and conf["tn"] == 23
    assert abs(conf["recall"] - 20/24) < 1e-9 and abs(conf["fp_rate"] - 1/24) < 1e-9
    g = ev.gate(conf); assert g["recall_ok"] and g["fp_ok"] and g["decision"] == "PASS"

def test_gate_negative_and_abstain_counts_as_pass():
    cases = _mk(12, 24, 4, 24)                          # the detector's numbers: recall .50, fp .167
    # turn 3 missed fails into ABSTAIN -> still count as PASS (miss)
    misses = [c for c in cases if c["label"] == "fail" and c["verdict"] == "PASS"][:3]
    for c in misses: c["verdict"] = "ABSTAIN"
    conf = ev.confusion(cases)
    assert conf["n_abstain"] == 3 and conf["tp"] == 12       # abstain not a catch
    # Pins the recall denominator: ABSTAIN must be counted as a miss (fn),
    # not dropped from the denominator -- 24 fail-labeled cases, 12 caught,
    # 12 missed (9 still PASS + 3 now ABSTAIN) => fn == 12, recall == 0.5.
    assert conf["fn"] == 12
    assert conf["recall"] == 0.5
    assert ev.gate(conf)["decision"] == "NEGATIVE"


def test_gate_fp_boundary_2_of_24_passes():
    # Spec §6.6: FP <= 0.083 (<= 2/24) PASSES. 2/24 == 0.08333... > 0.083,
    # so a naive "<= 0.083" comparison wrongly rejects the exact boundary.
    conf = ev.confusion(_mk(18, 24, 2, 24))              # recall 18/24=.75, fp 2/24=.0833
    assert conf["fp"] == 2 and conf["tp"] == 18
    assert conf["recall"] == 0.75
    g = ev.gate(conf)
    assert g["recall_ok"] and g["fp_ok"]
    assert g["decision"] == "PASS"


def test_gate_fp_3_of_24_is_negative():
    conf = ev.confusion(_mk(18, 24, 3, 24))              # recall .75, fp 3/24=.125
    assert conf["fp"] == 3
    g = ev.gate(conf)
    assert g["recall_ok"] and not g["fp_ok"]
    assert g["decision"] == "NEGATIVE"
