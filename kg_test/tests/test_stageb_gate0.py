import json, math
import numpy as np
from graft import stageb_gate0 as g


def test_load_build_attrs_visible_only(tmp_path):
    kg = {"concept": "X", "parts": [
        {"name": "bark", "attributes": [{"name": "texture", "value": "Rough "},
                                        {"name": "color", "value": "not visible"}]},
        {"name": "leaf", "attributes": [{"name": "shape", "value": "Elliptical"}]}]}
    d = tmp_path / "X"; d.mkdir(); (d / "kg.json").write_text(json.dumps(kg))
    out = g.load_build_attrs(str(tmp_path / "*" / "kg.json"))
    assert out["X"][("bark", "texture")] == "rough"
    assert ("bark", "color") not in out["X"]
    assert out["X"][("leaf", "shape")] == "elliptical"


def test_load_exemplars_l2_normed(tmp_path):
    kg = {"concept": "X", "parts": [{"name": "leaf", "embeddings": {"siglip2": [3.0, 4.0]}}]}
    d = tmp_path / "X"; d.mkdir(); (d / "kg.json").write_text(json.dumps(kg))
    out = g.load_exemplars(str(tmp_path / "*" / "kg.json"))
    assert np.allclose(np.linalg.norm(out["X"]["leaf"]), 1.0)


def test_heldout_label_majority_disjoint_and_ties():
    records = [{"ref_path": "a.jpg"}, {"ref_path": "a.jpg"},
               {"ref_path": "b.jpg"}, {"ref_path": "c.jpg"},
               {"ref_path": "BUILD.jpg"}]
    reads = {
        "a.jpg": {("bark", "texture"): "rough", ("leaf", "shape"): "ovate"},
        "b.jpg": {("bark", "texture"): "rough", ("leaf", "shape"): "elliptical"},
        "c.jpg": {("bark", "texture"): "not visible", ("leaf", "shape"): "ovate"},
        "BUILD.jpg": {("bark", "texture"): "smooth"},
    }
    calls = []
    def read_fn(p): calls.append(p); return reads[p]
    lab = g.heldout_label(records, build_set={"BUILD.jpg"}, read_fn=read_fn)
    assert "BUILD.jpg" not in calls
    assert sorted(calls) == ["a.jpg", "b.jpg", "c.jpg"]
    assert lab[("bark", "texture")] == "rough"          # rough 2 vs not-visible(dropped)
    assert lab[("leaf", "shape")] == "ovate"            # ovate 2 vs elliptical 1


def test_heldout_label_drops_true_tie():
    records = [{"ref_path": "a.jpg"}, {"ref_path": "b.jpg"}]
    reads = {"a.jpg": {("bark", "color"): "brown"}, "b.jpg": {("bark", "color"): "gray"}}
    lab = g.heldout_label(records, set(), lambda p: reads[p])
    assert ("bark", "color") not in lab                 # 1-1 tie dropped


def test_rare_values_threshold():
    labels = {}
    for i in range(16): labels[f"r{i}"] = {("bark", "texture"): "rough"}
    for i in range(2):  labels[f"f{i}"] = {("bark", "texture"): "flaky"}
    for i in range(2):  labels[f"p{i}"] = {("bark", "texture"): "peeling"}
    labels["nv"] = {}
    rare = g.rare_values(labels, "bark", "texture")
    assert rare == {"flaky", "peeling"}                 # N=20, thr=3


def test_arms_retrieval_vs_taxonomy(monkeypatch):
    import graft.taxonomy as tax
    monkeypatch.setattr(tax, "family_of",
                        lambda c: {"C": "F", "S1": "F", "S2": "F", "NN": "G"}.get(c))
    ex = {"C": {"bark": np.array([1.0, 0.0])}, "NN": {"bark": np.array([0.99, 0.01])},
          "S1": {"bark": np.array([0.2, 1.0])}, "S2": {"bark": np.array([0.3, 1.0])}}
    ba = {"NN": {("bark", "texture"): "smooth"}, "S1": {("bark", "texture"): "peeling"},
          "S2": {("bark", "texture"): "peeling"}, "C": {("bark", "texture"): "rough"}}
    assert g.retrieval_pred("C", "bark", "texture", ex, ba) == "smooth"
    assert g.mmkg_pred("C", "bark", "texture", ex, ba) == "peeling"
    monkeypatch.setattr(tax, "family_of", lambda c: {"C": "F"}.get(c))
    assert g.mmkg_pred("C", "bark", "texture", ex, ba) is None


def test_gate0_decision_and_mcnemar():
    cases = []
    for i in range(30):
        cases.append({"concept": f"C{i}", "part": "bark", "attr": "texture",
                      "label": "L", "rare": True,
                      "pred_mmkg": "L" if i < 24 else "X",
                      "pred_retrieval": "L" if i < 12 else "Y"})
    scored = g.score_cases(cases, lambda pred, lab, a: pred == lab)
    out = g.gate0(scored)
    assert out["n_rare"] == 30
    assert abs(out["rare_recall_mmkg"] - 24 / 30) < 1e-9
    assert abs(out["rare_recall_retrieval"] - 12 / 30) < 1e-9
    assert out["delta"] > 0.15 and out["mcnemar_p"] < 0.05 and out["decision"] == "PASS"


def test_gate0_winddown_when_underpowered():
    cases = [{"concept": "C", "part": "bark", "attr": "texture", "label": "L", "rare": True,
              "pred_mmkg": "L", "pred_retrieval": "X"}]
    scored = g.score_cases(cases, lambda pred, lab, a: pred == lab)
    out = g.gate0(scored)
    assert out["n_rare"] == 1 and out["decision"] == "WIND_DOWN"    # n_rare < 30


def test_feasibility_counts(monkeypatch):
    import graft.taxonomy as tax
    monkeypatch.setattr(tax, "family_of", lambda c: "F")
    ex = {c: {"bark": np.array([1.0, float(i)])} for i, c in enumerate(["A", "B", "C", "D"])}
    # rough x3 (common, count 3 > thr=2), flaky x1 (rare) -> N=4, thr=max(2,ceil(0.6))=2
    ba = {"A": {("bark", "texture"): "rough"}, "B": {("bark", "texture"): "rough"},
          "C": {("bark", "texture"): "rough"}, "D": {("bark", "texture"): "flaky"}}
    labels = {c: ba[c] for c in ba}
    out = g.feasibility(labels, ex, ba)
    assert out["n_included"] == 4 and out["n_rare"] == 1
