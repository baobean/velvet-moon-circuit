import json
from ragregen.mmkg import trace as tr

def test_build_and_write_roundtrip(tmp_path):
    rec = tr.build_trace_record(
        "agaric", "agaric", "rare", "fail",
        draft={"image_path": "d.png", "prompt": "a agaric"},
        retrieval={"flat": {}, "mmkg": {}, "pool_overlap": {}},
        mmkg_concept={"target_attributes": {}, "n_target_attributes": 0},
        attribute_reads={"draft": {}},
        verifier={"verdict": "ABSTAIN"},
        scores_reused={"semantic_ok": True},
        provenance={"vlm_id": "Qwen2.5-VL"})
    p = tr.write_trace(rec, str(tmp_path))
    back = json.load(open(p))
    assert back["concept"] == "agaric" and back["verifier"]["verdict"] == "ABSTAIN"

def test_build_trace_record_requires_sections():
    import pytest
    with pytest.raises(ValueError):
        tr.build_trace_record("c","c","rare","fail", draft={}, retrieval={},
                              mmkg_concept={}, attribute_reads={}, verifier={},
                              scores_reused={}, provenance={})  # empty draft/etc -> missing keys

def test_load_labels(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("case_id,prompt,concept,draft_path,verdict,verdict_identity,notes\n"
                 "agaric,x,agaric,d.png,,fail,\n")
    assert tr.load_labels(str(p)) == {"agaric": "fail"}
