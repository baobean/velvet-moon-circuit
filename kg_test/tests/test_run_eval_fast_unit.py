from graft.eval_common import build_manifest
from graft.run_eval_fast import assemble

def test_assemble_reuses_summarize_and_build_analysis():
    man = build_manifest(["A"], ["ours", "b1"], ["neutral"], [0.6], seeds=1)
    def metr(d): return {"dino": d, "siglip2": 0.7, "clip_i": 0.7, "clip_t": 0.2, "attribute_accuracy": 0.5}
    tables = {
        "A|ours|neutral|ip0.6|seed0": metr(0.6),
        "A|b1|neutral|ip0.6|seed0": metr(0.4),
    }
    out = assemble(man, tables, {"A": 10}, use_part_tree=False)
    assert out["expected_cells"] == 2 and out["missing_cells"] == []
    # summarize groups by method, excludes tags
    assert abs(out["summary"]["ours"]["dino"] - 0.6) < 1e-9
    # analysis wiring: ours beats b1 on this one species
    gv = out["analysis"]["graft_vs_b1"]["dino"]
    assert gv["n"] == 1 and abs(gv["mean_delta"] - 0.2) < 1e-9
