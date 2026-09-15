import json, subprocess, sys, pytest
from pathlib import Path

@pytest.mark.gpu
def test_partcrops_worker_emits_multi_ref_instances(tmp_path):
    out = tmp_path / "crops"
    subprocess.run([sys.executable, "-m", "graft.worker_build_partcrops",
                    "--root", "data/treevill/rawdata2", "--concept", "Bamboo",
                    "--out", str(out), "--config", "configs/pipeline_eval_run.yaml"],
                   check=True)
    recs = json.loads((out / "Bamboo.json").read_text())
    assert len(recs) >= 5                      # multiple parts x multiple build refs
    assert {r["concept"] for r in recs} == {"Bamboo"}
    assert len(recs[0]["siglip2"]) > 0
    # more than one distinct ref contributed (all-build-refs, not just medoid)
    assert len({r["ref_path"] for r in recs}) >= 2
