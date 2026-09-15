import json, subprocess, sys, pytest
from pathlib import Path

@pytest.mark.gpu
def test_starve_cell_emits_scored_row(tmp_path):
    out = tmp_path / "rows"
    subprocess.run([sys.executable, "-m", "graft.worker_starve_cell",
        "--concept", "Bamboo", "--level", "1", "--draw", "0", "--condition", "hub",
        "--graph", "outputs/mmkg/graph.json", "--kg", "outputs/Bamboo/kg.json",
        "--root", "data/treevill/rawdata2", "--out", str(out),
        "--config", "configs/pipeline_eval_run.yaml"], check=True)
    row = json.loads((out / "Bamboo_1_0_hub.json").read_text())
    assert row["condition"] == "hub" and row["level"] == 1
    assert 0.0 <= row["dino"] <= 1.0
