import subprocess, sys, shutil, pytest
from pathlib import Path
from graft.graph_schema import PartTypeGraph

@pytest.mark.gpu
def test_labeling_sets_labels_without_changing_membership(tmp_path):
    src = Path("outputs/mmkg/graph.json")
    g = tmp_path / "graph.json"; shutil.copy(src, g)
    before = PartTypeGraph.from_json(str(g))
    subprocess.run([sys.executable, "-m", "graft.worker_label_hubs",
                    "--graph", str(g), "--config", "configs/pipeline_eval_run.yaml"], check=True)
    after = PartTypeGraph.from_json(str(g))
    # membership identical; labels now populated
    assert [h.member_ids for h in after.part_types] == [h.member_ids for h in before.part_types]
    assert any(h.label for h in after.part_types)
