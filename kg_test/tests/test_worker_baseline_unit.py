# tests/test_worker_baseline_unit.py
from graft.config import GraftConfig
from graft.worker_baseline import cfg_for_method


def test_cfg_for_method_prunes_only_notree():
    base = GraftConfig()
    assert cfg_for_method(base, "ours").use_part_tree is True
    assert cfg_for_method(base, "ours_notree").use_part_tree is False
    assert cfg_for_method(base, "b1").use_part_tree is True
