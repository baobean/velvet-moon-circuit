import os, tempfile
from graft.config import GraftConfig


def test_restore_fields_defaults():
    c = GraftConfig()
    assert c.restore_k == 4
    assert list(c.restore_build_levels) == [0, 1, 2, 4]
    assert c.restore_n_draws == 3
    assert 0 < c.restore_min_mask_area_frac < c.restore_max_mask_area_frac <= 1.0


def test_restore_fields_roundtrip_yaml():
    c = GraftConfig()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "cfg.yaml")
        c.to_yaml(p)
        back = GraftConfig.from_yaml(p)
    assert back.restore_k == c.restore_k
    assert list(back.restore_build_levels) == [0, 1, 2, 4]
    assert back.restore_inpaint_strength == c.restore_inpaint_strength
