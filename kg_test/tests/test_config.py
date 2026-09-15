from graft.config import GraftConfig


def test_defaults_and_yaml_roundtrip(tmp_path):
    cfg = GraftConfig()
    assert cfg.sdxl_id == "stabilityai/stable-diffusion-xl-base-1.0"
    assert cfg.vlm_id == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert 0.0 < cfg.ip_scale <= 1.0

    p = tmp_path / "c.yaml"
    p.write_text("ip_scale: 0.4\nn_seeds: 2\n")
    cfg2 = GraftConfig.from_yaml(str(p))
    assert cfg2.ip_scale == 0.4 and cfg2.n_seeds == 2
    assert cfg2.vlm_id == cfg.vlm_id  # unspecified keys keep defaults


def test_to_yaml_roundtrips_every_field(tmp_path):
    cfg = GraftConfig(ip_scale=0.42, n_refine=3, outputs_dir="/tmp/x")
    p = tmp_path / "full.yaml"
    cfg.to_yaml(str(p))
    cfg2 = GraftConfig.from_yaml(str(p))
    assert cfg2 == cfg


def test_new_debias_defaults():
    cfg = GraftConfig()
    assert cfg.neutralize_name is True
    assert cfg.exemplar_selection == "medoid"
    assert cfg.use_part_tree is True


def test_from_yaml_accepts_new_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("neutralize_name: false\nexemplar_selection: text\nuse_part_tree: false\n")
    cfg = GraftConfig.from_yaml(str(p))
    assert cfg.neutralize_name is False and cfg.exemplar_selection == "text" and cfg.use_part_tree is False


def test_hub_label_decoding_defaults_are_locked():
    c = GraftConfig()
    assert c.hub_label_do_sample is False
    assert c.hub_label_max_new_tokens == 512
