from pathlib import Path

import pytest

from ragregen import config


def write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_load_dataset_resolves_gt_refs_against_images_root(tmp_path):
    p = write(tmp_path, "ds.yaml", f"""
name: t
images_root: {tmp_path}
cases:
  - id: amur_leopard_01
    prompt: "an Amur leopard on a snowy ridge"
    concept: "Amur leopard"
    coarse: leopard
    gt_refs:
      - amur/ref_01.jpg
""")
    ds = config.load_dataset(p)
    assert ds.name == "t"
    assert len(ds.cases) == 1
    c = ds.cases[0]
    assert c.id == "amur_leopard_01"
    assert c.concept == "Amur leopard"
    assert c.coarse == "leopard"
    assert c.gt_refs == [tmp_path / "amur" / "ref_01.jpg"]


def test_load_dataset_rejects_duplicate_case_ids(tmp_path):
    p = write(tmp_path, "ds.yaml", f"""
name: t
images_root: {tmp_path}
cases:
  - id: dup
    prompt: "a"
    concept: "a"
    coarse: b
    gt_refs: [x.jpg]
  - id: dup
    prompt: "b"
    concept: "b"
    coarse: c
    gt_refs: [y.jpg]
""")
    with pytest.raises(ValueError, match="duplicate case id: dup"):
        config.load_dataset(p)


def test_load_dataset_rejects_missing_gt_refs_key(tmp_path):
    p = write(tmp_path, "ds.yaml", f"""
name: t
images_root: {tmp_path}
cases:
  - id: a
    prompt: "a"
    concept: "a"
    coarse: b
""")
    with pytest.raises(ValueError, match="case 'a' has no gt_refs"):
        config.load_dataset(p)


def _dataset_yaml(tmp_path, coarse_line="    coarse: parrot"):
    img = tmp_path / "imgs"
    img.mkdir(exist_ok=True)
    (img / "ref0.jpg").write_bytes(b"x")
    text = (
        "name: t\n"
        f"images_root: {img}\n"
        "cases:\n"
        "  - id: parrot_case\n"
        "    prompt: an African grey parrot on a branch\n"
        "    concept: African grey parrot\n"
        f"{coarse_line}\n"
        "    gt_refs:\n"
        "      - ref0.jpg\n"
    )
    p = tmp_path / "dataset.yaml"
    p.write_text(text)
    return p


def test_case_carries_the_coarse_term(tmp_path):
    ds = config.load_dataset(_dataset_yaml(tmp_path))
    assert ds.cases[0].coarse == "parrot"


def test_missing_coarse_is_an_actionable_error(tmp_path):
    path = _dataset_yaml(tmp_path, coarse_line="    notes: none")
    with pytest.raises(ValueError) as e:
        config.load_dataset(path)
    assert "coarse" in str(e.value)
    assert "parrot_case" in str(e.value)


def test_coarse_equal_to_concept_is_rejected(tmp_path):
    path = _dataset_yaml(tmp_path, coarse_line="    coarse: African grey parrot")
    with pytest.raises(ValueError) as e:
        config.load_dataset(path)
    assert "superordinate" in str(e.value)


def test_coarse_equal_to_concept_is_rejected_case_insensitively(tmp_path):
    path = _dataset_yaml(tmp_path, coarse_line="    coarse: '  african GREY parrot '")
    with pytest.raises(ValueError) as e:
        config.load_dataset(path)
    assert "superordinate" in str(e.value)


def test_load_pipeline_defaults():
    cfg = config.load_pipeline(config.DEFAULT_PIPELINE_PATH)
    assert cfg.retry_budget == 3
    assert 0.0 < cfg.tau < 1.0
    assert cfg.steps == 28


def test_coarse_refs_are_keyed_by_term_and_resolved_against_images_root(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    cfg = tmp_path / "d.yaml"
    cfg.write_text(f"""
name: t
images_root: {tmp_path}
coarse_refs:
  parrot: [a.jpg]
cases:
  - id: c1
    prompt: p
    concept: "African grey parrot"
    coarse: parrot
    gt_refs: [a.jpg]
""")
    ds = config.load_dataset(cfg)
    assert ds.coarse_refs == {"parrot": [tmp_path / "a.jpg"]}


def test_coarse_refs_absent_is_not_an_error(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    cfg = tmp_path / "d.yaml"
    cfg.write_text(f"""
name: t
images_root: {tmp_path}
cases:
  - id: c1
    prompt: p
    concept: "African grey parrot"
    coarse: parrot
    gt_refs: [a.jpg]
""")
    assert config.load_dataset(cfg).coarse_refs == {}


def test_coarse_refs_for_an_unknown_term_names_the_term(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    cfg = tmp_path / "d.yaml"
    cfg.write_text(f"""
name: t
images_root: {tmp_path}
coarse_refs:
  wombat: [a.jpg]
cases:
  - id: c1
    prompt: p
    concept: "African grey parrot"
    coarse: parrot
    gt_refs: [a.jpg]
""")
    with pytest.raises(ValueError, match="wombat"):
        config.load_dataset(cfg)


def test_pipeline_carries_the_mask_dilation_knob(tmp_path):
    p = tmp_path / "pipeline.yaml"
    p.write_text("retry_budget: 3\ntau: 0.25\nmask_dilate_px: 20\n")
    assert config.load_pipeline(p).mask_dilate_px == 20


def test_mask_dilate_px_defaults_to_twelve(tmp_path):
    p = tmp_path / "pipeline.yaml"
    p.write_text("retry_budget: 3\ntau: 0.25\n")
    assert config.load_pipeline(p).mask_dilate_px == 12


def test_reference_preparation_defaults_to_crop(tmp_path):
    p = tmp_path / "pipeline.yaml"
    p.write_text("retry_budget: 3\ntau: 0.25\n")
    assert config.load_pipeline(p).ref_prep == "crop"


@pytest.mark.parametrize("mode", ["none", "crop", "crop_matte"])
def test_reference_preparation_modes_are_configurable(tmp_path, mode):
    p = tmp_path / "pipeline.yaml"
    p.write_text(f"retry_budget: 3\ntau: 0.25\nref_prep: {mode}\n")
    assert config.load_pipeline(p).ref_prep == mode


def test_unknown_reference_preparation_is_rejected(tmp_path):
    p = tmp_path / "pipeline.yaml"
    p.write_text("retry_budget: 3\ntau: 0.25\nref_prep: paste\n")
    with pytest.raises(ValueError, match="ref_prep"):
        config.load_pipeline(p)


def test_prototype_delta_defaults_off(tmp_path):
    p = tmp_path / "pipeline.yaml"
    p.write_text("retry_budget: 3\ntau: 0.25\n")
    assert config.load_pipeline(p).prototype_delta is None


def test_prototype_delta_is_loaded_when_explicit(tmp_path):
    p = tmp_path / "pipeline.yaml"
    p.write_text("retry_budget: 3\ntau: 0.25\nprototype_delta: 0.15\n")
    assert config.load_pipeline(p).prototype_delta == pytest.approx(0.15)


def test_kind_defaults_to_target(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text("""
name: t
images_root: /tmp
cases:
  - id: a
    prompt: "p"
    concept: "c"
    coarse: "co"
    gt_refs: [x.jpg]
""")
    ds = config.load_dataset(p)
    assert ds.cases[0].kind == "target"


def test_kind_is_read_when_present(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text("""
name: t
images_root: /tmp
cases:
  - id: a
    prompt: "p"
    concept: "c"
    coarse: "co"
    kind: control
    gt_refs: [x.jpg]
""")
    assert config.load_dataset(p).cases[0].kind == "control"


def test_an_unknown_kind_is_refused(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text("""
name: t
images_root: /tmp
cases:
  - id: a
    prompt: "p"
    concept: "c"
    coarse: "co"
    kind: banana
    gt_refs: [x.jpg]
""")
    #: A typo here would silently move a case between report strata.
    with pytest.raises(ValueError, match="kind"):
        config.load_dataset(p)


def test_eval_block_has_defaults(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("retry_budget: 3\ntau: 0.25\nsteps: 28\nseed: 0\n"
                 "mask_dilate_px: 12\ncrop_scorer: siglip_so400m_384\n"
                 "retriever: siglip_so400m_384\n")
    cfg = config.load_pipeline(p)
    assert cfg.eval_heldout_refs == 1
    assert "dinov3" in cfg.eval_dino


def test_an_eval_encoder_may_not_equal_the_retriever(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("retry_budget: 3\ntau: 0.25\nsteps: 28\nseed: 0\n"
                 "mask_dilate_px: 12\ncrop_scorer: siglip_so400m_384\n"
                 "retriever: siglip_so400m_384\n"
                 "eval:\n  siglip: siglip_so400m_384\n")
    #: Scoring an output with the model that chose its reference is
    #: self-marking. It must be impossible to configure, not merely discouraged.
    with pytest.raises(ValueError, match="self-marking|eval encoder"):
        config.load_pipeline(p)


def test_heldout_refs_must_leave_something_to_edit_with(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("retry_budget: 3\ntau: 0.25\nsteps: 28\nseed: 0\n"
                 "mask_dilate_px: 12\ncrop_scorer: siglip_so400m_384\n"
                 "retriever: siglip_so400m_384\n"
                 "eval:\n  heldout_refs: 0\n")
    with pytest.raises(ValueError, match="heldout_refs"):
        config.load_pipeline(p)


def _one_case_yaml(tmp_path, extra=""):
    p = tmp_path / "d.yaml"
    p.write_text(f"""
name: t
images_root: /tmp
cases:
  - id: a
    prompt: "p"
    concept: "African grey parrot"
    coarse: "parrot"
{extra}
    gt_refs: [x.jpg]
""")
    return p


def test_cohort_defaults_to_bridge(tmp_path):
    #: The 22 legacy cases carry no `cohort:` key and must keep behaving as
    #: they did, so the default is the legacy value, not the new one.
    ds = config.load_dataset(_one_case_yaml(tmp_path))
    assert ds.cases[0].cohort == "bridge"


def test_cohort_is_read_when_present(tmp_path):
    p = _one_case_yaml(tmp_path, "    cohort: common")
    assert config.load_dataset(p).cases[0].cohort == "common"


def test_an_unknown_cohort_is_refused(tmp_path):
    #: A typo silently merges a 256px-ref case into the full-res mean, which
    #: is exactly the comparison spec §5 forbids.
    p = _one_case_yaml(tmp_path, "    cohort: banana")
    with pytest.raises(ValueError, match="cohort"):
        config.load_dataset(p)


def test_a_relative_index_path_resolves_against_the_project_root(tmp_path, monkeypatch):
    from ragregen import config, env
    monkeypatch.chdir(tmp_path)                 # not the repo root
    db = config.load_retrieval_db(env.PROJECT_ROOT / "configs/retrieval_db.yaml")
    #: Not just .is_absolute() -- a regression to CWD-anchored
    #: Path(rel).resolve() would ALSO be absolute (tmp_path itself is), so
    #: that alone would still pass under this test's own monkeypatch.chdir.
    #: configs/retrieval_db.yaml's own index_path is "data/laion100k/index.faiss".
    assert db.index_path == env.PROJECT_ROOT / "data/laion100k/index.faiss"


def test_an_absolute_index_path_passes_through_unchanged(tmp_path):
    #: Back-compat: callers that already write an absolute index_path must
    #: not have it silently rewritten.
    abs_index = tmp_path / "some" / "index.faiss"
    p = write(tmp_path, "db.yaml", f"""
name: t
images_root: {tmp_path}
index_path: {abs_index}
encoder: siglip_so400m_384
""")
    db = config.load_retrieval_db(p)
    assert db.index_path == abs_index
