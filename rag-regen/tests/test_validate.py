import json
from pathlib import Path

from PIL import Image

from ragregen import config, validate


def make_img(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (127, 127, 127)).save(p)


def ds_with(tmp_path, refs, images_root=None):
    root = images_root or tmp_path
    return config.DatasetConfig(
        name="t", images_root=root,
        cases=[config.Case(id="c1", prompt="p", concept="k", coarse="thing",
                           gt_refs=[root / r for r in refs])],
    )


def test_missing_gt_ref_file_is_an_error(tmp_path):
    ds = ds_with(tmp_path, ["nope.jpg"])
    problems = validate.validate_dataset(ds)
    assert any(p.code == "gt_ref_missing" and p.severity == "error" for p in problems)


def test_readable_gt_ref_produces_no_error(tmp_path):
    make_img(tmp_path / "ok.jpg")
    ds = ds_with(tmp_path, ["ok.jpg"])
    problems = validate.validate_dataset(ds)
    assert [p for p in problems if p.severity == "error"] == []


def test_corrupt_gt_ref_is_an_error(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image")
    ds = ds_with(tmp_path, ["bad.jpg"])
    problems = validate.validate_dataset(ds)
    assert any(p.code == "gt_ref_unreadable" for p in problems)


def test_gt_ref_inside_retrieval_corpus_is_an_error(tmp_path):
    corpus = tmp_path / "corpus"
    make_img(corpus / "leak.jpg")
    ds = ds_with(tmp_path, ["corpus/leak.jpg"])
    db = config.RetrievalDBConfig(
        name="db", images_root=corpus,
        index_path=tmp_path / "i.faiss", encoder="siglip_so400m_384",
    )
    problems = validate.validate_leakage(ds, db)
    assert any(p.code == "gt_ref_in_corpus" and p.severity == "error"
               for p in problems)


def test_encoder_dim_mismatch_is_an_error(tmp_path):
    db = config.RetrievalDBConfig(
        name="db", images_root=tmp_path,
        index_path=tmp_path / "i.faiss", encoder="siglip_so400m_384",
    )
    problems = validate.validate_db(db, expected_dim=768)
    assert any(p.code == "index_dim_mismatch" for p in problems)


def test_unknown_encoder_is_an_error(tmp_path):
    db = config.RetrievalDBConfig(
        name="db", images_root=tmp_path,
        index_path=tmp_path / "i.faiss", encoder="nope",
    )
    problems = validate.validate_db(db, expected_dim=None)
    assert any(p.code == "unknown_encoder" for p in problems)


def test_sibling_prefix_not_detected_as_leakage(tmp_path):
    """Regression test: corpus=/data/corpus, ref=/data/corpus_holdout/x.jpg should NOT error."""
    corpus = tmp_path / "corpus"
    corpus_holdout = tmp_path / "corpus_holdout"
    make_img(corpus_holdout / "safe.jpg")
    ds = ds_with(tmp_path, ["corpus_holdout/safe.jpg"])
    db = config.RetrievalDBConfig(
        name="db", images_root=corpus,
        index_path=tmp_path / "i.faiss", encoder="siglip_so400m_384",
    )
    problems = validate.validate_leakage(ds, db)
    assert not any(p.code == "gt_ref_in_corpus" for p in problems)


def test_unresolvable_gt_ref_warns_instead_of_silently_skipping(tmp_path, monkeypatch):
    """OSError during ref resolve should emit leakage_check_skipped warning."""
    from pathlib import Path as _P
    make_img(tmp_path / "ok.jpg")
    ds = ds_with(tmp_path, ["ok.jpg"])
    db = config.RetrievalDBConfig(
        name="db", images_root=tmp_path / "corpus",
        index_path=tmp_path / "i.faiss", encoder="siglip_so400m_384",
    )
    real_resolve = _P.resolve
    def resolve_with_error(self, *args, **kwargs):
        if self.name == "ok.jpg":
            raise OSError("simulated symlink loop")
        return real_resolve(self, *args, **kwargs)
    monkeypatch.setattr(_P, "resolve", resolve_with_error)
    problems = validate.validate_leakage(ds, db)
    assert any(p.code == "leakage_check_skipped" for p in problems)
    assert all(p.severity == "warning" for p in problems if p.code == "leakage_check_skipped")


def test_unresolvable_corpus_warns_instead_of_silently_skipping(tmp_path, monkeypatch):
    """OSError during corpus resolve should emit leakage_check_skipped warning."""
    from pathlib import Path as _P
    make_img(tmp_path / "ok.jpg")
    ds = ds_with(tmp_path, ["ok.jpg"])
    db = config.RetrievalDBConfig(
        name="db", images_root=tmp_path / "corpus",
        index_path=tmp_path / "i.faiss", encoder="siglip_so400m_384",
    )
    real_resolve = _P.resolve
    def resolve_with_error(self, *args, **kwargs):
        if self.name == "corpus":
            raise OSError("simulated symlink loop")
        return real_resolve(self, *args, **kwargs)
    monkeypatch.setattr(_P, "resolve", resolve_with_error)
    problems = validate.validate_leakage(ds, db)
    assert any(p.code == "leakage_check_skipped" for p in problems)
    assert all(p.severity == "warning" for p in problems if p.code == "leakage_check_skipped")


def test_validate_disk_low_free_space_is_error(tmp_path, monkeypatch):
    """validate_disk with insufficient free space should return error."""
    import shutil
    from collections import namedtuple
    Usage = namedtuple("Usage", "total used free")
    # min_free_gb=20 -> free < 20 -> disk_low error
    monkeypatch.setattr(shutil, "disk_usage",
                        lambda p: Usage(total=int(500e9), used=int(495e9), free=int(5e9)))
    problems = validate.validate_disk(min_free_gb=20.0, path=tmp_path)
    assert [p.code for p in problems] == ["disk_low"]
    assert problems[0].severity == "error"


def test_validate_disk_tight_is_warning(tmp_path, monkeypatch):
    """validate_disk with tight free space should return warning."""
    import shutil
    from collections import namedtuple
    Usage = namedtuple("Usage", "total used free")
    # min_free_gb=20 -> tight band is 20 <= free < 100 GB
    monkeypatch.setattr(shutil, "disk_usage",
                        lambda p: Usage(total=int(500e9), used=int(450e9), free=int(50e9)))
    problems = validate.validate_disk(min_free_gb=20.0, path=tmp_path)
    assert [p.code for p in problems] == ["disk_tight"]
    assert problems[0].severity == "warning"


def test_validate_disk_sufficient_space_has_no_problems(tmp_path, monkeypatch):
    """validate_disk with abundant free space should return empty list."""
    import shutil
    from collections import namedtuple
    Usage = namedtuple("Usage", "total used free")
    # min_free_gb=20 -> free > 100 GB -> no problems
    monkeypatch.setattr(shutil, "disk_usage",
                        lambda p: Usage(total=int(500e9), used=int(50e9), free=int(450e9)))
    problems = validate.validate_disk(min_free_gb=20.0, path=tmp_path)
    assert problems == []


def test_validate_all_bad_tau_zero_is_error(tmp_path):
    """validate_all with tau=0 should return error."""
    make_img(tmp_path / "ok.jpg")
    ds = ds_with(tmp_path, ["ok.jpg"])
    db = config.RetrievalDBConfig(
        name="db", images_root=tmp_path,
        index_path=tmp_path / "i.faiss", encoder="siglip_so400m_384",
    )
    pipe = config.PipelineConfig(
        retry_budget=1, tau=0.0, steps=10, seed=42,
        crop_scorer="scorer", retriever="retriever",
    )
    problems = validate.validate_all(ds, db, pipe)
    assert any(p.code == "bad_tau" and p.severity == "error" for p in problems)


def test_validate_all_bad_tau_one_is_error(tmp_path):
    """validate_all with tau=1.0 should return error (must be strictly < 1)."""
    make_img(tmp_path / "ok.jpg")
    ds = ds_with(tmp_path, ["ok.jpg"])
    db = config.RetrievalDBConfig(
        name="db", images_root=tmp_path,
        index_path=tmp_path / "i.faiss", encoder="siglip_so400m_384",
    )
    pipe = config.PipelineConfig(
        retry_budget=1, tau=1.0, steps=10, seed=42,
        crop_scorer="scorer", retriever="retriever",
    )
    problems = validate.validate_all(ds, db, pipe)
    assert any(p.code == "bad_tau" and p.severity == "error" for p in problems)


def test_validate_all_bad_retry_budget_zero_is_error(tmp_path):
    """validate_all with retry_budget=0 should return error."""
    make_img(tmp_path / "ok.jpg")
    ds = ds_with(tmp_path, ["ok.jpg"])
    db = config.RetrievalDBConfig(
        name="db", images_root=tmp_path,
        index_path=tmp_path / "i.faiss", encoder="siglip_so400m_384",
    )
    pipe = config.PipelineConfig(
        retry_budget=0, tau=0.5, steps=10, seed=42,
        crop_scorer="scorer", retriever="retriever",
    )
    problems = validate.validate_all(ds, db, pipe)
    assert any(p.code == "bad_retry_budget" and p.severity == "error" for p in problems)


def test_validate_warns_when_a_coarse_term_is_thinly_referenced(tmp_path):
    from ragregen import validate
    from ragregen.config import Case, DatasetConfig

    ds = DatasetConfig(
        name="t", images_root=tmp_path,
        cases=[Case(id="c1", prompt="p", concept="African grey parrot",
                    coarse="parrot", gt_refs=[tmp_path / "a.png"])],
        coarse_refs={"parrot": [tmp_path / "a.png", tmp_path / "b.png"]},
    )
    warnings = validate.check_coarse_refs(ds)
    assert any("parrot" in w and "2" in w for w in warnings)


def test_validate_is_quiet_when_coarse_refs_are_absent_entirely(tmp_path):
    from ragregen import validate
    from ragregen.config import Case, DatasetConfig

    ds = DatasetConfig(
        name="t", images_root=tmp_path,
        cases=[Case(id="c1", prompt="p", concept="c", coarse="k",
                    gt_refs=[tmp_path / "a.png"])])
    assert validate.check_coarse_refs(ds) == []


def test_a_case_with_too_few_refs_for_the_holdout_is_reported():
    from ragregen import validate
    problems = validate.check_heldout_refs(
        [("thin", 1), ("fat", 3)], heldout_refs=1)
    #: 1 ref and h=1 leaves zero to edit with. Catch it in validate, not an
    #: hour into an encoding run.
    assert any("thin" in p for p in problems)
    assert not any("fat" in p for p in problems)


# Tests for validate_corpus_density


class _Case:
    def __init__(self, cid, concept):
        self.id = cid
        self.concept = concept


class _DS:
    def __init__(self, cases):
        self.cases = cases


def test_a_thin_concept_warns_and_does_not_error(tmp_path):
    #: The run must proceed: "the corpus had nothing here" is a result the
    #: report carries (doc 5 §10). Erroring would abort a booked GPU block.
    m = tmp_path / "corpus_manifest.json"
    m.write_text(json.dumps({"n_images": 10,
                             "per_concept": {"axolotl": 1, "fox": 900}}))
    got = validate.validate_corpus_density(
        m, _DS([_Case("axolotl", "axolotl"), _Case("fox", "fox")]), k=3)

    assert [p.severity for p in got] == ["warning"]
    assert got[0].code == "corpus_thin"
    assert "axolotl" in got[0].message


def test_a_dense_corpus_reports_nothing(tmp_path):
    m = tmp_path / "corpus_manifest.json"
    m.write_text(json.dumps({"n_images": 10, "per_concept": {"fox": 900}}))
    assert validate.validate_corpus_density(
        m, _DS([_Case("fox", "fox")]), k=3) == []


def test_a_concept_absent_from_the_manifest_is_treated_as_zero(tmp_path):
    #: Absent means the manifest predates the case, which is worse than thin,
    #: not better.
    m = tmp_path / "corpus_manifest.json"
    m.write_text(json.dumps({"n_images": 10, "per_concept": {}}))
    got = validate.validate_corpus_density(m, _DS([_Case("fox", "fox")]), k=3)
    assert got[0].code == "corpus_thin"


def test_a_missing_manifest_warns_rather_than_raising(tmp_path):
    #: The oracle arm needs no corpus at all, so a missing manifest must not
    #: block validate for a run that will never retrieve.
    got = validate.validate_corpus_density(
        tmp_path / "nope.json", _DS([_Case("fox", "fox")]), k=3)
    assert [p.code for p in got] == ["corpus_manifest_missing"]
    assert got[0].severity == "warning"


def test_mixed_case_concepts_are_looked_up_by_lowercase_key(tmp_path):
    #: Manifest keys are lowercase, while dataset concepts are mixed-case
    #: (e.g. "Boston bull" in dataset.yaml vs. "boston bull" in manifest).
    #: The lookup must normalize to lowercase; the message must report original.
    m = tmp_path / "corpus_manifest.json"
    m.write_text(json.dumps({"n_images": 10,
                             "per_concept": {"boston bull": 900}}))
    got = validate.validate_corpus_density(
        m, _DS([_Case("c1", "Boston bull")]), k=3)
    # Should NOT report thin (900 >= 3)
    assert got == []

    # Now test that a thin mixed-case concept is reported with its original casing
    m.write_text(json.dumps({"n_images": 10,
                             "per_concept": {"boston bull": 1}}))
    got = validate.validate_corpus_density(
        m, _DS([_Case("c1", "Boston bull")]), k=3)
    assert len(got) == 1
    assert got[0].code == "corpus_thin"
    # Message should contain the original casing, not the lowercased form
    assert "Boston bull" in got[0].message
    assert "boston bull" not in got[0].message


def test_validate_all_includes_corpus_density(tmp_path, monkeypatch):
    #: The operator must learn a concept is unservable BEFORE booking a GPU
    #: block, not nine hours into one. validate_cli prints whatever
    #: validate_all returns, so wiring it here is the whole change.
    called = {}

    def _fake(manifest_path, ds, k):
        called["manifest"] = manifest_path
        called["k"] = k
        return [validate._warn("corpus_thin", "axolotl is thin")]

    monkeypatch.setattr(validate, "validate_corpus_density", _fake)
    monkeypatch.setattr(validate, "validate_dataset", lambda ds: [])
    monkeypatch.setattr(validate, "validate_leakage", lambda ds, db: [])
    monkeypatch.setattr(validate, "validate_db", lambda db, expected_dim: [])
    monkeypatch.setattr(validate, "validate_disk", lambda: [])

    db = config.RetrievalDBConfig(
        name="c", images_root=tmp_path / "images",
        index_path=tmp_path / "corpus" / "index.faiss",
        encoder="siglip_so400m_384")
    pipe = config.load_pipeline(Path("configs/pipeline.yaml"))

    got = validate.validate_all(_DS([_Case("fox", "fox")]), db, pipe)

    assert [p.code for p in got] == ["corpus_thin"]
    #: The manifest lives beside the index, so one config key locates both.
    assert called["manifest"] == tmp_path / "corpus" / "corpus_manifest.json"
    assert called["k"] == pipe.retry_budget
