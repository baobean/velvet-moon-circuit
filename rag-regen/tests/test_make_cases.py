"""Dataset emission. No parquet, no HuggingFace -- just the YAML we produce.

The output has to survive config.load_dataset unchanged, so these tests round
-trip through it rather than asserting on strings.
"""
import pytest
import yaml

from ragregen import config
from scripts.make_cases import (_check_shortfall, _resolve_label_ids,
                                emit_yaml)


def _entry(cid, cohort="common", kind="control"):
    return {"id": cid, "prompt": f"a {cid} somewhere", "concept": cid,
            "coarse": "thing", "kind": kind, "cohort": cohort,
            "gt_refs": [f"{cid}/{cid}_0.jpg", f"{cid}/{cid}_1.jpg",
                        f"{cid}/{cid}_2.jpg"]}


def test_emitted_yaml_loads_as_a_dataset(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text(emit_yaml([_entry("fox")], images_root=tmp_path))
    ds = config.load_dataset(p)
    assert [c.id for c in ds.cases] == ["fox"]


def test_common_cases_are_labelled_common(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text(emit_yaml([_entry("fox")], images_root=tmp_path))
    assert config.load_dataset(p).cases[0].cohort == "common"


def test_bridge_cases_keep_their_cohort(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text(emit_yaml([_entry("durian", cohort="bridge", kind="target")],
                           images_root=tmp_path))
    got = config.load_dataset(p).cases[0]
    assert got.cohort == "bridge"
    assert got.kind == "target"


def test_three_refs_survive_the_heldout_split(tmp_path):
    #: metrics.split_refs with heldout=1 needs >= 2 refs. Emitting 2 would
    #: leave exactly one to edit with and pass; emitting 1 raises at run time,
    #: hours in. Three is the floor this file guarantees.
    p = tmp_path / "d.yaml"
    p.write_text(emit_yaml([_entry("fox")], images_root=tmp_path))
    assert len(config.load_dataset(p).cases[0].gt_refs) == 3


def test_a_duplicate_id_is_refused(tmp_path):
    #: A bridge concept re-derived as a common case would collide here.
    p = tmp_path / "d.yaml"
    p.write_text(emit_yaml([_entry("fox"), _entry("fox")],
                           images_root=tmp_path))
    with pytest.raises(ValueError, match="duplicate"):
        config.load_dataset(p)


def test_images_root_is_written_absolute(tmp_path):
    #: gt_refs resolve against it, and validate.py compares the resolved path
    #: against the corpus root to catch leakage.
    text = emit_yaml([_entry("fox")], images_root=tmp_path)
    assert yaml.safe_load(text)["images_root"] == str(tmp_path)


def test_full_lemma_parquet_names_resolve_to_the_coarse_maps_keys():
    #: Standard HuggingFace ImageNet ClassLabel names are the full
    #: comma-separated synset lemma string; configs/imagenet_coarse.yaml's
    #: keys are the first lemma alone. Matching them verbatim resolves
    #: neither, and the case is skipped with one print line.
    names = ["tench, Tinca tinca", "hammerhead, hammerhead shark",
             "goldfish, Carassius auratus"]
    want_ids, missing, ambiguous = _resolve_label_ids(
        names, ["hammerhead", "tench"])
    assert missing == []
    assert ambiguous == []
    #: The index into `names` IS the label id the batch loop looks up.
    assert want_ids == {1: "hammerhead", 0: "tench"}


def test_first_lemma_parquet_names_resolve_too():
    #: Normalising both sides means the command works whichever convention
    #: the parquet actually ships.
    names = ["tench", "hammerhead", "goldfish"]
    want_ids, missing, _ = _resolve_label_ids(names, ["goldfish"])
    assert missing == []
    assert want_ids == {2: "goldfish"}


def test_a_class_that_matches_nothing_is_reported_missing():
    names = ["tench, Tinca tinca", "goldfish, Carassius auratus"]
    _, missing, _ = _resolve_label_ids(names, ["tench", "axolotl"])
    assert missing == ["axolotl"]


def test_two_labels_normalising_alike_are_reported_not_shadowed():
    #: Folding two distinct ImageNet classes into one prompt is not something
    #: to discover from the refs. First id wins, deterministically, but the
    #: collision comes back to the caller.
    names = ["crane, wading bird", "crane, lifting machine"]
    want_ids, _, ambiguous = _resolve_label_ids(names, ["crane"])
    assert want_ids == {0: "crane"}
    assert ambiguous == [("crane", ["crane, wading bird",
                                    "crane, lifting machine"])]


def test_an_unresolvable_class_is_a_hard_failure():
    #: A truncated case set must never exit 0. Without this the run writes a
    #: valid-looking YAML holding only the 22 bridge cases, `validate` passes
    #: it, and `control/common` is discovered empty ~18 hours in.
    with pytest.raises(SystemExit, match="did not resolve"):
        _check_shortfall(["a", "b", "c"], ["b"], 0)


def test_the_failure_names_the_count_and_the_missing_classes():
    with pytest.raises(SystemExit) as exc:
        _check_shortfall(["a", "b"], ["a", "b"], 0)
    text = str(exc.value)
    assert "2/2" in text
    assert "a, b" in text
    assert "format" in text


def test_allow_short_tolerates_exactly_its_budget():
    #: Operable when a handful of classes genuinely are not in the parquet,
    #: without turning the guard off.
    _check_shortfall(["a", "b", "c"], ["a", "b"], 2)
    with pytest.raises(SystemExit):
        _check_shortfall(["a", "b", "c"], ["a", "b"], 1)


def test_a_class_named_by_a_later_lemma_still_resolves():
    #: configs/imagenet_coarse.yaml keys a class by whichever lemma reads
    #: naturally, not always the first. Label 437 really is
    #: "beacon, lighthouse, beacon light, pharos", and the map calls it
    #: `lighthouse` -- indexing first lemmas alone dropped it, which failed
    #: the whole campaign one class short of 70.
    names = ["tench, Tinca tinca", "beacon, lighthouse, beacon light, pharos"]
    want_ids, missing, _ = _resolve_label_ids(names, ["lighthouse"])
    assert missing == []
    assert want_ids == {1: "lighthouse"}


def test_a_first_lemma_is_not_hijacked_by_another_classes_synonym():
    #: "crane" is the first lemma of one class and a synonym inside another.
    #: The class actually named `crane` must win, or the refs would come from
    #: the wrong synset entirely.
    names = ["crane, Grus grus", "derrick, crane, hoist"]
    want_ids, missing, _ = _resolve_label_ids(names, ["crane"])
    assert missing == []
    assert want_ids == {0: "crane"}
