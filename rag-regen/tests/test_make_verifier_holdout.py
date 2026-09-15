from pathlib import Path

import pytest

from ragregen import config
from scripts.make_verifier_holdout import build_entries, emit_dataset


RARE_IDS = (
    "lagotto_romagnolo", "xoloitzcuintli", "azawakh",
    "bergamasco_shepherd", "khachapuri", "salak", "cherimoya",
    "hakka_tulou", "stave_church", "trullo", "okapi",
    "saiga_antelope",
)

CONTROL_DIRS = (
    "apple_fruit", "barn_building", "church_building", "daisy_flower",
    "deer", "donkey", "labrador_retriever", "lizard_on_a_rock",
    "pine_tree", "pizza", "stone_cottage", "tiger",
)


@pytest.fixture
def asset_tree(tmp_path):
    images = tmp_path / "images"
    scenes = tmp_path / "scenes"
    for case_id in RARE_IDS:
        folder = images / case_id
        folder.mkdir(parents=True)
        for index in range(4):
            (folder / f"wm_{index:02d}.jpg").touch()
    for folder_name in CONTROL_DIRS:
        folder = scenes / folder_name
        folder.mkdir(parents=True)
        for index in range(3):
            (folder / f"wm_{index:02d}.jpg").touch()
    return tmp_path


def test_build_entries_emits_twelve_rare_and_twelve_controls(asset_tree):
    rare, controls = build_entries(
        asset_tree / "images", asset_tree / "scenes")

    assert len(rare) == len(controls) == 12
    assert [row["id"] for row in rare] == list(RARE_IDS)
    assert {row["kind"] for row in rare} == {"target"}
    assert {row["cohort"] for row in rare} == {"bridge"}
    assert {row["kind"] for row in controls} == {"control"}
    assert {row["cohort"] for row in controls} == {"common"}


def test_reference_counts_preserve_edit_and_heldout_splits(asset_tree):
    rare, controls = build_entries(
        asset_tree / "images", asset_tree / "scenes")

    assert all(len(row["gt_refs"]) == 4 for row in rare)
    assert all(len(row["gt_refs"]) == 3 for row in controls)
    assert all(Path(path).is_absolute()
               for row in rare + controls for path in row["gt_refs"])


def test_missing_reference_is_rejected_before_a_gpu_run(asset_tree):
    (asset_tree / "images" / "okapi" / "wm_03.jpg").unlink()

    with pytest.raises(ValueError, match="okapi.*expected 4.*found 3"):
        build_entries(asset_tree / "images", asset_tree / "scenes")


def test_emitted_yaml_round_trips_with_unique_groundable_cases(
        asset_tree, tmp_path):
    rare, controls = build_entries(
        asset_tree / "images", asset_tree / "scenes")
    dataset_path = tmp_path / "holdout.yaml"
    dataset_path.write_text(emit_dataset("verifier_holdout_v1",
                                         rare + controls))

    dataset = config.load_dataset(dataset_path)

    assert len(dataset.cases) == 24
    assert len({case.id for case in dataset.cases}) == 24
    assert all(case.concept.lower() != case.coarse.lower()
               for case in dataset.cases)
    assert [case.id for case in dataset.cases[:12]] == list(RARE_IDS)
