"""Manifest arithmetic. No network -- img2dataset is never invoked here.

The manifest is what lets a bad result be attributed to retrieval failure
versus corpus sparsity (doc 5 §4), so it has to count what actually landed on
disk rather than what was requested.
"""
import json
import sys

import yaml
from PIL import Image

from scripts.fetch_corpus import build_manifest, cmd_manifest, download_cmd


def _corpus(root, entries):
    """entries: {filename: caption}. Writes images plus img2dataset sidecars."""
    root.mkdir(parents=True, exist_ok=True)
    for name, caption in entries.items():
        Image.new("RGB", (8, 8), (1, 2, 3)).save(root / f"{name}.jpg")
        (root / f"{name}.json").write_text(json.dumps({"caption": caption}))
    return root


def _dataset_yaml(path, concepts):
    """A minimal but valid dataset config, one case per concept."""
    cases = [{"id": f"c{i}", "prompt": f"a {c} somewhere", "concept": c,
              "coarse": "thing", "gt_refs": [f"{i}.jpg"]}
             for i, c in enumerate(concepts)]
    path.write_text(yaml.safe_dump(
        {"name": "t", "images_root": str(path.parent), "cases": cases}))
    return path


def _coarse_yaml(path, concepts):
    path.write_text(yaml.safe_dump({c: "thing" for c in concepts}))
    return path


class _Args:
    """Stand-in for the argparse.Namespace cmd_manifest receives."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_manifest_counts_images_that_actually_landed(tmp_path):
    root = _corpus(tmp_path / "images",
                   {"000": "a red fox", "001": "a violin", "002": "nothing"})
    got = build_manifest(root, ["fox", "violin", "panda"])
    assert got["n_images"] == 3


def test_manifest_counts_captions_per_concept(tmp_path):
    root = _corpus(tmp_path / "images",
                   {"000": "a red fox", "001": "another fox",
                    "002": "a violin"})
    got = build_manifest(root, ["fox", "violin", "panda"])
    assert got["per_concept"] == {"fox": 2, "violin": 1, "panda": 0}


def test_a_concept_the_download_missed_reports_zero_not_absent(tmp_path):
    #: Absent would read as "not asked for". Zero reads as "asked for and the
    #: web had nothing", which is the finding.
    root = _corpus(tmp_path / "images", {"000": "a red fox"})
    got = build_manifest(root, ["fox", "axolotl"])
    assert got["per_concept"]["axolotl"] == 0


def test_an_image_without_a_sidecar_still_counts_as_an_image(tmp_path):
    #: img2dataset occasionally writes the jpg and not the json. Dropping the
    #: image would understate corpus size; counting its caption would crash.
    root = _corpus(tmp_path / "images", {"000": "a red fox"})
    Image.new("RGB", (8, 8)).save(root / "orphan.jpg")
    got = build_manifest(root, ["fox"])
    assert got["n_images"] == 2
    assert got["per_concept"]["fox"] == 1


def test_manifest_counts_the_union_of_dataset_and_coarse_map(tmp_path, capsys):
    #: fetch_corpus.py:119 (pre-fix) only ever counted the 22 bridge concepts.
    #: rank_classes needs frequencies for the whole 532-entry candidate pool,
    #: or every common candidate scores zero and is dropped.
    images = _corpus(tmp_path / "images",
                     {"000": "a red fox", "001": "a violin"})
    dataset = _dataset_yaml(tmp_path / "dataset.yaml", ["violin"])
    coarse = _coarse_yaml(tmp_path / "coarse.yaml", ["fox", "panda"])
    args = _Args(dataset=dataset, coarse=coarse, image_dir=images,
                out=tmp_path / "manifest.json", k=3)
    cmd_manifest(args)
    got = json.loads(args.out.read_text())
    assert got["per_concept"] == {"violin": 1, "fox": 1, "panda": 0}


def test_the_union_is_sorted_and_deduplicated_on_overlap(tmp_path, capsys):
    #: "golden retriever" is both a bridge concept and a coarse-map key in the
    #: real configs. It must appear once, and the manifest's key order must
    #: not depend on which file mentioned it first.
    images = _corpus(tmp_path / "images", {"000": "a golden retriever"})
    dataset = _dataset_yaml(tmp_path / "dataset.yaml",
                            ["golden retriever", "zebra"])
    coarse = _coarse_yaml(tmp_path / "coarse.yaml",
                          ["golden retriever", "axolotl"])
    args = _Args(dataset=dataset, coarse=coarse, image_dir=images,
                out=tmp_path / "manifest.json", k=3)
    cmd_manifest(args)
    got = json.loads(args.out.read_text())
    assert list(got["per_concept"].keys()) == [
        "axolotl", "golden retriever", "zebra"]


def test_the_thin_warning_reflects_only_the_datasets_own_concepts(
        tmp_path, capsys):
    #: A "below k" list over all coarse-map candidates would be mostly noise.
    #: Only the dataset's own concepts -- the bridge cases that must work --
    #: belong in that warning.
    images = _corpus(tmp_path / "images", {"000": "a red fox"})
    dataset = _dataset_yaml(tmp_path / "dataset.yaml", ["fox", "violin"])
    coarse = _coarse_yaml(tmp_path / "coarse.yaml",
                          ["violin", "axolotl", "panda"])
    args = _Args(dataset=dataset, coarse=coarse, image_dir=images,
                out=tmp_path / "manifest.json", k=3)
    cmd_manifest(args)
    out = capsys.readouterr().out
    thin_line = next(l for l in out.splitlines() if "below k" in l)
    assert "violin" in thin_line
    assert "axolotl" not in thin_line
    assert "panda" not in thin_line


def test_a_case_differing_overlap_becomes_one_lowercase_manifest_key(
        tmp_path, capsys):
    #: The real configs have exactly this shape: dataset concepts are
    #: mixed-case ("Boston bull"), every coarse-map key is already lowercase
    #: ("boston bull"). A case-sensitive union would carry the same concept
    #: as two separate per_concept keys.
    images = _corpus(tmp_path / "images", {"000": "a boston bull terrier"})
    dataset = _dataset_yaml(tmp_path / "dataset.yaml", ["Boston bull"])
    coarse = _coarse_yaml(tmp_path / "coarse.yaml", ["boston bull"])
    args = _Args(dataset=dataset, coarse=coarse, image_dir=images,
                out=tmp_path / "manifest.json", k=3)
    cmd_manifest(args)
    got = json.loads(args.out.read_text())
    #: lowercase, sorted, duplicate-free -- exactly one key for the concept.
    assert list(got["per_concept"].keys()) == ["boston bull"]
    assert got["per_concept"]["boston bull"] == 1


def test_thin_warning_survives_the_case_normalization(tmp_path, capsys):
    #: The regression the normalization fix could most easily introduce: a
    #: case-sensitive lookup of a mixed-case dataset concept against the now
    #: -lowercase per_concept would KeyError, or worse, always read as zero
    #: and misreport a well-covered concept as thin.
    images = _corpus(tmp_path / "images",
                     {str(i): "a boston bull terrier" for i in range(5)})
    dataset = _dataset_yaml(tmp_path / "dataset.yaml", ["Boston bull"])
    coarse = _coarse_yaml(tmp_path / "coarse.yaml", ["boston bull"])
    args = _Args(dataset=dataset, coarse=coarse, image_dir=images,
                out=tmp_path / "manifest.json", k=3)
    cmd_manifest(args)
    got = json.loads(args.out.read_text())
    assert got["per_concept"]["boston bull"] == 5
    out = capsys.readouterr().out
    assert "below k" not in out


def test_the_download_targets_an_executable_module(tmp_path):
    #: `-m img2dataset` exits 1 with "cannot be directly executed" -- the
    #: package ships no __main__.py -- so the corpus download died before
    #: fetching one image. It went unnoticed because nothing ran this path
    #: until the campaign did. The executable target is img2dataset.main,
    #: which carries the __main__ guard over fire.Fire(download).
    cmd = download_cmd(tmp_path / "urls.parquet", tmp_path / "images",
                       processes=8, threads=32)
    assert cmd[:3] == [sys.executable, "-m", "img2dataset.main"]


def test_the_download_passes_the_paths_it_was_given(tmp_path):
    #: The args were read off an `args` namespace the helper no longer
    #: receives; a stale reference here would raise NameError at run time,
    #: hours into a campaign rather than in CI.
    cmd = download_cmd(tmp_path / "urls.parquet", tmp_path / "images",
                       processes=4, threads=16)
    assert cmd[cmd.index("--url_list") + 1] == str(tmp_path / "urls.parquet")
    assert cmd[cmd.index("--output_folder") + 1] == str(tmp_path / "images")
    assert cmd[cmd.index("--processes_count") + 1] == "4"
    assert cmd[cmd.index("--thread_count") + 1] == "16"
