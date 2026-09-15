import json

from ragregen.mmkg import inat_pilot
from ragregen.mmkg_store import manifests as m


# --- generate_treevill_manifest --------------------------------------------

def test_generate_treevill_manifest_reads_ref_paths_keyed_by_concept(tmp_path):
    outputs = tmp_path / "outputs"
    (outputs / "Ashore").mkdir(parents=True)
    (outputs / "Ashore" / "kg.json").write_text(json.dumps({
        "concept": "Ashore",
        "ref_paths": ["a1.jpg", "a2.jpg"],
        "parts": [],
    }))
    (outputs / "Bazna").mkdir(parents=True)
    (outputs / "Bazna" / "kg.json").write_text(json.dumps({
        "concept": "Bazna",
        "ref_paths": ["b1.jpg"],
        "parts": [],
    }))

    # fix-wave-2 I1: every ref_path is joined against treevill_root -- kg.json
    # stores paths relative to the kg_test repo root, not rag-regen.
    root = tmp_path / "kg_test_root"
    result = m.generate_treevill_manifest(str(outputs / "*" / "kg.json"), treevill_root=str(root))

    assert result == {
        "Ashore": {
            "eligible": [str(root / "a1.jpg"), str(root / "a2.jpg")],
            "source_split": "ref_paths_only",
        },
        "Bazna": {
            "eligible": [str(root / "b1.jpg")],
            "source_split": "ref_paths_only",
        },
    }


def test_generate_treevill_manifest_default_root_matches_kg_test_repo(tmp_path):
    # I1: with no treevill_root override, ref_paths resolve under the real
    # kg_test repo root (where kg.json's ref_paths/exemplar_crop actually
    # live), never left un-joined or resolved against some other root.
    outputs = tmp_path / "outputs"
    (outputs / "Akashmoni").mkdir(parents=True)
    (outputs / "Akashmoni" / "kg.json").write_text(json.dumps({
        "concept": "Akashmoni",
        "ref_paths": ["data/treevill/rawdata2/Akashmoni/1.jpg"],
        "parts": [],
    }))

    result = m.generate_treevill_manifest(str(outputs / "*" / "kg.json"))

    assert m.DEFAULT_TREEVILL_ROOT == "/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/kg_test"
    assert result["Akashmoni"]["eligible"] == [
        f"{m.DEFAULT_TREEVILL_ROOT}/data/treevill/rawdata2/Akashmoni/1.jpg"
    ]


def test_generate_treevill_manifest_empty_glob_yields_empty_manifest(tmp_path):
    result = m.generate_treevill_manifest(str(tmp_path / "nothing" / "*" / "kg.json"))
    assert result == {}


# --- generate_inat_manifest -------------------------------------------------

def _species_fixture():
    return [
        {"id": 1, "name": "Aveus testis", "common_name": "Test Warbler",
         "family": "Testidae", "genus": "Aveus"},
        {"id": 42, "name": "Mocka avis", "common_name": "Mock Finch",
         "family": "Mockidae", "genus": "Mocka"},
    ]


def _val_json_fixture():
    # 5 images for species 1, 4 for species 42 -- enough to exercise both the
    # eligible (build) and eval_set (test) slices at a small n_build.
    images = (
        [{"id": i, "file_name": f"sp1/{i}.jpg"} for i in range(1, 6)]
        + [{"id": i, "file_name": f"sp42/{i}.jpg"} for i in range(101, 105)]
    )
    annotations = (
        [{"image_id": i, "category_id": 1} for i in range(1, 6)]
        + [{"image_id": i, "category_id": 42} for i in range(101, 105)]
    )
    return {"categories": [], "images": images, "annotations": annotations}


def test_generate_inat_manifest_uses_selection_seam_and_splits_build_eval(monkeypatch, tmp_path):
    monkeypatch.setattr(inat_pilot, "select_pilot_species", lambda val_json: _species_fixture())

    val_json = _val_json_fixture()
    data_root = str(tmp_path / "data")

    result = m.generate_inat_manifest(val_json, data_root, n_build=3)

    assert set(result) == {"00001", "00042"}

    sp1 = result["00001"]
    assert sp1["eligible"] == [f"{data_root}/sp1/{i}.jpg" for i in (1, 2, 3)]
    assert sp1["eval_set"] == [f"{data_root}/sp1/{i}.jpg" for i in (4, 5)]
    # eligible and eval_set never overlap
    assert not (set(sp1["eligible"]) & set(sp1["eval_set"]))
    # I4: iNat keeps its real build/eval split, explicit and non-fabricated
    # (reflects the actual n_build/N_TEST used, not a hardcoded "7/3")
    assert sp1["source_split"] == "build_3_eval_3"

    sp42 = result["00042"]
    assert sp42["eligible"] == [f"{data_root}/sp42/{i}.jpg" for i in (101, 102, 103)]
    assert sp42["eval_set"] == [f"{data_root}/sp42/{i}.jpg" for i in (104,)]
    assert sp42["source_split"] == "build_3_eval_3"


def test_generate_inat_manifest_key_matches_species_key_format(monkeypatch, tmp_path):
    # species_key format ("%05d" of the category id) must match what
    # build_inat._species_key derives, so build.py's filtered manifest
    # lookups (keyed the same way) hit.
    from ragregen.mmkg_store.build_inat import _species_key

    species = _species_fixture()
    monkeypatch.setattr(inat_pilot, "select_pilot_species", lambda val_json: species)
    result = m.generate_inat_manifest(_val_json_fixture(), str(tmp_path / "data"), n_build=3)
    assert set(result) == {_species_key(sp) for sp in species}
