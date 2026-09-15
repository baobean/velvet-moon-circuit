"""Offline smoke test for the mmkg_store build orchestrator: FAKE encoder,
FAKE VLM (never called -- ``read_fn`` short-circuits it), 1 synthetic
Treevill concept + 2 synthetic iNat bird species. No torch, no GPU -- only
PIL/faiss (already required transitively) and the frozen mmkg_store modules
under test.
"""
import json
import sys

import numpy as np
import pytest
from PIL import Image

from ragregen.mmkg import inat_pilot
from ragregen.mmkg_store import reference_source as rs
from ragregen.mmkg_store.store import Store


def _mk_image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(path, format="PNG")


# ---------------------------------------------------------------------------
# import must not load a model
# ---------------------------------------------------------------------------

def test_import_does_not_instantiate_encoder_or_vlm(monkeypatch):
    # Force a fresh import so the module body actually re-executes here,
    # regardless of what earlier tests in this session already imported.
    sys.modules.pop("ragregen.mmkg_store.build", None)

    def _boom(*args, **kwargs):
        raise AssertionError("encoder/VLM must not be constructed at import time")

    monkeypatch.setattr("ragregen.encoders.build_encoder", _boom, raising=False)
    monkeypatch.setattr("ragregen.vlm.QwenVLM", _boom, raising=False)

    import ragregen.mmkg_store.build  # noqa: F401 -- must not raise


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class FakeEncoder:
    """Implements the encoder protocol: .encode_images(paths) -> np.ndarray."""

    def __init__(self, vecs):
        self.vecs = vecs
        self.calls = []

    def encode_images(self, paths):
        self.calls.append(list(paths))
        return np.array([self.vecs[p] for p in paths], dtype="float32")


def _boom_vlm_factory():
    raise AssertionError("vlm_factory must not be called when read_fn is given")


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------

def test_main_builds_store_for_one_tree_and_two_birds(tmp_path, monkeypatch):
    from ragregen.mmkg_store import build

    # --- Treevill fixture: 1 concept, 3 eligible refs, 1 part w/ crop -------
    outputs_dir = tmp_path / "outputs"
    tree_imgs_dir = tmp_path / "tree_imgs"
    p1, p2, p3 = (tree_imgs_dir / f"p{i}.png" for i in (1, 2, 3))
    _mk_image(p1, (255, 0, 0))
    _mk_image(p2, (0, 255, 0))
    _mk_image(p3, (0, 0, 255))

    kg = {
        "concept": "Ashore",
        "parts": [{
            "name": "leaf",
            "attributes": [
                {"name": "shape", "value": "ovate", "source": "vision"},
                {"name": "texture", "value": "not visible", "source": "vision"},
            ],
            "exemplar_crop": "crops/leaf.png",
        }],
        "ref_paths": [str(p1), str(p2), str(p3)],
    }
    concept_dir = outputs_dir / "Ashore"
    concept_dir.mkdir(parents=True)
    (concept_dir / "kg.json").write_text(json.dumps(kg))

    taxonomy_py = tmp_path / "taxonomy.py"
    taxonomy_py.write_text('TAXONOMY = {"Ashore": ("Ashorea testis", "Testaceae")}\n')

    # --- iNat fixture: 2 species, 3 build + 1 eval image each ---------------
    inat_data_root = tmp_path / "inat_data"
    sp1_imgs = [inat_data_root / "sp1" / f"{i}.png" for i in range(1, 5)]
    sp2_imgs = [inat_data_root / "sp2" / f"{i}.png" for i in range(1, 5)]
    for i, path in enumerate(sp1_imgs):
        _mk_image(path, (100 + i, 0, 0))
    for i, path in enumerate(sp2_imgs):
        _mk_image(path, (0, 100 + i, 0))

    val_json = {
        "categories": [],
        "images": (
            [{"id": i, "file_name": f"sp1/{i}.png"} for i in range(1, 5)]
            + [{"id": 10 + i, "file_name": f"sp2/{i}.png"} for i in range(1, 5)]
        ),
        "annotations": (
            [{"image_id": i, "category_id": 1} for i in range(1, 5)]
            + [{"image_id": 10 + i, "category_id": 2} for i in range(1, 5)]
        ),
    }
    val_json_path = tmp_path / "val.json"
    val_json_path.write_text(json.dumps(val_json))

    species = [
        {"id": 1, "name": "Aveus testis", "common_name": "Test Warbler",
         "family": "Testidae", "genus": "Aveus"},
        {"id": 2, "name": "Aveus fictus", "common_name": None,
         "family": "Testidae", "genus": "Aveus"},
    ]
    monkeypatch.setattr(inat_pilot, "select_pilot_species", lambda vj: species)

    # --- shared fake encoder over every path the build will embed ----------
    # treevill_root=str(tmp_path) below -- exemplar_crop ("crops/leaf.png")
    # is relative to that root (I1), so the fake encoder is keyed by the
    # resolved absolute path, not the raw kg.json string.
    vecs = {
        str(p1): [1.0, 0.0], str(p2): [0.0, 1.0], str(p3): [0.9, 0.1],
        str(tmp_path / "crops/leaf.png"): [0.5, 0.5],
    }
    for i, path in enumerate(sp1_imgs):
        vecs[str(path)] = [1.0, float(i) / 10.0]
    for i, path in enumerate(sp2_imgs):
        vecs[str(path)] = [0.0, 1.0 - float(i) / 10.0]
    fake_encoder = FakeEncoder(vecs)

    reads = {
        str(sp1_imgs[0]): {"primary_plumage_color": "red"},
        str(sp1_imgs[1]): {"primary_plumage_color": "red"},
        str(sp1_imgs[2]): {"primary_plumage_color": "red"},
        str(sp2_imgs[0]): {"primary_plumage_color": "blue"},
        str(sp2_imgs[1]): {"primary_plumage_color": "blue"},
        str(sp2_imgs[2]): {"primary_plumage_color": "blue"},
    }

    def fake_read_fn(path):
        base = reads.get(path, {"primary_plumage_color": "not visible"})
        return {
            "primary_plumage_color": base["primary_plumage_color"],
            "secondary_plumage_color": "not visible",
            "plumage_pattern": "not visible",
            "bill_shape": "not visible",
            "bill_color": "not visible",
            "distinctive_markings": "not visible",
        }

    out_dir = tmp_path / "store_out"

    summary = build.main(
        [],
        encoder_factory=lambda: fake_encoder,
        vlm_factory=_boom_vlm_factory,
        read_fn=fake_read_fn,
        kg_glob=str(outputs_dir / "*" / "kg.json"),
        val_json_path=str(val_json_path),
        data_root=str(inat_data_root),
        taxonomy_py=str(taxonomy_py),
        n_build=3,
        out_dir=str(out_dir),
        treevill_root=str(tmp_path),
        expected_dim=2,  # FakeEncoder emits toy 2-dim vectors, not real 1152
    )

    # --- artifacts on disk ---------------------------------------------------
    assert (out_dir / "inverted.json").is_file()
    assert (out_dir / "index.faiss").is_file()
    assert (out_dir / "sidecar.json").is_file()
    assert (out_dir / "build_summary.json").is_file()
    assert (out_dir / "store" / "treevill" / "Ashore.json").is_file()
    assert (out_dir / "store" / "inat" / "00001.json").is_file()
    assert (out_dir / "store" / "inat" / "00002.json").is_file()

    # --- summary shape (infrastructure counts only) --------------------------
    on_disk_summary = json.loads((out_dir / "build_summary.json").read_text())
    assert on_disk_summary == summary
    assert summary["n_species"] == 3
    assert summary["dataset_counts"] == {"treevill": 1, "inat": 2}
    assert summary["n_crops_indexed"] == 1          # 1 tree part crop
    assert summary["n_medoids_indexed"] == 3         # 1 tree + 2 birds
    for key in ("dino", "clip", "fidelity", "nn_baseline", "repair"):
        assert key not in json.dumps(summary).lower()

    # --- load the store back and exercise both adapters -----------------
    loaded = Store.load(out_dir)

    tree_img = rs.medoid_reference(loaded, "treevill:Ashore")
    assert tree_img is not None and tree_img.mode == "RGB"

    bird_img = rs.medoid_reference(loaded, "inat:00001")
    assert bird_img is not None and bird_img.mode == "RGB"

    assert rs.medoid_reference(loaded, "inat:99999") is None

    # fake VLM/encoder factories actually drove the build
    assert fake_encoder.calls  # embed_fn really used

    # --- fix-wave-2 I3/I4: per-record provenance shape ------------------
    tree_rec = loaded.get("treevill:Ashore")
    prov = tree_rec["provenance"]
    assert "build_manifest" in prov and set(prov["build_manifest"]) == {"path", "sha256"}
    assert prov["build_manifest"]["path"] is None  # manifest generated in-memory
    assert prov["build_manifest"]["sha256"]
    assert "excluded_images" in prov and isinstance(prov["excluded_images"], list)
    # I4: Treevill has no build/eval split of its own -- documented, not fabricated
    assert prov["source_split"] == "ref_paths_only"

    bird_rec = loaded.get("inat:00001")
    bird_prov = bird_rec["provenance"]
    assert "build_manifest" in bird_prov and set(bird_prov["build_manifest"]) == {"path", "sha256"}
    assert bird_prov["build_manifest"]["path"] is None
    assert bird_prov["build_manifest"]["sha256"]
    assert "excluded_images" in bird_prov and isinstance(bird_prov["excluded_images"], list)
    # I4: iNat keeps its real build/eval split, reflecting the actual n_build=3/N_TEST
    assert bird_prov["source_split"] == "build_3_eval_3"


# ---------------------------------------------------------------------------
# fix-round-1 regression: non-LOCKED encoder must never silently mislabel
# the store (the FAISS sidecar always stamps `LOCKED` regardless of which
# encoder produced the vectors)
# ---------------------------------------------------------------------------

def test_non_locked_encoder_raises_before_any_build_work():
    from ragregen.mmkg_store import build

    with pytest.raises(ValueError, match="locked encoder"):
        build.main(["--encoder", "not_the_locked_encoder"])


# ---------------------------------------------------------------------------
# fix-round-1 regression: a kg-glob concept missing from taxonomy.py must be
# skipped + tallied, never crash the (long, GPU) build
# ---------------------------------------------------------------------------

def test_treevill_concept_missing_from_taxonomy_is_skipped_not_crashed(tmp_path):
    from ragregen.mmkg_store import build

    outputs_dir = tmp_path / "outputs"
    imgs_dir = tmp_path / "imgs"

    def _mk_concept(name, color):
        p = imgs_dir / f"{name}.png"
        _mk_image(p, color)
        kg = {"concept": name, "parts": [], "ref_paths": [str(p)]}
        concept_dir = outputs_dir / name
        concept_dir.mkdir(parents=True)
        (concept_dir / "kg.json").write_text(json.dumps(kg))
        return str(p)

    mapped_img = _mk_concept("Ashore", (255, 0, 0))
    unmapped_img = _mk_concept("Ghost", (0, 255, 0))

    taxonomy_py = tmp_path / "taxonomy.py"
    taxonomy_py.write_text('TAXONOMY = {"Ashore": ("Ashorea testis", "Testaceae")}\n')

    # No Aves categories -> select_pilot_species(val_json) naturally returns
    # [] -- the iNat side is a genuine no-op, no monkeypatch/read_fn needed;
    # this test is about the Treevill taxonomy guard only.
    val_json_path = tmp_path / "val.json"
    val_json_path.write_text(json.dumps({"categories": [], "images": [], "annotations": []}))

    fake_encoder = FakeEncoder({mapped_img: [1.0, 0.0], unmapped_img: [0.0, 1.0]})
    out_dir = tmp_path / "store_out"

    summary = build.main(
        [],
        encoder_factory=lambda: fake_encoder,
        vlm_factory=_boom_vlm_factory,  # must never be called -- no bird species
        kg_glob=str(outputs_dir / "*" / "kg.json"),
        val_json_path=str(val_json_path),
        data_root=str(tmp_path / "inat_data"),
        taxonomy_py=str(taxonomy_py),
        n_build=3,
        out_dir=str(out_dir),
        expected_dim=2,  # FakeEncoder emits toy 2-dim vectors, not real 1152
    )

    # build did NOT raise; only the taxonomy-mapped concept made it into the store
    assert summary["dataset_counts"] == {"treevill": 1}
    assert (out_dir / "store" / "treevill" / "Ashore.json").is_file()
    assert not (out_dir / "store" / "treevill" / "Ghost.json").exists()

    treevill_tally = summary["eligibility"]["treevill"]
    assert treevill_tally["n_concepts"] == 2
    assert treevill_tally["excluded_reasons"]["no_taxonomy_entry"] == 1
    assert treevill_tally["excluded_concepts"] == ["Ghost"]


# ---------------------------------------------------------------------------
# fix-wave-2 #4: the first real embedding must be exactly LOCKED["dim"]
# (1152)-dimensional -- a mismatched encoder must fail loudly, not silently
# mislabel the store.
# ---------------------------------------------------------------------------

def test_first_embedding_dim_mismatch_raises_value_error(tmp_path):
    from ragregen.mmkg_store import build

    outputs_dir = tmp_path / "outputs"
    p = tmp_path / "Ashore.png"
    _mk_image(p, (255, 0, 0))
    kg = {"concept": "Ashore", "parts": [], "ref_paths": [str(p)]}
    concept_dir = outputs_dir / "Ashore"
    concept_dir.mkdir(parents=True)
    (concept_dir / "kg.json").write_text(json.dumps(kg))

    taxonomy_py = tmp_path / "taxonomy.py"
    taxonomy_py.write_text('TAXONOMY = {"Ashore": ("Ashorea testis", "Testaceae")}\n')

    val_json_path = tmp_path / "val.json"
    val_json_path.write_text(json.dumps({"categories": [], "images": [], "annotations": []}))

    # a toy 2-dim fake encoder standing in for a misconfigured/wrong encoder
    fake_encoder = FakeEncoder({str(p): [1.0, 0.0]})

    with pytest.raises(ValueError, match="1152"):
        build.main(
            [],
            encoder_factory=lambda: fake_encoder,
            vlm_factory=_boom_vlm_factory,
            kg_glob=str(outputs_dir / "*" / "kg.json"),
            val_json_path=str(val_json_path),
            data_root=str(tmp_path / "inat_data"),
            taxonomy_py=str(taxonomy_py),
            n_build=3,
            out_dir=str(tmp_path / "store_out"),
            # expected_dim intentionally NOT overridden -- exercises the real
            # production default (LOCKED["dim"] == 1152) against a 2-dim fake.
        )


# ---------------------------------------------------------------------------
# fix-wave-2 I2: an empty eligible pool (after manifest.eligible_pool) must be
# skipped and tallied, for both datasets -- the build must never crash by
# calling an adapter's `_centroid_nearest_index` on an empty array.
# ---------------------------------------------------------------------------

def test_treevill_empty_eligible_pool_is_skipped_not_crashed(tmp_path):
    from ragregen.mmkg_store import build

    outputs_dir = tmp_path / "outputs"

    p = tmp_path / "Ashore.png"
    _mk_image(p, (255, 0, 0))
    kg_ok = {"concept": "Ashore", "parts": [], "ref_paths": [str(p)]}
    (outputs_dir / "Ashore").mkdir(parents=True)
    (outputs_dir / "Ashore" / "kg.json").write_text(json.dumps(kg_ok))

    # taxonomy-mapped concept whose kg.json carries NO ref_paths at all --
    # eligible_pool trivially yields an empty pool.
    kg_empty = {"concept": "EmptyPool", "parts": [], "ref_paths": []}
    (outputs_dir / "EmptyPool").mkdir(parents=True)
    (outputs_dir / "EmptyPool" / "kg.json").write_text(json.dumps(kg_empty))

    taxonomy_py = tmp_path / "taxonomy.py"
    taxonomy_py.write_text(
        'TAXONOMY = {"Ashore": ("Ashorea testis", "Testaceae"), '
        '"EmptyPool": ("Emptia poolia", "Emptiaceae")}\n'
    )

    val_json_path = tmp_path / "val.json"
    val_json_path.write_text(json.dumps({"categories": [], "images": [], "annotations": []}))

    fake_encoder = FakeEncoder({str(p): [1.0, 0.0]})
    out_dir = tmp_path / "store_out"

    summary = build.main(
        [],
        encoder_factory=lambda: fake_encoder,
        vlm_factory=_boom_vlm_factory,
        kg_glob=str(outputs_dir / "*" / "kg.json"),
        val_json_path=str(val_json_path),
        data_root=str(tmp_path / "inat_data"),
        taxonomy_py=str(taxonomy_py),
        n_build=3,
        out_dir=str(out_dir),
        expected_dim=2,
    )

    # build did NOT raise; only the non-empty concept made it into the store
    assert summary["dataset_counts"] == {"treevill": 1}
    assert (out_dir / "store" / "treevill" / "Ashore.json").is_file()
    assert not (out_dir / "store" / "treevill" / "EmptyPool.json").exists()

    treevill_tally = summary["eligibility"]["treevill"]
    assert treevill_tally["excluded_reasons"]["empty_eligible_pool"] == 1
    assert "EmptyPool" in treevill_tally["excluded_concepts"]


def test_inat_empty_eligible_pool_is_skipped_not_crashed(tmp_path, monkeypatch):
    from ragregen.mmkg_store import build

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True)  # no Treevill concepts -- keep this test iNat-focused

    taxonomy_py = tmp_path / "taxonomy.py"
    taxonomy_py.write_text("TAXONOMY = {}\n")

    inat_data_root = tmp_path / "inat_data"
    sp1_imgs = [inat_data_root / "sp1" / f"{i}.png" for i in range(1, 5)]
    for i, path in enumerate(sp1_imgs):
        _mk_image(path, (100 + i, 0, 0))
    # species 2 has NO images at all in val.json -> its eligible pool is empty

    val_json = {
        "categories": [],
        "images": [{"id": i, "file_name": f"sp1/{i}.png"} for i in range(1, 5)],
        "annotations": [{"image_id": i, "category_id": 1} for i in range(1, 5)],
    }
    val_json_path = tmp_path / "val.json"
    val_json_path.write_text(json.dumps(val_json))

    species = [
        {"id": 1, "name": "Aveus testis", "common_name": "Test Warbler",
         "family": "Testidae", "genus": "Aveus"},
        {"id": 2, "name": "Aveus vacuus", "common_name": "Empty Finch",
         "family": "Testidae", "genus": "Aveus"},
    ]
    monkeypatch.setattr(inat_pilot, "select_pilot_species", lambda vj: species)

    vecs = {str(p): [1.0, float(i) / 10.0] for i, p in enumerate(sp1_imgs)}
    fake_encoder = FakeEncoder(vecs)

    def fake_read_fn(path):
        return {
            "primary_plumage_color": "red", "secondary_plumage_color": "not visible",
            "plumage_pattern": "not visible", "bill_shape": "not visible",
            "bill_color": "not visible", "distinctive_markings": "not visible",
        }

    out_dir = tmp_path / "store_out"

    summary = build.main(
        [],
        encoder_factory=lambda: fake_encoder,
        vlm_factory=_boom_vlm_factory,
        read_fn=fake_read_fn,
        kg_glob=str(outputs_dir / "*" / "kg.json"),
        val_json_path=str(val_json_path),
        data_root=str(inat_data_root),
        taxonomy_py=str(taxonomy_py),
        n_build=3,
        out_dir=str(out_dir),
        expected_dim=2,
    )

    # build did NOT raise; only the non-empty species made it into the store
    assert summary["dataset_counts"] == {"inat": 1}
    assert (out_dir / "store" / "inat" / "00001.json").is_file()
    assert not (out_dir / "store" / "inat" / "00002.json").exists()

    inat_tally = summary["eligibility"]["inat"]
    assert inat_tally["excluded_reasons"]["empty_eligible_pool"] == 1
    assert "00002" in inat_tally["excluded_species"]
