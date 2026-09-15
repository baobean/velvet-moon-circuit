import pytest

from ragregen.mmkg_store import schema as s
from ragregen.mmkg_store import store as st


def _record(species_key, genus, family, attr_value, part_type):
    return s.build_record(
        dataset="treevill",
        species_key=species_key,
        scientific_name=f"{genus} sp.",
        common_name=species_key,
        taxonomy={"genus": genus, "family": family},
        medoid={
            "image_path": f"{species_key}.jpg",
            "embedding_ref": 0,
            "k_images": 1,
            "selection": "siglip-centroid-nearest",
        },
        candidates=[{"image_path": f"{species_key}.jpg", "embedding_ref": 0}],
        part_crops=[
            {
                "part_type": part_type,
                "crop_path": f"{species_key}_{part_type}.jpg",
                "bbox": [0, 0, 1, 1],
                "embedding_ref": 0,
                "source_image": f"{species_key}.jpg",
            }
        ],
        attributes={
            "bark.texture": {
                "value": attr_value,
                "support": None,
                "visible_count": None,
                "source": "graft-kg",
            }
        },
        provenance={"source_split": "build", "build_manifest": {"path": "m.json", "sha256": "abc"}},
    )


def _build_three():
    # rec1 & rec2 share genus "Certhia" (same hub); rec1 & rec3 share attribute "rough".
    rec1 = _record("s1", "Certhia", "Certhiidae", "rough", "bark")
    rec2 = _record("s2", "Certhia", "Certhiidae", "smooth", "leaf")
    rec3 = _record("s3", "Quercus", "Fagaceae", "rough", "bark")
    return rec1, rec2, rec3


def test_write_store_and_load_roundtrip(tmp_path):
    rec1, rec2, rec3 = _build_three()
    out_dir = tmp_path / "out"
    st.write_store([rec1, rec2, rec3], out_dir)

    assert (out_dir / "store" / "treevill" / "s1.json").exists()
    assert (out_dir / "store" / "treevill" / "s2.json").exists()
    assert (out_dir / "store" / "treevill" / "s3.json").exists()
    assert (out_dir / "inverted.json").exists()
    # no index_builder given -> no faiss artifacts written
    assert not (out_dir / "index.faiss").exists()

    store = st.Store.load(out_dir)

    assert store.get("treevill:s1") == rec1
    assert store.get("treevill:missing") is None


def test_query_hub_and_attribute_and_intersection(tmp_path):
    rec1, rec2, rec3 = _build_three()
    out_dir = tmp_path / "out"
    st.write_store([rec1, rec2, rec3], out_dir)
    store = st.Store.load(out_dir)

    hub = "treevill:genus:Certhia"
    assert set(store.query(hub=hub)) == {"treevill:s1", "treevill:s2"}
    assert set(store.members(hub)) == {"treevill:s1", "treevill:s2"}

    assert set(store.query(attribute=("bark.texture", "rough"))) == {"treevill:s1", "treevill:s3"}

    # both filters together -> intersection
    assert set(store.query(hub=hub, attribute=("bark.texture", "rough"))) == {"treevill:s1"}

    # part_type filter
    assert set(store.query(part_type="bark")) == {"treevill:s1", "treevill:s3"}
    assert set(store.query(part_type="leaf")) == {"treevill:s2"}

    # a filter combination with no matches -> empty
    assert store.query(hub=hub, attribute=("bark.texture", "nonexistent-value")) == []


def test_provenance(tmp_path):
    rec1, rec2, rec3 = _build_three()
    out_dir = tmp_path / "out"
    st.write_store([rec1, rec2, rec3], out_dir)
    store = st.Store.load(out_dir)

    assert store.provenance("treevill:s1") == rec1["provenance"]


def test_inverted_index_shape(tmp_path):
    rec1, rec2, rec3 = _build_three()
    out_dir = tmp_path / "out"
    st.write_store([rec1, rec2, rec3], out_dir)

    import json
    inverted = json.loads((out_dir / "inverted.json").read_text())

    assert set(inverted.keys()) == {"attributes", "hubs", "parts"}
    assert set(inverted["attributes"]["bark.texture|rough"]) == {"treevill:s1", "treevill:s3"}
    assert set(inverted["hubs"]["treevill:genus:Certhia"]) == {"treevill:s1", "treevill:s2"}
    assert inverted["parts"]["bark"] == [0, 0]  # embedding_ref per crop (both use ref 0 in this fixture)


def test_nearest_crops_filters_by_part_type_and_hub(tmp_path):
    faiss = pytest.importorskip("faiss")
    import numpy as np

    from ragregen.mmkg_store import embed_index as ei

    builder = ei.IndexBuilder()
    ref_leaf1 = builder.add(
        np.array([1.0, 0.0], dtype="float32"),
        {"global_id": "treevill:oak", "kind": "part_crop", "path": "leaf1.jpg"},
    )
    ref_bark1 = builder.add(
        np.array([0.0, 1.0], dtype="float32"),
        {"global_id": "treevill:oak", "kind": "part_crop", "path": "bark1.jpg"},
    )
    ref_leaf2 = builder.add(
        np.array([0.9, 0.1], dtype="float32"),
        {"global_id": "treevill:pine", "kind": "part_crop", "path": "leaf2.jpg"},
    )

    rec_oak = s.build_record(
        dataset="treevill", species_key="oak", scientific_name="Quercus robur", common_name="Oak",
        taxonomy={"genus": "Quercus", "family": "Fagaceae"},
        medoid={"image_path": "oak.jpg", "embedding_ref": ref_leaf1, "k_images": 1,
                "selection": "siglip-centroid-nearest"},
        candidates=[], part_crops=[
            {"part_type": "leaf", "crop_path": "leaf1.jpg", "bbox": [0, 0, 1, 1],
             "embedding_ref": ref_leaf1, "source_image": "oak.jpg"},
            {"part_type": "bark", "crop_path": "bark1.jpg", "bbox": [0, 0, 1, 1],
             "embedding_ref": ref_bark1, "source_image": "oak.jpg"},
        ], attributes={}, provenance={},
    )
    rec_pine = s.build_record(
        dataset="treevill", species_key="pine", scientific_name="Pinus sylvestris", common_name="Pine",
        taxonomy={"genus": "Pinus", "family": "Pinaceae"},
        medoid={"image_path": "pine.jpg", "embedding_ref": ref_leaf2, "k_images": 1,
                "selection": "siglip-centroid-nearest"},
        candidates=[], part_crops=[
            {"part_type": "leaf", "crop_path": "leaf2.jpg", "bbox": [0, 0, 1, 1],
             "embedding_ref": ref_leaf2, "source_image": "pine.jpg"},
        ], attributes={}, provenance={},
    )

    out_dir = tmp_path / "out"
    st.write_store([rec_oak, rec_pine], out_dir, index_builder=builder)

    store = st.Store.load(out_dir)
    query_vec = np.array([1.0, 0.0], dtype="float32")

    hits = store.nearest_crops(query_vec, part_type="leaf", k=5)
    hit_gids = {meta["global_id"] for _, _, meta in hits}
    assert hit_gids == {"treevill:oak", "treevill:pine"}
    hit_refs = {ref for ref, _, _ in hits}
    assert ref_bark1 not in hit_refs  # bark excluded from a leaf-only query

    hits_hub = store.nearest_crops(query_vec, hub="treevill:genus:Quercus", k=5)
    assert {meta["global_id"] for _, _, meta in hits_hub} == {"treevill:oak"}
