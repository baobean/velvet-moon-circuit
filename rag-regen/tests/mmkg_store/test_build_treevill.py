import numpy as np

from ragregen.mmkg_store import build_treevill as bt


# --- fakes -------------------------------------------------------------

VECS = {
    "p1.jpg": [1.0, 0.0],
    "p2.jpg": [0.0, 1.0],
    "p3.jpg": [0.9, 0.1],   # nearest to the 3-point centroid -> medoid
    "crops/leaf.png": [0.5, 0.5],
    "crops/bark.png": [0.2, 0.8],
}


def fake_embed_fn(paths):
    return np.array([VECS[p] for p in paths], dtype="float32")


class FakeIndex:
    def __init__(self):
        self.calls = []

    def add(self, vec, meta):
        ref = len(self.calls)
        self.calls.append((np.asarray(vec, dtype="float32"), dict(meta)))
        return ref


KG = {
    "concept": "Ashore",
    "parts": [
        {
            "name": "leaf",
            "attributes": [
                {"name": "shape", "value": "ovate", "source": "vision"},
                {"name": "texture", "value": "not visible", "source": "vision"},
            ],
            "exemplar_crop": "crops/leaf.png",
        },
        {
            "name": "bark",
            "attributes": [
                {"name": "color", "value": "brown", "source": "vision"},
            ],
            "exemplar_crop": "crops/bark.png",
        },
    ],
}

TAX = {"Ashore": ("Ashorea testis", "Testaceae")}
ELIGIBLE = ["p1.jpg", "p2.jpg", "p3.jpg"]


def test_treevill_record_shape():
    idx = FakeIndex()
    # treevill_root="" is a no-op join (Path("") / "crops/leaf.png" ==
    # "crops/leaf.png") -- keeps this test's crop paths matching `VECS`
    # verbatim; the real-root join is exercised by
    # test_treevill_record_resolves_exemplar_crop_under_treevill_root below.
    rec = bt.treevill_record(
        "Ashore", KG, TAX, ELIGIBLE, fake_embed_fn, idx.add, treevill_root="",
    )

    # ids / namespacing
    assert rec["global_id"] == "treevill:Ashore"
    assert rec["dataset"] == "treevill"
    assert rec["species_key"] == "Ashore"
    assert rec["common_name"] == "Ashore"
    assert rec["scientific_name"] == "Ashorea testis"
    assert rec["taxonomy"] == {"genus": "Ashorea", "family": "Testaceae"}
    assert {"type": "instance_of", "target": "treevill:genus:Ashorea"} in rec["relations"]
    assert {"type": "instance_of", "target": "treevill:family:Testaceae"} in rec["relations"]

    # medoid = centroid-nearest of the 3 eligible embeddings -> p3.jpg
    assert rec["medoid"]["image_path"] == "p3.jpg"
    assert rec["medoid"]["k_images"] == 3
    assert rec["medoid"]["selection"] == "siglip-centroid-nearest"
    assert isinstance(rec["medoid"]["embedding_ref"], int)

    # candidates cover every eligible path, each with its own embedding_ref
    assert [c["image_path"] for c in rec["candidates"]] == ELIGIBLE
    assert all(isinstance(c["embedding_ref"], int) for c in rec["candidates"])

    # part_crops typed by part name, one embedding_ref each
    crops_by_type = {c["part_type"]: c for c in rec["part_crops"]}
    assert set(crops_by_type) == {"leaf", "bark"}
    assert crops_by_type["leaf"]["image_path"] == "crops/leaf.png"
    assert crops_by_type["bark"]["image_path"] == "crops/bark.png"
    assert all(isinstance(c["embedding_ref"], int) for c in rec["part_crops"])

    # attributes: built directly, not-visible dropped, null counts (not fabricated)
    assert rec["attributes"] == {
        "leaf.shape": {"value": "ovate", "support": None, "visible_count": None, "source": "vision"},
        "bark.color": {"value": "brown", "support": None, "visible_count": None, "source": "vision"},
    }

    # index_add seam: 3 candidates + 1 medoid + 2 crops = 6 calls, correctly kinded
    kinds = [meta["kind"] for _, meta in idx.calls]
    assert kinds.count("candidate") == 3
    assert kinds.count("medoid") == 1
    assert kinds.count("part_crop") == 2
    for _, meta in idx.calls:
        assert meta["global_id"] == "treevill:Ashore"

    # the medoid's indexed vector is p3's own embedding, not the centroid
    medoid_calls = [v for v, meta in idx.calls if meta["kind"] == "medoid"]
    assert np.allclose(medoid_calls[0], VECS["p3.jpg"])


def test_load_taxonomy_reads_real_file_without_import():
    tax = bt.load_taxonomy()
    assert tax["Ashok"] == ("Saraca asoca", "Fabaceae")
    assert isinstance(tax, dict) and len(tax) > 10


# --- fix-wave-2 I1: exemplar_crop resolved against treevill_root -----------

def test_treevill_record_resolves_exemplar_crop_under_treevill_root():
    # kg.json's exemplar_crop ("crops/leaf.png") is relative to the kg_test
    # repo root, NOT rag-regen -- treevill_record must join it against
    # treevill_root before embedding/indexing it, exactly like a real
    # "outputs/Akashmoni/crops/leaf.png" under kg_test.
    root = "/some/kg_test/root"
    resolved_vecs = dict(VECS)
    resolved_vecs[f"{root}/crops/leaf.png"] = VECS["crops/leaf.png"]
    resolved_vecs[f"{root}/crops/bark.png"] = VECS["crops/bark.png"]

    seen_paths = []

    def embed_fn_recording_paths(paths):
        seen_paths.extend(paths)
        return np.array([resolved_vecs[p] for p in paths], dtype="float32")

    idx = FakeIndex()
    rec = bt.treevill_record(
        "Ashore", KG, TAX, ELIGIBLE, embed_fn_recording_paths, idx.add,
        treevill_root=root,
    )

    # the raw eligible-pool paths are never joined (the manifest layer
    # already resolves those) -- only exemplar_crop is
    assert seen_paths[:3] == ELIGIBLE
    assert f"{root}/crops/leaf.png" in seen_paths
    assert f"{root}/crops/bark.png" in seen_paths

    crops_by_type = {c["part_type"]: c for c in rec["part_crops"]}
    assert crops_by_type["leaf"]["image_path"] == f"{root}/crops/leaf.png"
    assert crops_by_type["bark"]["image_path"] == f"{root}/crops/bark.png"

    # the index_add seam is called with the resolved path too, not the raw one
    crop_paths_indexed = {
        meta["path"] for _, meta in idx.calls if meta["kind"] == "part_crop"
    }
    assert crop_paths_indexed == {f"{root}/crops/leaf.png", f"{root}/crops/bark.png"}


def test_treevill_record_merges_provenance_extra():
    idx = FakeIndex()
    extra = {
        "source_split": None,
        "build_manifest": {"path": None, "sha256": "deadbeef"},
        "excluded_images": [{"path": "dup.jpg", "reason": "duplicate"}],
    }
    rec = bt.treevill_record(
        "Ashore", KG, TAX, ELIGIBLE, fake_embed_fn, idx.add, treevill_root="",
        provenance_extra=extra,
    )
    assert rec["provenance"]["source_split"] is None
    assert rec["provenance"]["build_manifest"] == {"path": None, "sha256": "deadbeef"}
    assert rec["provenance"]["excluded_images"] == [{"path": "dup.jpg", "reason": "duplicate"}]
    # the adapter's own fields survive the merge
    assert rec["provenance"]["kg_concept"] == "Ashore"
