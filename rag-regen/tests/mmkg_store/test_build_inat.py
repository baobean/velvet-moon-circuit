import numpy as np

from ragregen.mmkg import inat_pilot
from ragregen.mmkg_store import build_inat as bi


# --- fakes ---------------------------------------------------------------

# 3 synthetic species (NOT the locked 25) -- exercises the production
# selection seam via monkeypatch, per the brief.
SPECIES = [
    {"id": 1, "name": "Aveus testis", "common_name": "Test Warbler",
     "family": "Testidae", "genus": "Aveus", "image_dir_name": "00001_Testidae_Aveus_testis"},
    {"id": 2, "name": "Aveus fictus", "common_name": None,
     "family": "Testidae", "genus": "Aveus", "image_dir_name": "00002_Testidae_Aveus_fictus"},
    {"id": 42, "name": "Mocka avis", "common_name": "Mock Finch",
     "family": "Mockidae", "genus": "Mocka", "image_dir_name": "00042_Mockidae_Mocka_avis"},
]

VECS = {
    "sp1/a.jpg": [1.0, 0.0],
    "sp1/b.jpg": [0.0, 1.0],
    "sp1/c.jpg": [0.9, 0.1],   # nearest to the 3-point centroid -> medoid
    "sp2/a.jpg": [1.0, 0.0],
    "sp2/b.jpg": [1.0, 0.0],
    "sp2/c.jpg": [0.0, 1.0],   # centroid (0.667,0.333); a/b tie for nearest -> first (a) wins via argmax
    "sp42/a.jpg": [0.2, 0.8],
    "sp42/b.jpg": [0.3, 0.7],
    "sp42/c.jpg": [0.9, 0.1],
}

MANIFEST = {
    "00001": {"eligible": ["sp1/a.jpg", "sp1/b.jpg", "sp1/c.jpg"]},
    "00002": {"eligible": ["sp2/a.jpg", "sp2/b.jpg", "sp2/c.jpg"]},
    "00042": {"eligible": ["sp42/a.jpg", "sp42/b.jpg", "sp42/c.jpg"]},
}

# Reads: consistent "red" primary_plumage_color for all 3 sp1 images (is_target
# True, visible_count=3, support=3); sp2/sp42 vary so their consensus is absent
# for that slot but present for another.
READS = {
    "sp1/a.jpg": {"primary_plumage_color": "red", "secondary_plumage_color": "not visible",
                  "plumage_pattern": "not visible", "bill_shape": "not visible",
                  "bill_color": "not visible", "distinctive_markings": "not visible"},
    "sp1/b.jpg": {"primary_plumage_color": "red", "secondary_plumage_color": "not visible",
                  "plumage_pattern": "not visible", "bill_shape": "not visible",
                  "bill_color": "not visible", "distinctive_markings": "not visible"},
    "sp1/c.jpg": {"primary_plumage_color": "red", "secondary_plumage_color": "not visible",
                  "plumage_pattern": "not visible", "bill_shape": "not visible",
                  "bill_color": "not visible", "distinctive_markings": "not visible"},
    "sp2/a.jpg": {"primary_plumage_color": "blue", "secondary_plumage_color": "not visible",
                  "plumage_pattern": "not visible", "bill_shape": "not visible",
                  "bill_color": "not visible", "distinctive_markings": "not visible"},
    "sp2/b.jpg": {"primary_plumage_color": "blue", "secondary_plumage_color": "not visible",
                  "plumage_pattern": "not visible", "bill_shape": "not visible",
                  "bill_color": "not visible", "distinctive_markings": "not visible"},
    "sp2/c.jpg": {"primary_plumage_color": "blue", "secondary_plumage_color": "not visible",
                  "plumage_pattern": "not visible", "bill_shape": "not visible",
                  "bill_color": "not visible", "distinctive_markings": "not visible"},
    "sp42/a.jpg": {"primary_plumage_color": "not visible", "secondary_plumage_color": "not visible",
                   "plumage_pattern": "not visible", "bill_shape": "not visible",
                   "bill_color": "not visible", "distinctive_markings": "not visible"},
    "sp42/b.jpg": {"primary_plumage_color": "not visible", "secondary_plumage_color": "not visible",
                   "plumage_pattern": "not visible", "bill_shape": "not visible",
                   "bill_color": "not visible", "distinctive_markings": "not visible"},
    "sp42/c.jpg": {"primary_plumage_color": "not visible", "secondary_plumage_color": "not visible",
                   "plumage_pattern": "not visible", "bill_shape": "not visible",
                   "bill_color": "not visible", "distinctive_markings": "not visible"},
}


def fake_embed_fn(paths):
    return np.array([VECS[p] for p in paths], dtype="float32")


def fake_read_fn(path):
    return READS[path]


class FakeIndex:
    def __init__(self):
        self.calls = []

    def add(self, vec, meta):
        ref = len(self.calls)
        self.calls.append((np.asarray(vec, dtype="float32"), dict(meta)))
        return ref


def test_inat_records_uses_selection_seam_and_builds_medoid_attrs(monkeypatch):
    monkeypatch.setattr(inat_pilot, "select_pilot_species", lambda val_json: SPECIES)

    idx = FakeIndex()
    records = bi.inat_records(
        val_json={"fake": True}, data_root="/data", manifest=MANIFEST,
        embed_fn=fake_embed_fn, read_fn=fake_read_fn, index_add=idx.add,
    )

    # exactly 3 records -- the synthetic selection, not the locked 25
    assert len(records) == 3

    by_key = {r["species_key"] for r in records}
    assert by_key == {"00001", "00002", "00042"}

    for rec in records:
        assert rec["dataset"] == "inat"
        assert rec["part_crops"] == []
        assert rec["global_id"] == f"inat:{rec['species_key']}"
        for rel in rec["relations"]:
            assert rel["target"].startswith("inat:")

    by_species_key = {r["species_key"]: r for r in records}

    # taxonomy / relations from the val category
    rec1 = by_species_key["00001"]
    assert rec1["taxonomy"] == {"genus": "Aveus", "family": "Testidae"}
    assert {"type": "instance_of", "target": "inat:genus:Aveus"} in rec1["relations"]
    assert {"type": "instance_of", "target": "inat:family:Testidae"} in rec1["relations"]
    assert rec1["scientific_name"] == "Aveus testis"
    assert rec1["common_name"] == "Test Warbler"

    # common_name may be None
    rec2 = by_species_key["00002"]
    assert rec2["common_name"] is None

    # medoid = deterministic centroid-nearest of the eligible embeddings
    assert rec1["medoid"]["image_path"] == "sp1/c.jpg"
    assert rec1["medoid"]["k_images"] == 3
    assert rec1["medoid"]["selection"] == "siglip-centroid-nearest"
    assert isinstance(rec1["medoid"]["embedding_ref"], int)

    # candidates cover the full eligible pool, each with its own embedding_ref
    assert [c["image_path"] for c in rec1["candidates"]] == MANIFEST["00001"]["eligible"]
    assert all(isinstance(c["embedding_ref"], int) for c in rec1["candidates"])

    # bird attributes: present (is_target consensus), integer counts, source tag
    assert "primary_plumage_color" in rec1["attributes"]
    attr = rec1["attributes"]["primary_plumage_color"]
    assert attr["value"] == "red"
    assert attr["support"] == 3
    assert attr["visible_count"] == 3
    assert isinstance(attr["support"], int)
    assert isinstance(attr["visible_count"], int)
    assert attr["source"] == "inat-consensus"

    rec42 = by_species_key["00042"]
    # no visible reads for any slot -> no target attributes at all
    assert rec42["attributes"] == {}

    # index_add seam: 3 species x (3 candidates + 1 medoid) = 12 calls
    assert len(idx.calls) == 12
    for _, meta in idx.calls:
        assert meta["global_id"].startswith("inat:")
        assert meta["kind"] in {"candidate", "medoid"}


# --- fix-wave-2 I2/I3 -------------------------------------------------------

def test_inat_records_skips_species_absent_from_manifest(monkeypatch):
    # build.py omits a species from the filtered manifest entirely when its
    # eligible pool ended up empty (I2) -- inat_records must skip it, not
    # KeyError, and never call the adapter with an empty pool.
    monkeypatch.setattr(inat_pilot, "select_pilot_species", lambda val_json: SPECIES)

    manifest_missing_00042 = {k: v for k, v in MANIFEST.items() if k != "00042"}
    idx = FakeIndex()

    records = bi.inat_records(
        val_json={"fake": True}, data_root="/data", manifest=manifest_missing_00042,
        embed_fn=fake_embed_fn, read_fn=fake_read_fn, index_add=idx.add,
    )

    assert {r["species_key"] for r in records} == {"00001", "00002"}


def test_inat_record_merges_provenance_extra(monkeypatch):
    monkeypatch.setattr(inat_pilot, "select_pilot_species", lambda val_json: SPECIES)
    idx = FakeIndex()
    extra_by_species = {
        "00001": {
            "source_split": "build_3_eval_3",
            "build_manifest": {"path": None, "sha256": "cafef00d"},
            "excluded_images": [{"path": "sp1/dup.jpg", "reason": "duplicate"}],
        },
    }

    records = bi.inat_records(
        val_json={"fake": True}, data_root="/data", manifest=MANIFEST,
        embed_fn=fake_embed_fn, read_fn=fake_read_fn, index_add=idx.add,
        provenance_extra_by_species=extra_by_species,
    )

    by_key = {r["species_key"]: r for r in records}
    rec1 = by_key["00001"]
    assert rec1["provenance"]["source_split"] == "build_3_eval_3"
    assert rec1["provenance"]["build_manifest"] == {"path": None, "sha256": "cafef00d"}
    assert rec1["provenance"]["excluded_images"] == [{"path": "sp1/dup.jpg", "reason": "duplicate"}]
    # the adapter's own fields survive the merge
    assert rec1["provenance"]["category_id"] == 1

    # a species with no entry in the extras map gets no extra fields at all
    rec42 = by_key["00042"]
    assert "build_manifest" not in rec42["provenance"]
