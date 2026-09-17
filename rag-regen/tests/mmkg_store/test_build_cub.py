import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from ragregen.mmkg_store import build_cub as bc
from ragregen.mmkg_store.store import Store


# --- unit tests: part-type grouping -----------------------------------------

def test_all_15_cub_parts_map_to_exactly_one_group():
    all_names = {
        name for names in bc.PART_GROUPS.values() for name in names
    }
    assert all_names == {
        "back", "beak", "belly", "breast", "crown", "forehead", "left eye",
        "left leg", "left wing", "nape", "right eye", "right leg",
        "right wing", "tail", "throat",
    }
    assert set(bc.PART_NAME_TO_GROUP) == all_names
    assert bc.PART_NAME_TO_GROUP["beak"] == "head"
    assert bc.PART_NAME_TO_GROUP["left wing"] == "wing"
    assert bc.PART_NAME_TO_GROUP["right leg"] == "leg"


def test_common_name_strips_numeric_prefix_and_underscores():
    assert bc._common_name("001.Black_footed_Albatross") == "Black footed Albatross"
    assert bc._common_name("NoDot") == "NoDot"


# --- unit tests: source-photo selection -------------------------------------

def test_pick_source_image_picks_most_visible_groups():
    part_locs = {
        1: {"beak": (1, 1, True)},  # 1 group (head)
        2: {"left wing": (1, 1, True), "right wing": (1, 1, True), "tail": (1, 1, True)},  # 2 groups
        3: {},  # 0 groups
    }
    assert bc._pick_source_image([1, 2, 3], part_locs) == 2


def test_pick_source_image_ties_broken_by_lowest_id():
    # both image 5 and image 2 see exactly 2 groups -> lowest id (2) wins
    part_locs = {
        2: {"beak": (1, 1, True), "tail": (1, 1, True)},           # head, tail
        5: {"left leg": (1, 1, True), "right leg": (1, 1, True)},  # leg, back(no) -> leg only... see below
    }
    # correct image 5 to also carry 2 groups (leg + back) to make it a real tie
    part_locs[5]["back"] = (1, 1, True)
    assert bc._pick_source_image([2, 5], part_locs) == 2


def test_pick_source_image_ignores_non_visible_points():
    part_locs = {
        1: {"beak": (10, 10, False)},  # present but not visible -> 0 groups
        2: {"beak": (10, 10, True)},   # 1 group
    }
    assert bc._pick_source_image([1, 2], part_locs) == 2


# --- unit tests: crop-box derivation / clamping -----------------------------

def test_clamp_axis_no_clamp_needed():
    lo, hi = bc._clamp_axis(center=50, side=20, size=100)
    assert (lo, hi) == (40, 60)


def test_clamp_axis_shifts_when_over_low_edge():
    lo, hi = bc._clamp_axis(center=5, side=20, size=100)
    assert lo == 0
    assert hi == 20  # shifted right, not shrunk


def test_clamp_axis_shifts_when_over_high_edge():
    lo, hi = bc._clamp_axis(center=95, side=20, size=100)
    assert hi == 100
    assert lo == 80  # shifted left, not shrunk


def test_clamp_axis_uses_full_extent_when_side_exceeds_size():
    lo, hi = bc._clamp_axis(center=50, side=150, size=100)
    assert (lo, hi) == (0, 100)


def test_crop_box_for_group_none_when_no_visible_subparts():
    part_locs = {"beak": (10, 10, False)}
    box = bc._crop_box_for_group(part_locs, {"beak", "crown"}, (0, 0, 60, 50), (100, 80))
    assert box is None


def test_crop_box_for_group_averages_visible_points():
    part_locs = {"left wing": (30.0, 40.0, True), "right wing": (60.0, 42.0, True)}
    box = bc._crop_box_for_group(part_locs, {"left wing", "right wing"}, (0, 0, 60, 50), (100, 80))
    # side = max(32, round(0.45*60)) = max(32, 27) = 32 -> half=16
    # center = (45, 41) -> box = (29, 25, 61, 57), well within [0,100]x[0,80]
    assert box == (29, 25, 61, 57)


# --- hermetic fixture: tiny CUB-layout tree + real (fake-embedded) build ----

IMG_W, IMG_H = 100, 80
BBOX = (10.0, 10.0, 60.0, 50.0)  # same whole-bird box for every fixture image

PARTS_TXT = """\
1 back
2 beak
3 belly
4 breast
5 crown
6 forehead
7 left eye
8 left leg
9 left wing
10 nape
11 right eye
12 right leg
13 right wing
14 tail
15 throat
"""


def _write_cub_fixture(root: Path) -> None:
    (root / "parts").mkdir(parents=True)
    images_dir = root / "images"

    species = {
        "001.Test_Sparrow": [1, 2, 3],
        "002.Test_Warbler": [4, 5, 6],
    }

    images_lines = []
    labels_lines = []
    bbox_lines = []
    for cid, (class_name, image_ids) in enumerate(species.items(), start=1):
        species_dir = images_dir / class_name
        species_dir.mkdir(parents=True)
        for n, iid in enumerate(image_ids, start=1):
            fname = f"img{n}.jpg"
            # distinct-ish solid color per image, real on-disk image
            color = (10 * iid % 256, 20 * iid % 256, 30 * iid % 256)
            Image.new("RGB", (IMG_W, IMG_H), color=color).save(species_dir / fname)
            images_lines.append(f"{iid} {class_name}/{fname}")
            labels_lines.append(f"{iid} {cid}")
            bbox_lines.append(f"{iid} {BBOX[0]} {BBOX[1]} {BBOX[2]} {BBOX[3]}")

    (root / "images.txt").write_text("\n".join(images_lines) + "\n")
    (root / "image_class_labels.txt").write_text("\n".join(labels_lines) + "\n")
    (root / "bounding_boxes.txt").write_text("\n".join(bbox_lines) + "\n")
    (root / "classes.txt").write_text(
        "1 001.Test_Sparrow\n2 002.Test_Warbler\n"
    )
    (root / "parts" / "parts.txt").write_text(PARTS_TXT)

    # part_locs.txt: sparse -- only the images we care about get entries;
    # an image with no lines here simply has zero visible groups.
    part_locs_lines = [
        # species 1, image 1: head (beak+crown), wing (l/r wing), tail -> 3 groups
        "1 2 40.0 20.0 1",
        "1 5 42.0 15.0 1",
        "1 9 30.0 40.0 1",
        "1 13 60.0 42.0 1",
        "1 14 50.0 70.0 1",
        # species 1, image 2: leg (l/r leg), back, tail -> 3 groups (TIE with image 1;
        # lowest image_id (1) must win as the source photo)
        "2 8 25.0 55.0 1",
        "2 12 28.0 58.0 1",
        "2 1 45.0 25.0 1",
        "2 14 55.0 65.0 1",
        # species 2, image 4: back only -> 1 group
        "4 1 50.0 20.0 1",
        # species 2, image 5: leg (l/r leg), breast (breast+belly) -> 2 groups (wins)
        "5 8 20.0 60.0 1",
        "5 12 25.0 62.0 1",
        "5 4 45.0 35.0 1",
        "5 3 48.0 38.0 1",
        # species 2, image 6: no entries -> 0 groups
    ]
    (root / "parts" / "part_locs.txt").write_text("\n".join(part_locs_lines) + "\n")


def _fake_embed_fn(paths):
    """Deterministic, content-independent embedding: md5(path) -> 2D vector.

    Works for arbitrary paths (including derived crop paths the test never
    names up front) without a lookup table.
    """
    vecs = []
    for p in paths:
        h = hashlib.md5(str(p).encode()).hexdigest()
        v1 = int(h[:8], 16) % 1000 / 1000.0
        v2 = int(h[8:16], 16) % 1000 / 1000.0
        vecs.append([v1, v2])
    return np.array(vecs, dtype="float32")


def test_build_cub_store_hermetic_fixture(tmp_path):
    cub_root = tmp_path / "cub"
    out_dir = tmp_path / "store_out"
    cub_root.mkdir()
    _write_cub_fixture(cub_root)

    records = bc.build_cub_store(cub_root, out_dir, _fake_embed_fn)
    assert len(records) == 2

    loaded = Store.load(out_dir)
    all_ids = loaded.query()
    assert set(all_ids) == {"cub:001.Test_Sparrow", "cub:002.Test_Warbler"}

    sparrow = loaded.get("cub:001.Test_Sparrow")
    warbler = loaded.get("cub:002.Test_Warbler")

    # basic record shape
    for rec in (sparrow, warbler):
        assert rec["dataset"] == "cub"
        assert rec["scientific_name"] is None
        # CUB has no genus/family -- minimum acceptable taxonomy, no fabricated
        # hubs (schema.build_record only emits instance_of relations when
        # "genus"/"family" keys are present in taxonomy).
        assert rec["taxonomy"] == {}
        assert rec["relations"] == []
        assert rec["attributes"] == {}
        # medoid present and drawn from this species' own images
        assert rec["medoid"]["image_path"]
        assert rec["medoid"]["selection"] == "siglip-centroid-nearest"
        assert rec["medoid"]["k_images"] == 3
        assert isinstance(rec["medoid"]["embedding_ref"], int)
        assert len(rec["candidates"]) == 3

    assert sparrow["common_name"] == "Test Sparrow"
    assert warbler["common_name"] == "Test Warbler"

    # part_crops: species 1's source photo (tie broken to lowest id -> image 1)
    # exposes head/wing/tail; species 2's source photo (image 5) exposes leg/breast
    sparrow_types = {c["part_type"] for c in sparrow["part_crops"]}
    warbler_types = {c["part_type"] for c in warbler["part_crops"]}
    assert sparrow_types == {"head", "wing", "tail"}
    assert warbler_types == {"leg", "breast"}

    assert rec["provenance"]  # sanity: provenance block present (checked generically above via rec loop)
    assert sparrow["provenance"]["source_image_id"] == 1
    assert warbler["provenance"]["source_image_id"] == 5

    # every part crop is a real, on-disk, croppable image with an embedding_ref
    for rec in (sparrow, warbler):
        for crop in rec["part_crops"]:
            crop_path = Path(crop["image_path"])
            assert crop_path.exists()
            with Image.open(crop_path) as im:
                assert im.size[0] > 0 and im.size[1] > 0
            assert isinstance(crop["embedding_ref"], int)

    # FAISS index + sidecar were actually persisted and are loadable
    assert (out_dir / "index.faiss").exists()
    assert (out_dir / "sidecar.json").exists()
    # 2 species x (3 candidates + 1 medoid) + 3 sparrow crops + 2 warbler crops = 13
    assert len(loaded._sidecar["entries"]) == 13


def test_build_cub_records_tail_crop_is_clamped_not_shrunk(tmp_path):
    # Direct check on the sparrow tail crop: point (50, 70) with side=32 would
    # naturally run to y in [54, 86], which overflows img_h=80 -- the box must
    # be shifted up to [48, 80], not shrunk.
    cub_root = tmp_path / "cub"
    out_dir = tmp_path / "store_out"
    cub_root.mkdir()
    _write_cub_fixture(cub_root)

    bc.build_cub_store(cub_root, out_dir, _fake_embed_fn)
    loaded = Store.load(out_dir)
    sparrow = loaded.get("cub:001.Test_Sparrow")
    tail_crop = next(c for c in sparrow["part_crops"] if c["part_type"] == "tail")
    with Image.open(tail_crop["image_path"]) as im:
        assert im.size == (32, 32)  # full side preserved, not shrunk by clamping
