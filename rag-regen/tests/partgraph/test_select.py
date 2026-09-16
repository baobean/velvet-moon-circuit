from pathlib import Path
from PIL import Image
from ragregen.partgraph.select import select_part_images


class _FakeStore:
    def __init__(self, records): self._r = records
    def get(self, gid): return self._r.get(gid)


def _png(tmp_path, name, color):
    p = tmp_path / name
    Image.new("RGB", (32, 32), color).save(p)
    return str(p)


def test_selects_one_image_per_part_type(tmp_path):
    leaf = _png(tmp_path, "leaf.png", (0, 128, 0))
    bark = _png(tmp_path, "bark.png", (128, 64, 0))
    store = _FakeStore({"ds:sp_A": {"part_crops": [
        {"part_type": "leaf", "image_path": leaf, "embedding_ref": 0},
        {"part_type": "bark", "image_path": bark, "embedding_ref": 1},
    ]}})
    got = select_part_images(store, "ds:sp_A")
    assert set(got) == {"leaf", "bark"}
    assert got["leaf"].mode == "RGB" and got["leaf"].size == (32, 32)


def test_missing_record_returns_empty(tmp_path):
    assert select_part_images(_FakeStore({}), "ds:nope") == {}


def test_parts_filter(tmp_path):
    leaf = _png(tmp_path, "leaf.png", (0, 128, 0))
    bark = _png(tmp_path, "bark.png", (128, 64, 0))
    store = _FakeStore({"ds:sp_A": {"part_crops": [
        {"part_type": "leaf", "image_path": leaf, "embedding_ref": 0},
        {"part_type": "bark", "image_path": bark, "embedding_ref": 1},
    ]}})
    got = select_part_images(store, "ds:sp_A", parts=["leaf"])
    assert set(got) == {"leaf"}
