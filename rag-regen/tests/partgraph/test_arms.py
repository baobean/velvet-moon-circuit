import pytest
from PIL import Image
from ragregen.partgraph.arms import reference_for, ARMS


class _Store:
    def __init__(self, rec, medoid=None): self._rec, self._medoid = rec, medoid
    def get(self, gid): return self._rec


def _crop(tmp_path, name, color):
    p = tmp_path / name
    Image.new("RGB", (40, 40), color).save(p)
    return str(p)


def test_arms_registry():
    assert ARMS == ("single_medoid", "partgraph")


def test_partgraph_arm_composes(tmp_path):
    rec = {"part_crops": [
        {"part_type": "leaf", "image_path": _crop(tmp_path, "l.png", (0, 128, 0)), "embedding_ref": 0},
        {"part_type": "bark", "image_path": _crop(tmp_path, "b.png", (128, 64, 0)), "embedding_ref": 1},
    ]}
    out = reference_for("partgraph", _Store(rec), "ds:sp_A", tile=64)
    # 2 parts -> cols=ceil(sqrt(2))=2, rows=ceil(2/2)=1 -> a 2x1 grid of 64
    assert out.mode == "RGB" and out.size == (128, 64)


def test_partgraph_no_parts_raises(tmp_path):
    with pytest.raises(LookupError):
        reference_for("partgraph", _Store({"part_crops": []}), "ds:sp_A")


def test_unknown_arm_raises(tmp_path):
    with pytest.raises(ValueError):
        reference_for("magic", _Store({}), "ds:sp_A")
