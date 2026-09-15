from ragregen.mmkg_store import reference_source as rs
from PIL import Image


class FakeStore:
    def __init__(self, recs):
        self.recs = recs
        self.calls = []

    def get(self, gid):
        self.calls.append(gid)
        return self.recs.get(gid)

    def nearest_crops(self, *a, **k):
        raise AssertionError("adapter must not call nearest_crops")


def test_returns_medoid_image_and_none(tmp_path):
    m = tmp_path / "m.jpg"
    Image.new("RGB", (8, 8), "green").save(m)
    st = FakeStore({"inat:04486": {"medoid": {"image_path": str(m)}}})
    img = rs.medoid_reference(st, "inat:04486")
    assert img is not None and img.size == (8, 8) and img.mode == "RGB"
    assert rs.medoid_reference(st, "inat:missing") is None  # never a fallback species
    assert st.calls == ["inat:04486", "inat:missing"]  # only get(); nearest_crops would AssertionError
