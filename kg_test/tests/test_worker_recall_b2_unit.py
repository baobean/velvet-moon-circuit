# tests/test_worker_recall_b2_unit.py
from graft.worker_recall_b2 import recall_all


def test_recall_all_maps_species_to_attrs():
    calls = []
    def describe(concept):
        calls.append(concept)
        return [f"{concept}-attr1", f"{concept}-attr2"]
    out = recall_all(["Guava", "Mango"], describe)
    assert out == {"Guava": ["Guava-attr1", "Guava-attr2"],
                   "Mango": ["Mango-attr1", "Mango-attr2"]}
    assert calls == ["Guava", "Mango"]
