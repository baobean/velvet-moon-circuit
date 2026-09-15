from ragregen.mmkg import schema as s

def test_norm_and_slots():
    assert s.SLOTS[0] == "primary_color" and len(s.SLOTS) == 6
    assert s.norm(" Rough ") == "rough"
    assert "" in s.NOT_VISIBLE and "n/a" in s.NOT_VISIBLE

def test_target_attributes_consensus_and_decidable():
    # primary_color: red x3, blue x1 over 4 visible -> thr=max(2,ceil(2.0))=2 -> red target
    # surface_texture: rough x2 over 2 visible -> V_vis=2 (<3) -> NOT a target
    reads = [
        {"primary_color": "red",  "surface_texture": "rough"},
        {"primary_color": "Red ", "surface_texture": "rough"},
        {"primary_color": "red",  "surface_texture": "not visible"},
        {"primary_color": "blue", "surface_texture": "none"},
    ]
    t = s.target_attributes(reads)
    assert t["primary_color"]["is_target"] is True
    assert t["primary_color"]["value"] == "red" and t["primary_color"]["support"] == 3
    assert t["primary_color"]["visible_count"] == 4
    assert t["surface_texture"]["is_target"] is False   # V_vis=2 < 3
    assert s.n_targets(t) == 1 and s.decidable(t) is False
