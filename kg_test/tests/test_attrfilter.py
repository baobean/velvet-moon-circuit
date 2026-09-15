from graft.attrfilter import is_generic, count_generic

def test_flags_generic_values():
    for v in ["green", "full", "tree", "not visible", "normal", ""]:
        assert is_generic(v) is True

def test_keeps_discriminative_values():
    for v in ["symmetric candelabra silhouette", "stiff triangular scales", "reddish flaking bark"]:
        assert is_generic(v) is False

def test_count_generic():
    assert count_generic(["green", "stiff triangular scales", "full"]) == 2
