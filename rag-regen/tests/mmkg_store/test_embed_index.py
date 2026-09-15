import numpy as np, pytest
faiss = pytest.importorskip("faiss")
from ragregen.mmkg_store import embed_index as ei


def test_add_search_roundtrip_and_config_guard(tmp_path):
    b = ei.IndexBuilder()
    r0 = b.add(np.array([1.,0.],dtype="float32"), {"global_id":"d:x","kind":"medoid","path":"x.jpg"})
    r1 = b.add(np.array([0.,1.],dtype="float32"), {"global_id":"d:y","kind":"medoid","path":"y.jpg"})
    # (dim guard is checked on real 1152 vectors in the live build; here we assert id + meta wiring)
    b.save(str(tmp_path))
    idx, side = ei.load_index(str(tmp_path))
    hits = ei.search(idx, side, np.array([1.,0.],dtype="float32"), k=1)
    assert hits[0][0] == r0 and hits[0][2]["global_id"] == "d:x"
    # config guard
    import json; s=json.load(open(tmp_path/"sidecar.json")); s["config"]["encoder"]="other"; json.dump(s,open(tmp_path/"sidecar.json","w"))
    with pytest.raises(ValueError): ei.load_index(str(tmp_path))
