import numpy as np
from graft.graph_schema import PartInstanceRec, PartTypeRec, PartTypeGraph
from graft.build_graph import top_centroid_members

def test_top_centroid_members_picks_nearest():
    c = [1.0, 0.0]
    insts = [
        PartInstanceRec("m0","A","leaf","a.jpg","a.png", [0.99, 0.14]),   # near centroid
        PartInstanceRec("m1","B","leaf","b.jpg","b.png", [0.98, 0.20]),   # near
        PartInstanceRec("m2","C","leaf","c.jpg","c.png", [0.20, 0.98]),   # far
    ]
    hub = PartTypeRec(id=0, part="leaf", member_ids=["m0","m1","m2"], centroid=c, coherence=0.7)
    g = PartTypeGraph(part_instances=insts, part_types=[hub])
    assert top_centroid_members(g, hub, k=2) == ["m0", "m1"]
