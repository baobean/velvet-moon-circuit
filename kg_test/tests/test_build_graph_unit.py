import numpy as np
from graft.graph_schema import PartInstanceRec
from graft.build_graph import assemble_graph

def _rec(cid, part, ri, vec):
    return PartInstanceRec(id=f"{cid}::{part}::{ri}", concept=cid, part=part,
                           ref_path=f"{cid}{ri}.jpg", crop_path="x.png",
                           siglip2=list(vec/np.linalg.norm(vec)))

def test_assemble_builds_valid_cross_concept_leaf_hub():
    rng = np.random.default_rng(1)
    insts = []
    # 12 leaves near [1,0,0] across 3 concepts -> one valid hub (>=2 concepts, coherent)
    for i, c in enumerate(["A","A","A","A","B","B","B","B","C","C","C","C"]):
        insts.append(_rec(c, "leaf", i, np.array([1.,0,0]) + 0.01*rng.standard_normal(3)))
    g = assemble_graph(insts)
    leaf_hubs = [h for h in g.part_types if h.part == "leaf"]
    assert len(leaf_hubs) == 1
    assert len({insts[i].concept for i in range(12)}) == 3
    assert leaf_hubs[0].coherence >= 0.5
