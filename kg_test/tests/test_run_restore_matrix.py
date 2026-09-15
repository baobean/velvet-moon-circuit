import numpy as np
from graft.graph_schema import PartTypeGraph, PartInstanceRec, PartTypeRec
from graft import run_restore_resident as R


def _graph():
    insts = []
    for c in ("C1", "C2"):
        for ri in range(3):
            v = np.random.default_rng(hash((c, ri)) % 999).normal(size=8); v /= np.linalg.norm(v)
            insts.append(PartInstanceRec(f"{c}::leaf::{ri}", c, "leaf", f"{c}/r{ri}", "x.png", v.tolist()))
    return PartTypeGraph(part_instances=insts,
                         part_types=[PartTypeRec(0, "leaf", [i.id for i in insts], insts[0].siglip2, 0.9)])


def test_iter_cells_skips_nonborrowable_and_clamps_levels():
    graph = _graph()
    held = [{"concept": "C1", "part": "leaf", "ref_path": "r", "box": [0, 0, 4, 4],
             "mask_path": "m", "crop_path": "c", "siglip2": graph.part_instances[0].siglip2}]
    cells = list(R.iter_restore_cells(graph, held, concepts=["C1"], levels=[0, 1, 2, 4],
                                      n_draws=2, conditions=["isolated", "random", "rawnn", "hub"], k=4))
    got_levels = sorted({lvl for (_c, _p, lvl, _d, _cond, _t, _k) in cells})
    assert got_levels == [0, 1, 2, 3]                 # C1 has 3 own leaf crops -> level 4 clamps to 3
    assert {cond for (_c, _p, _l, _d, cond, _t, _k) in cells} == {"isolated", "random", "rawnn", "hub"}
    assert all(k > 0 for (*_rest, k) in cells)
