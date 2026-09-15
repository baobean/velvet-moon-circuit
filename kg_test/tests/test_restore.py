import numpy as np
from graft.graph_schema import PartTypeGraph, PartInstanceRec, PartTypeRec
from graft import restore


def _emb(seed, n=8):
    v = np.random.default_rng(seed).normal(size=n)
    return (v / np.linalg.norm(v)).tolist()


def _graph():
    insts = [
        PartInstanceRec("C1::leaf::0", "C1", "leaf", "C1/r0", "c10.png", _emb(1)),
        PartInstanceRec("C1::leaf::1", "C1", "leaf", "C1/r1", "c11.png", _emb(2)),
        PartInstanceRec("C2::leaf::0", "C2", "leaf", "C2/r0", "c20.png", _emb(3)),
        PartInstanceRec("C2::leaf::1", "C2", "leaf", "C2/r1", "c21.png", _emb(4)),
        PartInstanceRec("C2::leaf::2", "C2", "leaf", "C2/r2", "c22.png", _emb(5)),
    ]
    hub = PartTypeRec(0, "leaf", [i.id for i in insts], _emb(6), 0.9)
    return PartTypeGraph(part_instances=insts, part_types=[hub])


def test_hub_of_and_own_crops():
    g = _graph()
    assert restore.hub_of(g, "C1::leaf::0") == 0
    assert [o.id for o in restore.own_part_crops(g, "C1", "leaf")] == ["C1::leaf::0", "C1::leaf::1"]


def test_k_eff_bounded_by_hub_siblings():
    g = _graph()
    assert restore.compute_k_eff(g, "C1", "leaf", k=4) == 2


def test_select_refs_starves_own_and_matches_borrowed_count():
    g = _graph()
    k_eff = restore.compute_k_eff(g, "C1", "leaf", k=4)
    for cond in ("random", "rawnn", "hub"):
        own, bidx, pool = restore.select_refs(g, "C1", "leaf", build_P=1, condition=cond, k_eff=k_eff, draw=0)
        assert len(own) == 1
        assert len(bidx) == k_eff
        assert all(pool[i].concept != "C1" for i in bidx)
    own0, b0, _ = restore.select_refs(g, "C1", "leaf", build_P=0, condition="isolated", k_eff=k_eff, draw=0)
    assert own0 == [] and b0 == []


def test_assemble_weights_sum_to_one(tmp_path):
    from PIL import Image
    g = _graph()
    for inst in g.part_instances:
        inst.crop_path = str(tmp_path / (inst.id.replace("::", "_") + ".png"))
        Image.new("RGB", (16, 16)).save(inst.crop_path)
    own, bidx, pool = restore.select_refs(g, "C1", "leaf", build_P=1, condition="hub", k_eff=2, draw=0)
    imgs, w = restore.assemble_ref_images(own, bidx, pool, g)
    assert len(imgs) == 1 + 2
    assert abs(sum(w) - 1.0) < 1e-9
