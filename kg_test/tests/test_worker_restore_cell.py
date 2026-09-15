import numpy as np
from PIL import Image
from graft.graph_schema import PartTypeGraph, PartInstanceRec, PartTypeRec
from graft import worker_restore_cell as w


def _setup(tmp_path):
    insts, held = [], []
    for c in ("C1", "C2"):
        for ri in range(3):
            cp = str(tmp_path / f"{c}_leaf_{ri}.png"); Image.new("RGB", (16, 16)).save(cp)
            v = np.random.default_rng(hash((c, ri)) % 999).normal(size=8); v /= np.linalg.norm(v)
            insts.append(PartInstanceRec(f"{c}::leaf::{ri}", c, "leaf", f"{c}/r{ri}", cp, v.tolist()))
    hub = PartTypeRec(0, "leaf", [i.id for i in insts], insts[0].siglip2, 0.9)
    graph = PartTypeGraph(part_instances=insts, part_types=[hub])
    for ri in range(2):
        ref = tmp_path / f"C1_held_{ri}.png"; Image.new("RGB", (64, 64), (200, 100, 50)).save(ref)
        m = np.zeros((64, 64), bool); m[10:30, 12:34] = True
        mp = str(tmp_path / f"C1_leaf_held_{ri}.npy"); np.save(mp, m)
        cp = str(tmp_path / f"C1_leaf_heldcrop_{ri}.png"); Image.new("RGB", (22, 20)).save(cp)
        held.append({"concept": "C1", "part": "leaf", "ref_path": str(ref), "box": [12, 10, 34, 30],
                     "mask_path": mp, "crop_path": cp, "siglip2": insts[0].siglip2})
    return graph, held


def test_build_restore_inputs_masks_and_excludes_target(tmp_path):
    graph, held = _setup(tmp_path)
    from graft import restore
    k_eff = restore.compute_k_eff(graph, "C1", "leaf", k=4)
    out = w.build_restore_inputs(graph, held, "C1", "leaf", target_idx=0,
                                 build_P=1, condition="hub", k_eff=k_eff, draw=0)
    base = np.asarray(Image.open(held[0]["ref_path"]).convert("RGB"))
    masked = np.asarray(out["masked_img"])
    assert not (base[10:30, 12:34] == masked[10:30, 12:34]).all()
    assert (base[0:10, :] == masked[0:10, :]).all()
    assert len(out["scoring_crops"]) == 1
    assert len(out["ref_imgs"]) == 1 + k_eff
    assert all(bid.split("::")[0] != "C1" for bid in out["borrowed_ids"])


def test_build_restore_inputs_isolated_build0_has_no_refs(tmp_path):
    graph, held = _setup(tmp_path)
    out = w.build_restore_inputs(graph, held, "C1", "leaf", target_idx=0,
                                 build_P=0, condition="isolated", k_eff=2, draw=0)
    assert out["ref_imgs"] == []


def test_restore_row_schema():
    r = w.restore_row("C1", "leaf", 1, "hub", 0, 2, {"dino": 0.5, "siglip2": 0.4, "clip_i": 0.3})
    assert r == {"concept": "C1", "part": "leaf", "level": 1, "draw": 0, "condition": "hub",
                 "k_eff": 2, "dino": 0.5, "siglip2": 0.4, "clip_i": 0.3}
