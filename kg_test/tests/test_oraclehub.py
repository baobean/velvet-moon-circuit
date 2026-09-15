import numpy as np
from graft import oraclehub_run as orh
from graft.oraclehub_analysis import contrast, summarize


def _norm(a): return a / np.linalg.norm(a, axis=-1, keepdims=True)


def _pool_part(vecs, concepts, fam_ids):
    return [{"concept": c, "crop_path": f"/x/{c}.png", "siglip2": _norm(np.asarray(v, float)),
             "fam_id": fid} for v, c, fid in zip(vecs, concepts, fam_ids)]


def test_sibling_rank_and_stratum():
    # own is fam 0; its sibling 'sibA' (fam 0) is embedding-FAR; fam-1 crops are near.
    pool_part = _pool_part(
        [[1, 0, 0], [0.1, 1, 0], [0.99, 0.05, 0], [0.98, 0.1, 0], [0.97, 0.12, 0]],
        ["own", "sibA", "b1", "b2", "b3"], [0, 0, 1, 1, 1])
    q = pool_part[0]["siglip2"]
    br, n = orh.sibling_rank(q, pool_part, "own", 0)
    assert n == 1 and br == 4               # sibA is the 4th-nearest other-concept crop
    assert orh.stratum(4) == "near" and orh.stratum(2) == "near"   # rank <=4 = near
    assert orh.stratum(8) == "mid" and orh.stratum(20) == "far"


def test_oraclehub_recovers_distant_sibling_rawnn_misses():
    pool_part = _pool_part(
        [[1, 0, 0], [0.1, 1, 0], [0.99, 0.05, 0], [0.98, 0.1, 0], [0.97, 0.12, 0]],
        ["own", "sibA", "b1", "b2", "b3"], [0, 0, 1, 1, 1])
    q = pool_part[0]["siglip2"]
    oracle = orh._arm_select("oraclehub", q, pool_part, "own", 0, k_eff=1, seed=0)
    rawnn = orh._arm_select("rawnn", q, pool_part, "own", 0, k_eff=1, seed=0)
    assert [pool_part[i]["concept"] for i in oracle] == ["sibA"]     # taxonomy recovers it
    assert all(pool_part[i]["concept"] != "sibA" for i in rawnn)     # NN misses it


def test_iter_cells_gate_excludes_near_and_singletons():
    # exemplars for one part 'leaf'; own=A(fam0), sibling B(fam0) is NN-near (rank1) -> excluded.
    pool = {"leaf": _pool_part([[1, 0, 0], [0.99, 0.01, 0], [0.2, 1, 0]],
                               ["A", "B", "C"], [0, 0, 1])}
    exemplar = {"A": {"leaf": pool["leaf"][0]["siglip2"]}}
    # monkeypatch family_id so A,B are fam 0 (need >=2 held-out crops for A/leaf)
    import graft.taxonomy as tax
    orig = tax.TAXONOMY.copy()
    tax.TAXONOMY.clear(); tax.TAXONOMY.update({"A": ("x", "Fam"), "B": ("y", "Fam"), "C": ("z", "Other")})
    tax._FAMILY_IDS.clear(); tax._FAMILY_IDS.update({"Fam": 0, "Other": 1})
    try:
        held = [{"concept": "A", "part": "leaf"}, {"concept": "A", "part": "leaf"}]
        # B (sibling, fam 0) is NN-near (rank 1) -> gated out
        near = _pool_part([[1, 0, 0], [0.99, 0.01, 0]] + [[0.9, 0.4, 0]] * 5,
                          ["A", "B"] + [f"c{i}" for i in range(5)], [0, 0] + [1] * 5)
        pool2 = {"leaf": near}
        exemplar2 = {"A": {"leaf": near[0]["siglip2"]}}
        assert list(orh.iter_cells(pool2, exemplar2, held, k=4, gate_k=4)) == []
        # B embedding-FAR: 5 fam-1 crops sit nearer the query, pushing B to rank 6 (>4) -> passes
        far = _pool_part([[1, 0, 0], [0.1, 1, 0]] + [[0.99, 0.05 + 0.01 * i, 0] for i in range(5)],
                         ["A", "B"] + [f"c{i}" for i in range(5)], [0, 0] + [1] * 5)
        pool3 = {"leaf": far}
        exemplar3 = {"A": {"leaf": far[0]["siglip2"]}}
        cells3 = list(orh.iter_cells(pool3, exemplar3, held, k=4, gate_k=4))
        assert len(cells3) == 1 and cells3[0][0] == "A" and cells3[0][3] == 1  # k_eff=1
        assert cells3[0][4] > 4          # sibling rank is embedding-distant
    finally:
        tax.TAXONOMY.clear(); tax.TAXONOMY.update(orig)
        tax._FAMILY_IDS.clear()
        tax._FAMILY_IDS.update({f: i for i, f in enumerate(sorted({v[1] for v in orig.values()}))})


def test_contrast_stratifies():
    # 2 far cells where oraclehub beats rawnn, 1 near cell where it ties
    rows = []
    for (c, st, oh, rn) in [("A", "far", 0.30, 0.10), ("B", "far", 0.25, 0.15), ("C", "near", 0.20, 0.20)]:
        for d in range(2):
            rows.append({"concept": c, "part": "leaf", "draw": d, "stratum": st,
                         "condition": "oraclehub", "dino": oh})
            rows.append({"concept": c, "part": "leaf", "draw": d, "stratum": st,
                         "condition": "rawnn", "dino": rn})
    out = contrast(rows, "oraclehub", "rawnn")
    assert out["overall"]["n"] == 3
    assert out["far"]["n"] == 2 and out["far"]["mean_delta"] > 0.1
    assert abs(out["near"]["mean_delta"]) < 1e-9
