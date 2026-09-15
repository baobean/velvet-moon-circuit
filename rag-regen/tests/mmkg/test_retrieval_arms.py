from ragregen.mmkg import retrieval_arms as ra

def test_flat_and_mmkg_arms_and_divergence():
    hits = [{"path": "a.jpg", "score": 0.9, "rank": 0},
            {"path": "b.jpg", "score": 0.5, "rank": 1}]
    flat = ra.flat_arm(hits)
    assert flat["selected_ref"]["path"] == "a.jpg"                # top by score
    inst = [{"path": "b.jpg", "score": 0.5, "part_type": "body"},
            {"path": "c.jpg", "score": 0.7, "part_type": "head"}]
    targets = {"primary_color": {"is_target": True}, "surface_texture": {"is_target": True}}
    mmkg = ra.mmkg_arm(inst, targets)
    assert mmkg["selected_ref"]["path"] == "c.jpg"               # highest score in pool
    d = ra.divergence(flat, mmkg)
    assert d["n_flat"] == 2 and d["n_mmkg"] == 2
    assert abs(d["jaccard"] - 1/3) < 1e-9                        # {a,b} vs {b,c} -> 1/3
    assert d["selected_ref_identical"] is False
    cov = ra.coverage(flat, mmkg)
    assert cov["flat_unit_size"] == 1 and cov["mmkg_part_types"] == 2
    assert cov["flat_top1_in_mmkg_pool"] is False               # a.jpg not in {b,c}

def test_mmkg_arm_dedups_part_type_by_max_score():
    inst = [
        {"path": "b.jpg", "score": 0.5, "part_type": "body"},
        {"path": "d.jpg", "score": 0.8, "part_type": "body"},   # same part_type, higher score
        {"path": "c.jpg", "score": 0.7, "part_type": "head"},
    ]
    mmkg = ra.mmkg_arm(inst, targets={})
    sel_paths = sorted(i["path"] for i in mmkg["selected_set"])
    assert sel_paths == ["c.jpg", "d.jpg"]        # one per part_type; body -> d (0.8) beats b (0.5)
    flat = ra.flat_arm([{"path": "z.jpg", "score": 0.1, "rank": 0}])
    cov = ra.coverage(flat, mmkg)
    assert cov["mmkg_unit_size"] == 2 and cov["mmkg_part_types"] == 2   # 3 instances -> 2 after dedup
