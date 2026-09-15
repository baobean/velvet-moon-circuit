from ragregen.mmkg import build as b

def test_feasibility_counts_decidable():
    reads = {
        # decidable: 2 slots each with 3 visible agreeing
        "A": [{"primary_color":"red","surface_texture":"rough"}]*3,
        # undecidable: only 1 slot reliable
        "B": [{"primary_color":"red"}]*3 + [{"surface_texture":"x"}],
    }
    out = b.feasibility(reads)
    assert out["by_concept"]["A"] == 2 and out["by_concept"]["B"] == 1
    assert out["decidable_concepts"] == 1 and out["undecidable"] == ["B"]

def test_build_concept_uses_readfn_and_targets(tmp_path):
    paths = []
    reads = {}
    vals = [{"primary_color":"red","surface_texture":"rough"},
            {"primary_color":"red","surface_texture":"rough"},
            {"primary_color":"red","surface_texture":"smooth"}]
    from PIL import Image
    colors = ["white", "black", "red"]  # distinct bytes so the real sha256 guard keeps all 3
    for i, v in enumerate(vals):
        p = tmp_path / f"s{i}.jpg"; Image.new("RGB",(8,8),colors[i]).save(p)
        paths.append(str(p)); reads[str(p)] = v
    rec = b.build_concept("A", "q", paths, gt_ref_paths=[], read_fn=lambda p: reads[p])
    assert rec["slice_size"] == 3
    assert rec["target_attributes"]["primary_color"]["is_target"] is True
    assert rec["n_target_attributes"] >= 1

def test_dedup_and_guard_drops_unhashable_dup_and_contaminated(tmp_path):
    from PIL import Image
    # a readable, unique image (kept)
    p_keep = tmp_path / "keep.jpg"; Image.new("RGB", (8, 8), "red").save(p_keep)
    # a byte-identical duplicate of keep (second occurrence dropped)
    p_dup = tmp_path / "dup.jpg"; p_dup.write_bytes(p_keep.read_bytes())
    # a path that cannot be hashed (missing) -> dropped
    p_missing = str(tmp_path / "missing.jpg")
    # a file whose bytes match a ground-truth reference -> contamination drop
    p_contam = tmp_path / "contam.jpg"; Image.new("RGB", (8, 8), "blue").save(p_contam)
    gt = tmp_path / "gt.jpg"; gt.write_bytes(p_contam.read_bytes())   # byte-identical GT ref

    kept = b.dedup_and_guard([str(p_keep), str(p_dup), p_missing, str(p_contam)],
                             gt_ref_paths=[str(gt)])
    assert kept == [str(p_keep)]   # dup dropped (same bytes as keep), missing dropped, contaminated dropped
