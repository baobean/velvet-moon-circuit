from ragregen.mmkg_store import manifest as m
from PIL import Image


def test_eligible_pool_drops_dup_and_contaminated(tmp_path):
    keep = tmp_path/"a.jpg"; Image.new("RGB",(8,8),"red").save(keep)
    dup = tmp_path/"b.jpg"; dup.write_bytes(keep.read_bytes())          # sha dup -> dropped
    contam = tmp_path/"c.jpg"; Image.new("RGB",(8,8),"blue").save(contam)
    gt = tmp_path/"gt.jpg"; gt.write_bytes(contam.read_bytes())         # eval ref == contam
    kept, excl = m.eligible_pool({"eligible":[str(keep),str(dup),str(contam)]}, gt_eval_paths=[str(gt)])
    assert kept == [str(keep)]
    assert {e["reason"] for e in excl} == {"duplicate","contamination"}


def test_manifest_rejects_eligible_in_eval(tmp_path):
    p = tmp_path/"a.jpg"; Image.new("RGB",(8,8),"red").save(p)
    import pytest
    with pytest.raises(ValueError):
        m.eligible_pool({"eligible":[str(p)], "eval_set":[str(p)]})


# --- fix-wave-2 I3: provenance_block over an in-memory (no on-disk file) manifest ---

def test_provenance_block_with_real_manifest_file(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"eligible": ["a.jpg"]}')
    entry = {"eligible": ["a.jpg"], "source_split": "build"}
    kept = ["a.jpg"]
    excluded = [{"path": "b.jpg", "reason": "duplicate"}]

    block = m.provenance_block(entry, manifest_path, kept, excluded)

    assert block["source_split"] == "build"
    assert block["build_manifest"]["path"] == str(manifest_path)
    assert block["build_manifest"]["sha256"]  # a real file hash, non-empty
    assert block["eligible_images"] == kept
    assert block["excluded_images"] == excluded


def test_provenance_block_manifest_path_none_uses_in_memory_marker():
    # mmkg_store's Treevill/iNat manifests are generated in-memory each build
    # run -- there is no on-disk manifest file to hash, so `manifest_path`
    # may be None and `build_manifest` carries a clear in-memory marker
    # instead ({"path": None, "sha256": <hash of the entry itself>}).
    entry = {"eligible": ["a.jpg", "b.jpg"], "source_split": None}
    kept = ["a.jpg"]
    excluded = [{"path": "b.jpg", "reason": "duplicate"}]

    block = m.provenance_block(entry, None, kept, excluded)

    assert block["source_split"] is None
    assert block["build_manifest"]["path"] is None
    assert isinstance(block["build_manifest"]["sha256"], str) and block["build_manifest"]["sha256"]
    assert block["eligible_images"] == kept
    assert block["excluded_images"] == excluded

    # deterministic: same entry -> same in-memory sha256
    block2 = m.provenance_block(entry, None, kept, excluded)
    assert block2["build_manifest"]["sha256"] == block["build_manifest"]["sha256"]

    # a different entry -> a different sha256 (not just a constant placeholder)
    other = m.provenance_block({"eligible": ["z.jpg"]}, None, kept, excluded)
    assert other["build_manifest"]["sha256"] != block["build_manifest"]["sha256"]
