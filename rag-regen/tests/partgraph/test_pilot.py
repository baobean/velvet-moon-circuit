from pathlib import Path
from PIL import Image
from ragregen.partgraph.pilot import run_case
from ragregen import regen


class _Store:
    def __init__(self, rec): self._rec = rec
    def get(self, gid): return self._rec


class _ConstEncoder:      # deterministic, offline stand-in for DinoEncoder
    def embed(self, image):
        import numpy as np
        a = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
        return a / (np.linalg.norm(a) + 1e-12)


def _png(p, color, size=(64, 64)):
    Image.new("RGB", size, color).save(p); return p


def test_run_case_stitch_arm_scores_and_preserves(tmp_path):
    leaf = tmp_path / "leaf.png"; _png(leaf, (0, 200, 0))
    rec = {"part_crops": [{"part_type": "leaf", "image_path": str(leaf), "embedding_ref": 0}],
           "medoid": {"image_path": str(leaf)}}
    draft = Image.new("RGB", (128, 128), (10, 10, 10))
    mask = Image.new("L", (128, 128), 0)
    for x in range(40, 88):
        for y in range(40, 88):
            mask.putpixel((x, y), 255)
    r1 = tmp_path / "r1.png"; _png(r1, (0, 200, 0))
    r2 = tmp_path / "r2.png"; _png(r2, (0, 190, 0))

    def sink(d, m, ref):
        return regen.stitch(d, m, ref)

    out = run_case(arm="partgraph", store=_Store(rec), global_id="ds:sp_A",
                   draft=draft, mask=mask, gt_refs=[Path(r1), Path(r2)],
                   encoder=_ConstEncoder(), sink=sink, heldout=1, tile=64)
    assert out["arm"] == "partgraph"
    assert 0.0 <= out["dino"] <= 1.0
    assert abs(out["preservation"] - 1.0) < 1e-6      # stitch preserves ~mask exactly
