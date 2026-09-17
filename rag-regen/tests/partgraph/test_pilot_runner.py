"""CPU-only test of the wk2 GO/NO-GO pilot runner (scripts/partgraph_pilot.py).

Hermetic: a fake drafter/masker/inpainter/encoder (no torch, no models) plus a
real on-disk mmkg_store built via schema.build_record + store.write_store,
loaded through the real store_factory default (Store.load) -- this exercises
config.load_dataset's new `global_id` field, the real store integration, and
`run_case`/`gain_verdict` end to end, only the heavy models are faked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ragregen import regen  # noqa: E402
from ragregen.mmkg_store import schema  # noqa: E402
from ragregen.mmkg_store.store import write_store  # noqa: E402
from scripts import partgraph_pilot  # noqa: E402


# --- fixture helpers ---------------------------------------------------------

def _png(path, color, size=(64, 64)):
    Image.new("RGB", size, color).save(path)
    return path


class _FakeDrafter:
    """Deterministic per-prompt color, no torch."""

    def draft(self, prompt):
        h = abs(hash(prompt))
        color = (h % 256, (h // 256) % 256, (h // 65536) % 256)
        return Image.new("RGB", (64, 64), color)


def _fixed_mask_factory(never_none_for: tuple = ()):
    """masker_factory(device) -> a fixed centered-rectangle mask fn.

    Returns None for any case whose id is in `never_none_for`, to exercise
    the drop-and-continue path; otherwise always grounds.
    """
    def _factory(device):
        def _mask_fn(draft_image, case):
            if case.id in never_none_for:
                return None
            w, h = draft_image.size
            m = Image.new("L", draft_image.size, 0)
            for x in range(w // 4, 3 * w // 4):
                for y in range(h // 4, 3 * h // 4):
                    m.putpixel((x, y), 255)
            return m
        return _mask_fn
    return _factory


class _FakeInpainter:
    """stitch-based fake, exactly like Task 5's test uses `stitch` as `sink`."""

    def regen(self, draft, mask, reference, prompt):
        return regen.stitch(draft, mask, reference)

    def free(self):
        pass


class _ConstEncoder:
    """Deterministic, offline stand-in for DinoEncoder (mean-RGB based)."""

    def embed(self, image):
        a = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
        return a / (np.linalg.norm(a) + 1e-12)

    def free(self):
        pass


def _write_dataset_yaml(tmp_path: Path, *, drop_case_b_global_id: bool = False) -> Path:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    _png(images_dir / "a_ref1.png", (10, 200, 10))
    _png(images_dir / "a_ref2.png", (12, 190, 15))
    _png(images_dir / "b_ref1.png", (200, 10, 10))
    _png(images_dir / "b_ref2.png", (190, 15, 12))

    case_b = {
        "id": "case_b",
        "prompt": "a photo of a rare thing B",
        "concept": "rare thing B",
        "coarse": "thing",
        "gt_refs": ["b_ref1.png", "b_ref2.png"],
    }
    if not drop_case_b_global_id:
        case_b["global_id"] = "testds:species_b"

    data = {
        "name": "pilot_fixture",
        "images_root": str(images_dir),
        "cases": [
            {
                "id": "case_a",
                "prompt": "a photo of a rare thing A",
                "concept": "rare thing A",
                "coarse": "thing",
                "gt_refs": ["a_ref1.png", "a_ref2.png"],
                "global_id": "testds:species_a",
            },
            case_b,
        ],
    }
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _write_pipeline_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump({"steps": 2, "seed": 0}))
    return path


def _write_store(tmp_path: Path, species_keys=("species_a", "species_b")) -> Path:
    src = tmp_path / "store_src"
    src.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "store"
    records = []
    for i, key in enumerate(species_keys):
        medoid_path = _png(src / f"{key}_medoid.png", (50 + i, 60 + i, 70 + i))
        part_a = _png(src / f"{key}_partA.png", (0, 128 + i, 0))
        part_b = _png(src / f"{key}_partB.png", (128 + i, 64, 0))
        records.append(schema.build_record(
            dataset="testds",
            species_key=key,
            scientific_name=f"Testus {key}",
            common_name=key,
            taxonomy={},
            medoid={"image_path": str(medoid_path), "embedding_ref": i * 10,
                   "k_images": 1, "selection": "test-fixture"},
            candidates=[],
            part_crops=[
                {"part_type": "partA", "image_path": str(part_a), "embedding_ref": i * 10 + 1},
                {"part_type": "partB", "image_path": str(part_b), "embedding_ref": i * 10 + 2},
            ],
            attributes={},
            provenance={},
        ))
    write_store(records, out_dir)
    return out_dir


def _latest_run_dir(output_root: Path) -> Path:
    runs = [r for r in output_root.glob("partgraph_pilot_*")
           if r.is_dir() and not r.is_symlink()]
    assert len(runs) == 1, f"expected exactly one run dir, found {runs}"
    return runs[0]


# --- tests --------------------------------------------------------------

def test_happy_path_writes_result_with_four_rows_and_a_verdict(tmp_path):
    dataset_path = _write_dataset_yaml(tmp_path)
    pipeline_path = _write_pipeline_yaml(tmp_path)
    store_dir = _write_store(tmp_path)
    output_root = tmp_path / "outputs"

    argv = ["--dataset", str(dataset_path), "--pipeline", str(pipeline_path),
           "--store", str(store_dir), "--output-root", str(output_root),
           "--tile", "32"]

    rc = partgraph_pilot.main(
        argv,
        drafter_factory=lambda pipe_cfg, device: _FakeDrafter(),
        masker_factory=_fixed_mask_factory(),
        inpainter_factory=lambda pipe_cfg, device: _FakeInpainter(),
        dino_factory=lambda pipe_cfg, device: _ConstEncoder(),
    )
    assert rc == 0

    run_dir = _latest_run_dir(output_root)
    result = json.loads((run_dir / "result.json").read_text())

    assert len(result["per_case"]) == 4  # 2 species x 2 arms
    assert len(result["deltas"]) == 2
    assert result["decision"]["verdict"] in {
        "GAIN", "LOSS", "NO_LARGE_EFFECT", "UNDERPOWERED"}
    assert result["dropped"] == []

    for case_id in ("case_a", "case_b"):
        case_dir = run_dir / case_id
        assert (case_dir / "draft.png").exists()
        assert (case_dir / "mask.png").exists()
        for arm in ("single_medoid", "partgraph"):
            assert (case_dir / f"reference_{arm}.png").exists()
            assert (case_dir / f"output_{arm}.png").exists()

    rows = [json.loads(line) for line in
           (run_dir / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert {r["arm"] for r in rows} == {"single_medoid", "partgraph"}

    run_json = json.loads((run_dir / "run.json").read_text())
    assert run_json["status"] == "ok"
    assert run_json["results"]["decision"] == result["decision"]["verdict"]


def test_ungrounded_case_is_dropped_and_the_other_case_still_completes(tmp_path):
    dataset_path = _write_dataset_yaml(tmp_path)
    pipeline_path = _write_pipeline_yaml(tmp_path)
    store_dir = _write_store(tmp_path)
    output_root = tmp_path / "outputs"

    argv = ["--dataset", str(dataset_path), "--pipeline", str(pipeline_path),
           "--store", str(store_dir), "--output-root", str(output_root)]

    rc = partgraph_pilot.main(
        argv,
        drafter_factory=lambda pipe_cfg, device: _FakeDrafter(),
        masker_factory=_fixed_mask_factory(never_none_for=("case_b",)),
        inpainter_factory=lambda pipe_cfg, device: _FakeInpainter(),
        dino_factory=lambda pipe_cfg, device: _ConstEncoder(),
    )
    assert rc != 0

    run_dir = _latest_run_dir(output_root)
    result = json.loads((run_dir / "result.json").read_text())

    assert any(d["case_id"] == "case_b" and d["arm"] is None
              and d["reason"] == "not grounded" for d in result["dropped"])

    case_a_rows = [r for r in result["per_case"] if r["case_id"] == "case_a"]
    assert len(case_a_rows) == 2
    assert {r["arm"] for r in case_a_rows} == {"single_medoid", "partgraph"}
    assert all(r["case_id"] != "case_b" for r in result["per_case"])
    # case_b drafted (masking runs after drafting) but never got a mask.
    assert (run_dir / "case_b" / "draft.png").exists()
    assert not (run_dir / "case_b" / "mask.png").exists()


def test_case_missing_global_id_is_dropped_before_reaching_the_store(tmp_path):
    dataset_path = _write_dataset_yaml(tmp_path, drop_case_b_global_id=True)
    pipeline_path = _write_pipeline_yaml(tmp_path)
    # No "species_b" record at all -- if the runner ever looked case_b up in
    # the store despite its missing global_id, that would be a bug this store
    # cannot mask.
    store_dir = _write_store(tmp_path, species_keys=("species_a",))
    output_root = tmp_path / "outputs"

    argv = ["--dataset", str(dataset_path), "--pipeline", str(pipeline_path),
           "--store", str(store_dir), "--output-root", str(output_root)]

    rc = partgraph_pilot.main(
        argv,
        drafter_factory=lambda pipe_cfg, device: _FakeDrafter(),
        masker_factory=_fixed_mask_factory(),
        inpainter_factory=lambda pipe_cfg, device: _FakeInpainter(),
        dino_factory=lambda pipe_cfg, device: _ConstEncoder(),
    )
    assert rc != 0

    run_dir = _latest_run_dir(output_root)
    result = json.loads((run_dir / "result.json").read_text())

    drop = next(d for d in result["dropped"] if d["case_id"] == "case_b")
    assert drop["arm"] is None
    assert "global_id" in drop["reason"]

    assert all(r["case_id"] != "case_b" for r in result["per_case"])
    # Dropped before drafting -- case_b never even gets a case_dir.
    assert not (run_dir / "case_b").exists()

    case_a_rows = [r for r in result["per_case"] if r["case_id"] == "case_a"]
    assert len(case_a_rows) == 2
