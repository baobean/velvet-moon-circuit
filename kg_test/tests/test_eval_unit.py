import dataclasses
import json
import os
import subprocess

import pytest
from PIL import Image

from graft import run_eval
from graft.config import GraftConfig
from graft.dataset import Species
from graft.run_eval import evaluate, rank_species_by_unique_count, summarize
from graft.schema import ConceptKG

FIXTURE_ROOT = os.path.join(os.path.dirname(__file__), "fixtures", "mini_treevill")


def test_summarize_means_per_method():
    rows = [
        {"method": "ours", "dino": 0.8},
        {"method": "ours", "dino": 0.6},
        {"method": "b0", "dino": 0.2},
    ]
    s = summarize(rows)
    assert abs(s["ours"]["dino"] - 0.7) < 1e-9
    assert abs(s["b0"]["dino"] - 0.2) < 1e-9


def test_summarize_ignores_non_numeric_and_method_keys():
    rows = [{"method": "ours", "species": "Akashmoni", "dino": 0.5}]
    s = summarize(rows)
    assert s["ours"] == {"dino": 0.5}


def test_summarize_excludes_matrix_tag_numerics():
    rows = [
        {"method": "ours", "species": "A", "name_mode": "neutral",
         "ip_scale": 0.4, "n_heldout": 2, "dino": 0.4},
        {"method": "ours", "species": "B", "name_mode": "named",
         "ip_scale": 0.8, "n_heldout": 6, "dino": 0.6},
        {"method": "b1", "species": "A", "name_mode": "neutral",
         "ip_scale": 0.4, "n_heldout": 2, "dino": 0.1},
    ]
    s = summarize(rows)
    assert s["ours"] == {"dino": 0.5}
    assert s["b1"] == {"dino": 0.1}
    for per_method in s.values():
        assert "ip_scale" not in per_method
        assert "n_heldout" not in per_method


def test_rank_species_by_unique_count_orders_and_limits():
    ranked = rank_species_by_unique_count(FIXTURE_ROOT)
    # both fixture species have 3 unique images -> deterministic name tie-break
    assert ranked == ["Akashmoni", "Debdaru"]
    assert rank_species_by_unique_count(FIXTURE_ROOT, limit=1) == ["Akashmoni"]


# --------------------------------------------------------------------------
# Matrix driver: every GPU phase is a subprocess, so the whole loop is
# exercisable on CPU by stubbing subprocess.run + the KG build subprocess.
# --------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, returncode=0, stderr=""):
        self.returncode, self.stderr, self.stdout = returncode, stderr, ""


def _fixture_species():
    return Species("Akashmoni", sorted(
        os.path.join(FIXTURE_ROOT, "Akashmoni", f) for f in ("1.jpg", "2.jpg", "3.jpg")
    ))


def _write_kg(path, ref_paths, *, with_embeddings=True):
    embs = [[1.0, 0.0]] * len(ref_paths) if with_embeddings else []
    ConceptKG("Akashmoni", [], [], {}, "", [], list(ref_paths),
              ref_embeddings=embs).to_json(path)


def _stub_workers(monkeypatch, *, fail_gen=()):
    """Simulate worker_baseline / worker_metrics. `fail_gen` holds
    (method, ip_scale) cells whose GENERATION subprocess fails."""
    import yaml

    def fake_run(cmd, capture_output=False, text=False):
        module = cmd[2]
        if module == "graft.worker_baseline":
            method, _species, cfg_yaml, _kg_arg, image_path = cmd[3:8]
            with open(cfg_yaml) as f:
                ip_scale = yaml.safe_load(f)["ip_scale"]
            if (method, ip_scale) in fail_gen:
                return _FakeProc(1, "simulated generation failure")
            Image.new("RGB", (8, 8)).save(image_path)
            return _FakeProc()
        if module == "graft.worker_metrics":
            metrics_path = cmd[7]
            with open(metrics_path, "w") as f:
                json.dump({"dino": 0.5}, f)
            return _FakeProc()
        raise AssertionError(f"unexpected worker: {module}")

    monkeypatch.setattr(subprocess, "run", fake_run)


def _stub_build_kg(monkeypatch, calls):
    import graft.pipeline

    def fake_build(concept, build_refs, cfg, outputs_dir):
        calls.append(concept)
        _write_kg(os.path.join(outputs_dir, "kg.json"), build_refs)

    monkeypatch.setattr(graft.pipeline, "_build_kg_subprocess", fake_build)


def _cfg(tmp_path):
    return dataclasses.replace(GraftConfig(), outputs_dir=str(tmp_path), k_build_refs=2)


def test_evaluate_reports_a_complete_matrix(tmp_path, monkeypatch):
    _stub_workers(monkeypatch)
    _stub_build_kg(monkeypatch, [])
    out = evaluate([_fixture_species()], ["ours", "b0"], None, _cfg(tmp_path),
                   name_modes=["neutral"], ip_scales=[0.4, 0.6])
    # ours is swept (2 ip points), b0 runs once per name_mode
    assert out["expected_cells"] == 3
    assert out["missing_cells"] == []
    assert len(out["rows"]) == 3


def test_evaluate_retries_unswept_method_after_a_transient_failure(tmp_path, monkeypatch):
    _stub_workers(monkeypatch, fail_gen={("b0", 0.4)})
    _stub_build_kg(monkeypatch, [])
    out = evaluate([_fixture_species()], ["ours", "b0"], None, _cfg(tmp_path),
                   name_modes=["neutral"], ip_scales=[0.4, 0.6])
    # b0 failed at ip 0.4 -> it must be retried at 0.6, not marked done
    assert out["missing_cells"] == []
    assert sorted(r["method"] for r in out["rows"]) == ["b0", "ours", "ours"]


def test_evaluate_records_cells_that_never_produced_a_row(tmp_path, monkeypatch):
    _stub_workers(monkeypatch, fail_gen={("ours", 0.4), ("ours", 0.6)})
    _stub_build_kg(monkeypatch, [])
    out = evaluate([_fixture_species()], ["ours", "b0"], None, _cfg(tmp_path),
                   name_modes=["neutral"], ip_scales=[0.4, 0.6])
    assert out["expected_cells"] == 3
    assert [(c["method"], c["ip_scale"]) for c in out["missing_cells"]] == [
        ("ours", 0.4), ("ours", 0.6)
    ]


def test_evaluate_rebuilds_a_stale_kg_without_ref_embeddings(tmp_path, monkeypatch, capsys):
    sp = _fixture_species()
    outputs_dir = os.path.join(str(tmp_path), sp.name)
    os.makedirs(outputs_dir)
    _write_kg(os.path.join(outputs_dir, "kg.json"), sp.images, with_embeddings=False)

    _stub_workers(monkeypatch)
    calls = []
    _stub_build_kg(monkeypatch, calls)
    out = evaluate([sp], ["b0"], None, _cfg(tmp_path),
                   name_modes=["neutral"], ip_scales=[0.6])
    assert calls == ["Akashmoni"]                     # stale KG => rebuilt
    assert "predates ref_embeddings" in capsys.readouterr().out
    assert len(out["rows"]) == 1


def test_evaluate_reuses_a_kg_that_already_has_ref_embeddings(tmp_path, monkeypatch):
    sp = _fixture_species()
    outputs_dir = os.path.join(str(tmp_path), sp.name)
    os.makedirs(outputs_dir)
    _write_kg(os.path.join(outputs_dir, "kg.json"), sp.images)

    _stub_workers(monkeypatch)
    calls = []
    _stub_build_kg(monkeypatch, calls)
    evaluate([sp], ["b0"], None, _cfg(tmp_path), name_modes=["neutral"], ip_scales=[0.6])
    assert calls == []


def test_evaluate_resume_skips_generation_for_an_existing_image(tmp_path, monkeypatch):
    sp = _fixture_species()
    images_dir = os.path.join(str(tmp_path), sp.name, "eval_images")
    os.makedirs(images_dir)
    Image.new("RGB", (8, 8)).save(os.path.join(images_dir, "ours_neutral_ip0.6.png"))

    # every generation fails: only the pre-existing image can yield a row
    _stub_workers(monkeypatch, fail_gen={("ours", 0.6), ("b0", 0.6)})
    _stub_build_kg(monkeypatch, [])
    out = evaluate([sp], ["ours", "b0"], None, _cfg(tmp_path),
                   name_modes=["neutral"], ip_scales=[0.6], resume=True)
    assert [r["method"] for r in out["rows"]] == ["ours"]
    assert [c["method"] for c in out["missing_cells"]] == ["b0"]


# --------------------------------------------------------------------------
# CLI validation (a typo must not silently run the whole matrix in the wrong arm)
# --------------------------------------------------------------------------

def test_main_rejects_unknown_name_mode():
    with pytest.raises(SystemExit):
        run_eval.main(["--root", FIXTURE_ROOT, "--name-modes", "nutral"])


def test_main_rejects_empty_sweep_lists():
    with pytest.raises(SystemExit):
        run_eval.main(["--root", FIXTURE_ROOT, "--ip-scales", " , "])
    with pytest.raises(SystemExit):
        run_eval.main(["--root", FIXTURE_ROOT, "--name-modes", ""])


def test_main_species_selector_bypasses_the_expensive_ranking(tmp_path, monkeypatch):
    def _explode(*a, **k):
        raise AssertionError("rank_species_by_unique_count must not run with --species")

    monkeypatch.setattr(run_eval, "rank_species_by_unique_count", _explode)
    monkeypatch.setattr(run_eval.env, "setup", lambda: None)

    seen = {}

    def fake_evaluate(species_list, methods, models, cfg, **kwargs):
        seen["names"] = [sp.name for sp in species_list]
        seen["kwargs"] = kwargs
        return {"rows": [], "summary": {}, "expected_cells": 0, "missing_cells": []}

    monkeypatch.setattr(run_eval, "evaluate", fake_evaluate)
    rc = run_eval.main([
        "--root", FIXTURE_ROOT, "--species", "Debdaru", "--resume",
        "--out", str(tmp_path),
    ])
    assert rc == 0
    assert seen["names"] == ["Debdaru"]
    assert seen["kwargs"]["resume"] is True
    assert json.loads((tmp_path / "results.json").read_text())["expected_cells"] == 0
