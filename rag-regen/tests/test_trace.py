import json
from pathlib import Path

from ragregen import trace


def test_open_run_creates_timestamped_dir_and_latest_symlink(tmp_path):
    rd = trace.open_run("unit", argv=["x"], args={"a": 1}, root=tmp_path)
    assert rd.path.is_dir()
    assert rd.path.name.startswith("unit_")
    latest = tmp_path / "unit_latest"
    assert latest.is_symlink()
    assert latest.resolve() == rd.path.resolve()


def test_run_json_records_argv_and_args(tmp_path):
    rd = trace.open_run("unit", argv=["a", "b"], args={"k": "v"}, root=tmp_path)
    rd.finish("ok", {"mean": 1.0})
    data = json.loads((rd.path / "run.json").read_text())
    assert data["argv"] == ["a", "b"]
    assert data["args"] == {"k": "v"}
    assert data["status"] == "ok"
    assert data["results"] == {"mean": 1.0}


def test_second_run_moves_the_latest_symlink(tmp_path):
    first = trace.open_run("unit", argv=[], args={}, root=tmp_path)
    second = trace.open_run("unit", argv=[], args={}, root=tmp_path)
    assert first.path != second.path  # They must have different paths
    assert (tmp_path / "unit_latest").resolve() == second.path.resolve()
    assert first.path.is_dir()


def test_case_trace_save_writes_through_a_tmp_file_and_replaces(tmp_path,
                                                                 monkeypatch):
    """CaseTrace.save became a live write path once run_pipeline started
    appending to it every round -- a SIGKILL mid bare write_text could
    truncate a case's trace.json. Queue.save's tmp-file + os.replace
    pattern is the fix. A killed process mid-write isn't something a unit
    test can simulate directly, so this pins the MECHANISM instead: the
    final file lands via os.replace of a sibling .tmp, exactly like
    Queue.save, not via a direct write_text(out) that a bare write_text
    would also satisfy."""
    import os as os_module

    replaced = []
    real_replace = os_module.replace

    def spy_replace(src, dst):
        replaced.append((Path(src), Path(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(trace.os, "replace", spy_replace)

    ct = trace.CaseTrace("c1")
    ct.vlm("verify", "a reply")
    out = tmp_path / "trace.json"
    ct.save(out)

    assert replaced == [(out.with_name("trace.json.tmp"), out)]
    assert out.is_file()
    assert not (tmp_path / "trace.json.tmp").exists()
    json.loads(out.read_text())


def test_case_trace_flags_garbage_vlm_output(tmp_path):
    ct = trace.CaseTrace("c1")
    ct.vlm("verify", "!!!! !!!!")
    ct.vlm("query", "an Amur leopard")
    out = tmp_path / "trace.json"
    ct.save(out)
    data = json.loads(out.read_text())
    assert data["vlm"][0]["garbage"] is True
    assert data["vlm"][1]["garbage"] is False


def test_same_second_runs_get_different_dirs(tmp_path):
    """When two runs happen in the same second, they must get different directories."""
    import time
    first = trace.open_run("unit", argv=[], args={}, root=tmp_path)
    # Force second run to happen before next second boundary
    second = trace.open_run("unit", argv=[], args={}, root=tmp_path)
    assert first.path != second.path, "Runs in same second must have different directories"


def test_directory_at_latest_raises_error(tmp_path):
    """If someone created a directory at <tag>_latest, opening a run must raise RuntimeError."""
    (tmp_path / "unit_latest").mkdir()
    try:
        trace.open_run("unit", argv=[], args={}, root=tmp_path)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "is a directory" in str(e)


def test_regular_file_at_latest_is_replaced(tmp_path):
    """If a regular file exists at <tag>_latest, it should be replaced with the symlink."""
    latest = tmp_path / "unit_latest"
    latest.write_text("old content")
    rd = trace.open_run("unit", argv=[], args={}, root=tmp_path)
    assert latest.is_symlink(), "Latest should be a symlink now"
    assert latest.resolve() == rd.path.resolve()


def test_broken_symlink_at_latest_is_replaced(tmp_path):
    """If a broken symlink exists at <tag>_latest, it should be replaced with the new symlink."""
    latest = tmp_path / "unit_latest"
    latest.symlink_to("/nonexistent/path")
    assert not latest.exists(), "Symlink should be broken"
    rd = trace.open_run("unit", argv=[], args={}, root=tmp_path)
    assert latest.is_symlink(), "Latest should be a symlink"
    assert latest.resolve() == rd.path.resolve()
