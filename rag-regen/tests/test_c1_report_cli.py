import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import c1_report  # noqa: E402

HEADER = "case_id,prompt,concept,draft_path,verdict,verdict_identity,notes\n"


@pytest.fixture
def run_dir(tmp_path):
    (tmp_path / "labels.csv").write_text(
        HEADER
        + "a,a prompt,fox,/d/a.png,fail,fail,\n"
        + "b,b prompt,fox,/d/b.png,pass,pass,\n")
    (tmp_path / "stream_a.json").write_text(json.dumps({
        "a": {"fox": {"sim": 0.10, "kind": "subject", "box": None,
                      "dino_conf": None, "state": "MISSING"}},
        "b": {"fox": {"sim": 0.90, "kind": "subject", "box": None,
                      "dino_conf": None, "state": "PRESENT"}}}))
    (tmp_path / "stream_b.json").write_text(json.dumps({
        "a": {"ok": False, "degenerate": False, "raw": "", "issues": []},
        "b": {"ok": True, "degenerate": False, "raw": "", "issues": []}}))
    return tmp_path


def test_report_runs_and_writes_both_artifacts(run_dir, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["c1_report.py", "--run", str(run_dir)])
    assert c1_report.main() == 0
    assert (run_dir / "c1.md").is_file()
    assert (run_dir / "c1.json").is_file()


def test_report_covers_both_ground_truth_columns(run_dir, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["c1_report.py", "--run", str(run_dir)])
    c1_report.main()
    md = (run_dir / "c1.md").read_text()
    assert "verdict_identity" in md and "verdict`" in md


def test_missing_stream_a_exits_two_with_an_actionable_message(
        run_dir, monkeypatch, capsys):
    (run_dir / "stream_a.json").unlink()
    monkeypatch.setattr(sys, "argv", ["c1_report.py", "--run", str(run_dir)])
    assert c1_report.main() == 2
    out = capsys.readouterr().out
    assert "stream_a.json" in out and "score-a" in out


def test_missing_stream_b_exits_two(run_dir, monkeypatch, capsys):
    (run_dir / "stream_b.json").unlink()
    monkeypatch.setattr(sys, "argv", ["c1_report.py", "--run", str(run_dir)])
    assert c1_report.main() == 2
    assert "score-b" in capsys.readouterr().out


def test_json_artifact_carries_the_arms_at_tau_star(run_dir, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["c1_report.py", "--run", str(run_dir)])
    c1_report.main()
    data = json.loads((run_dir / "c1.json").read_text())
    assert "verdict_identity" in data
    arms = data["verdict_identity"]["arms_at_tau_star"]
    assert set(arms) == {"grounded", "semantic", "fused"}
    assert "balanced_accuracy" in arms["fused"]


def test_alternate_streams_and_stem_do_not_overwrite_baseline(run_dir):
    alt_a = run_dir / "stream_a_retrieved.json"
    alt_b = run_dir / "stream_b_ref.json"
    alt_a.write_text((run_dir / "stream_a.json").read_text())
    alt_b.write_text((run_dir / "stream_b.json").read_text())

    assert c1_report.main([
        "--run", str(run_dir), "--stream-a", str(alt_a),
        "--stream-b", str(alt_b), "--stem", "c1_retrieved_ref",
    ]) == 0
    assert (run_dir / "c1_retrieved_ref.json").is_file()
    assert (run_dir / "c1_retrieved_ref.md").is_file()


def test_report_can_write_outside_source_run(run_dir, tmp_path):
    output = tmp_path / "eval_today"
    assert c1_report.main([
        "--run", str(run_dir), "--output-dir", str(output),
    ]) == 0
    assert (output / "c1.json").is_file()
    assert (output / "c1.md").is_file()
