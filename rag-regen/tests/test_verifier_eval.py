from ragregen import c1, verifier_eval
from scripts import verifier_report


def _row(cid, verdict):
    return c1.LabelRow(cid, "prompt", cid, "draft.png", verdict, verdict, "")


def test_reports_nonpass_recall_and_pass_specificity():
    rows = [_row("caught", "fail"), _row("missed", "fail"),
            _row("good", "pass"), _row("false_alarm", "pass")]
    stream = {
        "caught": {"ok": False}, "missed": {"ok": True},
        "good": {"ok": True}, "false_alarm": {"ok": False},
    }
    got = verifier_eval.evaluate(rows, stream)
    assert got["non_pass"]["recall"] == 0.5
    assert got["non_pass"]["detected"] == 1
    assert got["non_pass"]["missed"] == 1
    assert got["pass"]["specificity"] == 0.5
    assert got["overall"]["balanced_accuracy"] == 0.5


def test_per_case_output_names_false_negatives():
    got = verifier_eval.evaluate([_row("rare", "fail")],
                                 {"rare": {"ok": True}})
    assert got["cases"][0]["false_negative"] is True
    assert "FALSE NEGATIVE" in verifier_eval.render(got)


def test_report_cli_accepts_an_alternate_decision_stream(tmp_path):
    import json

    (tmp_path / "labels.csv").write_text(
        "case_id,prompt,concept,draft_path,verdict,verdict_identity,notes\n"
        "rare,p,c,draft.png,fail,fail,\n")
    decisions = tmp_path / "stream_b_ref_oracle.json"
    decisions.write_text(json.dumps({"rare": {"ok": False}}))

    assert verifier_report.main([
        "--run", str(tmp_path), "--decisions", str(decisions),
        "--stem", "verifier_ref_oracle",
    ]) == 0
    assert (tmp_path / "verifier_ref_oracle.json").is_file()
    assert "non-pass" in (tmp_path / "verifier_ref_oracle.md").read_text()


def test_report_cli_can_write_outside_source_run(tmp_path):
    import json

    source = tmp_path / "screen_older"
    output = tmp_path / "eval_today"
    source.mkdir()
    (source / "labels.csv").write_text(
        "case_id,prompt,concept,draft_path,verdict,verdict_identity,notes\n"
        "rare,p,c,draft.png,fail,fail,\n")
    (source / "stream_b.json").write_text(json.dumps({"rare": {"ok": False}}))

    assert verifier_report.main([
        "--run", str(source), "--output-dir", str(output),
    ]) == 0
    assert (output / "verifier_verdict_identity.json").is_file()
    assert not (source / "verifier_verdict_identity.json").exists()
