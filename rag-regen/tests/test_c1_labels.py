import pytest

from ragregen import c1

HEADER = "case_id,prompt,concept,draft_path,verdict,verdict_identity,notes\n"


def _csv(tmp_path, body):
    p = tmp_path / "labels.csv"
    p.write_text(HEADER + body)
    return p


def test_load_labels_reads_both_verdict_columns(tmp_path):
    p = _csv(tmp_path, "fox,a fox,fox,/d/fox.png,pass,fail,note text\n")
    rows = c1.load_labels(p)
    assert len(rows) == 1
    assert rows[0].case_id == "fox"
    assert rows[0].verdict == "pass"
    assert rows[0].verdict_identity == "fail"
    assert rows[0].notes == "note text"


def test_resolve_draft_uses_recorded_path_when_it_exists(tmp_path):
    draft = tmp_path / "recorded.png"
    draft.write_bytes(b"image")
    p = _csv(tmp_path, f"fox,a fox,fox,{draft},pass,fail,note\n")
    row = c1.load_labels(p)[0]
    assert c1.resolve_draft(row, p) == draft


def test_resolve_draft_falls_back_beside_moved_labels(tmp_path):
    case_dir = tmp_path / "fox"
    case_dir.mkdir()
    local = case_dir / "draft.png"
    local.write_bytes(b"image")
    p = _csv(tmp_path,
             "fox,a fox,fox,/old/worktree/fox/draft.png,pass,fail,note\n")
    row = c1.load_labels(p)[0]
    assert c1.resolve_draft(row, p) == local


def test_ground_truth_maps_fail_to_true(tmp_path):
    p = _csv(tmp_path, "a,p,c,/d/a.png,fail,fail,\n"
                       "b,p,c,/d/b.png,pass,pass,\n")
    rows = c1.load_labels(p)
    kept, flags, excluded = c1.ground_truth(rows, "verdict_identity")
    assert flags == [True, False]
    assert excluded == 0
    assert len(kept) == 2


def test_ground_truth_excludes_blank_verdicts_and_counts_them(tmp_path):
    """A blank verdict is never guessed at -- it is dropped and counted."""
    p = _csv(tmp_path, "a,p,c,/d/a.png,fail,fail,\n"
                       "b,p,c,/d/b.png,,,\n"
                       "c,p,c,/d/c.png,pass,pass,\n")
    rows = c1.load_labels(p)
    kept, flags, excluded = c1.ground_truth(rows, "verdict_identity")
    assert [r.case_id for r in kept] == ["a", "c"]
    assert flags == [True, False]
    assert excluded == 1


def test_ground_truth_rejects_an_unknown_column(tmp_path):
    p = _csv(tmp_path, "a,p,c,/d/a.png,fail,fail,\n")
    with pytest.raises(ValueError, match="unknown verdict column"):
        c1.ground_truth(c1.load_labels(p), "verdict_vibes")


def test_ground_truth_rejects_an_unparseable_verdict(tmp_path):
    """Anything that is neither pass, fail, nor blank is an operator typo and
    must be named, not silently coerced to one side."""
    p = _csv(tmp_path, "a,p,c,/d/a.png,maybe,maybe,\n")
    with pytest.raises(ValueError, match="case 'a'.*'maybe'"):
        c1.ground_truth(c1.load_labels(p), "verdict")
