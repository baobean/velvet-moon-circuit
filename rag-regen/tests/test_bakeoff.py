import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen.retrieve import Hit  # noqa: E402
from scripts import bakeoff  # noqa: E402


def hits(*stems):
    return [Hit(Path(f"/c/{s}.jpg"), 1.0 - i * 0.1, i)
            for i, s in enumerate(stems)]


def test_recall_at_1_hits():
    assert bakeoff.recall_at_k(hits("fox", "panda"), {Path("/c/fox.jpg")}, 1) == 1.0


def test_recall_at_1_misses():
    assert bakeoff.recall_at_k(hits("panda", "fox"), {Path("/c/fox.jpg")}, 1) == 0.0


def test_recall_at_5_finds_a_later_hit():
    assert bakeoff.recall_at_k(hits("a", "b", "fox"), {Path("/c/fox.jpg")}, 5) == 1.0


def test_mrr_is_reciprocal_of_the_first_correct_rank():
    assert bakeoff.mrr(hits("a", "fox"), {Path("/c/fox.jpg")}) == 0.5


def test_mrr_is_zero_when_absent():
    assert bakeoff.mrr(hits("a", "b"), {Path("/c/fox.jpg")}) == 0.0


# --- metric edges ----------------------------------------------------------

def test_recall_at_k_truncates_at_k_rather_than_scanning_everything():
    """A hit at rank 2 must not count towards R@1 or R@2."""
    h = hits("a", "b", "fox")
    assert bakeoff.recall_at_k(h, {Path("/c/fox.jpg")}, 1) == 0.0
    assert bakeoff.recall_at_k(h, {Path("/c/fox.jpg")}, 2) == 0.0
    assert bakeoff.recall_at_k(h, {Path("/c/fox.jpg")}, 3) == 1.0


def test_recall_and_mrr_on_empty_hits():
    assert bakeoff.recall_at_k([], {Path("/c/fox.jpg")}, 5) == 0.0
    assert bakeoff.mrr([], {Path("/c/fox.jpg")}) == 0.0


def test_mrr_uses_the_first_correct_hit_not_a_later_one():
    h = hits("a", "fox", "vixen")
    correct = {Path("/c/fox.jpg"), Path("/c/vixen.jpg")}
    assert bakeoff.mrr(h, correct) == 0.5


def test_mrr_is_one_when_the_top_hit_is_correct():
    assert bakeoff.mrr(hits("fox", "a"), {Path("/c/fox.jpg")}) == 1.0


# --- summary rendering -----------------------------------------------------
# A skipped arm must be visibly skipped. If it rendered as 0.000, the operator
# would read "FG-CLIP lost" when the truth is "FG-CLIP never ran".

def test_summary_reports_both_roles_separately():
    md = bakeoff.render_summary(
        [{"encoder": "siglip_so400m_384", "R@1": 0.5, "R@5": 0.9, "MRR": 0.7, "n": 10}],
        [{"encoder": "siglip_so400m_384", "accuracy": 0.8, "n": 20, "n_classes": 4}],
    )
    assert "Retriever" in md and "Crop scorer" in md
    assert "0.500" in md and "0.800" in md


def test_summary_marks_a_skipped_arm_as_skipped_not_as_zero():
    md = bakeoff.render_summary(
        [{"encoder": "fgclip", "skipped": "no fgclip env given"}],
        [{"encoder": "fgclip", "skipped": "no fgclip env given"}],
    )
    assert "skipped" in md.lower()
    assert "no fgclip env given" in md
    assert "0.000" not in md


def test_summary_tells_the_operator_the_winners_need_not_match():
    md = bakeoff.render_summary([], [])
    assert "pipeline.yaml" in md
    assert "need not be the same" in md
