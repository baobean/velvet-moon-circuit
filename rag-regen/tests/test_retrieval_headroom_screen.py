import numpy as np

from scripts.retrieval_headroom_screen import leave_one_out_top1


def test_perfect_separation_scores_100_percent():
    vecs = np.array([
        [1.0, 0.0], [0.9, 0.1],   # species A
        [0.0, 1.0], [0.1, 0.9],   # species B
    ])
    labels = ["A", "A", "B", "B"]
    per_label, acc = leave_one_out_top1(vecs, labels)
    assert acc == 1.0
    assert per_label == {"A": [2, 2], "B": [2, 2]}


def test_indistinguishable_species_score_near_zero():
    # Constructed so every row's nearest OTHER row is of the opposite label.
    vecs = np.array([
        [1.0, 0.0], [1.01, 0.0],   # A0, B0 (mislabelled on purpose: nearest of A0 is B0)
        [0.0, 1.0], [0.01, 1.0],   # A1, B1
    ])
    labels = ["A", "B", "A", "B"]
    per_label, acc = leave_one_out_top1(vecs, labels)
    assert acc == 0.0


def test_self_excluded_from_neighbor_search():
    # A single unique row per label -- its only possible neighbour is a
    # DIFFERENT label's row, proving self is excluded (an unguarded search
    # would trivially return itself with similarity 1.0).
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = ["A", "B"]
    per_label, acc = leave_one_out_top1(vecs, labels)
    assert acc == 0.0
    assert per_label == {"A": [0, 1], "B": [0, 1]}
