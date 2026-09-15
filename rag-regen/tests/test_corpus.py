"""Corpus selection. No network, no parquet -- just which rows we keep.

The corpus does two jobs (doc 5 §4): the seeded slice guarantees the answer
exists, the random bulk makes finding it non-trivial. Both are tested here
because getting either wrong produces a plausible-looking corpus that
measures nothing.
"""
import pytest

from ragregen import corpus


def test_a_word_boundary_match_hits():
    assert corpus.caption_matches("a red fox in the snow", "fox")


def test_matching_is_case_insensitive():
    assert corpus.caption_matches("A Golden Retriever puppy",
                                  "golden retriever")


def test_a_substring_inside_a_word_is_not_a_match():
    #: "foxglove" is a flower. Without word boundaries the fox slice fills
    #: with plants and retrieval looks broken for reasons that are ours.
    assert not corpus.caption_matches("purple foxglove blooms", "fox")


def test_multi_word_concepts_match_as_a_phrase():
    assert corpus.caption_matches("my golden retriever", "golden retriever")
    assert not corpus.caption_matches("a retriever of golden things",
                                      "golden retriever")


def test_count_per_concept_counts_every_matching_caption():
    caps = ["a fox", "another fox", "a violin", "nothing here"]
    got = corpus.count_per_concept(caps, ["fox", "violin", "panda"])
    assert got == {"fox": 2, "violin": 1, "panda": 0}


def test_select_takes_the_requested_number_per_concept():
    caps = [f"a fox number {i}" for i in range(50)] + ["unrelated"] * 50
    idx = corpus.select_rows(caps, ["fox"], n_random=0, n_per_concept=10,
                             seed=0)
    assert len(idx) == 10
    assert all(caps[i].startswith("a fox") for i in idx)


def test_select_adds_random_bulk_on_top_of_the_seeded_slice():
    caps = [f"a fox number {i}" for i in range(50)] + ["unrelated"] * 50
    idx = corpus.select_rows(caps, ["fox"], n_random=20, n_per_concept=10,
                             seed=0)
    #: The bulk is drawn from everything, so it may re-draw a fox row. The
    #: guarantee is >= the seeded slice and <= seeded + bulk, never double
    #: counting.
    assert 10 <= len(idx) <= 30
    assert len(idx) == len(set(idx))


def test_select_is_deterministic_for_a_seed():
    caps = [f"caption {i}" for i in range(200)]
    a = corpus.select_rows(caps, [], n_random=25, n_per_concept=0, seed=7)
    b = corpus.select_rows(caps, [], n_random=25, n_per_concept=0, seed=7)
    assert a == b


def test_a_different_seed_selects_differently():
    caps = [f"caption {i}" for i in range(200)]
    a = corpus.select_rows(caps, [], n_random=25, n_per_concept=0, seed=7)
    b = corpus.select_rows(caps, [], n_random=25, n_per_concept=0, seed=8)
    assert a != b


def test_a_concept_with_too_few_captions_takes_what_exists():
    #: Not an error. A thin concept is a finding the report must show, and
    #: raising here would abort a 100k download over one rare word.
    caps = ["a lone axolotl"]
    idx = corpus.select_rows(caps, ["axolotl"], n_random=0, n_per_concept=10,
                             seed=0)
    assert idx == [0]


def test_returned_indices_are_sorted():
    #: Parquet take() is much faster on sorted indices, and a sorted list
    #: makes the URL file diffable between runs.
    caps = [f"caption {i}" for i in range(200)]
    idx = corpus.select_rows(caps, [], n_random=40, n_per_concept=0, seed=3)
    assert idx == sorted(idx)
