"""The coarse map and case generation.

`config.load_dataset` rejects coarse == concept because identical terms make
the fine-grained margin identically zero, silently disabling the test. Every
entry here has to clear that bar before it can become a case.
"""
from pathlib import Path

import yaml

from ragregen import casegen

COARSE_PATH = Path("configs/imagenet_coarse.yaml")


def _coarse_map():
    return yaml.safe_load(COARSE_PATH.read_text())


def test_the_map_offers_more_candidates_than_the_70_we_need():
    #: Ranking picks 70 by corpus frequency. A pool of exactly 70 would make
    #: the ranking decorative.
    assert len(_coarse_map()) >= 120


def test_no_entry_has_coarse_equal_to_concept():
    #: config.load_dataset:111 raises on this, so a bad entry aborts
    #: make_cases rather than producing a degenerate case.
    bad = [k for k, v in _coarse_map().items()
           if k.strip().lower() == str(v).strip().lower()]
    assert bad == []


def test_every_coarse_term_is_a_non_empty_string():
    assert all(isinstance(v, str) and v.strip() for v in _coarse_map().values())


def test_concepts_are_lowercase_and_unique():
    #: Ranking matches captions case-insensitively; storing mixed case here
    #: would make the map's keys disagree with the manifest's.
    m = _coarse_map()
    assert all(k == k.lower() for k in m)
    assert len(m) == len(set(m))


def test_no_coarse_term_is_an_abstract_category():
    #: mask_draft grounds the coarse term with GroundingDINO, so a coarse
    #: value has to name something a detector can find. Function words
    #: ("support", "signal") and scene words ("landscape") ground to nothing,
    #: which yields an empty or arbitrary mask rather than a visible failure.
    abstract = {"mammal", "structure", "support", "landscape", "signal",
                "accessory", "cycle"}
    found = sorted({v for v in _coarse_map().values() if v in abstract})
    assert found == []


def test_no_key_is_a_homonym_with_a_dominant_non_imagenet_sense():
    #: corpus.caption_matches is a word-boundary regex with no sense
    #: disambiguation, and rank_classes takes the top 70 by descending count
    #: -- so frequency ranking systematically selects FOR classes whose names
    #: are common English words meaning something else. The consequence is
    #: directional and it corrupts the headline: retrieval returns toy kites
    #: and construction cranes, the pipeline pastes one over a bird draft,
    #: DINO scores it against the ImageNet bird answer key and drops, and the
    #: stratum reads `harm` -- the method blamed for a corpus-builder
    #: artifact. corpus_density looks healthy for exactly these cases, so
    #: per-case attribution will not catch it.
    #: `file` was not caught by inspection -- it was caught by the corpus
    #: itself. It ranked 13th of the selected 70 on 105 caption matches, and
    #: every sampled one was a Wikimedia filename prefix ("File:Johnsmith.png",
    #: "File:KIPOT FALLS...jpg"). Not one filing cabinet. Sample the captions
    #: before trusting a count; a high score is evidence of a common WORD, not
    #: of a well-covered concept.
    homonyms = {"kite", "crane", "drake", "jay", "robin", "hen", "bow",
                "mouse", "notebook", "jaguar", "impala", "cougar", "beaver",
                "dam", "file"}
    found = sorted(homonyms & set(_coarse_map()))
    assert found == []


def test_ranking_prefers_the_more_common_class():
    counts = {"fox": 500, "axolotl": 3, "violin": 120}
    coarse = {"fox": "canine", "axolotl": "salamander",
              "violin": "string instrument"}
    got = casegen.rank_classes(counts, coarse, exclude=set(), n=2)
    assert got == ["fox", "violin"]


def test_ranking_only_considers_classes_in_the_coarse_map():
    #: A class with no groundable superordinate cannot become a case at all,
    #: so it must not consume one of the 70 slots.
    counts = {"fox": 500, "nematode": 900}
    coarse = {"fox": "canine"}
    assert casegen.rank_classes(counts, coarse, exclude=set(), n=5) == ["fox"]


def test_ranking_excludes_the_bridge_concepts():
    #: The 22 legacy cases are carried forward verbatim; re-deriving one as a
    #: `common` case would duplicate the id and load_dataset would raise.
    counts = {"fox": 500, "violin": 400}
    coarse = {"fox": "canine", "violin": "string instrument"}
    got = casegen.rank_classes(counts, coarse, exclude={"fox"}, n=5)
    assert got == ["violin"]


def test_ranking_breaks_ties_alphabetically():
    #: Determinism matters more than which one wins: the same manifest must
    #: produce the same 70 classes on a re-run.
    counts = {"beta": 10, "alpha": 10}
    coarse = {"alpha": "a", "beta": "b"}
    assert casegen.rank_classes(counts, coarse, exclude=set(),
                                n=2) == ["alpha", "beta"]


def test_ranking_returns_fewer_than_asked_when_the_pool_is_small():
    counts = {"fox": 5}
    assert casegen.rank_classes(counts, {"fox": "canine"}, exclude=set(),
                                n=70) == ["fox"]


def test_ranking_excludes_classes_with_zero_count():
    #: A class the corpus never mentions is not a common one. Including it
    #: would give the `full` arm's retrieval nothing to find, and corrupt
    #: the benchmark by measuring phantom commonness. The default min_count
    #: of 1 is exactly this rule, generalised.
    counts = {"fox": 500, "violin": 400}
    coarse = {"fox": "canine", "violin": "string instrument", "nematode": "worm"}
    got = casegen.rank_classes(counts, coarse, exclude=set(), n=5)
    assert got == ["fox", "violin"]
    assert "nematode" not in got


def test_ranking_drops_a_concept_the_corpus_cannot_serve_k_refs_for():
    #: cmd_select seeds per-concept rows for the DATASET's 22 concepts only,
    #: so a `common` concept has to be found in the random bulk -- where
    #: corpus.py's own docstring says a 100k sample yields single digits. A
    #: 1-caption concept becoming a case then fails to serve `k` references
    #: hours into the run. The boundary is inclusive: exactly min_count is
    #: enough.
    counts = {"fox": 500, "violin": 3, "oboe": 2}
    coarse = {"fox": "canine", "violin": "string instrument",
              "oboe": "wind instrument"}
    got = casegen.rank_classes(counts, coarse, exclude=set(), n=5,
                               min_count=3)
    assert got == ["fox", "violin"]
    assert "oboe" not in got


def test_a_prompt_mentions_its_concept():
    assert "golden retriever" in casegen.frame_prompt("golden retriever", 0)


def test_prompts_are_deterministic_for_an_index():
    assert (casegen.frame_prompt("fox", 3) == casegen.frame_prompt("fox", 3))


def test_consecutive_indices_use_different_frames():
    #: 70 identical sentence shapes would let the verifier key on the frame
    #: rather than the concept.
    a = casegen.frame_prompt("fox", 0)
    b = casegen.frame_prompt("fox", 1)
    assert a != b


def test_the_frame_pool_is_large_enough_to_vary():
    assert len(casegen.SCENE_FRAMES) >= 8
