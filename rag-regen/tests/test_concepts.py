from ragregen import concepts


def kinds(cs):
    return {c.phrase: c.kind for c in cs}


def test_target_concept_is_always_first_and_a_subject():
    cs = concepts.parse("an Amur leopard on a snowy ridge", target="Amur leopard")
    assert cs[0].phrase == "Amur leopard"
    assert cs[0].kind == "subject"


def test_numeric_words_produce_a_count_concept():
    cs = concepts.parse("three polar bears on an ice floe", target="polar bear")
    assert any(c.kind == "count" for c in cs)


def test_digits_produce_a_count_concept():
    cs = concepts.parse("2 axolotls in a tank", target="axolotl")
    assert any(c.kind == "count" for c in cs)


def test_spatial_words_produce_a_relation_concept():
    cs = concepts.parse("a durian behind a wooden fence", target="durian")
    assert any(c.kind == "relation" for c in cs)


def test_repeated_digit_is_emitted_once():
    """Repeated digits must not produce duplicate count concepts."""
    cs = concepts.parse("2 cats and 2 dogs", target="cat")
    phrases = [c.phrase for c in cs]
    assert phrases.count("2") == 1, "Digit '2' must appear exactly once despite two occurrences in prompt"
    assert len(phrases) == len(set(phrases)), "No phrases should be duplicated"


def test_target_overlapping_a_count_word_is_not_duplicated():
    """When target word matches a count word, it must be emitted only once as subject."""
    cs = concepts.parse("two dogs and two cats", target="two")
    phrases = [c.phrase for c in cs]
    assert phrases.count("two") == 1, "Target 'two' must appear exactly once despite being both target and count word"
    assert len(phrases) == len(set(phrases)), "No phrases should be duplicated"


def test_every_kind_is_known():
    cs = concepts.parse("three foxes behind a tree", target="fox")
    assert all(c.kind in concepts.KINDS for c in cs)


def test_multi_word_number_phrase_matches():
    """Test that multi-word number phrases like 'a pair of' are detected correctly."""
    cs = concepts.parse("a pair of oranges on a table", target="orange")
    phrase_kinds = kinds(cs)
    assert "a pair of" in phrase_kinds
    assert phrase_kinds["a pair of"] == "count"


def test_multi_word_spatial_phrase_matches():
    """Test that multi-word spatial phrases like 'in front of' are detected correctly."""
    cs = concepts.parse("a cat in front of a door", target="cat")
    phrase_kinds = kinds(cs)
    assert "in front of" in phrase_kinds
    assert phrase_kinds["in front of"] == "relation"


def test_number_word_inside_a_longer_word_is_not_matched():
    """Number words inside longer words must not produce count concepts due to word boundaries."""
    cs = concepts.parse("a kitten on a mat", target="kitten")
    phrase_kinds = kinds(cs)
    assert all(c.kind != "count" for c in cs), \
        "'ten' inside 'kitten' must not register as a count concept"


def test_relation_word_inside_a_longer_word_is_not_matched():
    """Relation words inside longer words must not produce relation concepts due to word boundaries."""
    cs = concepts.parse("a nearby fox underneath a shelf", target="fox")
    phrase_kinds = kinds(cs)
    assert all(c.kind != "relation" for c in cs), \
        "'near' in 'nearby' and 'under' in 'underneath' must not register as relations"


def test_target_concept_carries_the_coarse_term():
    cs = concepts.parse("an African grey parrot behind a branch",
                        target="African grey parrot", coarse="parrot")
    assert cs[0].phrase == "African grey parrot"
    assert cs[0].coarse == "parrot"


def test_non_target_concepts_have_no_coarse_term():
    cs = concepts.parse("an African grey parrot behind a wooden branch",
                        target="African grey parrot", coarse="parrot")
    others = [c for c in cs[1:]]
    assert others, "prompt should yield at least one non-target concept"
    assert all(c.coarse is None for c in others)


def test_coarse_defaults_to_none_when_not_supplied():
    cs = concepts.parse("a fox in the snow", target="fox")
    assert all(c.coarse is None for c in cs)
