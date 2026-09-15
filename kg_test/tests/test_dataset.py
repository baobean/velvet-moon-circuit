from graft.dataset import Species, list_species, load_species, load_treevill, split_refs

FIX = "tests/fixtures/mini_treevill"
FIX_DUP = "tests/fixtures/mini_treevill_dup"


def test_load_finds_species():
    sp = {s.name: s for s in load_treevill(FIX)}
    assert set(sp) == {"Akashmoni", "Debdaru"}
    assert len(sp["Akashmoni"].images) == 3


def test_split_is_disjoint_and_deterministic():
    sp = load_treevill(FIX)[0]
    b1, h1 = split_refs(sp, k_build=2, seed=0)
    b2, h2 = split_refs(sp, k_build=2, seed=0)
    assert b1 == b2 and h1 == h2  # deterministic
    assert set(b1).isdisjoint(h1)  # disjoint
    assert len(b1) == 2 and len(h1) == 1


def test_load_dedupes_byte_identical_images():
    # SpeciesX/1.jpg and 3.jpg are byte-identical; only one should survive.
    # Real Treevill species were hand-checked to have exactly this shape:
    # a handful of unique images copied hundreds of times each (e.g.
    # Akashmoni: 2 unique / 2000 files; Aloe Wood: 3 unique / 2000).
    sp = load_treevill(FIX_DUP)[0]
    assert sp.name == "SpeciesX"
    assert len(sp.images) == 2  # deduped from 3 files


def test_list_species_is_cheap_name_only_listing():
    assert list_species(FIX) == ["Akashmoni", "Debdaru"]


def test_load_species_matches_load_treevill_for_that_species():
    sp = load_species(FIX_DUP, "SpeciesX")
    assert sp.name == "SpeciesX"
    assert len(sp.images) == 2


def test_adaptive_split_includes_small_species():
    two = Species("two", ["a.jpg", "b.jpg"])
    b, h = split_refs(two, k_build=5, seed=0)
    assert len(b) == 1 and len(h) == 1            # build n-1, hold out 1

    one = Species("one", ["only.jpg"])
    b1, h1 = split_refs(one, k_build=5, seed=0)
    assert b1 == [] and len(h1) == 1              # can't build+hold-out => empty build (eval skips)


def test_adaptive_split_caps_large_species():
    big = Species("big", [f"{i}.jpg" for i in range(15)])
    b, h = split_refs(big, k_build=5, seed=0)
    assert len(b) == 5 and len(h) == 10           # unchanged vs Phase 1 for n>=6
