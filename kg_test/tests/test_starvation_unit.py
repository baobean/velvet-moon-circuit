from graft.starvation import keep_draw, iter_cells, STARVE_CONCEPTS, STARVE_LEVELS, N_DRAWS

def test_keep_draw_deterministic_and_sized():
    refs = [f"r{i}.jpg" for i in range(5)]
    assert keep_draw(refs, 2, seed=0) == keep_draw(refs, 2, seed=0)   # deterministic
    assert len(keep_draw(refs, 2, seed=0)) == 2
    assert keep_draw(refs, 9, seed=0) == refs                          # ceiling
    assert keep_draw(refs, 2, seed=0) != keep_draw(refs, 2, seed=1)    # seed varies draw

def test_cells_cover_grid():
    cells = list(iter_cells())
    assert len(cells) == len(STARVE_CONCEPTS) * len(STARVE_LEVELS) * N_DRAWS
