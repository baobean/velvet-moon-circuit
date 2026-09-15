import numpy as np
from graft.selection import medoid_index

def test_medoid_picks_central_row():
    # rows 0,1 cluster together; row 2 is an outlier -> medoid is 0 or 1, not 2
    embs = np.array([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]])
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    assert medoid_index(embs) in (0, 1)
    assert medoid_index(embs) != 2

def test_medoid_single_row():
    assert medoid_index(np.array([[0.1, 0.2, 0.3]])) == 0
