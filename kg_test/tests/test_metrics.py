import numpy as np

from graft.metrics import mean_cosine


def test_mean_cosine_normalized_vectors():
    gen = np.array([1.0, 0.0])
    refs = np.array([[1.0, 0.0], [0.0, 1.0]])  # sims 1.0 and 0.0
    assert abs(mean_cosine(gen, refs) - 0.5) < 1e-6
