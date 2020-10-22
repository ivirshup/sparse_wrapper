import numpy as np

from fixtures import random_array

def test_transpose(random_array):
    assert np.array_equal(np.asarray(random_array.T), np.asarray(random_array).T)
