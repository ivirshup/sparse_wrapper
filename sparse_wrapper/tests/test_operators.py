import numpy as np
from scipy import sparse as ss

from sparse_wrapper import SparseArray

def test_equality():
    mtx = ss.random(100, 100, format="csr")
    arr = SparseArray(mtx)

    assert np.all(arr == arr.copy())
    # assert np.all(mtx == arr) # TODO
    assert np.all(arr == mtx)
