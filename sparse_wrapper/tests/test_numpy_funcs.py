import numpy as np
from scipy import sparse as ss

# import pytest

from sparse_wrapper import SparseArray

from fixtures import random_array, matrix_type

npTrue = np.bool_(True)
npFalse = np.bool_(False)


def test_all(matrix_type):
    mtx = matrix_type(np.eye(100))
    arr = SparseArray(mtx)
    assert not np.all(arr)
    assert np.all(SparseArray(matrix_type(np.ones((10, 10))))) is npTrue


def test_any(matrix_type):
    assert not np.any(SparseArray(matrix_type(np.zeros((100, 100)))))
    assert np.any(SparseArray(matrix_type(np.eye(100))))
