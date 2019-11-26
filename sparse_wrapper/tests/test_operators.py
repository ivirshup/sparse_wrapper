import numpy as np
from scipy import sparse as ss

from sparse_wrapper import SparseArray

from fixtures import random_array, matrix_type

another_matrix_type = matrix_type

# boooooo
npTrue = np.bool_(True)
npFalse = np.bool_(False)


def test_equality():
    mtx = ss.random(100, 100, format="csr")
    arr = SparseArray(mtx)

    assert np.all(arr == arr.copy()) is np.bool_(True)
    # assert np.all(mtx == arr) # TODO
    assert np.all(arr == mtx)


def test_add(matrix_type, another_matrix_type):
    mtx1 = matrix_type(ss.random(100, 100))
    mtx2 = another_matrix_type(ss.random(100, 100))
    arr1, arr2 = SparseArray(mtx1), SparseArray(mtx2)

    assert np.all(arr1 + arr2 == mtx1 + mtx2)


def test_subtract(matrix_type, another_matrix_type):
    mtx1 = matrix_type(ss.random(100, 100))
    mtx2 = another_matrix_type(ss.random(100, 100))
    arr1, arr2 = SparseArray(mtx1), SparseArray(mtx2)

    assert np.all(arr1 - arr2 == mtx1 - mtx2)


def test_divide_scalar(matrix_type):
    mtx = matrix_type(ss.random(100, 100))
    arr = SparseArray(mtx)

    assert np.all(arr / 2 == mtx / 2)


# def test_divide(matrix_type, another_matrix_type):
#     mtx1 = matrix_type(ss.random(100, 100))
#     mtx2 = another_matrix_type(ss.random(100, 100))
#     arr1, arr2 = SparseArray(mtx1), SparseArray(mtx2)
#     assert np.all(arr1 / arr2 == mtx1 / mtx2)


def test_matmul_identity(random_array):
    assert np.all(random_array @ np.eye(random_array.shape[0]) == random_array)


def test_matmul(matrix_type, another_matrix_type):
    mtx1 = matrix_type(ss.random(100, 100))
    mtx2 = another_matrix_type(ss.random(100, 100))
    arr1, arr2 = SparseArray(mtx1), SparseArray(mtx2)

    assert np.all(arr1 @ arr2 == mtx1 @ mtx2)
