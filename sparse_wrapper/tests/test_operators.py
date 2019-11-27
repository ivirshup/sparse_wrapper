# TODO: Also the inplace versions
# TODO: Use of random arrays is a little flaky
from operator import mul, matmul, add, sub, pow, truediv, floordiv
from operator import eq, ne, gt, ge, lt, le  # Comparison
from operator import and_, or_, xor  # Binary
from operator import lshift, rshift
from operator import pos, neg, abs, invert # Unary

import numpy as np
from sparse import COO
from scipy import sparse as ss
import pytest

from sparse_wrapper import SparseArray

from fixtures import matrix_type, random_array

another_matrix_type = matrix_type
another_random_array = random_array

UNARY_OPERATORS = [pos, neg, abs]
BINARY_OPERATORS = [and_, or_, xor]  # Note, xor equivalent to neq(sparsepat1, sparsepat2)
COMPARISON_OPERATORS = [eq, ne, gt, ge, lt, le]
MATH_OPERATORS = [mul, matmul, add, sub, pow, truediv, floordiv]
SCIPYSPARSE_NOT_IMPLEMENTED = [and_, or_, xor, invert, floordiv, mul, pow]
# mul and pow work when the other is a scalar

@pytest.fixture(params=UNARY_OPERATORS)
def unary_op(request):
    return request.param


@pytest.fixture(params=BINARY_OPERATORS)
def binary_op(request):
    return request.param


@pytest.fixture(params=MATH_OPERATORS)
def math_op(request):
    return request.param


@pytest.fixture(params=COMPARISON_OPERATORS)
def comparison_op(request):
    return request.param


@pytest.fixture(
    params=[np.asarray, lambda x: x.value.copy(), lambda x: COO(x.value)],
    ids=["ndarray", "spmatrix", "COO"],
)
def alternate_form(request):
    return request.param


# Not all of these are necessarily going to work with sparse matrices, all should work for np.ndarray
def test_math_op(random_array, another_random_array, math_op, alternate_form):
    sarr1, sarr2 = random_array, another_random_array
    alt1, alt2 = alternate_form(sarr1), alternate_form(sarr2)

    sarr_res = math_op(sarr1, sarr2)
    alt_res = math_op(alt1, alt2)
    assert not np.any(sarr_res != alt_res)


def test_heterogenous_math_op(
    random_array, another_random_array, math_op, alternate_form
):
    """
    Test math operator for mixed container operations
    """
    sarr1, sarr2 = random_array, another_random_array
    alt1, alt2 = alternate_form(sarr1), alternate_form(sarr2)

    sarr_res = math_op(sarr1, alt2)
    alt_res = math_op(alt1, alt2)
    assert not np.any(sarr_res != alt_res)


def test_binary_op(matrix_type, another_matrix_type, binary_op, alternate_form):
    sarr1 = SparseArray(matrix_type(ss.random(100, 100, dtype=bool)))
    sarr2 = SparseArray(another_matrix_type(ss.random(100, 100, dtype=bool)))
    alt1, alt2 = alternate_form(sarr1), alternate_form(sarr2)

    sarr_res = binary_op(sarr1, sarr2)
    alt_res = binary_op(alt1, alt2)
    assert not np.any(sarr_res != alt_res)


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
