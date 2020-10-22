# TODO: Also the inplace versions
# TODO: Use of random arrays is a little flaky
from operator import mul, matmul, add, sub, pow, truediv, floordiv
from operator import eq, ne, gt, ge, lt, le  # Comparison
from operator import and_, or_, xor  # Binary
from operator import lshift, rshift
from operator import pos, neg, abs, invert  # Unary

from types import FunctionType

import numpy as np
from sparse import COO
from scipy import sparse as ss
import pytest

from sparse_wrapper import CompressedSparseArray, assparsearray

from fixtures import matrix_type, random_array, dtype

another_dtype = dtype
another_matrix_type = matrix_type
another_random_array = random_array

# NOTE: xor equivalent to neq(sparsepat1, sparsepat2)
# NOTE: invert equivalent to neq(sparsepat1, fillvalue)

UNARY_OPERATORS = {pos, neg, abs}
BOOLEAN_OPERATORS = {and_, or_, xor}
COMPARISON_OPERATORS = {eq, ne, gt, ge, lt, le}
MATH_OPERATORS = {mul, matmul, add, sub, pow, truediv, floordiv}
SCIPYSPARSE_NOT_IMPLEMENTED = {and_, or_, xor, invert, floordiv, mul, pow, pos}
# mul and pow work when the other is a scalar


@pytest.fixture(params=UNARY_OPERATORS)
def unary_op(request):
    return request.param


@pytest.fixture(params=BOOLEAN_OPERATORS)
def boolean_op(request):
    return request.param


@pytest.fixture(params=MATH_OPERATORS)
def math_op(request):
    return request.param


@pytest.fixture(params=COMPARISON_OPERATORS)
def comparison_op(request):
    return request.param


def asspmatrix(x):
    return x.value.copy()


# All tests with this currently fail: https://github.com/pydata/sparse/issues/305
def asCOO(x):
    return COO(x.value)


@pytest.fixture(
    params=[
        np.asarray,
        asspmatrix,
        # asCOO,
    ],
    ids=[
        "ndarray",
        "spmatrix",
        # "COO",
    ],
)
def alternate_form(request):
    return request.param


# Not all of these are necessarily going to work with sparse matrices, all should work for np.ndarray
def test_math_op(random_array, another_random_array, math_op, alternate_form, dtype, another_dtype):
    if math_op in SCIPYSPARSE_NOT_IMPLEMENTED and alternate_form == asspmatrix:
        pytest.skip("Not supported")

    sarr1, sarr2 = random_array.astype(dtype), another_random_array.astype(dtype)
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
    if math_op in SCIPYSPARSE_NOT_IMPLEMENTED and alternate_form == asspmatrix:
        pytest.skip("Not supported")

    sarr1, sarr2 = random_array, another_random_array
    alt1, alt2 = alternate_form(sarr1), alternate_form(sarr2)

    sarr_res = math_op(sarr1, alt2)
    alt_res = math_op(alt1, alt2)
    assert not np.any(sarr_res != alt_res)


def test_boolean_op(matrix_type, another_matrix_type, boolean_op, alternate_form):
    if boolean_op in SCIPYSPARSE_NOT_IMPLEMENTED and alternate_form == asspmatrix:
        pytest.skip("Not supported")

    sarr1 = assparsearray(matrix_type(ss.random(100, 100, dtype=bool)))
    sarr2 = assparsearray(another_matrix_type(ss.random(100, 100, dtype=bool)))
    alt1, alt2 = alternate_form(sarr1), alternate_form(sarr2)

    sarr_res = boolean_op(sarr1, sarr2)
    alt_res = boolean_op(alt1, alt2)
    assert not np.any(sarr_res != alt_res)


# These will all fail with COO: https://github.com/pydata/sparse/issues/305
def test_comparison_op(
    random_array, another_random_array, comparison_op, alternate_form
):
    sarr1, sarr2 = random_array, another_random_array
    alt1, alt2 = alternate_form(sarr1), alternate_form(sarr2)

    sarr_res = comparison_op(sarr1, sarr2)
    alt_res = comparison_op(alt1, alt2)
    assert not np.any(sarr_res != alt_res)

    sarr_res_same = comparison_op(sarr1, sarr1.copy())
    alt_res_same = comparison_op(alt1, alt1.copy())
    assert not np.any(sarr_res_same != alt_res_same)


def test_unary_op(random_array, unary_op, alternate_form):
    if unary_op in SCIPYSPARSE_NOT_IMPLEMENTED and alternate_form == asspmatrix:
        pytest.skip("Not supported")
    sarr = random_array
    alt = alternate_form(sarr)

    sarr_res = unary_op(sarr)
    alt_res = unary_op(alt)
    assert not np.any(sarr_res != alt_res)


# boooooo
npTrue = np.bool_(True)
npFalse = np.bool_(False)


def test_equality():
    mtx = ss.random(100, 100, format="csr")
    arr = assparsearray(mtx)

    assert np.all(arr == arr.copy()) is np.bool_(True)
    # assert np.all(mtx == arr) # TODO
    assert np.all(arr == mtx)


def test_add(matrix_type, another_matrix_type):
    mtx1 = matrix_type(ss.random(100, 100))
    mtx2 = another_matrix_type(ss.random(100, 100))
    arr1, arr2 = assparsearray(mtx1), assparsearray(mtx2)

    assert np.all(arr1 + arr2 == mtx1 + mtx2)


def test_subtract(matrix_type, another_matrix_type):
    mtx1 = matrix_type(ss.random(100, 100))
    mtx2 = another_matrix_type(ss.random(100, 100))
    arr1, arr2 = assparsearray(mtx1), assparsearray(mtx2)

    assert np.all(arr1 - arr2 == mtx1 - mtx2)


def test_divide_scalar(matrix_type):
    mtx = matrix_type(ss.random(100, 100))
    arr = assparsearray(mtx)

    assert np.all(arr / 2 == mtx / 2)


# def test_divide(matrix_type, another_matrix_type):
#     mtx1 = matrix_type(ss.random(100, 100))
#     mtx2 = another_matrix_type(ss.random(100, 100))
#     arr1, arr2 = CompressedSparseArray(mtx1), CompressedSparseArray(mtx2)
#     assert np.all(arr1 / arr2 == mtx1 / mtx2)


def test_matmul_identity(random_array):
    assert np.all(random_array @ np.eye(random_array.shape[0]) == random_array)


def test_matmul(matrix_type, another_matrix_type):
    mtx1 = matrix_type(ss.random(100, 100))
    mtx2 = another_matrix_type(ss.random(100, 100))
    arr1, arr2 = assparsearray(mtx1), assparsearray(mtx2)

    assert np.all(arr1 @ arr2 == mtx1 @ mtx2)
