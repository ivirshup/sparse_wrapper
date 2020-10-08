"""
These are not ideal tests, but are useful for development. Placeholders until I can write some higher level tests.
"""

from numba import njit
import numpy as np
from scipy import sparse
import operator as op

from sparse_wrapper._core.coordinate_ops import (
    op_intersect_indices,
    op_union_indices,
    difference_indices,
)
from fixtures import matrix_type

matrix_type2 = matrix_type


def test_intersect(matrix_type, matrix_type2):
    a = matrix_type(sparse.random(1000, 1000, density=0.2))
    b = matrix_type2(sparse.random(1000, 1000, density=0.2))

    truth = a.multiply(b)
    test = op_intersect_indices(njit(lambda x, y: x * y), a, b)

    assert (truth != test).sum() == 0


def test_union(matrix_type, matrix_type2):
    a = matrix_type(sparse.random(1000, 1000, density=0.2, dtype=bool))
    b = matrix_type2(sparse.random(1000, 1000, density=0.2, dtype=bool))

    truth = a + b
    test = op_union_indices(njit(lambda x, y: x | y), a, b)

    assert (truth != test).sum() == 0

    a = matrix_type(sparse.random(1000, 1000, density=0.2))
    b = matrix_type2(sparse.random(1000, 1000, density=0.2))

    truth = a + b
    test = op_union_indices(njit(lambda x, y: x + y), a, b)

    assert (truth != test).sum() == 0


def test_difference(matrix_type, matrix_type2):
    a = matrix_type(sparse.random(1000, 1000, density=0.2, dtype=bool))
    b = matrix_type2(sparse.random(1000, 1000, density=0.2, dtype=bool))

    truth = a > b
    test = difference_indices(a, b)

    assert (truth != test).sum() == 0
