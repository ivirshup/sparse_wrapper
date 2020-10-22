from fixtures import matrix_type

from functools import singledispatch
from sparse_wrapper import assparsearray
import numpy as np
from scipy import sparse

from sparse import COO

import pytest

@singledispatch
def asnparray(x):
    return np.asarray(x)

@asnparray.register(sparse.spmatrix)
def _(x):
    return x.toarray()

@asnparray.register(COO)
def _(x):
    return x.todense()


def check_indexer(a, idx, idx_like):
    assert np.array_equal(np.asarray(a[idx]), idx_like(a.value)[idx].toarray())


def make_idx_array(maxn, n=10, repeat=False, sorted=False):
    idx = np.random.choice(maxn, size=n, replace=~repeat)
    if sorted:
        idx.sort()
    return idx

@pytest.mark.parametrize(
    "idx",
    [
        1,
        (1, 4),
        slice(None, None, 2),
        np.array([1, 40, 2]),
        (slice(5), slice(5)),
        (slice(None, None, 2), 1),
        (1, slice(None, None, 2)),
        (slice(None), slice(None)),
        (make_idx_array(50), slice(None)),
        (slice(None), make_idx_array(99)),
        np.ix_(make_idx_array(49, 20), make_idx_array(99, 50))
    ]
)
def test_indexing(matrix_type, idx):
    a = assparsearray(matrix_type(sparse.random(50, 100, format="csr", density=0.4)))

    true_result = a.value.toarray()[idx]
    curr_result = a[idx]
    assert np.array_equal(asnparray(curr_result), true_result)

