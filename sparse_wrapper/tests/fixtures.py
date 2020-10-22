import numpy as np
from scipy import sparse
import pytest

from sparse_wrapper import SparseArray


@pytest.fixture(params=["csr", "csc"])
def random_array(request):
    return SparseArray(sparse.random(100, 100, format=request.param))


@pytest.fixture(params=[bool, int, float])
def dtype(request):
    return request.param


@pytest.fixture(params=[sparse.csr_matrix, sparse.csc_matrix])
def matrix_type(request):
    return request.param
