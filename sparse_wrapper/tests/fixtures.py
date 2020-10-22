import numpy as np
from scipy import sparse
import pytest

from sparse_wrapper import CSC, CSR, assparsearray


@pytest.fixture(params=[CSR, CSC])
def random_array(request):
    return request.param(sparse.random(100, 100))


@pytest.fixture(params=[bool, int, float])
def dtype(request):
    return request.param


@pytest.fixture(params=[sparse.csr_matrix, sparse.csc_matrix])
def matrix_type(request):
    return request.param
