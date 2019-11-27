import numpy as np
from scipy import sparse as ss
import pytest

from sparse_wrapper import SparseArray


@pytest.fixture(params=["csr", "csc"])
def random_array(request):
    return SparseArray(ss.random(100, 100, format=request.param))


@pytest.fixture(params=[ss.csr_matrix, ss.csc_matrix])
def matrix_type(request):
    return request.param
