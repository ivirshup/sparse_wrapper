from os import major
from typing import Union, Tuple, Sequence
import scipy.sparse as ss
from numba import njit, generated_jit, types, typeof
import numpy as np
from sparse import COO

# from .sparsearray import CompressedSparseArray

from typing import Union

# CompressedMatrices2D = Union[CompressedSparseArray, ss.csr_matrix, ss.csc_matrix]


def major_index_int(x, idx: int) -> COO:  #: "CompressedMatrices2D",
    s = slice(x.indptr[idx], x.indptr[idx + 1])
    data = x.data[s]
    indices = x.indices[s]
    return COO(
        indices[None, :],
        data,
        shape=x.shape[abs(1 - x._compressed_dim)],
        sorted=True,
        has_duplicates=False,
    )


def minor_index_int(x, idx, major_idx: Union[slice, np.ndarray] = slice(None)):
    indices, data, shape = minor_idx_adv(
        x.data, x.indices, x.indptr, idx, major_idx=major_idx
    )
    return COO(
        indices[None, :],
        data=data,
        shape=shape,
        sorted=True,
        has_duplicates=False,
    )


@generated_jit(cache=True)
def resolve_indices(a, l):
    if isinstance(a, (types.SliceType, types.SliceLiteral)):
    # if isinstance(a, types.SliceType):
        return lambda a, l: range(*a.indices(l))
    else:
        return lambda a, _: a


@njit(cache=True)
def minor_idx_adv(
    data, indices, indptr, minor_idx, major_idx: Union[slice, np.ndarray] = slice(None)
):
    out_indices = []
    out_data = []

    row_iter = resolve_indices(major_idx, len(indptr) - 1)

    out_indptr = np.zeros(len(row_iter) + 1, dtype=np.int64)
    found = 0

    for idx, i in enumerate(row_iter):

        start = indptr[i]
        stop = indptr[i + 1]
        row_indices = indices[start:stop]
        possible_idx = np.searchsorted(row_indices, minor_idx)
        for j, p_idx in enumerate(possible_idx):
            if row_indices[p_idx] == minor_idx[j]:
                found += 1
                out_indices.append(j)
                out_data.append(data[start + p_idx])
        out_indptr[idx+1] = found

    return (
        np.array(out_data),
        np.array(out_indices),
        out_indptr
    )