from numba import njit
import numpy as np
from scipy import sparse

from typing import Callable, Tuple, Union

ScipyCompressedSparse = Union[sparse.csr_matrix, sparse.csc_matrix]


def intersect_indices(a: sparse.csr_matrix, b: sparse.csr_matrix):
    assert a.shape == b.shape
    indptr, indices = intersect_indices_csr_csr(
        a.shape,
        a.indptr,
        a.indices,
        b.indptr,
        b.indices,
    )
    return sparse.csr_matrix(
        (np.ones(len(indices), dtype=bool), indices, indptr), shape=a.shape
    )


@njit
def intersect_indices_csr_csr(
    shape: Tuple[int, int],
    a_indptr: np.ndarray,
    a_indices: np.ndarray,
    # a_data: np.ndarray,
    b_indptr: np.ndarray,
    b_indices: np.ndarray,
    # b_data: np.ndarray,
):
    out_indptr = np.zeros_like(a_indptr)
    out_indices = np.zeros(max(len(a_indices), len(b_indices)), dtype=a_indices.dtype)

    out_idx = 0

    for i in range(shape[0]):

        a_idx = a_indptr[i]
        a_end = a_indptr[i + 1]
        b_idx = b_indptr[i]
        b_end = b_indptr[i + 1]

        while (a_idx < a_end) and (b_idx < b_end):
            a_j = a_indices[a_idx]
            b_j = b_indices[b_idx]
            j = min(a_j, b_j)
            if (a_j == j) and (b_j == j):
                out_indices[out_idx] = j
                out_idx += 1
            a_idx += a_j == j
            b_idx += b_j == j

        out_indptr[i + 1] = out_idx

    out_indices = out_indices[: out_idx + 1]

    return out_indptr, out_indices


def op_intersect_indices(
    op: Callable, a: ScipyCompressedSparse, b: ScipyCompressedSparse
) -> ScipyCompressedSparse:
    assert a.shape == b.shape
    if type(a) != type(b):
        b = type(a)(b)

    a.sort_indices()
    b.sort_indices()

    return type(a)(
        op_intersect_indices_csr_csr(
            op,
            a.indptr,
            a.indices,
            a.data,
            b.indptr,
            b.indices,
            b.data,
            out_dtype=np.array(op(a.data[0], b.data[0])).dtype,
        ),
        a.shape,
    )


@njit
def op_intersect_indices_csr_csr(
    op: Callable,
    a_indptr: np.ndarray,
    a_indices: np.ndarray,
    a_data: np.ndarray,
    b_indptr: np.ndarray,
    b_indices: np.ndarray,
    b_data: np.ndarray,
    out_dtype,
):
    out_indptr = np.zeros_like(a_indptr)
    out_indices = np.zeros(max(len(a_indices), len(b_indices)), dtype=a_indices.dtype)
    out_data = np.zeros(len(out_indices), dtype=out_dtype)

    out_idx = 0

    for i in range(len(a_indptr) - 1):

        a_idx = a_indptr[i]
        a_end = a_indptr[i + 1]
        b_idx = b_indptr[i]
        b_end = b_indptr[i + 1]

        while (a_idx < a_end) and (b_idx < b_end):
            a_j = a_indices[a_idx]
            b_j = b_indices[b_idx]
            j = min(a_j, b_j)
            if (a_j == j) and (b_j == j):
                out_indices[out_idx] = j
                out_data[out_idx] = op(a_data[a_idx], b_data[b_idx])
                out_idx += 1
            a_idx += a_j == j
            b_idx += b_j == j

        out_indptr[i + 1] = out_idx

    out_indices = out_indices[: out_idx + 1]
    out_data = out_data[: out_idx + 1]

    return out_data, out_indices, out_indptr


# def op_intersect_indices_csr_csc(
#     op: Callable,
#     shape: Tuple[int, int],
#     csr_indptr: np.ndarray,
#     csr_indices: np.ndarray,
#     csr_data: np.ndarray,
#     csc_indptr: np.ndarray,
#     csc_indices: np.ndarray,
#     csc_data: np.ndarray,
#     out_dtype,
# ):
#     out_indptr = np.zeros_like(csr_indptr)
#     out_indices = np.zeros(max(len(csr_indices), len(csc_indices)), dtype=csr_indices.dtype)
#     out_data = np.zeros(len(out_indices), dtype=out_dtype)

#     out_idx = 0

#     for i in range(shape[0]):

#         csr_idx = csr_indptr[i]
#         csr_end = csr_indptr[i+1]
#         csc_idx = csc_indptr[i]
#         csc_end = csc_indptr[i+1]

#         while (csr_idx < csr_end) and (csc_idx < csc_end):
#             csr_j = csr_indices[csr_idx]
#             csc_j = csc_indices[csc_idx]
#             j = min(csr_j, csc_j)
#             if (csr_j == j) and (csc_j == j):
#                 out_indices[out_idx] = j
#                 out_data[out_idx] = op(csr_data[csr_idx], csc_data[csc_idx])
#                 out_idx += 1
#             csr_idx += (csr_j == j)
#             csc_idx += (csc_j == j)

#         out_indptr[i + 1] = out_idx

#     out_indices = out_indices[:out_idx+1]
#     out_data = out_data[:out_idx+1]

#     return out_data, out_indices, out_indptr


def union_indices(a: sparse.csr_matrix, b: sparse.csr_matrix):
    assert a.shape == b.shape
    indptr, indices = union_indices_csr_csr(
        a.shape,
        a.indptr,
        a.indices,
        b.indptr,
        b.indices,
    )
    return sparse.csr_matrix(
        (np.ones(len(indices), dtype=bool), indices, indptr), shape=a.shape
    )


@njit
def union_indices_csr_csr(
    shape: Tuple[int, int],
    a_indptr: np.ndarray,
    a_indices: np.ndarray,
    # a_data: np.ndarray,
    b_indptr: np.ndarray,
    b_indices: np.ndarray,
    # b_data: np.ndarray,
):
    out_indptr = np.zeros_like(a_indptr)
    out_indices = np.zeros(len(a_indices) + len(b_indices), dtype=a_indices.dtype)

    out_idx = 0

    for i in range(shape[0]):

        a_idx = a_indptr[i]
        a_end = a_indptr[i + 1]
        b_idx = b_indptr[i]
        b_end = b_indptr[i + 1]

        while (a_idx < a_end) and (b_idx < b_end):
            a_j = a_indices[a_idx]
            b_j = b_indices[b_idx]
            if a_j < b_j:
                out_indices[out_idx] = a_j
                a_idx += 1
            elif b_j < a_j:
                out_indices[out_idx] = b_j
                b_idx += 1
            else:
                out_indices[out_idx] = a_j
                a_idx += 1
                b_idx += 1
            out_idx += 1

        # Catch up the other set
        while a_idx < a_end:
            a_j = a_indices[a_idx]
            out_indices[out_idx] = a_j
            a_idx += 1
            out_idx += 1

        while b_idx < b_end:
            b_j = b_indices[b_idx]
            out_indices[out_idx] = b_j
            b_idx += 1
            out_idx += 1

        out_indptr[i + 1] = out_idx

    out_indices = out_indices[: out_idx + 1]

    return out_indptr, out_indices


def op_union_indices(
    op: Callable, a: sparse.csr_matrix, b: sparse.csr_matrix, *, default_value=0
):
    assert a.shape == b.shape

    if type(a) != type(b):
        b = type(a)(b)
    a.sort_indices()
    b.sort_indices()

    # TODO: numpy is weird with bools here
    out_dtype = np.array(op(a.data[0], b.data[0])).dtype
    default_value = out_dtype.type(default_value)
    return type(a)(
        op_union_indices_csr_csr(
            op,
            a.indptr,
            a.indices,
            a.data,
            b.indptr,
            b.indices,
            b.data,
            out_dtype=out_dtype,
            default_value=default_value,
        ),
        a.shape,
    )


@njit
def op_union_indices_csr_csr(
    op: Callable,
    a_indptr: np.ndarray,
    a_indices: np.ndarray,
    a_data: np.ndarray,
    b_indptr: np.ndarray,
    b_indices: np.ndarray,
    b_data: np.ndarray,
    out_dtype,
    default_value,
):
    out_indptr = np.zeros_like(a_indptr)
    out_indices = np.zeros(len(a_indices) + len(b_indices), dtype=a_indices.dtype)
    out_data = np.zeros(len(out_indices), dtype=out_dtype)

    out_idx = 0

    for i in range(len(a_indptr) - 1):

        a_idx = a_indptr[i]
        a_end = a_indptr[i + 1]
        b_idx = b_indptr[i]
        b_end = b_indptr[i + 1]

        while (a_idx < a_end) and (b_idx < b_end):
            a_j = a_indices[a_idx]
            b_j = b_indices[b_idx]
            if a_j < b_j:
                out_indices[out_idx] = a_j
                out_data[out_idx] = op(a_data[a_idx], default_value)
                a_idx += 1
            elif b_j < a_j:
                out_indices[out_idx] = b_j
                out_data[out_idx] = op(default_value, b_data[b_idx])
                b_idx += 1
            else:
                out_indices[out_idx] = a_j
                out_data[out_idx] = op(a_data[a_idx], b_data[b_idx])
                a_idx += 1
                b_idx += 1
            out_idx += 1

        # Catch up the other set
        while a_idx < a_end:
            a_j = a_indices[a_idx]
            out_indices[out_idx] = a_j
            out_data[out_idx] = op(a_data[a_idx], default_value)
            a_idx += 1
            out_idx += 1

        while b_idx < b_end:
            b_j = b_indices[b_idx]
            out_indices[out_idx] = b_j
            out_data[out_idx] = op(default_value, b_data[b_idx])
            b_idx += 1
            out_idx += 1

        out_indptr[i + 1] = out_idx

    out_indices = out_indices[: out_idx + 1]
    out_data = out_data[: out_idx + 1]

    return out_data, out_indices, out_indptr


def difference_indices(
    a: ScipyCompressedSparse, b: ScipyCompressedSparse
) -> ScipyCompressedSparse:
    assert a.shape == b.shape

    if type(a) != type(b):
        b = type(a)(b)
    a.sort_indices()
    b.sort_indices()

    indices, indptr = difference_indices_csr_csr(
        a.indptr, a.indices, b.indptr, b.indices
    )

    return type(a)((np.ones(len(indices), dtype=bool), indices, indptr), shape=a.shape)


@njit
def difference_indices_csr_csr(
    a_indptr: np.ndarray,
    a_indices: np.ndarray,
    b_indptr: np.ndarray,
    b_indices: np.ndarray,
):
    out_indptr = np.zeros_like(a_indptr)
    out_indices = np.zeros_like(a_indices)

    out_idx = 0

    for i in range(len(a_indptr) - 1):

        a_idx = a_indptr[i]
        a_end = a_indptr[i + 1]
        b_idx = b_indptr[i]
        b_end = b_indptr[i + 1]

        while (a_idx < a_end) and (b_idx < b_end):
            a_j = a_indices[a_idx]
            b_j = b_indices[b_idx]
            if a_j < b_j:
                out_indices[out_idx] = a_j
                a_idx += 1
                out_idx += 1
            elif b_j < a_j:
                b_idx += 1
            else:
                a_idx += 1
                b_idx += 1

        # Catch up the other set
        while a_idx < a_end:
            a_j = a_indices[a_idx]
            out_indices[out_idx] = a_j
            a_idx += 1
            out_idx += 1

        if b_idx < b_end:
            b_idx = b_end

        out_indptr[i + 1] = out_idx

    out_indices = out_indices[: out_idx + 1]

    return out_indices, out_indptr


def symdifference_indices(a, b):
    assert a.shape == b.shape

    if type(a) != type(b):
        b = type(a)(b)
    a.sort_indices()
    b.sort_indices()

    indices, indptr = symdifference_indices_csr_csr(
        a.indptr, a.indices, b.indptr, b.indices
    )

    return type(a)((np.ones(len(indices), dtype=bool), indices, indptr), shape=a.shape)


@njit
def symdifference_indices_csr_csr(
    a_indptr: np.ndarray,
    a_indices: np.ndarray,
    b_indptr: np.ndarray,
    b_indices: np.ndarray,
):
    out_indptr = np.zeros_like(a_indptr)
    out_indices = np.zeros(len(a_indices) + len(b_indices), dtype=a_indices.dtype)

    out_idx = 0

    for i in range(len(a_indptr) - 1):

        a_idx = a_indptr[i]
        a_end = a_indptr[i + 1]
        b_idx = b_indptr[i]
        b_end = b_indptr[i + 1]

        while (a_idx < a_end) and (b_idx < b_end):
            a_j = a_indices[a_idx]
            b_j = b_indices[b_idx]
            if a_j < b_j:
                out_indices[out_idx] = a_j
                a_idx += 1
                out_idx += 1
            elif b_j < a_j:
                out_indices[out_idx] = b_j
                b_idx += 1
                out_idx += 1
            else:
                a_idx += 1
                b_idx += 1

        # Catch up the other set
        while a_idx < a_end:
            a_j = a_indices[a_idx]
            out_indices[out_idx] = a_j
            a_idx += 1
            out_idx += 1

        while b_idx < b_end:
            b_j = b_indices[b_idx]
            out_indices[out_idx] = b_j
            b_idx += 1
            out_idx += 1

        out_indptr[i + 1] = out_idx

    out_indices = out_indices[: out_idx + 1]

    return out_indices, out_indptr
