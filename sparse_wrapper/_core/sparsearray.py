from abc import ABC, abstractmethod
from math import sin
import numbers
from functools import singledispatch
from numbers import Number

import numpy as np
from numba import njit
import scipy.sparse as ss
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs

from sparse import COO

from .indexing import major_index_int, minor_index_int
from .coordinate_ops import op_union_indices


def sparse_dask(arr, chunks):
    return assparsearray(arr).asdask(chunks)


def row_scale(sparse_dask_array, scale):
    def row_scale_block(X, block_info=None):
        if block_info == "__block_info_dummy__":
            return X
        loc = block_info[0]["array-location"][0]
        if isinstance(X, CompressedSparseArray):
            return X.inplace_row_scale(scale[loc[0] : loc[1]])
        else:
            return X / scale[loc[0] : loc[1]][:, np.newaxis]

    return sparse_dask_array.map_blocks(row_scale_block, dtype=sparse_dask_array.dtype)


def _convert_to_numpy_array(arr, dtype=None):
    if isinstance(arr, np.ndarray):
        ret = arr
    else:
        ret = arr.toarray()
    if dtype and ret.dtype != dtype:
        ret = ret.astype(dtype)
    return ret


def _calculation_method(name):
    def calc(self, axis=None, out=None, dtype=None, **kwargs):
        if axis is None:
            return getattr(self.value, name)(axis)
        elif axis == 0 or axis == 1:
            return getattr(self.value, name)(axis).A.squeeze()
        elif (
            isinstance(axis, tuple)
            and len(axis) == 1
            and (axis[0] == 0 or axis[0] == 1)
        ):
            return getattr(self.value, name)(axis[0]).A
        elif isinstance(axis, tuple):
            v = self.value
            for ax in axis:
                v = getattr(v, name)(ax)
            return assparsearray(ss.csr_matrix(v))
        return assparsearray(ss.csr_matrix(getattr(self.value, name)(axis)))

    return calc


class CompressedSparseArray(np.lib.mixins.NDArrayOperatorsMixin, ABC):
    """
    An wrapper around scipy.sparse to allow sparse arrays to be the chunks in a dask array.
    """

    @property
    @abstractmethod
    def _compressed_dim(self) -> int:
        pass

    __array_priority__ = 10.0

    def __array__(self, dtype=None, **kwargs):
        # respond to np.asarray
        return _convert_to_numpy_array(self.value, dtype)

    _HANDLED_FUNCTIONS = {}

    def __array_function__(self, func, types, args, kwargs):
        if func in self._HANDLED_FUNCTIONS:
            result = self._HANDLED_FUNCTIONS[func](*args, **kwargs)
        else:
            result = func(
                *(x.value if isinstance(x, CompressedSparseArray) else x for x in args),
                **kwargs,
            )
        if issparse(result):
            result = assparsearray(result)
        elif isinstance(result, np.matrix):
            result = np.asarray(result)
        return result

    # One might also consider adding the built-in list type to this
    # list, to support operations like np.add(array_like, list)
    _HANDLED_TYPES = (np.ndarray, numbers.Number, COO, ss.spmatrix)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get("out", ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            if not isinstance(x, self._HANDLED_TYPES + (CompressedSparseArray,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(
            x.value if isinstance(x, CompressedSparseArray) else x for x in inputs
        )
        if out:
            kwargs["out"] = tuple(
                x.value if isinstance(x, CompressedSparseArray) else x for x in out
            )
        # special case multiplication for sparse input, so it is elementwise, not matrix multiplication
        if (
            ufunc.__name__ == "multiply"
            and len(inputs) == 2
            and issparse(inputs[0])
            and issparse(inputs[1])
        ):
            result = inputs[0].multiply(inputs[1])
        elif (
            ufunc.__name__ == "true_divide"
            and len(inputs) == 2
            and issparse(inputs[0])
            and issparse(inputs[1])
        ):
            result = inputs[0] / inputs[1]
        elif (
            ufunc.__name__ == "floor_divide"
            and len(inputs) == 2
            and issparse(inputs[0])
            and issparse(inputs[1])
        ):

            result = inputs[0] / inputs[1]
        elif ufunc.__name__ == "floor_divide" and len(inputs) == 2:
            result = inputs[0]._divide(inputs[1])
        elif ufunc.__name__ == "matmul" and len(inputs) == 2:
            arg1, arg2 = (
                arg.value if isinstance(arg, CompressedSparseArray) else arg
                for arg in inputs
            )
            result = arg1 @ arg2
        elif ufunc.__name__ == "power" and len(inputs) == 2:
            arg1, arg2 = (
                arg.value if isinstance(arg, CompressedSparseArray) else arg
                for arg in inputs
            )
            result = op_union_indices(njit(lambda x, y: np.power(x, y)), arg1, arg2)
        else:
            result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x) for x in result)
        elif method == "at":
            # no return value
            return None
        else:
            # one return value
            return type(self)(result)

    def __repr__(self):
        return f"<{type(self).__name__}: shape={self.shape}, dtype={self.dtype}>"

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def data(self):
        return self.value.data

    @property
    def indices(self):
        return self.value.indices

    @property
    def indptr(self):
        return self.value.indptr

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, item):
        dense_dim = abs(self._compressed_dim - 1)
        if not isinstance(item, tuple):
            item = (item, slice(None))
        assert len(item) == 2
        # If one or more index is an int
        if isinstance(item[0], Number) and isinstance(item[1], Number):
            return self.value[item]
        if isinstance(item[self._compressed_dim], Number):
            return major_index_int(self, item[self._compressed_dim])[item[dense_dim]]
        elif isinstance(item[abs(self._compressed_dim - 1)], Number):
            return minor_index_int(self, item[dense_dim], item[self._compressed_dim])
        # replace slices that span the entire column or row with slice(None) to ensure cupy sparse doesn't blow up
        if (
            isinstance(item[0], slice)
            and item[0].start == 0
            and item[0].stop == self.shape[0]
            and item[0].step is None
        ):
            item0 = slice(None)
        else:
            item0 = item[0]
        if (
            isinstance(item[1], slice)
            and item[1].start == 0
            and item[1].stop == self.shape[1]
            and item[1].step is None
        ):
            item1 = slice(None)
        else:
            item1 = item[1]
        return assparsearray(self.value.__getitem__((item0, item1)))

    def __lt__(self, other):
        return assparsearray(self.value < _get_value(other))

    def __le__(self, other):
        return assparsearray(self.value <= _get_value(other))

    def __eq__(self, other):
        return assparsearray(self.value == _get_value(other))

    def __ne__(self, other):
        return assparsearray(self.value != _get_value(other))

    def __gt__(self, other):
        return assparsearray(self.value > _get_value(other))

    def __ge__(self, other):
        return assparsearray(self.value >= _get_value(other))

    def copy(self):
        return assparsearray(self.value.copy())

    def astype(self, dtype, copy=True):
        dtype = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
        if copy:
            return assparsearray(self.value.astype(dtype))
        else:
            self.value = self.value.astype(dtype, copy=copy)
            return self

    mean = _calculation_method("mean")
    argmax = _calculation_method("argmax")
    min = _calculation_method("min")
    argmin = _calculation_method("argmin")
    sum = _calculation_method("sum")
    prod = _calculation_method("prod")
    all = _calculation_method("all")
    any = _calculation_method("any")

    def inplace_row_scale(self, scale):
        sparsefuncs.inplace_row_scale(self.value, scale)
        return self

    def getnnz(self):
        return self.value.getnnz()

    def asdask(self, chunks):
        import dask.array as da

        return da.from_array(self, chunks=chunks, asarray=False, fancy=False)

    def todense(self):
        return np.asarray(self)

    def transpose(self):
        return assparsearray(self.value.transpose())


class CSR(CompressedSparseArray):
    @property
    def _compressed_dim(self) -> int:
        return 0

    def __init__(self, a):
        if isinstance(a, CSC):
            self.value = CSC.value.tocsr()
        elif isinstance(a, ss.csr_matrix):
            self.value = a
        elif isinstance(a, COO):
            self.value = a.tocsr()
        else:
            self.value = ss.csr_matrix(a)


class CSC(CompressedSparseArray):
    @property
    def _compressed_dim(self) -> int:
        return 1

    def __init__(self, a):
        if isinstance(a, CSR):
            self.value = CSR.value.tocsr()
        elif isinstance(a, ss.csc_matrix):
            self.value = a
        elif isinstance(a, COO):
            self.value = a.tocsc()
        else:
            self.value = ss.csc_matrix(a)


def implements(np_function):
    "Register an __array_function__ implementation for CompressedSparseArray objects."

    def decorator(func):
        CompressedSparseArray._HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


def _concatenate(L, axis=0):
    if len(L) == 1:
        return L[0]
    if axis == 0:
        return assparsearray(scipy.sparse.vstack(tuple([sa.value for sa in L])))
    elif axis == 1:
        return assparsearray(scipy.sparse.hstack(tuple([sa.value for sa in L])))
    else:
        msg = (
            "Can only concatenate sparse matrices for axis in " "{0, 1}.  Got %s" % axis
        )
        raise ValueError(msg)


# register concatenate if Dask is installed
try:
    from dask.array.core import concatenate_lookup

    concatenate_lookup.register(CompressedSparseArray, _concatenate)
except ImportError:
    pass


@implements(np.all)
def _all(a):
    if a.getnnz() != np.multiply(*a.shape):
        return np.bool_(False)
    else:
        return np.all(a.value.data)


@implements(np.any)
def _any(a):
    return np.any(a.value.data)


@singledispatch
def _get_value(a):
    return a


@_get_value.register(CompressedSparseArray)
def _(a):
    return a.value


@singledispatch
def assparsearray(a):
    return NotImplemented


@assparsearray.register(ss.csr_matrix)
def _(a: ss.csr_matrix):
    return CSR(a)


@assparsearray.register(ss.csc_matrix)
def _(a: ss.csc_matrix):
    return CSC(a)


@assparsearray.register(np.ndarray)
def _(a: np.ndarray):
    return CSR(ss.csr_matrix(a))


@assparsearray.register(ss.coo_matrix)
def _(a: ss.coo_matrix):
    return COO(a)
