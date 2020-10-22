import numbers
import numpy as np
import scipy.sparse as ss
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs

from sparse import COO


def sparse_dask(arr, chunks):
    return SparseArray(arr).asdask(chunks)


def row_scale(sparse_dask_array, scale):
    def row_scale_block(X, block_info=None):
        if block_info == "__block_info_dummy__":
            return X
        loc = block_info[0]["array-location"][0]
        if isinstance(X, SparseArray):
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
            return SparseArray(ss.csr_matrix(v))
        return SparseArray(ss.csr_matrix(getattr(self.value, name)(axis)))

    return calc


class SparseArray(np.lib.mixins.NDArrayOperatorsMixin):
    """
    An wrapper around scipy.sparse to allow sparse arrays to be the chunks in a dask array.
    """

    __array_priority__ = 10.0

    def __init__(self, value):
        if isinstance(value, (np.ndarray, np.matrix)):
            value = ss.csr_matrix(value)
        elif isinstance(value, COO):
            value = value.tocsr()
        elif not issparse(value):
            raise ValueError(
                f"SparseArray only takes a scipy.sparse value, but given {type(value)}"
            )
        self.value = value

    def __array__(self, dtype=None, **kwargs):
        # respond to np.asarray
        return _convert_to_numpy_array(self.value, dtype)

    _HANDLED_FUNCTIONS = {}

    def __array_function__(self, func, types, args, kwargs):
        if func in self._HANDLED_FUNCTIONS:
            result = self._HANDLED_FUNCTIONS[func](*args, **kwargs)
        else:
            result = func(
                *(x.value if isinstance(x, SparseArray) else x for x in args), **kwargs
            )
        if issparse(result):
            result = SparseArray(result)
        elif isinstance(result, np.matrix):
            result = np.asarray(result)
        return result

    # One might also consider adding the built-in list type to this
    # list, to support operations like np.add(array_like, list)
    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get("out", ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use SparseArray instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle SparseArray objects.
            if not isinstance(x, self._HANDLED_TYPES + (SparseArray,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.value if isinstance(x, SparseArray) else x for x in inputs)
        if out:
            kwargs["out"] = tuple(
                x.value if isinstance(x, SparseArray) else x for x in out
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
        elif ufunc.__name__ == "matmul" and len(inputs) == 2:
            arg1, arg2 = (
                arg.value if isinstance(arg, SparseArray) else arg for arg in inputs
            )
            result = arg1 @ arg2
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
        return "%s(%r)" % (type(self).__name__, self.value)

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
        return self.value.indices

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, item):
        if isinstance(item, numbers.Number):
            return _convert_to_numpy_array(self.value.__getitem__(item)).squeeze()
        elif isinstance(item, tuple) and (
            isinstance(item[0], numbers.Number) or isinstance(item[1], numbers.Number)
        ):
            return _convert_to_numpy_array(self.value.__getitem__(item)).squeeze()
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
        return SparseArray(self.value.__getitem__((item0, item1)))

    def _get_value(self, other):
        # get the value if a SparseArray, or just return other
        return other.value if isinstance(other, SparseArray) else other

    def __lt__(self, other):
        return SparseArray(self.value < self._get_value(other))

    def __le__(self, other):
        return SparseArray(self.value <= self._get_value(other))

    def __eq__(self, other):
        return SparseArray(self.value == self._get_value(other))

    def __ne__(self, other):
        return SparseArray(self.value != self._get_value(other))

    def __gt__(self, other):
        return SparseArray(self.value > self._get_value(other))

    def __ge__(self, other):
        return SparseArray(self.value >= self._get_value(other))

    def copy(self):
        return SparseArray(self.value.copy())

    def astype(self, dtype, copy=True):
        dtype = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
        if copy:
            return SparseArray(self.value.astype(dtype))
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
        return SparseArray(self.value.transpose())


def implements(np_function):
    "Register an __array_function__ implementation for SparseArray objects."

    def decorator(func):
        SparseArray._HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


def _concatenate(L, axis=0):
    if len(L) == 1:
        return L[0]
    if axis == 0:
        return SparseArray(scipy.sparse.vstack(tuple([sa.value for sa in L])))
    elif axis == 1:
        return SparseArray(scipy.sparse.hstack(tuple([sa.value for sa in L])))
    else:
        msg = (
            "Can only concatenate sparse matrices for axis in " "{0, 1}.  Got %s" % axis
        )
        raise ValueError(msg)


# register concatenate if Dask is installed
try:
    from dask.array.core import concatenate_lookup

    concatenate_lookup.register(SparseArray, _concatenate)
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
