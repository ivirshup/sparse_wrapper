# sparse_wrapper

Array-like wrapper for sparse matrices with a compressed axis.

```python
from sparse_wrapper import CSR
from sparse import COO
from scipy import sparse

a = sparse.random(5, 10, format="csr", density=.4)

# Numpy semantics

CSR(a).mean(axis=1)
# array([0.12135885, 0.12187619, 0.37112213, 0.21135995, 0.31450332])
CSR(a) * CSR(a)
# <CSR: shape=(5, 10), dtype=float64>
CSR(a) + COO(a)
# <CSR: shape=(5, 10), dtype=float64>

# Sparse operations

CSR(a) @ CSR(a).T
# <CSR: shape=(5, 5), dtype=float64>
```

Very WIP.