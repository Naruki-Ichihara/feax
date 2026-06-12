---
sidebar_label: csr
title: feax.csr
---

Lightweight CSR sparse-matrix carrier for the assembled-operator path.

``CSRMatrix`` is the assembled representation produced by the CSR-direct
assembly path (:func:`feax.assembler._get_J_csr` + the CSR-space BC application).
It exists so the Jacobian can flow through the solver stack as a deduplicated
CSR triple ``(data, indptr, indices)`` — exactly what direct backends (cuDSS,
cholmod, umfpack, spsolve) consume — without ever materializing a JAX ``BCOO``
or re-running ``sum_duplicates`` per solve.

The class is intentionally minimal: it is a JAX pytree (so it traces/vmaps/jits
cleanly, with ``data``/``indptr``/``indices`` as children and the static
``shape`` as aux) and supports just the operations the solver stack needs —
matrix-vector products (for iterative solvers and residual/convergence checks)
and a dense materialization (for tests and small dense paths). It is the seed of
the eventual &quot;assembled operator&quot; abstraction.

## CSRMatrix Objects

```python
@register_pytree_node_class
class CSRMatrix()
```

Compressed Sparse Row matrix with a fixed (precomputed) structure.

Parameters
----------
- **data** (*ndarray (nse,)*): Nonzero values, in CSR order (row-major, within-row sorted by column).
- **indptr** (*ndarray (num_rows + 1,)*): Row pointer; row ``r`` owns slots ``indptr[r] : indptr[r+1]``.
- **indices** (*ndarray (nse,)*): Column index of each slot.
- **shape** (*tuple(int, int)*): ``(num_rows, num_cols)`` — static.


#### with\_data

```python
def with_data(data) -> 'CSRMatrix'
```

Return a CSRMatrix sharing this structure but with new values.

#### \_\_matmul\_\_

```python
def __matmul__(v)
```

``A @ v``.

Row segments are contiguous and ascending, so the sum is deterministic
(``indices_are_sorted=True``) — no atomics.

#### rmatvec

```python
def rmatvec(w)
```

``A.T @ w`` without materializing the transpose.

Segments are the (unsorted) column indices, so this is a scatter-add.

#### transpose

```python
def transpose() -> 'CSRTranspose'
```

Lazy transpose: a matvec-only view whose ``@`` is ``self.rmatvec``.

Suitable for iterative solvers / convergence checks. Direct solvers that
need an explicit transposed CSR structure should instead solve with a
transpose flag on the assembled ``self``.

#### diagonal

```python
def diagonal()
```

Main diagonal as a dense vector (length ``num_rows``).

#### transpose\_with\_maps

```python
def transpose_with_maps(J: CSRMatrix, T_perm, T_indptr,
                        T_indices) -> CSRMatrix
```

Build the explicit CSR transpose ``J.T`` from precomputed static maps.

``T_perm`` / ``T_indptr`` / ``T_indices`` come from
:meth:`Problem._build_csr_assembly_structure`. The transposed values are a
pure gather ``J.data[T_perm]`` (no sort), and the result is a real
:class:``1 — so a direct backend can factorize ``J.T`` for the
adjoint solve. When the sparsity pattern is structurally symmetric the
transpose shares ``J``&#x27;s structure (``T_indptr == indptr`` etc.), so a
stateful backend (e.g. cuDSS) can reuse the same symbolic factorization.

## CSRTranspose Objects

```python
@register_pytree_node_class
class CSRTranspose()
```

Matvec-only lazy transpose of a :class:`CSRMatrix`.

Only ``@`` (matrix-vector product) is defined; it delegates to the parent&#x27;s
``rmatvec``. This is enough for iterative Krylov solves and residual checks,
which never need the explicit transposed structure.

