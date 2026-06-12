"""Lightweight CSR sparse-matrix carrier for the assembled-operator path.

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
the eventual "assembled operator" abstraction.
"""

import jax
import jax.numpy as np
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class CSRMatrix:
    """Compressed Sparse Row matrix with a fixed (precomputed) structure.

    Parameters
    ----------
    data : ndarray (nse,)
        Nonzero values, in CSR order (row-major, within-row sorted by column).
    indptr : ndarray (num_rows + 1,)
        Row pointer; row ``r`` owns slots ``indptr[r] : indptr[r+1]``.
    indices : ndarray (nse,)
        Column index of each slot.
    shape : tuple(int, int)
        ``(num_rows, num_cols)`` — static.
    """

    def __init__(self, data, indptr, indices, shape):
        self.data = data
        self.indptr = indptr
        self.indices = indices
        self.shape = shape

    # --- pytree -----------------------------------------------------------
    def tree_flatten(self):
        # data/indptr/indices are dynamic (indptr/indices are constant in
        # practice but carrying them as children keeps trace/vmap trivial);
        # shape is static aux.
        return (self.data, self.indptr, self.indices), self.shape

    @classmethod
    def tree_unflatten(cls, aux, children):
        data, indptr, indices = children
        return cls(data, indptr, indices, aux)

    # --- basic properties -------------------------------------------------
    @property
    def nse(self) -> int:
        return self.indices.shape[0]

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self) -> int:
        return 2

    def with_data(self, data) -> 'CSRMatrix':
        """Return a CSRMatrix sharing this structure but with new values."""
        return CSRMatrix(data, self.indptr, self.indices, self.shape)

    def _row_of_slot(self):
        """Row index of every slot (length nse). Static structure → cheap."""
        n = self.shape[0]
        return np.repeat(np.arange(n), np.diff(self.indptr),
                         total_repeat_length=self.indices.shape[0])

    # --- matrix-vector products ------------------------------------------
    def __matmul__(self, v):
        """``A @ v``.

        Row segments are contiguous and ascending, so the sum is deterministic
        (``indices_are_sorted=True``) — no atomics.
        """
        n = self.shape[0]
        rows = self._row_of_slot()
        return jax.ops.segment_sum(self.data * v[self.indices], rows,
                                   num_segments=n, indices_are_sorted=True)

    def rmatvec(self, w):
        """``A.T @ w`` without materializing the transpose.

        Segments are the (unsorted) column indices, so this is a scatter-add.
        """
        m = self.shape[1]
        rows = self._row_of_slot()
        return jax.ops.segment_sum(self.data * w[rows], self.indices,
                                   num_segments=m, indices_are_sorted=False)

    def transpose(self) -> 'CSRTranspose':
        """Lazy transpose: a matvec-only view whose ``@`` is ``self.rmatvec``.

        Suitable for iterative solvers / convergence checks. Direct solvers that
        need an explicit transposed CSR structure should instead solve with a
        transpose flag on the assembled ``self``.
        """
        return CSRTranspose(self)

    @property
    def T(self) -> 'CSRTranspose':
        return self.transpose()

    def diagonal(self):
        """Main diagonal as a dense vector (length ``num_rows``)."""
        n = self.shape[0]
        rows = self._row_of_slot()
        is_diag = rows == self.indices
        return jax.ops.segment_sum(np.where(is_diag, self.data, 0.0), rows,
                                   num_segments=n, indices_are_sorted=True)

    # --- dense (tests / small problems) ----------------------------------
    def todense(self):
        n, m = self.shape
        rows = self._row_of_slot()
        return np.zeros((n, m), self.data.dtype).at[rows, self.indices].add(self.data)


def transpose_with_maps(J: CSRMatrix, T_perm, T_indptr, T_indices) -> CSRMatrix:
    """Build the explicit CSR transpose ``J.T`` from precomputed static maps.

    ``T_perm`` / ``T_indptr`` / ``T_indices`` come from
    :meth:`Problem._build_csr_assembly_structure`. The transposed values are a
    pure gather ``J.data[T_perm]`` (no sort), and the result is a real
    :class:`CSRMatrix` — so a direct backend can factorize ``J.T`` for the
    adjoint solve. When the sparsity pattern is structurally symmetric the
    transpose shares ``J``'s structure (``T_indptr == indptr`` etc.), so a
    stateful backend (e.g. cuDSS) can reuse the same symbolic factorization.
    """
    return CSRMatrix(J.data[T_perm], T_indptr, T_indices, (J.shape[1], J.shape[0]))


@register_pytree_node_class
class CSRTranspose:
    """Matvec-only lazy transpose of a :class:`CSRMatrix`.

    Only ``@`` (matrix-vector product) is defined; it delegates to the parent's
    ``rmatvec``. This is enough for iterative Krylov solves and residual checks,
    which never need the explicit transposed structure.
    """

    def __init__(self, parent: CSRMatrix):
        self.parent = parent
        self.shape = (parent.shape[1], parent.shape[0])

    def tree_flatten(self):
        return (self.parent,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0])

    @property
    def dtype(self):
        return self.parent.dtype

    def __matmul__(self, v):
        return self.parent.rmatvec(v)

    def todense(self):
        return self.parent.todense().T
