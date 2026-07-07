"""Automatic Sparse Differentiation (ASD) utilities, built on `asdex`.

feax's standard volume assembly does NOT go through this module: the
element-dense ``jacfwd`` + slot-map scatter is already optimal when the
sparsity is the mesh connectivity (a global coloring would need ``max row nnz``
colors — more kernel evaluations than the element dimension). This module
covers the places where a sparse operator is needed but its pattern is *not*
the plain connectivity:

* **extra residual terms** — arbitrary user coupling with unknown sparsity:
  :func:`detect_jacobian_pattern` / :func:`sparse_jacobian_fn` (used by the
  assembled ``extra_residual_fn`` solver path, enabling direct solvers).
* **reduced periodic operators** ``PᵀJP`` — pattern by boolean triple product
  (:func:`reduced_operator_pattern`), values by colored probes of the
  matrix-free action (:func:`operator_assembler`); enables direct/AMG solves
  for periodic problems.
* **design-space Hessians** ``d²J/dρ²`` — :func:`sparse_hessian_fn` with
  symmetric (star) coloring + HVPs, for second-order design optimization.
* **verification** — :func:`verify_jacobian_pattern` checks feax's hand-built
  CSR pattern against detection on the actual residual.

All factories return functions with a FIXED sparsity structure (jit-safe) that
produce :class:`feax.csr.CSRMatrix` — the operator type the feax solver stack
consumes.
"""

from typing import Callable, Optional, Tuple

import numpy as onp
import scipy.sparse as sp

import jax
import jax.numpy as np
import asdex

from .csr import CSRMatrix

__all__ = [
    "detect_jacobian_pattern",
    "detect_hessian_pattern",
    "connectivity_pattern",
    "reduced_operator_pattern",
    "sparse_jacobian_fn",
    "sparse_hessian_fn",
    "operator_assembler",
    "merge_csr_patterns",
    "verify_jacobian_pattern",
]


# ---------------------------------------------------------------------------
# Pattern coercion / algebra
# ---------------------------------------------------------------------------

def _to_scipy_pattern(pattern) -> sp.csr_matrix:
    """Coerce a pattern (scipy sparse / BCOO / dense / asdex SparsityPattern /
    CSRMatrix) to a canonical boolean scipy CSR with sorted indices."""
    from jax.experimental.sparse import BCOO
    if isinstance(pattern, asdex.SparsityPattern):
        A = sp.coo_matrix(
            (onp.ones(len(pattern.rows), bool),
             (onp.asarray(pattern.rows), onp.asarray(pattern.cols))),
            shape=pattern.shape)
    elif isinstance(pattern, CSRMatrix):
        A = sp.csr_matrix(
            (onp.ones(pattern.nse, bool), onp.asarray(pattern.indices),
             onp.asarray(pattern.indptr)), shape=pattern.shape)
    elif sp.issparse(pattern):
        A = pattern.astype(bool)
    elif isinstance(pattern, BCOO):
        idx = onp.asarray(pattern.indices)
        A = sp.coo_matrix((onp.ones(idx.shape[0], bool), (idx[:, 0], idx[:, 1])),
                          shape=pattern.shape)
    else:
        A = sp.csr_matrix(onp.asarray(pattern) != 0)
    A = A.tocsr()
    A.sum_duplicates()
    A.sort_indices()
    return A


def _sparsity_from_scipy(A: sp.csr_matrix) -> "asdex.SparsityPattern":
    coo = A.tocoo()
    return asdex.SparsityPattern(coo.row.astype(onp.int32),
                                 coo.col.astype(onp.int32),
                                 tuple(int(s) for s in coo.shape))


def _slot_keys(pat: sp.csr_matrix) -> onp.ndarray:
    """Sorted ``row*ncols + col`` key of every CSR slot (indices sorted)."""
    rows = onp.repeat(onp.arange(pat.shape[0], dtype=onp.int64),
                      onp.diff(pat.indptr))
    return rows * pat.shape[1] + pat.indices.astype(onp.int64)


def _csr_slots(pat: sp.csr_matrix, rows, cols) -> onp.ndarray:
    """CSR data-slot index of each ``(row, col)`` pair (all must be in ``pat``)."""
    keys = _slot_keys(pat)
    q = onp.asarray(rows, onp.int64) * pat.shape[1] + onp.asarray(cols, onp.int64)
    slots = onp.searchsorted(keys, q)
    assert q.size == 0 or (keys[slots] == q).all(), \
        "(row, col) entries not contained in the CSR pattern"
    return slots.astype(onp.int32)


def detect_jacobian_pattern(f: Callable, x_sample) -> sp.csr_matrix:
    """Global Jacobian sparsity of ``f`` (valid for all inputs) as boolean CSR.

    Uses asdex's jaxpr abstract interpretation — no derivative evaluation.
    """
    return _to_scipy_pattern(asdex.jacobian_sparsity(f, x_sample))


def detect_hessian_pattern(f: Callable, x_sample) -> sp.csr_matrix:
    """Global Hessian sparsity of scalar ``f`` as boolean CSR."""
    return _to_scipy_pattern(asdex.hessian_sparsity(f, x_sample))


def connectivity_pattern(problem) -> sp.csr_matrix:
    """feax's assembled CSR pattern (mesh connectivity) as boolean CSR.

    Requires ``MatrixView.FULL`` (UPPER/LOWER store a triangular view whose
    pattern is not the full operator's).
    """
    from .problem import MatrixView
    if problem.matrix_view is not MatrixView.FULL:
        raise ValueError("connectivity_pattern requires MatrixView.FULL")
    n = problem.num_total_dofs_all_vars
    indptr = onp.asarray(problem.csr_indptr)
    indices = onp.asarray(problem.csr_indices)
    A = sp.csr_matrix((onp.ones(indices.size, bool), indices, indptr), shape=(n, n))
    A.sort_indices()
    return A


def reduced_operator_pattern(P, K_pattern) -> sp.csr_matrix:
    """Sparsity of the Galerkin product ``PᵀKP`` by boolean triple product.

    ``P`` is the (n_full, n_reduced) prolongation (BCOO/scipy/dense);
    ``K_pattern`` any pattern accepted by this module (e.g.
    :func:`connectivity_pattern`). Exact for boolean algebra — a superset of the
    numerical pattern, which is what coloring/decompression need.
    """
    Pp = _to_scipy_pattern(P).astype(onp.float64)
    Kp = _to_scipy_pattern(K_pattern).astype(onp.float64)
    R = (Pp.T @ Kp @ Pp).tocsr()
    return _to_scipy_pattern(R)


def merge_csr_patterns(pattern_a, pattern_b) -> dict:
    """Union of two CSR patterns plus the maps to assemble/transpose on it.

    Returns a dict with the merged ``indptr``/``indices`` (int32 JAX arrays),
    ``nnz``, ``shape``, data-slot maps ``slots_a``/``slots_b`` (aligned with each
    input pattern's CSR order), and transpose maps ``T_perm``/``T_indptr``/
    ``T_indices`` for :func:`feax.csr.transpose_with_maps`.
    """
    A = _to_scipy_pattern(pattern_a)
    B = _to_scipy_pattern(pattern_b)
    assert A.shape == B.shape, "patterns must share a shape"
    M = (A + B).tocsr()
    M.sum_duplicates()
    M.sort_indices()

    rows_a = onp.repeat(onp.arange(A.shape[0], dtype=onp.int64), onp.diff(A.indptr))
    rows_b = onp.repeat(onp.arange(B.shape[0], dtype=onp.int64), onp.diff(B.indptr))
    slots_a = _csr_slots(M, rows_a, A.indices)
    slots_b = _csr_slots(M, rows_b, B.indices)

    # transpose maps: CSR of Mᵀ whose data carries the source slot index
    nnz = int(M.nnz)
    S = sp.csr_matrix((onp.arange(nnz, dtype=onp.int64), M.indices, M.indptr),
                      shape=M.shape)
    ST = S.T.tocsr()
    ST.sort_indices()

    return dict(
        indptr=np.asarray(M.indptr.astype(onp.int32)),
        indices=np.asarray(M.indices.astype(onp.int32)),
        nnz=nnz,
        shape=tuple(int(s) for s in M.shape),
        slots_a=np.asarray(slots_a),
        slots_b=np.asarray(slots_b),
        T_perm=np.asarray(ST.data.astype(onp.int32)),
        T_indptr=np.asarray(ST.indptr.astype(onp.int32)),
        T_indices=np.asarray(ST.indices.astype(onp.int32)),
    )


# ---------------------------------------------------------------------------
# Sparse function factories (fixed structure -> CSRMatrix, jit-safe)
# ---------------------------------------------------------------------------

def _csr_producer(coloring, compressed_fn) -> Tuple[Callable, sp.csr_matrix]:
    """Wrap an asdex compressed-AD function into ``x -> CSRMatrix``."""
    pat = _to_scipy_pattern(coloring.sparsity)
    slots = np.asarray(_csr_slots(pat, coloring.sparsity.rows, coloring.sparsity.cols))
    indptr = np.asarray(pat.indptr.astype(onp.int32))
    indices = np.asarray(pat.indices.astype(onp.int32))
    nnz, shape = int(pat.nnz), tuple(int(s) for s in pat.shape)

    def fn(x):
        if nnz == 0:
            return CSRMatrix(np.zeros(0), indptr, indices, shape)
        B = compressed_fn(x)
        data_pat = asdex.decompress_data(B, coloring)
        data = np.zeros(nnz, data_pat.dtype).at[slots].set(data_pat)
        return CSRMatrix(data, indptr, indices, shape)

    return fn, pat


def sparse_jacobian_fn(f: Callable, x_sample=None, pattern=None, *,
                       mode=None) -> Tuple[Callable, sp.csr_matrix]:
    """Sparse Jacobian of ``f`` with a fixed structure.

    Detects the sparsity from ``f``/``x_sample`` (or uses the given ``pattern``
    superset), colors it, and returns ``(jac_fn, pattern_csr)`` where
    ``jac_fn(x) -> CSRMatrix`` runs one JVP/VJP per color (jit-safe, fixed
    structure). Cost per call: ``num_colors`` AD passes of ``f``.
    """
    if pattern is None:
        if x_sample is None:
            raise ValueError("sparse_jacobian_fn needs x_sample (for detection) "
                             "or an explicit pattern")
        sparsity = asdex.jacobian_sparsity(f, x_sample)
    else:
        sparsity = _sparsity_from_scipy(_to_scipy_pattern(pattern))
    coloring = asdex.jacobian_coloring_from_sparsity(sparsity, mode=mode)
    compressed_fn = asdex.compressed_jacobian_from_coloring(f, coloring)
    return _csr_producer(coloring, compressed_fn)


def sparse_hessian_fn(f: Callable, x_sample=None, pattern=None, *,
                      mode=None, symmetric=True) -> Tuple[Callable, sp.csr_matrix]:
    """Sparse Hessian of scalar ``f`` with a fixed structure.

    Star (symmetric) coloring + one HVP per color. Returns
    ``(hess_fn, pattern_csr)`` with ``hess_fn(x) -> CSRMatrix``. Intended e.g.
    for design-space Hessians ``d²J/dρ²`` in second-order topology
    optimization, where the pattern is the filter-stencil overlap.
    """
    if pattern is None:
        if x_sample is None:
            raise ValueError("sparse_hessian_fn needs x_sample (for detection) "
                             "or an explicit pattern")
        sparsity = asdex.hessian_sparsity(f, x_sample)
    else:
        sparsity = _sparsity_from_scipy(_to_scipy_pattern(pattern))
    coloring = asdex.hessian_coloring_from_sparsity(sparsity, mode=mode,
                                                    symmetric=symmetric)
    compressed_fn = asdex.compressed_hessian_from_coloring(f, coloring)
    return _csr_producer(coloring, compressed_fn)


def operator_assembler(pattern, *, mode="fwd") -> Callable:
    """Assembler for LINEAR operators known only through their matvec.

    Colors ``pattern`` once; the returned ``assemble(matvec) -> CSRMatrix``
    materializes any linear operator with that sparsity using ``num_colors``
    matvec probes (colored JVPs at 0). Used for the reduced periodic operator
    ``PᵀJP``, whose action exists matrix-free but whose assembled form is
    needed for direct/AMG solves.
    """
    pat = _to_scipy_pattern(pattern)
    sparsity = _sparsity_from_scipy(pat)
    coloring = asdex.jacobian_coloring_from_sparsity(sparsity, mode=mode)
    x0 = np.zeros(pat.shape[1])

    def assemble(matvec: Callable) -> CSRMatrix:
        compressed_fn = asdex.compressed_jacobian_from_coloring(matvec, coloring)
        fn, _ = _csr_producer(coloring, compressed_fn)
        return fn(x0)

    return assemble


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_jacobian_pattern(problem, traced_params, ts=None) -> dict:
    """Check feax's hand-built CSR pattern against detection on the residual.

    Detects the true Jacobian sparsity of the assembled (bulk, no-BC) residual
    via asdex and compares with :func:`connectivity_pattern`. Soundness requires
    ``detected ⊆ connectivity``; ``coverage`` reports how much of the
    connectivity pattern is actually used (element blocks may hold structural
    zeros). Returns ``dict(ok, num_detected, num_pattern, num_missing,
    coverage)``.
    """
    import jax.flatten_util
    from .assembler import get_res

    n = problem.num_total_dofs_all_vars

    def residual(u):
        sol_list = problem.unflatten_fn_sol_list(u)
        res_list = get_res(problem, sol_list, traced_params, ts)
        return jax.flatten_util.ravel_pytree(res_list)[0]

    detected = detect_jacobian_pattern(residual, np.zeros(n))
    conn = connectivity_pattern(problem)
    missing = (detected.astype(onp.int8) - conn.astype(onp.int8)) > 0
    return dict(
        ok=int(missing.nnz) == 0,
        num_detected=int(detected.nnz),
        num_pattern=int(conn.nnz),
        num_missing=int(missing.nnz),
        coverage=float(detected.nnz / max(conn.nnz, 1)),
    )
