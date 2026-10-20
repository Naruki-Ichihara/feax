"""Batched preconditioned conjugate gradients on a shared sparsity pattern.

The kernel behind :func:`feax.vmap` for *design* batches (one stiffness matrix
per case). The ``N`` matrices share one CSR pattern (``indptr``, ``indices``)
and differ only in their values ``(N, nnz)``, so every CG iteration is one
batched sparse matrix-vector product plus one application of the batched
preconditioner ``M`` (e.g. one back-substitution with a shared factorization,
see :func:`feax.solvers.preconditioners.shared_factor`). Each case has its own
convergence flag and iteration count; the loop runs until every case has
converged or ``maxiter`` is reached.
"""

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp


class BatchedSolveInfo(NamedTuple):
    """Per-case diagnostics of :func:`batched_pcg`."""
    iterations: jnp.ndarray     # (N,) int32: CG iterations taken by each case
    converged: jnp.ndarray      # (N,) bool: reached the tolerance within maxiter
    residual: jnp.ndarray       # (N,) float: final relative residual ||r|| / ||b||


def batched_spmv(values, indptr, indices):
    """``spmv(V, X) -> (N, n)`` for CSR values ``V`` of shape ``(N, nnz)`` (or
    ``(nnz,)``) on the pattern ``(indptr, indices)`` and vectors ``X`` ``(N, n)``."""
    n = indptr.shape[0] - 1
    nnz = indices.shape[0]
    rows = jnp.repeat(jnp.arange(n), jnp.diff(indptr), total_repeat_length=nnz)

    def one(v, x):
        return jax.ops.segment_sum(v * x[indices], rows, num_segments=n)

    in_axes = (0 if jnp.ndim(values) == 2 else None, 0)
    return jax.vmap(one, in_axes=in_axes)


def batched_pcg(values, indptr, indices, b, M: Optional[Callable] = None, *,
                tol: float = 1e-6, atol: float = 0.0, maxiter: int = 1000, x0=None):
    """Solve ``K_i x_i = b_i`` for ``i = 1..N`` with preconditioned CG.

    Parameters
    ----------
    values : array (N, nnz) or (nnz,)
        CSR values of the ``N`` symmetric positive definite matrices on the
        shared pattern (a single ``(nnz,)`` matrix is shared by every case).
    indptr, indices : arrays
        The shared CSR pattern.
    b : array (N, n)
        Right-hand sides.
    M : callable, optional
        Batched preconditioner ``M(R) -> Z`` on ``(N, n)`` arrays (an
        approximate inverse applied to every case at once). ``None`` is plain CG.
    tol, atol : float
        Case ``i`` has converged when ``||r_i|| <= max(tol * ||b_i||, atol)``.
    maxiter : int
        Iteration cap for the batch (a converged case stops updating).
    x0 : array (N, n), optional
        Initial iterates (zeros by default).

    Returns
    -------
    X : array (N, n)
    info : BatchedSolveInfo
    """
    b = jnp.asarray(b)
    if b.ndim != 2:
        raise ValueError(f"b must be (N, n), got shape {b.shape}.")
    N, n = b.shape
    spmv = batched_spmv(values, indptr, indices)
    apply_M = (lambda R: R) if M is None else M
    bnorm = jnp.linalg.norm(b, axis=1)
    threshold = jnp.maximum(tol * bnorm, atol)

    X = jnp.zeros_like(b) if x0 is None else jnp.asarray(x0)
    R = b - spmv(values, X) if x0 is not None else b
    Z = apply_M(R)
    rz = jnp.sum(R * Z, axis=1)
    done0 = jnp.linalg.norm(R, axis=1) <= threshold
    state = (X, R, Z, rz, done0, jnp.zeros(N, jnp.int32), jnp.array(0, jnp.int32))

    def cond(s):
        return (s[6] < maxiter) & ~jnp.all(s[4])

    def body(s):
        X, R, P, rz, done, its, k = s
        AP = spmv(values, P)
        pAp = jnp.sum(P * AP, axis=1)
        alpha = jnp.where(done, 0.0, rz / jnp.where(pAp == 0.0, 1.0, pAp))
        X = X + alpha[:, None] * P
        R = R - alpha[:, None] * AP
        now = jnp.linalg.norm(R, axis=1) <= threshold
        its = jnp.where(done, its, k + 1)
        Z = apply_M(R)
        rz_new = jnp.sum(R * Z, axis=1)
        fin = done | now
        beta = jnp.where(fin, 0.0, rz_new / jnp.where(rz == 0.0, 1.0, rz))
        P = jnp.where(fin[:, None], P, Z + beta[:, None] * P)
        return X, R, P, jnp.where(fin, rz, rz_new), fin, its, k + 1

    X, R, _, _, done, its, _ = jax.lax.while_loop(cond, body, state)
    rel = jnp.linalg.norm(R, axis=1) / jnp.where(bnorm == 0.0, 1.0, bnorm)
    return X, BatchedSolveInfo(iterations=its, converged=done, residual=rel)


__all__ = ["BatchedSolveInfo", "batched_pcg", "batched_spmv"]
