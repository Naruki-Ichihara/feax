---
sidebar_label: batched
title: feax.solvers.batched
---

Batched preconditioned conjugate gradients on a shared sparsity pattern.

The kernel behind :func:`feax.vmap` for *design* batches (one stiffness matrix
per case). The ``N`` matrices share one CSR pattern (``indptr``, ``indices``)
and differ only in their values ``(N, nnz)``, so every CG iteration is one
batched sparse matrix-vector product plus one application of the batched
preconditioner ``M`` (e.g. one back-substitution with a shared factorization,
see :func:``1). Each case has its own
convergence flag and iteration count; the loop runs until every case has
converged or ``maxiter`` is reached.

## BatchedSolveInfo Objects

```python
class BatchedSolveInfo(NamedTuple)
```

Per-case diagnostics of :func:`batched_pcg`.

#### iterations

(N,) int32: CG iterations taken by each case

#### converged

(N,) bool: reached the tolerance within maxiter

#### residual

(N,) float: final relative residual ||r|| / ||b||

#### batched\_spmv

```python
def batched_spmv(values, indptr, indices)
```

``spmv(V, X) -&gt; (N, n)`` for CSR values ``V`` of shape ``(N, nnz)`` (or
``(nnz,)``) on the pattern ``(indptr, indices)`` and vectors ``X`` ``(N, n)``.

#### batched\_pcg

```python
def batched_pcg(values,
                indptr,
                indices,
                b,
                M: Optional[Callable] = None,
                *,
                tol: float = 1e-6,
                atol: float = 0.0,
                maxiter: int = 1000,
                x0=None)
```

Solve ``K_i x_i = b_i`` for ``i = 1..N`` with preconditioned CG.

Parameters
----------
- **values** (*array (N, nnz) or (nnz,)*): CSR values of the ``N`` symmetric positive definite matrices on the shared pattern (a single ``(nnz,)`` matrix is shared by every case).
- **b** (*array (N, n)*): Right-hand sides.
- **M** (*callable, optional*): Batched preconditioner ``M(R) -&gt; Z`` on ``(N, n)`` arrays (an approximate inverse applied to every case at once). ``None`` is plain CG.
- **maxiter** (*int*): Iteration cap for the batch (a converged case stops updating).
- **x0** (*array (N, n), optional*): Initial iterates (zeros by default).


Returns
-------
- **X** (*array (N, n)*)

