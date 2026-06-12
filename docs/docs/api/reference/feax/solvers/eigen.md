---
sidebar_label: eigen
title: feax.solvers.eigen
---

Linear-buckling eigensolvers for feax, in JAX.

Solves the linear-buckling pencil ``(K + λ K_g) φ = 0`` for the lowest positive
buckling factors ``λ`` and their mode shapes, with three forward backends:

* ``solver=&quot;sparse&quot;`` (:func:`_sparse_buckling`, **default**) — matrix-free shift-invert
  ``T = -K⁻¹ K_g`` solved by SciPy ARPACK (:func:`scipy.sparse.linalg.eigs`) with one
  sparse factorization for ``K⁻¹``. ``O(nf · ncv)`` memory, never a dense ``N×N``. Runs
  purely host-side (no nested ``jax.pure_callback``), so it is leak-free under repeated
  calls (e.g. inside topology optimization).
* ``solver=&quot;dense&quot;`` (:func:``0) — Cholesky reduction of the generalized
  symmetric-definite pencil to a standard ``jnp.linalg.eigh`` on the free DoFs.
  Exact and robust, but ``O(nf²)`` memory; for small/medium reduced systems.
* ``solver=&quot;matfree&quot;`` (:func:``7, opt-in) — the same shift-invert via
  matfree&#x27;s JAX Arnoldi. Equivalent results, but applies ``K⁻¹`` through a nested
  ``jax.pure_callback`` that accumulates in the XLA runtime across calls (memory leak in
  long loops); prefer ``&quot;sparse&quot;`` unless you specifically need the JAX path.

The differentiable driver :func:``4 wraps any backend with
an analytical eigenvalue sensitivity, so ``λ`` is ``jax.grad``-able w.r.t. the assembled
``K`` / ``K_g`` (and ``bf`` itself stays jittable — the eigensolve is an opaque host
callback regardless of backend).

## BucklingConvergenceError Objects

```python
class BucklingConvergenceError(RuntimeError)
```

Raised when the matrix-free buckling eigensolve fails to converge.

Lets a caller (e.g. a topology-optimization loop) detect a bad solve early and
react — instead of silently propagating garbage eigenvalues / sensitivities.

#### generalized\_eigh

```python
def generalized_eigh(A: jnp.ndarray,
                     B: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]
```

Solve the generalized symmetric eigenproblem ``A x = λ B x`` (``B ≻ 0``).

Parameters
----------
- **A** (*(n, n) array*): Symmetric matrix (symmetrised internally for round-off).
- **B** (*(n, n) array*): Symmetric positive-definite matrix.


Returns
-------
- **w** (*(n,) array*): Eigenvalues ``λ`` in ascending order.
- **X** (*(n, n) array*): Eigenvectors as columns, B-orthonormal (``Xᵀ B X = I``).


Notes
-----
Implemented via the Cholesky reduction ``C = L⁻¹ A L⁻ᵀ`` with ``B = L Lᵀ``; the
eigenvalues of ``C`` equal those of the pencil and ``x = L⁻ᵀ u``.

#### create\_linear\_buckling\_solver

```python
def create_linear_buckling_solver(free_dofs,
                                  num_modes: int = 3,
                                  solver: str = "sparse",
                                  **solver_kw)
```

Return a ``jax.grad``-able function ``bf(K, Kg) -&gt; λ`` for buckling factors.

The forward eigensolve (sparse matrix-free :func:`_matfree_buckling` or dense
:func:`_dense_buckling`, selected by ``solver``) runs through a
:func:`jax.pure_callback`; differentiability is supplied by a
:func:`jax.custom_vjp` using the **analytical eigenvalue sensitivity**

.. code-block:: text

    (K + λ K_g) φ = 0,   φᵀ K φ = 1   ⇒   dλ = λ φᵀ dK φ + λ² φᵀ dK_g φ.

Because only eigenvalues (not eigenvectors) are differentiated, this is robust to
the degenerate buckling-mode pairs that make a naive ``eigh`` VJP blow up. Gradients
flow to ``K.data`` and ``K_g.data`` (hence to any upstream design parameters that
assemble them), e.g. for buckling-load optimization.

Parameters
----------
- **free_dofs** (*int array*): Unconstrained DoFs (captured; the returned function differentiates only K, Kg).
- **num_modes** (*int*): Number of buckling factors returned.
- **solver** (*{`&quot;sparse&quot;, &quot;dense&quot;, &quot;matfree&quot;`}*): Forward eigensolver. ``&quot;sparse&quot;`` (default) forwards ``solver_kw`` to :func:``0 (SciPy ARPACK matrix-free shift-invert — low memory, leak-free; e.g. ``num_matvecs=40``); ``&quot;dense&quot;`` uses :func:`_dense_buckling` (exact, ``O(nf²)`` memory); ``&quot;matfree&quot;`` uses :func:`_matfree_buckling` (JAX matfree Arnoldi — equivalent, opt-in; leaks memory in long loops).


Returns
-------
- **bf** (*Callable[[BCOO, BCOO], Tuple[jax.Array, jax.Array]]*): ``bf(K, Kg) -&gt; (lambdas, modes)``. ``lambdas`` is the ``(num_modes,)`` ascending positive buckling factors, **differentiable** w.r.t. ``K`` and ``Kg``. ``modes`` is the ``(num_modes, N)`` full-DoF mode shapes for visualization and carries **no gradient** (``stop_gradient``; eigenvector derivatives are unstable at the degenerate buckling pairs).


Notes
-----
``bf`` is jittable (``jax.jit(bf)`` / ``jax.jit(jax.grad(...))``): the forward
eigensolve runs host-side in a :func:`jax.pure_callback` and only ``K.data`` /
``K_g.data`` are traced. The (constant) sparsity ``indices`` are captured on the
first **eager** call, so call ``bf(K, Kg)`` once outside ``jit`` before jitting it
(a clear error is raised otherwise).

