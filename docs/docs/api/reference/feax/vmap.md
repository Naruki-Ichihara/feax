---
sidebar_label: vmap
title: feax.vmap
---

``feax.vmap``: batch a solver over its parameters, solving the batch well.

``jax.vmap(solver)`` already batches a feax solver, but it does not know what
the batch means: when the batched parameters change only the right-hand side
(load cases), a direct solver factorizes once and solves many; when they
change the stiffness matrix (designs), it factorizes every case — ``N``
factorizations held at once, which runs out of memory quickly.

:func:`vmap` looks at which parameters are batched, decides which of the two
it is, and picks the solve accordingly:

* **rhs batch** (the operator does not depend on the batched parameters):
  plain ``jax.vmap`` of the solver — with a direct solver that is one
  factorization and a multi-right-hand-side solve.
* **lhs batch** (one operator per case): the ``N`` operators are assembled on
  their shared sparsity pattern and solved together by
  :func:``1, preconditioned by a batch-level
  preconditioner built from all of them — by default one shared factorization
  of their mean (:func:``2), or :func:``3 without
  cuDSS. Each case reports its CG iteration count.

A Newton solver is batched the same way: all cases iterate together in one
traced loop, with one shared factorization of the mean tangent per Newton step
and a per-case line search (a nonlinear batch is always an lhs batch).

The returned callable is differentiable (custom VJP: the adjoint batch is
solved the same way) and composes with ``jax.jit``.

## BatchedSolver Objects

```python
class BatchedSolver()
```

What :func:`vmap` returns. ``kind`` is ``&quot;rhs&quot;`` or ``&quot;lhs&quot;``.

#### operator\_signature

```python
def operator_signature(traced_params,
                       initial_guess=None,
                       traced_structure=None)
```

A signature of every case&#x27;s operator, ``(B, n_dof)``: the log of the
diagonal of its (BC-applied) stiffness matrix at the initial guess.

Independent of how the design is parametrized (density, modulus,
orientation, ...): whatever enters the operator shows up here. Feed it
to :func:`feax.gene.similarity_order` to order the batch so that
``chunk`` groups similar operators for the shared preconditioner.

#### vmap

```python
def vmap(solver: Solver,
         in_axes: Any = 0,
         *,
         batch: str = "auto",
         options: Optional[Any] = None,
         preconditioner: Optional[Callable] = None,
         chunk: Optional[int] = None,
         return_info: bool = False) -> BatchedSolver
```

Batch a feax solver over ``traced_params``, solving the batch well.

Parameters
----------
- **solver** (*Solver*): A solver from :func:`feax.create_solver`: linear (``linear=True``) or the standard traced Newton solver (no ``extra_residual_fn``, no host-side AMG rebuild). A Newton batch is always an lhs batch (the tangent depends on the state): all cases iterate together, with one shared factorization of the mean tangent per Newton step and a per-case line search; ``info`` is then a :class:`BatchedNewtonInfo` (Newton iterations, ``converged``, ``line_search_failed``, CG iterations) and every case&#x27;s final iterate is returned, converged or not.
- **in_axes** (*int, None or a TracedParams-shaped prefix*): Which leaves of ``traced_params`` carry the batch axis (as for ``jax.vmap``; the initial guess and the TracedStructure are shared).
- **batch** (*{`&quot;auto&quot;, &quot;rhs&quot;, &quot;lhs&quot;`}*): What the batch changes: ``&quot;rhs&quot;`` only the right-hand side (same operator for every case), ``&quot;lhs&quot;`` the operator too. ``&quot;auto&quot;`` detects it from the solver&#x27;s sample parameters.
- **options** (*KrylovSolverOptions or DirectSolverOptions, optional*): How an *lhs* batch is solved. Default: batched CG (``tol``, ``atol``, ``maxiter`` from the solver&#x27;s own Krylov options, or cg 1e-8 / 1e-10 / 1000). ``DirectSolverOptions`` solves each case with its own factorization (``N`` at once unless ``chunk`` is set — a warning).
- **preconditioner** (*callable, optional*): The batch-level preconditioner of an lhs batch, ``preconditioner(op) -&gt; M`` where ``op.values`` is ``(N, nnz)`` (every case&#x27;s operator on the shared pattern) and ``M`` acts on ``(N, n)`` arrays. Default: :func:``6 (one factorization of the mean operator) with cuDSS, else :func:``7. Any callable of the same contract (e.g. a per-group factorization) can be used. A solver built with ``preconditioner=fe.jacobi()`` / ``fe.shared_factor()`` / an ``AMGPreconditioner`` keeps it; any other solver preconditioner is ignored here (it may not accept a batched operator).
- **chunk** (*int, optional*): Process the batch in groups of this size (``lax.map`` over groups): a preconditioner per group and a memory bound of ``chunk`` operators.
- **return_info** (*bool*): Also return a :class:``8 with the per-case CG iteration counts (``rhs``: zeros).


Returns
-------
BatchedSolver
    ``batched(traced_params, initial_guess=None, traced_structure=None)
    -&gt; solution`` (``(solution, info)`` with ``return_info``), with the
    batch axis in front of the solution; differentiable w.r.t.
    ``traced_params`` and jit-able.

