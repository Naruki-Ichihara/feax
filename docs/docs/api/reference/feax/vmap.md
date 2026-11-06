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

#### \_\_call\_\_

```python
def __call__(traced_params,
             initial_guess=None,
             traced_structure=None,
             factors=None)
```

Solve the batch. ``factors`` (from :meth:`factorize`) are the
group factorizations to precondition with -- no mean is assembled and
nothing is factorized; without them each call factorizes its own.

#### factorize

```python
def factorize(traced_params, initial_guess=None, traced_structure=None)
```

Factorize the mean operator of every group of ``share`` cases (one
cuDSS entry each, kept alive) and return them as ``factors`` for
``batched(..., factors=factors)``.

The preconditioner only has to be close, not current: reuse the
factors over several calls -- e.g. the iterations of an optimization
-- and factorize again (at the current parameters) when the CG counts
have grown too much. The batch must keep its order and ``share``
while the factors are reused. spineax keeps ``SPINEAX_FACTOR_CACHE``
entries (read at each new entry): set it to at least the number of
groups, or an evicted group is re-factorized from its stored mean on
every use (correct, but no reuse).

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
         share: Optional[int] = None,
         return_info: bool = False) -> BatchedSolver
```

Batch a feax solver over ``traced_params``, solving the batch well.

Parameters
----------
- **solver** (*Solver, homogenization solver or LinearSystemBatch*): The solver of :func:`feax.flat.solver.create_homogenization_solver` is batched over unit-cell designs: the result is the :class:`feax.flat.solver.BatchedHomogenizationSolver` (``C_hom`` of every design, one shared factorization per ``share`` designs; see :func:`feax.flat.solver.create_batched_homogenization_solver`, whose ``tol`` / ``atol`` / ``maxiter`` come from ``options`` here). A :class:``7 (a batched linear problem given directly by its operators and right-hand sides, several per case allowed -- see :func:`feax.flat.solver.create_batched_homogenization_solver`) is solved by the same lhs engine; the call then returns the solutions ``(N, n_rhs, n)``. Otherwise a solver from :func:`feax.flat.solver.create_homogenization_solver`1: linear (``linear=True``) or the standard traced Newton solver (no ``extra_residual_fn``, no host-side AMG rebuild). A Newton batch is always an lhs batch (the tangent depends on the state): all cases iterate together, with one shared factorization of the mean tangent per Newton step and a per-case line search; ``info`` is then a :class:`feax.flat.solver.create_homogenization_solver`8 (Newton iterations, ``converged``, ``line_search_failed``, CG iterations) and every case&#x27;s final iterate is returned, converged or not.
- **in_axes** (*int, None or a TracedParams-shaped prefix*): Which leaves of ``traced_params`` carry the batch axis (as for ``jax.vmap``; the initial guess and the TracedStructure are shared).
- **batch** (*{`&quot;auto&quot;, &quot;rhs&quot;, &quot;lhs&quot;`}*): What the batch changes: ``&quot;rhs&quot;`` only the right-hand side (same operator for every case), ``&quot;lhs&quot;`` the operator too. ``&quot;auto&quot;`` detects it from the solver&#x27;s sample parameters.
- **options** (*KrylovSolverOptions or DirectSolverOptions, optional*): How an *lhs* batch is solved. Default: batched CG (``tol``, ``atol``, ``maxiter`` from the solver&#x27;s own Krylov options, or cg 1e-8 / 1e-10 / 1000). ``DirectSolverOptions`` solves each case with its own factorization (``N`` at once unless ``chunk`` is set — a warning).
- **preconditioner** (*callable, optional*): The batch-level preconditioner of an lhs batch, ``preconditioner(op) -&gt; M`` where ``op.values`` is ``(N, nnz)`` (every case&#x27;s operator on the shared pattern) and ``M`` acts on ``(N, n)`` arrays. Default: :func:``5 (one factorization of the mean operator) with cuDSS, else :func:``6. Any callable of the same contract (e.g. a per-group factorization) can be used. A solver built with ``preconditioner=fe.jacobi()`` / ``fe.shared_factor()`` / an ``AMGPreconditioner`` keeps it; any other solver preconditioner is ignored here (it may not accept a batched operator).
- **chunk** (*int, optional*): Process the batch in groups of this size (``lax.map`` over groups): a memory bound of ``chunk`` operators, and -- unless ``share`` is set -- a preconditioner per group.
- **share** (*int, optional*): Cases per shared preconditioner, independent of ``chunk`` (lhs linear batch with :func:``5). The batch is split into groups of ``share`` cases; each group&#x27;s mean operator is accumulated ``chunk`` operators at a time, factorized once, and that one ``M`` preconditions the CG of all of the group&#x27;s chunks. So ``share=N, chunk=4`` is one factorization for the whole batch with only 4 operators in memory. The groups re-factorize one cuDSS entry in turn (one symbolic analysis, one factorization&#x27;s memory), and the adjoint runs the groups in reverse from the forward&#x27;s last factorization (with one group, the adjoint factorizes nothing). Default: ``chunk`` (a preconditioner per chunk). Typical use: start with the whole batch while the cases are alike and lower ``share`` as the CG counts grow.
- **return_info** (*bool*): Also return a :class:``4 with the per-case CG iteration counts (``rhs``: zeros).


Returns
-------
BatchedSolver
    ``batched(traced_params, initial_guess=None, traced_structure=None)
    -&gt; solution`` (``(solution, info)`` with ``return_info``), with the
    batch axis in front of the solution; differentiable w.r.t.
    ``traced_params`` and jit-able.

## LinearSystemBatch Objects

```python
class LinearSystemBatch(NamedTuple)
```

A batch of linear systems ``K(t_i) x_ik = b_k(t_i)`` (``i`` the cases,
``k = 1..n_rhs`` right-hand sides per case) on one CSR pattern -- what the
lhs engine of :func:`vmap` solves. :func:`vmap` builds one from a feax
solver (``n_rhs = 1``); other batched linear problems (e.g. the periodic
unit-cell homogenization of :mod:``0, six unit-strain right-hand
sides per cell on the reduced operator) build their own and call
:func:`vmap` on it, so they share the grouping, the shared factorization,
its reuse and the adjoint.

Attributes
----------
- **values** (*``(tp_chunk, u0, ts) -&gt; (C, nnz)``*): CSR values of the operators of a chunk of cases (SPD).
- **rhs** (*``(tp_chunk, u0, ts) -&gt; (C, n_rhs, n)``*): Their right-hand sides.
- **pattern** (*``ts -&gt; (indptr, indices)``*): The shared CSR pattern.
- **diag_slots** (*``ts -&gt; (n,)``*): Slot of every diagonal entry in the CSR values (diagonal scaling, Jacobi, signatures).
- **n_rhs** (*int*): Right-hand sides per case.
- **cotangents** (*``(tp, X, lam, u0, ts) -&gt; tp cotangent``, optional*): ``traced_params`` cotangents from the solutions ``X`` and adjoints ``lam`` (both ``(N, n_rhs, n)``), for a spec that assembles its operators chunk by chunk inside the solve (as :func:`vmap` does for a feax solver). Default ``None``: :func:`vmap` assembles the values and right-hand sides of the whole batch *outside* the solve -- every case&#x27;s operator is held at once -- passes them in as arrays, and JAX differentiates ``values`` and ``rhs`` itself (the solve&#x27;s VJP w.r.t. them is ``-Σ_k lam_k ⊗ x_k`` on the CSR slots and ``lam``).


## \_Engine Objects

```python
class _Engine(NamedTuple)
```

#### solve

(tp, u0, ts, factors) -&gt; (X (N, K, n), info (N, K))

#### factorize

(tp, u0, ts) -&gt; factors

## BatchedSystemSolver Objects

```python
class BatchedSystemSolver()
```

What :func:`vmap` returns for a :class:`LinearSystemBatch`:
``batched(traced_params, factors=None) -&gt; X`` with ``X`` of shape
``(N, n_rhs, n)`` (``(X, info)`` with ``return_info``; ``info`` per case
and right-hand side), differentiable w.r.t. ``traced_params``. The
operators of the whole batch are assembled outside the solve (memory:
``N`` operators); ``share`` / ``chunk`` set the groups and the CG chunks.

#### solve

```python
def solve(traced_params, factors=None)
```

Unjitted, for use inside a caller&#x27;s own jit / grad: ``(X, info)``.

#### factorize

```python
def factorize(traced_params)
```

Group factorizations for ``batched(traced_params, factors=...)`` (see
:meth:`BatchedSolver.factorize`).

#### operator\_signature

```python
def operator_signature(traced_params)
```

``(N, n)``: log of the diagonal of every case&#x27;s operator, for
:func:`feax.gene.similarity_order`.

