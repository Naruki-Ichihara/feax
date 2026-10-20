---
sidebar_label: preconditioners
title: feax.solvers.preconditioners
---

Preconditioners for the Krylov solvers: one callable, one contract.

A preconditioner is a function ``preconditioner(op) -&gt; M`` that is called
*inside* the solve, with the operator of that solve, and returns the
preconditioner action ``M(r) -&gt; z`` (an approximate inverse) for the Krylov
method. ``op`` is a :class:`LinearOperator`: its matvec is always available,
and its assembled CSR values / diagonal are assembled on demand, so a
preconditioner can be built from the current operator even on the matrix-free
Krylov path. Because it runs under tracing, anything built from ``op`` (a
Jacobi diagonal, a factorization of a batch mean, ...) is rebuilt for every
solve and never goes stale.

This single entry point is passed as ``preconditioner=`` to
:func:``1 (and may be overridden per call,
``solver(tp, u0, preconditioner=P)``); it replaces the former option fields and
the preconditioner state that used to ride along ``TracedStructure``.

Built-in preconditioners
------------------------
* :func:``6 — diagonal scaling from the current operator.
* :func:``7 — a smoothed-aggregation AMG hierarchy built once on the host
  (PyAMG, from a sample operator) and applied as one multigrid cycle. The
  result, :class:``8, is a pytree, so it can be a traced
  argument of a jitted solve instead of a closure constant.
* a fixed ``M``: ``lambda op: M``.

## LinearOperator Objects

```python
class LinearOperator()
```

The operator of one linear solve, as a preconditioner sees it.

Attributes
----------
- **matvec** (*callable*): ``A @ v`` (also available as ``op @ v`` / ``op(v)``).
- **csr** (*CSRMatrix*): The assembled, BC-applied operator. Assembled on first access when the solve is matrix-free (one assembly; the matvec stays matrix-free).
- **diag** (*array*): The diagonal of ``csr`` (gathered through the problem&#x27;s diagonal slot map when available).
- **transpose** (*bool*): True for the adjoint operator ``Jᵀ``. ``csr`` then still holds ``J`` (its diagonal is the same; AMG and Jacobi do not need the transpose).
- **batched** (*bool*): True inside :func:``7 over a design batch: ``values`` is then ``(N, nnz)`` (one matrix per case on the shared pattern), ``diag`` is ``(N, n)``, and the ``M`` a preconditioner returns must act on the whole batch, ``M(R) -&gt; Z`` with ``R`` of shape ``(N, n)``.


#### wrap

```python
@classmethod
def wrap(cls, A, **kw)
```

Wrap whatever a solve function receives (CSRMatrix, its transpose,
a matvec callable, a sparse/dense matrix) into a LinearOperator.

#### values

```python
@property
def values()
```

CSR values: ``(nnz,)``, or ``(N, nnz)`` when ``batched``.

#### diag

```python
@property
def diag()
```

Diagonal of the operator: ``(n,)``, or ``(N, n)`` when ``batched``.

#### has\_csr\_pattern\_of

```python
def has_csr_pattern_of(src) -> bool
```

Whether the operator is on the problem&#x27;s own CSR pattern (so the
slot maps of ``src`` apply) rather than e.g. a merged or reduced pattern.

#### as\_pytree

```python
def as_pytree(preconditioner)
```

Return ``preconditioner`` in a form whose pytree leaves are all arrays
(``None`` passes through).

#### jacobi

```python
def jacobi(shift: float = 1e-12) -> Callable
```

Diagonal (Jacobi) preconditioner, ``M = diag(A)⁻¹``, from the current operator.

The diagonal is read from the assembled operator of *each* solve (one
assembly on the matrix-free Krylov path), so it follows the parameters —
e.g. the SIMP stiffness contrast in topology optimization. ``shift``
regularizes near-zero diagonal entries.

#### shared\_factor

```python
def shared_factor(matrix_type: str = "spd", *, device_id: int = 0) -> Callable
```

One factorization shared by a batch: ``M = K_avg⁻¹`` with
``K_avg = mean_i K_i``.

Meant for :func:`feax.vmap` over a design batch (``op.values`` of shape
``(N, nnz)``): the values are averaged over the batch, the mean matrix is
factorized once with cuDSS, and ``M`` applies its back-substitution to all
``N`` right-hand sides at once. When the ``K_i`` are close to each other
the preconditioned CG converges in a few iterations, so the batch costs
about one factorization plus a few batched SpMV / back-substitutions —
instead of ``N`` factorizations. On a single operator (``(nnz,)`` values)
the &quot;mean&quot; is the operator itself and CG converges in one iteration.

Known as a mean-based preconditioner (stochastic FEM) / approximate
reanalysis; it is only as good as the spread of the batch — watch the
iteration counts :func:`feax.vmap` returns.

Parameters
----------
- **matrix_type** (*{`&quot;spd&quot;, &quot;symmetric&quot;, &quot;general&quot;`}*): cuDSS factorization of the mean matrix (an SPD elasticity / diffusion operator after symmetric Dirichlet elimination is ``&quot;spd&quot;``).
- **device_id** (*int*): CUDA device of the factorization.


## AMGOptions Objects

```python
@dataclass(frozen=True, eq=False)
class AMGOptions()
```

Configuration of the smoothed-aggregation AMG preconditioner (:func:`amg`).

Parameters
----------
- **near_nullspace** (*array, str, or None, default None*): The AMG near-null-space ``B`` — the low-energy modes the coarse grid must represent. Accepts three kinds of input:
- **num_nullspace** (*int, optional*): Number of candidate vectors for ``near_nullspace=&quot;adaptive_sa&quot;``.
- **cycle** (*str, default &quot;V&quot;*): Multigrid cycle type (&quot;V&quot;, &quot;W&quot;, &quot;F&quot;).
- **smoother_omega** (*float, default 0.67*): Damped-Jacobi relaxation factor (undamped Jacobi diverges on elasticity).
- **smoother_sweeps** (*int, default 2*): Jacobi sweeps per pre/post smoothing.
- **coarse_solver** (*str, default &quot;pinv&quot;*): AMJax coarse-grid solver (&quot;pinv&quot;, &quot;lu&quot;, &quot;qr&quot;, &quot;jacobi&quot;).
- **strength** (*float, optional*): SA strength-of-connection threshold (PyAMG default when None).
- **rebuild_every** (*int or None, default None*): Newton-only: how often to rebuild the hierarchy from the *current* tangent during a nonlinear solve (the near-null-space is always reused).
- **lag_tol** (*float, default 1e-3*): Adaptive-lag trigger: rebuild when the per-step linear solve&#x27;s relative residual exceeds this.


## AMGPreconditioner Objects

```python
@jax.tree_util.register_pytree_node_class
class AMGPreconditioner()
```

A built AMG hierarchy as a preconditioner (``preconditioner(op) -&gt; M``).

Created by :func:`amg`. A pytree whose leaves are the hierarchy&#x27;s arrays, so
a jitted caller can pass it as an argument of the solve
(``solver(tp, u0, preconditioner=P)``) instead of baking every level in as a
constant. ``rebuild(A)`` builds a new hierarchy from a concrete operator on
the host (used by the Newton host loop; not traceable).

#### rebuild

```python
def rebuild(A) -> "AMGPreconditioner"
```

A new preconditioner whose hierarchy is built from the concrete
(BC-applied) CSR operator ``A`` — host-side, not traceable.

#### amg

```python
def amg(problem,
        bc,
        traced_params,
        options: Optional[AMGOptions] = None,
        *,
        traced_structure=None,
        symmetric_elimination: bool = True,
        projection=None) -> AMGPreconditioner
```

Build a smoothed-aggregation AMG preconditioner from a sample operator.

Assembles one BC-applied Jacobian at ``traced_params`` (on the host, via
PyAMG — requires the optional ``feax[amg]`` extra) and converts the
hierarchy to a JAX-native ``amjax.MultilevelSolver``. The returned
:class:`AMGPreconditioner` applies one multigrid cycle as ``M``; the outer
Krylov applies the current operator matrix-free, so the fixed hierarchy
stays usable as the operator changes between solves (parameter sweeps,
Newton steps — see ``AMGOptions.rebuild_every`` for rebuilding it).

Parameters
----------
- **traced_params** (*TracedParams*): Sample parameters the hierarchy is built at.
- **options** (*AMGOptions, optional*): Hierarchy / smoother / near-null-space configuration.
- **traced_structure** (*TracedStructure, optional*): Assemble the sample on the TracedStructure path (recommended when the solver uses one).
- **symmetric_elimination** (*bool, default True*): Must match the solver&#x27;s Dirichlet elimination.
- **projection** (*sparse matrix, optional*): The reduced solver&#x27;s projection (``create_solver(projection=...)``): the sample is then the reduced operator ``PᵀJP`` and the near-null-space is mapped into the reduced space.


Examples
--------
```python
>>> P = fe.amg(problem, bc, tp, fe.AMGOptions(near_nullspace=&quot;rigid_body&quot;))
>>> solver = fe.create_solver(problem, bc, fe.KrylovSolverOptions(solver=&quot;cg&quot;),
...                           preconditioner=P, linear=True,
...                           traced_params=tp, traced_structure=ts)
>>> u = jax.jit(lambda tp, ts, P: solver(tp, u0, traced_structure=ts,
...                                      preconditioner=P))(tp, ts, P)
```

