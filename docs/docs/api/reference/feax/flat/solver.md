---
sidebar_label: solver
title: feax.flat.solver
---

Computational homogenization solver for periodic unit cell analysis.

Computes the homogenized stiffness matrix C_hom for a linear elastic unit
cell using asymptotic homogenization theory with periodic boundary conditions.

The homogenized stiffness relates volume-averaged stress to volume-averaged
strain:

    &lt;σ&gt; = C_hom : &lt;ε&gt;

For each of the n independent unit strain cases ε^(k), the periodic
fluctuation problem is solved and the volume-averaged stress response is
assembled into C_hom.

#### macro\_displacement

```python
def macro_displacement(mesh, epsilon_macro: np.ndarray) -> np.ndarray
```

Affine displacement field u_i = ε_ij X_j for each mesh node.

Parameters
----------
- **mesh** (*FEAX mesh with ``points`` array of shape ``(num_nodes, 2)`` or*): ``(num_nodes, 3)``.
- **epsilon_macro** (*ndarray, shape (3, 3)*): Symmetric macroscopic strain tensor.


Returns
-------
ndarray
    Flattened affine displacement vector with the same number of
    components per node as the mesh spatial dimension.

#### average\_stress

```python
def average_stress(problem, u_total: np.ndarray, traced_params,
                   dim: int) -> np.ndarray
```

Compute the volume-averaged Cauchy stress in Voigt notation.

Parameters
----------
- **problem** (*FEAX Problem instance.*)
- **u_total** (*ndarray, shape (num_dofs,)*): Total displacement field.
- **traced_params** (*FEAX TracedParams.*)
- **dim** (*int*): Problem dimension (2 or 3).


Returns
-------
ndarray
    Volume-averaged stress in Voigt notation.
    Shape ``(3,)`` for 2D: ``[σ11, σ22, σ12]``.
    Shape ``(6,)`` for 3D: ``[σ11, σ22, σ33, σ23, σ13, σ12]``.

## HomogenizationResult Objects

```python
class HomogenizationResult(NamedTuple)
```

Result of a computational homogenization analysis.

Attributes
----------
- **C_hom** (*ndarray, shape (3, 3) or (6, 6)*): Homogenized stiffness matrix in Voigt notation.
- **u_totals** (*ndarray, shape (n_cases, num_dofs)*): Total displacement fields for each unit strain case. ``u_totals[k]`` has shape ``(num_dofs,)``.
- **u_macros** (*ndarray, shape (n_cases, num_dofs)*): Macroscopic (affine) displacement fields for each unit strain case.


## \_ReducedSystem Objects

```python
class _ReducedSystem(NamedTuple)
```

The pinned, periodic-reduced homogenization system of one unit cell,
as functions of the traced parameters (shared by the single and the
batched solvers).

#### operator\_data

tp -&gt; (nnz,) CSR values of the pinned P^T K P

#### case\_rhs

(unit_strain, tp) -&gt; ((n_red,) rhs, (ndof,) u_macro)

#### zero\_mean

(ndof,) fluctuation -&gt; zero-mean fluctuation

#### indptr

static CSR pattern of the reduced operator

#### rows

(nnz,) row of every CSR slot (numpy)

#### diag\_slots

(n_red,) CSR slot of every diagonal entry

#### pins

pinned reduced dofs (one per translation)

#### create\_homogenization\_solver

```python
def create_homogenization_solver(problem: Any,
                                 bc: Any,
                                 P: Any,
                                 mesh: Any,
                                 solver_options: KrylovSolverOptions = None,
                                 dim: int = 3)
```

Create a computational homogenization solver for a linear elastic unit cell.

For each of the n independent unit strain cases the periodic fluctuation
problem is solved, the volume-averaged stress is computed, and the results
are assembled into the homogenized stiffness matrix C_hom:

    C_hom[:, k] = &lt;σ&gt;  when  ε^(k) is applied

where ε^(k) is the k-th unit strain in Voigt order.

Parameters
----------
- **problem** (*FEAX Problem instance (linear elasticity).*)
- **bc** (*FEAX DirichletBC. Leave EMPTY for periodic unit cells: the reduced*): system is singular (rigid translations) but consistent, and the Krylov solve handles it. Do NOT pin a node to suppress the translations — the affine initial guess ``u_macro`` does not satisfy such a pin, which corrupts the fluctuation field around the pinned node (invisible for a homogeneous cell, wrong C_hom for a heterogeneous one).
- **P** (*Prolongation matrix from ``feax.flat.pbc.prolongation_matrix``.*): Shape ``(num_dofs, num_reduced_dofs)``.
- **mesh** (*FEAX mesh of the unit cell.*)
- **solver_options** (*KrylovSolverOptions or DirectSolverOptions, optional*): Solver configuration. Default (``None``) is matrix-free Krylov (CG), which handles the singular periodic system directly. Pass ``DirectSolverOptions`` (GPU/cuDSS) to use the factor-once / solve-many direct path — one zero-mean-regularized factorization of ``PᵀKP`` reused across all strain cases and adjoints (machine-precise, faster at moderate-to-large cells; falls back to Krylov if cuDSS is unavailable).
- **dim** (*int*): Problem dimension. ``2`` or ``3``. Default: ``3``.


Returns
-------
callable
    ``solve(traced_params) -&gt; HomogenizationResult``. :func:``0
    batches it over designs (``fe.vmap(solve, in_axes=...,
    preconditioner=fe.shared_factor(scaling=True), share=...)``), giving
    the :class:``3 of
    :func:``4.

Examples
--------
```python
>>> solve = create_homogenization_solver(problem, bc, P, mesh, dim=3)
>>> result = solve(traced_params)
>>> result.C_hom   # ndarray, shape (6, 6)
>>> result.u_totals[0]  # displacement field for first strain case
```

## BatchedHomogenizationResult Objects

```python
class BatchedHomogenizationResult(NamedTuple)
```

Result of :class:`BatchedHomogenizationSolver`.

Attributes
----------
- **C_hom** (*ndarray, shape (B, 3, 3) or (B, 6, 6)*): Homogenized stiffness of every design (Voigt).
- **iterations** (*ndarray, shape (B, n_cases)*): CG iterations of every design and unit-strain case.
- **converged** (*ndarray, shape (B, n_cases)*): Whether each case reached the tolerance.
- **u_totals** (*ndarray (B, n_cases, num_dofs) or None*): Total displacement fields (only with ``return_fields=True``).


## BatchedHomogenizationSolver Objects

```python
class BatchedHomogenizationSolver()
```

Homogenized stiffness of a batch of unit cells that share one mesh and
differ in their parameters (density, modulus, ... fields).

Built by :func:`create_batched_homogenization_solver`. Every design has its
own reduced periodic operator ``PᵀK_bP`` (translations pinned, SPD) and the
``n_cases`` unit-strain right-hand sides. The batch is a
:class:`feax.LinearSystemBatch` solved by :func:`feax.vmap`&#x27;s lhs engine:
all ``B × n_cases`` systems by conjugate gradients, preconditioned -- per
group of ``share`` designs -- by ONE cuDSS factorization of the group&#x27;s
mean operator (:func:``1, diagonally scaled per design by
default), or by Jacobi without cuDSS.

``solver(traced_params, factors=None) -&gt; BatchedHomogenizationResult`` is
jit-compiled and differentiable w.r.t. ``traced_params``;
``solver.evaluate`` is the same function unjitted (to compose with a
caller&#x27;s own ``jax.grad`` / ``jax.jit``). ``factorize(traced_params)``
returns the group factorizations for reuse over many evaluations;
``operator_signature(traced_params)`` gives a per-design feature vector for
:func:``6.

#### evaluate

```python
def evaluate(traced_params, factors=None) -> BatchedHomogenizationResult
```

The batched homogenization, unjitted (differentiable).

#### \_\_call\_\_

```python
def __call__(traced_params, factors=None) -> BatchedHomogenizationResult
```

Homogenized stiffness of every design. ``factors`` (from
:meth:`factorize`, same batch size and ``share``) precondition without
assembling a mean or factorizing.

#### factorize

```python
def factorize(traced_params)
```

One factorization of the mean reduced operator per group of
``share`` designs (one cuDSS entry each, kept alive), for
``solver(traced_params, factors=...)``. Keep the batch order and size
while reusing them; set ``SPINEAX_FACTOR_CACHE`` (before spineax is
imported) to at least the number of groups.

#### operator\_signature

```python
def operator_signature(traced_params)
```

``(B, n_red)``: log of the diagonal of every design&#x27;s reduced
operator -- a parametrization-independent feature vector for
:func:`feax.gene.similarity_order`.

#### create\_batched\_homogenization\_solver

```python
def create_batched_homogenization_solver(
        problem,
        bc,
        P,
        mesh,
        dim: int = 3,
        *,
        in_axes=0,
        share: int = None,
        chunk: int = None,
        preconditioner: str = None,
        tol: float = 1e-8,
        atol: float = 0.0,
        maxiter: int = 2000,
        return_fields: bool = False,
        scaling: bool = True) -> BatchedHomogenizationSolver
```

Batched computational homogenization: ``C_hom`` of many unit cells at once.

The cells share the mesh, the periodic prolongation ``P`` and the material
law; they differ in ``traced_params`` (e.g. a lattice density or modulus
field per design), which carries the batch along ``in_axes`` (as in
:func:`feax.vmap`; e.g. ``fe.TracedParams(volume_vars=(0, None))`` for a
batched modulus and a shared Poisson ratio).

Instead of factorizing every design&#x27;s reduced operator ``PᵀK_bP`` (B
factorizations, memory-bound quickly), the ``B × n_cases`` unit-strain
systems are solved together by CG preconditioned by one factorization of
the mean operator per group of ``share`` designs -- the mean-based
preconditioner of :func:``7, run by :func:`feax.vmap`
on a :class:``9 (the same engine as a batch of
topology-optimization designs). It is exact for identical
designs and degrades as they spread: watch ``result.iterations`` and lower
``share`` (after ordering the batch by :meth:``4 and
:func:``5) when the counts grow.

Parameters
----------
- **in_axes** (*int, None or a TracedParams-shaped prefix*): Which leaves of ``traced_params`` carry the batch axis.
- **share** (*int, optional*): Designs per shared factorization (default: the whole batch).
- **chunk** (*int, optional*): Designs assembled and solved at a time (memory bound: ``chunk`` reduced operators held at once; the group mean is accumulated chunk by chunk, as in :func:`feax.vmap`). Default: ``share``.
- **preconditioner** (*{`&quot;shared_factor&quot;, &quot;jacobi&quot;`} or preconditioner object, optional*): Default: ``&quot;shared_factor&quot;`` with cuDSS, else ``&quot;jacobi&quot;``. A :func:``7 / :func:``9 object is used as is (its ``scaling`` wins over the argument below).
- **return_fields** (*bool*): Also return the total displacement fields ``(B, n_cases, num_dofs)``. The solver call checks ``converged`` and re-evaluates once when a design did not converge (see the note on platform reliability below); ``solver.evaluate`` (for use under your own ``jax.grad`` / ``jax.jit``) performs no such check -- inspect ``result.converged`` yourself.
- **scaling** (*bool*): Shared factorization with a per-design symmetric diagonal scaling, ``M_b = S_b K̄⁻¹ S_b`` with ``S_b = sqrt(diag K̄ / diag K_b)`` (default). It costs one diagonal product per CG iteration and keeps the shared preconditioner effective when the designs&#x27; stiffness differs by orders of magnitude in places (lattice struts vs void); ``False`` is the plain mean preconditioner of :func:``7.


Returns
-------
BatchedHomogenizationSolver
    ``solver(traced_params, factors=None) -&gt; BatchedHomogenizationResult``,
    plus ``solver.factorize(traced_params)`` and
    ``solver.operator_signature(traced_params)``.

Notes
-----
On one platform (NVIDIA GB10 with driver 580 and a JAX 0.11 development
build) the compiled gradient of this solver intermittently -- mostly on the
first execution of the executable -- produced a corrupted mean operator
(a race in XLA:GPU kernels outside feax; feax&#x27;s own assembly uses no
atomics), visible as a non-converging batched CG. It was seen with the
operators assembled inside the solve&#x27;s custom VJP; the shipped structure
(operators assembled outside, see :func:`feax.vmap`) has not reproduced
it. ``XLA_FLAGS=--xla_gpu_deterministic_ops=true`` makes it rare at some
cost in speed; the solver call also re-evaluates once when ``converged``
is not all true.

