---
sidebar_label: amg
title: feax.solvers.amg
---

Algebraic multigrid preconditioner backend (PyAMG hierarchy -&gt; AMJax/JAX).

This builds a smoothed-aggregation AMG hierarchy *once* on the host (PyAMG, from
a sample assembled CSR Jacobian), converts it to a JAX-native
``amjax.MultilevelSolver``, and returns one multigrid V-cycle as a preconditioner
callable ``M(x) -&gt; y`` for a matrix-free outer Krylov solve.

Why this shape:
- AMG for vector elasticity/structural problems needs the **near-null-space**
  (rigid body modes) to build a good coarse space. PyAMG&#x27;s
  ``smoothed_aggregation_solver(A, B=...)`` takes it as a first-class argument;
  we generate the rigid body modes from the mesh node coordinates.
- AMJax&#x27;s JAX-side V-cycle implements only a Jacobi smoother, and *undamped*
  Jacobi diverges on elasticity, so we use a damped Jacobi smoother by default.
- The hierarchy is fixed (built from a sample matrix); the outer Krylov applies
  the current operator matrix-free and uses this AMG cycle as the preconditioner.
  An outer Krylov (gmres/cg/bicgstab) makes the fixed-hierarchy preconditioner
  robust to the operator changing between solves (Newton / parameter sweeps).

Requires the optional ``feax[amg]`` dependency (``amjax`` + ``pyamg``).

#### rigid\_body\_modes

```python
def rigid_body_modes(problem, bc=None)
```

Rigid-body near-null-space modes from mesh node coordinates.

Returns an ``(n_dof, k)`` array (k=6 in 3D: 3 translations + 3 rotations;
k=3 in 2D: 2 translations + 1 rotation) for a single vector field with
``vec == dim`` (node-major DOF layout ``dof = node*vec + component``). Returns
``None`` when the problem is not a single ``vec == dim`` field (e.g. scalar
Poisson, or a multi-field mixed problem), in which case AMG falls back to the
default constant near-null-space.

Dirichlet DOFs (``bc.bc_rows``) are zeroed: constrained DOFs are not part of
the reduced operator&#x27;s near-null-space.

#### build\_amg\_preconditioner

```python
def build_amg_preconditioner(problem, bc, sample_csr, options)
```

Build an AMG V-cycle preconditioner ``M(x) -&gt; y`` from a sample Jacobian.

The near-null-space ``B`` is resolved from ``options.near_nullspace`` and may
be (a) a user-defined array, (b) a known-physics preset (rigid body modes for
elasticity / constant for scalar), or (c) ``&quot;auto&quot;`` — estimated numerically
by adaptive smoothed aggregation (relaxing ``A x = 0`` from random starts).

Parameters
----------
- **sample_csr** (*feax.csr.CSRMatrix*): A representative assembled (BC-applied) Jacobian; the SA-AMG hierarchy is built from its values + pattern on the host.
- **options** (*AMGSolverOptions*): Hierarchy / smoother / near-null-space configuration.


Returns
-------
callable
    ``M(b) -&gt; x`` applying one multigrid cycle (an approximate inverse),
    suitable as the ``M=`` preconditioner for a JAX Krylov solver.

#### make\_self\_preconditioned\_amg\_solve

```python
def make_self_preconditioned_amg_solve(problem, bc, options)
```

Return a ``solve(A, b, x0) -&gt; x`` that rebuilds the AMG V-cycle from ``A``.

Each call builds a fresh AMG preconditioner from the *given* operator ``A``
(a feax ``CSRMatrix`` or its transpose) and runs the outer Krylov method on
it. This is what makes the Newton ``rebuild_every`` path and the adjoint solve
use a preconditioner matched to the current/converged tangent rather than a
stale one built from the initial state. Runs eagerly (a host PyAMG build per
call), so it is not meant for a jitted while-loop.

#### amg\_to\_krylov\_options

```python
def amg_to_krylov_options(amg_options,
                          problem,
                          bc,
                          traced_params,
                          traced_structure=None,
                          symmetric_elimination: bool = True)
```

Lower an :class:`AMGSolverOptions` to a matrix-free Krylov solve + AMG ``M``.

Assembles one sample Jacobian (so the AMG hierarchy + rigid-body near-null-space
can be built on the host), then returns a :class:`KrylovSolverOptions` whose
``preconditioner`` is the AMG V-cycle. The outer solver is the requested
Krylov method (``&quot;auto&quot;`` -&gt; cg for an SPD sample, else gmres). The rest of the
solver stack then treats it as an ordinary matrix-free Krylov solve.

