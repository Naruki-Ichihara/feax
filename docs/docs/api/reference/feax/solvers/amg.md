---
sidebar_label: amg
title: feax.solvers.amg
---

Algebraic multigrid hierarchy builder (PyAMG hierarchy -&gt; AMJax/JAX).

Builds a smoothed-aggregation AMG hierarchy on the host (PyAMG, from an
assembled CSR operator) and converts it to a JAX-native
``amjax.MultilevelSolver``. The user-facing preconditioner built on it is
:func:`feax.solvers.preconditioners.amg` (:class:`AMGPreconditioner`), which
applies one multigrid cycle as ``M`` for a matrix-free outer Krylov solve.

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

#### build\_amg\_hierarchy

```python
def build_amg_hierarchy(problem, bc, sample_csr, options)
```

Build the AMG hierarchy (an ``amjax.MultilevelSolver`` pytree).

The near-null-space ``B`` is resolved from ``options.near_nullspace`` and may
be (a) a user-defined array, (b) a known-physics preset (rigid body modes for
elasticity / constant for scalar), or (c) ``&quot;auto&quot;`` — estimated numerically
by adaptive smoothed aggregation (relaxing ``A x = 0`` from random starts).

Parameters
----------
- **sample_csr** (*feax.csr.CSRMatrix*): A representative assembled (BC-applied) Jacobian; the SA-AMG hierarchy is built from its values + pattern on the host.
- **options** (*AMGOptions*): Hierarchy / smoother / near-null-space configuration.


Returns
-------
amjax.MultilevelSolver
    The hierarchy, registered as a pytree; ``.aspreconditioner(cycle)`` gives
    ``M(b) -&gt; x`` applying one multigrid cycle (an approximate inverse).

