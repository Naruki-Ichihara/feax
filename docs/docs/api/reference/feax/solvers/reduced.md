---
sidebar_label: reduced
title: feax.solvers.reduced
---

Reduced solver path for periodic constraints.

This module contains the P-matrix reduced solve path used by
``feax.solver.create_solver`` when ``P is not None``.

Two operator representations:

* **matrix-free** (``KrylovSolverOptions``): the reduced operator ``PᵀJP`` is
  applied as three matvecs ``Pᵀ(J(P v))`` — never assembled.
* **assembled** (``DirectSolverOptions`` / an AMG preconditioner that rebuilds): the reduced
  pattern is computed by a boolean triple product of the known ``P`` and
  connectivity patterns (:func:``4), and the
  operator is materialized from its matrix-free action by colored probing
  (:func:``5) — enabling direct factorization and
  AMG preconditioning for periodic problems.

#### create\_reduced\_solver

```python
def create_reduced_solver(problem,
                          bc,
                          projection,
                          solver_options,
                          adjoint_solver_options,
                          preconditioner=None,
                          adjoint_preconditioner=None)
```

Create the reduced solver for periodic boundary conditions.

``KrylovSolverOptions`` runs fully matrix-free (three matvecs per Krylov
iteration); a ``preconditioner(op) -&gt; M`` sees the reduced operator (its
values assembled on demand by colored probing). ``DirectSolverOptions``,
and an :class:`~feax.solvers.preconditioners.AMGPreconditioner` that
rebuilds its hierarchy, assemble the reduced operator ``PᵀJP``: its
sparsity is the boolean triple product of the known patterns, its values
come from colored probes of the matrix-free action (``num_colors`` matvecs,
once per solve) — see :mod:``1. Periodic operators are symmetric
after symmetric elimination, so the adjoint reuses the same (assembled or
matrix-free) operator.

