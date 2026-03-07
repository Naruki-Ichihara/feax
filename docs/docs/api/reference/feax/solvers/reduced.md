---
sidebar_label: reduced
title: feax.solvers.reduced
---

Reduced (matrix-free) solver path for periodic constraints.

This module contains the P-matrix reduced solve path used by
``feax.solver.create_solver`` when ``P is not None``.

#### create\_reduced\_solver

```python
def create_reduced_solver(problem, bc, P, solver_options,
                          adjoint_solver_options)
```

Create matrix-free reduced solver for periodic boundary conditions.

