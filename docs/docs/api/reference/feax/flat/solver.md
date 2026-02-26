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

#### create\_homogenization\_solver

```python
def create_homogenization_solver(problem: Any,
                                 bc: Any,
                                 P: Any,
                                 mesh: Any,
                                 solver_options: IterativeSolverOptions = None,
                                 dim: int = 3)
```

Create a computational homogenization solver for a linear elastic unit cell.

For each of the n independent unit strain cases the periodic fluctuation
problem is solved, the volume-averaged stress is computed, and the results
are assembled into the homogenized stiffness matrix C_hom:

    C_hom[:, k] = &lt;σ&gt;  when  ε^(k) is applied

where ε^(k) is the k-th unit strain in Voigt order.

Internally uses ``feax.solver.create_solver`` with the prolongation matrix
P.  For each macroscopic strain ε^(k), the fluctuation solver returns the
periodic fluctuation P u&#x27;_red, and the total displacement is reconstructed
as u_total = P u&#x27;_red + u_macro.

Parameters
----------
- **problem** (*FEAX Problem instance (linear elasticity).*)
- **bc** (*FEAX DirichletBC (typically empty for periodic unit cells).*)
- **P** (*Prolongation matrix from ``feax.flat.pbc.prolongation_matrix``.*): Shape ``(num_dofs, num_reduced_dofs)``.
- **mesh** (*FEAX mesh of the unit cell.*)
- **solver_options** (*IterativeSolverOptions, optional*): Iterative solver configuration. Default: ``IterativeSolverOptions()`` which uses ``solver=&quot;auto&quot;`` (→ CG, since P^T K P is SPD), ``tol=1e-10``, ``atol=1e-10``, ``maxiter=10000``.
- **dim** (*int*): Problem dimension. ``2`` → output shape ``(3, 3)``; ``3`` → output shape ``(6, 6)``. Default: ``3``.


Returns
-------
callable
    ``compute_C_hom(internal_vars) -&gt; ndarray`` of shape
    ``(3, 3)`` (2D) or ``(6, 6)`` (3D).

Raises
------
ValueError
    If ``dim`` is not 2 or 3.

Examples
--------
```python
>>> from feax.flat.pbc import periodic_bc_3D, prolongation_matrix
>>> from feax.solver_option import IterativeSolverOptions
>>> pbc = periodic_bc_3D(unitcell, vec=3, dim=3)
>>> P = prolongation_matrix(pbc, mesh, vec=3)
>>> opts = IterativeSolverOptions(solver=&quot;cg&quot;, tol=1e-10, maxiter=5000)
>>> compute_C_hom = create_homogenization_solver(
...     problem, bc, P, mesh, solver_options=opts, dim=3
... )
>>> C_hom = compute_C_hom(internal_vars)  # ndarray, shape (6, 6)
```

