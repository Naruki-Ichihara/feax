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
def average_stress(problem, u_total: np.ndarray, internal_vars,
                   dim: int) -> np.ndarray
```

Compute the volume-averaged Cauchy stress in Voigt notation.

Parameters
----------
- **problem** (*FEAX Problem instance.*)
- **u_total** (*ndarray, shape (num_dofs,)*): Total displacement field.
- **internal_vars** (*FEAX InternalVars.*)
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

Parameters
----------
- **problem** (*FEAX Problem instance (linear elasticity).*)
- **bc** (*FEAX DirichletBC (typically empty for periodic unit cells).*)
- **P** (*Prolongation matrix from ``feax.flat.pbc.prolongation_matrix``.*): Shape ``(num_dofs, num_reduced_dofs)``.
- **mesh** (*FEAX mesh of the unit cell.*)
- **solver_options** (*IterativeSolverOptions, optional*): Iterative solver configuration.
- **dim** (*int*): Problem dimension. ``2`` or ``3``. Default: ``3``.


Returns
-------
callable
    ``solve(internal_vars) -&gt; HomogenizationResult``

Examples
--------
```python
>>> solve = create_homogenization_solver(problem, bc, P, mesh, dim=3)
>>> result = solve(internal_vars)
>>> result.C_hom   # ndarray, shape (6, 6)
>>> result.u_totals[0]  # displacement field for first strain case
```

