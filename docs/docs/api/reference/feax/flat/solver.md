---
sidebar_label: solver
title: feax.flat.solver
---

Solver utilities for periodic lattice structures and homogenization problems.

This module provides specialized solvers for unit cell analysis and computational
homogenization with periodic boundary conditions. It implements the macro term
approach for handling prescribed macroscopic strains without rigid body motion.

Key Functions:
create_homogenization_solver: Create solver for unit cell homogenization
create_affine_displacement_solver: Create solver for affine displacement using JAX linear solvers
create_macro_displacement_field: Generate macro displacement from strain

**Example**:

  Basic homogenization solver usage:
  
  &gt;&gt;&gt; from feax.flat.solver import create_homogenization_solver
  &gt;&gt;&gt; from feax.flat.pbc import periodic_bc_3D, prolongation_matrix
  &gt;&gt;&gt;
  &gt;&gt;&gt; # Setup periodic boundary conditions
  &gt;&gt;&gt; pbc = periodic_bc_3D(unitcell, vec=3, dim=3)
  &gt;&gt;&gt; P = prolongation_matrix(pbc, mesh, vec=3)
  &gt;&gt;&gt;
  &gt;&gt;&gt; # Define macroscopic strain
  &gt;&gt;&gt; epsilon_macro = np.array([[0.01, 0.0, 0.0],
  &gt;&gt;&gt;                          [0.0, 0.0, 0.0],
  &gt;&gt;&gt;                          [0.0, 0.0, 0.0]])
  &gt;&gt;&gt;
  &gt;&gt;&gt; # Create homogenization solver
  &gt;&gt;&gt; solver = create_homogenization_solver(
  &gt;&gt;&gt;     problem, bc, P, epsilon_macro, solver_options, mesh
  &gt;&gt;&gt; )
  &gt;&gt;&gt;
  &gt;&gt;&gt; # Solve for total displacement (fluctuation + macro)
  &gt;&gt;&gt; solution = solver(internal_vars, initial_guess)

#### create\_macro\_displacement\_field

```python
def create_macro_displacement_field(mesh,
                                    epsilon_macro: np.ndarray) -> np.ndarray
```

Create macroscopic displacement field from macroscopic strain tensor.

Computes the affine displacement field u_macro = epsilon_macro @ X for each
node position X in the mesh. This represents the displacement that would
occur under pure macroscopic deformation without any fluctuations.

**Arguments**:

- `mesh` - FEAX mesh object with points attribute
- `epsilon_macro` _np.ndarray_ - Macroscopic strain tensor of shape (3, 3)
  

**Returns**:

- `np.ndarray` - Flattened displacement field of shape (num_nodes * 3,)
  

**Example**:

  &gt;&gt;&gt; # Pure extension in x-direction
  &gt;&gt;&gt; epsilon_macro = np.array([[0.01, 0.0, 0.0],
  &gt;&gt;&gt;                          [0.0, 0.0, 0.0],
  &gt;&gt;&gt;                          [0.0, 0.0, 0.0]])
  &gt;&gt;&gt; u_macro = create_macro_displacement_field(mesh, epsilon_macro)

#### create\_affine\_displacement\_solver

```python
def create_affine_displacement_solver(
    problem: Any,
    bc: Any,
    P: np.ndarray,
    epsilon_macro: np.ndarray,
    mesh: Any,
    solver_options: SolverOptions = None
) -> Callable[[Any, np.ndarray], np.ndarray]
```

Create an affine displacement solver for LINEAR elasticity homogenization.

This solver computes the affine displacement problem for linear elasticity:
1. Solves: K @ u_fluctuation = -K @ u_macro  (in reduced space)
2. Returns: u_total = u_fluctuation + u_macro
3. Fully differentiable via JAX&#x27;s implicit differentiation

**Note**: This solver is for LINEAR problems only (constant stiffness K).
For nonlinear problems with periodic BCs, use create_solver(problem, bc, P=P) directly.
Homogenization (computing C_hom) only makes sense for linear elasticity where
the stiffness tensor is constant.

**Arguments**:

- `problem` - FEAX Problem instance (must be linear elasticity)
- `bc` - Dirichlet boundary conditions (typically empty for periodic problems)
- `P` _np.ndarray_ - Prolongation matrix from periodic boundary conditions
- `epsilon_macro` _np.ndarray_ - Macroscopic strain tensor of shape (3, 3)
- `mesh` - FEAX mesh object for computing macro displacement field
- `solver_options` _SolverOptions, optional_ - Linear solver configuration
  

**Returns**:

- `Callable` - Differentiable solver (internal_vars, initial_guess) -&gt; total displacement
  

**Example**:

  &gt;&gt;&gt; pbc = periodic_bc_3D(unitcell, vec=3, dim=3)
  &gt;&gt;&gt; P = prolongation_matrix(pbc, mesh, vec=3)
  &gt;&gt;&gt; epsilon_macro = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
  &gt;&gt;&gt; solver = create_affine_displacement_solver(problem, bc, P, epsilon_macro, mesh)
  &gt;&gt;&gt; u_total = solver(internal_vars, initial_guess)

#### create\_homogenization\_solver

```python
def create_homogenization_solver(problem: Any,
                                 bc: Any,
                                 P: np.ndarray,
                                 solver_options: SolverOptions,
                                 mesh: Any,
                                 dim: int = 3) -> Callable[[Any], np.ndarray]
```

Create a computational homogenization solver for LINEAR elasticity.

This solver runs multiple macroscopic strain cases and computes the volume-averaged
stress response to determine the homogenized stiffness matrix C_hom.

For 2D: Runs 3 independent strain cases (11, 22, 12)
For 3D: Runs 6 independent strain cases (11, 22, 33, 23, 13, 12)

The homogenized stiffness relates average stress to average strain via:
&lt;σ&gt; = C_hom : &lt;ε&gt;

**Important**: This solver is for LINEAR elasticity only (constant C_hom).
For nonlinear materials, C depends on strain state and homogenization requires
tangent/secant stiffness computation at each strain level. For nonlinear periodic
problems, use create_solver(problem, bc, P=P) directly.

**Differentiability**:
The homogenization solver is fully differentiable w.r.t. internal_vars (material
properties) via JAX&#x27;s built-in implicit differentiation. This enables topology
optimization with homogenized properties as objectives.

**Arguments**:

- `problem` - FEAX Problem instance (must be LINEAR elasticity)
- `bc` - Dirichlet boundary conditions (typically empty for periodic problems)
- `P` _np.ndarray_ - Prolongation matrix from periodic boundary conditions
- `solver_options` _SolverOptions_ - Linear solver configuration
- `mesh` - FEAX mesh object for computing macro displacement field
- `dim` _int_ - Problem dimension (2 or 3). Defaults to 3.
  

**Returns**:

- `Callable` - Differentiable function that takes internal_vars and returns homogenized stiffness matrix
- `Shape` - (3, 3) for 2D in Voigt notation, (6, 6) for 3D in Voigt notation
  

**Raises**:

- `ValueError` - If dim is not 2 or 3
  

**Example**:

  &gt;&gt;&gt; # 3D homogenization
  &gt;&gt;&gt; from feax.flat.pbc import periodic_bc_3D, prolongation_matrix
  &gt;&gt;&gt; pbc = periodic_bc_3D(unitcell, vec=3, dim=3)
  &gt;&gt;&gt; P = prolongation_matrix(pbc, mesh, vec=3)
  &gt;&gt;&gt;
  &gt;&gt;&gt; # Create homogenization solver
  &gt;&gt;&gt; compute_C_hom = create_homogenization_solver(
  &gt;&gt;&gt;     problem, bc, P, solver_options, mesh, dim=3
  &gt;&gt;&gt; )
  &gt;&gt;&gt;
  &gt;&gt;&gt; # Compute homogenized stiffness
  &gt;&gt;&gt; C_hom = compute_C_hom(internal_vars)
  &gt;&gt;&gt; print(f&quot;Homogenized stiffness matrix: {`C_hom.shape`}&quot;)  # (6, 6) for 3D

#### create\_unit\_cell\_solver

```python
def create_unit_cell_solver(
        problem: Any,
        bc: Any,
        P: np.ndarray,
        solver_options: SolverOptions = None,
        adjoint_solver_options: SolverOptions = None,
        epsilon_macro: np.ndarray = None,
        mesh: Any = None,
        iter_num: int = 1) -> Callable[[Any, np.ndarray], np.ndarray]
```

Create a differentiable unit cell solver for lattice structures with custom VJP.

This solver handles periodic boundary conditions and optional macroscopic strains,
with full gradient support through the adjoint method. It follows the same pattern
as FEAX&#x27;s main create_solver but specialized for unit cell problems.

**Arguments**:

- `problem` - FEAX Problem instance
- `bc` - Dirichlet boundary conditions (typically empty for periodic problems)
- `P` _np.ndarray_ - Prolongation matrix from periodic BCs
- `solver_options` _SolverOptions, optional_ - Forward solver configuration
- `adjoint_solver_options` _SolverOptions, optional_ - Adjoint solver configuration
- `epsilon_macro` _np.ndarray, optional_ - Macroscopic strain tensor. If provided,
  adds affine displacement term.
- `mesh` _optional_ - FEAX mesh object. Required if epsilon_macro is provided.
- `iter_num` _int_ - Number of iterations (default 1 for linear problems)
  

**Returns**:

- `Callable` - Differentiable unit cell solver function with signature
  (internal_vars, initial_guess) -&gt; solution
  

**Raises**:

- `ValueError` - If epsilon_macro is provided but mesh is None
  

**Example**:

  Standard periodic solver (no macro strain):
  &gt;&gt;&gt; solver = create_unit_cell_solver(problem, bc, P, solver_options)
  &gt;&gt;&gt; sol = solver(internal_vars, initial_guess)
  
  With macro strain (affine displacement):
  &gt;&gt;&gt; epsilon = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
  &gt;&gt;&gt; solver = create_unit_cell_solver(problem, bc, P, solver_options,
  &gt;&gt;&gt;                                  epsilon_macro=epsilon, mesh=mesh)
  &gt;&gt;&gt;
  &gt;&gt;&gt; # Use in optimization with gradients
  &gt;&gt;&gt; def loss_fn(internal_vars):
  &gt;&gt;&gt;     sol = solver(internal_vars, initial_guess)
  &gt;&gt;&gt;     return compute_compliance(sol)
  &gt;&gt;&gt; grad_fn = jax.grad(loss_fn)

