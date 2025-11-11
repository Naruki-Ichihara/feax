# Lattice Structure Homogenization

This tutorial demonstrates computational homogenization of lattice structures using FEAX's `flat` toolkit. We compute the effective stiffness tensor of a BCC (Body-Centered Cubic) lattice using periodic boundary conditions and graph-based structure definition.

## Overview

**Computational homogenization** determines effective material properties of periodic microstructures by:

1. Defining a representative unit cell with periodic boundary conditions
2. Applying prescribed macroscopic strain states
3. Computing volume-averaged stress response
4. Assembling the homogenized stiffness tensor $\mathbf{C}_{\text{hom}}$

The relation between average stress and strain: $\langle \boldsymbol{\sigma} \rangle = \mathbf{C}_{\text{hom}} : \langle \boldsymbol{\epsilon} \rangle$

## The `feax.flat` Toolkit

FEAX provides the `flat` module for periodic structures and homogenization:

```python
import feax.flat as flat
```

**Key modules:**

- **`flat.unitcell`** - Unit cell base class with boundary detection
- **`flat.graph`** - Graph-based lattice structure generation
- **`flat.pbc`** - Periodic boundary condition utilities
- **`flat.solver`** - Specialized homogenization solvers
- **`flat.utils`** - Visualization tools (stiffness sphere, etc.)

## Problem Setup

### Material Properties

```python
import feax as fe
import feax.flat as flat
import jax.numpy as np

E_base = 210e9  # Pa (steel)
nu = 0.3
```

### Linear Elasticity Problem

```python
class LinearElasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E, nu_val):
            mu = E / (2.0 * (1.0 + nu_val))
            lmbda = E * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress
```

## Unit Cell Definition

Use `flat.unitcell.UnitCell` to define the computational domain:

```python
class BCCUnitCell(flat.unitcell.UnitCell):
    """BCC lattice unit cell."""

    def mesh_build(self, mesh_size):
        return fe.mesh.box_mesh(size=1.0, mesh_size=mesh_size, element_type='HEX8')

# Create unit cell
unitcell = BCCUnitCell(mesh_size=0.05)
mesh = unitcell.mesh
```

**Key features of `UnitCell`:**
- Automatically computes bounding box (`unitcell.lb`, `unitcell.ub`)
- Provides boundary detection methods (`is_corner`, `is_edge`, `is_face`)
- Generates mapping functions for periodic pairings
- Compatible with `flat.pbc.periodic_bc_3D()`

## Graph-Based Lattice Structure

Use `flat.graph` to define strut-based lattice structures:

```python
# Define BCC lattice: 8 corners + 1 center node
corners = np.array([[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]], dtype=np.float32)
center = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
nodes = np.vstack([corners, center])

# BCC edges: all corners connect to center
edges = np.array([[i, 8] for i in range(8)])

# Create problem first
problem = LinearElasticity(mesh=mesh, vec=3, dim=3, ele_type='HEX8', location_fns=[])

# Create lattice density field using graph
lattice_func = flat.graph.create_lattice_function(nodes, edges, radius=0.05)
rho = flat.graph.create_lattice_density_field(problem, lattice_func,
                                               density_solid=1.0, density_void=0.01)
```

**How `flat.graph` works:**

1. **`create_lattice_function(nodes, edges, radius)`** - Creates function that evaluates if point is near any strut
2. **`create_lattice_density_field(problem, lattice_func, ...)`** - Evaluates lattice at element centroids
3. Returns element-based density array `(num_elements,)`

**Advantages:**
- Clean node-edge representation
- Differentiable through JAX
- Efficient vectorized evaluation
- Works with arbitrary lattice topologies

## Periodic Boundary Conditions

Use `flat.pbc.periodic_bc_3D()` for full 3D periodicity:

```python
pairings = flat.pbc.periodic_bc_3D(unitcell, vec=3, dim=3)
P = flat.pbc.prolongation_matrix(pairings, mesh, vec=3)
```

**What `periodic_bc_3D()` does:**
- Creates pairings for all 3 face pairs (x, y, z directions)
- Creates pairings for all 12 edge pairs
- Creates pairings for all 7 corner pairs (origin excluded)
- Total: 25 geometric pairings × 3 components = 75 periodic constraints

The prolongation matrix $\mathbf{P}$ maps reduced DOFs to full DOFs:

$$
\mathbf{u}_{\text{full}} = \mathbf{P} \, \mathbf{u}_{\text{reduced}}
$$

## Internal Variables with Density

Use element-based variables for density-dependent properties:

```python
bc_config = fe.DCboundary.DirichletBCConfig([])
bc = bc_config.create_bc(problem)

# Density-based Young's modulus
E_field = fe.internal_vars.InternalVars.create_cell_var(problem, E_base * rho)
nu_field = fe.internal_vars.InternalVars.create_cell_var(problem, nu)
internal_vars = fe.internal_vars.InternalVars(volume_vars=(E_field, nu_field), surface_vars=())
```

**Why cell-based variables?**
- Density field from `flat.graph` is element-based
- More efficient than quad-point based for homogenization
- Natural for topology optimization

## Homogenization Solver

Use `flat.solver.create_homogenization_solver()` to compute $\mathbf{C}_{\text{hom}}$:

```python
solver_options = fe.solver.SolverOptions(tol=1e-8, linear_solver="cg", verbose=False)

compute_C_hom = flat.solver.create_homogenization_solver(
    problem, bc, P, solver_options, mesh, dim=3
)

C_hom = compute_C_hom(internal_vars)
```

**How it works:**

For 3D, the solver:
1. Applies 6 unit strain cases: $\epsilon_{11}, \epsilon_{22}, \epsilon_{33}, \gamma_{23}, \gamma_{13}, \gamma_{12}$
2. Solves each case with periodic BCs: $\mathbf{K} \mathbf{u} = -\mathbf{K} \mathbf{u}_{\text{macro}}$
3. Computes volume-averaged stress: $\langle \boldsymbol{\sigma} \rangle$
4. Assembles stiffness matrix: $\mathbf{C}_{\text{hom}}$ (6×6 in Voigt notation)

**Key properties:**
- Fully differentiable w.r.t. `internal_vars` (topology optimization)
- Uses affine displacement method for efficiency
- Automatically handles periodic constraints via $\mathbf{P}$ matrix

## Extract Engineering Constants

For cubic symmetry materials:

```python
C11 = C_hom[0, 0]
C12 = C_hom[0, 1]
C44 = C_hom[3, 3]

E_eff = (C11 - C12) * (C11 + 2*C12) / (C11 + C12)
nu_eff = C12 / (C11 + C12)
G_eff = C44

print(f"Effective Young's modulus: {E_eff/1e9:.2f} GPa")
print(f"Effective Poisson's ratio: {nu_eff:.3f}")
print(f"Effective shear modulus: {G_eff/1e9:.2f} GPa")
print(f"Relative stiffness: {E_eff/E_base:.3f}")
```

## Visualization

### Save Lattice Structure

```python
import os

output_dir = os.path.join(os.path.dirname(__file__), "data", "vtk")
os.makedirs(output_dir, exist_ok=True)

lattice_file = os.path.join(output_dir, "bcc_lattice_structure.vtu")
fe.utils.save_sol(
    mesh=mesh,
    sol_file=lattice_file,
    cell_infos=[("density", rho)]
)
```

### Visualize Stiffness Sphere

Use `flat.utils.visualize_stiffness_sphere()` for directional stiffness:

```python
sphere_file = os.path.join(output_dir, "bcc_stiffness_sphere.vtk")
flat.utils.visualize_stiffness_sphere(
    C_hom,
    output_file=sphere_file,
)
```

The stiffness sphere shows Young's modulus in each direction:

$$
E(\mathbf{n}) = \frac{1}{\mathbf{n}^T \mathbf{C}_{\text{hom}}^{-1} \mathbf{n}}
$$

**Interpretation:**
- Sphere radius = directional stiffness
- Perfectly spherical = isotropic material
- Elongated = anisotropic (stiffer in certain directions)

## Complete Code

```python
import os
import feax as fe
import feax.flat as flat
import jax.numpy as np

# Material properties
E_base = 210e9  # Pa (steel)
nu = 0.3

class LinearElasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E, nu_val):
            mu = E / (2.0 * (1.0 + nu_val))
            lmbda = E * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress

class BCCUnitCell(flat.unitcell.UnitCell):
    def mesh_build(self, mesh_size):
        return fe.mesh.box_mesh(size=1.0, mesh_size=mesh_size, element_type='HEX8')

# Create unit cell
unitcell = BCCUnitCell(mesh_size=0.05)
mesh = unitcell.mesh

# Define BCC lattice structure
corners = np.array([[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]], dtype=np.float32)
center = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
nodes = np.vstack([corners, center])
edges = np.array([[i, 8] for i in range(8)])

# Create problem and density field
problem = LinearElasticity(mesh=mesh, vec=3, dim=3, ele_type='HEX8', location_fns=[])
lattice_func = flat.graph.create_lattice_function(nodes, edges, radius=0.05)
rho = flat.graph.create_lattice_density_field(problem, lattice_func, density_solid=1.0, density_void=0.01)

# Periodic boundary conditions
pairings = flat.pbc.periodic_bc_3D(unitcell, vec=3, dim=3)
P = flat.pbc.prolongation_matrix(pairings, mesh, vec=3)

# Boundary conditions and internal variables
bc = fe.DCboundary.DirichletBCConfig([]).create_bc(problem)
E_field = fe.internal_vars.InternalVars.create_cell_var(problem, E_base * rho)
nu_field = fe.internal_vars.InternalVars.create_cell_var(problem, nu)
internal_vars = fe.internal_vars.InternalVars(volume_vars=(E_field, nu_field), surface_vars=())

# Homogenization
solver_options = fe.solver.SolverOptions(tol=1e-8, linear_solver="cg")
compute_C_hom = flat.solver.create_homogenization_solver(problem, bc, P, solver_options, mesh, dim=3)
C_hom = compute_C_hom(internal_vars)

# Extract properties
C11, C12, C44 = C_hom[0, 0], C_hom[0, 1], C_hom[3, 3]
E_eff = (C11 - C12) * (C11 + 2*C12) / (C11 + C12)
nu_eff = C12 / (C11 + C12)

print(f"Effective Young's modulus: {E_eff/1e9:.2f} GPa")
print(f"Relative stiffness: {E_eff/E_base:.3f}")

# Save results
output_dir = os.path.join(os.path.dirname(__file__), "data", "vtk")
os.makedirs(output_dir, exist_ok=True)
fe.utils.save_sol(mesh, os.path.join(output_dir, "bcc_lattice.vtu"), cell_infos=[("density", rho)])
flat.utils.visualize_stiffness_sphere(C_hom, output_file=os.path.join(output_dir, "stiffness_sphere.vtk"))
```

## Advanced Topics

### Custom Lattice Topologies

Define any lattice using node-edge graphs:

```python
# Octet truss lattice
nodes = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],  # Bottom
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],  # Top
    [0.5, 0.5, 0.5]  # Center
])
edges = np.array([
    [0, 8], [1, 8], [2, 8], [3, 8],  # Bottom to center
    [4, 8], [5, 8], [6, 8], [7, 8],  # Top to center
])

lattice_func = flat.graph.create_lattice_function(nodes, edges, radius=0.05)
```

### 2D Homogenization

For 2D problems (plane stress/strain):

```python
mesh = fe.mesh.rectangle_mesh(Nx=32, Ny=32, domain_x=1.0, domain_y=1.0)
unitcell = MyUnitCell2D()  # Implement mesh_build() for 2D

# 2D periodic BCs (only x, y directions)
compute_C_hom = flat.solver.create_homogenization_solver(
    problem, bc, P, solver_options, mesh, dim=2
)
# Returns 3×3 stiffness matrix (ε11, ε22, γ12)
```

## Summary

**Key concepts:**
- **`flat.unitcell.UnitCell`** - Abstract base for unit cell definition
- **`flat.graph`** - Node-edge graph → density field
- **`flat.pbc.periodic_bc_3D()`** - Automatic 3D periodic constraints
- **`flat.solver.create_homogenization_solver()`** - Computes $\mathbf{C}_{\text{hom}}$
- **`flat.utils.visualize_stiffness_sphere()`** - Directional stiffness visualization

**Workflow:**
1. Define `UnitCell` subclass with `mesh_build()`
2. Create lattice structure using `flat.graph`
3. Apply periodic BCs with `flat.pbc.periodic_bc_3D()`
4. Compute homogenized stiffness with `flat.solver`
5. Visualize results with `flat.utils`

## Further Reading

- [Periodic Boundary Conditions](periodic_boundary_conditions.md) - Detailed PBC tutorial
- [examples/advance/lattice_homogenization.py](../../examples/advance/lattice_homogenization.py) - Complete working example
- [API: flat.graph](../api/reference/feax/flat/graph.md) - Graph-based structure generation
- [API: flat.solver](../api/reference/feax/flat/solver.md) - Homogenization solvers
