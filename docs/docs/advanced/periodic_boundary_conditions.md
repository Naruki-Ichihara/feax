# Periodic Boundary Conditions

This tutorial demonstrates how to apply periodic boundary conditions in FEAX using prolongation matrices.

## Problem Description

Consider the 2D Poisson equation on a unit square $\Omega = [0, 1] \times [0, 1]$:

$$
-\nabla \cdot (\theta \nabla u) = f \quad \text{in } \Omega
$$

with periodic BCs on left-right boundaries: $u(0, y) = u(1, y)$, and Dirichlet BCs on top-bottom: $u(x, 0) = u(x, 1) = 0$.

## Mathematical Formulation

The prolongation matrix $\mathbf{P}$ relates full and reduced DOFs:

$$
\mathbf{u}_{\text{full}} = \mathbf{P} \, \mathbf{u}_{\text{reduced}}
$$

The system is solved in reduced space: $(\mathbf{P}^T \mathbf{K} \mathbf{P}) \, \mathbf{u}_{\text{reduced}} = \mathbf{P}^T \mathbf{f}$

## Implementation

### Step 1: Mesh Generation

```python
import feax as fe
import jax.numpy as np

mesh = fe.mesh.rectangle_mesh(Nx=32, Ny=32, domain_x=1.0, domain_y=1.0)
```

### Step 2: Problem Definition

```python
class PoissonParametric(fe.problem.Problem):
    def get_tensor_map(self):
        def tensor_map(u_grad, theta):
            return theta * u_grad
        return tensor_map

    def get_mass_map(self):
        def mass_map(u, x, theta):
            dx, dy = x[0] - 0.5, x[1] - 0.5
            val = x[0]*np.sin(5.0*np.pi*x[1]) + np.exp(-(dx*dx + dy*dy)/0.02)
            return np.array([-val])
        return mass_map

problem = PoissonParametric(mesh=mesh, vec=1, dim=2, ele_type='QUAD4', location_fns=[])
```

### Step 3: Periodic Boundary Conditions

```python
import feax.flat as flat

def left_boundary(point):
    return np.isclose(point[0], 0.0, atol=1e-5)

def right_boundary(point):
    return np.isclose(point[0], 1.0, atol=1e-5)

def mapping_x(point_A):
    return np.array([point_A[0] + 1.0, point_A[1]])

periodic_pairing = flat.pbc.PeriodicPairing(
    location_master=left_boundary,
    location_slave=right_boundary,
    mapping=mapping_x,
    vec=0
)

P = flat.pbc.prolongation_matrix([periodic_pairing], mesh, vec=1)
```

### Step 4: Dirichlet Boundary Conditions

```python
def bottom_boundary(point):
    return np.isclose(point[1], 0.0, atol=1e-5)

def top_boundary(point):
    return np.isclose(point[1], 1.0, atol=1e-5)

bc_config = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(bottom_boundary, 0, 0.0),
    fe.DCboundary.DirichletBCSpec(top_boundary, 0, 0.0),
])
bc = bc_config.create_bc(problem)
```

### Step 5: Internal Variables

```python
theta = 1.0
theta_array = fe.internal_vars.InternalVars.create_uniform_volume_var(problem, theta)
internal_vars = fe.internal_vars.InternalVars(volume_vars=(theta_array,), surface_vars=())
```

### Step 6: Solver

```python
solver_options = fe.solver.SolverOptions(tol=1e-8, linear_solver="cg")
solver = fe.solver.create_solver(problem, bc, solver_options, iter_num=1, P=P)
```

Pass prolongation matrix `P` to `create_solver()`.

### Step 7: Solve

```python
initial_guess = np.zeros(problem.num_total_dofs_all_vars)
sol_full = solver(internal_vars, initial_guess)
fe.utils.save_sol(mesh, "periodic_poisson.vtu", point_infos=[("u", sol_full.reshape(-1, 1))])
```

## 3D Periodic Boundary Conditions

```python
from feax.lattice_toolkit.pbc import periodic_bc_3D

mesh_3d = fe.mesh.box_mesh(size=1.0, mesh_size=0.1)
pairings_3d = periodic_bc_3D(mesh_3d, vec=3, dim=3)
P_3d = flat.pbc.prolongation_matrix(pairings_3d, mesh_3d, vec=3)
```

## Complete Code

```python
import feax as fe
import feax.flat as flat
import jax.numpy as np

# Problem definition
class PoissonParametric(fe.problem.Problem):
    def get_tensor_map(self):
        def tensor_map(u_grad, theta):
            return theta * u_grad
        return tensor_map

    def get_mass_map(self):
        def mass_map(u, x, theta):
            dx = x[0] - 0.5
            dy = x[1] - 0.5
            val = x[0]*np.sin(5.0*np.pi*x[1]) + np.exp(-(dx*dx + dy*dy)/0.02)
            return np.array([-val])
        return mass_map

# Mesh
Nx, Ny = 32, 32
mesh = fe.mesh.rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=1.0, domain_y=1.0)

# Create problem
problem = PoissonParametric(mesh=mesh, vec=1, dim=2, ele_type='QUAD4', location_fns=[])

# Periodic boundary conditions (left-right)
def left_boundary(point):
    return np.isclose(point[0], 0.0, atol=1e-5)

def right_boundary(point):
    return np.isclose(point[0], 1.0, atol=1e-5)

def mapping_x(point_A):
    return np.array([point_A[0] + 1.0, point_A[1]])

periodic_pairing = flat.pbc.PeriodicPairing(
    location_master=left_boundary,
    location_slave=right_boundary,
    mapping=mapping_x,
    vec=0
)

P = flat.pbc.prolongation_matrix([periodic_pairing], mesh, vec=1)

# Dirichlet boundary conditions (top-bottom = 0)
def bottom_boundary(point):
    return np.isclose(point[1], 0.0, atol=1e-5)

def top_boundary(point):
    return np.isclose(point[1], 1.0, atol=1e-5)

bc_config = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(bottom_boundary, 0, 0.0),
    fe.DCboundary.DirichletBCSpec(top_boundary, 0, 0.0),
])
bc = bc_config.create_bc(problem)

# Internal variables
theta = 1.0
theta_array = fe.internal_vars.InternalVars.create_uniform_volume_var(problem, theta)
internal_vars = fe.internal_vars.InternalVars(volume_vars=(theta_array,), surface_vars=())

# Solver
solver_options = fe.solver.SolverOptions(tol=1e-8, linear_solver="cg")
solver = fe.solver.create_solver(problem, bc, solver_options=solver_options, iter_num=1, P=P)

# Solve
initial_guess = np.zeros(problem.num_total_dofs_all_vars)
sol_full = solver(internal_vars, initial_guess)

# Save
fe.utils.save_sol(mesh, "periodic_poisson.vtu", point_infos=[("u", sol_full.reshape(-1, 1))])
```

## Vector Problems

For vector problems, apply periodicity to each component:

```python
pairings = [
    flat.pbc.PeriodicPairing(left, right, mapping_x, vec=0),
    flat.pbc.PeriodicPairing(left, right, mapping_x, vec=1),
    flat.pbc.PeriodicPairing(left, right, mapping_x, vec=2),
]
P = flat.pbc.prolongation_matrix(pairings, mesh, vec=3)
```
