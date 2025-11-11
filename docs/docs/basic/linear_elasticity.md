# Linear Elasticity Problem

This tutorial demonstrates solving a linear elasticity problem using FEAX. We consider a cantilever beam subjected to a traction force on one end while the other end is fixed.

## Problem Description

Consider a rectangular beam with length $L = 100$, width $W = 10$, and height $H = 10$. The beam is fixed at $x = 0$ and subjected to a downward traction force at $x = L$. The governing equation is:

$$
-\nabla \cdot \boldsymbol{\sigma}(\mathbf{u}) = \mathbf{0} \quad \text{in } \Omega
$$

where $\boldsymbol{\sigma}$ is the stress tensor and $\mathbf{u}$ is the displacement field. For linear elastic materials, the constitutive relation is:

$$
\boldsymbol{\sigma} = \lambda \text{tr}(\boldsymbol{\epsilon}) \mathbf{I} + 2\mu \boldsymbol{\epsilon}
$$

where $\boldsymbol{\epsilon} = \frac{1}{2}(\nabla \mathbf{u} + \nabla \mathbf{u}^T)$ is the strain tensor, and $\lambda$ and $\mu$ are the Lamé parameters:

$$
\mu = \frac{E}{2(1+\nu)}, \quad \lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}
$$

with Young's modulus $E = 70 \times 10^3$ and Poisson's ratio $\nu = 0.3$.

## Mesh Generation

Create a hexahedral mesh using the built-in mesh generator:

```python
import feax as fe
import jax.numpy as np

L, W, H = 100, 10, 10
box_size = (L, W, H)
mesh = fe.mesh.box_mesh(box_size, mesh_size=1)
```

The `box_mesh` function generates a structured hexahedral mesh with element size approximately equal to `mesh_size`. The resulting mesh uses HEX8 (8-node hexahedral) elements.

## Problem Definition

Define the linear elasticity problem by implementing the `get_tensor_map` method, which returns the stress tensor as a function of the displacement gradient:

```python
E = 70e3
nu = 0.3

class LinearElasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, *args):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress
```

The `get_tensor_map` method returns a function that takes the displacement gradient `u_grad` and returns the stress tensor. Additional arguments in `*args` can be used for internal variables such as spatially varying material properties.

### Surface Traction

Apply a traction force on the right face by implementing `get_surface_maps`:

```python
    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return np.array([0., 0., traction_mag])
        return [surface_map]
```

The surface map returns a traction vector as a function of displacement `u`, position `x`, and internal surface variables (here, `traction_mag`). The traction is applied in the negative $z$-direction.

Create the problem instance:

```python
right = lambda point: np.isclose(point[0], L, 1e-5)
problem = LinearElasticity(mesh, vec=3, dim=3, location_fns=[right])
```

The `location_fns` argument specifies which boundaries have surface tractions. The order corresponds to the surface maps returned by `get_surface_maps`.

## Boundary Conditions

Fix all displacement components at the left face:

```python
left = lambda point: np.isclose(point[0], 0., 1e-5)

left_fix = fe.DCboundary.DirichletBCSpec(
    location=left,
    component="all",
    value=0.
)
bc_config = fe.DCboundary.DirichletBCConfig([left_fix])
bc = bc_config.create_bc(problem)
```

The `DirichletBCSpec` defines a Dirichlet boundary condition at a specified location. The `component` parameter can be:
- `"all"`: Apply to all components
- `0`, `1`, `2`: Apply to specific $x$, $y$, or $z$ component

## Solver Configuration

FEAX provides iterative linear solvers for large-scale problems. Configure the solver with:

```python
solver_option = fe.solver.SolverOptions(linear_solver="bicgstab")
solver = fe.solver.create_solver(problem, bc, solver_option, iter_num=1)
```

### Available Linear Solvers

FEAX implements the following Krylov subspace methods:

- **`"bicgstab"`** (BiConjugate Gradient Stabilized): Suitable for general non-symmetric systems. Recommended for most applications.
- **`"cg"`** (Conjugate Gradient): For symmetric positive definite systems. Faster convergence but requires symmetry.
- **`"gmres"`** (Generalized Minimal Residual): For general systems with memory of previous iterations.

### Solver Options

Additional options can be specified:

```python
solver_option = fe.solver.SolverOptions(
    linear_solver="bicgstab",
    tol=1e-8,                          # Convergence tolerance
    atol=1e-8,                         # Absolute tolerance
    max_iter=1000,                     # Maximum iterations
    use_jacobi_preconditioner=False,   # Enable diagonal preconditioning
    jacobi_shift=1e-12                 # Regularization for preconditioner
)
```

The parameter `iter_num=1` indicates a linear problem solved in a single Newton iteration. For nonlinear problems, omit this parameter to enable Newton iterations.

## Internal Variables

Set the traction magnitude using internal variables:

```python
traction = 1.0
traction_array = fe.internal_vars.InternalVars.create_uniform_surface_var(problem, traction)

internal_vars = fe.internal_vars.InternalVars(
    volume_vars=(),
    surface_vars=[(traction_array,)]
)
```

Internal variables separate the problem structure from parameter values, enabling efficient parameter studies and gradient-based optimization. Surface variables must be provided as a list of tuples, with each tuple corresponding to a surface defined in `location_fns`.

## Solving the Problem

Solve the system:

```python
initial = fe.utils.zero_like_initial_guess(problem, bc)
solution = solver(internal_vars, initial)
```

The initial guess is constructed to satisfy the boundary conditions. For linear problems with `iter_num=1`, the initial guess does not affect the solution but is required by the solver interface.

Extract the displacement field:

```python
sol_unflat = problem.unflatten_fn_sol_list(solution)
displacement = sol_unflat[0]
```

The `unflatten_fn_sol_list` method converts the flat solution vector into a list of solution fields. For single-variable problems, the displacement is `sol_unflat[0]`.

## Visualization

Save the solution in VTK format for visualization in ParaView:

```python
fe.utils.save_sol(
    mesh=mesh,
    sol_file="displacement.vtu",
    point_infos=[("displacement", displacement)]
)
```

The `point_infos` parameter accepts a list of `(name, data)` tuples for point-based data. For cell-based data, use the `cell_infos` parameter.

## Complete Code

```python
import feax as fe
import jax.numpy as np

# Material and loading parameters
E = 70e3
nu = 0.3
traction = 1.0

# Mesh
L, W, H = 100, 10, 10
mesh = fe.mesh.box_mesh((L, W, H), mesh_size=1)

# Boundary location functions
left = lambda point: np.isclose(point[0], 0., 1e-5)
right = lambda point: np.isclose(point[0], L, 1e-5)

# Problem definition
class LinearElasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, *args):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress

    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return np.array([0., 0., traction_mag])
        return [surface_map]

problem = LinearElasticity(mesh, vec=3, dim=3, location_fns=[right])

# Boundary conditions
bc_config = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(left, "all", 0.)
])
bc = bc_config.create_bc(problem)

# Solver
solver_option = fe.solver.SolverOptions(linear_solver="bicgstab")
solver = fe.solver.create_solver(problem, bc, solver_option, iter_num=1)

# Internal variables
traction_array = fe.internal_vars.InternalVars.create_uniform_surface_var(problem, traction)
internal_vars = fe.internal_vars.InternalVars(volume_vars=(), surface_vars=[(traction_array,)])

# Solve
initial = fe.utils.zero_like_initial_guess(problem, bc)
solution = solver(internal_vars, initial)

# Extract displacement
displacement = problem.unflatten_fn_sol_list(solution)[0]

# Save
fe.utils.save_sol(mesh, "displacement.vtu", point_infos=[("displacement", displacement)])
```

## Further Reading

- Explore spatially varying material properties using `InternalVars.create_node_var_from_fn`
- Compute gradients through the solver using `jax.grad` for optimization
- Implement nonlinear constitutive laws by modifying the stress function
