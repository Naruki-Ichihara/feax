# Linear Elasticity Problem

This tutorial demonstrates solving a linear elasticity problem using FEAX. We consider a cantilever beam subjected to a traction force on one end while the other end is fixed.

## Problem Description

Consider a rectangular beam with length $L = 100$, width $W = 10$, and height $H = 10$. The beam is fixed at $x = 0$ and subjected to a traction force at $x = L$. The governing equation is:

$$
-\nabla \cdot \boldsymbol{\sigma}(\mathbf{u}) = \mathbf{0} \quad \text{in } \Omega
$$

For linear elastic materials:

$$
\boldsymbol{\sigma} = \lambda \text{tr}(\boldsymbol{\epsilon}) \mathbf{I} + 2\mu \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} = \frac{1}{2}(\nabla \mathbf{u} + \nabla \mathbf{u}^T)
$$

with Young's modulus $E = 70 \times 10^3$ and Poisson's ratio $\nu = 0.3$.

## Mesh Generation

```python
import feax as fe
import jax.numpy as np

E        = 70e3
nu       = 0.3
traction = 1e-3
tol      = 1e-5

L, W, H = 100, 10, 10
mesh = fe.mesh.box_mesh((L, W, H), mesh_size=1)

left  = lambda point: np.isclose(point[0], 0., tol)
right = lambda point: np.isclose(point[0], L,  tol)
```

## Problem Definition

Implement `get_tensor_map` to return the stress function, and `get_surface_maps` to define the traction:

```python
class LinearElasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, *args):
            mu      = E / (2. * (1. + nu))
            lmbda   = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress

    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return np.array([0., 0., traction_mag])
        return [surface_map]

problem = LinearElasticity(mesh, vec=3, dim=3, location_fns=[right])
```

`location_fns` specifies which boundaries carry surface tractions, corresponding to the list returned by `get_surface_maps`.

## Boundary Conditions

```python
left_fix = fe.DCboundary.DirichletBCSpec(
    location=left,
    component="all",
    value=0.
)
bc_config = fe.DCboundary.DirichletBCConfig([left_fix])
bc = bc_config.create_bc(problem)
```

`component` can be `"all"` or a specific axis `"x"`, `"y"`, `"z"` (equivalently `0`, `1`, `2`).

## Internal Variables

Internal variables separate problem structure from parameter values, enabling efficient parameter studies and gradient-based optimization.

```python
traction_array = fe.InternalVars.create_uniform_surface_var(problem, traction)
internal_vars  = fe.InternalVars(
    volume_vars=(),
    surface_vars=[(traction_array,)]
)
```

Surface variables are provided as a list of tuples, with each tuple corresponding to a surface in `location_fns`.

## Solver

`DirectSolverOptions` automatically selects the best available backend (cuDSS on GPU, sparse direct on CPU):

```python
solver_opts = fe.DirectSolverOptions()
solver = fe.create_solver(
    problem, bc,
    solver_options=solver_opts,
    iter_num=1,
    internal_vars=internal_vars
)
initial = fe.zero_like_initial_guess(problem, bc)
```

`iter_num=1` solves a linear problem in a single Newton iteration. Omit it for nonlinear problems.

## Solving

```python
def solve_forward(iv):
    return solver(iv, initial)

sol          = solve_forward(internal_vars)
displacement = problem.unflatten_fn_sol_list(sol)[0]
```

## Visualization

Save the solution in VTK format for ParaView:

```python
import os

data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)

fe.utils.save_sol(
    mesh=mesh,
    sol_file=os.path.join(data_dir, 'vtk/u.vtu'),
    point_infos=[("displacement", displacement)]
)
```

## Complete Code

```python
import feax as fe
import jax.numpy as np
import os

E        = 70e3
nu       = 0.3
traction = 1e-3
tol      = 1e-5

L, W, H = 100, 10, 10
mesh = fe.mesh.box_mesh((L, W, H), mesh_size=1)

left  = lambda point: np.isclose(point[0], 0., tol)
right = lambda point: np.isclose(point[0], L,  tol)

class LinearElasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, *args):
            mu      = E / (2. * (1. + nu))
            lmbda   = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress

    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return np.array([0., 0., traction_mag])
        return [surface_map]

problem = LinearElasticity(mesh, vec=3, dim=3, location_fns=[right])

bc_config = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(location=left, component="all", value=0.)
])
bc = bc_config.create_bc(problem)

traction_array = fe.InternalVars.create_uniform_surface_var(problem, traction)
internal_vars  = fe.InternalVars(volume_vars=(), surface_vars=[(traction_array,)])

solver_opts = fe.DirectSolverOptions()
solver      = fe.create_solver(problem, bc, solver_options=solver_opts,
                               iter_num=1, internal_vars=internal_vars)
initial     = fe.zero_like_initial_guess(problem, bc)

sol          = solver(internal_vars, initial)
displacement = problem.unflatten_fn_sol_list(sol)[0]

data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)
fe.utils.save_sol(mesh=mesh, sol_file=os.path.join(data_dir, 'vtk/u.vtu'),
                  point_infos=[("displacement", displacement)])
```

## Further Reading

- [JIT Transform](./jit_transform.md) — accelerate repeated solves with `jax.jit`
- [Vectorization Transform](./vmap_transform.md) — batch parameter studies with `jax.vmap`
- [Nonlinear Hyperelasticity](./hyperelasticity.md) — nonlinear problems with Newton's method
