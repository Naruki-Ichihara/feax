# Overview

This page summarizes FEAX's core workflow and key concepts. Every FEAX simulation follows the same pattern: **Mesh → Problem → Boundary Conditions → Internal Variables → Solver → Solution**.

## Workflow at a Glance

```python
import feax as fe
import jax
import jax.numpy as np

# 1. Mesh
mesh = fe.mesh.box_mesh((10, 5, 5), mesh_size=1)

# 2. Problem (constitutive law)
class MyProblem(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, *args):
            ...
            return sigma
        return stress

problem = MyProblem(mesh, vec=3, dim=3)

# 3. Boundary conditions
bc = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(location_fn, component="all", value=0.)
]).create_bc(problem)

# 4. Internal variables (parameters)
internal_vars = fe.InternalVars(volume_vars=(), surface_vars=[])

# 5. Solver
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(),
    iter_num=1, internal_vars=internal_vars)
initial = fe.zero_like_initial_guess(problem, bc)

# 6. Solve
sol = solver(internal_vars, initial)
```

## Problem Definition

A `Problem` subclass defines the physics by overriding one or more methods. The `Problem` constructor takes:

```python
problem = MyProblem(
    mesh,                     # Mesh or List[Mesh] for multi-variable
    vec=3,                    # DOFs per node (3 for 3D displacement)
    dim=3,                    # Spatial dimension (2 or 3)
    ele_type='HEX8',          # Element type (see table below)
    location_fns=[right],     # Boundaries with surface loads
    matrix_view='FULL',       # 'FULL', 'UPPER', or 'LOWER'
    additional_info=(E, nu),  # Extra args passed to custom_init()
)
```

### Supported Element Types

| Element | Type string | Dimension | Nodes |
|---|---|---|---|
| Hexahedron (linear) | `HEX8` | 3D | 8 |
| Hexahedron (quadratic) | `HEX27` | 3D | 27 |
| Tetrahedron (linear) | `TET4` | 3D | 4 |
| Tetrahedron (quadratic) | `TET10` | 3D | 10 |
| Quadrilateral (linear) | `QUAD4` | 2D | 4 |
| Quadrilateral (quadratic) | `QUAD9` | 2D | 9 |
| Triangle (linear) | `TRI3` | 2D | 3 |
| Triangle (quadratic) | `TRI6` | 2D | 6 |

### Physics Methods

Override these methods to define the constitutive law and loading:

| Method | Signature | Use case |
|---|---|---|
| `get_tensor_map()` | `(u_grad, *iv) → σ` | Stress tensor from displacement gradient |
| `get_energy_density()` | `(u_grad) → scalar` | Strain energy density (stress derived via `jax.grad`) |
| `get_mass_map()` | `(u, x, *iv) → f` | Body forces / reaction terms (no gradient) |
| `get_surface_maps()` | `(u, x, *iv) → t` | Surface tractions (Neumann BCs) |
| `get_weak_form()` | `(vals, grads, x, *iv) → (mass, grad)` | Multi-variable coupled physics |
| `get_surface_weak_forms()` | `(vals, x, *iv) → tractions` | Multi-variable surface loads |
| `custom_init(*args)` | — | Custom setup using `additional_info` |

The `*iv` arguments are internal variables (volume or surface), passed automatically by the assembler.

### Single-Variable Problems

For single-variable problems (e.g., displacement only), use `get_tensor_map()` or `get_energy_density()`:

**Stress-based** — return the stress tensor directly:

```python
class LinearElasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, *args):
            eps = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(eps) * np.eye(self.dim) + 2 * mu * eps
        return stress
```

**Energy-based** — return the scalar energy density $\psi(\nabla\mathbf{u})$. The stress tensor is derived automatically via `jax.grad`:

```python
class Elasticity(fe.problem.Problem):
    def get_energy_density(self):
        def psi(u_grad):
            eps = 0.5 * (u_grad + u_grad.T)
            return 0.5 * lmbda * np.trace(eps)**2 + mu * np.sum(eps * eps)
        return psi
```

`get_energy_density()` works with **both** solver paths:

- **Assembly path** — when `get_tensor_map()` returns `None`, the assembler automatically computes `σ = jax.grad(ψ)` and uses it for residual/Jacobian assembly, exactly as if you had defined `get_tensor_map()` yourself.
- **Matrix-free path** — `create_energy_fn(problem)` integrates $\psi$ over the domain to build the total energy function for `MatrixFreeOptions`.

If both `get_tensor_map()` and `get_energy_density()` are defined, `get_tensor_map()` takes precedence for the assembly path.

### Surface Loads

Define `get_surface_maps()` for Neumann boundary conditions. Each function corresponds to a surface specified in `location_fns`:

```python
class BeamWithTraction(fe.problem.Problem):
    def get_tensor_map(self):
        ...

    def get_surface_maps(self):
        def traction(u, x, magnitude):
            return np.array([0., 0., magnitude])
        return [traction]  # one per location_fn

# location_fns=[right] means traction is applied on the right face
problem = BeamWithTraction(mesh, vec=3, dim=3, location_fns=[right])
```

The third argument `magnitude` comes from the surface internal variables.

### Multi-Variable Problems

For coupled multi-physics (e.g., Stokes flow, Cahn-Hilliard), use `get_weak_form()` with multiple meshes/variables.

#### `get_weak_form()` Interface

The weak form function operates at a **single quadrature point** and is automatically `jax.vmap`-ed over all quadrature points by the framework. Its signature is:

```python
def weak_form(vals, grads, x, *internal_vars):
    ...
    return mass_terms, grad_terms
```

**Input arguments:**

| Argument | Shape | Description |
|---|---|---|
| `vals[i]` | `(vec_i,)` | Interpolated solution of variable $i$ at the quadrature point |
| `grads[i]` | `(vec_i, dim)` | Gradient of variable $i$ at the quadrature point |
| `x` | `(dim,)` | Physical coordinate of the quadrature point |
| `*internal_vars` | scalar (interpolated) | Volume internal variables, interpolated to the quadrature point |

**Return values:**

| Return | Shape | Integrated as |
|---|---|---|
| `mass_terms[i]` | `(vec_i,)` | $\int \text{mass\_terms}_i \cdot v_i \, d\Omega$ |
| `grad_terms[i]` | `(vec_i, dim)` | $\int \text{grad\_terms}_i : \nabla v_i \, d\Omega$ |

Both `mass_terms` and `grad_terms` are lists with one entry per variable. The framework handles:
1. Interpolating the solution and its gradient from nodal values to quadrature points
2. Interpolating internal variables (node-based or cell-based) to quadrature points
3. Calling the weak form at each quadrature point (via `vmap`)
4. Integrating the returned terms with the appropriate test functions and weights

#### Example: Cahn-Hilliard

```python
class CahnHilliard(fe.problem.Problem):
    def get_weak_form(self):
        def weak_form(vals, grads, x, c_old):
            c, mu = vals[0], vals[1]        # solution variables
            grad_c, grad_mu = grads[0], grads[1]
            # mass_terms[i] → ∫ · v_i dΩ
            mass_terms = [(c - c_old) / dt, mu - (c**3 - c)]
            # grad_terms[i] → ∫ · ∇v_i dΩ
            grad_terms = [M * grad_mu, -kappa * grad_c]
            return mass_terms, grad_terms
        return weak_form

problem = CahnHilliard(
    mesh=[mesh, mesh],       # one mesh per variable
    vec=[1, 1],              # scalar c and scalar μ
    dim=2,
    ele_type=['QUAD4', 'QUAD4'],
)
```

#### `get_surface_weak_forms()` Interface

For multi-variable surface loads, override `get_surface_weak_forms()`. Each function operates at a single surface quadrature point:

```python
def surface_weak_form(vals, x, *internal_vars):
    ...
    return tractions  # list of (vec_i,) arrays
```

| Argument | Shape | Description |
|---|---|---|
| `vals[i]` | `(vec_i,)` | Interpolated solution of variable $i$ at the surface point |
| `x` | `(dim,)` | Physical coordinate of the surface quadrature point |
| `tractions[i]` | `(vec_i,)` | Surface load integrated as $\int t_i \cdot v_i \, d\Gamma$ |

```python
class StokesProblem(fe.problem.Problem):
    def get_surface_weak_forms(self):
        def inlet(vals, x):
            return [np.array([p_in, 0.]), np.zeros(1)]  # [velocity_traction, pressure_traction]
        return [inlet]  # one per location_fn
```

Multi-variable problems require `get_weak_form()` (or `get_universal_kernel()`) — the single-variable methods (`get_tensor_map`, etc.) are not used and will produce a warning if defined.

### Using `additional_info` and `custom_init`

Pass extra parameters at construction time via `additional_info` and process them in `custom_init()`:

```python
class ParametricProblem(fe.problem.Problem):
    def custom_init(self, E, nu):
        self.mu = E / (2 * (1 + nu))
        self.lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

    def get_tensor_map(self):
        def stress(u_grad, *args):
            eps = 0.5 * (u_grad + u_grad.T)
            return self.lmbda * np.trace(eps) * np.eye(self.dim) + 2 * self.mu * eps
        return stress

problem = ParametricProblem(mesh, vec=3, dim=3, additional_info=(210e3, 0.3))
```

## Boundary Conditions

### Dirichlet BCs

Specified declaratively with `DirichletBCSpec`:

```python
bc_config = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(location=left_face, component="all", value=0.),
    fe.DCboundary.DirichletBCSpec(location=top_face,  component="y",   value=1.0),
])
bc = bc_config.create_bc(problem)
```

- **`location`**: function `point → bool` identifying boundary nodes (e.g., `lambda p: np.isclose(p[0], 0.)`)
- **`component`**: `"all"`, `"x"`, `"y"`, `"z"` (or integer `0`, `1`, `2`)
- **`value`**: prescribed value (float)

### Neumann BCs

Defined via `get_surface_maps()` in the Problem class (see [Surface Loads](#surface-loads) above).

### Multipoint Constraints (Prolongation Matrix `P`)

FEAX supports multipoint constraints via a prolongation matrix `P` that maps a reduced DOF set to the full DOF set. When `P` is provided to `create_solver`, the solver operates in the reduced space, enforcing the constraints exactly.

Periodic boundary conditions are a common application — `P` ties DOFs on opposite faces of a unit cell so that $\mathbf{u}^+ - \mathbf{u}^- = \bar{\boldsymbol{\varepsilon}} \cdot \Delta\mathbf{x}$:

```python
from feax.flat.pbc import prolongation_matrix

P = prolongation_matrix(mesh, problem)
solver = fe.create_solver(problem, bc, P=P,
    solver_options=fe.IterativeSolverOptions())
```

The reduced system is solved matrix-free (matvec via `P^T K P`), so `P` requires `IterativeSolverOptions`.

See [Periodic Boundary Conditions](../advanced/periodic_boundary_conditions.md) for details.

## Internal Variables

`InternalVars` separates problem structure from parameter values. This separation is what makes FEAX differentiable — parameters flow through the solver as JAX arrays, enabling `jax.grad` and `jax.vmap`.

### Creation Methods

| Method | Shape | Use case |
|---|---|---|
| `create_node_var(problem, value)` | `(num_nodes,)` | Node-based properties (most efficient) |
| `create_cell_var(problem, value)` | `(num_cells,)` | Element-wise properties (e.g., topology density) |
| `create_uniform_surface_var(problem, value)` | `(num_faces, num_quad)` | Uniform surface loads |
| `create_node_var_from_fn(problem, fn)` | `(num_nodes,)` | Spatially varying node properties |
| `create_cell_var_from_fn(problem, fn)` | `(num_cells,)` | Spatially varying element properties |
| `create_spatially_varying_surface_var(problem, fn)` | `(num_faces, num_quad)` | Spatially varying surface loads |

### Structure

```python
# Volume variables → passed as *args to get_tensor_map() stress function
E  = fe.InternalVars.create_node_var(problem, 210e3)
nu = fe.InternalVars.create_node_var(problem, 0.3)

# Surface variables → passed as *args to get_surface_maps() traction function
traction = fe.InternalVars.create_uniform_surface_var(problem, 1e-3)

internal_vars = fe.InternalVars(
    volume_vars=(E, nu),            # tuple of arrays
    surface_vars=[(traction,)]      # list of tuples, one per location_fn
)
```

The stress function receives volume variables as extra arguments:

```python
def get_tensor_map(self):
    def stress(u_grad, E, nu):  # E, nu come from volume_vars
        mu = E / (2 * (1 + nu))
        lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        eps = 0.5 * (u_grad + u_grad.T)
        return lmbda * np.trace(eps) * np.eye(self.dim) + 2 * mu * eps
    return stress
```

### Differentiability

Since `InternalVars` is a JAX pytree, you can differentiate with respect to any parameter:

```python
def objective(iv):
    sol = solver(iv, initial)
    return np.sum(sol ** 2)

grad_fn = jax.grad(objective)
grads = grad_fn(internal_vars)
# grads.volume_vars[0] → sensitivity w.r.t. E at each node
```

## Solvers

### Solver Options

FEAX provides three solver paths via `fe.create_solver`:

| Solver Options | Method | Best for |
|---|---|---|
| `fe.DirectSolverOptions()` | Sparse direct (cuDSS on GPU, spsolve on CPU) | Small-medium problems |
| `fe.IterativeSolverOptions()` | Iterative (CG/BiCGSTAB/GMRES) | Large problems, periodic BCs |
| `fe.MatrixFreeOptions()` | Matrix-free Newton (JVP-based) | Custom energy, no assembly needed |

### `iter_num` Parameter

The `iter_num` parameter controls the Newton iteration strategy:

| `iter_num` | Behavior | `jax.vmap` compatible |
|---|---|---|
| `1` | Single linear solve (linear problems) | Yes |
| `> 1` | Fixed N Newton iterations | Yes |
| `None` (default) | Adaptive Newton with while loop | No |

```python
# Linear problem — one solve, vmappable
solver = fe.create_solver(problem, bc, solver_options=fe.DirectSolverOptions(),
    iter_num=1, internal_vars=internal_vars)

# Nonlinear problem — adaptive Newton (not vmappable)
solver = fe.create_solver(problem, bc, solver_options=fe.DirectSolverOptions(),
    internal_vars=internal_vars)

# Nonlinear problem — fixed iterations (vmappable)
solver = fe.create_solver(problem, bc, solver_options=fe.DirectSolverOptions(),
    iter_num=10, internal_vars=internal_vars)
```

### Matrix-Free Solver

For problems with custom energy contributions (e.g., cohesive zones), use `MatrixFreeOptions`:

```python
from feax.solvers.matrix_free import MatrixFreeOptions, LinearSolverOptions, create_energy_fn

elastic_energy = create_energy_fn(problem)  # from get_energy_density()

def total_energy(u_flat, delta_max):
    return elastic_energy(u_flat) + cohesive_energy(u_flat, delta_max)

solver = fe.create_solver(problem, bc,
    solver_options=MatrixFreeOptions(
        newton_tol=1e-8,
        newton_max_iter=200,
        linear_solver=LinearSolverOptions(solver='cg', atol=1e-8),
    ),
    energy_fn=total_energy)
```

The tangent operator is computed via `jax.jvp` (forward-mode AD) of the residual — no sparse matrix is ever assembled.

### Solver Calling Convention

All solvers share the same signature:

```python
sol = solver(internal_vars, initial_guess)
```

This uniform interface enables `jax.jit`, `jax.grad`, and `jax.vmap` to work with any solver path.

### `MatrixView` for Symmetric Problems

For symmetric problems, use `matrix_view='UPPER'` to reduce memory by ~50% and enable optimized solvers (Cholesky):

```python
problem = MyProblem(mesh, vec=3, dim=3, matrix_view='UPPER')
```

## JAX Transformations

FEAX solvers are compatible with JAX's functional transformations:

```python
import jax

# JIT compilation — eliminates Python overhead
fast_solver = jax.jit(solver)
sol = fast_solver(internal_vars, initial)

# Differentiation — gradients through the entire solve
grad_fn = jax.grad(lambda iv: np.sum(solver(iv, initial)**2))
grads = grad_fn(internal_vars)

# Vectorization — batch parameter studies (requires iter_num != None)
batched_solver = jax.vmap(solver, in_axes=(0, None))
sols = batched_solver(batched_internal_vars, initial)
```

**Notes:**
- `jax.grad` is supported (first-order). `jax.hessian` (second-order) is not, because solvers use `custom_vjp` internally.
- `jax.vmap` requires fixed iteration count (`iter_num=1` or `iter_num=N`). Adaptive Newton (`iter_num=None`) uses a while loop and is not vmappable.

## Post-Processing

### Unflattening the Solution

The solver returns a flat DOF vector. Use `unflatten_fn_sol_list` to reshape it per variable:

```python
sol_list = problem.unflatten_fn_sol_list(sol)
displacement = sol_list[0]  # shape (num_nodes, vec)
```

For multi-variable problems, `sol_list[i]` gives the i-th variable's solution.

### VTK Output

Save results as VTK files for visualization in ParaView:

```python
fe.utils.save_sol(
    mesh=mesh,
    sol_file='output.vtu',
    point_infos=[("displacement", displacement)]
)
```

## Next Steps

- **[Linear Elasticity](../basic/linear_elasticity.md)** — full walkthrough of a first problem
- **[JIT Transform](../basic/jit_transform.md)** — accelerate solves with `jax.jit`
- **[Vectorization Transform](../basic/vmap_transform.md)** — batch parameter studies with `jax.vmap`
- **[Hyperelasticity](../basic/hyperelasticity.md)** — nonlinear problems with energy-based formulation
- **[Cohesive Fracture](../advanced/cohesive_fracture.md)** — matrix-free solver with cohesive zone model
