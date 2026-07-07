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
traced_params = fe.TracedParams(volume_vars=(), surface_vars=[])

# 5. Solver
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(),
    linear=True, traced_params=traced_params)
initial = fe.zero_like_initial_guess(problem, bc)

# 6. Solve
sol = solver(traced_params, initial)
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
| `get_universal_kernel()` | element-level kernel (see below) | Custom quadrature, ANS, EAS, Hessian regularization |
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

`get_energy_density()` feeds the solver in two ways:

- **Residual/Jacobian assembly** — when `get_tensor_map()` returns `None`, the assembler automatically computes `σ = jax.grad(ψ)` and uses it for residual and Jacobian assembly, exactly as if you had defined `get_tensor_map()` yourself.
- **Scalar energy evaluation** — `fe.create_energy_fn(problem)` integrates $\psi$ over the domain to build the total stored-energy function, useful for objective evaluation (e.g. compliance) and post-processing.

If both `get_tensor_map()` and `get_energy_density()` are defined, `get_tensor_map()` takes precedence for assembly.

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
def weak_form(vals, grads, x, *traced_params):
    ...
    return mass_terms, grad_terms
```

**Input arguments:**

| Argument | Shape | Description |
|---|---|---|
| `vals[i]` | `(vec_i,)` | Interpolated solution of variable $i$ at the quadrature point |
| `grads[i]` | `(vec_i, dim)` | Gradient of variable $i$ at the quadrature point |
| `x` | `(dim,)` | Physical coordinate of the quadrature point |
| `*traced_params` | scalar (interpolated) | Volume internal variables, interpolated to the quadrature point |

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
def surface_weak_form(vals, x, *traced_params):
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

#### `get_universal_kernel()` — element-level control

`get_weak_form()` operates at a single quadrature point and lets the framework handle interpolation and integration. `get_universal_kernel()` drops one level lower: it receives the full **element** data — all nodal values, all quadrature points, shape function gradients and weights — and returns the element residual vector directly. The framework then handles global assembly and automatic differentiation.

Use `get_universal_kernel()` when the physics at a quadrature point cannot be expressed independently of the other quadrature points in the same element, for example:

- custom quadrature rules (different from the element's default Gauss points)
- assumed natural strain (ANS) tying across quadrature points
- element-level static condensation (enhanced assumed strain, EAS)
- element-level Hessian regularization (biharmonic, see the Third-Medium Contact example)

##### Kernel signature

The kernel function receives the following arguments per element:

```python
def kernel(
    cell_sol_flat,          # (num_dofs_per_cell,)  nodal DOF values, flattened
    physical_quad_points,   # (num_quads, dim)       physical coords of native Gauss pts
    cell_shape_grads,       # (num_quads, num_nodes, dim)  dN/dx at native Gauss pts
    cell_JxW,               # (num_quads,)           det(J) × weight at native Gauss pts
    cell_v_grads_JxW,       # (num_quads, num_nodes, vec, dim)  pre-weighted test grad
    *cell_internal_vars,    # per-cell internal variables (gathered from TracedParams)
):
    ...
    return residual_flat    # (num_dofs_per_cell,)  element residual, flattened
```

The kernel is **vmapped over all cells** by the framework: it sees one element at a time. `cell_sol_flat` is the concatenation of all nodal DOF values for that element; use `self.unflatten_fn_dof(cell_sol_flat)` to recover the per-variable, per-node arrays.

##### Return value

Return a 1-D JAX array of length `num_dofs_per_cell` — the element's contribution to the global residual. Use `jax.flatten_util.ravel_pytree` to flatten structured arrays before returning:

```python
import jax.flatten_util

R = ...  # (num_nodes, vec) element residual
return jax.flatten_util.ravel_pytree(R)[0]
```

The global stiffness matrix (Jacobian) is assembled automatically via `jax.jacrev` on the kernel, so no manual linearisation is required.

##### Accessing nodal data

`cell_sol_flat` contains all element DOFs in a flat vector. `unflatten_fn_dof` splits it back into per-variable arrays:

```python
class MyProblem(fe.Problem):
    def get_universal_kernel(self):
        unflatten = self.unflatten_fn_dof

        def kernel(cell_sol_flat, physical_quad_points,
                   cell_shape_grads, cell_JxW, cell_v_grads_JxW,
                   *cell_internal_vars):
            cell_sol_list = unflatten(cell_sol_flat)
            cell_sol = cell_sol_list[0]   # (num_nodes, vec) for variable 0
            ...
        return kernel
```

##### Per-cell internal variables

`cell_internal_vars` contains **one entry per variable in `TracedParams.volume_vars`**, already gathered to the element level:

| Global shape | Per-cell shape passed to kernel |
|---|---|
| `(num_nodes,)` | `(num_nodes_per_elem,)` — gathered via connectivity |
| `(num_cells,)` | scalar (the cell's value) |
| `(num_cells, num_quads, ...)` | `(num_quads, ...)` — the element's rows |
| Any other shape | passed through unchanged |

This makes it straightforward to pass, for example, per-cell node coordinates — which would be the physical positions of the element's nodes — as a volume variable, and use them to build a custom isoparametric Jacobian inside the kernel.

##### Minimal example

A linear-elastic kernel that recomputes the isoparametric Jacobian from per-cell node coordinates stored as a volume internal variable, using a custom quadrature rule precomputed at construction time:

```python
import jax
import jax.numpy as np
import jax.flatten_util
import feax as fe
from feax.traced_params import TracedParams

class CustomQuadratureProblem(fe.Problem):
    def custom_init(self, dNdxi, weights, C):
        # dNdxi  : (nq, num_nodes, 3)  reference shape gradients at custom quad pts
        # weights: (nq,)               quadrature weights (sum to reference-cell volume)
        # C      : (3,3,3,3)           stiffness tensor
        self._dNdxi = dNdxi
        self._w = weights
        self._C = C

    def get_universal_kernel(self):
        dNdxi = self._dNdxi      # (nq, num_nodes, 3)
        w = self._w              # (nq,)
        C = self._C              # (3,3,3,3)
        unflatten = self.unflatten_fn_dof

        def kernel(cell_sol_flat, physical_quad_points,
                   cell_shape_grads, cell_JxW, cell_v_grads_JxW,
                   cell_nodes):
            # cell_nodes: (num_nodes, 3) — per-cell physical coords from TracedParams
            cell_sol = unflatten(cell_sol_flat)[0]          # (num_nodes, 3)

            # Isoparametric Jacobian and its inverse at each custom quad point
            J    = np.einsum("ai,qaI->qiI", cell_nodes, dNdxi)  # (nq, 3, 3)
            Jinv = np.linalg.inv(J)
            detJ = np.linalg.det(J)

            # Physical shape gradients dN/dx
            dNdx = np.einsum("qaI,qIi->qai", dNdxi, Jinv)       # (nq, num_nodes, 3)

            # Small-strain stress
            grad_u = np.einsum("ai,qaj->qij", cell_sol, dNdx)   # (nq, 3, 3)
            eps    = 0.5 * (grad_u + np.transpose(grad_u, (0, 2, 1)))
            sigma  = np.einsum("ijkl,qkl->qij", C, eps)          # (nq, 3, 3)

            # Element residual R_ai = Σ_q (w · detJ)_q σ_ij (dN_a/dx_j)_q
            JxW = w * detJ
            R = np.einsum("q,qij,qaj->ai", JxW, sigma, dNdx)    # (num_nodes, 3)
            return jax.flatten_util.ravel_pytree(R)[0]

        return kernel


# Construction: pass custom quad data via additional_info,
# node coords via TracedParams
problem = CustomQuadratureProblem(
    mesh, vec=3, dim=3, ele_type="HEX8",
    additional_info=(dNdxi, weights, C),
)
cell_nodes = np.asarray(mesh.points)[np.asarray(mesh.cells)]  # (nc, nn, 3)
iv = TracedParams(volume_vars=(cell_nodes,))
```

:::note
`get_universal_kernel()` can coexist with `get_tensor_map()` for single-variable problems: the assembler adds their contributions. For multi-variable problems only `get_universal_kernel()` (or `get_weak_form()`) is used.
:::

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
from feax.flat.pbc import PeriodicPairing, prolongation_matrix

pairings = [PeriodicPairing(location_master=left, location_slave=right,
                            mapping=mapping_x, vec=0)]
P = prolongation_matrix(pairings, mesh, vec=1)
solver = fe.create_solver(problem, bc, P=P,
    solver_options=fe.KrylovSolverOptions())
```

With `KrylovSolverOptions` the reduced system is solved matrix-free (matvec via `P^T K P`); with `DirectSolverOptions` or `AMGSolverOptions` the reduced operator `PᵀJP` is assembled sparsely and factorized (or used to build the AMG hierarchy).

See [Periodic Boundary Conditions](../advanced/periodic_boundary_conditions.md) for details.

## Internal Variables

`TracedParams` separates problem structure from parameter values. This separation is what makes FEAX differentiable — parameters flow through the solver as JAX arrays, enabling `jax.grad` and `jax.vmap`.

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
E  = fe.TracedParams.create_node_var(problem, 210e3)
nu = fe.TracedParams.create_node_var(problem, 0.3)

# Surface variables → passed as *args to get_surface_maps() traction function
traction = fe.TracedParams.create_uniform_surface_var(problem, 1e-3)

traced_params = fe.TracedParams(
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

Since `TracedParams` is a JAX pytree, you can differentiate with respect to any parameter:

```python
def objective(iv):
    sol = solver(iv, initial)
    return np.sum(sol ** 2)

grad_fn = jax.grad(objective)
grads = grad_fn(traced_params)
# grads.volume_vars[0] → sensitivity w.r.t. E at each node
```

## Solvers

### Solver Options

FEAX has three solver-option classes for `fe.create_solver`. When `solver_options` is omitted, a direct solver is selected automatically (cuDSS on GPU, cholmod/umfpack/spsolve on CPU). See the [Solver Guide](./solver.md) for the full selection logic.

| Solver Options | Method | Operator | Best for |
|---|---|---|---|
| `fe.DirectSolverOptions()` | Sparse direct (cuDSS on GPU, cholmod/umfpack/spsolve on CPU) | Assembled CSR | Default; robust when memory permits |
| `fe.KrylovSolverOptions()` | Iterative (CG/BiCGSTAB/GMRES) | Matrix-free (residual JVP) | Large/memory-bound problems, periodic BCs |
| `fe.AMGSolverOptions()` | Krylov preconditioned by smoothed-aggregation AMG | Assembled CSR (hierarchy) + matrix-free Krylov | Large scalar-elliptic problems; elasticity via `near_nullspace="rigid_body"`. Requires `feax[amg]` |

### `linear` Flag

The `linear` flag selects the solve path:

| `linear` | Behavior |
|---|---|
| `True` | A single linear solve (linear problems). |
| `False` (default) | Adaptive Newton iteration with line search. |

```python
# Linear problem — one solve
solver = fe.create_solver(problem, bc, solver_options=fe.DirectSolverOptions(),
    linear=True, traced_params=traced_params)

# Nonlinear problem — adaptive Newton (the default)
solver = fe.create_solver(problem, bc, solver_options=fe.DirectSolverOptions(),
    traced_params=traced_params)
```

Both paths compose with `jax.jit`, `jax.vmap`, and `jax.grad`. The Newton forward is a traced `jax.lax.while_loop` — one Newton step per loop body, with no `pure_callback` node — so adaptive convergence and line search jit and vmap natively.

### Custom Residual Contributions

For an additional residual term that does not come from the standard element weak form (e.g. a cohesive-zone traction), pass `extra_residual_fn`. With `KrylovSolverOptions` FEAX assembles the bulk Jacobian and applies the extra term's tangent matrix-free via `jax.jvp`; with `DirectSolverOptions` the extra term's sparse Jacobian is detected and assembled onto the merged CSR pattern so the direct solver factorizes the exact combined tangent:

```python
def cohesive_residual(u_flat):
    return jax.grad(lambda u: cohesive_energy(u, delta_max))(u_flat)

solver = fe.create_solver(problem, bc,
    solver_options=fe.KrylovSolverOptions(solver='cg'),
    newton_options=fe.NewtonOptions(tol=1e-8, max_iter=200),
    extra_residual_fn=cohesive_residual,
    linear=False)
```

### Solver Calling Convention

All solvers share the same signature:

```python
sol = solver(traced_params, initial_guess)
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
sol = fast_solver(traced_params, initial)

# Differentiation — gradients through the entire solve
grad_fn = jax.grad(lambda iv: np.sum(solver(iv, initial)**2))
grads = grad_fn(traced_params)

# Vectorization — batch parameter studies
batched_solver = jax.vmap(solver, in_axes=(0, None))
sols = batched_solver(batched_internal_vars, initial)
```

**Notes:**
- `jax.grad` is supported (first-order). `jax.hessian` (second-order) is not, because solvers use `custom_vjp` internally.
- `jax.vmap` works for all solver paths — linear, Newton, and reduced. Batching is over BC **values** and parameters; BC **locations** must be identical within a batch (see the [Solver Guide](./solver.md)).

## Post-Processing

### Unflattening the Solution

The solver returns a `fe.Solution` — it behaves like the flat DOF vector (arithmetic, `np.asarray`, initial guess for the next solve) and carries the DOF layout, so no `Problem` is needed to interpret it:

```python
displacement = sol.field(0)          # shape (num_nodes, vec)
sol_flat = sol.dofs                  # the raw flat DOF vector
temperature = sol.node_var()         # (num_nodes,) — feed the next solve's TracedParams
```

For multi-variable problems, `sol.field(i)` gives the i-th variable's solution (equivalent to `problem.unflatten_fn_sol_list(sol)[i]`). Pass `return_solution=False` to `create_solver` to get the raw flat vector instead.

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

- **[Solver Guide](./solver.md)** — solver options, Newton settings, BC elimination, and incremental loading
- **[Linear Elasticity](../basic/linear_elasticity.md)** — full walkthrough of a first problem
- **[JIT Transform](../basic/jit_transform.md)** — accelerate solves with `jax.jit`
- **[Vectorization Transform](../basic/vmap_transform.md)** — batch parameter studies with `jax.vmap`
- **[Hyperelasticity](../basic/hyperelasticity.md)** — nonlinear problems with energy-based formulation
- **[Cohesive Fracture](../advanced/cohesive_fracture.md)** — matrix-free solver with cohesive zone model
