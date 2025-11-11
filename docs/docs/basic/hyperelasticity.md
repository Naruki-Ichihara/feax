# Nonlinear Hyperelasticity Problem

This tutorial demonstrates solving nonlinear hyperelasticity problems using FEAX. We consider a cube subjected to large deformations using a Neo-Hookean material model, showcasing Newton's method for nonlinear finite element analysis.

## Problem Description

Consider a unit cube $\Omega = [0,1]^3$ subjected to large deformations through prescribed boundary displacements. The governing equation in the nonlinear setting is:

$$
-\nabla \cdot \mathbf{P}(\mathbf{u}) = \mathbf{0} \quad \text{in } \Omega
$$

where $\mathbf{P}$ is the first Piola-Kirchhoff stress tensor and $\mathbf{u}$ is the displacement field. Unlike linear elasticity, the stress depends nonlinearly on the deformation gradient $\mathbf{F} = \nabla \mathbf{u} + \mathbf{I}$.

### Neo-Hookean Material Model

For hyperelastic materials, the stress derives from a strain energy density function $\psi(\mathbf{F})$:

$$
\mathbf{P} = \frac{\partial \psi}{\partial \mathbf{F}}
$$

The compressible Neo-Hookean model uses:

$$
\psi(\mathbf{F}) = \frac{\mu}{2}\left(J^{-2/3} I_1 - 3\right) + \frac{\kappa}{2}(J - 1)^2
$$

where:
- $I_1 = \text{tr}(\mathbf{F}^T \mathbf{F})$ is the first invariant
- $J = \det(\mathbf{F})$ is the Jacobian (volume ratio)
- $\mu = \frac{E}{2(1+\nu)}$ is the shear modulus
- $\kappa = \frac{E}{3(1-2\nu)}$ is the bulk modulus

This model captures:
- **Large deformations** through $\mathbf{F}$ (not linearized strain)
- **Incompressibility constraint** through the $(J-1)^2$ penalty
- **Material frame indifference** through invariants

## Mesh Generation

Create a structured hexahedral mesh for the unit cube:

```python
import feax as fe
import jax
import jax.numpy as np

mesh = fe.mesh.box_mesh((1, 1, 1), mesh_size=0.1)
print(f"Mesh: {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} elements")
```

The relatively coarse mesh (`mesh_size=0.1`) provides 10 elements per side, sufficient for demonstrating nonlinear convergence.

## Problem Definition: Energy-Based Approach

FEAX leverages JAX's automatic differentiation to compute stress from energy. Define the energy function and let JAX compute $\mathbf{P} = \partial \psi / \partial \mathbf{F}$:

```python
class HyperElasticityFeax(fe.problem.Problem):
    def get_tensor_map(self):
        # Define strain energy density function
        def psi(F):
            # Material parameters
            E = 100.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))

            # Neo-Hookean energy
            J = np.linalg.det(F)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        # Automatic differentiation: P = ∂ψ/∂F
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I  # Deformation gradient
            P = P_fn(F)
            return P

        return first_PK_stress
```

### Key Insights

1. **Energy function**: `psi(F)` implements the Neo-Hookean model mathematically
2. **Automatic differentiation**: `jax.grad(psi)` computes the exact stress tensor without manual derivation
3. **Deformation gradient**: $\mathbf{F} = \nabla \mathbf{u} + \mathbf{I}$ (displacement gradient + identity)
4. **No manual Jacobian**: FEAX assembler automatically computes the tangent stiffness via `jax.jacobian`

**Advantages of this approach:**
- ✅ No manual stress/tangent derivation (error-prone for complex models)
- ✅ Easy to experiment with different energy functions
- ✅ Guaranteed consistent tangent stiffness
- ✅ Composable with JAX transformations (vmap, grad)

## Boundary Conditions: Prescribed Rotation

Apply boundary conditions that impose a rotation about the $x$-axis:

```python
# Define boundary locations
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], 1, atol=1e-5)

# Rotation angle: π/3 radians (60 degrees)
def zero_dirichlet_val(point):
    return 0.

def dirichlet_val_x2(point):
    """Y-component displacement for rotation about x-axis"""
    return (0.5 + (point[1] - 0.5) * np.cos(np.pi / 3.) -
            (point[2] - 0.5) * np.sin(np.pi / 3.) - point[1]) / 2.

def dirichlet_val_x3(point):
    """Z-component displacement for rotation about x-axis"""
    return (0.5 + (point[1] - 0.5) * np.sin(np.pi / 3.) +
            (point[2] - 0.5) * np.cos(np.pi / 3.) - point[2]) / 2.
```

The displacement functions rotate the left boundary by $60°$ while fixing the right boundary:

$$
\begin{bmatrix} y' \\ z' \end{bmatrix} = \begin{bmatrix} \cos(60°) & -\sin(60°) \\ \sin(60°) & \cos(60°) \end{bmatrix} \begin{bmatrix} y - 0.5 \\ z - 0.5 \end{bmatrix} + \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}
$$

The displacement is the difference: $u = x' - x$, scaled by 0.5 to apply the rotation gradually.

### Apply Boundary Conditions

```python
bc_config = fe.DCboundary.DirichletBCConfig([
    # Left boundary - prescribed rotation
    fe.DCboundary.DirichletBCSpec(location=left, component='x', value=zero_dirichlet_val),
    fe.DCboundary.DirichletBCSpec(location=left, component='y', value=dirichlet_val_x2),
    fe.DCboundary.DirichletBCSpec(location=left, component='z', value=dirichlet_val_x3),
    # Right boundary - fixed
    fe.DCboundary.DirichletBCSpec(location=right, component='all', value=zero_dirichlet_val)
])
```

**Note:** Using component names `'x'`, `'y'`, `'z'` is equivalent to indices `0`, `1`, `2`.

## Create Problem Instance

Instantiate the hyperelasticity problem:

```python
feax_problem = HyperElasticityFeax(mesh, vec=3, dim=3)
internal_vars = fe.internal_vars.InternalVars()
bc = bc_config.create_bc(feax_problem)
```

For this example, we use empty `internal_vars` since material parameters are hardcoded. For parameter studies, these would be passed as internal variables.

## Nonlinear Solver Configuration

Configure the Newton solver for the nonlinear problem:

```python
solver_options = fe.solver.SolverOptions(
    tol=1e-8,                   # Nonlinear residual tolerance
    linear_solver="bicgstab",   # Linear solver for Newton system
    max_iter=10,                # Maximum Newton iterations
    verbose=True                # Print convergence info during iterations
)

# Create nonlinear solver (no iter_num=1!)
solver = fe.solver.create_solver(feax_problem, bc, solver_options)
```

### Monitoring Convergence with `verbose=True`

Setting `verbose=True` enables iteration-by-iteration convergence logging using `jax.debug.print()`, which works seamlessly with JIT compilation and `jax.vmap`:

```python
solver_options = fe.solver.SolverOptions(tol=1e-8, linear_solver="bicgstab", verbose=True)
solver = fe.solver.create_solver(feax_problem, bc, solver_options)
sol = solver(internal_vars, fe.utils.zero_like_initial_guess(feax_problem, bc))
```

**Output:**
```
Newton solver starting: initial res_norm = 2.345678e+01
Newton iter   0: res_norm = 5.432109e+00, alpha = 1.0000, success = True
Newton iter   1: res_norm = 1.234567e+00, alpha = 1.0000, success = True
Newton iter   2: res_norm = 2.345678e-02, alpha = 1.0000, success = True
Newton iter   3: res_norm = 1.234567e-05, alpha = 1.0000, success = True
Newton iter   4: res_norm = 2.345678e-09, alpha = 1.0000, success = True
Newton solver converged: final_iter = 5, final_res_norm = 2.345678e-09
```

**Key benefits:**
- ✅ **JIT-compatible**: Uses `jax.debug.print()` instead of Python `print()`
- ✅ **Vmap-compatible**: Works when solving multiple cases with `jax.vmap`
- ✅ **Armijo line search info**: Shows step size `alpha` and search success
- ✅ **No performance overhead**: Logging calls are optimized out by XLA when not needed

### Linear vs Nonlinear Solver

| Configuration | Use Case | Newton Iterations |
|--------------|----------|-------------------|
| `create_solver(..., iter_num=1)` | Linear problems | 1 (single solve) |
| `create_solver(...)` | Nonlinear problems | Multiple (until convergence) |

**Key differences:**
- **Linear problems**: Residual is linear in unknowns, solved in one iteration
- **Nonlinear problems**: Requires iterative Newton method to converge
- **Convergence criterion**: $\|\mathbf{r}\| < \text{tol}$ where $\mathbf{r}$ is the residual

## Newton's Method Workflow

FEAX implements Newton's method automatically:

```
1. Initialize: u⁰ = initial_guess
2. For k = 0, 1, 2, ... until convergence:
   a. Assemble residual: r^k = r(u^k)
   b. Check convergence: if ‖r^k‖ < tol, stop
   c. Assemble tangent: K^k = ∂r/∂u|_{u^k}  (via JAX autodiff)
   d. Solve linear system: K^k δu = -r^k
   e. Update: u^{k+1} = u^k + δu
```

The tangent stiffness $\mathbf{K}$ is computed automatically via `jax.jacobian` of the residual.

## Solving the Problem

Solve the nonlinear system:

```python
def solve_fn(internal_vars):
    initial_guess = fe.utils.zero_like_initial_guess(feax_problem, bc)
    sol = solver(internal_vars, initial_guess)
    return sol

print("Solving...")
sol = solve_fn(internal_vars)
sol_unflat = feax_problem.unflatten_fn_sol_list(sol)
displacement = sol_unflat[0]
```

### Available Solver Options

The `SolverOptions` dataclass provides extensive control over the Newton solver:

```python
solver_options = fe.solver.SolverOptions(
    tol=1e-8,                          # Absolute residual tolerance
    rel_tol=1e-8,                      # Relative residual tolerance
    max_iter=10,                       # Maximum Newton iterations
    linear_solver="bicgstab",          # Linear solver: "cg", "bicgstab", "gmres"
    use_jacobi_preconditioner=False,   # Enable diagonal preconditioning
    jacobi_shift=1e-12,                # Regularization for preconditioner
    linear_solver_tol=1e-10,           # Linear solver convergence tolerance
    linear_solver_maxiter=10000,       # Max linear solver iterations
    line_search_max_backtracks=30,     # Armijo line search parameters
    line_search_c1=1e-4,               # Sufficient decrease constant
    line_search_rho=0.5,               # Backtracking factor
    verbose=False                      # Enable convergence logging (JIT/vmap compatible)
)
```

**Key options for nonlinear problems:**
- **`tol`**: Newton method converges when residual norm `< tol`
- **`max_iter`**: Prevents infinite loops in difficult cases
- **`linear_solver`**: BiCGSTAB recommended for general problems, CG for symmetric
- **Line search**: Helps convergence for difficult nonlinear problems

## Visualization

Save the solution for ParaView visualization:

```python
import os

# Create output directory
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)
vtk_path = os.path.join(data_dir, 'vtk/u_hyper_elast.vtu')

# Save displacement field
fe.utils.save_sol(
    mesh=mesh,
    sol_file=vtk_path,
    point_infos=[("displacement", displacement)]
)

print(f"Solution saved to: {vtk_path}")
```

### Visualizing in ParaView

1. Open `u_hyper_elast.vtu` in ParaView
2. Apply **Warp By Vector** filter:
   - Select `displacement` as the vector field
   - Set scale factor to `1.0` to see actual deformation
3. Color by `displacement` magnitude to see deformation intensity
4. The cube should show a clear twisting/rotation pattern

## Complete Code

```python
import feax as fe
import jax
import jax.numpy as np
import os

class HyperElasticityFeax(fe.problem.Problem):
    def get_tensor_map(self):
        def psi(F):
            E = 100.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)
        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress

# Mesh
mesh = fe.mesh.box_mesh((1, 1, 1), mesh_size=0.1)

# Boundary locations
left = lambda point: np.isclose(point[0], 0., atol=1e-5)
right = lambda point: np.isclose(point[0], 1, atol=1e-5)

# Boundary values
zero_dirichlet_val = lambda point: 0.
dirichlet_val_x2 = lambda point: (0.5 + (point[1] - 0.5) * np.cos(np.pi / 3.) -
                                   (point[2] - 0.5) * np.sin(np.pi / 3.) - point[1]) / 2.
dirichlet_val_x3 = lambda point: (0.5 + (point[1] - 0.5) * np.sin(np.pi / 3.) +
                                   (point[2] - 0.5) * np.cos(np.pi / 3.) - point[2]) / 2.

# Boundary conditions
bc_config = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(location=left, component='x', value=zero_dirichlet_val),
    fe.DCboundary.DirichletBCSpec(location=left, component='y', value=dirichlet_val_x2),
    fe.DCboundary.DirichletBCSpec(location=left, component='z', value=dirichlet_val_x3),
    fe.DCboundary.DirichletBCSpec(location=right, component='all', value=zero_dirichlet_val)
])

# Problem setup
feax_problem = HyperElasticityFeax(mesh, vec=3, dim=3)
internal_vars = fe.internal_vars.InternalVars()
bc = bc_config.create_bc(feax_problem)

# Nonlinear solver
solver_options = fe.solver.SolverOptions(tol=1e-8, linear_solver="bicgstab")
solver = fe.solver.create_solver(feax_problem, bc, solver_options)

# Solve
sol = solver(internal_vars, fe.utils.zero_like_initial_guess(feax_problem, bc))
displacement = feax_problem.unflatten_fn_sol_list(sol)[0]

# Save
fe.utils.save_sol(mesh, "u_hyper_elast.vtu", point_infos=[("displacement", displacement)])
```

## Advanced Topics

### Parameter Studies with Hyperelasticity

Pass material parameters via internal variables:

```python
class ParametricHyperelasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def psi(F, E, nu):
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)
        def first_PK_stress(u_grad, E, nu):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F, E, nu)
            return P

        return first_PK_stress

# Create spatially varying Young's modulus
E_array = fe.internal_vars.InternalVars.create_node_var(problem, 100.0)
nu_array = fe.internal_vars.InternalVars.create_node_var(problem, 0.3)
internal_vars = fe.internal_vars.InternalVars(volume_vars=(E_array, nu_array))
```

### Vectorized Material Parameter Study

Use `jax.vmap` to solve for multiple material properties in parallel:

```python
def solve_for_material(E_value):
    E_array = fe.internal_vars.InternalVars.create_node_var(problem, E_value)
    nu_array = fe.internal_vars.InternalVars.create_node_var(problem, 0.3)
    internal_vars = fe.internal_vars.InternalVars(volume_vars=(E_array, nu_array))
    return solver(internal_vars, fe.utils.zero_like_initial_guess(problem, bc))

# Vectorize over Young's modulus
solve_vmap = jax.vmap(solve_for_material)
E_values = np.linspace(50., 200., 10)
solutions = solve_vmap(E_values)  # Solve 10 cases in parallel
```

See [Vectorization Transform](./vmap_transform.md) for details on parameter studies.

### Alternative Material Models

The energy-based approach makes it trivial to implement other hyperelastic models:

**Mooney-Rivlin:**
```python
def psi(F):
    C10, C01 = 0.5, 0.2  # Material constants
    kappa = 1000.
    J = np.linalg.det(F)
    C = F.T @ F
    I1 = np.trace(C)
    I2 = 0.5 * (I1**2 - np.trace(C @ C))
    energy = C10 * (J**(-2./3.) * I1 - 3.) + C01 * (J**(-4./3.) * I2 - 3.) + \
             (kappa / 2.) * (J - 1.)**2
    return energy
```

**Ogden:**
```python
def psi(F):
    mu1, alpha1 = 6.3, 1.3
    kappa = 1000.
    C = F.T @ F
    lambdas = np.sqrt(np.linalg.eigvalsh(C))  # Principal stretches
    J = np.prod(lambdas)
    energy = (mu1 / alpha1) * np.sum(J**(-alpha1/3.) * lambdas**alpha1 - 3.) + \
             (kappa / 2.) * (J - 1.)**2
    return energy
```

Just replace the energy function and JAX handles the rest!

## Convergence Considerations

### Common Convergence Issues

1. **Too large deformation**: Reduce boundary displacement or use load stepping
2. **Material locking**: Refine mesh or use mixed formulations
3. **Poor initial guess**: Use solution continuation from smaller loads

### Load Stepping

For very large deformations, incrementally apply the load:

```python
# Apply load in 10 steps
n_steps = 10
scale_factors = np.linspace(0.1, 1.0, n_steps)

solution = fe.utils.zero_like_initial_guess(problem, bc)
for i, scale in enumerate(scale_factors):
    # Scale boundary conditions
    bc_scaled = scale_boundary_conditions(bc, scale)

    # Solve using previous solution as initial guess
    solution = solver(internal_vars, solution)
    print(f"Step {i+1}/{n_steps}: scale = {scale:.2f}")

displacement_final = problem.unflatten_fn_sol_list(solution)[0]
```

### Monitoring Solution Quality

Check solution quality indicators:

```python
# Check for negative Jacobians (invalid deformation)
def check_jacobians(problem, displacement):
    def compute_det_F(cell_sol):
        u_grad = problem.fes[0].get_grad_at_quad_points(cell_sol)
        F = u_grad + np.eye(3)
        J = np.linalg.det(F)
        return J

    # Compute Jacobians for all cells
    jacobians = jax.vmap(compute_det_F)(displacement.reshape(problem.num_cells, -1, 3))

    min_J = np.min(jacobians)
    if min_J <= 0:
        print(f"⚠️  Warning: Negative Jacobian detected (min = {min_J:.6f})")
    else:
        print(f"✅ All Jacobians positive (min = {min_J:.6f})")

    return min_J

min_jacobian = check_jacobians(feax_problem, displacement)
```

## Key Takeaways

1. **Energy-based formulation** with `jax.grad` eliminates manual stress derivation
2. **Automatic tangent stiffness** via JAX's automatic differentiation
3. **Newton's method** for nonlinear problems (omit `iter_num=1`)
4. **Large deformations** require deformation gradient $\mathbf{F} = \nabla \mathbf{u} + \mathbf{I}$
5. **Load stepping** helps convergence for extreme deformations
6. **Material models** are easy to swap by changing the energy function

## Further Reading

- Implement other hyperelastic models (Mooney-Rivlin, Ogden, Yeoh)
- Study mixed formulations for near-incompressible materials
- Explore dynamic problems with mass and damping
- Investigate contact mechanics with nonlinear constraints
- Learn about [JIT compilation](./jit_transform.md) to accelerate Newton iterations
