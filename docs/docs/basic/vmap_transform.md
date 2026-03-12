# Vectorization Transform

This tutorial demonstrates using `jax.vmap` to solve multiple finite element problems in parallel. As an example, we study how structural stiffness varies with material density using the SIMP interpolation scheme.

## Overview

`jax.vmap` automatically vectorizes a function to operate over a batch of inputs. For FEA solvers this eliminates explicit loops and enables hardware parallelism.

## Problem Setup: Density-Dependent Elasticity

We define a problem where Young's modulus varies with a density field via SIMP:

$$
E(\rho) = (E_0 - E_\varepsilon)\,\rho^p + E_\varepsilon
$$

```python
import feax as fe
import jax
import jax.numpy as np
import time

E0    = 70e3   # Young's modulus at full density
E_eps = 1e-3   # Residual stiffness
nu    = 0.3
p     = 3      # SIMP penalization exponent
T     = 1e2    # Traction magnitude (fixed)

mesh = fe.mesh.box_mesh((2, 1, 1), mesh_size=0.1)
print(f"Mesh: {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} elements")

left  = lambda point: np.isclose(point[0], 0, atol=1e-5)
right = lambda point: np.isclose(point[0], 2, atol=1e-5)
```

## Problem Definition

The density `rho` is received as a volume internal variable at each quadrature point:

```python
class DensityElasticityProblem(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, rho):
            E      = (E0 - E_eps) * rho**p + E_eps
            mu     = E / (2.0 * (1.0 + nu))
            lam    = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            strain = 0.5 * (u_grad + u_grad.T)
            return lam * np.trace(strain) * np.eye(self.dim) + 2.0 * mu * strain
        return stress

    def get_surface_maps(self):
        def traction_map(u_grad, surface_quad_point, traction_magnitude):
            return np.array([0.0, 0.0, -traction_magnitude])
        return [traction_map]

bc_config = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(location=left, component='all', value=0.0)
])

problem = DensityElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    location_fns=[right]
)
bc = bc_config.create_bc(problem)
print(f"Problem: {problem.num_total_dofs_all_vars} DOFs")
```

## Solver

```python
solver_options = fe.DirectSolverOptions(solver="cudss")
solver = fe.create_solver(problem, bc, solver_options, iter_num=1)
```

## Single Solve Function

Encapsulate a solve for one density value:

```python
def single_solve(density):
    rho       = fe.InternalVars.create_uniform_volume_var(problem, density)
    traction_z = fe.InternalVars.create_uniform_surface_var(problem, T)
    iv = fe.InternalVars(
        volume_vars=[rho],
        surface_vars=[(traction_z,)]
    )
    return solver(iv, fe.zero_like_initial_guess(problem, bc))
```

## Vectorization with vmap

Transform the single-solve function to handle a batch of densities automatically:

```python
solve_vmap = jax.vmap(single_solve)

# Pre-compile
_ = solve_vmap(np.array([0.5]))
jax.block_until_ready(_)

# Solve for a batch of density values in parallel
density_values = np.linspace(0.1, 1.0, 10)
solutions = solve_vmap(density_values)
print(f"Solutions shape: {solutions.shape}")  # (10, num_dofs)
```

`jax.vmap` automatically vectorizes the entire solve pipeline — no loop modification needed.

## Benchmarking

```python
batch_sizes = [1, 10, 20, 30, 40]

for batch_size in batch_sizes:
    density_values = np.linspace(0.1, 1.0, batch_size)

    start_time    = time.time()
    vmap_solutions = solve_vmap(density_values)
    jax.block_until_ready(vmap_solutions)
    vmap_time = time.time() - start_time

    print(f"Batch {batch_size:3d}: vmap = {vmap_time:.4f}s")
```

## Combining with JIT

```python
solve_vmap_jit = jax.jit(jax.vmap(single_solve))

# First call triggers compilation
_ = solve_vmap_jit(np.array([0.5]))
jax.block_until_ready(_)

# Subsequent calls are very fast
density_values = np.linspace(0.1, 1.0, 40)
solutions = solve_vmap_jit(density_values)
```

## Key Takeaways

1. **`jax.vmap` vectorizes an entire solve pipeline** without code changes
2. **Volume internal variables** pass spatially uniform or varying fields (e.g., density) to the constitutive law
3. **Surface internal variables** pass loads that remain fixed across the batch
4. **Compose with `jax.jit`** for maximum throughput on repeated batches
5. **Compose with `jax.grad`** for gradient-based optimization over the density field

## Further Reading

- [JIT Transform](./jit_transform.md) — `jax.jit` for repeated single solves
- [Nonlinear Hyperelasticity](./hyperelasticity.md) — nonlinear problems with Newton's method
