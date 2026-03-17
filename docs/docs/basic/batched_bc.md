# Batched Dirichlet BC

This tutorial demonstrates using `jax.vmap` to solve the same linear elasticity problem for **multiple prescribed boundary displacements** in a single vectorised call. While the [Vectorization Transform](./vmap_transform.md) tutorial vmaps over material parameters (`InternalVars`), this example vmaps over **Dirichlet boundary condition values**.

## Overview

The key insight is that `DirichletBC` is registered as a **JAX pytree**. You can swap its prescribed values via `bc.replace_vals(new_vals)` and vmap over a batch of values — the mesh, DOF locations, and stiffness matrix structure stay fixed while only the BC values change.

## Problem Setup

A 2D plane-stress cantilever beam:
- Left face: fully fixed ($u_x = u_y = 0$)
- Right face: prescribed $u_x$ displacement (varies across the batch)

```python
import jax
import jax.numpy as np
import feax as fe

E, nu = 70e3, 0.3
batch_size = 1000

class LinearElasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad):
            mu = E / (2.0 * (1.0 + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(eps) * np.eye(self.dim) + 2 * mu * eps
        return stress

mesh = fe.mesh.rectangle_mesh(Nx=20, Ny=5, domain_x=10.0, domain_y=2.0)

left = lambda p: np.isclose(p[0], 0.0, atol=1e-5)
right = lambda p: np.isclose(p[0], 10.0, atol=1e-5)

problem = LinearElasticity(mesh, vec=2, dim=2, ele_type='QUAD4')
```

## Boundary Conditions

Define BCs with a **placeholder** value (0.0) for the right-face x-displacement. This value will be replaced per batch entry:

```python
bc_config = fe.DirichletBCConfig([
    fe.DirichletBCSpec(location=left, component="all", value=0.0),
    fe.DirichletBCSpec(location=right, component="x", value=0.0),  # placeholder
])
bc = bc_config.create_bc(problem)
```

## Solver

```python
iv = fe.InternalVars(volume_vars=())
solver = fe.create_solver(
    problem, bc,
    solver_options=fe.DirectSolverOptions(),
    iter_num=1,
    internal_vars=iv,
)
```

## Building the Batch

Locate the right-face x-DOFs inside `bc.bc_vals` and build a batch of BC value vectors — one per prescribed displacement:

```python
displacements = np.linspace(0.1, 100, batch_size)

# Find right-face x-DOF positions in bc_vals
right_nodes = np.argwhere(
    jax.vmap(right)(mesh.points)
).reshape(-1)
right_x_dofs = right_nodes * problem.fes[0].vec  # x-component DOF indices

def make_bc_vals(disp):
    return bc.bc_vals.at[
        np.searchsorted(bc.bc_rows, right_x_dofs)
    ].set(disp)

bc_vals_batch = jax.vmap(make_bc_vals)(displacements)  # (batch_size, n_bc_dofs)
```

`bc.bc_vals` is sorted by `bc.bc_rows`, so `np.searchsorted` maps the global DOF indices to their positions in the BC value array.

## Vectorised Solve

Use `bc.replace_vals()` inside a `jax.vmap` to solve all cases in parallel:

```python
@jax.jit
def solve_batch(vals_batch):
    return jax.vmap(lambda v: solver(iv, bc=bc.replace_vals(v)))(vals_batch)

sols = solve_batch(bc_vals_batch)  # (batch_size, total_dofs)
```

The solver is called once per batch entry, but `jax.vmap` + `jax.jit` fuse the computation into a single XLA program with batch-level parallelism.

## How `replace_vals` Works

`DirichletBC` is a frozen dataclass with four fields:

| Field | Description | Changes across batch? |
|---|---|---|
| `bc_rows` | DOF indices with BCs | No |
| `bc_mask` | Boolean mask for BC DOFs | No |
| `bc_vals` | Prescribed values | **Yes** |
| `total_dofs` | Total DOF count | No |

`replace_vals(new_vals)` returns a new `DirichletBC` with only `bc_vals` swapped. Since `DirichletBC` is a JAX pytree, `jax.vmap` traces through the replacement and batches the solve automatically.

## Key Takeaways

1. **`DirichletBC` is a JAX pytree** — it works with `jax.vmap`, `jax.jit`, and `jax.grad`
2. **`replace_vals()`** swaps prescribed values while keeping DOF locations fixed
3. **`np.searchsorted`** maps global DOF indices to positions in `bc_vals`
4. **Compose `vmap` + `jit`** for maximum throughput on batched BC sweeps

## Running the Example

```bash
python examples/basic/batched_bc.py
```

## Further Reading

- [Vectorization Transform](./vmap_transform.md) — vmap over material parameters (`InternalVars`)
- [JIT Transform](./jit_transform.md) — `jax.jit` for repeated single solves
