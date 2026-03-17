# Solver Guide

This page explains how to configure and use `fe.create_solver` — the central entry point for solving finite element problems in FEAX.

## Basic Usage

Every FEAX solve follows the same pattern:

```python
import feax as fe

# 1. Build the solver
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(),
    iter_num=1,
    internal_vars=internal_vars)

# 2. Create initial guess
initial = fe.zero_like_initial_guess(problem, bc)

# 3. Solve
sol = solver(internal_vars, initial)
```

The returned `solver` is a callable with a `custom_vjp`, so it works seamlessly with `jax.jit`, `jax.grad`, and `jax.vmap`.

## Solver Options

FEAX offers three solver option classes, each suited to different problem types:

| Option class | Method | Best for |
|---|---|---|
| `fe.DirectSolverOptions()` | Sparse direct (cuDSS on GPU, cholmod / umfpack / spsolve on CPU) | Default choice; robust and fast when memory permits |
| `fe.IterativeSolverOptions()` | Iterative (CG / BiCGSTAB / GMRES) | Memory-constrained problems, periodic BCs with `P` |
| `fe.MatrixFreeOptions()` | Matrix-free Newton via JVP | Custom energy, no assembly |

### Direct Solvers

```python
# Auto-select: cuDSS on GPU, spsolve on CPU
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(),
    iter_num=1, internal_vars=internal_vars)

# Explicit backend
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(solver="spsolve"),
    iter_num=1)
```

Available backends: `"auto"`, `"cudss"`, `"spsolve"`, `"cholmod"`, `"umfpack"`.

:::note cuDSS and `symmetric_bc=False`
When `symmetric_bc=False` is used, the Jacobian becomes non-symmetric (GENERAL). The auto-selection detects this and configures cuDSS in LU mode automatically. If you manually specify `DirectSolverOptions(solver="cudss")`, ensure that the `CUDSSOptions(matrix_type=...)` matches the actual matrix symmetry.
:::

### Iterative Solvers

```python
solver = fe.create_solver(problem, bc,
    solver_options=fe.IterativeSolverOptions(solver="cg"),
    iter_num=1, internal_vars=internal_vars)
```

Available backends: `"auto"`, `"cg"`, `"bicgstab"`, `"gmres"`.

Use `"cg"` for SPD matrices (symmetric problems), `"bicgstab"` or `"gmres"` for general matrices.

## Automatic Solver Selection

When `solver_options` is omitted or set to `solver="auto"`, FEAX automatically selects the best solver based on the hardware backend and matrix properties. The selection follows the principle: **try direct solvers first, fall back to iterative only when explicitly requested**.

### Selection Flow

```
solver_options=None  (or DirectSolverOptions(solver="auto"))
        │
        ▼
  Assemble sample Jacobian
        │
        ▼
  detect_matrix_property(J)
  ┌─────────┬─────────────┬──────────┐
  │   SPD   │  SYMMETRIC  │ GENERAL  │
  └────┬────┴──────┬──────┴────┬─────┘
       │           │           │
       ▼           ▼           ▼
  ┌─── GPU available? ───────────────┐
  │  Yes → cuDSS (Cholesky/LDLT/LU) │  ← highest priority
  └──────────────────────────────────┘
       │           │           │
       ▼           ▼           ▼
  ┌─── CPU fallback ─────────────────┐
  │  SPD:                            │
  │    cholmod → umfpack → spsolve   │
  │  SYMMETRIC / GENERAL:            │
  │    umfpack → spsolve             │
  └──────────────────────────────────┘
```

### Direct Solver Priority

The direct solver is selected in the following priority order:

| Priority | Solver | Platform | Matrix types | Method |
|---|---|---|---|---|
| 1 | **cuDSS** | GPU | SPD / SYMMETRIC / GENERAL | Cholesky / LDLT / LU |
| 2 | **cholmod** | CPU | SPD only | Supernodal Cholesky |
| 3 | **umfpack** | CPU | SPD / SYMMETRIC / GENERAL | Multifrontal LU |
| 4 | **spsolve** | CPU | SPD / SYMMETRIC / GENERAL | SciPy sparse LU |

cuDSS automatically adapts its factorization to the matrix property:
- **SPD** → Cholesky (fastest, lowest memory)
- **SYMMETRIC** → LDLT (no pivoting overhead)
- **GENERAL** → LU with partial pivoting

### Iterative Solver Selection

When `IterativeSolverOptions(solver="auto")` is used, the iterative solver is selected by matrix property:

| Matrix property | Solver | Notes |
|---|---|---|
| SPD | `cg` | Optimal for symmetric positive definite |
| SYMMETRIC | `bicgstab` | Robust for indefinite symmetric systems |
| GENERAL | `gmres` | General-purpose Krylov method |

### Matrix Property Detection

`detect_matrix_property(J)` performs two numerical checks on the assembled Jacobian:

1. **Symmetry test**: Compares $\|\mathbf{A}\mathbf{x} - \mathbf{A}^T\mathbf{x}\|$ for a random vector. Skipped when `matrix_view='UPPER'` or `'LOWER'` (symmetry guaranteed by construction).
2. **Positive-definiteness heuristic**: Checks if all diagonal entries are positive (necessary but not sufficient for SPD).

### When to Override Auto Selection

| Situation | Recommendation |
|---|---|
| Problem is too large for direct solver (out of memory) | `IterativeSolverOptions()` |
| Need symmetric solver but using `symmetric_bc=False` | `DirectSolverOptions(solver="spsolve")` or `"umfpack"` |
| Periodic BCs with prolongation matrix `P` | `IterativeSolverOptions()` (required) |
| Custom energy with matrix-free Newton | `MatrixFreeOptions()` |
| Extreme stiffness contrast (ill-conditioned Jacobian) | `DirectSolverOptions(solver="spsolve")` on CPU |

### Numerical Stability: cuDSS vs CPU Direct Solvers

cuDSS (GPU) uses LU factorization optimized for throughput, but its pivoting strategy can be less robust than CPU-based solvers for extremely ill-conditioned systems. Problems with large stiffness contrasts (e.g., third-medium contact with γ₀ ≈ 1e-6, multi-material topology optimization) may exhibit:

- Newton convergence stalling after the first iteration
- Line search returning very small step sizes (α ≈ 0)
- Residual not decreasing despite non-zero Newton increment

In such cases, switch to a CPU direct solver with more robust pivoting:

```python
# CPU: SciPy sparse LU (most robust pivoting)
solver_options = fe.DirectSolverOptions(solver="spsolve")

# CPU: UMFPACK multifrontal LU (good balance of speed and stability)
solver_options = fe.DirectSolverOptions(solver="umfpack")
```

On GPU, if a direct solver is required, consider running the ill-conditioned part on CPU or using an iterative solver with preconditioning:

```python
# GPU: GMRES with Jacobi preconditioner
solver_options = fe.IterativeSolverOptions(
    solver="gmres", use_jacobi_preconditioner=True)
```

## Linear vs. Nonlinear: `iter_num`

The `iter_num` parameter controls the Newton iteration strategy:

| `iter_num` | Behavior | `jax.vmap` compatible |
|---|---|---|
| `1` | Single linear solve | Yes |
| `> 1` | Fixed N Newton iterations | Yes |
| `None` (default) | Adaptive Newton with while loop | No |

### Linear Problems (`iter_num=1`)

For linear elasticity and other linear PDEs, one linear solve is sufficient:

```python
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(),
    iter_num=1, internal_vars=internal_vars)
sol = solver(internal_vars, initial)
```

### Nonlinear Problems (Newton Solver)

For hyperelasticity and other nonlinear problems, use the Newton solver:

```python
# Adaptive Newton — stops when converged (not vmappable)
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(),
    newton_options=fe.NewtonOptions(tol=1e-8, max_iter=50),
    iter_num=None, internal_vars=internal_vars)

# Fixed iterations — always runs exactly N steps (vmappable)
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(),
    iter_num=10, internal_vars=internal_vars)
```

#### Newton Options

`fe.NewtonOptions` controls convergence criteria and line search:

```python
fe.NewtonOptions(
    tol=1e-8,                      # absolute residual tolerance
    rel_tol=1e-8,                  # relative residual tolerance
    max_iter=50,                   # maximum Newton iterations
    line_search_c1=1e-4,           # Armijo condition parameter
    line_search_rho=0.5,           # backtracking factor
    line_search_max_backtracks=10, # max line search steps
    internal_jit=True,             # JIT-compile each component separately
)
```

Setting `internal_jit=True` (default) JIT-compiles the residual, Jacobian, and linear solve functions individually, avoiding the long monolithic JIT compilation that can occur with large problems.

## Boundary Condition Elimination: `symmetric_bc`

The `symmetric_bc` parameter controls how Dirichlet boundary conditions are enforced in the Jacobian matrix. This choice can be critical for Newton convergence in nonlinear problems.

### `symmetric_bc=True` (default) — Symmetric Elimination

Zeros both BC rows **and** columns in the Jacobian, then sets BC diagonal entries to 1:

$$
\mathbf{K}_\text{mod} = \begin{pmatrix} \mathbf{I} & \mathbf{0} \\ \mathbf{0} & \mathbf{K}_{11} \end{pmatrix}
$$

- Preserves matrix symmetry → symmetric solvers (CG, Cholesky) can be used
- Removes K₁₀ coupling between BC DOFs and interior DOFs
- Suitable for: fixed BCs, linear problems, problems where BC values are pre-applied to the initial guess

```python
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(),
    iter_num=1, symmetric_bc=True)  # default
```

### `symmetric_bc=False` — Non-symmetric Elimination

Zeros only BC **rows**, keeping BC columns (K₁₀ coupling) intact:

$$
\mathbf{K}_\text{mod} = \begin{pmatrix} \mathbf{I} & \mathbf{0} \\ \mathbf{K}_{10} & \mathbf{K}_{11} \end{pmatrix}
$$

The Newton solver drives BC DOFs to their prescribed values through the modified residual: `res[bc_dof] = sol[bc_dof] - bc_val`.

- Maintains K₁₀ coupling → more accurate Newton linearization
- Produces a non-symmetric Jacobian → CG cannot be used; use `spsolve`, `umfpack`, `bicgstab`, or `gmres`

```python
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(solver="spsolve"),
    newton_options=fe.NewtonOptions(tol=1e-6, max_iter=20),
    iter_num=None, symmetric_bc=False,
    internal_vars=internal_vars)
```

### When to Use `symmetric_bc=False`

Use non-symmetric elimination when the K₁₀ coupling is important for Newton convergence:

1. **Incremental loading** — BC values change per load step and the previous solution is reused as the initial guess. K₁₀ ensures that prescribed displacement changes propagate correctly to interior DOFs.

2. **Large stiffness contrast** — e.g., third-medium contact where background medium stiffness is scaled by γ₀ ≈ 1e-6. Without K₁₀, the first Newton increment overshoots in soft regions, causing divergence.

3. **Large-deformation nonlinear problems** — where BC DOF displacements are large and strongly coupled to interior DOFs.

:::tip Rule of thumb
If your Newton solver converges with `symmetric_bc=True`, keep the default — it enables symmetric solvers and is slightly more efficient. Switch to `symmetric_bc=False` when you observe divergence or poor convergence that you suspect is caused by the boundary condition treatment.
:::

## Incremental Loading

For problems where the prescribed displacement or load is applied gradually over multiple steps, use `bc.replace_vals()` with `symmetric_bc=False`:

```python
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(solver="spsolve"),
    newton_options=fe.NewtonOptions(tol=1e-6, max_iter=20),
    iter_num=None, symmetric_bc=False,
    internal_vars=internal_vars)

sol = fe.zero_like_initial_guess(problem, bc)

for step in range(1, num_steps + 1):
    # Update prescribed values (same DOF locations, different values)
    scale = step / num_steps
    new_vals = bc.bc_vals.at[move_bc_pos].set(max_disp * scale)
    bc_step = bc.replace_vals(new_vals)

    # Solve with updated BCs, reusing previous solution as initial guess
    sol = solver(internal_vars, sol, bc=bc_step)
```

Key points:
- `bc.replace_vals(new_vals)` creates a new `DirichletBC` with updated values but the same DOF locations — no solver rebuild needed.
- The previous solution `sol` is passed as the initial guess, giving Newton a good starting point.
- The solver's optional `bc=` keyword argument overrides the BC values without re-compiling.

## Matrix-Free Solver

For problems with custom energy contributions (e.g., cohesive zones), use `MatrixFreeOptions`. The tangent operator is computed via `jax.jvp` — no sparse matrix is ever assembled:

```python
from feax.solvers.matrix_free import MatrixFreeOptions, LinearSolverOptions, create_energy_fn

elastic_energy = create_energy_fn(problem)

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

## `MatrixView` for Symmetric Problems

When the problem is symmetric (most single-variable problems with `symmetric_bc=True`), set `matrix_view='UPPER'` on the Problem to store only the upper triangle of the stiffness matrix. This reduces memory by ~50% and enables optimized solvers (Cholesky):

```python
problem = MyProblem(mesh, vec=3, dim=3, matrix_view='UPPER')
```

:::caution
Do not use `matrix_view='UPPER'` with `symmetric_bc=False`, as the modified Jacobian is no longer symmetric.
:::

## `jax.vmap` Compatibility

FEAX solvers support `jax.vmap` for batched parameter studies — solving the same problem with different boundary condition values, material parameters, or loads in a single vectorized call. This section details vmap compatibility for each solver path and how to use it.

### Compatibility Matrix

| Solver path | `iter_num` | `make_jittable` | `bc=` override | `jax.vmap` | Notes |
|---|---|---|---|---|---|
| **Linear** (Direct) | `1` | — | Yes | Yes | cuDSS, spsolve, cholmod, umfpack all supported |
| **Linear** (Iterative) | `1` | — | Yes | Yes | CG, BiCGSTAB, GMRES |
| **Reduced** (`P≠None`) | `1` | — | Yes | Yes | Iterative only (matrix-free matvec) |
| **Newton** (fori_loop) | `> 1` | `True` | Yes | Yes | Fixed iterations with `make_jittable=True` |
| **Newton** (while_loop) | `None` | — | Yes | **No** | Python/while loop not traceable by vmap |
| **Newton** (Python loop) | `> 1` | `False` | Yes | **No** | Python loop not traceable by vmap |
| **Matrix-free** | `≠ 1` | — | No | **No** | Python while loop, no `bc=` override |

:::tip Key rule
To use `jax.vmap` with a solver, avoid Python-level control flow. Use `iter_num=1` (linear), `P` (reduced), or `iter_num > 1` with `NewtonOptions(make_jittable=True)` (Newton fori_loop).
:::

### How `bc=` Override Works

All vmap-compatible solvers accept an optional `bc=` keyword argument. This overrides the boundary condition **values** without rebuilding the solver. Combined with `jax.vmap`, this enables batched solves over different prescribed displacements.

The key mechanism is the `DirichletBC.replace_vals()` method, which creates a new `DirichletBC` with different `bc_vals` but the same DOF locations (`bc_rows`) and mask (`bc_mask`):

```python
bc1 = bc.replace_vals(bc.bc_vals.at[-1].set(0.1))
bc2 = bc.replace_vals(bc.bc_vals.at[-1].set(0.5))
```

Internally, the solver uses **parametric** Jacobian and residual functions (`create_J_bc_parametric`, `create_res_bc_parametric`) that take `bc` as an explicit argument rather than capturing it in a closure. This allows JAX to trace through the BC values under `jax.vmap`.

### Limitation: BC Locations Must Be Identical Within a Batch

:::danger Important constraint
`jax.vmap` can only batch over BC **values** (`bc_vals`). The BC **locations** (`bc_rows`, `bc_mask`) must be identical across all elements in the batch.
:::

This is a fundamental constraint, not a temporary limitation:

1. **Jacobian sparsity pattern depends on BC locations** — `apply_boundary_to_J` zeros out rows and columns corresponding to `bc_rows`. Different BC locations produce different sparsity patterns (different `indices` in BCOO, different `indptr`/`indices` in BCSR). JAX's vmap requires all array shapes and structures to be identical across the batch.

2. **Direct solvers require a shared sparsity structure** — cuDSS batched factorization shares a single set of CSR `offsets` and `columns` across the batch; only the `values` and right-hand side `b` vary. Different BC locations would require different CSR structures, which is not supported.

When you need to solve with **different BC locations**, create a separate solver for each location pattern and use vmap within each group:

```python
# Group A: left edge fixed, right edge loaded
bc_a = fe.DirichletBCConfig([
    fe.DirichletBCSpec(location=left_edge, component='all', value=0.),
    fe.DirichletBCSpec(location=right_edge, component='x', value=0.),
]).create_bc(problem)
solver_a = fe.create_solver(problem, bc_a, iter_num=1, ...)

# Group B: bottom edge fixed, top edge loaded
bc_b = fe.DirichletBCConfig([
    fe.DirichletBCSpec(location=bottom_edge, component='all', value=0.),
    fe.DirichletBCSpec(location=top_edge, component='y', value=0.),
]).create_bc(problem)
solver_b = fe.create_solver(problem, bc_b, iter_num=1, ...)

# vmap within each group (values vary, locations fixed)
sols_a = jax.vmap(lambda v: solver_a(iv, bc=bc_a.replace_vals(v)))(vals_batch_a)
sols_b = jax.vmap(lambda v: solver_b(iv, bc=bc_b.replace_vals(v)))(vals_batch_b)
```

The solver construction cost (Jacobian assembly, cuDSS symbolic factorization, etc.) is incurred only once per BC location pattern. Within each group, all solves with different `bc_vals` are efficiently batched via vmap. Since the number of distinct BC location patterns is typically small (e.g., a few load cases or boundary condition types), this approach is practical and efficient.

### Basic vmap Example: Linear Solver

```python
import jax
import jax.numpy as np
import feax as fe

# Setup
problem = MyProblem(mesh, vec=2, dim=2, ele_type='QUAD4')
bc = fe.DirichletBCConfig([...]).create_bc(problem)
iv = fe.InternalVars(volume_vars=())

# Create solver (any solver backend works)
solver = fe.create_solver(problem, bc,
    solver_options=fe.IterativeSolverOptions(solver='cg'),
    iter_num=1)

# Stack bc_vals into a batch (shape: [batch_size, num_bc_dofs])
vals_batch = np.stack([
    bc.bc_vals.at[-1].set(0.1),
    bc.bc_vals.at[-1].set(0.5),
])

# vmap over bc_vals
solutions = jax.vmap(
    lambda v: solver(iv, bc=bc.replace_vals(v))
)(vals_batch)
# solutions.shape = (batch_size, num_dofs)
```

### vmap with Direct Solver (cuDSS)

cuDSS supports vmap natively. When the sparsity pattern (CSR offsets and column indices) is the same across the batch — which is always the case when only `bc_vals` changes — cuDSS uses an efficient batched factorization:

```python
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(),  # auto-selects cuDSS on GPU
    iter_num=1, internal_vars=iv)

solutions = jax.vmap(
    lambda v: solver(iv, bc=bc.replace_vals(v))
)(vals_batch)
```

### vmap with Newton Solver (fori_loop)

For nonlinear problems, use fixed iterations with `make_jittable=True` to enable vmap:

```python
from feax.solvers.options import NewtonOptions

solver = fe.create_solver(problem, bc,
    solver_options=fe.IterativeSolverOptions(solver='cg'),
    iter_num=10,
    internal_vars=iv,
    newton_options=NewtonOptions(make_jittable=True))

initial = fe.zero_like_initial_guess(problem, bc)

solutions = jax.vmap(
    lambda v: solver(iv, initial, bc=bc.replace_vals(v))
)(vals_batch)
```

:::note Newton solver requires explicit initial guess
Unlike the linear solver, the Newton solver always requires an explicit `initial_guess` argument. Use `fe.zero_like_initial_guess(problem, bc)` to create a suitable initial guess.
:::

:::caution Fixed iteration count
With `make_jittable=True`, the Newton solver runs exactly `iter_num` iterations regardless of convergence. Choose `iter_num` large enough for your problem. The adaptive while-loop path (`iter_num=None`) supports `bc=` override for sequential use but is **not** compatible with `jax.vmap`.
:::

### vmap with Reduced Solver (Periodic BCs)

The reduced solver (activated by passing a prolongation matrix `P`) also supports vmap:

```python
solver = fe.create_solver(problem, bc,
    solver_options=fe.IterativeSolverOptions(solver='cg'),
    iter_num=1, P=P)

solutions = jax.vmap(
    lambda v: solver(iv, bc=bc.replace_vals(v))
)(vals_batch)
```

### What Can Be Vmapped

| Batched input | How to vmap | Example |
|---|---|---|
| BC values (`bc_vals`) | `bc.replace_vals(v)` | Parametric displacement studies |
| Material parameters | Batch `InternalVars` | Material property sweeps |
| Both | Combine in lambda | Full parameter studies |

Example: vmap over both BC values and material parameters:

```python
def solve_one(bc_vals, material_params):
    iv = fe.InternalVars(volume_vars=(material_params,))
    return solver(iv, bc=bc.replace_vals(bc_vals))

solutions = jax.vmap(solve_one)(bc_vals_batch, material_params_batch)
```

### Combining vmap with jax.grad

Solvers with `custom_vjp` support both `jax.vmap` and `jax.grad`. A common pattern is to differentiate a batched loss:

```python
def batched_loss(material_params):
    iv = fe.InternalVars(volume_vars=(material_params,))
    # vmap over BC values
    sols = jax.vmap(
        lambda v: solver(iv, bc=bc.replace_vals(v))
    )(vals_batch)
    return np.sum(sols ** 2)

grad = jax.grad(batched_loss)(material_params)
```

### Solver Paths Not Compatible with vmap

The following solver paths use Python-level control flow that cannot be traced by `jax.vmap`:

- **Newton with `iter_num=None`** (adaptive while loop): Uses `jax.lax.while_loop` with dynamic termination, which is not vmap-compatible.
- **Newton with `make_jittable=False`** (default): Uses a Python `for` loop with `jax.debug.print` and early stopping — useful for debugging but not traceable.
- **Matrix-free solver** (`MatrixFreeOptions`): Uses a Python while loop internally for Newton iteration.

These paths still support the `bc=` override for **sequential** use (e.g., incremental loading), but cannot be batched with `jax.vmap`.

## `jax.grad` Compatibility

All FEAX solvers implement `jax.custom_vjp`, enabling `jax.grad` (reverse-mode AD) through the solve. The backward pass uses the **implicit function theorem**: instead of differentiating through Newton iterations, it solves a single adjoint linear system to obtain exact gradients efficiently.

### What Can Be Differentiated

| Parameter | Supported | Solver paths |
|---|---|---|
| `InternalVars` (material params, loads) | Yes | All solver paths |
| `bc_vals` (BC prescribed values) | Yes | All parametric paths (see below) |
| `initial_guess` | No | Gradient is `None` (not meaningful) |

### Differentiating w.r.t. `bc_vals`

To compute gradients through the solver with respect to boundary condition values, pass `bc=` as a keyword argument with a `DirichletBC` created via `replace_vals`:

```python
solver = fe.create_solver(problem, bc,
    solver_options=fe.IterativeSolverOptions(solver='cg'),
    iter_num=3, internal_vars=iv,
    newton_options=NewtonOptions(make_jittable=True))

initial = fe.zero_like_initial_guess(problem, bc)

def loss(bc_vals_arg):
    sol = solver(iv, initial, bc=bc.replace_vals(bc_vals_arg))
    return np.sum(sol ** 2)

grad_bc = jax.grad(loss)(bc.bc_vals)
```

### Compatibility Matrix

| Solver path | `iter_num` | `make_jittable` | `jax.grad` (iv) | `jax.grad` (bc_vals) |
|---|---|---|---|---|
| **Linear** (Direct/Iterative) | `1` | — | Yes | Yes |
| **Newton** (fori_loop) | `> 1` | `True` | Yes | Yes |
| **Newton** (Python loop) | `> 1` | `False` | Yes | Yes |
| **Newton** (while_loop) | `None` | — | Yes | Yes |
| **Reduced** (`P≠None`) | `1` | — | Yes | Yes |
| **Matrix-free** | `≠ 1` | — | Yes | No (`bc=` not supported) |

### Gradient Correctness with `symmetric_bc`

When `symmetric_bc=True` (the default), the forward Jacobian uses symmetric elimination which zeros out BC coupling columns (K₁₀). The backward pass automatically applies a correction to the adjoint solution at BC DOFs so that gradients w.r.t. `bc_vals` remain exact. No user action is required — the correction is transparent and preserves compatibility with symmetric solvers (CG, Cholesky).

:::tip Verifying gradients
For critical applications, you can verify analytic gradients against finite differences:

```python
def loss(bc_vals_arg):
    sol = solver(iv, initial, bc=bc.replace_vals(bc_vals_arg))
    return np.sum(sol ** 2)

analytic = jax.grad(loss)(bc_vals)

# Central finite difference
eps = 1e-5
for i in range(len(bc_vals)):
    p1 = bc_vals.at[i].add(eps)
    p2 = bc_vals.at[i].add(-eps)
    fd = (loss(p1) - loss(p2)) / (2 * eps)
    print(f"  i={i}: analytic={float(analytic[i]):.8f}, fd={float(fd):.8f}")
```
:::

## `jax.jit` Compatibility

All solver paths can be wrapped with `jax.jit` for compilation:

```python
solver = fe.create_solver(problem, bc,
    solver_options=fe.IterativeSolverOptions(solver='cg'),
    iter_num=3, internal_vars=iv,
    newton_options=NewtonOptions(make_jittable=True))

initial = fe.zero_like_initial_guess(problem, bc)

@jax.jit
def solve(iv, bc_vals):
    return solver(iv, initial, bc=bc.replace_vals(bc_vals))

sol = solve(iv, bc.bc_vals)  # first call triggers compilation
```

### Compatibility Matrix

| Solver path | `iter_num` | `make_jittable` | `jax.jit` | Notes |
|---|---|---|---|---|
| **Linear** (Direct/Iterative) | `1` | — | Yes | Single linear solve, fast compilation |
| **Newton** (fori_loop) | `> 1` | `True` | Yes | Entire Newton loop compiled into one XLA program |
| **Newton** (Python loop) | `> 1` | `False` | Partial | Each component (residual, Jacobian, solve) is JIT-compiled individually; Python loop runs on host |
| **Newton** (while_loop) | `None` | — | Yes | `jax.lax.while_loop` is XLA-traceable |
| **Reduced** (`P≠None`) | `1` | — | Yes | Matrix-free matvec, iterative only |
| **Matrix-free** | `≠ 1` | — | Partial | Python while loop on host, inner operations JIT-compiled |

### `make_jittable` and `internal_jit`

The Newton solver offers two JIT strategies via `NewtonOptions`:

| Option | Effect | Use case |
|---|---|---|
| `make_jittable=True` | Entire Newton loop is traced into a single XLA program via `jax.lax.fori_loop`. Requires fixed `iter_num`. | Small–medium problems; required for `jax.vmap`. |
| `make_jittable=False` (default) | Python-level Newton loop; each component (residual, Jacobian, linear solve) compiled separately. | Large 3-D problems where monolithic compilation is too slow. |
| `internal_jit=True` (default) | Wraps each component with `jax.jit` when `make_jittable=False`. | Ensures compiled execution even in the Python-loop path. |

:::note JIT compilation time
For large problems, `make_jittable=True` can cause very long initial compilation times because the entire Newton loop is fused into one XLA graph. In such cases, use the default `make_jittable=False` with `internal_jit=True` to keep compilation fast while still running compiled kernels.
:::

### Composing `jax.jit` with `jax.grad`

`jax.jit` and `jax.grad` compose naturally:

```python
@jax.jit
def loss_and_grad(bc_vals_arg):
    def loss(v):
        sol = solver(iv, initial, bc=bc.replace_vals(v))
        return np.sum(sol ** 2)
    return jax.value_and_grad(loss)(bc_vals_arg)

val, grad = loss_and_grad(bc.bc_vals)
```

## Summary

| Scenario | `iter_num` | `symmetric_bc` | Solver options |
|---|---|---|---|
| Linear, fixed BCs | `1` | `True` | `DirectSolverOptions()` |
| Nonlinear, fixed BCs | `None` | `True` | `DirectSolverOptions()` |
| Nonlinear, incremental loading | `None` | `False` | `DirectSolverOptions(solver="spsolve")` |
| Large problem, periodic BCs | `1` | `True` | `IterativeSolverOptions()` with `P` |
| Custom energy (cohesive) | — | — | `MatrixFreeOptions()` |
| Batched parameter study | `1` or `> 1` | `True` | any (must be vmappable) |
