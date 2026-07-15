# Solver Guide

This page explains how to configure and use `fe.create_solver` — the central entry point for solving finite element problems in FEAX.

## Basic Usage

Every FEAX solve follows the same pattern:

```python
import feax as fe

# 1. Build the solver
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(),
    linear=True,
    traced_params=traced_params)

# 2. Create an initial guess
initial = fe.zero_like_initial_guess(problem, bc)

# 3. Solve
sol = solver(traced_params, initial)
```

The returned `solver` is a callable with a `custom_vjp`, so it composes with `jax.jit`, `jax.grad`, and `jax.vmap`.

By default the solver returns a `fe.Solution` object rather than a bare array (`return_solution=True`). A `Solution` behaves like the flat DOF vector everywhere — arithmetic, `np.asarray`, passing as the next call's `initial_guess` — and adds structured accessors: `sol.dofs` (the raw flat vector), `sol.field(i)` (variable `i` as `(num_nodes, vec)`, replacing `problem.unflatten_fn_sol_list(sol)[i]`), and `sol.node_var(component=...)` (a `(num_nodes,)` node variable ready to feed the next solve's `TracedParams`). Pass `return_solution=False` for the raw flat vector.

`create_solver` has two orthogonal choices:

- **How to solve the linear system** — `solver_options`: a **direct** factorization (`DirectSolverOptions`), a **Krylov** iterative method (`KrylovSolverOptions`), or **algebraic multigrid** (`AMGSolverOptions`).
- **Linear or nonlinear** — the `linear` flag: a single linear solve (`linear=True`) or an adaptive Newton iteration (`linear=False`, the default).

## Solver Options

FEAX has three solver-option classes for `create_solver`, plus a dedicated geometric-multigrid solver class for structured grids:

| Option class | Method | Operator | Best for |
|---|---|---|---|
| `fe.DirectSolverOptions()` | Sparse direct factorization (cuDSS on GPU; cholmod / umfpack / spsolve on CPU) | Assembled CSR matrix | Default choice; robust and fast when memory permits |
| `fe.KrylovSolverOptions()` | Krylov iterative (CG / BiCGSTAB / GMRES) | **Matrix-free** (residual JVP) | Memory-bound problems, periodic BCs with `P`, custom residual terms |
| `fe.AMGSolverOptions()` | Krylov preconditioned by smoothed-aggregation **AMG** ([AMJax](https://github.com/vboussange/AMJax) + [PyAMG](https://github.com/pyamg/pyamg)) | Assembled CSR (for the AMG hierarchy) + matrix-free outer Krylov | Large scalar-elliptic problems (Poisson / thermal / diffusion); elasticity via a rigid-body near-null-space. Requires `feax[amg]` |
| `fe.NarrowBandCMG` ([GMG](#geometric-multigrid-gmg--narrowbandcmg)) | CG preconditioned by **geometric multigrid** (grid-coarsening transfers, MGPCG) | **Matrix-free** on a `StructuredGrid` band (one shared element stiffness) | Very large voxel domains / moving narrow bands (giga-voxel topology optimization). Not a `create_solver` route — a dedicated solver class |

This page focuses on **configuring** the solvers and **choosing** between them. For how each one works under the hood — the cuDSS factorization phases, the AMG V-cycle, the matrix-free GMG operator — see [Solver Internals](../solver_internals.mdx).

The distinction is the operator representation:

- **Direct** solvers need the matrix entries (to factorize), so the Jacobian is assembled straight into a deduplicated CSR matrix.
- **Krylov** solvers need only a matrix–vector product, so FEAX never assembles the Jacobian — the tangent action `J · v` is a forward-mode `jax.jvp` of the residual. This keeps memory low (no element Jacobian is ever materialized) at the cost of having no matrix entries for an entry-based preconditioner.
- **AMG** is an outer Krylov method preconditioned by a smoothed-aggregation multigrid hierarchy. The hierarchy is built (on the host, via PyAMG) from an *assembled* sample Jacobian and run on the GPU through AMJax; the outer Krylov applies the current operator. Its near-linear scaling makes it the fastest option for large scalar-elliptic systems, where it overtakes a direct factorization as the problem grows.

### Direct Solvers

```python
# Auto-select: cuDSS on GPU, cholmod/umfpack/spsolve on CPU
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(),
    linear=True, traced_params=traced_params)

# Explicit backend
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(solver="spsolve"),
    linear=True)
```

Available backends: `"auto"`, `"cudss"`, `"spsolve"`, `"cholmod"`, `"umfpack"`.

:::note cuDSS and `symmetric_elimination=False`
When `symmetric_elimination=False` is used, the Jacobian becomes non-symmetric (GENERAL). The auto-selection detects this and configures cuDSS in LU mode automatically. If you manually specify `DirectSolverOptions(solver="cudss")`, ensure that `CUDSSOptions(matrix_type=...)` matches the actual matrix symmetry.
:::

#### Factorization reuse (`reuse_factorization`)

`DirectSolverOptions(reuse_factorization=True)` (cuDSS only, **off by default**) keeps the cuDSS factorization alive after the forward solve and reuses it instead of factorizing again, in two situations:

- **Forward + adjoint (gradients).** The adjoint solves `Jᵀλ = v`. For a symmetric operator `J = Jᵀ`, so `λ = J⁻¹v` reuses the forward factors — one factorization per `value_and_grad` instead of two.
- **Multiple right-hand sides under `jax.vmap`.** A batch of load cases sharing the same matrix is factorized once and solved as a single multi-RHS cuDSS solve — see [Factorize once, solve many](#factorize-once-solve-many-reuse_factorization).

```python
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(solver="cudss", reuse_factorization=True),
    linear=True, traced_params=traced_params)
```

Reuse is applied **only** when the cuDSS `matrix_type` is `SYMMETRIC` or `SPD` (the default is `SYMMETRIC`). A `GENERAL` matrix — a genuinely non-symmetric operator, or one produced by `symmetric_elimination=False` — automatically falls back to two independent factorizations, so the flag is harmless to leave on in that case.

:::caution `reuse_factorization=True` assumes a truly symmetric operator
Reuse substitutes `J⁻¹` for `J⁻ᵀ` in the adjoint, which is exact **only** when `J = Jᵀ`. If the operator is actually non-symmetric but its `matrix_type` is (mis)declared `SYMMETRIC`/`SPD` — e.g. you forced `CUDSSOptions(matrix_type=...)`, or a near-symmetric tangent was detected as symmetric — the **gradient will be wrong**. When you cannot guarantee symmetry, leave `reuse_factorization=False` (the default) or set `matrix_type=GENERAL`.
:::

:::caution Memory: reused factors stay resident
A reused factorization (LU/Cholesky fill-in, plus device copies of the matrix) is held in a process-global cache (`SPINEAX_FACTOR_CACHE` entries, default 8) rather than freed after the solve. For large 3D problems the factors can be several GB; on unified-memory devices (e.g. GB10) this can trigger OOM. If memory is the binding constraint, keep `reuse_factorization=False` so each solve frees its factors immediately, or shrink the cache.
:::

Leave it **off** (the default) for: single forward solves with no gradient (nothing to reuse), non-symmetric or uncertain-symmetry operators, memory-tight large problems, and pipelines that interleave many distinct factorizations (a token whose factorization has been evicted from the cache raises at solve time). Turn it **on** when the operator is genuinely symmetric/SPD, you take gradients or `vmap` over many load cases, and the factors fit in memory.

### Krylov Solvers

```python
solver = fe.create_solver(problem, bc,
    solver_options=fe.KrylovSolverOptions(solver="cg"),
    linear=True, traced_params=traced_params)
```

Available backends: `"auto"`, `"cg"`, `"bicgstab"`, `"gmres"`.

Use `"cg"` for SPD matrices (symmetric problems), `"bicgstab"` or `"gmres"` for general matrices.

Because the Krylov path is matrix-free, there is no assembled matrix to draw a Jacobi (diagonal) preconditioner from on the standard path; convergence relies on the conditioning of the system itself. Reach for a direct solver when the problem is well-conditioned and fits in memory, and Krylov when memory is the binding constraint.

### AMG Solvers

`fe.AMGSolverOptions()` runs an outer Krylov method preconditioned by a smoothed-aggregation algebraic-multigrid (AMG) hierarchy. The hierarchy is built once from an assembled sample Jacobian via [PyAMG](https://github.com/pyamg/pyamg) and executed on the GPU through [AMJax](https://github.com/vboussange/AMJax). It is the fastest option for **large scalar-elliptic problems** (Poisson, steady heat conduction, implicit diffusion, the pressure-Poisson step of a flow solver), where its near-linear scaling overtakes a direct factorization as the mesh grows.

Requires the optional dependency: `pip install feax[amg]` (pulls in `amjax` + `pyamg`).

```python
solver = fe.create_solver(problem, bc,
    solver_options=fe.AMGSolverOptions(),
    linear=True, traced_params=traced_params)
```

The outer Krylov method is set by `solver` (`"auto"` → `cg` for SPD, else `gmres`; or `"cg"` / `"bicgstab"` / `"gmres"`).

#### Near-null-space (`near_nullspace`)

AMG quality hinges on the operator's **near-null-space** — the low-energy modes the coarse grid must represent. Plain (scalar) AMG works for scalar elliptic problems but **fails on vector elasticity**, which needs the rigid-body modes. `near_nullspace` accepts:

| Value | Meaning |
|---|---|
| `None` (default) | Smart default: rigid-body modes for a single `vec == dim` field (elasticity), else the constant near-null-space |
| `"rigid_body"` | Rigid-body modes built from the mesh node coordinates (6 in 3D, 3 in 2D) — the right choice for continuum elasticity |
| `"constant"` | The constant near-null-space (PyAMG default; correct for scalar Poisson / heat) |
| `"adaptive_sa"` | Estimate the near-null-space numerically (adaptive smoothed aggregation, relaxing `A x = 0`); use when no analytic modes are known. Set `num_nullspace` for the count |
| array `(n_dof, k)` | A user-supplied near-null-space, used verbatim |

```python
# Elasticity: rigid-body modes are auto-generated for vec == dim
solver = fe.create_solver(problem, bc,
    solver_options=fe.AMGSolverOptions(near_nullspace="rigid_body", solver="gmres"),
    linear=True, traced_params=traced_params)
```

#### Nonlinear solves and `rebuild_every`

For Newton solves (`linear=False`) the tangent changes each iteration, so a hierarchy built from the initial tangent can go stale. `rebuild_every` controls how the hierarchy is refreshed (the near-null-space is always reused):

| `rebuild_every` | Behavior |
|---|---|
| `None` (default) | **Adaptive lag**: reuse the hierarchy across Newton steps and rebuild only when a step's linear residual exceeds `lag_tol`. Few rebuilds, robust for strong nonlinearity. Runs as a host Newton loop (the per-step assembly and solve are individually JIT-compiled). |
| `0` | Build the hierarchy once and reuse it as a fixed preconditioner. Cheapest, and fully traced — composes with `jax.jit` / `jax.vmap` / `jax.grad`. Best when the operator changes little. |
| `k >= 1` | Rebuild every `k` Newton iterations (fixed lag). |

```python
# Large-deformation hyperelasticity: adaptive lag (default) keeps the
# preconditioner fresh as the tangent changes.
solver = fe.create_solver(problem, bc,
    solver_options=fe.AMGSolverOptions(near_nullspace="rigid_body", solver="gmres"),
    newton_options=fe.NewtonOptions(),
    traced_params=traced_params)
```

:::note JIT / vmap with AMG
`rebuild_every=0` lowers to a matrix-free Krylov solve with a fixed AMG preconditioner and is fully traced (`jit` / `vmap` / `grad`). The adaptive-lag and fixed-period Newton paths run a host loop (PyAMG setup cannot be traced) — they JIT each per-iteration kernel and are meant for concrete (non-`vmap`) calls; the adjoint still builds its preconditioner from the converged tangent.
:::

AMG is never chosen by auto-selection — it must be requested explicitly with `AMGSolverOptions`.

### Geometric Multigrid (GMG) — `NarrowBandCMG`

For domains that live on a **structured voxel grid** (a [`StructuredGrid`](./data_storage.md#structuredgrid--the-implicit-voxel-grid)), feax provides a matrix-free geometric multigrid solver, `fe.NarrowBandCMG`. Unlike the options above it is not a `solver_options` route through `create_solver` — it is a dedicated solver class that exploits the grid structure directly: transfers and smoothers come from grid coarsening (no algebraic setup), every level is matrix-free (one shared element stiffness), and the cost and memory are **O(active band)**, not O(domain). It currently targets 3D SIMP-style elasticity (`vec=3`, HEX8 voxels).

```python
grid   = fe.StructuredGrid((128, 64, 64))
cmg    = fe.NarrowBandCMG(grid, fixed_pred=lambda ni, nj, nk, nx, ny, nz: ni == 0,
                          nu=0.3, penal=3.0)
levels = cmg.build(active_cells)                    # cells carrying material
b      = cmg.load_vector(levels, tip_nodes, comp=2, value=-1.0)
solver = cmg.create_solver(levels, b)               # None -> cuDSS coarsest level
u      = solver(rho_cells)                          # bare array OR TracedParams
dc     = jax.grad(lambda r: np.dot(b, solver(r)))(rho_cells)   # differentiable
```

The MGPCG iteration is jittable and differentiable (implicit-diff `custom_vjp`), and the coarsest level is solved by cuDSS (`None` / `DirectSolverOptions`) or a matrix-free block-Jacobi Krylov method (`KrylovSolverOptions`, no cuDSS dependency). Where it fits:

| Situation | Use |
|---|---|
| Unstructured mesh | `Direct` / `Krylov` / `AMG` options above |
| Structured grid, moderate size | `Direct` (cuDSS) is usually simplest |
| Structured grid, very large / moving active band (topology optimization) | `NarrowBandCMG` — see the [Narrow-Band & Giga-Voxel tutorial](../advanced/narrowband_topology_optimization.md) |

See the [`solvers.cmg` API reference](../api/reference/feax/solvers/cmg.md) for `n_levels`, `bucket` (recompile amortization for moving bands), and the smoother knobs.

## Automatic Solver Selection

When `solver_options` is omitted or set to `solver="auto"`, FEAX selects a concrete solver from the hardware backend and the matrix properties. The default is `DirectSolverOptions(solver="auto")`: **prefer a direct solver, and use Krylov only when explicitly requested.**

### Selection Flow

```
solver_options=None  (or DirectSolverOptions(solver="auto"))
        │
        ▼
  Assemble sample Jacobian (needs traced_params)
        │
        ▼
  detect_matrix_property(J)
  ┌─────────┬─────────────┬──────────┐
  │   SPD   │  SYMMETRIC  │ GENERAL  │
  └────┬────┴──────┬──────┴────┬─────┘
       │           │           │
       ▼           ▼           ▼
  ┌─── GPU available? ───────────────┐
  │  Yes → cuDSS (Cholesky/LDLT/LU)  │  ← highest priority
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

:::note Auto-selection assembles a sample Jacobian
Resolving `"auto"` evaluates the Jacobian once to inspect its properties, so pass `traced_params` to `create_solver` when you rely on auto-selection (or specify the solver explicitly to skip the probe).
:::

### Direct Solver Priority

| Priority | Solver | Platform | Matrix types | Method |
|---|---|---|---|---|
| 1 | **cuDSS** | GPU | SPD / SYMMETRIC / GENERAL | Cholesky / LDLT / LU |
| 2 | **cholmod** | CPU | SPD only | Supernodal Cholesky |
| 3 | **umfpack** | CPU | SPD / SYMMETRIC / GENERAL | Multifrontal LU |
| 4 | **spsolve** | CPU | SPD / SYMMETRIC / GENERAL | SciPy sparse LU |

cuDSS adapts its factorization to the matrix property:

- **SPD** → Cholesky (fastest, lowest memory)
- **SYMMETRIC** → LDLT (no pivoting overhead)
- **GENERAL** → LU with partial pivoting

### Krylov Solver Selection

With `KrylovSolverOptions(solver="auto")`, the iterative method is chosen by matrix property:

| Matrix property | Solver | Notes |
|---|---|---|
| SPD | `cg` | Optimal for symmetric positive definite |
| SYMMETRIC | `bicgstab` | Robust for indefinite symmetric systems |
| GENERAL | `gmres` | General-purpose Krylov method |

### Matrix Property Detection

`detect_matrix_property(J)` performs two numerical checks on the assembled Jacobian (it accepts both the `CSRMatrix` from the assembled path and a JAX `BCOO`):

1. **Symmetry test** — compares $\|\mathbf{A}\mathbf{x} - \mathbf{A}^T\mathbf{x}\|$ for a random vector. Skipped when `matrix_view='UPPER'` or `'LOWER'` (symmetry guaranteed by construction).
2. **Positive-definiteness heuristic** — checks that all diagonal entries are positive (necessary but not sufficient for SPD).

### When to Override Auto Selection

| Situation | Recommendation |
|---|---|
| Problem too large for a direct solver (out of memory) | `KrylovSolverOptions()` |
| Large scalar-elliptic problem (Poisson / thermal / diffusion) | `AMGSolverOptions()` |
| Large elasticity / structural problem | `AMGSolverOptions(near_nullspace="rigid_body")` |
| Need a symmetric solver but using `symmetric_elimination=False` | `DirectSolverOptions(solver="spsolve")` or `"umfpack"` |
| Periodic BCs with prolongation matrix `P` | `KrylovSolverOptions()` (matrix-free) or `DirectSolverOptions()` / `AMGSolverOptions()` (assembled `PᵀJP`) |
| Structured voxel grid too large even for matrix-free Krylov / moving narrow band | [`fe.NarrowBandCMG`](#geometric-multigrid-gmg--narrowbandcmg) (GMG, O(band)) |
| Extra residual term (e.g. cohesive interface) | `KrylovSolverOptions()` (hybrid matrix-free) or `DirectSolverOptions()` (assembled sparse tangent) with `extra_residual_fn` |
| Extreme stiffness contrast (ill-conditioned Jacobian) | `DirectSolverOptions(solver="spsolve")` on CPU |

### Numerical Stability: cuDSS vs CPU Direct Solvers

cuDSS (GPU) uses LU factorization tuned for throughput, but its pivoting can be less robust than CPU solvers for extremely ill-conditioned systems. Problems with large stiffness contrasts (e.g. third-medium contact with γ₀ ≈ 1e-6, multi-material topology optimization) may show:

- Newton convergence stalling after the first iteration
- Line search returning very small step sizes (α ≈ 0)
- Residual not decreasing despite a non-zero Newton increment

In such cases, switch to a CPU direct solver with more robust pivoting:

```python
# CPU: SciPy sparse LU (most robust pivoting)
solver_options = fe.DirectSolverOptions(solver="spsolve")

# CPU: UMFPACK multifrontal LU (good balance of speed and stability)
solver_options = fe.DirectSolverOptions(solver="umfpack")
```

## Linear vs. Nonlinear: the `linear` flag

The `linear` flag selects the solve path:

| `linear` | Behavior |
|---|---|
| `True` | A single linear solve `J · Δu = -r`, then `u = u₀ + Δu`. |
| `False` (default) | Adaptive Newton iteration with Armijo line search, run to `NewtonOptions.tol` / `rel_tol` (capped at `max_iter`). |

Both paths are differentiable and compose with `jax.jit` / `jax.vmap` / `jax.grad`.

### Linear Problems (`linear=True`)

For linear elasticity and other linear PDEs, one linear solve is sufficient:

```python
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(),
    linear=True, traced_params=traced_params)
sol = solver(traced_params, initial)
```

### Nonlinear Problems (`linear=False`)

For hyperelasticity and other nonlinear problems, use the Newton path (the default):

```python
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(),
    newton_options=fe.NewtonOptions(tol=1e-8, max_iter=50),
    traced_params=traced_params)

initial = fe.zero_like_initial_guess(problem, bc)
sol = solver(traced_params, initial)
```

The Newton iteration runs adaptively (it stops as soon as the residual is converged). The forward is a traced `jax.lax.while_loop` — one Newton step per loop body, with no `pure_callback` node — so it composes with `jax.jit` and `jax.vmap` natively while still using data-dependent convergence and line search.

#### Newton Options

`fe.NewtonOptions` controls convergence criteria and line search:

```python
fe.NewtonOptions(
    tol=1e-8,                          # absolute residual tolerance
    rel_tol=1e-8,                      # relative residual tolerance
    max_iter=50,                       # maximum Newton iterations
    line_search_c1=1e-4,               # Armijo sufficient-decrease constant
    line_search_rho=0.5,               # backtracking shrink factor
    line_search_max_backtracks=30,     # max line search steps
    raise_on_line_search_failure=True, # raise if no descent step is found
)
```

`raise_on_line_search_failure=True` raises `NewtonLineSearchError` when Armijo backtracking exhausts `line_search_max_backtracks` without a descent step (effectively `α → 0`). A failed line search almost always signals an inconsistent Jacobian or a bad linear solve, so failing loudly is the safer default; set it to `False` to let the iteration continue with the best step found.

## Boundary Condition Elimination: `symmetric_elimination`

The `symmetric_elimination` parameter controls how Dirichlet boundary conditions are enforced in the Jacobian matrix. This choice can be critical for Newton convergence in nonlinear problems.

### `symmetric_elimination=True` (default) — Symmetric Elimination

Zeros both BC rows **and** columns in the Jacobian, then sets BC diagonal entries to 1:

$$
\mathbf{K}_\text{mod} = \begin{pmatrix} \mathbf{I} & \mathbf{0} \\ \mathbf{0} & \mathbf{K}_{11} \end{pmatrix}
$$

- Preserves matrix symmetry → symmetric solvers (CG, Cholesky) can be used
- Removes the K₁₀ coupling between BC DOFs and interior DOFs
- Suitable for: fixed BCs, linear problems, and problems where BC values are pre-applied to the initial guess

```python
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(),
    linear=True, symmetric_elimination=True)  # default
```

### `symmetric_elimination=False` — Non-symmetric Elimination

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
    symmetric_elimination=False,
    traced_params=traced_params)
```

### When to Use `symmetric_elimination=False`

Use non-symmetric elimination when the K₁₀ coupling matters for Newton convergence:

1. **Incremental loading** — BC values change per load step and the previous solution is reused as the initial guess. K₁₀ ensures prescribed displacement changes propagate correctly to interior DOFs.
2. **Large stiffness contrast** — e.g. third-medium contact where the background medium stiffness is scaled by γ₀ ≈ 1e-6. Without K₁₀, the first Newton increment overshoots in soft regions, causing divergence.
3. **Large-deformation nonlinear problems** — where BC DOF displacements are large and strongly coupled to interior DOFs.

:::tip Rule of thumb
If your Newton solver converges with `symmetric_elimination=True`, keep the default — it enables symmetric solvers and is slightly more efficient. Switch to `symmetric_elimination=False` when you see divergence or poor convergence you suspect is caused by the boundary-condition treatment.
:::

## Incremental Loading

For problems where the prescribed displacement or load is applied gradually over several steps, use `bc.replace_vals()` with `symmetric_elimination=False`:

```python
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(solver="spsolve"),
    newton_options=fe.NewtonOptions(tol=1e-6, max_iter=20),
    symmetric_elimination=False,
    traced_params=traced_params)

sol = fe.zero_like_initial_guess(problem, bc)

for step in range(1, num_steps + 1):
    # Update prescribed values (same DOF locations, different values)
    scale = step / num_steps
    new_vals = bc.bc_vals.at[move_bc_pos].set(max_disp * scale)
    bc_step = bc.replace_vals(new_vals)

    # Solve with updated BCs, reusing the previous solution as initial guess
    sol = solver(traced_params, sol, bc=bc_step)
```

Key points:

- `bc.replace_vals(new_vals)` creates a new `DirichletBC` with updated values but the same DOF locations — no solver rebuild needed.
- The previous solution `sol` is passed as the initial guess, giving Newton a good starting point.
- The solver's optional `bc=` keyword overrides the BC values without re-compiling.

## Custom Residual Contributions: `extra_residual_fn`

Some problems add a contribution to the global residual that does not come from the standard element weak form — for example a cohesive-zone traction on an interface, or a penalty/contact term. Supply it as `extra_residual_fn(sol_flat) -> residual_flat`:

```python
def cohesive_residual(u_flat):
    # e.g. the gradient of an interface potential, ∂Φ/∂u
    return jax.grad(lambda u: cohesive_energy(u, delta_max))(u_flat)

solver = fe.create_solver(problem, bc,
    solver_options=fe.KrylovSolverOptions(solver='cg', maxiter=200),
    newton_options=fe.NewtonOptions(tol=1e-6, max_iter=1000),
    extra_residual_fn=cohesive_residual,
    linear=False)
```

With `KrylovSolverOptions` this runs a **hybrid matrix-free Newton–Krylov** iteration: FEAX assembles the bulk Jacobian (a CSR matrix, which also supplies a Jacobi preconditioner), while the extra term's tangent is applied matrix-free via `jax.jvp`. The combined operator is

$$
\mathbf{J}_\text{total} \cdot \mathbf{v} = \mathbf{J}_\text{bulk} \cdot \mathbf{v} + \mathrm{jvp}(\text{extra\_residual},\, \mathbf{u},\, \mathbf{v}).
$$

With `DirectSolverOptions`, the extra term's sparsity is detected once via automatic sparse differentiation (asdex), colored, and assembled onto the merged (bulk ∪ extra) CSR pattern — the direct solver then factorizes the exact combined tangent.

Requirements: `extra_residual_fn` needs the nonlinear path (`linear=False`) and either `KrylovSolverOptions` or `DirectSolverOptions`; it cannot be combined with `P`. Dirichlet rows of the extra residual are zeroed automatically.

## Periodic Boundary Conditions: the reduced solver

Periodic constraints are imposed with a prolongation matrix `P` that maps a reduced (independent) DOF vector to the full DOF vector, `u = P · u_reduced`. Pass it via `P=`:

```python
solver = fe.create_solver(problem, bc,
    solver_options=fe.KrylovSolverOptions(solver='cg'),
    linear=True, P=P)
```

The reduced solver solves the reduced system `Pᵀ J P · u_reduced = -Pᵀ r` and supports two operator representations:

- **`KrylovSolverOptions`** — fully matrix-free: three matvecs (`P`, `J`, `Pᵀ`) per Krylov iteration, no assembled matrix.
- **`DirectSolverOptions` / `AMGSolverOptions`** — the reduced operator `PᵀJP` is *assembled* into a sparse matrix (pattern from a boolean triple product, values from colored probes of the matrix-free action, via asdex) and handed to the direct factorization or AMG hierarchy.

A Dirichlet BC that pins only *part* of a periodic equivalence class (e.g. one node of a tied pair) is contradictory and now raises a `ValueError` at build time — constrain either interior nodes only or the entire class. See [Periodic Boundary Conditions](../advanced/periodic_boundary_conditions.md) for building `P`.

## `MatrixView` for Symmetric Problems

When the problem is symmetric (most single-variable problems with `symmetric_elimination=True`), set `matrix_view='UPPER'` on the Problem to store only the upper triangle of the stiffness matrix. This roughly halves memory and enables optimized solvers (Cholesky):

```python
problem = MyProblem(mesh, vec=3, dim=3, matrix_view='UPPER')
```

:::caution
Do not combine `matrix_view='UPPER'` with `symmetric_elimination=False`, as the modified Jacobian is no longer symmetric.
:::

## Composing with `jax.jit`, `jax.vmap`, and `jax.grad`

Every solver path — linear, Newton, reduced, and hybrid — is built around a `custom_vjp` and composes with all three JAX transformations. The Newton forward pass is a traced `jax.lax.while_loop` (one Newton step per loop body) that vmaps natively as a batched while-loop, so data-dependent convergence and line search remain compatible with tracing.

```python
@jax.jit
def solve(iv, bc_vals):
    return solver(iv, initial, bc=bc.replace_vals(bc_vals))

sol = solve(iv, bc.bc_vals)  # first call triggers compilation
```

### Batched solves with `jax.vmap`

`jax.vmap` batches a solve over different boundary-condition values, material parameters, or loads in one vectorized call. All solver paths support it.

```python
import jax
import jax.numpy as np
import feax as fe

solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(),
    linear=True, traced_params=iv)

# Stack bc_vals into a batch (shape: [batch_size, num_bc_dofs])
vals_batch = np.stack([
    bc.bc_vals.at[-1].set(0.1),
    bc.bc_vals.at[-1].set(0.5),
])

solutions = jax.vmap(
    lambda v: solver(iv, initial, bc=bc.replace_vals(v))
)(vals_batch)
# solutions.shape = (batch_size, num_dofs)
```

The `bc=` keyword accepts a `DirichletBC` whose `bc_vals` differ from the original; `bc.replace_vals(v)` produces it. Internally the solver uses **parametric** Jacobian and residual functions that take `bc` as explicit data (rather than capturing it in a closure), so JAX can trace through the BC values under `vmap`.

:::danger BC locations must be identical within a batch
`jax.vmap` can batch over BC **values** (`bc_vals`) only. The BC **locations** (`bc_rows`, `bc_mask`) must be identical across the batch.
:::

This is a structural constraint, not a temporary limitation:

1. **The Jacobian sparsity pattern depends on BC locations** — BC elimination zeros the rows/columns at `bc_rows`. Different BC locations give different sparsity patterns, and `vmap` requires identical array shapes/structures across the batch.
2. **Direct solvers require a shared sparsity structure** — batched cuDSS factorization shares a single set of CSR offsets/columns across the batch; only the values and right-hand side vary.

When you need **different BC locations**, build a separate solver per location pattern and `vmap` within each group:

```python
# Group A: left edge fixed, right edge loaded
bc_a = fe.DirichletBCConfig([
    fe.DirichletBCSpec(location=left_edge, component='all', value=0.),
    fe.DirichletBCSpec(location=right_edge, component='x', value=0.),
]).create_bc(problem)
solver_a = fe.create_solver(problem, bc_a, linear=True, ...)

# Group B: bottom edge fixed, top edge loaded
bc_b = fe.DirichletBCConfig([
    fe.DirichletBCSpec(location=bottom_edge, component='all', value=0.),
    fe.DirichletBCSpec(location=top_edge, component='y', value=0.),
]).create_bc(problem)
solver_b = fe.create_solver(problem, bc_b, linear=True, ...)

# vmap within each group (values vary, locations fixed)
sols_a = jax.vmap(lambda v: solver_a(iv, bc=bc_a.replace_vals(v)))(vals_batch_a)
sols_b = jax.vmap(lambda v: solver_b(iv, bc=bc_b.replace_vals(v)))(vals_batch_b)
```

The construction cost (assembly, symbolic factorization) is paid once per BC-location pattern; within each group all `bc_vals` solves are batched efficiently.

#### Factorize once, solve many (`reuse_factorization`)

When the batched axis changes only the **right-hand side** — different `bc_vals`, loads, or source terms on the *same* matrix — add `reuse_factorization=True` (cuDSS). JAX keeps the unbatched matrix out of the batch, so FEAX factorizes **once** and solves the whole batch as a single multi-RHS cuDSS solve, in both the forward and the adjoint pass:

```python
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(solver="cudss", reuse_factorization=True),
    linear=True, traced_params=iv)

solutions = jax.vmap(lambda v: solver(iv, initial, bc=bc.replace_vals(v)))(vals_batch)
grads     = jax.vmap(jax.grad(loss))(vals_batch)
```

The `vmap` wall-clock then stays almost flat in the batch size `B` — one factorization amortized over `B` cheap solves, instead of `B` independent factor-and-solve calls. On a ~50k-DOF elasticity problem this is ≈15–25× faster at `B = 32` for both the forward solve and `value_and_grad`. No new API is needed; the reuse happens transparently inside `jax.vmap`.

:::note Only when the matrix is shared across the batch
This fast path needs the matrix to be **invariant** across the batch (load / BC-value / source-term studies). If the batched parameter changes the matrix itself — per-sample material or density, as in topology optimization — each sample needs its own factorization, and FEAX falls back to factorizing per batch element automatically. See [Factorization reuse](#factorization-reuse-reuse_factorization).
:::

Without `reuse_factorization`, `jax.vmap` still works but re-factorizes each batch element, so a batch of same-matrix load cases sees little speedup — the flag is what turns `vmap` into factor-once / solve-many.

### Gradients with `jax.grad`

All solver paths implement `jax.custom_vjp`. The backward pass uses the **implicit function theorem**: instead of differentiating through the Newton iterations, it solves a single adjoint linear system, giving exact gradients efficiently.

| Parameter | Differentiable | Notes |
|---|---|---|
| `TracedParams` (material params, loads) | Yes | All solver paths |
| `bc_vals` (BC prescribed values) | Yes | Pass `bc=bc.replace_vals(...)` |
| `initial_guess` | No | Gradient is `None` (not meaningful) |

To differentiate with respect to BC values, pass `bc=` built from `replace_vals`:

```python
def loss(bc_vals_arg):
    sol = solver(iv, initial, bc=bc.replace_vals(bc_vals_arg))
    return np.sum(sol ** 2)

grad_bc = jax.grad(loss)(bc.bc_vals)
```

#### Gradient correctness with `symmetric_elimination`

With `symmetric_elimination=True` (the default), the forward Jacobian uses symmetric elimination, which zeros the BC coupling columns (K₁₀). The backward pass automatically corrects the adjoint solution at BC DOFs so that gradients w.r.t. `bc_vals` stay exact — no user action is required, and the correction preserves compatibility with symmetric solvers (CG, Cholesky).

:::tip Verifying gradients
For critical applications, check analytic gradients against finite differences:

```python
analytic = jax.grad(loss)(bc_vals)

eps = 1e-5
for i in range(len(bc_vals)):
    p1 = bc_vals.at[i].add(eps)
    p2 = bc_vals.at[i].add(-eps)
    fd = (loss(p1) - loss(p2)) / (2 * eps)
    print(f"  i={i}: analytic={float(analytic[i]):.8f}, fd={float(fd):.8f}")
```
:::

## Summary

| Scenario | `linear` | `symmetric_elimination` | Solver options |
|---|---|---|---|
| Linear, fixed BCs | `True` | `True` | `DirectSolverOptions()` |
| Nonlinear, fixed BCs | `False` | `True` | `DirectSolverOptions()` |
| Nonlinear, incremental loading | `False` | `False` | `DirectSolverOptions(solver="spsolve")` |
| Large / memory-bound problem | `True`/`False` | `True` | `KrylovSolverOptions()` |
| Large scalar-elliptic (Poisson / thermal) | `True`/`False` | `True` | `AMGSolverOptions()` |
| Large elasticity / structural | `True`/`False` | `True` | `AMGSolverOptions(near_nullspace="rigid_body")` |
| Periodic BCs (prolongation `P`) | `True` | `True` | `KrylovSolverOptions()` (or `DirectSolverOptions()` / `AMGSolverOptions()`) with `P` |
| Extra residual term (cohesive) | `False` | — | `KrylovSolverOptions()` or `DirectSolverOptions()` with `extra_residual_fn` |
| Batched parameter study | `True`/`False` | `True` | any (all paths vmap) |
| Many load cases (same matrix), `vmap` | `True` | `True` | `DirectSolverOptions(solver="cudss", reuse_factorization=True)` |
