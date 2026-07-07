---
sidebar_label: Sparse AD & Assembled Solvers
---

# Automatic Sparse Differentiation & Assembled Solver Paths

This tutorial covers `feax.asd`, FEAX's automatic sparse differentiation (ASD) module built on `asdex` (jaxpr-level sparsity detection + graph coloring), and the two assembled solver paths it unlocks in `fe.create_solver`:

1. **Assembled `extra_residual_fn`** — arbitrary coupling terms outside the mesh connectivity (springs, follower loads, penalty terms) solved with **direct** factorization instead of hybrid Krylov.
2. **Assembled reduced operator** $\mathbf{P}^T \mathbf{J} \mathbf{P}$ — periodic problems solved with **direct** or **AMG-preconditioned** solvers instead of matrix-free CG.

## Overview

FEAX's standard volume assembly does *not* use this module: the element-dense `jacfwd` + slot-map scatter is already optimal when the sparsity is the mesh connectivity (a global coloring would need "max row nnz" colors — more kernel evaluations than the element dimension). `feax.asd` covers the places where a sparse operator is needed but its pattern is **not** the plain connectivity:

- **Extra residual terms** — arbitrary user coupling with unknown sparsity. The pattern is *detected* from the function itself.
- **Reduced periodic operators** $\mathbf{P}^T \mathbf{J} \mathbf{P}$ — the pattern is a boolean triple product of known patterns; the values come from colored probes of the matrix-free action.
- **Design-space Hessians** $d^2 J / d\rho^2$ — symmetric (star) coloring + HVPs for second-order design optimization.
- **Verification** — checking FEAX's hand-built CSR pattern against detection on the actual residual.

All factories return functions with a **fixed sparsity structure** (jit-safe) that produce `feax.csr.CSRMatrix` — the operator type the FEAX solver stack consumes.

```python
from feax import asd
```

## The ASD Toolbox

### Sparse Jacobians and Hessians

`asd.sparse_jacobian_fn(f, x_sample)` detects the global sparsity of `f` by abstract interpretation of its jaxpr (no derivative evaluation), colors the pattern, and returns a fixed-structure producer:

```python
import jax
import jax.numpy as jnp
from feax import asd

def f(x):
    return jnp.sin(x[1:] * x[:-1]) + x[1:] ** 2

x = jnp.linspace(0.3, 1.2, 30)
jac_fn, pattern = asd.sparse_jacobian_fn(f, x)   # pattern: boolean scipy CSR
J = jax.jit(jac_fn)(x)                           # J: feax.csr.CSRMatrix
```

Each call costs `num_colors` AD passes of `f` — for a banded or local coupling this is a small constant, independent of the problem size. An explicit `pattern=` superset can be passed instead of `x_sample` to skip detection.

`asd.sparse_hessian_fn(g, x_sample)` is the second-order analogue for scalar `g`: star (symmetric) coloring + one HVP per color:

```python
def g(x):
    return jnp.sum((x[1:] - x[:-1]) ** 2 * x[1:]) + jnp.sum(x ** 3)

hess_fn, pattern = asd.sparse_hessian_fn(g, x)
H = jax.jit(hess_fn)(x)                          # CSRMatrix, matches jax.hessian(g)(x)
```

This is intended e.g. for design-space Hessians in second-order topology optimization, where the pattern is the filter-stencil overlap.

### Materializing a Linear Operator from Its Matvec

`asd.operator_assembler(pattern)` colors a pattern once and returns an `assemble(matvec) -> CSRMatrix` function that materializes **any** linear operator with that sparsity, using `num_colors` matvec probes (colored JVPs at zero):

```python
assemble = asd.operator_assembler(pattern)       # color once
K = assemble(lambda v: A @ v)                    # num_colors matvecs
```

This is how the reduced periodic operator $\mathbf{P}^T \mathbf{J} \mathbf{P}$ gets assembled: its action exists matrix-free, but its assembled form is needed for direct factorization or AMG hierarchy construction.

### Pattern Algebra

`asd.merge_csr_patterns(pattern_a, pattern_b)` computes the union of two CSR patterns plus everything needed to assemble and transpose on it: the merged `indptr`/`indices`, data-slot maps `slots_a`/`slots_b` (aligned with each input's CSR order), and transpose maps `T_perm`/`T_indptr`/`T_indices` for `feax.csr.transpose_with_maps`:

```python
from feax.csr import CSRMatrix, transpose_with_maps

m = asd.merge_csr_patterns(bulk_pattern, extra_pattern)
data = (jnp.zeros(m["nnz"])
        .at[m["slots_a"]].add(bulk_values)
        .at[m["slots_b"]].add(extra_values))
M = CSRMatrix(data, m["indptr"], m["indices"], m["shape"])
MT = transpose_with_maps(M, m["T_perm"], m["T_indptr"], m["T_indices"])
```

Two more pattern helpers round out the algebra:

- **`asd.connectivity_pattern(problem)`** — FEAX's assembled CSR pattern (mesh connectivity) as boolean CSR. Requires `MatrixView.FULL`.
- **`asd.reduced_operator_pattern(P, K_pattern)`** — the sparsity of the Galerkin product $\mathbf{P}^T \mathbf{K} \mathbf{P}$ by boolean triple product. Exact for boolean algebra, i.e. a superset of the numerical pattern — which is what coloring and decompression need.

### Verifying the Hand-Built Pattern

`asd.verify_jacobian_pattern(problem, traced_params)` detects the true Jacobian sparsity of the assembled (bulk, no-BC) residual and compares it with the connectivity pattern. Soundness requires the detected pattern to be contained in the connectivity pattern:

```python
report = asd.verify_jacobian_pattern(problem, traced_params)
assert report["ok"]              # detected ⊆ connectivity
print(report["coverage"])        # fraction of the pattern actually used
```

This is a useful sanity check when developing custom kernels or new element types: a `False` result means the residual couples DOFs the assembler does not account for.

## Path 1: Assembled `extra_residual_fn` (Direct Solvers)

`fe.create_solver(..., extra_residual_fn=...)` adds an arbitrary residual contribution `extra_residual_fn(sol_flat) -> residual_flat` on top of the FEM residual — discrete springs, ground springs, follower loads, penalty couplings. These terms can couple DOFs that share no element, so their Jacobian lies **outside** the mesh CSR pattern.

Previously only the **hybrid** path worked: with `KrylovSolverOptions`, the bulk Jacobian is assembled and the extra contribution enters as a `jax.jvp` matvec — restricting the linear solve to Krylov methods. With `DirectSolverOptions`, FEAX now takes the **assembled** path instead:

1. The extra term's Jacobian sparsity is detected **once** via `asd.sparse_jacobian_fn` (asdex jaxpr analysis + coloring), on a sample solution.
2. The extra pattern is merged with the bulk CSR pattern via `asd.merge_csr_patterns`.
3. Every Newton step assembles $\mathbf{J}_\text{total} = \mathbf{J}_\text{bulk} + \mathbf{J}_\text{extra}$ on the merged pattern, and the direct solver (cuDSS / spsolve / UMFPACK / CHOLMOD) factorizes the **exact** tangent.
4. Gradients flow through an adjoint solve on the assembled *transposed* merged operator (precomputed transpose maps), so `jax.grad` through the solver is exact.

### Example: Cubic Springs Between Non-Adjacent Nodes

Adapted from `tests/test_asd.py` — a cubic spring between two nodes that share no element, plus a cubic ground spring:

```python
import jax
import jax.numpy as jnp
import feax as fe

class Elasticity(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E):
            nu = 0.3
            mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lam * jnp.trace(eps) * jnp.eye(3) + 2 * mu * eps
        return stress

grid = fe.StructuredGrid((4, 3, 3))
mesh = grid.to_mesh()
problem = Elasticity(mesh, vec=3, dim=3, ele_type="HEX8")
tp = fe.TracedParams(volume_vars=(jnp.full(len(mesh.cells), 100.0),))

bc = fe.DirichletBCConfig([
    fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], 0.0), "all", 0.0),
    fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], 4.0), "x", 0.05),
]).create_bc(problem)

# DOF indices of two NON-adjacent nodes (y-components) and a ground node
i = int(grid.node_id(0, 1, 1)) * 3 + 1
j = int(grid.node_id(4, 2, 2)) * 3 + 1
g = int(grid.node_id(2, 1, 1)) * 3

def extra_residual_fn(sol):
    d = sol[i] - sol[j]
    f = 5.0 * (d + d ** 3)                       # cubic spring i <-> j
    r = jnp.zeros_like(sol)
    r = r.at[i].add(f).at[j].add(-f)
    return r.at[g].add(2.0 * sol[g] ** 3 + 0.5 * sol[g])   # ground spring

solver = fe.create_solver(
    problem, bc,
    solver_options=fe.DirectSolverOptions(),     # assembled ASD path
    newton_options=fe.NewtonOptions(tol=1e-10, max_iter=30),
    extra_residual_fn=extra_residual_fn,
    traced_params=tp,
)

ig = fe.zero_like_initial_guess(problem, bc)
sol = solver(tp, ig)

# Gradients flow through the merged-pattern adjoint:
loss = lambda tp_: jnp.sum(solver(tp_, ig) ** 2)
grads = jax.grad(loss)(tp)
```

Swapping `DirectSolverOptions()` for `KrylovSolverOptions(solver="cg", ...)` gives the hybrid matrix-free path — both produce the same solution and the same gradients; the assembled path is the one to use when a direct factorization is desired (ill-conditioned tangents, many Newton steps, stiff local nonlinearities).

:::note Requirements of the assembled extra path
- Requires `MatrixView.FULL` — the merged pattern is general nonsymmetric.
- Requires the nonlinear path (`linear=False`, the default).
- Dirichlet BC rows of the extra residual are zeroed automatically; under symmetric elimination the BC columns are also removed from the tangent, and the dropped coupling re-enters through the adjoint BC correction — so gradients w.r.t. `bc_vals` remain correct.
- The Newton loop is host-driven (like the hybrid path) with jitted per-iteration kernels: it composes with `jax.grad` but is not `vmap`-able.
:::

## Path 2: Assembled Reduced Operator $\mathbf{P}^T \mathbf{J} \mathbf{P}$ (Periodic Problems)

When `fe.create_solver(..., P=P)` is given a prolongation matrix (see [Periodic Boundary Conditions](periodic_boundary_conditions.md)), the solve happens in the reduced space. Two operator representations, chosen by `solver_options`:

- **`KrylovSolverOptions`** (matrix-free): the reduced operator is applied as three matvecs $\mathbf{P}^T(\mathbf{J}(\mathbf{P} v))$ — never assembled. Fully traced, jit/vmap/grad-friendly.
- **`DirectSolverOptions` / `AMGSolverOptions`** (assembled): the reduced pattern comes from the boolean triple product `asd.reduced_operator_pattern(P, asd.connectivity_pattern(problem))`, and the operator is materialized from its matrix-free action by `asd.operator_assembler` — `num_colors` matvec probes per assembly (roughly the max row nnz of the reduced pattern). The assembled operator is then factorized directly, or used to build an AMG hierarchy.

Periodic operators are symmetric after symmetric Dirichlet elimination, so the adjoint reuses the same operator, assembled at the converged state (never stale).

### Example: Direct and AMG Solves on a Periodic Cell

Adapted from `tests/test_asd.py` — a periodic cell in $x$, compressed in $z$:

```python
from feax.flat.pbc import PeriodicPairing, prolongation_matrix

L = 4
grid = fe.StructuredGrid((L, 3, 3))
mesh = grid.to_mesh()
problem = Elasticity(mesh, vec=3, dim=3, ele_type="HEX8")

pairings = [PeriodicPairing(
    location_master=lambda p: jnp.isclose(p[0], 0.0, atol=1e-8),
    location_slave=lambda p: jnp.isclose(p[0], float(L), atol=1e-8),
    mapping=lambda p: p + jnp.array([float(L), 0.0, 0.0]),
    vec=c) for c in range(3)]
P = prolongation_matrix(pairings, mesh, vec=3)

bc = fe.DirichletBCConfig([
    fe.DirichletBCSpec(lambda p: jnp.isclose(p[2], 0.0), "all", 0.0),
    fe.DirichletBCSpec(lambda p: jnp.isclose(p[2], 3.0), "z", -0.05),
]).create_bc(problem)

tp = fe.TracedParams(volume_vars=(E_cells,))     # e.g. a heterogeneous field
ig = fe.zero_like_initial_guess(problem, bc)

# Assembled + direct factorization of PᵀJP:
solver_direct = fe.create_solver(problem, bc, P=P,
    solver_options=fe.DirectSolverOptions())

# Assembled + AMG-preconditioned CG on PᵀJP (requires feax[amg]):
solver_amg = fe.create_solver(problem, bc, P=P,
    solver_options=fe.AMGSolverOptions(solver="cg", tol=1e-12, atol=1e-14))

sol = solver_direct(tp, ig)
grads = jax.grad(lambda tp_: jnp.sum(solver_direct(tp_, ig) ** 2))(tp)
```

Both agree with the matrix-free CG reference to solver tolerance, enforce periodicity exactly, and their gradients match the matrix-free adjoint and finite differences.

:::note Dirichlet BCs on periodically paired DOFs
The reduced path validates that no Dirichlet row lands on *part* of a periodic equivalence class: the $\mathbf{P}^T(\cdot)\mathbf{P}$ reduction would fold the eliminated row together with its periodic partners and silently dilute the constraint. Pin an unpaired DOF (e.g. an interior node) or constrain the entire periodic class — otherwise `create_solver` raises a `ValueError`.
:::

:::note Execution model
The assembled reduced path runs eagerly (host orchestration) and re-materializes the current operator each solve — appropriate for the "few solves, many right-hand-side" style of homogenization workloads. Keep `KrylovSolverOptions` when you need the fully traced, `jax.vmap`-able reduced solve.
:::

## AMG Solver Options

`fe.AMGSolverOptions` is a third linear-solver family alongside `DirectSolverOptions` and `KrylovSolverOptions`: a matrix-free outer Krylov solve preconditioned by one smoothed-aggregation AMG cycle. A sample Jacobian is assembled once, a PyAMG hierarchy is built on the host, converted to a JAX-native `amjax.MultilevelSolver`, and one V-cycle serves as the preconditioner `M`. Requires the optional `feax[amg]` dependency (`amjax` + `pyamg`).

```python
options = fe.AMGSolverOptions(
    solver="cg",                  # outer Krylov: "auto", "cg", "gmres", "bicgstab"
    tol=1e-10, atol=1e-12,        # outer Krylov tolerances
    maxiter=500,
    near_nullspace="rigid_body",  # the key knob for elasticity
    cycle="V",                    # "V", "W", "F"
    smoother_omega=0.67,          # damped Jacobi (undamped diverges on elasticity)
    smoother_sweeps=2,
    coarse_solver="pinv",         # "pinv", "lu", "qr", "jacobi"
    rebuild_every=None,           # Newton-only: hierarchy rebuild policy
)
```

The single most important field is **`near_nullspace`** — the low-energy modes $\mathbf{B}$ that the coarse grid must represent:

- **`"rigid_body"`** — rigid body modes built from the mesh node coordinates (6 in 3D, 3 in 2D). The right choice for continuum elasticity; without it, plain aggregation AMG fails on vector problems.
- **`"constant"`** — the constant near-null-space (PyAMG default; correct for scalar elliptic problems like Poisson/heat).
- **`"adaptive_sa"`** — estimate the modes numerically by adaptive smoothed aggregation (relaxing $\mathbf{A}x = 0$ from random starts). Expensive; worth it when no analytic near-null-space is known. `num_nullspace` sets how many.
- **an `(n_dof, k)` array** — a user-defined near-null-space, used verbatim.
- **`None`** (default) — smart default: rigid body modes for a single `vec == dim` field (elasticity), otherwise constant.

On the reduced path, the near-null-space is automatically projected into the reduced space (a weighted pullback through $\mathbf{P}$, exact for the 0/1 master-slave prolongations from `feax.flat.pbc.prolongation_matrix`).

For nonlinear (Newton) solves without `P`, **`rebuild_every`** controls how often the hierarchy is rebuilt from the current tangent: `None` (default) is adaptive lag — rebuild only when the lagged preconditioner degrades past `lag_tol`; `0` builds once and keeps the solve fully traced/jit/vmap/grad-able; `k >= 1` rebuilds every `k` iterations.

## When to Use Which

| Path | Options | Best for | Trade-offs |
|---|---|---|---|
| Matrix-free Krylov | `KrylovSolverOptions` | Well-conditioned SPD systems, large problems, anything that must `jit`/`vmap`/scan | No assembly, lowest memory; convergence degrades with ill conditioning / high material contrast |
| Assembled direct | `DirectSolverOptions` | Exact tangents, ill-conditioned systems, small-to-medium DOF counts, factor-reuse workflows | Factorization memory grows with fill-in; extra/reduced paths are host-driven (grad yes, vmap no) |
| Assembled AMG | `AMGSolverOptions` | Large elasticity / periodic problems where direct factorization is too big and plain CG stalls | Needs the right `near_nullspace`; host hierarchy build; optional `feax[amg]` dependency |

Rules of thumb:

- Start with the default (`DirectSolverOptions(solver="auto")` in `create_solver`) or CG for large SPD problems.
- Reach for the **assembled extra path** when an `extra_residual_fn` makes the hybrid Krylov solve fragile (stiff springs, penalty terms that wreck conditioning) — the direct solver factorizes the exact merged tangent.
- Reach for the **assembled reduced path** when periodic solves with matrix-free CG converge slowly (high-contrast unit cells): direct for exactness, AMG with `near_nullspace="rigid_body"` for scale.
- Use `asd.verify_jacobian_pattern` whenever you suspect the assembled pattern and the actual residual disagree.

## Summary

**Key concepts:**

- **`asd.sparse_jacobian_fn` / `asd.sparse_hessian_fn`** — coloring-based sparse derivatives with a fixed, jit-safe structure, returning `CSRMatrix` + pattern
- **`asd.operator_assembler`** — materialize any linear operator from its matvec by colored probing
- **`asd.merge_csr_patterns`** — pattern union with slot maps and transpose maps
- **`asd.reduced_operator_pattern` / `asd.connectivity_pattern`** — boolean pattern algebra for $\mathbf{P}^T \mathbf{J} \mathbf{P}$
- **`asd.verify_jacobian_pattern`** — runtime coverage check of the hand-built CSR pattern

**Solver paths unlocked:**

1. `extra_residual_fn` + `DirectSolverOptions` — exact merged tangent, direct factorization, merged-pattern adjoint
2. `P` + `DirectSolverOptions` / `AMGSolverOptions` — assembled $\mathbf{P}^T \mathbf{J} \mathbf{P}$ for periodic problems

## Further Reading

- [Periodic Boundary Conditions](periodic_boundary_conditions.md) — the `P` matrix and pairings
- [Lattice Structure Homogenization](lattice_homogenization.md) — a full periodic workflow
- `tests/test_asd.py` — complete working examples of every path on this page
- [API: feax.asd](../api/reference/feax/asd.md) — full ASD module reference
- [API: solver options](../api/reference/feax/solvers/options.md) — `DirectSolverOptions`, `KrylovSolverOptions`, `AMGSolverOptions`
