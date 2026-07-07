---
sidebar_label: Linear Buckling
---

# Differentiable Linear Buckling Analysis

This tutorial demonstrates linear (eigenvalue) buckling analysis with FEAX's differentiable buckling solver. We solve the buckling pencil for the lowest critical load factors and mode shapes, and differentiate the buckling factors with respect to the assembled stiffness matrices — the building block for buckling-constrained design and topology optimization.

## Overview

**Linear buckling** predicts the load level at which a structure loses stability. Given a reference load, a static solve produces a prestress state $\boldsymbol{\sigma}_0$; the question is by what factor $\lambda$ the reference load can be scaled before the structure buckles. This is the generalized eigenproblem (the *buckling pencil*)

$$
(\mathbf{K} + \lambda \mathbf{K}_g)\,\boldsymbol{\phi} = \mathbf{0}
$$

where:

- $\mathbf{K}$ is the **material tangent stiffness** (symmetric positive definite on the free DoFs)
- $\mathbf{K}_g$ is the **geometric (initial-stress) stiffness**, assembled from the prestress $\boldsymbol{\sigma}_0$ of the reference solution
- $\lambda$ is the **buckling factor**: the smallest positive $\lambda$ scales the reference load to the critical load, and $\boldsymbol{\phi}$ is the corresponding buckling mode

Internally, the solver recasts the pencil as a **shift-invert** standard eigenproblem

$$
\mathbf{T} = -\mathbf{K}^{-1}\mathbf{K}_g, \qquad \mathbf{T}\boldsymbol{\phi} = \mu\,\boldsymbol{\phi}, \qquad \mu = 1/\lambda
$$

so the *smallest positive* buckling factors become the *largest-magnitude* eigenvalues $\mu$ — exactly what Krylov eigensolvers (ARPACK, Arnoldi) converge fastest to. One sparse factorization of $\mathbf{K}$ supplies $\mathbf{K}^{-1}$ for all matrix-vector products.

## The Buckling Solver

`create_linear_buckling_solver()` (in [`feax.solvers.eigen`](../api/reference/feax/solvers/eigen.md), also exported as `fe.create_linear_buckling_solver`) returns a differentiable, jittable function `bf`:

```python
import feax as fe

bf = fe.create_linear_buckling_solver(
    free_dofs,        # int array: indices of unconstrained DoFs
    num_modes=4,      # number of lowest positive buckling factors
    solver="sparse",  # "sparse" | "dense" | "matfree" | "cudss"
    # extra backend kwargs forwarded via **solver_kw, e.g. num_matvecs=60
)

lambdas, modes = bf(K, Kg)   # K, Kg: feax.csr.CSRMatrix (no BCs applied)
```

- **`lambdas`** — `(num_modes,)` ascending positive buckling factors, **differentiable** w.r.t. `K` and `Kg`. If fewer positive modes exist than requested, the tail is padded with `inf`.
- **`modes`** — `(num_modes, N)` mode shapes scattered onto the full DoF vector (zero on constrained DoFs), normalized so $\boldsymbol{\phi}^T \mathbf{K} \boldsymbol{\phi} = 1$. Modes are for visualization and carry **no gradient** (`stop_gradient`).

`free_dofs` are the DoFs *not* fixed by Dirichlet BCs or rigid-body pins — $\mathbf{K}$ restricted to them must be positive definite. With a `DirichletBC` in hand:

```python
import numpy as onp

N = problem.num_total_dofs_all_vars
free_dofs = onp.setdiff1d(onp.arange(N), onp.asarray(bc.bc_rows))
```

### Analytic Eigenvalue Sensitivity

Differentiability is supplied by a `jax.custom_vjp` around the forward eigensolve, using the analytical sensitivity of a simple eigenvalue of the pencil (with the normalization $\boldsymbol{\phi}^T \mathbf{K} \boldsymbol{\phi} = 1$):

$$
\mathrm{d}\lambda = \lambda\, \boldsymbol{\phi}^T \mathrm{d}\mathbf{K}\, \boldsymbol{\phi} + \lambda^2\, \boldsymbol{\phi}^T \mathrm{d}\mathbf{K}_g\, \boldsymbol{\phi}
$$

Gradients flow into `K.data` and `Kg.data` — the CSR value arrays — and from there through the assembler to any upstream design parameters.

**Key properties:**

- Only *eigenvalues* are differentiated, never eigenvectors. This makes the VJP robust to the **degenerate buckling-mode pairs** (repeated eigenvalues from symmetry) that make a naive `eigh` VJP blow up.
- The cotangent matrices share the exact CSR structure (`indptr`/`indices`) of the primal `K`/`Kg`, so they compose cleanly with FEAX's CSR assembly VJP.
- The same sensitivity is used by every backend, so gradients agree across `"dense"`, `"sparse"`, and `"cudss"`.

### Jittability Contract

`bf` is jittable — `jax.jit(bf)` and `jax.jit(jax.grad(...))` both work — because only `K.data` / `Kg.data` are traced; the eigensolve runs as an opaque callback. The (constant) CSR sparsity `indices` must be captured on the first **eager** call:

```python
lambdas, modes = bf(K, Kg)      # 1st call OUTSIDE jit: captures CSR structure
bf_jit = jax.jit(bf)            # now safe to jit
lambdas, modes = bf_jit(K, Kg)
```

Calling a jitted `bf` before an eager call raises a clear `ValueError` telling you to do exactly this.

## Choosing a Backend

| `solver=` | Where it runs | Memory | When to use |
|---|---|---|---|
| `"sparse"` (default) | Host (SciPy ARPACK + CHOLMOD/SuperLU) | $O(n_f \cdot \text{ncv})$ | The workhorse. Leak-free under repeated calls — safe inside optimization loops. |
| `"dense"` | Device (`jnp.linalg.eigh`) | $O(n_f^2)$ | Exact reference for small/medium reduced systems; also the fallback when there are too few free DoFs for a Krylov solve. |
| `"matfree"` | Host + nested JAX callbacks | $O(n_f \cdot \text{num\_matvecs})$ | Opt-in JAX Arnoldi (requires `matfree`). Equivalent results, but **leaks memory in long loops** — reserve for single solves. |
| `"cudss"` | GPU (spineax cuDSS + traced Arnoldi) | sparse factor + Krylov basis | Opt-in GPU shift-invert. Fastest for large pencils; requires spineax + CUDA. |

### `"sparse"` — SciPy ARPACK shift-invert (default)

Matrix-free shift-invert: $\mathbf{K}$ restricted to the free DoFs is factorized once (CHOLMOD via `sksparse` when available, else SuperLU), and ARPACK drives a `LinearOperator` $v \mapsto -\mathbf{K}^{-1}(\mathbf{K}_g v)$. Never forms a dense $N \times N$ matrix. Everything is plain SciPy/NumPy on the host — no nested `jax.pure_callback` — so repeated calls (e.g. every iteration of a topology optimization) accumulate nothing.

Kwargs: `num_matvecs=30` (lower bound on the ARPACK Arnoldi basis size `ncv`; `ncv ≥ 2k+1` is enforced), `conv_tol=1e-2` (residual tolerance on the fundamental mode; `None` disables the check), `seed=0`, `verbose=False`.

### `"dense"` — exact generalized `eigh`

Densifies only the reduced free-DoF blocks and solves the symmetric-definite generalized problem $\mathbf{K}_g \mathbf{x} = \gamma \mathbf{K}\mathbf{x}$ (with $\lambda = -1/\gamma$) via a Cholesky reduction to a standard `jnp.linalg.eigh` — the `generalized_eigh()` utility, which runs on GPU. Exact and robust, but $O(n_f^2)$ memory: use it for small/medium reduced systems and as a reference to validate the Krylov backends.

### `"matfree"` — JAX Arnoldi (opt-in)

The same shift-invert via matfree's JAX Arnoldi with full reorthogonalization. Results match `"sparse"`, but $\mathbf{K}^{-1}$ is applied through a nested `jax.pure_callback` that accumulates state in the XLA runtime across calls — a **memory leak in long loops**. Prefer `"sparse"` unless you specifically need this path. Kwargs: `num_matvecs`, `conv_tol`, `seed`, `verbose`.

### `"cudss"` — GPU shift-invert (opt-in)

The GPU backend keeps the entire forward on the device: $\mathbf{K}$ (restricted to the free DoFs) is factorized **once per call** by cuDSS through spineax's factor-once/solve-many API, and each Arnoldi iteration inside a `lax.scan` is one device SpMV ($\mathbf{K}_g v$) plus one cuDSS SOLVE reusing those factors, with full (two-pass) reorthogonalization. Only the small $(m, m)$ Hessenberg eigendecomposition and the convergence check run in host callbacks. Factors live in spineax's fixed-capacity LRU cache, so nothing accumulates over an optimization loop.

```python
bf = fe.create_linear_buckling_solver(
    free_dofs, num_modes=4, solver="cudss",
    num_matvecs=60,          # Arnoldi subspace size (min nf, max(num_matvecs, 2*num_modes+10))
    matrix_type="spd",       # "spd" (Cholesky, default) or "symmetric" (LDLᵀ)
    device_id=0,             # CUDA device
)
```

Use `matrix_type="symmetric"` when a SIMP void floor leaves $\mathbf{K}$ near-indefinite. Requires spineax + CUDA; a clear `ImportError` is raised otherwise.

**Measured performance** — a 125k-DoF pencil, 4 modes, `num_matvecs=60`, on a GB10:

| Backend | Time | Eigenvalues |
|---|---|---|
| `"sparse"` | 124 s | reference |
| `"cudss"` (eager) | 3.2 s | identical |
| `"cudss"` (jitted) | 2.7 s | identical |

That is roughly a **40× speedup** for the same answer.

## Convergence Failures

All Krylov backends **fail loudly** instead of silently propagating garbage eigenvalues (and hence garbage sensitivities) into an optimizer. A `BucklingConvergenceError` is raised when:

- the sparse factorization of $\mathbf{K}$ fails ($\mathbf{K}$ near-singular on the free DoFs — e.g. a SIMP void floor set too low, or unconstrained rigid modes),
- no positive eigenvalue is found, or
- the fundamental Ritz pair's residual $\lVert \mathbf{T}\boldsymbol{\phi} - \mu\boldsymbol{\phi} \rVert / \lVert \boldsymbol{\phi} \rVert$ exceeds `conv_tol` (default `1e-2`; pass `conv_tol=None` to disable the check).

The usual fix is to **increase `num_matvecs`** (a larger Krylov subspace), or to verify that `free_dofs` really excludes all Dirichlet and rigid-body DoFs so $\mathbf{K}$ is SPD on them.

```python
from feax import BucklingConvergenceError

try:
    lambdas, modes = bf(K, Kg)
except BucklingConvergenceError:
    ...  # e.g. retry with a larger num_matvecs, or reject this design iterate
```

:::warning cudss and async dispatch
With `solver="cudss"` the convergence check runs inside a host callback on the GPU stream. Under JAX's asynchronous dispatch the error does not surface at the `bf(K, Kg)` call itself — it surfaces (possibly wrapped by the runtime) when the result is **synchronized**, e.g. at `jax.block_until_ready(lambdas)` or when the values are read on the host. Synchronize before treating a cudss solve as successful.
:::

## Worked Example

### Where K and Kg come from

Both inputs are `feax.csr.CSRMatrix` operators over the **full** DoF set, assembled **without** Dirichlet BCs — the constraint handling happens through `free_dofs`. FEAX's entry point for a raw assembled tangent is `fe.get_jacobian()`:

```python
import jax.numpy as jnp
import feax as fe

ts = fe.TracedStructure.from_problem(problem)
zero = jnp.zeros((num_nodes, 3))

# Material tangent stiffness of the (linear) problem, no BCs applied
K = fe.get_jacobian(problem, [zero], traced_params, ts)
```

The geometric stiffness $\mathbf{K}_g$ is the Jacobian of the initial-stress residual — the gradient of the geometric energy

$$
E_g(\mathbf{u}) = \int_\Omega \sigma_{0,ij}\, \tfrac{1}{2}\, \frac{\partial u_k}{\partial x_i} \frac{\partial u_k}{\partial x_j} \,\mathrm{d}V
$$

evaluated at the prestress $\boldsymbol{\sigma}_0$ from a reference static solve. This residual is linear in $\mathbf{u}$, so its Jacobian is exact and constant: assemble it at $\mathbf{u} = 0$.

For **layered composite solids**, FEAX ships a turnkey $\mathbf{K}_g$ element: `feax.mechanics.GeometricStiffnessSolid`, built by `feax.mechanics.create_layered_solid_geometric_stiffness(mesh, cell_sigma0, ply_thicknesses, ...)` (HEX8), with the per-quad prestress evaluated by `feax.mechanics.layered_quad_stress` on the reference solution. The prestress enters through `TracedParams`, so `Kg` stays differentiable w.r.t. the design:

```python
from feax.mechanics import create_layered_solid_geometric_stiffness, layered_quad_stress

problem_g, _ = create_layered_solid_geometric_stiffness(
    mesh, cell_sigma0=onp.zeros((num_cells, nq, 3, 3)),
    ply_thicknesses=ply_thicknesses)
ts_g = fe.TracedStructure.from_problem(problem_g)

sol = solver(traced_params, traced_structure=ts)          # reference static solve
u = problem.unflatten_fn_sol_list(sol)[0]
sigma0 = layered_quad_stress(problem._dNdxi_ref, cell_nodes, u[cells], C_quad)

K  = fe.get_jacobian(problem, [zero], traced_params, ts)
Kg = fe.get_jacobian(problem_g, [zero],
                     fe.TracedParams(volume_vars=(cell_nodes, sigma0)), ts_g)

lambdas, modes = bf(K, Kg)
```

For **other element types** there is no general-purpose $\mathbf{K}_g$ assembler yet: define a small `fe.Problem` whose residual is $\partial E_g / \partial \mathbf{u}$ for your element (the `GeometricStiffnessSolid` kernel is a good template) and assemble it with `fe.get_jacobian` at $\mathbf{u} = 0$ — or build the `CSRMatrix` yourself from any external assembly, as shown next.

### A minimal CSRMatrix-level pencil

The solver only needs the CSR triples, so it works with matrices from *any* source. Here is a self-contained pencil (SPD banded $\mathbf{K}$, negative-definite mass-like $\mathbf{K}_g$ — the same fixture used in `tests/test_eigen.py`):

```python
import numpy as onp
import scipy.sparse as sp
import jax.numpy as jnp

import feax as fe
from feax.csr import CSRMatrix

n = 150
rng = onp.random.default_rng(0)

# K: banded SPD (Laplacian-like)
K_sp = sp.diags([2.0 + rng.random(n), -0.5 * rng.random(n - 1),
                 -0.5 * rng.random(n - 1)], [0, 1, -1]).tocsr()
K_sp = (K_sp + K_sp.T) * 0.5 + 2.0 * sp.eye(n)

# Kg: symmetric negative definite (mass-like, sign-flipped)
M = sp.diags([4.0 + rng.random(n), onp.ones(n - 1), onp.ones(n - 1)],
             [0, 1, -1]).tocsr() * (1.0 / n)
Kg_sp = (-(M + M.T) * 0.5).tocsr()

def to_csr(A):
    A = A.tocsr(); A.sort_indices()
    return CSRMatrix(jnp.asarray(A.data), jnp.asarray(A.indptr),
                     jnp.asarray(A.indices), A.shape)

K, Kg = to_csr(K_sp), to_csr(Kg_sp)
free_dofs = onp.arange(3, n - 3)          # constrain a few end DoFs

bf = fe.create_linear_buckling_solver(free_dofs, num_modes=4, solver="sparse",
                                      num_matvecs=60)
lambdas, modes = bf(K, Kg)
print(lambdas)                            # ascending positive buckling factors
```

## Gradients for Design Sensitivities

Because `bf` is a `custom_vjp` function, buckling factors can sit directly inside a `jax.grad`-ed objective. Differentiating $\sum_i \lambda_i$ with respect to the CSR value arrays:

```python
import jax

bf = fe.create_linear_buckling_solver(free_dofs, num_modes=3, solver="sparse")
bf(K, Kg)                                 # eager call captures the CSR structure

def loss(k_data, kg_data):
    lam, _ = bf(CSRMatrix(k_data, K.indptr, K.indices, K.shape),
                CSRMatrix(kg_data, Kg.indptr, Kg.indices, Kg.shape))
    return jnp.sum(lam)

gK, gKg = jax.grad(loss, argnums=(0, 1))(K.data, Kg.data)
```

`gK` and `gKg` are the per-nonzero sensitivities $\partial(\sum \lambda) / \partial K_{rc}$ and $\partial(\sum \lambda) / \partial K_{g,rc}$. In a design loop you rarely stop there: `K.data` and `Kg.data` themselves come out of FEAX's differentiable assembly (`fe.get_jacobian` of `TracedParams`-parameterized problems), so composing the pieces —

```python
def objective(rho):
    tp = params_from_density(rho)                 # e.g. SIMP-scaled stiffness
    K = fe.get_jacobian(problem, [zero], tp, ts)
    Kg = assemble_Kg(tp)                          # prestress solve + Kg assembly
    lam, _ = bf(K, Kg)
    return -lam[0]                                # maximize the critical load

g = jax.grad(objective)(rho)
```

— gives end-to-end sensitivities of the critical buckling load w.r.t. the design variables. Modes carry no gradient, so keep them out of differentiated objectives (use `lambdas` only).

:::tip Backend consistency
The analytic sensitivity uses the backend's own $\boldsymbol{\phi}$, and `"dense"`, `"sparse"`, and `"cudss"` produce matching gradients (verified against finite differences in `tests/test_eigen.py`). You can prototype with `"dense"`, run production with `"sparse"`, and accelerate with `"cudss"` without touching the optimization code.
:::

## Summary

**Key concepts:**

- **Buckling pencil** $(\mathbf{K} + \lambda \mathbf{K}_g)\boldsymbol{\phi} = 0$, solved via shift-invert $\mathbf{T} = -\mathbf{K}^{-1}\mathbf{K}_g$, $\mu = 1/\lambda$
- **`fe.create_linear_buckling_solver(free_dofs, num_modes, solver=...)`** — differentiable, jittable `bf(K, Kg) -> (lambdas, modes)`
- **Analytic sensitivity** $\mathrm{d}\lambda = \lambda\,\boldsymbol{\phi}^T \mathrm{d}\mathbf{K}\,\boldsymbol{\phi} + \lambda^2\,\boldsymbol{\phi}^T \mathrm{d}\mathbf{K}_g\,\boldsymbol{\phi}$ — eigenvalues only, robust to degenerate mode pairs
- **Backends**: `"sparse"` (default, leak-free host ARPACK), `"dense"` (exact GPU `eigh`, small systems), `"matfree"` (opt-in, single solves only), `"cudss"` (GPU factor-once/solve-many, ~40× on a 125k-DoF pencil)
- **`BucklingConvergenceError`** — loud failure; increase `num_matvecs`, check that $\mathbf{K}$ is SPD on `free_dofs`

**Workflow:**

1. Solve the reference static problem for the prestress $\boldsymbol{\sigma}_0$
2. Assemble `K` (material tangent) and `Kg` (initial-stress Jacobian at $\mathbf{u}=0$) with `fe.get_jacobian` — no BCs applied
3. Build `free_dofs` from the constrained rows (`bc.bc_rows` plus rigid-body pins)
4. Call `bf(K, Kg)` once eagerly, then jit / `jax.grad` freely

## Further Reading

- [API: solvers.eigen](../api/reference/feax/solvers/eigen.md) — full docstrings for `create_linear_buckling_solver`, `generalized_eigh`, `BucklingConvergenceError`
- `tests/test_eigen.py` — backend cross-validation, gradient checks, and convergence-failure tests
- [Lattice Structure Homogenization](lattice_homogenization.md) — differentiable assembly with `TracedParams`, the upstream half of a buckling-constrained design loop
