---
sidebar_label: Batched Solves (feax.vmap)
---

# Batched Solves: `feax.vmap`

`fe.vmap` batches a FEAX solver over its `TracedParams` and solves the batch well: a batch of **load cases** on one stiffness matrix is one factorization and a multi-right-hand-side solve, a batch of **designs** (one stiffness matrix per case) is a batched conjugate-gradient solve preconditioned by one shared factorization. It looks at which parameters are batched, decides which of the two it is, and picks the solve accordingly. A **Newton** solver is batched the same way, one shared factorization of the mean tangent per Newton step. The result is differentiable and composes with `jax.jit`.

```python
batched = fe.vmap(solver, in_axes=0, *, batch="auto", options=None,
                  preconditioner=None, chunk=None, return_info=False)
U = batched(traced_params, initial_guess=None, traced_structure=None)
```

## Why not `jax.vmap`?

`jax.vmap(solver)` already batches a FEAX solver, and it is the right tool for anything that is not a solve of this kind (see the [Vectorization Transform](../basic/vmap_transform.md) tutorial). But `jax.vmap` does not know what the batch *means*:

- When the batched parameters change only the **right-hand side** — tractions, source terms, boundary values on the same design — a direct solver can factorize once and back-substitute `N` right-hand sides. With `reuse_factorization=True` cuDSS does this under `jax.vmap` (see [Factorize once, solve many](../getting-started/solver.md#factorize-once-solve-many-reuse_factorization)).
- When the batched parameters change the **stiffness matrix** — material fields, SIMP densities, one design per case — `jax.vmap` of a direct solver factorizes every case and holds all `N` factorizations at once. That runs out of memory quickly: an A100 ran out at `N = 10` for a 254k-DOF cantilever. `jax.vmap` of a Krylov solver works, but every case iterates on its own with a per-case preconditioner and nothing is shared between the cases.

`fe.vmap` tells the two apart and, for the second case, solves all `N` systems together with a preconditioner built from the whole batch.

## Setting up

`fe.vmap` takes the `fe.Solver` object returned by `fe.create_solver(..., traced_params=tp)`, linear (`linear=True`) or the standard traced Newton solver. The solver carries what it was built from (`problem`, `bc`, `options`, `newton_options`, `sample_params`, `traced_structure`, `preconditioner`, `linear`) and can assemble its own operator (`solver.assemble(u, tp)`) and residual (`solver.residual(u, tp)`); `fe.vmap` uses these to inspect the batch and to assemble the batched operators. The sample `traced_params=` is required: it gives the parameter structure that `in_axes` is matched against and the point at which the batch kind is detected.

```python
import feax as fe

ts = fe.TracedStructure.from_problem(problem)
tp0 = fe.TracedParams(volume_vars=(E0,), surface_vars=[(t0,)])    # sample
solver = fe.create_solver(problem, bc, fe.DirectSolverOptions(), linear=True,
                          traced_params=tp0, traced_structure=ts)
u0 = fe.zero_like_initial_guess(problem, bc)
```

### `in_axes`

`in_axes` works like `jax.vmap`'s but applies to `traced_params` only; the initial guess and the `TracedStructure` are shared by every case. It is an `int` (the same axis on every leaf), `None`, or a `TracedParams`-shaped prefix of ints and `None`s:

```python
fe.vmap(solver, in_axes=0)                                                      # every leaf batched on axis 0
fe.vmap(solver, in_axes=fe.TracedParams(volume_vars=(None,), surface_vars=[(0,)]))   # only the traction
fe.vmap(solver, in_axes=fe.TracedParams(volume_vars=(1,), surface_vars=[(0,)]))      # E batched on axis 1
```

Every batched leaf must have the same batch size, and at least one leaf must be batched. The batch axis of the solution is always in front.

### rhs / lhs detection

With `batch="auto"` (the default) `fe.vmap` decides whether the operator depends on the batched leaves: it takes a JVP of the assembled CSR values with respect to those leaves at the solver's sample parameters, in a random direction (non-floating leaves get a zero tangent). A nonzero JVP means an **lhs** batch (one operator per case); zero means an **rhs** batch (the operator is shared). `batch="rhs"` or `batch="lhs"` overrides the detection. The result is the `kind` attribute of the returned `BatchedSolver`:

```python
loads_batch = fe.vmap(solver, in_axes=fe.TracedParams(volume_vars=(None,), surface_vars=[(0,)]))
loads_batch.kind       # "rhs"
fe.vmap(solver, in_axes=0).kind       # "lhs" — E changes the stiffness
```

## rhs batches: one factorization, many right-hand sides

An rhs batch is a plain `jax.vmap` of the solver, with the batch axes moved to the front. With a direct solver that is one factorization and a multi-right-hand-side solve; with a Krylov solver it is the solver's own iteration per case. The solver's own options and preconditioner are used — passing `options=` or `preconditioner=` for an rhs batch raises a `ValueError`.

```python
# load cases: same design, batched tractions -> detected as rhs
loads_batch = fe.vmap(solver, in_axes=fe.TracedParams(volume_vars=(None,), surface_vars=[(0,)]))
U = loads_batch(fe.TracedParams(volume_vars=(E,), surface_vars=[(loads,)]), u0, ts)   # (N, n)
```

## lhs batches: batched CG with a shared factorization

For an lhs batch the `N` operators are assembled on the shared CSR pattern of the problem — one `(N, nnz)` array of values on one `(indptr, indices)` — and solved together by batched preconditioned conjugate gradients (`feax.solvers.batched.batched_pcg`). Every CG iteration is one batched sparse matrix-vector product plus one application of a **batch-level preconditioner**; each case has its own convergence test and iteration count, and the loop runs until every case has converged or `maxiter` is reached.

```python
# designs: batched modulus fields -> detected as lhs
designs = fe.vmap(solver, in_axes=0, chunk=10, return_info=True,
                  options=fe.KrylovSolverOptions(solver="cg", tol=1e-8, maxiter=500))
U, info = designs(fe.TracedParams(volume_vars=(E_batch,), surface_vars=[(loads,)]), u0, ts)
print(info.iterations)      # CG iterations per case
```

### The batch-level preconditioner

The preconditioner follows the same contract as the single-solve preconditioners of `fe.create_solver` ([Preconditioners](../getting-started/solver.md#preconditioners)): a callable `preconditioner(op) -> M`. Inside `fe.vmap`, `op` is a `fe.LinearOperator` with `op.batched = True`, so

- `op.values` has shape `(N, nnz)` — every case's operator on the shared pattern `op.indptr`, `op.indices`,
- `op.diag` has shape `(N, n)`,
- `op.structure`, `op.problem`, `op.bc` are the solve's `TracedStructure`, `Problem` and `DirichletBC`,
- and the returned `M` must act on the whole batch: `M(R) -> Z` with `R` of shape `(N, n)`.

The default is **`fe.shared_factor()`** when cuDSS is available, else **`fe.jacobi()`**. A solver created with `preconditioner=fe.jacobi()`, `fe.shared_factor()` or an `fe.AMGPreconditioner` keeps that one; any other preconditioner of the solver is ignored here, because a single-solve `M` would not accept a batched operator — pass a batch-level one explicitly.

**`fe.shared_factor(matrix_type="spd", device_id=0)`** averages the `N` operators, factorizes the mean matrix `K_avg = mean_i K_i` once with cuDSS, and applies its back-substitution to all `N` right-hand sides at once: `M = K_avg⁻¹`. When the `K_i` are close to each other the preconditioned CG converges in a few iterations, so the batch costs about one factorization plus a few batched SpMVs and back-substitutions instead of `N` factorizations. This is the mean-based preconditioner of stochastic FEM, also known as approximate reanalysis, and it is only as good as the spread of the batch: as designs separate into different black-and-white topologies the iteration counts grow (tens to several hundred in the batched topology-optimization benchmark), so watch `info.iterations`. On a single operator (`(nnz,)` values) the "mean" is the operator itself and `M` is its exact inverse — CG converges in one iteration — so `fe.shared_factor()` can also be passed as the `preconditioner=` of an ordinary Krylov solver.

**`fe.jacobi()`** scales by the per-case diagonal, `(N, n)`, with no factorization. Far more iterations than the shared factorization at large sizes, but no cuDSS dependency.

**`fe.AMGPreconditioner`** (from `fe.amg(...)`) is accepted too: its fixed hierarchy is vmapped over the cases.

Any callable of the same contract can be passed — see [Writing a batch-level preconditioner](#writing-a-batch-level-preconditioner).

### `options`

`options=` sets how the lhs batch is solved:

- `fe.KrylovSolverOptions(solver="cg", tol=..., atol=..., maxiter=...)` configures the batched CG. Case `i` has converged when `‖r_i‖ ≤ max(tol · ‖b_i‖, atol)`. Only `solver="cg"` (or `"auto"`) is accepted: the batched solve is conjugate gradients, for SPD operators, so the solver must use `symmetric_elimination=True` (the default). The default is the solver's own `KrylovSolverOptions` if it has them, otherwise `cg` with `tol=1e-8`, `atol=1e-10`, `maxiter=1000`.
- `fe.DirectSolverOptions(...)` instead factorizes **every** case (`jax.vmap` of a direct solver, built from the batched solver's problem if the solver is not itself direct). Without `chunk=` all `N` factorizations are held at once and `fe.vmap` warns; with `chunk=` the memory is bounded by one group of factorizations. `preconditioner=` is an error with a direct solver.

### `chunk`

`chunk=C` processes the batch in groups of `C` cases with `jax.lax.map` over the groups (the last group is padded by repeating the last case; the padded results are dropped). Each group builds its own preconditioner — with `fe.shared_factor()` that is one factorization per group of the mean of that group — and only one group's operators are in memory at a time, so the memory bound is `C` operators (plus one factorization) regardless of `N`. Grouping similar cases together makes the group mean a closer preconditioner: in the batched topology-optimization benchmark, neighbouring load angles give similar designs, and one factorization per group of 10 needs far fewer CG iterations than one global factorization for all 100. `chunk` also applies to rhs batches and to the direct-solver path.

### `return_info`

With `return_info=True` the call returns `(U, info)`, where `info` is a `feax.solvers.batched.BatchedSolveInfo` named tuple with one entry per case:

| Field | Shape | Meaning |
|---|---|---|
| `iterations` | `(N,)` int32 | CG iterations taken by the case (rhs batch: zeros; direct options: ones) |
| `converged` | `(N,)` bool | reached the tolerance within `maxiter` |
| `residual` | `(N,)` float | final relative residual `‖r‖ / ‖b‖` |

## The returned callable

`fe.vmap` returns a `BatchedSolver`, called as `batched(traced_params, initial_guess=None, traced_structure=None)`; it exposes `kind`, `solver`, `in_axes`, `options`, `preconditioner` and `chunk`. The initial guess defaults to `fe.zero_like_initial_guess(problem, bc)` and the `TracedStructure` to the solver's; a `fe.Solution` is accepted as the initial guess. The result is a `fe.Solution` with the batch axis in front (`U.dofs` is `(N, n)`, `U.field(0)` is `(N, num_nodes, vec)`), or a raw `(N, n)` array if the solver was created with `return_solution=False`.

The callable is **differentiable** with respect to `traced_params`. An lhs batch has a custom VJP: the adjoint systems `K_i λ_i = v_i` are solved as one batch exactly like the forward (same preconditioner, same options, same chunking), and the cotangent of an unbatched leaf (an `in_axes` entry of `None`) is summed over the cases. An rhs batch differentiates through the solver's own `custom_vjp` under `jax.vmap`. The `info` output carries no gradient.

```python
tp_batch = fe.TracedParams(volume_vars=(E_batch,), surface_vars=[(loads,)])
dJ = jax.grad(lambda tp: jnp.sum(designs(tp, u0, ts)[0].dofs ** 2))(tp_batch)
dJ.volume_vars[0].shape    # (N, n_nodes): one sensitivity per design
```

It composes with `jax.jit` — put the batch axes wherever they are in your data and let `in_axes` describe them:

```python
solve = jax.jit(lambda E, L, ts: designs(fe.TracedParams(volume_vars=(E,), surface_vars=[(L,)]), None, ts)[0].dofs)
U = solve(E_batch, loads, ts)
```

## Full example

This mirrors `tests/test_feax_vmap.py`: a 3D cantilever with a nodal Young's modulus field as a volume variable and a tip traction as a surface variable.

```python
import jax
import jax.numpy as jnp
import feax as fe

class Elasticity(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E):
            nu = 0.3
            mu = E / (2 * (1 + nu))
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lam * jnp.trace(eps) * jnp.eye(3) + 2 * mu * eps
        return stress

    def get_surface_maps(self):
        return [lambda u, x, t: jnp.array([0.0, 0.0, -t])]

mesh = fe.mesh.box_mesh((2.0, 0.5, 0.5), mesh_size=0.25, element_type="HEX8")
right = lambda p: jnp.isclose(p[0], 2.0, atol=1e-6)
problem = Elasticity(mesh, vec=3, dim=3, ele_type="HEX8", location_fns=[right])
bc = fe.DirichletBCConfig([fe.DirichletBCSpec(
    lambda p: jnp.isclose(p[0], 0.0, atol=1e-6), "all", 0.0)]).create_bc(problem)
ts = fe.TracedStructure.from_problem(problem)
u0 = fe.zero_like_initial_guess(problem, bc)

# N designs (log-normal modulus fields) and N tip loads
N = 5
n_nodes = mesh.points.shape[0]
E_batch = jnp.exp(0.2 * jax.random.normal(jax.random.PRNGKey(1), (N, n_nodes)))
t1 = fe.TracedParams.create_uniform_surface_var(problem, 1e-3)
loads = jnp.stack([t1 * (1.0 + 0.2 * k) for k in range(N)])

tp0 = fe.TracedParams(volume_vars=(E_batch[0],), surface_vars=[(t1,)])
solver = fe.create_solver(problem, bc, fe.DirectSolverOptions(), linear=True,
                          traced_params=tp0, traced_structure=ts)

# load cases: same design, batched tractions -> detected as rhs, one factorization
loads_batch = fe.vmap(solver, in_axes=fe.TracedParams(volume_vars=(None,), surface_vars=[(0,)]))
U = loads_batch(fe.TracedParams(volume_vars=(E_batch[0],), surface_vars=[(loads,)]), u0, ts)
print(loads_batch.kind, U.dofs.shape)          # rhs (5, n)

# designs: batched modulus fields -> detected as lhs, batched CG with one shared factorization
designs = fe.vmap(solver, in_axes=0, chunk=10, return_info=True,
                  options=fe.KrylovSolverOptions(solver="cg", tol=1e-8, maxiter=500))
tp_batch = fe.TracedParams(volume_vars=(E_batch,), surface_vars=[(loads,)])
U, info = designs(tp_batch, u0, ts)
print(designs.kind, info.iterations, info.converged.all())   # lhs [..] True

# sensitivities of a batch objective: one adjoint batch, solved the same way
dJ = jax.grad(lambda tp: jnp.sum(designs(tp, u0, ts)[0].dofs ** 2))(tp_batch)
```

Each case of the lhs batch matches a direct solve of that case to the CG tolerance, and `dJ.volume_vars[0][i]` matches `jax.grad` of the single-case solver at case `i`.

## Writing a batch-level preconditioner

A batch-level preconditioner is any `preconditioner(op) -> M` that handles `op.batched = True`: it may read `op.values` (`(N, nnz)`), `op.diag` (`(N, n)`), the shared pattern, and the problem data, and must return an `M` that maps `(N, n)` residuals to `(N, n)` corrections. `N` is static under tracing, so the preconditioner can split the batch however it likes. Two examples.

A per-case Jacobi (what `fe.jacobi()` does, written out):

```python
def scaled_jacobi(shift=1e-12):
    def preconditioner(op):
        inv = 1.0 / (op.diag + shift)              # (N, n)
        return lambda R: inv * R
    return preconditioner

batched = fe.vmap(solver, in_axes=0, preconditioner=scaled_jacobi())
```

Per-group factorizations built from `fe.shared_factor()` — one cuDSS factorization per group of `G` consecutive cases, each applied to its own rows of the batch. This is what `chunk=G` does implicitly, but written out it keeps all cases in a single CG loop (one convergence test over the whole batch) and lets the groups be chosen freely:

```python
def grouped_factor(G):
    def preconditioner(op):
        N = op.values.shape[0]
        Ms = []
        for start in range(0, N, G):
            sub = fe.LinearOperator(values=op.values[start:start + G], indptr=op.indptr,
                                    indices=op.indices, structure=op.structure,
                                    problem=op.problem, bc=op.bc, batched=True)
            Ms.append((start, fe.shared_factor()(sub)))
        def M(R):
            return jnp.concatenate([M_g(R[s:s + G]) for s, M_g in Ms], axis=0)
        return M
    return preconditioner

batched = fe.vmap(solver, in_axes=0, preconditioner=grouped_factor(10))
```

The same contract is what `fe.create_solver(preconditioner=...)` uses for a single solve, where `op.batched` is `False` and `op.values` is `(nnz,)`: a preconditioner that branches on `op.batched` (or uses `...` indexing like `fe.jacobi` does) works in both places.

Underneath, the kernel is available directly as `feax.solvers.batched.batched_pcg(values, indptr, indices, b, M, *, tol, atol, maxiter, x0) -> (X, BatchedSolveInfo)` for CSR values of shape `(N, nnz)` (or a single `(nnz,)` matrix shared by all right-hand sides) and `b` of shape `(N, n)`, and `feax.solvers.batched.batched_spmv(values, indptr, indices)` returns the batched matrix-vector product `spmv(V, X) -> (N, n)`.

## Newton solvers

A Newton solver (`create_solver(..., linear=False)`) is batched by the same call. For a nonlinear problem the tangent depends on the state, which depends on every batched parameter, so a Newton batch is always an **lhs** batch (`batch="rhs"` still forces plain `jax.vmap`). All cases iterate together in one traced loop:

1. per Newton step, the residuals and the `N` tangents of the current iterates are assembled on the shared pattern;
2. the batch-level preconditioner is built from the `N` tangents — by default one shared factorization of their mean — and the `N` Newton directions come from one batched CG (`options`);
3. the Armijo line search of the solver's `NewtonOptions` runs per case;
4. a case that has converged (`tol` / `rel_tol`), or whose line search failed, keeps its iterate while the others go on, until every case is done or `max_iter`.

Every case's **final iterate is returned whether or not it converged**, and `info` is a `BatchedNewtonInfo` that says what happened to each case: `iterations` (Newton steps), `converged` (residual reached `tol` / `rel_tol`), `line_search_failed` (stopped because the Armijo search found no step), `cg_iterations` (CG iterations summed over the steps) and `residual` (final relative residual). A case that is neither converged nor failed ran out of `max_iter`. Under tracing no exception can be raised, so `raise_on_line_search_failure` does not apply here; check the flags. The custom VJP solves the `N` adjoint systems of the converged tangents the same way, so the tangents are assumed symmetric (a hyperelastic tangent derived from an energy is), with `symmetric_elimination=True`. `chunk=` groups the cases exactly as for a linear batch, with one preconditioner per group and per step. The initial guess may be shared, `(n,)`, or given per case, `(N, n)`.

```python
newton = fe.create_solver(problem, bc, fe.DirectSolverOptions(), linear=False,
                          newton_options=fe.NewtonOptions(tol=1e-8), traced_params=tp0,
                          traced_structure=ts)
batched = fe.vmap(newton, in_axes=0, return_info=True,
                  options=fe.KrylovSolverOptions(solver="cg", tol=1e-10, maxiter=500))
U, info = batched(fe.TracedParams((), [(loads,)]), u0, ts)   # N load levels, N Newton iterations each
```

## Limitations

- **Newton: the traced path only.** A Newton solver with `extra_residual_fn`, or with an AMG preconditioner that rebuilds its hierarchy (`rebuild_every != 0`), runs a host loop and cannot be batched here; use `jax.vmap`.
- **The sample parameters are required.** The solver must have been created with `traced_params=`; it is how `fe.vmap` sees the parameter structure and detects the batch kind.
- **No reduced solvers.** A solver with `projection=` (periodic boundary conditions) is not supported yet.
- **CG only, SPD operators.** The lhs batch is conjugate gradients on the symmetric, BC-eliminated operator: `options.solver` must be `"cg"` (or `"auto"`) and the solver must use `symmetric_elimination=True`. Non-symmetric problems can use `options=fe.DirectSolverOptions()` with `chunk=`.
- **One batch kind per `fe.vmap`.** The rhs/lhs decision is made once, at the sample parameters, from the leaves `in_axes` marks as batched; if a leaf changes the operator only away from the sample, pass `batch="lhs"` explicitly.
- **The batch is over `traced_params`.** Boundary-condition *values* (`bc=bc.replace_vals(v)`) are not a batch axis of `fe.vmap`; use `jax.vmap` with `reuse_factorization=True` for those (see the [Solver Guide](../getting-started/solver.md#batched-solves-with-jaxvmap)).

## Benchmarks

`examples/benchmark/bench_topopt_batched_oc.py` runs 100 independent topology optimizations (one design per load case) at 500k DOF through `fe.vmap` with `fe.shared_factor`, in groups of `--case-chunk` designs (`chunk=`), with `--k0 group` (one factorization per group) or `--k0 global` (one for the whole batch). On an A100, 100 designs in 10 OC iterations took 1970 s, about 21 s per design, with CG growing to ~210 (mean) / ~420 (max) iterations as the designs separated. See the [benchmark README](https://github.com/Naruki-Ichihara/feax/tree/main/examples/benchmark) for the numbers behind the shared factorization alone (`bench_shared_factor_pcg.py`: ~27 CG iterations and a 6.9–19.5× lower cost per case than a direct solve of each case at 500k DOF for a mild modulus spread).
