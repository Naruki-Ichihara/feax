---
sidebar_label: Batched Topology Optimization (shared PCG)
---

# Many Topology Optimizations at Once: Shared-Factor PCG

This tutorial optimizes **B independent designs of the same structure together**, one per load direction. Every design has its own stiffness matrix, so each iteration needs B forward solves with B different matrices. Instead of B factorizations, the solves go through `fe.vmap`: the B matrices are assembled on their shared sparsity pattern and solved by batched conjugate gradients preconditioned by **one** cuDSS factorization of their mean (`fe.shared_factor`) per group of `SHARE` designs, at first the whole batch. The factorizations are made once and reused across iterations, and only refreshed when the CG counts say they have gone stale. Everything else is the ordinary `feax.gene` toolkit, which accepts a batch of designs `(B, n)` directly.

The complete script is [`examples/advance/batched_topology_optimization_shared_pcg.py`](https://github.com/Naruki-Ichihara/feax/blob/main/examples/advance/batched_topology_optimization_shared_pcg.py). The batching itself is explained in [Batched Solves: feax.vmap](batched_solves.md).

## Overview

1. A SIMP cantilever (HEX8, 48×12×12 elements, about 25k DOF, by default) and `B = 8` load directions, 0° to 157.5° from −z in the y-z plane, applied on a patch of the free end.
2. One linear solver, batched with `fe.vmap` over the per-design modulus field and load. `share=` sets how many designs share one factorization (`SHARE=0`, the default, is the whole batch: the designs start identical, so the mean is exact), `chunk=` how many stiffness matrices are assembled at a time (`CHUNK=4`); the group mean is accumulated chunk by chunk, so memory is bounded by `CHUNK` matrices whatever the group size.
3. The group factorizations are made once (`batched.factorize`) and reused by the forward and adjoint solves of every iteration (`batched(..., factors=factors)`).
4. The `gene` responses on the batch: density filter, Heaviside projection (fixed `BETA`, no continuation), compliance (one load per design) and volume, all on `(B, n)` arrays.
5. Sensitivities by `jax.value_and_grad` through the batched solver (its custom VJP solves the B adjoint systems the same way, with the same factors).
6. `gene.oc_update`: the optimality-criteria update with the volume constraint met per design, pure JAX.
7. When the mean CG count exceeds `CG_THRESHOLD` (default 200), the batch is regrouped by **operator similarity**, `SHARE` is **halved**, and the new groups are factorized at the current designs. The companion script [`batched_topology_optimization_shared_pcg_gcmma.py`](https://github.com/Naruki-Ichihara/feax/blob/main/examples/advance/batched_topology_optimization_shared_pcg_gcmma.py) drives the same problem with `BatchedGCMMA` from `feax.experimental.mma` (see below).

Requires cuDSS (`spineax`). Without it `fe.vmap` falls back to `fe.jacobi()`, there is nothing to factorize or share (the batch just runs `CHUNK` designs at a time), and the CG iteration counts are much higher.

## Settings

Both scripts read their size and strategy from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `NX`, `NY`, `NZ` | 48, 12, 12 | elements along the cantilever (e.g. `NX=80 NY=20 NZ=20` for ~100k DOF) |
| `B` | 8 | designs, one load direction each |
| `SHARE` | 0 | designs per shared factorization; 0 is the whole batch |
| `CHUNK` | 4 | stiffness matrices assembled at a time; 0 is all at once |
| `ITERS` | 40 | optimization iterations |
| `BETA` | 2.0 | Heaviside projection sharpness, fixed for the whole run (no continuation) |
| `CG_THRESHOLD` | 200 | when the mean CG count exceeds it: regroup, halve `SHARE`, factorize again (0: never) |
| `MAX_GROUPS` | 0 | cap on the number of groups the halving may create (0: down to one design per group) |
| `REORDER_EVERY` | 0 | also re-sort and re-factorize every this many iterations (0: only when regrouping) |
| `REPORT_EVERY` | 5 | print a progress line every this many iterations |
| `OUT_DIR`, `CSV_DIR`, `RUN_TAG` | | where the VTU and the CSV logs go, and the run label in the CSVs |

Every iteration is logged to `history.csv` (one row per iteration: compliance ratios, CG mean / min / max, `share`, number of groups, and the update / factorize / evaluate / regroup times) and `designs.csv` (one row per design per iteration), flushed as they are written.

## Loads: one per design

The surface map takes the two in-plane traction components. For the batch they are arrays with a leading design axis:

```python
class SIMPElasticity(fe.Problem):
    def get_surface_maps(self):
        def traction(u, x, ty, tz):          # returns minus the applied traction
            return jnp.array([0.0, ty, tz])
        return [traction]

theta = onp.pi * onp.arange(B) / B
unit = fe.TracedParams.create_uniform_surface_var(problem, 1.0)
lead = (B,) + (1,) * unit.ndim
TY = jnp.asarray(-TRACTION * onp.sin(theta)).reshape(lead) * unit[None]   # (B, ...)
TZ = jnp.asarray(TRACTION * onp.cos(theta)).reshape(lead) * unit[None]
loads = (TY, TZ)
```

## The solver, batched

```python
cg = fe.KrylovSolverOptions(solver="cg", tol=1e-7, atol=1e-12, maxiter=3000)
solver = fe.create_solver(problem, bc, cg, linear=True, traced_params=tp0, traced_structure=ts)
in_axes = fe.TracedParams(volume_vars=(0,), surface_vars=[(0, 0)])

def make_batched(share):
    cudss = fe.solvers.options.has_cudss()
    return fe.vmap(
        solver, in_axes=in_axes, options=cg,
        preconditioner=fe.shared_factor() if cudss else fe.jacobi(),
        chunk=CHUNK, share=(share or B) if cudss else None, return_info=True)

share = SHARE                  # None (SHARE=0): the whole batch
batched = make_batched(share)
```

The single-case solver is only the template `fe.vmap` batches. Built with the batch's own CG options it holds nothing; a `DirectSolverOptions` template would keep a pre-warmed cuDSS factorization the batch never uses (7 GB at 500k DOF). The modulus field and both load components carry the batch axis; `fe.vmap` probes that the modulus changes the operator and takes the lhs path. `return_info=True` returns the CG iteration count of every design, the number to watch.

## `gene` on a batch of designs

The `gene` functions do not know about `fe.vmap`; they only see the shapes. A density filter, the volume response and the compliance response all take `(B, n)` arrays, and the compliance takes one load per design through a tuple entry of `surface_load_params` whose elements carry the design axis:

```python
filt = gene.create_density_filter(mesh, FILTER_RADIUS)          # DensityFilter: (B, n) -> (B, n)
volume = gene.create_volume_fn(problem)                           # (B, n) -> (B,)
compliance = gene.create_compliance_fn(
    problem, surface_load_params=[(jnp.asarray(-TRACTION * onp.sin(theta)),
                                   jnp.asarray(TRACTION * onp.cos(theta)))])   # (B, n_dof) -> (B,)
vol_w = jax.grad(volume)(jnp.zeros(n_nodes))                      # nodal volume weights
dv = jax.vjp(filt, jnp.zeros(n_nodes))[1](vol_w)[0]              # ... pulled back through the linear filter
```

The volume constraint is imposed on the filtered density. The filter is linear, so its weight on the design variables is the filter's transpose applied to the nodal weights, one vector-Jacobian product.

`create_density_filter` finds the neighbours within the radius with a k-d tree and returns a `DensityFilter`, a pytree whose leaves are the sparse weights. It can therefore be an *argument* of a jitted function, which is how the scripts use it below.

## Factorize once, evaluate many times

```python
def make_evaluate(batched):
    @jax.jit
    def _evaluate(X, beta, order, factors, ts, u0, filt):
        back = jnp.argsort(order)

        def objective(X):
            rho = gene.heaviside_projection(filt(X), beta=beta)
            E = gene.simp(rho, E0, PENAL, EPS)
            tp = fe.TracedParams(volume_vars=(E[order],), surface_vars=[(TY[order], TZ[order])])
            U, info = batched(tp, u0, ts, factors=factors)
            c = compliance(U.dofs[back])                           # (B,)
            return jnp.sum(c), (c, volume(filt(X)), info.iterations[back])
        (_, (c, v, its)), dc = jax.value_and_grad(objective, has_aux=True)(X)
        return c, dc, v, its

    def evaluate(X, beta, order, factors):
        return _evaluate(X, beta, order, factors, ts, u0, filt)
    return evaluate


def factorize(X, beta, order):
    if not fe.solvers.options.has_cudss():
        return None                                   # Jacobi: nothing to factorize
    n_groups = -(-B // (share or B))
    os.environ["SPINEAX_FACTOR_CACHE"] = str(n_groups)   # every group's entry stays resident
    rho = gene.heaviside_projection(filt(X), beta=beta)
    tp = fe.TracedParams(volume_vars=(gene.simp(rho, E0, PENAL, EPS)[order],),
                         surface_vars=[(TY[order], TZ[order])])
    return batched.factorize(tp, u0, ts)
```

Three things in this evaluation matter at scale:

- **The mesh-sized arrays are arguments.** The `TracedStructure`, the initial guess, the filter and the factors go into the jitted function as arguments. Closed over, they would be compiled in as constants, and at ~10⁵ DOF and up XLA spends minutes, on one thread, folding them. `fe.vmap` reads the CSR pattern from the `ts` of the call for the same reason.
- **The factors are reused.** `factorize` factorizes the mean stiffness matrix of every group once, at the current designs, in the solve order. Every evaluation after that passes the same `factors`: the forward and adjoint solves only assemble the designs' own matrices for CG, with no mean assembly and no factorization. The mean-matrix preconditioner only has to be close, not current. `SPINEAX_FACTOR_CACHE` is set to the number of groups so that every group's cuDSS entry stays on the GPU while it is reused (the scripts start with it at 1, before spineax is imported).
- **The solve order is a permutation.** The batch is solved in the order `order` (so that each group of `share` consecutive designs is similar) and the results are permuted back, so the optimizer never sees the ordering.

Summing the B compliances gives the B sensitivities in one backward pass, since the designs are independent (`dc[i]` depends on `X[i]` only).

## The optimization loop: regroup only when CG says so

```python
X = jnp.full((B, n_nodes), VOLFRAC)
order = jnp.arange(B)
factors = factorize(X, BETA, order)            # the uniform start: one exact factorization for all
c, dc, v, its = evaluate(X, BETA, order, factors)
refactor = False
for k in range(1, ITERS + 1):
    X = gene.oc_update(X, dc, VOLFRAC, dv=dv, move=MOVE)
    if refactor:                               # new groups: factorize them at the current designs
        factors = factorize(X, BETA, order)
        refactor = False
    c, dc, v, its = evaluate(X, BETA, order, factors)
    if CG_THRESHOLD and float(its.mean()) > CG_THRESHOLD:
        order = similarity_order(X, BETA)      # regroup by operator similarity
        new_share = max(1, (share or B) // 2)  # halve SHARE (down to MAX_GROUPS groups)
        refactor = True
        if new_share != (share or B):
            share = new_share
            batched = make_batched(share)      # a new batched solver (recompiles once)
            evaluate = make_evaluate(batched)
```

(Simplified from the script, which also times and logs every step.) `gene.oc_update` on a `(B, n)` array runs one bisection per design, so every design meets the volume fraction exactly; `VOLFRAC` may also be an array `(B,)`.

The regrouping takes effect at the next evaluation, which factorizes the new groups; the sensitivities just computed stay valid. Halving `SHARE` down to 1 ends at one exact factorization per design (CG converges in one iteration); more, closer factorizations beat hundreds of CG iterations. `MAX_GROUPS` caps the number of groups, since every group's factorization stays on the GPU while it is reused.

**Why order by operator similarity.** A shared factorization of a group mean works for design `i` as well as `K_i` is close to that mean: the eigenvalues of `M K_i` lie between the extreme element-wise ratios of the two stiffnesses. So the right notion of "close" is in the operator, not in the design variables, and it is the same for any parametrization (density, modulus, fibre orientation, ...). `batched.operator_signature(tp)` returns the log-diagonal of every design's stiffness matrix, `(B, n_dof)` (assembled `CHUNK` designs at a time), and `gene.similarity_order` sorts the batch along the principal direction of those signatures, so that consecutive designs, and hence each group of `share`, are similar:

```python
def similarity_order(X, beta):
    rho = gene.heaviside_projection(filt(X), beta=beta)
    sig = batched.operator_signature(
        fe.TracedParams(volume_vars=(gene.simp(rho, E0, PENAL, EPS),), surface_vars=[loads]))
    return gene.similarity_order(sig)
```

`gene.group_spread` gives the within-group spread of any feature as a one-number diagnostic. `REORDER_EVERY` additionally re-sorts and re-factorizes on a fixed schedule.

## GCMMA over the batch

The GCMMA version lives in its own script, `batched_topology_optimization_shared_pcg_gcmma.py`, with the same problem, loads, batched solver, factorization strategy and `gene` responses.

`jax.vmap` of `GCMMA` would evaluate every design on its own — B separate solves per evaluation, exactly what the shared factorization avoids. `BatchedGCMMA` turns this around: the value-and-gradient function receives the whole batch `X` `(B, n)` and returns `(B,)`, `(B, n)`, `(B, m)`, `(B, m, n)` from **one** batched evaluation, while the asymptotes, the convex subproblems and the conservativity (inner) loop are per design, vectorized. A design whose approximation is already conservative keeps its candidate while the others re-solve, so the loop runs until every design is conservative or `max_inner` is reached.

```python
from feax.experimental.mma import BatchedGCMMA

mesh_args = (ts, u0, filt)              # mesh-sized: threaded through the GCMMA step as an argument

def value_and_grad(X, beta, order, factors, mesh):
    c, dc, v, its = evaluate(X, beta, order, factors, mesh)   # one batched solve; dc[i] depends on X[i] only
    g = v / VOLFRAC - 1.0                                       # (B,)  volume constraint per design
    dg = jnp.broadcast_to(dv / (VOLFRAC * vol_w.sum()), X.shape)
    return c / c0, dc / c0[:, None], g[:, None], dg[:, None, :], (c, v, its)

opt = BatchedGCMMA(value_and_grad, 0.0, 1.0, n_constraints=1, max_inner=MAX_INNER,
                   move=MOVE, has_aux=True)
step = jax.jit(opt.step)
state = opt.init(X, beta, order, factors, mesh_args)
for k in range(1, ITERS + 1):
    state = step(state, beta, order, factors, mesh_args)   # extra args ride along to value_and_grad
    c, v, its = state.aux
```

Here `evaluate` is the jitted function itself and takes `(ts, u0, filt)` as one `mesh` argument. The regrouping is the same as in the OC loop; when `SHARE` changes, the batched solver, the evaluation and the jitted step are rebuilt, while the GCMMA state is unaffected (the designs keep their original order, only the solve is permuted).

The state is an `MMAState` with a leading batch axis; `solve`, `solve_with_history` and `run` work on it unchanged, and a stopping test fires when every design passes it. Since the B designs are independent, the sum of the B scaled compliances has the block-diagonal gradient the batch needs, one backward pass for all designs. **Every inner iteration is one more batched evaluation for all designs**, so `MAX_INNER` is kept small (3); with hundreds of designs the slowest one sets the count, so watch `state.inner_iterations` next to the CG counts.

## Result at scale

With the current strategy at **503k DOF × 100 designs** on an A100 40GB (one shared factorization for the whole batch, `SHARE=0`), the factorization of the uniform initial design, reused throughout, kept CG at ~130 iterations, below `CG_THRESHOLD`, so the batch was never regrouped. 8 iterations took **1303 s**, against **2191 s** when every evaluation re-factorizes. Before the changes behind this strategy (the CSR pattern read from the call's `TracedStructure`, the filter and mesh arrays passed into `jit`, chunked adjoint cotangents, `share=` separate from `chunk=`), this configuration did not compile in reasonable time and then ran out of memory.

## Earlier runs (previous strategy)

:::note Measured with the previous version of the scripts
The tables below were made before the factor reuse described above: every evaluation built its own factorizations (one per `chunk` of neighbouring designs, starting at `chunk=4`), the threshold halved `chunk` itself, and the Heaviside sharpness followed a continuation (`beta` 1 → 2 → 4 → 8). The scripts now use a fixed `BETA` (default 2), start with one shared factorization for the whole batch and reuse it, so the counts and times differ. They are kept for what they show about the shared preconditioner.
:::

OC, fixed grouping by load angle (`chunk=4`), 8 designs at 25k DOF:

```
iter  beta  c/c0 mean  c/c0 max          vol  CG iters mean/max    time
   1     1     0.6386    0.6505  [0.400, 0.400]      13.2 /   15     3.3s
  10     1     0.3698    0.3957  [0.400, 0.400]      39.5 /   47    14.7s
  20     2     0.2415    0.2856  [0.400, 0.400]     154.0 /  172    44.2s
  30     4     0.1641    0.1915  [0.400, 0.400]     314.8 /  339    95.5s
  40     8     0.1386    0.1566  [0.400, 0.400]     344.8 /  365   149.1s

40 iterations of 8 designs in 149.1 s (0.466 s per design per iteration)
final c/c0 per design: [0.1566 0.1369 0.1242 0.1369 0.1566 0.1369 0.1242 0.1369]
```

OC with similarity ordering and the group size halved above the threshold:

```
iter  beta  c/c0 mean  c/c0 max          vol  CG iters mean/max  chunk    time  order (groups of CHUNK)
   1     1     0.6386    0.6505  [0.400, 0.400]      13.2 /   15      4     5.4s  [2, 1, 3, 4, 0, 5, 7, 6]
  15     2     0.2631    0.3280  [0.400, 0.400]     137.5 /  164      4    29.1s  [6, 5, 7, 4, 0, 3, 1, 2]
  20     2     0.2415    0.2856  [0.400, 0.400]     102.5 /  121      2    48.4s  [6, 5, 7, 4, 0, 3, 1, 2]
  25     4     0.1650    0.1930  [0.400, 0.400]       1.0 /    1      1    68.1s  [6, 5, 7, 4, 0, 3, 1, 2]
  40     8     0.1386    0.1566  [0.400, 0.400]       1.0 /    1      1   104.9s  [2, 3, 1, 4, 0, 5, 7, 6]

40 iterations of 8 designs in 104.9 s (0.328 s per design per iteration)
```

GCMMA (`MAX_INNER = 3`), fixed grouping:

```
iter  beta  c/c0 mean  c/c0 max          vol  CG iters mean/max    time   inner iters (per design)
   1     1     0.7464    0.7706  [0.365, 0.368]      10.0 /   11    11.1s  [0 0 0 0 0 0 0 0]
  20     2     0.2353    0.2682  [0.400, 0.400]     159.8 /  180   133.3s  [0 0 0 0 0 0 0 0]
  35     8     0.1339    0.1443  [0.400, 0.400]     344.5 /  364   327.4s  [3 2 3 2 3 2 3 2]
  40     8     0.1329    0.1432  [0.400, 0.400]     345.2 /  362   413.8s  [3 2 2 2 3 2 2 2]

40 iterations of 8 designs in 413.8 s (1.293 s per design per iteration)
```

What these runs show, independent of the strategy:

- Early on, all designs are close to the uniform start and the mean matrix is an excellent preconditioner: a dozen CG iterations per solve.
- As the layouts sharpen, the designs separate and the counts grow to a few hundred, then plateau. This is the cost of one factorization standing in for several different matrices; it is still one factorization in memory instead of several.
- The mirror-symmetric load pairs (0° and 90°, 22.5° and 67.5°, ...) converge to the same compliance, a useful check.
- Smaller groups paid off once the counts plateaued: the designs and compliances were identical to the fixed-grouping run (the solves are exact to tolerance either way), and the time fell from 129 s to 105 s, all of it after the group size dropped to one. On this problem the similarity ordering alone did not change the counts (the load angles already order the designs along one path); it matters when the batch has no such natural order. At larger DOF a factorization is relatively dearer, so a higher threshold is right — hence the default of 200 with reuse.
- GCMMA reached a slightly lower compliance in the same 40 iterations (mean `c/c0` 0.133 vs 0.139) and met the volume constraint through the constraint rather than by construction. Once the projection sharpened, two to three inner iterations per design appeared and each step cost three to four batched evaluations, which is why that run took 2.8× the OC time.

The final designs are written to `output_batched_topopt_shared_pcg/designs.vtu` (`output_batched_topopt_shared_pcg_gcmma/` for GCMMA), one point field per design.
