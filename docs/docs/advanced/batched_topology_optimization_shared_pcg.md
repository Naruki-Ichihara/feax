---
sidebar_label: Batched Topology Optimization (shared PCG)
---

# Many Topology Optimizations at Once: Shared-Factor PCG

This tutorial optimizes **B independent designs of the same structure together**, one per load direction. Every design has its own stiffness matrix, so each iteration needs B forward solves with B different matrices. Instead of B factorizations, the solves go through `fe.vmap`: the B matrices are assembled on their shared sparsity pattern and solved by batched conjugate gradients preconditioned by **one** cuDSS factorization of their mean (`fe.shared_factor`), in groups of neighbouring designs. Everything else is the ordinary `feax.gene` toolkit, which accepts a batch of designs `(B, n)` directly.

The complete script is [`examples/advance/batched_topology_optimization_shared_pcg.py`](https://github.com/Naruki-Ichihara/feax/blob/main/examples/advance/batched_topology_optimization_shared_pcg.py). The batching itself is explained in [Batched Solves: feax.vmap](batched_solves.md).

## Overview

1. A SIMP cantilever (HEX8, about 25k DOF) and `B = 8` load directions, 0° to 157.5° from −z in the y-z plane, applied on a patch of the free end.
2. One linear solver, batched with `fe.vmap` over the per-design modulus field and load; `chunk=4` groups designs with neighbouring load angles so that each group shares one factorization.
3. The `gene` responses on the batch: density filter, Heaviside projection, compliance (one load per design) and volume, all on `(B, n)` arrays.
4. Sensitivities by `jax.value_and_grad` through the batched solver (its custom VJP solves the B adjoint systems the same way).
5. `gene.oc_update`: the optimality-criteria update with the volume constraint met per design, pure JAX.
6. Two measures that keep the shared preconditioner effective as the designs separate, both driven by the CG count: the batch is re-ordered by **operator similarity** so that each group shares a factorization close to its members, and the group size is **halved** whenever the mean CG count exceeds a threshold. The companion script [`batched_topology_optimization_shared_pcg_gcmma.py`](https://github.com/Naruki-Ichihara/feax/blob/main/examples/advance/batched_topology_optimization_shared_pcg_gcmma.py) drives the same problem with `BatchedGCMMA` from `feax.experimental.mma` (see below).

Requires cuDSS (`spineax`). Without it `fe.vmap` falls back to `fe.jacobi()` and the CG iteration counts are much higher.

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
TY = jnp.asarray(-TRACTION * onp.sin(theta)).reshape(lead) * unit[None]   # (B, ...)
TZ = jnp.asarray(TRACTION * onp.cos(theta)).reshape(lead) * unit[None]
loads = (TY, TZ)
```

## The solver, batched

```python
solver = fe.create_solver(problem, bc, fe.DirectSolverOptions(), linear=True,
                          traced_params=tp0, traced_structure=ts)
batched = fe.vmap(
    solver, in_axes=fe.TracedParams(volume_vars=(0,), surface_vars=[(0, 0)]),
    options=fe.KrylovSolverOptions(solver="cg", tol=1e-7, atol=1e-12, maxiter=3000),
    preconditioner=fe.shared_factor(), chunk=CHUNK, return_info=True)
```

The modulus field and both load components carry the batch axis. `fe.vmap` probes that the modulus changes the operator and takes the lhs path; `return_info=True` returns the CG iteration count of every design, the number to watch.

## `gene` on a batch of designs

The `gene` functions do not know about `fe.vmap`; they only see the shapes. A density filter, the volume response and the compliance response all take `(B, n)` arrays, and the compliance takes one load per design through a tuple entry of `surface_load_params` whose elements carry the design axis:

```python
filt = gene.create_density_filter(mesh, FILTER_RADIUS)          # (B, n) -> (B, n)
volume = gene.create_volume_fn(problem)                           # (B, n) -> (B,)
compliance = gene.create_compliance_fn(
    problem, surface_load_params=[(-TRACTION * sin(theta), TRACTION * cos(theta))])   # (B, n_dof) -> (B,)
vol_w = jax.grad(volume)(jnp.zeros(n_nodes))                      # nodal volume weights
dv = jax.vjp(filt, jnp.zeros(n_nodes))[1](vol_w)[0]              # ... pulled back through the linear filter
```

The volume constraint is imposed on the filtered density. The filter is linear, so its weight on the design variables is the filter's transpose applied to the nodal weights, one vector-Jacobian product.

## Evaluate and update

```python
@jax.jit
def evaluate(X, beta):
    def objective(X):
        rho = gene.heaviside_projection(filt(X), beta=beta)
        tp = fe.TracedParams(volume_vars=(gene.simp(rho, E0, PENAL, EPS),), surface_vars=[loads])
        U, info = batched(tp, u0, ts)
        c = compliance(U.dofs)                                     # (B,)
        return jnp.sum(c), (c, volume(filt(X)), info)
    (_, (c, v, info)), dc = jax.value_and_grad(objective, has_aux=True)(X)
    return c, dc, v, info

X = jnp.full((B, n_nodes), VOLFRAC)
for k in range(1, ITERS + 1):
    beta = BETA[...]                                               # Heaviside continuation
    X = gene.oc_update(X, dc, VOLFRAC, dv=dv, move=MOVE)
    c, dc, v, info = evaluate(X, beta)
```

Summing the B compliances gives the B sensitivities in one backward pass, since the designs are independent (`dc[i]` depends on `X[i]` only). `gene.oc_update` on a `(B, n)` array runs one bisection per design, so every design meets the volume fraction exactly; `VOLFRAC` may also be an array `(B,)`.

## Keeping the preconditioner effective: similar groups, adaptive group size

The counts above come from a fixed grouping by load angle. Two things, both read off `info.iterations`, do better.

**Order the batch by operator similarity.** A shared factorization of a group mean works for design `i` as well as `K_i` is close to that mean: the eigenvalues of `M K_i` lie between the extreme element-wise ratios of the two stiffnesses. So the right notion of "close" is in the operator, not in the design variables, and it is the same for any parametrization (density, modulus, fibre orientation, ...). `batched.operator_signature(tp)` returns the log-diagonal of every design's stiffness matrix, `(B, n_dof)`, and `gene.similarity_order` sorts the batch along the principal direction of those signatures, so that consecutive designs, and hence each `chunk`, are similar:

```python
def similarity_order(X, beta):
    rho = gene.heaviside_projection(filt(X), beta=beta)
    sig = batched.operator_signature(
        fe.TracedParams(volume_vars=(gene.simp(rho, E0, PENAL, EPS),), surface_vars=[loads]))
    return gene.similarity_order(sig)

if k % REORDER_EVERY == 1:
    order = similarity_order(X, beta)
c, dc, v, its = evaluate(X, beta, order)        # solves in `order`, returns in the original order
```

The evaluation permutes the batch (`E[order]`, `TY[order]`, ...) and un-permutes the results, so the optimizer never sees the ordering. `gene.group_spread` gives the within-group spread of any feature as a one-number diagnostic.

**Halve the group size when the count is still high.** Once the designs have separated into different layouts, even similar neighbours are far from their group mean and the count plateaus in the hundreds. Then more, closer factorizations are cheaper than the CG iterations: a factorization of this cantilever costs a fraction of a second, a few hundred CG iterations for the batch several seconds. The loop halves `chunk` whenever the mean count exceeds `CG_THRESHOLD`, down to one design per group, which is one exact factorization per design (CG converges in one iteration):

```python
if float(its.mean()) > CG_THRESHOLD and chunk > 1:
    chunk = max(1, chunk // 2)
    batched = make_batched(chunk)                # a new batched solver (recompiles once)
    evaluate = make_evaluate(batched)
```

```
iter  beta  c/c0 mean  c/c0 max          vol  CG iters mean/max  chunk    time  order (groups of CHUNK)
   1     1     0.6386    0.6505  [0.400, 0.400]      13.2 /   15      4     5.4s  [2, 1, 3, 4, 0, 5, 7, 6]
  15     2     0.2631    0.3280  [0.400, 0.400]     137.5 /  164      4    29.1s  [6, 5, 7, 4, 0, 3, 1, 2]
  20     2     0.2415    0.2856  [0.400, 0.400]     102.5 /  121      2    48.4s  [6, 5, 7, 4, 0, 3, 1, 2]
  25     4     0.1650    0.1930  [0.400, 0.400]       1.0 /    1      1    68.1s  [6, 5, 7, 4, 0, 3, 1, 2]
  40     8     0.1386    0.1566  [0.400, 0.400]       1.0 /    1      1   104.9s  [2, 3, 1, 4, 0, 5, 7, 6]

40 iterations of 8 designs in 104.9 s (0.328 s per design per iteration)
```

The designs and compliances are identical to the fixed-grouping run (the solves are exact to tolerance either way); the time fell from 129 s to 105 s, all of it after the group size dropped to one. On this problem the similarity ordering alone did not change the counts (the load angles already order the designs along one path, and the mirror-symmetric pairs it finds, such as 0° with 90°, were already in each other's groups); it matters when the batch has no such natural order. The adaptive group size is what pays off here, and the threshold is the knob: at larger DOF the factorization is relatively dearer and a higher threshold is right.

## GCMMA over the batch

The GCMMA version lives in its own script, `batched_topology_optimization_shared_pcg_gcmma.py`, with the same problem, loads, batched solver and `gene` responses.

`jax.vmap` of `GCMMA` would evaluate every design on its own — B separate solves per evaluation, exactly what the shared factorization avoids. `BatchedGCMMA` turns this around: the value-and-gradient function receives the whole batch `X` `(B, n)` and returns `(B,)`, `(B, n)`, `(B, m)`, `(B, m, n)` from **one** batched evaluation, while the asymptotes, the convex subproblems and the conservativity (inner) loop are per design, vectorized. A design whose approximation is already conservative keeps its candidate while the others re-solve, so the loop runs until every design is conservative or `max_inner` is reached.

```python
from feax.experimental.mma import BatchedGCMMA

def value_and_grad(X, beta):
    c, dc, v, info = evaluate(X, beta)                 # one batched solve; dc[i] depends on X[i] only
    g = v / VOLFRAC - 1.0                               # (B,)  volume constraint per design
    dg = jnp.broadcast_to(dv / (VOLFRAC * vol_w.sum()), X.shape)
    return c / c0, dc / c0[:, None], g[:, None], dg[:, None, :], (c, v, info)

opt = BatchedGCMMA(value_and_grad, 0.0, 1.0, n_constraints=1, max_inner=MAX_INNER, move=MOVE, has_aux=True)
state = opt.init(X, beta)
for k in range(1, ITERS + 1):
    state = jax.jit(opt.step)(state, beta_at(k))        # beta rides along as an argument
    c, v, info = state.aux
```

The state is an `MMAState` with a leading batch axis; `solve`, `solve_with_history` and `run` work on it unchanged, and a stopping test fires when every design passes it. Since the B designs are independent, the sum of the B scaled compliances has the block-diagonal gradient the batch needs, one backward pass for all designs.

```
iter  beta  c/c0 mean  c/c0 max          vol  CG iters mean/max    time   inner iters (per design)
   1     1     0.7464    0.7706  [0.365, 0.368]      10.0 /   11    11.1s  [0 0 0 0 0 0 0 0]
  20     2     0.2353    0.2682  [0.400, 0.400]     159.8 /  180   133.3s  [0 0 0 0 0 0 0 0]
  35     8     0.1339    0.1443  [0.400, 0.400]     344.5 /  364   327.4s  [3 2 3 2 3 2 3 2]
  40     8     0.1329    0.1432  [0.400, 0.400]     345.2 /  362   413.8s  [3 2 2 2 3 2 2 2]

40 iterations of 8 designs in 413.8 s (1.293 s per design per iteration)
```

Two things to read off this run, next to the OC one:

- GCMMA reached a slightly lower compliance in the same 40 iterations (mean `c/c0` 0.133 vs 0.139) and meets the volume constraint exactly, through the constraint rather than by construction.
- **Every inner iteration is one more batched evaluation for all designs.** While the approximations are conservative (`beta` ≤ 4) no inner iterations are needed and an outer step costs one evaluation; once the projection sharpens (`beta = 8`) two to three inner iterations per design appear and the step costs three to four. That, not the subproblem solve, is why the run took 2.8× the OC time. With hundreds of designs the slowest one sets the count, so keep `MAX_INNER` small and watch `state.inner_iterations`.

## What the run shows

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

- Early on, all designs are close to the uniform start and the mean matrix is an excellent preconditioner: a dozen CG iterations per solve.
- As the Heaviside continuation sharpens the layouts (`beta` 2 → 8), the designs separate and the counts grow to a few hundred, then plateau. This is the cost of one factorization standing in for eight different matrices; it is still one factorization in memory instead of eight.
- The mirror-symmetric load pairs (0° and 90°, 22.5° and 67.5°, ...) converge to the same compliance, a useful check.
- To keep the counts down, group similar designs: `chunk` is the knob, here one factorization per four neighbouring load angles. The [batched benchmarks](https://github.com/Naruki-Ichihara/feax/tree/main/examples/benchmark) measure the same effect at 500k DOF and 100 designs.

The final designs are written to `output_batched_topopt_shared_pcg/designs.vtu`, one point field per design.
