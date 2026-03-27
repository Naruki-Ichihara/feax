# Batched Topology Optimization with Surface Loads

This tutorial demonstrates how to optimize **multiple load cases in parallel** on a single mesh using `jax.vmap`. Each case has its own SIREN neural density field and a different surface traction, while the FE solve, compliance evaluation, and gradient-based update are all batched.

## Overview

In multi-load topology optimization, the same structure must be optimized for several loading scenarios simultaneously. A naive approach would loop over load cases sequentially, but JAX's `vmap` lets us vectorize the entire forward solve and backward pass across all cases in a single JIT-compiled call.

This example solves 10 cantilever beam problems on a $60 \times 30$ QUAD4 mesh. Each problem applies a downward traction at a different patch along the right edge:

```
 ┌──────────────────────────────────┐ ── y = LY
 │                                  ├── patch 9
 │                                  ├── patch 8
 │                                  ├── ...
 │  fixed (left)                    ├── patch 1
 │                                  ├── patch 0
 └──────────────────────────────────┘ ── y = 0
x = 0                             x = LX
```

**Key ingredients:**

1. **SIREN** (Sinusoidal Representation Network) maps cell centroids to a density field per case
2. **SIMP** material interpolation penalizes intermediate densities
3. **Augmented Lagrangian** enforces a volume fraction constraint
4. **`jax.vmap`** batches the FE solve, compliance, and optimizer across all cases

## Problem Setup

### Material and Geometry

```python
LX, LY = 60.0, 30.0
SCALE = 5
NX, NY = int(LX * SCALE), int(LY * SCALE)

E0 = 70e3           # Young's modulus (solid)
E_EPS = 7.0         # Ersatz stiffness (void)
NU = 0.3            # Poisson's ratio
SIMP_PENALTY = 3.0  # SIMP penalization exponent
TRACTION_MAG = 1e2  # Applied traction magnitude
```

### Load Patch Locations

Ten patches are distributed evenly along the right edge. Each patch is defined by a `location_fn` that tests whether a point lies within the patch band:

```python
NUM_CASES = 10
PATCH_HALF_BAND = 0.5 * LY / NUM_CASES + Y_TOL
PATCH_CENTERS = [LY * (i + 0.5) / NUM_CASES for i in range(NUM_CASES)]

def _make_right_patch(center):
    def loc_fn(pt):
        return jnp.isclose(pt[0], LX, atol=X_TOL) & (
            jnp.abs(pt[1] - center) <= PATCH_HALF_BAND
        )
    return loc_fn

LOAD_LOCATIONS = tuple(_make_right_patch(c) for c in PATCH_CENTERS)
```

!!! warning "location_fn must be single-argument"
    FEAX inspects `co_argcount` to distinguish 1-argument (`point`) from 2-argument (`point, node_index`) location functions. Using `lambda pt, _c=center: ...` yields `co_argcount=2`, causing FEAX to misinterpret the default parameter as a node index. Always use a closure instead.

### Linear Elasticity with SIMP

The problem class receives the density $\rho$ as a cell-based internal variable (volume variable). The surface map applies a downward traction whose magnitude is controlled by the surface variable `load`:

```python
class LinearElasticityProblem(fe.Problem):
    def custom_init(self, E0, E_eps, nu, p):
        self.E0, self.E_eps, self.nu, self.p = E0, E_eps, nu, p

    def get_tensor_map(self):
        def stress(u_grad, rho):
            E = (self.E0 - self.E_eps) * rho**self.p + self.E_eps
            mu = E / (2.0 * (1.0 + self.nu))
            lam = E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
            strain = 0.5 * (u_grad + u_grad.T)
            return lam * jnp.trace(strain) * jnp.eye(self.dim) + 2.0 * mu * strain
        return stress

    def get_surface_maps(self):
        def surface_map(u, x, load):
            traction = jnp.zeros(self.dim)
            return traction.at[-1].set(load)
        return [surface_map for _ in range(max(1, len(self.location_fns)))]
```

All 10 surface maps share the same physics — the per-case loading is controlled entirely by which surface variables are nonzero.

## Density Parameterization with SIREN

Each load case has its own SIREN network that maps normalized cell centroids $(x, y) \in [-1, 1]^2$ to a scalar logit. The density is obtained via sigmoid:

$$
\rho_i = \sigma\bigl(\text{SIREN}_i(\mathbf{x})\bigr)
$$

The SIREN architecture uses sinusoidal activations with a frequency parameter $\omega$:

```python
class SIREN(eqx.Module):
    weights: tuple[jax.Array, ...]
    biases: tuple[jax.Array, ...]
    omega: float = eqx.field(static=True)

    def __call__(self, x):
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = jnp.sin(self.omega * (x @ weight + bias))
        return x @ self.weights[-1] + self.biases[-1]
```

The 10 models are created independently and then stacked into a single batched PyTree using `jax.tree_util.tree_map`:

```python
def create_model_batch():
    keys = jax.random.split(jax.random.PRNGKey(MODEL_RNG_SEED), NUM_CASES)
    models = [SIREN(..., key=key) for key in keys]
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *models)
```

After stacking, each leaf array has an extra leading batch dimension of size `NUM_CASES`, which aligns with `jax.vmap`.

## Batched Surface Variables

The central data-engineering step is assembling `batched_surface_vars` — a PyTree where each leaf has shape `(NUM_CASES, ...)`. For case $k$, only the $k$-th surface patch carries the traction; all other patches are zeroed out:

```python
tractions = [
    fe.InternalVars.create_uniform_surface_var(problem, TRACTION_MAG, surface_index=i)
    for i in range(NUM_CASES)
]

per_case_surface_vars = []
for case_idx in range(NUM_CASES):
    sv = tuple(
        (tractions[i],) if i == case_idx else (jnp.zeros_like(tractions[i]),)
        for i in range(NUM_CASES)
    )
    per_case_surface_vars.append(sv)

batched_surface_vars = jax.tree_util.tree_map(
    lambda *xs: jnp.stack(xs, axis=0),
    *per_case_surface_vars,
)
```

This produces a structure where `batched_surface_vars[case_idx]` activates only the traction at patch `case_idx`.

## cuDSS Pre-warming

When using the `cudss` direct solver, the solver must be initialized with a concrete sparse matrix **before** any `jax.vmap` tracing occurs. Pass a sample `internal_vars` to `create_solver`:

```python
sample_rho = jnp.full(problem.num_cells, TARGET_VOLUME_FRACTION)
sample_surface_vars = tuple((t,) for t in tractions)
sample_internal_vars = fe.InternalVars(
    volume_vars=(sample_rho,),
    surface_vars=sample_surface_vars,
)

solver = fe.create_solver(
    problem, bc=bc,
    solver_options=fe.DirectSolverOptions(solver="cudss"),
    adjoint_solver_options=fe.DirectSolverOptions(solver="cudss"),
    iter_num=1,
    internal_vars=sample_internal_vars,
)
```

!!! warning "ConcretizationTypeError without pre-warming"
    If `internal_vars` is omitted, cuDSS initialization is deferred to the first solve call. When that first call happens inside `jax.vmap`, JAX tracer values are passed where concrete values are required, resulting in a `ConcretizationTypeError`. Always provide `internal_vars` when combining cuDSS with `vmap`.

## Batched Loss and Optimization

### Forward Solve

Each case calls the same solver with different `(rho, surface_vars)`:

```python
def solve_forward(rho, surface_vars):
    internal_vars = fe.InternalVars(volume_vars=(rho,), surface_vars=surface_vars)
    solution = solver(internal_vars, initial_guess)
    return compliance_fn(solution, surface_vars)
```

### Loss Function

The batched loss uses `jax.vmap` over `solve_forward` and applies an augmented Lagrangian volume constraint:

$$
\mathcal{L} = \sum_{k=1}^{N} \left[ c_k + \lambda_k \max(v_k - v^*, 0) + \frac{p}{2} \max(v_k - v^*, 0)^2 \right]
$$

where $c_k$ is the compliance, $v_k$ is the volume fraction, $v^*$ is the target, $\lambda_k$ is the Lagrange multiplier, and $p$ is the penalty.

```python
@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def batched_loss(models, coords, target_volume_fractions, lams, penalties, batched_surface_vars):
    densities = predict_densities(models, coords)
    compliances = jax.vmap(solve_forward)(densities, batched_surface_vars)
    volume_fractions = jax.vmap(volume_fraction_fn)(densities)

    volume_errors = volume_fractions - target_volume_fractions
    violations = jnp.maximum(volume_errors, 0.0)
    losses = compliances + lams * violations + 0.5 * penalties * violations**2
    return jnp.sum(losses), {...}
```

`eqx.filter_value_and_grad` differentiates through all array leaves of `models` (the SIREN weights and biases), while `eqx.filter_jit` JIT-compiles the entire computation.

### Training Loop

The optimizer (AdaBelief via Optax) is also vmapped so each case has independent optimizer state:

```python
models = create_model_batch()
optimizer = optax.chain(
    optax.zero_nans(),
    optax.clip_by_global_norm(GRAD_CLIP_NORM),
    optax.adabelief(LEARNING_RATE),
)
opt_states = jax.vmap(lambda m: optimizer.init(eqx.filter(m, eqx.is_array)))(models)

for iteration in range(NUM_ITERATIONS):
    (total_loss, aux), grads = batched_loss(models, coords, ...)
    updates, opt_states = jax.vmap(optimizer.update)(grads, opt_states, value=aux["losses"])
    models = jax.vmap(eqx.apply_updates)(models, updates)
    lams = lams + penalties * aux["violations"]
```

The Lagrange multipliers $\lambda_k$ are updated after each iteration using the standard augmented Lagrangian update rule.

## Output

Optimized densities are saved as VTU files for visualization in ParaView:

```python
final_densities = predict_densities(models, coords)
for case_index, case_name in enumerate(CASE_NAMES):
    fe.utils.save_sol(
        mesh,
        f"output/density_{case_index + 1}_{case_name}.vtu",
        cell_infos=[("density", final_densities[case_index])],
    )
```

## Summary

| Concept | How it is used |
|---|---|
| `location_fns` | Define multiple surface load regions on one `Problem` |
| `InternalVars.surface_vars` | Per-surface-patch traction arrays, zeroed selectively per case |
| `jax.tree_util.tree_map` + `jnp.stack` | Stack per-case PyTrees into batched arrays |
| `jax.vmap(solve_forward)` | Vectorize the FE solve + compliance over all load cases |
| `eqx.filter_jit` / `eqx.filter_value_and_grad` | JIT-compile and differentiate through Equinox models |
| `jax.vmap(optimizer.update)` | Independent optimizer state per case |
| cuDSS pre-warming | Pass `internal_vars` to `create_solver` to avoid tracer errors under `vmap` |

## Complete Code

See `examples/advance/topology_optimization_batched_surface_loads.py`.
