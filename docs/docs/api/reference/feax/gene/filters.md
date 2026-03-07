---
sidebar_label: filters
title: feax.gene.filters
---

Filters for topology optimization using FEAX framework.

Provides both PDE-based (Helmholtz) and distance-based (density) filters
for design variable smoothing in generative design workflows.

## HelmholtzFilterProblem Objects

```python
class HelmholtzFilterProblem(fe.problem.Problem)
```

Helmholtz filter problem: ρ̃ - r² ∇²ρ̃ = ρ_source

This is a helper Problem class for internal use by helmholtz_filter().

#### create\_helmholtz\_filter

```python
def create_helmholtz_filter(mesh, radius, P=None, solver_options=None)
```

Create a differentiable Helmholtz filter function (node-based).

This factory function creates the filter problem and solver once, returning
a pure function that can be used with jax.jit, jax.vmap, and jax.grad.

Solves: ρ̃ - r² ∇²ρ̃ = ρ_source

**Arguments**:

- `mesh` - Mesh object
- `radius` - Filter radius (controls smoothness - larger = smoother)
- `P` - Optional prolongation matrix for periodic boundary conditions (default None)
- `solver_options` - Optional IterativeSolverOptions (default: cg, tol=1e-8)


**Returns**:

- `filter_fn` - A pure function (rho_source) -&gt; rho_filtered that can be
  used with JAX transformations (jit, vmap, grad)
- `Input` - (num_nodes,) node-based density field
- `Output` - (num_nodes,) filtered node-based density field


**Example**:

```python
>>> # Create filter function once
>>> filter_fn = create_helmholtz_filter(mesh, radius=0.1)
>>>
>>> # Use in differentiable objective
>>> def objective(rho):
...     rho_filtered = filter_fn(rho)
...     # ... use rho_filtered in FE solve
...     return compliance
>>>
>>> # Compute gradients
>>> grad_fn = jax.grad(objective)
>>> gradient = grad_fn(rho)
>>>
>>> # With periodic BCs
>>> P = flat.pbc.prolongation_matrix(pairings, mesh, vec=1)
>>> filter_fn = create_helmholtz_filter(mesh, radius=0.1, P=P)
```

#### helmholtz\_filter

```python
def helmholtz_filter(rho_source, mesh, radius, P=None, solver_options=None)
```

Apply Helmholtz filter to node-based density field.

WARNING: This function creates problem/solver each call. For use inside jax.grad,
use create_helmholtz_filter() instead to create the filter function once.

Solves: ρ̃ - r² ∇²ρ̃ = ρ_source

**Arguments**:

- `rho_source` - (num_nodes,) array of node-based source density field
- `mesh` - Mesh object
- `radius` - Filter radius (controls smoothness - larger = smoother)
- `P` - Optional prolongation matrix for periodic boundary conditions (default None)
- `solver_options` - Optional IterativeSolverOptions (default: cg, tol=1e-8)


**Returns**:

  (num_nodes,) array of filtered node-based density field


**Example**:

```python
>>> # For one-time use (NOT inside jax.grad)
>>> rho_filtered = helmholtz_filter(rho_source, mesh, radius=0.1)
```
```python
>>> # For use with jax.grad, use create_helmholtz_filter instead:
>>> filter_fn = create_helmholtz_filter(mesh, radius=0.1)
>>> def objective(rho):
...     return np.sum(filter_fn(rho))
>>> grad_fn = jax.grad(objective)
```

#### create\_density\_filter

```python
def create_density_filter(mesh,
                          radius: float,
                          weight_type: str = "cone") -> Callable
```

Create a standard density filter using distance-based weighted averaging.

This is the classic topology optimization filter that computes weighted
averages of design variables based on spatial proximity. It&#x27;s computationally
lighter than Helmholtz filter but provides similar smoothing effects.

The filter computes: ρ̃_i = Σ_j w_ij ρ_j / Σ_j w_ij
where w_ij is a distance-based weight function.

**Arguments**:

- `mesh` - Mesh object
- `radius` - Filter radius (controls neighborhood size)
- `weight_type` - Type of weight function:
  - &quot;cone&quot;: w = max(0, r - d) (default, most common)
  - &quot;gaussian&quot;: w = exp(-(d/r)^2)
  - &quot;constant&quot;: w = 1 if d &lt;= r, else 0


**Returns**:

- `filter_fn` - A JIT-compiled function (rho) -&gt; rho_filtered
- `Input` - (num_nodes,) node-based density field
- `Output` - (num_nodes,) filtered node-based density field


**Example**:

```python
>>> # Create filter function once
>>> filter_fn = create_density_filter(mesh, radius=3.0)
>>>
>>> # Use in optimization
>>> def objective(rho):
...     rho_filtered = filter_fn(rho)
...     # ... use rho_filtered in FE solve
...     return compliance
>>>
>>> # Automatic differentiation works seamlessly
>>> grad_fn = jax.grad(objective)
>>> gradient = grad_fn(rho)
```
**Notes**:
- This filter is linear and does not require solving a PDE
- The cone weight function (default) is most common in literature
- For large meshes, pre-computation of weights may require significant memory
- The filter preserves the integral of the density field (mass conservation)

#### density\_filter

```python
def density_filter(rho_source,
                   mesh,
                   radius: float,
                   weight_type: str = "cone") -> np.ndarray
```

Apply standard density filter to node-based density field.

WARNING: This function creates the filter each call. For use inside jax.grad
or in optimization loops, use create_density_filter() instead to create the
filter function once.

**Arguments**:

- `rho_source` - (num_nodes,) array of node-based source density field
- `mesh` - Mesh object
- `radius` - Filter radius
- `weight_type` - Weight function type (&quot;cone&quot;, &quot;gaussian&quot;, &quot;constant&quot;)


**Returns**:

  (num_nodes,) array of filtered node-based density field


**Example**:

```python
>>> # For one-time use (NOT inside jax.grad or loops)
>>> rho_filtered = density_filter(rho_source, mesh, radius=3.0)
>>>
>>> # For repeated use or with jax.grad, use create_density_filter instead:
>>> filter_fn = create_density_filter(mesh, radius=3.0)
>>> def objective(rho):
...     return np.sum(filter_fn(rho))
>>> grad_fn = jax.grad(objective)
```

#### heaviside\_projection

```python
@jax.jit
def heaviside_projection(rho, beta=10.0, threshold=0.5)
```

Apply Heaviside projection to density field for sharp void/solid boundaries.

H(ρ) = (tanh(β*(ρ-threshold)) + 1) / 2

This function is pure and JIT-compiled for efficient batched processing.

**Arguments**:

- `rho` - Density field (normalized to [0, 1])
- `beta` - Sharpness parameter (higher = sharper transition, default 10.0)
- `threshold` - Transition threshold (default 0.5)


**Returns**:

  Projected density field with sharp boundaries


**Example**:

```python
>>> # Single field
>>> rho_sharp = heaviside_projection(rho_smooth, beta=10.0, threshold=0.5)
```
```python
>>> # Vectorized batch processing
>>> rho_batch_sharp = jax.vmap(lambda r: heaviside_projection(r, beta=10.0))(rho_batch)
```

#### compute\_volume\_fraction\_threshold

```python
def compute_volume_fraction_threshold(rho, target_volume_fraction)
```

Compute density threshold to achieve target volume fraction.

Uses percentile-based approach to ensure exact volume fraction after Heaviside projection.

**Arguments**:

- `rho` - Density field
- `target_volume_fraction` - Target solid volume fraction (0.0 to 1.0)


**Returns**:

  Threshold value for Heaviside projection


**Example**:

```python
>>> # Normalize density to [0, 1]
>>> rho_normalized = (rho - rho.min()) / (rho.max() - rho.min())
>>>
>>> # Compute threshold for 50% volume fraction
>>> threshold = compute_volume_fraction_threshold(rho_normalized, 0.5)
>>>
>>> # Apply Heaviside projection
>>> rho_projected = heaviside_projection(rho_normalized, beta=10.0, threshold=threshold)
```

