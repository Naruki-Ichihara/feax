---
sidebar_label: filters
title: feax.flat.filters
---

Density filtering utilities for periodic structures.

This module provides filtering functions commonly used in topology optimization
and microstructure design, with support for periodic boundary conditions.

## HelmholtzFilterProblem Objects

```python
class HelmholtzFilterProblem(fe.problem.Problem)
```

Helmholtz filter problem: ρ̃ - r² ∇²ρ̃ = ρ_source

This is a helper Problem class for internal use by helmholtz_filter().

#### helmholtz\_filter

```python
def helmholtz_filter(rho_source, mesh, radius, P=None, solver_options=None)
```

Apply Helmholtz filter to density field with optional periodic boundary conditions.

Solves: ρ̃ - r² ∇²ρ̃ = ρ_source

This function is pure (no side effects) and can be used with jax.jit and jax.vmap
when P and solver_options are provided as static arguments.

**Arguments**:

- `rho_source` - (num_cells,) array of source density field
- `mesh` - Mesh object
- `radius` - Filter radius (controls smoothness - larger = smoother)
- `P` - Optional prolongation matrix for periodic boundary conditions (default None)
- `solver_options` - Optional SolverOptions (default: tol=1e-8, cg solver)


**Returns**:

  (num_nodes,) array of filtered density field


**Example**:

```python
>>> # Without periodic BCs
>>> rho_filtered = helmholtz_filter(rho_source, mesh, radius=0.1)
```
```python
>>> # With periodic BCs
>>> P = flat.pbc.prolongation_matrix(pairings, mesh, vec=1)
>>> rho_filtered = helmholtz_filter(rho_source, mesh, radius=0.1, P=P)
```
```python
>>> # Vectorized filtering with vmap
>>> filter_fn = lambda rho: helmholtz_filter(rho, mesh, radius=0.1, P=P)
>>> rho_batch_filtered = jax.vmap(filter_fn)(rho_batch)
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

