---
sidebar_label: cohesive
title: feax.mechanics.cohesive
---

Cohesive zone interface for matrix-free fracture simulations.

This module separates two concerns:

1. **Cohesive potential** (material model):
   Pure functions ``potential(delta, delta_max, **params) -&gt; phi_per_node``
   that define the traction-separation law.

2. **CohesiveInterface** (geometry):
   Holds node pairs, integration weights, and interface normals.
   Composes with a potential to produce an energy function
   ``(u_flat, delta_max) -&gt; scalar`` for use with ``newton_solve``.

Supports mixed-mode fracture by decomposing the displacement jump
into normal and tangential components:

```python
δ_n = jump · n          (normal opening, scalar)
δ_t = |jump - δ_n · n|  (tangential sliding, scalar)
δ   = sqrt(⟨δ_n⟩₊² + β² δ_t²)  (effective opening)
```

where β is the mode-mixity ratio (default 1.0).

Usage:

```python
from feax.mechanics.cohesive import CohesiveInterface, exponential_potential

# Axis-aligned interface (simple)
interface = CohesiveInterface.from_axis(
    top_nodes, bottom_nodes, weights, normal_axis=1, vec=2)

# Arbitrary interface with per-node normals
interface = CohesiveInterface(
    top_nodes, bottom_nodes, weights, normals=normals, vec=2)

cohesive_energy = interface.create_energy_fn(
    exponential_potential, Gamma=15.0, sigma_c=20000.0,
)

def total_energy(u, delta_max):
    return elastic_energy(u) + cohesive_energy(u, delta_max)
```

#### exponential\_potential

```python
def exponential_potential(delta, delta_max, *, Gamma, sigma_c)
```

Xu-Needleman exponential cohesive potential with irreversibility.

Loading:

```python
φ(δ) = Γ [1 - (1 + δ/δc) exp(-δ/δc)]
```

where ``δc = Γ / (e · σc)``.

Unloading follows a secant path back to the origin:

```python
φ_unload(δ) = φ(δ_max) / δ_max² · δ²
```

Parameters
----------
- **delta** (*jax.Array*): Current effective opening, shape ``(n_nodes,)``.
- **delta_max** (*jax.Array*): Historical maximum opening, shape ``(n_nodes,)``.
- **Gamma** (*float*): Fracture energy [J/m²].
- **sigma_c** (*float*): Critical cohesive traction [Pa].


Returns
-------
- **phi** (*jax.Array*): Cohesive energy density per node, shape ``(n_nodes,)``.


#### bilinear\_potential

```python
def bilinear_potential(delta, delta_max, *, Gamma, sigma_c)
```

Bilinear cohesive potential with irreversibility.

Loading:

```python
φ(δ) = 0.5 k₀ δ²                                  for δ < δ₀
φ(δ) = Γ - 0.5 σc (δ_f - δ)² / (δ_f - δ₀)        for δ₀ ≤ δ < δ_f
φ(δ) = Γ                                            for δ ≥ δ_f
```

where ``δ_f = 2Γ / σc``, ``δ₀ = σc / k₀``, ``k₀ = σc² / (2Γ)``.

Unloading follows a secant path back to the origin.

Parameters
----------
- **delta** (*jax.Array*): Current effective opening, shape ``(n_nodes,)``.
- **delta_max** (*jax.Array*): Historical maximum opening, shape ``(n_nodes,)``.
- **Gamma** (*float*): Fracture energy [J/m²].
- **sigma_c** (*float*): Critical cohesive traction [Pa].


Returns
-------
- **phi** (*jax.Array*): Cohesive energy density per node, shape ``(n_nodes,)``.


#### compute\_trapezoidal\_weights

```python
def compute_trapezoidal_weights(coords_1d)
```

Compute trapezoidal integration weights from sorted 1D coordinates.

For 2D cohesive interfaces where nodes are arranged along a line.

Parameters
----------
- **coords_1d** (*array_like*): Sorted 1D coordinates of interface nodes.


Returns
-------
- **weights** (*jax.Array*): Integration weights, shape ``(n,)``.


#### compute\_lumped\_area\_weights

```python
def compute_lumped_area_weights(node_ids, coords, quads)
```

Compute lumped area weights from quad elements on a planar interface.

For 3D cohesive interfaces where quad elements define the surface.
Each quad contributes 1/4 of its area to each of its 4 nodes.

Parameters
----------
- **node_ids** (*array_like*): Node indices for which to compute weights, shape ``(n,)``.
- **coords** (*array_like*): Full coordinate array, shape ``(n_total, 3)``.
- **quads** (*array_like*): Quad element connectivity (indices into ``coords``), shape ``(n_quads, 4)``.


Returns
-------
- **weights** (*jax.Array*): Area weights per node, shape ``(n,)``.


## CohesiveInterface Objects

```python
class CohesiveInterface()
```

Cohesive zone interface with mixed-mode support.

Decomposes the displacement jump into normal and tangential components
and computes an effective opening for the cohesive potential:

```python
δ_n = jump · n                     (normal)
δ_t = |jump - δ_n · n|             (tangential)
δ   = sqrt(⟨δ_n⟩₊² + β² δ_t²)    (effective)
```

where ``⟨·⟩₊ = max(·, 0)`` is the Macaulay bracket (no energy in
compression) and β is the mode-mixity ratio.

Parameters
----------
- **top_nodes** (*array_like*): Node indices on the ``+`` side of the interface, shape ``(n,)``.
- **bottom_nodes** (*array_like*): Node indices on the ``-`` side of the interface, shape ``(n,)``.
- **weights** (*array_like*): Integration weights per node pair, shape ``(n,)``.
- **normals** (*array_like*): Unit normal vectors pointing from bottom to top, shape ``(n, vec)``.
- **vec** (*int*): Number of displacement components per node (2 or 3).
- **beta** (*float, optional*): Mode-mixity ratio for tangential contribution (default 1.0). β=0 gives pure Mode I; β=1 gives equal weight to Mode I and II.


Examples
--------
```python
>>> # Axis-aligned interface (convenience)
>>> interface = CohesiveInterface.from_axis(
...     top_nodes, bottom_nodes, weights, normal_axis=1, vec=2)
>>>
>>> # Arbitrary normals
>>> normals = compute_normals(...)  # (n, 2)
>>> interface = CohesiveInterface(
...     top_nodes, bottom_nodes, weights, normals, vec=2, beta=0.5)
```

#### from\_axis

```python
@classmethod
def from_axis(cls,
              top_nodes,
              bottom_nodes,
              weights,
              normal_axis,
              vec,
              beta=1.0)
```

Create interface with axis-aligned normal.

Convenience constructor for interfaces aligned with a coordinate
axis (e.g., y=0 plane with normal in y-direction).

Parameters
----------
- **normal_axis** (*int*): Coordinate axis index for the normal direction (0=x, 1=y, 2=z).


#### get\_jump

```python
def get_jump(u_flat)
```

Compute displacement jump at interface nodes.

Parameters
----------
- **u_flat** (*jax.Array*): Flat displacement vector.


Returns
-------
- **jump** (*jax.Array*): Displacement jump (top - bottom), shape ``(n_nodes, vec)``.


#### get\_opening

```python
def get_opening(u_flat)
```

Compute effective opening at interface nodes.

Returns the mixed-mode effective opening:

```python
δ = sqrt(⟨δ_n⟩₊² + β² δ_t²)
```

Parameters
----------
- **u_flat** (*jax.Array*): Flat displacement vector.


Returns
-------
- **delta** (*jax.Array*): Effective opening, shape ``(n_nodes,)``.


#### get\_opening\_components

```python
def get_opening_components(u_flat)
```

Compute normal and tangential opening components.

Parameters
----------
- **u_flat** (*jax.Array*): Flat displacement vector.


Returns
-------
- **delta_n** (*jax.Array*): Normal opening (signed), shape ``(n_nodes,)``.
- **delta_t** (*jax.Array*): Tangential opening (unsigned), shape ``(n_nodes,)``.


#### create\_energy\_fn

```python
def create_energy_fn(potential, **params)
```

Create a cohesive energy function.

Parameters
----------
- **potential** (*callable*): Cohesive potential function with signature ``potential(delta, delta_max, **params) -&gt; phi_per_node``.


Returns
-------
- **energy_fn** (*callable*): Function ``energy_fn(u_flat, delta_max) -&gt; scalar`` suitable for use with ``newton_solve``.

