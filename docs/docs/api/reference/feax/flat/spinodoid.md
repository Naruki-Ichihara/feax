---
sidebar_label: spinodoid
title: feax.flat.spinodoid
---

Spinodoid structure generation using Gaussian Random Fields.

This module provides tools for generating anisotropic Gaussian Random Fields (GRF)
for periodic spinodoid microstructures. For filtering and projection operations,
see the `filters` module.

#### evaluate\_grf\_field

```python
@jax.jit
def evaluate_grf_field(cell_centers, n_vectors, gamma, beta)
```

Evaluate Gaussian Random Field at given points (JIT-compiled).

φ(x) = sqrt(2/N) * Σ cos(β*n_i·x + γ_i)

**Arguments**:

- `cell_centers` - (num_cells, 3) array of evaluation points
- `n_vectors` - (N, 3) array of random unit vectors
- `gamma` - (N,) array of random phases
- `beta` - scalar wave number (controls characteristic length scale)


**Returns**:

  (num_cells,) array of zero-mean GRF values

#### generate\_direction\_vectors

```python
def generate_direction_vectors(theta1, theta2, theta3, N, key)
```

Generate random unit vectors in constrained cone regions.

Vectors are distributed among cones around x, y, z axes based on
the specified angle constraints.

**Arguments**:

- `theta1` - Cone angle for x-axis (radians, 0 = no constraint)
- `theta2` - Cone angle for y-axis (radians, 0 = no constraint)
- `theta3` - Cone angle for z-axis (radians, 0 = no constraint)
- `N` - Total number of vectors to generate
- `key` - JAX random key


**Returns**:

  (N, 3) array of unit vectors

- `Note` - At least one theta must be &gt; 0.

#### generate\_grf\_source

```python
def generate_grf_source(mesh,
                        beta=10.0,
                        N=100,
                        theta1=None,
                        theta2=None,
                        theta3=None,
                        seed=0)
```

Generate anisotropic Gaussian Random Field (GRF) source for spinodoid structures.

φ(x) = sqrt(2/N) * Σ cos(β*n_i·x + γ_i)
where n_i ~ U(constrained region on S²)

**Arguments**:

- `mesh` - Mesh object with points and cells
- `beta` - Wave number (controls characteristic length scale, default 10.0)
- `N` - Number of random waves (default 100)
- `theta1` - Cone angle constraint for x-direction (radians, None = no constraint)
- `theta2` - Cone angle constraint for y-direction (radians, None = no constraint)
- `theta3` - Cone angle constraint for z-direction (radians, None = no constraint)
- `seed` - Random seed (default 0)


**Returns**:

  (num_cells,) array of GRF values (zero mean)

- `Note` - At least one theta must be specified (not None).


**Example**:

```python
>>> mesh = box_mesh(size=1.0, mesh_size=0.1)
>>> source = generate_grf_source(mesh, beta=10.0, N=100,
...                              theta1=np.pi/4, theta2=np.pi/4, theta3=np.pi/4)
```

