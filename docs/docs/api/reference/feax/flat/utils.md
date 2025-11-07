---
sidebar_label: utils
title: feax.flat.utils
---

Utility functions for lattice toolkit visualization and analysis.

#### compute\_directional\_youngs\_modulus

```python
def compute_directional_youngs_modulus(C, n)
```

Compute Young&#x27;s modulus in direction n from stiffness matrix C.

For a given direction n, E(n) = 1/S(n) where S(n) = n^T @ S @ n
and S is the compliance tensor in full 3x3x3x3 form.

**Arguments**:

- `C` - 6x6 stiffness matrix in Voigt notation
- `n` - Direction vector (3,)
  

**Returns**:

- `float` - Young&#x27;s modulus in direction n

#### visualize\_stiffness\_sphere

```python
def visualize_stiffness_sphere(C, output_file, n_theta=30, n_phi=60)
```

Create 3D sphere visualization of directional Young&#x27;s modulus.

Creates a VTK file showing how Young&#x27;s modulus varies with direction.
The surface is shaped by E(n) values - a perfect sphere indicates isotropy.

**Arguments**:

- `C` - 6x6 stiffness matrix in Voigt notation
- `output_file` - Path to output VTK file (e.g., &#x27;stiffness_sphere.vtu&#x27;)
- `n_theta` - Number of theta divisions (default 30)
- `n_phi` - Number of phi divisions (default 60)
  

**Returns**:

- `dict` - Statistics including E_max, E_min, anisotropy_ratio

