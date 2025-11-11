---
sidebar_label: pbc
title: feax.flat.pbc
---

Periodic boundary condition implementation for finite element analysis.

This module provides functionality for enforcing periodic boundary conditions (PBCs)
in finite element computations, particularly for unit cell-based homogenization and
multiscale analysis. It handles the construction of prolongation matrices that map
between reduced (independent) and full degree-of-freedom spaces.

## PeriodicPairing Objects

```python
@dataclass
class PeriodicPairing()
```

Represents a single periodic boundary condition pair between master and slave regions.

This class encapsulates the geometric and topological relationship between paired
regions in a periodic boundary condition. It defines which nodes are considered
&#x27;master&#x27; (independent) and &#x27;slave&#x27; (dependent), along with the mapping function
that relates their coordinates.

**Attributes**:

- `location_master` _Callable[[onp.ndarray], bool]_ - Function that identifies points
  on the master side of the periodic boundary. Takes a point coordinate array
  and returns True if the point lies on the master boundary.
- `location_slave` _Callable[[onp.ndarray], bool]_ - Function that identifies points
  on the slave side of the periodic boundary. Takes a point coordinate array
  and returns True if the point lies on the slave boundary.
- `mapping` _Callable[[onp.ndarray], onp.ndarray]_ - Function that maps a point from
  the master boundary to its corresponding location on the slave boundary.
  Essential for establishing the periodic relationship.
- `vec` _int_ - Index of the degree-of-freedom component affected by this pairing.
  For vector problems: 0=x-component, 1=y-component, 2=z-component, etc.


**Example**:

```python
>>> # Create a pairing for x-direction periodicity
>>> pairing = PeriodicPairing(
...     location_master=lambda p: np.isclose(p[0], 0.0),  # Left face
...     location_slave=lambda p: np.isclose(p[0], 1.0),   # Right face
...     mapping=lambda p: p + np.array([1.0, 0.0, 0.0]),  # Translate by 1 in x
...     vec=0  # Apply to x-component of displacement
... )
```

#### prolongation\_matrix

```python
def prolongation_matrix(periodic_pairings: Iterable[PeriodicPairing],
                        mesh: Mesh,
                        vec: int,
                        offset: int = 0) -> BCOO
```

Constructs the prolongation matrix P for applying periodic boundary conditions.

The prolongation matrix P maps reduced (independent) degrees of freedom to the full
set of DoFs, enforcing periodic constraints. This is essential for homogenization
and multiscale analysis where opposite boundaries must have compatible displacements.

The matrix enforces: u_slave = u_master for paired nodes, reducing the system size
while maintaining periodicity. Given reduced DoFs ū, the full DoF vector is u = P @ ū.

**Arguments**:

- `periodic_pairings` _Iterable[PeriodicPairing]_ - Collection of periodic constraint
  definitions specifying master/slave boundary pairs, their geometric mapping,
  and affected DoF components.
- `mesh` _Mesh_ - Finite element mesh containing node coordinates (points) and
  element connectivity information.
- `vec` _int_ - Number of degrees of freedom per node (e.g., 2 for 2D elasticity,
  3 for 3D elasticity).
- `offset` _int_ - Global DoF offset for constructing the matrix. Defaults to 0.


**Returns**:

- `BCOO` - Prolongation matrix P with shape (N, M) where:
  - N = total number of DoFs before applying constraints
  - M = number of independent DoFs after applying constraints
  JAX BCOO sparse matrix format for compatibility with FEAX ecosystem.


**Raises**:

- `AssertionError` - If master and slave boundary node counts don&#x27;t match.
- `ValueError` - If the DoF reduction is inconsistent with the pairing structure.


**Example**:

```python
>>> # Create prolongation matrix for 3D periodic unit cell
>>> pairings = periodic_bc_3D(unit_cell, vec=3, dim=3)
>>> P = prolongation_matrix(pairings, mesh, vec=3)
>>> print(f&quot;DoF reduction: {`P.shape[0]`} -> {`P.shape[1]`}&quot;)
```
**Notes**:
The matrix construction process:
1. Identifies master and slave nodes for each pairing
2. Establishes point-to-point correspondence using geometric mapping
3. Builds sparse matrix entries to enforce u_slave = u_master
4. Results in a rectangular matrix for DoF space transformation
5. Returns JAX BCOO sparse matrix for JAX transformations compatibility

#### periodic\_bc\_3D

```python
def periodic_bc_3D(unitcell: UnitCell,
                   vec: int = 1,
                   dim: int = 3) -> List[PeriodicPairing]
```

Generate periodic boundary condition pairings for a 3D unit cell.

Creates a complete set of periodic pairings for all faces, edges, and corners of a
3D unit cell. This ensures full periodicity where opposite boundaries are constrained
to have compatible displacements, essential for RVE-based homogenization.

The function systematically pairs:
- Opposite faces (6 face pairs for 3D)
- Corresponding edges (12 edge pairs for 3D)
- Corresponding corners (7 corner pairs for 3D, excluding origin)

**Arguments**:

- `unitcell` _UnitCell_ - The unit cell object providing boundary identification
  functions and geometric mapping capabilities.
- `vec` _int_ - Number of degrees of freedom per node. Defaults to 1.
- `dim` _int_ - Spatial dimension of the problem. Defaults to 3.


**Returns**:

- `List[PeriodicPairing]` - Complete list of periodic pairings ordered as:
  1. Corner pairings (excluding origin as master)
  2. Edge pairings (excluding corners)
  3. Face pairings (excluding edges and corners)


**Example**:

```python
>>> unit_cell = UnitCell()
>>> ...
>>> pairings = periodic_bc_3D(unit_cell, vec=3)  # 3D elasticity
>>> print(f&quot;Total pairings: {`len(pairings)`}&quot;)
>>> # Should be: 7 corners + 12 edges + 6 faces = 25 per DoF component
```
**Notes**:
- The origin corner serves as the master for all other corners
- Edge and face exclusions prevent double-counting of constraints
- Each geometric pairing is replicated for each DoF component

