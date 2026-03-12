---
sidebar_label: adaptive
title: feax.gene.adaptive
---

Adaptive remeshing for topology optimization using Gmsh size callback.

Regenerates TET4 meshes with element sizes controlled by a refinement
field: high-value regions receive fine elements, low-value regions
receive coarse elements.  The field can be any node-based scalar in
[0, 1] — filtered density, density-gradient magnitude, etc.

Two entry points:

- ``adaptive_mesh`` — general-purpose; accepts any geometry
  (STEP/BREP file or a Gmsh model-builder callable).
- ``adaptive_box_mesh`` — convenience wrapper for axis-aligned boxes
  with optional periodic face meshing (for unit-cell homogenisation).

#### interpolate\_field

```python
def interpolate_field(values_old, mesh, points_new, clip=None)
```

Transfer a node-based scalar field to a new point set via TET4
shape functions (barycentric coordinates).

For each query point, finds the containing tetrahedron and
interpolates using barycentric weights.  Points outside the mesh
fall back to nearest-node value.

Parameters
----------
- **values_old** (*ndarray, shape (num_old_nodes,)*): Field values on the old mesh (density, temperature, etc.).
- **mesh** (*Mesh*): Old TET4 mesh (``cells`` must have 4 columns).
- **points_new** (*ndarray, shape (num_new_nodes, 3)*): Node coordinates of the new mesh.
- **clip** (*(float, float), optional*): If given, clamp output to ``(min, max)``.


Returns
-------
values_new : ndarray, shape (num_new_nodes,)

Raises
------
ValueError
    If the mesh is not TET4 (cells must have exactly 4 nodes).

#### gradient\_refinement

```python
def gradient_refinement(rho, mesh)
```

Compute normalised gradient magnitude as a refinement field.

Uses TET4 shape functions (constant gradient per element), then
averages to nodes and normalises to [0, 1].

Parameters
----------
- **rho** (*ndarray, shape (num_nodes,)*): Node-based scalar field (e.g. filtered density).
- **mesh** (*Mesh*): TET4 mesh (``cells`` must have 4 columns).


Returns
-------
ndarray, shape (num_nodes,)
    Normalised gradient magnitude in [0, 1], suitable for use as
    ``refinement_field`` in :func:`adaptive_mesh`.

Raises
------
ValueError
    If the mesh is not TET4 (cells must have exactly 4 nodes).

#### adaptive\_mesh

```python
def adaptive_mesh(geometry: Union[str, Callable[[], None]],
                  refinement_field: Optional[onp.ndarray] = None,
                  old_mesh: Optional[Mesh] = None,
                  h_min: float = 0.02,
                  h_max: float = 0.15) -> Mesh
```

Generate adaptive TET4 mesh for an arbitrary geometry.

Element sizes are driven by a refinement field (any node-based
scalar in [0, 1]): high values produce fine elements (``h_min``),
low values produce coarse elements (``h_max``).  Typical choices
include filtered density, density-gradient magnitude, or an
indicator function.

Parameters
----------
- **geometry** (*str or callable*)
- **refinement_field** (*ndarray, shape (num_old_nodes,), optional*)
- **old_mesh** (*Mesh, optional*)
- **h_min** (*float*)
- **h_max** (*float*)


Returns
-------
Mesh
New TET4 mesh with adaptive element sizes.

**Examples**:


```python
# From a STEP file — refine by density
mesh = adaptive_mesh(&quot;bracket.step&quot;, rho_filtered, old_mesh,
                     h_min=0.5, h_max=3.0)

# Refine by density-gradient magnitude
grad_mag = compute_gradient_magnitude(rho_filtered, old_mesh)
grad_norm = grad_mag / grad_mag.max()  # normalise to [0, 1]
mesh = adaptive_mesh(geometry, grad_norm, old_mesh,
                     h_min=0.5, h_max=3.0)
```

#### adaptive\_box\_mesh

```python
def adaptive_box_mesh(size: Union[float, Tuple[float, float, float]],
                      refinement_field: Optional[onp.ndarray] = None,
                      old_mesh: Optional[Mesh] = None,
                      h_min: float = 0.02,
                      h_max: float = 0.15,
                      target_nodes: Optional[int] = None,
                      origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                      periodic: bool = False,
                      max_retries: int = 3) -> Mesh
```

Generate adaptive TET4 box mesh (with optional periodic faces).

Convenience wrapper around :func:`adaptive_mesh` for axis-aligned
boxes.  Adds periodic face meshing for unit-cell homogenisation and
optional node-count tuning.

Parameters
----------
- **size** (*float or (float, float, float)*): Domain dimensions (cube side length, or ``(lx, ly, lz)``).
- **refinement_field** (*ndarray, shape (num_old_nodes,), optional*): Node-based scalar in [0, 1] on the old mesh (see :func:`adaptive_mesh` for details).
- **old_mesh** (*Mesh, optional*): Previous mesh.
- **h_min** (*float*): Minimum element size.
- **h_max** (*float*): Maximum element size.
- **target_nodes** (*int, optional*): If set, iteratively scale h_min/h_max to approximately match this node count (within ±30 %).
- **origin** (*(float, float, float)*): Domain origin.
- **periodic** (*bool*): If True, enforce periodic face meshing via Gmsh.
- **max_retries** (*int*): Maximum mesh generation attempts for node-count tuning.


Returns
-------
Mesh
    New TET4 mesh.

