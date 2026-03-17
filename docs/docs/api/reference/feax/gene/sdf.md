---
sidebar_label: sdf
title: feax.gene.sdf
---

Signed distance field conversion for topology optimization results.

Converts node-based density fields from topology optimization into
queryable signed distance fields (SDFs) and exports surface meshes
via marching cubes.

The interface follows jaxcad conventions: an SDF is a callable
``f(p) -&gt; distance`` where negative values indicate the interior.

Supports all element types in feax (TET4, TET10, HEX8, HEX20,
TRI3, TRI6, QUAD4, QUAD8) via basix shape functions and Newton
inverse mapping.

Example
-------
```python
>>> result = gene.optimizer.run(...)
>>> field = DensityField(result.rho_filtered, result.mesh, threshold=0.5)
>>> verts, faces = field.to_mesh(resolution=80)
>>> field.to_stl(&quot;optimized.stl&quot;, resolution=80)
```

## SDF Objects

```python
class SDF(ABC)
```

Abstract signed distance function.

Convention: ``f(p) &lt; 0`` inside, ``f(p) = 0`` on surface,
``f(p) &gt; 0`` outside.

#### \_\_call\_\_

```python
@abstractmethod
def __call__(p: onp.ndarray) -> float
```

Evaluate signed distance at a single point *p*.

#### evaluate

```python
def evaluate(points: onp.ndarray) -> onp.ndarray
```

Vectorised evaluation on an array of points ``(N, dim) -&gt; (N,)``.

#### \_\_or\_\_

```python
def __or__(other: SDF) -> SDF
```

Union: ``self | other``.

#### \_\_and\_\_

```python
def __and__(other: SDF) -> SDF
```

Intersection: ``self &amp; other``.

#### \_\_sub\_\_

```python
def __sub__(other: SDF) -> SDF
```

Difference: ``self - other``.

#### to\_mesh

```python
def to_mesh(resolution: int = 80,
            bounds: Optional[Tuple[onp.ndarray, onp.ndarray]] = None,
            level: float = 0.0,
            padding: float = 0.0,
            watertight: bool = False) -> Tuple[onp.ndarray, onp.ndarray]
```

Extract a triangle surface mesh via marching cubes.

Parameters
----------
- **resolution** (*int*): Number of grid points per axis.
- **bounds** (*(lower, upper), optional*): Axis-aligned bounding box as two arrays. Defaults to the bounding box stored on the SDF (if any).
- **level** (*float*): Iso-level for marching cubes (default 0.0 = surface).
- **padding** (*float*): Extra space added around bounds in each direction.
- **watertight** (*bool*): If ``True``, post-process the mesh with trimesh to guarantee a watertight (manifold, closed) surface. Merges duplicate vertices, removes degenerate faces, fills holes, and fixes normals.  Requires ``trimesh``.


Returns
-------
- **vertices** (*ndarray, shape (V, 3)*)


#### to\_stl

```python
def to_stl(path: str,
           resolution: int = 80,
           bounds: Optional[Tuple[onp.ndarray, onp.ndarray]] = None,
           level: float = 0.0,
           padding: float = 0.0,
           watertight: bool = False) -> None
```

Export the iso-surface as an STL file.

Parameters are forwarded to :meth:`to_mesh`.

## Union Objects

```python
class Union(SDF)
```

Boolean union of two SDFs (``min``).

## Intersection Objects

```python
class Intersection(SDF)
```

Boolean intersection of two SDFs (``max``).

## Difference Objects

```python
class Difference(SDF)
```

Boolean difference ``a - b``  (i.e. ``max(a, -b)``).

## DensityField Objects

```python
class DensityField(SDF)
```

Convert a node-based density field on an FE mesh into an SDF.

The signed distance is approximated as ``threshold - density(p)``:

* density &gt; threshold  →  SDF &lt; 0  (inside / solid)
* density &lt; threshold  →  SDF &gt; 0  (outside / void)
* density = threshold  →  SDF = 0  (surface)

Supports all element types available in feax (TET4, TET10, HEX8,
HEX20, TRI3, TRI6, QUAD4, QUAD8).  Uses basix shape functions
for interpolation and Newton iteration for inverse mapping from
physical to reference coordinates.

For linear simplex elements (TET4, TRI3) a fast direct
barycentric path is used instead of Newton iteration.

Parameters
----------
- **rho** (*ndarray, shape (num_nodes,)*): Node-based density (filtered / projected).
- **mesh** (*feax.Mesh*): Finite element mesh.
- **threshold** (*float*): Iso-density level that defines the solid boundary.
- **close_boundary** (*bool*): If ``True`` (default), points outside the FE mesh are treated as void (density = 0) so that the marching-cubes surface closes at the domain boundary.  If ``False``, the nearest-node density is used instead (legacy behaviour).


#### evaluate

```python
def evaluate(points: onp.ndarray) -> onp.ndarray
```

Evaluate SDF at many points ``(N, dim) -&gt; (N,)``.

Uses batched Newton inverse mapping with basix shape functions.
For TET4/TRI3 (linear simplex), falls back to the fast direct
barycentric path.

#### bounds

```python
@property
def bounds() -> Tuple[onp.ndarray, onp.ndarray]
```

Axis-aligned bounding box ``(lower, upper)`` of the FE mesh.

#### from\_opt\_result

```python
def from_opt_result(result, threshold: float = 0.5) -> DensityField
```

Create a :class:`DensityField` from an :class:`~feax.gene.optimizer.OptResult`.

Uses the filtered density (``result.rho_filtered``) and the final
mesh (``result.mesh``).

Parameters
----------
- **result** (*OptResult*): Output of ``feax.gene.optimizer.run()``.
- **threshold** (*float*): Iso-density level (default 0.5).


Returns
-------
DensityField

#### from\_vtk

```python
def from_vtk(path: str,
             field_name: str = "density",
             threshold: float = 0.5) -> DensityField
```

Create a :class:`DensityField` from a VTK/VTU file.

Reads a mesh file (typically ``.vtu`` saved by
``feax.utils.save_sol``) containing a node-based scalar field
and wraps it as an SDF.

Supports all element types that feax supports (TET4, TET10,
HEX8, HEX20, TRI3, TRI6, QUAD4, QUAD8).

Parameters
----------
- **path** (*str*): Path to the VTK/VTU file.
- **field_name** (*str*): Name of the point-data array to use as density (default ``&quot;density&quot;``).
- **threshold** (*float*): Iso-density level (default 0.5).


Returns
-------
DensityField

Raises
------
KeyError
    If *field_name* is not found in the file&#x27;s point data.

Example
-------
```python
>>> field = from_vtk(&quot;output/final.vtu&quot;)
>>> field.to_stl(&quot;result.stl&quot;, resolution=100)
```

