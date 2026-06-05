---
sidebar_label: mesh_extraction
title: feax.gene.mesh_extraction
---

Surface and volume mesh extraction from topology optimization results.

Extracts smooth iso-surfaces from node-based density fields using VTK&#x27;s
contour filter on the unstructured FE mesh. Unlike grid-based marching
cubes, this interpolates along element edges and produces smooth surfaces
that respect the FE shape function continuity.

Requires ``pyvista`` (``pip install pyvista``).

Example
-------
```python
>>> result = gene.optimizer.run(...)
>>> surface = extract_surface(result.rho_filtered, result.mesh, threshold=0.5)
>>> surface.save(&quot;optimized.stl&quot;)
>>>
>>> # With Gmsh volume remeshing
>>> fe_mesh = extract_volume_mesh(result.rho_filtered, result.mesh, mesh_size=0.05)
```

#### extract\_surface

```python
def extract_surface(density: onp.ndarray,
                    mesh,
                    threshold: float = 0.5,
                    smooth_iterations: int = 0) -> 'pyvista.PolyData'
```

Extract a closed, manifold surface mesh from a density field.

Uses VTK&#x27;s ``clip_scalar`` to keep the solid region (density &gt;= threshold),
then extracts the outer surface. The result is a closed (manifold) triangle
mesh where:

- **Interior faces** are smooth iso-surfaces interpolated along element edges
- **Boundary faces** are the domain boundary where the solid meets the box edge

Parameters
----------
- **density** (*ndarray, shape (num_nodes,)*): Node-based density field (e.g. ``result.rho_filtered``).
- **mesh** (*feax.Mesh*): The finite element mesh.
- **threshold** (*float*): Iso-density level defining the solid boundary (default 0.5).
- **smooth_iterations** (*int*): Laplacian smoothing passes on the extracted surface (default 0).


Returns
-------
pyvista.PolyData
    Closed triangle surface mesh.

Example
-------
```python
>>> surface = extract_surface(result.rho_filtered, result.mesh)
>>> surface.save(&quot;optimized.stl&quot;)
>>> print(f&quot;Manifold: {`surface.is_manifold`}&quot;)
```

#### extract\_volume\_mesh

```python
def extract_volume_mesh(density: onp.ndarray,
                        mesh,
                        threshold: float = 0.5,
                        mesh_size: Optional[float] = None,
                        element_type: str = 'TET4') -> 'feax.Mesh'
```

Extract a clean volume mesh from a density field via surface remeshing.

Pipeline: density field → VTK iso-surface → STL → Gmsh volume mesh → feax Mesh.

Parameters
----------
- **density** (*ndarray, shape (num_nodes,)*): Node-based density field.
- **mesh** (*feax.Mesh*): The finite element mesh.
- **threshold** (*float*): Iso-density level (default 0.5).
- **mesh_size** (*float, optional*): Target element size for Gmsh. Defaults to average edge length of the input mesh.
- **element_type** (*str*): ``&#x27;TET4&#x27;`` (default) or ``&#x27;TET10&#x27;``.


Returns
-------
feax.Mesh
    Volumetric finite element mesh of the solid region.

#### from\_opt\_result

```python
def from_opt_result(result, threshold: float = 0.5) -> 'pyvista.PolyData'
```

Extract surface mesh from an OptResult.

Convenience wrapper around :func:`extract_surface` that uses
``result.rho_filtered`` and ``result.mesh``.

Parameters
----------
- **result** (*OptResult*): Output of ``feax.gene.optimizer.run()``.
- **threshold** (*float*): Iso-density level (default 0.5).


Returns
-------
pyvista.PolyData

