---
sidebar_label: mesh
title: feax.mesh
---

Mesh management and generation utilities for FEAX finite element framework.

This module provides the Mesh class for managing finite element meshes and utility
functions for mesh generation, validation, and format conversion.

## Mesh Objects

```python
class Mesh()
```

Finite element mesh manager.

This class manages mesh data including node coordinates, element connectivity,
and element type information. It provides methods for querying mesh properties
and analyzing boundary conditions.

Parameters
----------
- **points** (*Array*): Node coordinates with shape (num_total_nodes, dim)
- **cells** (*Array*): Element connectivity with shape (num_cells, num_nodes_per_element)
- **ele_type** (*str, optional*): Element type identifier (default: &#x27;TET4&#x27;)


Attributes
----------
- **points** (*Array*): Node coordinates with shape (num_total_nodes, dim)
- **cells** (*Array*): Element connectivity with shape (num_cells, num_nodes_per_element)
- **ele_type** (*str*): Element type identifier


Notes
-----
The element connectivity array should follow the standard node ordering
conventions for each element type.

#### from\_gmsh

```python
@staticmethod
def from_gmsh(gmsh_mesh: meshio.Mesh,
              element_type: Optional[str] = None) -> 'Mesh'
```

Convert meshio.Mesh (from Gmsh) to FEAX Mesh.

This static method converts a meshio.Mesh object (typically from reading
a Gmsh file or using meshio&#x27;s mesh generation) to a FEAX Mesh object.

Parameters
----------
- **gmsh_mesh** (*meshio.Mesh*): Meshio mesh object containing points and cells
- **element_type** (*str, optional*): FEAX element type to use. If None, automatically detects from gmsh_mesh. Supported: &#x27;TET4&#x27;, &#x27;TET10&#x27;, &#x27;HEX8&#x27;, &#x27;HEX20&#x27;, &#x27;HEX27&#x27;, &#x27;TRI3&#x27;, &#x27;TRI6&#x27;, &#x27;QUAD4&#x27;, &#x27;QUAD8&#x27;


Returns
-------
- **mesh** (*Mesh*): FEAX Mesh object


Raises
------
ValueError
    If element_type is not found in gmsh_mesh or is unsupported

Examples
--------
Read Gmsh .msh file and convert:
&gt;&gt;&gt; import meshio
&gt;&gt;&gt; gmsh_mesh = meshio.read(&quot;mesh.msh&quot;)
&gt;&gt;&gt; mesh = Mesh.from_gmsh(gmsh_mesh)

Convert meshio mesh with specific element type:
&gt;&gt;&gt; mesh = Mesh.from_gmsh(gmsh_mesh, element_type=&#x27;HEX8&#x27;)

Notes
-----
- The method automatically maps meshio cell types to FEAX element types
- Only volume elements (3D) and surface elements (2D) are supported
- If multiple element types exist, specify element_type to select one

#### count\_selected\_faces

```python
def count_selected_faces(location_fn: Callable[[np.ndarray], bool]) -> int
```

Count faces that satisfy a location function.

This method is useful for setting up distributed load conditions by
identifying boundary faces that meet specified geometric criteria.

Parameters
----------
- **location_fn** (*Callable[[np.ndarray], bool]*): Function that takes face centroid coordinates and returns True if the face is on the desired boundary


Returns
-------
- **face_count** (*int*): Number of faces satisfying the location function


Notes
-----
This method uses vectorized operations for efficient face selection
and works with all supported element types.

#### check\_mesh\_TET4

```python
def check_mesh_TET4(points: Array, cells: Array) -> np.ndarray
```

Check the node ordering of TET4 elements by computing signed volumes.

This function computes the signed volume of each tetrahedral element to verify
proper node ordering. Negative volumes indicate inverted elements.

Parameters
----------
- **points** (*Array*): Node coordinates with shape (num_nodes, 3)
- **cells** (*Array*): Element connectivity with shape (num_elements, 4)


Returns
-------
- **qualities** (*np.ndarray*): Signed volumes for each element. Positive values indicate proper ordering, negative values indicate inverted elements


Notes
-----
The quality metric is computed as the scalar triple product of edge vectors
from the first node to the other three nodes.

#### get\_meshio\_cell\_type

```python
def get_meshio_cell_type(ele_type: str) -> str
```

Convert FEAX element type to meshio-compatible cell type string.

This function maps FEAX element type identifiers to the corresponding
cell type names used by the meshio library for file I/O operations.

Parameters
----------
- **ele_type** (*str*): FEAX element type identifier (e.g., &#x27;TET4&#x27;, &#x27;HEX8&#x27;, &#x27;TRI3&#x27;, &#x27;QUAD4&#x27;)


Returns
-------
- **cell_type** (*str*): Meshio-compatible cell type name


Raises
------
NotImplementedError
    If the element type is not supported

Notes
-----
Supported element types include:
- TET4, TET10: Tetrahedral elements
- HEX8, HEX20, HEX27: Hexahedral elements  
- TRI3, TRI6: Triangular elements
- QUAD4, QUAD8: Quadrilateral elements

#### box\_mesh

```python
def box_mesh(size: Union[float, Tuple[float, float, float]],
             origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
             mesh_size: float = 0.1,
             element_type: str = 'HEX8',
             recombine: bool = True) -> Mesh
```

Generate structured or unstructured mesh for box domain using Gmsh.

Creates high-quality meshes using Gmsh, supporting both hexahedral (HEX8)
and tetrahedral (TET4) elements with structured or unstructured meshing.

Parameters
----------
- **size** (*float or tuple of 3 floats*): If float: creates cube with side length = size If tuple: (length_x, length_y, length_z) for rectangular box
- **origin** (*tuple of 3 floats, optional*): Origin point (x0, y0, z0) of the box. Default is (0, 0, 0)
- **mesh_size** (*float, optional*): Target element size. Smaller values create finer meshes. Default is 0.1
- **element_type** (*str, optional*): &#x27;HEX8&#x27; for hexahedral elements (default) or &#x27;TET4&#x27; for tetrahedral
- **recombine** (*bool, optional*): If True and element_type=&#x27;HEX8&#x27;, use structured recombination algorithm. Default is True for better quality hexahedral meshes.


Returns
-------
- **mesh** (*Mesh*): Mesh with HEX8 or TET4 elements


Raises
------
ImportError
    If gmsh is not installed
ValueError
    If element_type is not &#x27;HEX8&#x27; or &#x27;TET4&#x27;

Examples
--------
Create structured HEX8 mesh:
&gt;&gt;&gt; mesh = box_mesh_gmsh(1.0, mesh_size=0.1, element_type=&#x27;HEX8&#x27;)

Create unstructured TET4 mesh:
&gt;&gt;&gt; mesh = box_mesh_gmsh((2.0, 1.0, 0.5), mesh_size=0.05, element_type=&#x27;TET4&#x27;)

Notes
-----
- Gmsh provides superior mesh quality compared to simple structured meshes
- HEX8 meshes are preferred for most applications (better accuracy per DOF)
- TET4 meshes are more flexible for complex geometries
- For simple boxes with uniform elements, use box_mesh() for speed

#### rectangle\_mesh

```python
def rectangle_mesh(
    Nx: int,
    Ny: int,
    domain_x: float = 1.0,
    domain_y: float = 1.0,
    origin: Tuple[float, float] = (0.0, 0.0)) -> Mesh
```

Generate structured 2D rectangular mesh with QUAD4 elements.

Creates a simple structured quadrilateral mesh for rectangular domains.
This is a lightweight alternative to Gmsh for simple 2D problems.

Parameters
----------
- **Nx** (*int*): Number of elements in x-direction
- **Ny** (*int*): Number of elements in y-direction
- **domain_x** (*float, optional*): Length of domain in x-direction. Default is 1.0
- **domain_y** (*float, optional*): Length of domain in y-direction. Default is 1.0
- **origin** (*tuple of 2 floats, optional*): Origin point (x0, y0) of the rectangle. Default is (0, 0)


Returns
-------
- **mesh** (*Mesh*): Mesh with QUAD4 elements


Examples
--------
Create 32x32 mesh on unit square:
&gt;&gt;&gt; mesh = rectangle_mesh(Nx=32, Ny=32, domain_x=1.0, domain_y=1.0)

Notes
-----
- Generates (Nx+1) × (Ny+1) nodes
- Generates Nx × Ny QUAD4 elements
- Node ordering follows standard QUAD4 convention

#### sphere\_mesh

```python
def sphere_mesh(radius: float,
                center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                mesh_size: float = 0.1,
                element_type: str = 'TET4') -> Mesh
```

Generate mesh for sphere using Gmsh.

Creates a tetrahedral or hexahedral mesh for a spherical domain.
Note: Hexahedral meshing of spheres is challenging and may produce
lower quality elements.

Parameters
----------
- **radius** (*float*)
- **center** (*tuple of 3 floats, optional*)
- **mesh_size** (*float, optional*)
- **element_type** (*str, optional*)


Returns
-------
- **mesh** (*Mesh*)


Raises
------
ImportError
If gmsh is not installed

Examples
--------
Create TET4 sphere mesh:
&gt;&gt;&gt; mesh = sphere_mesh_gmsh(1.0, mesh_size=0.1)

Notes
-----
- TET4 is strongly recommended for spheres (better quality)
- HEX8 meshes for spheres may have distorted elements

#### cylinder\_mesh

```python
def cylinder_mesh(radius: float,
                  height: float,
                  center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                  axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
                  mesh_size: float = 0.1,
                  element_type: str = 'TET4') -> Mesh
```

Generate mesh for cylinder using Gmsh.

Creates a tetrahedral or hexahedral mesh for a cylindrical domain.

Parameters
----------
- **radius** (*float*): Radius of the cylinder
- **height** (*float*): Height of the cylinder along the axis
- **center** (*tuple of 3 floats, optional*): Center point (x, y, z) of the cylinder base. Default is (0, 0, 0)
- **axis** (*tuple of 3 floats, optional*): Direction vector of cylinder axis. Default is (0, 0, 1) (z-axis)
- **mesh_size** (*float, optional*): Target element size. Default is 0.1
- **element_type** (*str, optional*): &#x27;TET4&#x27; for tetrahedral (default) or &#x27;HEX8&#x27; for hexahedral


Returns
-------
- **mesh** (*Mesh*): Mesh with TET4 or HEX8 elements


Raises
------
ImportError
    If gmsh is not installed

Examples
--------
Create TET4 cylinder mesh:
&gt;&gt;&gt; mesh = cylinder_mesh_gmsh(0.5, 2.0, mesh_size=0.1)

Create HEX8 cylinder mesh:
&gt;&gt;&gt; mesh = cylinder_mesh_gmsh(0.5, 2.0, mesh_size=0.1, element_type=&#x27;HEX8&#x27;)

