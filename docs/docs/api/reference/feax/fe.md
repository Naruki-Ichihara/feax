---
sidebar_label: fe
title: feax.fe
---

Finite element class implementation for FEAX framework.

This module provides the FiniteElement class that handles shape functions,
quadrature rules, and geometric computations for individual variables in
finite element problems.

## FiniteElement Objects

```python
@dataclass
class FiniteElement()
```

Finite element class for a single variable with shape functions and quadrature.

This class handles all geometric and computational aspects for one variable in
a finite element problem, including shape functions, quadrature rules, and
transformations between reference and physical domains.

Parameters
----------
- **mesh** (*Mesh*): Finite element mesh containing node coordinates and element connectivity
- **vec** (*int*): Number of vector components in solution (e.g., 3 for 3D displacement, 1 for temperature)
- **dim** (*int*): Spatial dimension of the problem (2D or 3D)
- **ele_type** (*str, optional*): Element type identifier (default: &#x27;HEX8&#x27;) Supported: &#x27;TET4&#x27;, &#x27;TET10&#x27;, &#x27;HEX8&#x27;, &#x27;HEX20&#x27;, &#x27;HEX27&#x27;, &#x27;TRI3&#x27;, &#x27;TRI6&#x27;, &#x27;QUAD4&#x27;, &#x27;QUAD8&#x27;
- **gauss_order** (*Optional[int], optional*): Gaussian quadrature order (default: determined by element type)


Attributes
----------
- **num_cells** (*int*): Number of elements in the mesh
- **num_nodes** (*int*): Number of nodes per element
- **num_total_nodes** (*int*): Total number of nodes in the mesh
- **num_total_dofs** (*int*): Total degrees of freedom for this variable
- **num_quads** (*int*): Number of quadrature points per element
- **shape_vals** (*np.ndarray*): Shape function values at quadrature points
- **shape_grads** (*np.ndarray*): Shape function gradients in physical coordinates
- **JxW** (*np.ndarray*): Jacobian determinant times quadrature weights


#### get\_shape\_grads

```python
def get_shape_grads() -> Tuple[Array, Array]
```

Compute shape function gradients in physical coordinates.

Transforms shape function gradients from reference coordinates to physical
coordinates using the Jacobian transformation. Also computes the Jacobian
determinant times quadrature weights for integration.

References
----------
Hughes, Thomas JR. The finite element method: linear static and dynamic
finite element analysis. Courier Corporation, 2012. Page 147, Eq. (3.9.3)

Returns
-------
- **shape_grads_physical** (*np.ndarray*): Shape function gradients in physical coordinates. Shape: (num_cells, num_quads, num_nodes, dim)
- **JxW** (*np.ndarray*): Jacobian determinant times quadrature weights for integration. Shape: (num_cells, num_quads)


#### get\_face\_shape\_grads

```python
def get_face_shape_grads(boundary_inds: Array) -> Tuple[Array, Array]
```

Compute face shape function gradients and surface integration scaling.

Uses Nanson&#x27;s formula to transform surface integrals from physical domain
to reference domain. Computes shape function gradients on boundary faces
and the scaling factor needed for surface integration.

References
----------
Wikiversity: Continuum mechanics/Volume change and area change
https://en.wikiversity.org/wiki/Continuum_mechanics/Volume_change_and_area_change

Parameters
----------
- **boundary_inds** (*np.ndarray*): Boundary face indices with shape (num_selected_faces, 2). First column: element index, Second column: local face index


Returns
-------
- **face_shape_grads_physical** (*np.ndarray*): Face shape function gradients in physical coordinates. Shape: (num_selected_faces, num_face_quads, num_nodes, dim)
- **nanson_scale** (*np.ndarray*): Surface integration scaling factor (Jacobian * weights). Shape: (num_selected_faces, num_face_quads)


#### get\_physical\_quad\_points

```python
def get_physical_quad_points() -> Array
```

Compute physical coordinates of quadrature points.

Maps quadrature points from reference element to physical coordinates
using shape function interpolation.

Returns
-------
- **physical_quad_points** (*np.ndarray*): Physical coordinates of quadrature points. Shape: (num_cells, num_quads, dim)


#### get\_physical\_surface\_quad\_points

```python
def get_physical_surface_quad_points(boundary_inds: Array) -> Array
```

Compute physical coordinates of surface quadrature points.

Maps surface quadrature points from reference faces to physical coordinates
using face shape function interpolation.

Parameters
----------
- **boundary_inds** (*np.ndarray*): Boundary face indices with shape (num_selected_faces, 2). First column: element index, Second column: local face index


Returns
-------
- **physical_surface_quad_points** (*np.ndarray*): Physical coordinates of surface quadrature points. Shape: (num_selected_faces, num_face_quads, dim)


#### get\_boundary\_conditions\_inds

```python
def get_boundary_conditions_inds(
        location_fns: Optional[List[Callable]]) -> List[Array]
```

Identify boundary faces that satisfy location function conditions.

Determines which element faces lie on boundaries defined by location functions.
Used internally for surface integral computations.

Parameters
----------
- **location_fns** (*list[callable] or None*)


#### convert\_from\_dof\_to\_quad

```python
def convert_from_dof_to_quad(sol: Array) -> Array
```

Interpolate nodal solution values to quadrature points.

Uses shape functions to interpolate solution from nodes to quadrature
points within each element.

Parameters
----------
- **sol** (*np.ndarray*): Nodal solution values with shape (num_total_nodes, vec)


Returns
-------
- **u** (*np.ndarray*): Solution values at quadrature points. Shape: (num_cells, num_quads, vec)


#### convert\_from\_dof\_to\_face\_quad

```python
def convert_from_dof_to_face_quad(sol: Array, boundary_inds: Array) -> Array
```

Interpolate nodal solution to surface quadrature points.

Uses face shape functions to interpolate solution from nodes to
quadrature points on boundary faces.

Parameters
----------
- **sol** (*np.ndarray*): Nodal solution values with shape (num_total_nodes, vec)
- **boundary_inds** (*np.ndarray*): Boundary face indices with shape (num_selected_faces, 2)


Returns
-------
- **u** (*np.ndarray*): Solution values at surface quadrature points. Shape: (num_selected_faces, num_face_quads, vec)


#### sol\_to\_grad

```python
def sol_to_grad(sol: Array) -> Array
```

Compute solution gradients at quadrature points.

Uses shape function gradients to compute spatial derivatives of the
solution at quadrature points within each element.

Parameters
----------
- **sol** (*np.ndarray*): Nodal solution values with shape (num_total_nodes, vec)


Returns
-------
- **u_grads** (*np.ndarray*): Solution gradients at quadrature points. Shape: (num_cells, num_quads, vec, dim)


#### print\_BC\_info

```python
def print_BC_info() -> None
```

Print boundary condition information for debugging.

Note: This method is deprecated and may not work correctly.
Use DirichletBC class from DCboundary module for BC handling.

