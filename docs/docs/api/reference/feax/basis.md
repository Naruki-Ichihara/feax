---
sidebar_label: basis
title: feax.basis
---

Shape function computation and element basis definitions for FEAX.

This module provides shape function values and gradients for various finite
element types using the FEniCS Basix library. It handles the conversion
between different node ordering conventions (meshio vs basix) and supports
quadrature rule generation for both volume and surface integrals.

Note: This implementation is adapted from JAX-FEM.

#### get\_elements

```python
def get_elements(
    ele_type: str
) -> Tuple[basix.ElementFamily, basix.CellType, basix.CellType, int, int,
           List[int]]
```

Get element configuration data for basix library integration.

Provides element family, cell types, integration orders, and node re-ordering
transformations needed to properly interface with the FEniCS Basix library.

The re-ordering is necessary because mesh files (Gmsh, Abaqus) use different
node ordering conventions than basix. This function handles the mapping between
meshio ordering (same as Abaqus) and basix ordering.

References
----------
- Abaqus node ordering: https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node33.html
- Basix element definitions: https://defelement.com/elements/lagrange.html

Parameters
----------
- **ele_type** (*str*): Element type identifier (e.g., &#x27;HEX8&#x27;, &#x27;TET4&#x27;, &#x27;QUAD4&#x27;, &#x27;TRI3&#x27;)


Returns
-------
- **element_family** (*basix.ElementFamily*): Basix element family (Lagrange, serendipity, etc.)
- **basix_ele** (*basix.CellType*): Basix cell type for the element
- **basix_face_ele** (*basix.CellType*): Basix cell type for element faces
- **gauss_order** (*int*): Default Gaussian quadrature order
- **degree** (*int*): Polynomial degree of shape functions
- **re_order** (*list[int]*): Node re-ordering transformation from meshio to basix convention


Raises
------
NotImplementedError
    If element type is not supported

Examples
--------
```python
>>> family, elem, face, order, deg, reorder = get_elements(&#x27;HEX8&#x27;)
>>> print(reorder)  # [0, 1, 3, 2, 4, 5, 7, 6]
```

#### reorder\_inds

```python
def reorder_inds(inds: Array, re_order: List[int]) -> Array
```

Apply node re-ordering transformation to match basix conventions.

Converts node indices between meshio ordering (used by mesh files)
and basix ordering (used by FEniCS Basix library).

Parameters
----------
- **inds** (*np.ndarray*): Node indices in original ordering
- **re_order** (*list[int]*): Re-ordering transformation mapping


Returns
-------
np.ndarray
    Node indices in basix ordering

#### get\_shape\_vals\_and\_grads

```python
def get_shape_vals_and_grads(
        ele_type: str,
        gauss_order: Optional[int] = None) -> Tuple[Array, Array, Array]
```

Compute shape function values and gradients using basix.

Generates shape function values, reference gradients, and quadrature
weights for specified element type and integration order.

Parameters
----------
- **ele_type** (*str*): Element type identifier (e.g., &#x27;HEX8&#x27;, &#x27;TET4&#x27;, &#x27;QUAD4&#x27;)
- **gauss_order** (*int, optional*): Gaussian quadrature order. If None, uses element-specific default


Returns
-------
- **shape_values** (*np.ndarray*): Shape function values at quadrature points. Shape: (num_quads, num_nodes)
- **shape_grads_ref** (*np.ndarray*): Shape function gradients in reference coordinates. Shape: (num_quads, num_nodes, dim)
- **weights** (*np.ndarray*): Quadrature weights. Shape: (num_quads,)


Examples
--------
```python
>>> vals, grads, weights = get_shape_vals_and_grads(&#x27;HEX8&#x27;, gauss_order=2)
>>> print(vals.shape)  # (8, 8) for HEX8 with 2x2x2 quadrature
>>> print(grads.shape) # (8, 8, 3) for 3D gradients
```

#### get\_face\_shape\_vals\_and\_grads

```python
def get_face_shape_vals_and_grads(
    ele_type: str,
    gauss_order: Optional[int] = None
) -> Tuple[Array, Array, Array, Array, Array]
```

Compute face shape functions and geometric data for surface integrals.

Generates shape function values, gradients, quadrature data, and geometric
information for element faces needed for boundary/surface integral computations.

Parameters
----------
- **ele_type** (*str*): Element type identifier (e.g., &#x27;HEX8&#x27;, &#x27;TET4&#x27;, &#x27;QUAD4&#x27;)
- **gauss_order** (*int, optional*): Gaussian quadrature order for faces. If None, uses element-specific default


Returns
-------
- **face_shape_vals** (*np.ndarray*): Shape function values at face quadrature points. Shape: (num_faces, num_face_quads, num_nodes)
- **face_shape_grads_ref** (*np.ndarray*): Shape function gradients at face quadrature points in reference coordinates. Shape: (num_faces, num_face_quads, num_nodes, dim)
- **face_weights** (*np.ndarray*): Quadrature weights for face integration (includes Jacobian scaling). Shape: (num_faces, num_face_quads)
- **face_normals** (*np.ndarray*): Outward normal vectors for each face. Shape: (num_faces, dim)
- **face_inds** (*np.ndarray*): Local node indices defining each face. Shape: (num_faces, num_face_vertices)


Examples
--------
```python
>>> vals, grads, weights, normals, inds = get_face_shape_vals_and_grads(&#x27;HEX8&#x27;)
>>> print(vals.shape)    # (6, 4, 8) for HEX8: 6 faces, 4 quad points each
>>> print(normals.shape) # (6, 3) for 6 face normals in 3D
```

