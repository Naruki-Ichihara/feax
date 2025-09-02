"""
Shape function computation and element basis definitions for FEAX.

This module provides shape function values and gradients for various finite
element types using the FEniCS Basix library. It handles the conversion
between different node ordering conventions (meshio vs basix) and supports
quadrature rule generation for both volume and surface integrals.

Note: This implementation is adapted from JAX-FEM.
"""

import basix
import numpy as onp
from typing import Tuple, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

def get_elements(ele_type: str) -> Tuple[basix.ElementFamily, basix.CellType, basix.CellType, int, int, List[int]]:
    """Get element configuration data for basix library integration.
    
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
    ele_type : str
        Element type identifier (e.g., 'HEX8', 'TET4', 'QUAD4', 'TRI3')

    Returns
    -------
    element_family : basix.ElementFamily
        Basix element family (Lagrange, serendipity, etc.)
    basix_ele : basix.CellType
        Basix cell type for the element
    basix_face_ele : basix.CellType
        Basix cell type for element faces
    gauss_order : int
        Default Gaussian quadrature order
    degree : int
        Polynomial degree of shape functions
    re_order : list[int]
        Node re-ordering transformation from meshio to basix convention
        
    Raises
    ------
    NotImplementedError
        If element type is not supported
        
    Examples
    --------
    >>> family, elem, face, order, deg, reorder = get_elements('HEX8')
    >>> print(reorder)  # [0, 1, 3, 2, 4, 5, 7, 6]
    """
    element_family = basix.ElementFamily.P
    if ele_type == 'HEX8':
        re_order = [0, 1, 3, 2, 4, 5, 7, 6]
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 2 # 2x2x2, TODO: is this full integration?
        degree = 1
    elif ele_type == 'HEX20':
        re_order = [0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 15, 14]
        element_family = basix.ElementFamily.serendipity
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 2 # 6x6x6, full integration
        degree = 2
    elif ele_type == 'TET4':
        re_order = [0, 1, 2, 3]
        basix_ele = basix.CellType.tetrahedron
        basix_face_ele = basix.CellType.triangle
        gauss_order = 0 # 1, full integration
        degree = 1
    elif ele_type == 'TET10':
        re_order = [0, 1, 2, 3, 9, 6, 8, 7, 5, 4]
        basix_ele = basix.CellType.tetrahedron
        basix_face_ele = basix.CellType.triangle
        gauss_order = 2 # 4, full integration
        degree = 2
    elif ele_type == 'QUAD4':
        re_order = [0, 1, 3, 2]
        basix_ele = basix.CellType.quadrilateral
        basix_face_ele = basix.CellType.interval
        gauss_order = 2
        degree = 1
    elif ele_type == 'QUAD8':
        re_order = [0, 1, 3, 2, 4, 6, 7, 5]
        element_family = basix.ElementFamily.serendipity
        basix_ele = basix.CellType.quadrilateral
        basix_face_ele = basix.CellType.interval
        gauss_order = 2
        degree = 2
    elif ele_type == 'TRI3':
        re_order = [0, 1, 2]
        basix_ele = basix.CellType.triangle
        basix_face_ele = basix.CellType.interval
        gauss_order = 0 # 1, full integration
        degree = 1
    elif ele_type == 'TRI6':
        re_order = [0, 1, 2, 5, 3, 4]
        basix_ele = basix.CellType.triangle
        basix_face_ele = basix.CellType.interval
        gauss_order = 2 # 3, full integration
        degree = 2
    else:
        raise NotImplementedError(f"Element type '{ele_type}' is not supported. "
                                f"Supported types: HEX8, HEX20, TET4, TET10, "
                                f"QUAD4, QUAD8, TRI3, TRI6")

    return element_family, basix_ele, basix_face_ele, gauss_order, degree, re_order


def reorder_inds(inds: 'NDArray', re_order: List[int]) -> 'NDArray':
    """Apply node re-ordering transformation to match basix conventions.
    
    Converts node indices between meshio ordering (used by mesh files)
    and basix ordering (used by FEniCS Basix library).
    
    Parameters
    ----------
    inds : np.ndarray
        Node indices in original ordering
    re_order : list[int]
        Re-ordering transformation mapping
        
    Returns
    -------
    np.ndarray
        Node indices in basix ordering
    """

    new_inds = []
    for ind in inds.reshape(-1):
        new_inds.append(onp.argwhere(re_order == ind))
    new_inds = onp.array(new_inds).reshape(inds.shape)
    return new_inds


def get_shape_vals_and_grads(ele_type: str, gauss_order: Optional[int] = None) -> Tuple['NDArray', 'NDArray', 'NDArray']:
    """Compute shape function values and gradients using basix.
    
    Generates shape function values, reference gradients, and quadrature
    weights for specified element type and integration order.

    Parameters
    ----------
    ele_type : str
        Element type identifier (e.g., 'HEX8', 'TET4', 'QUAD4')
    gauss_order : int, optional
        Gaussian quadrature order. If None, uses element-specific default

    Returns
    -------
    shape_values : np.ndarray
        Shape function values at quadrature points.
        Shape: (num_quads, num_nodes)
    shape_grads_ref : np.ndarray
        Shape function gradients in reference coordinates.
        Shape: (num_quads, num_nodes, dim)
    weights : np.ndarray
        Quadrature weights.
        Shape: (num_quads,)
        
    Examples
    --------
    >>> vals, grads, weights = get_shape_vals_and_grads('HEX8', gauss_order=2)
    >>> print(vals.shape)  # (8, 8) for HEX8 with 2x2x2 quadrature
    >>> print(grads.shape) # (8, 8, 3) for 3D gradients
    """
    element_family, basix_ele, basix_face_ele, gauss_order_default, degree, re_order = get_elements(ele_type)

    if gauss_order is None:
        gauss_order = gauss_order_default

    quad_points, weights = basix.make_quadrature(basix_ele, gauss_order)
    element = basix.create_element(element_family, basix_ele, degree)
    vals_and_grads = element.tabulate(1, quad_points)[:, :, re_order, :]
    shape_values = vals_and_grads[0, :, :, 0]
    shape_grads_ref = onp.transpose(vals_and_grads[1:, :, :, 0], axes=(1, 2, 0))
    return shape_values, shape_grads_ref, weights


def get_face_shape_vals_and_grads(ele_type: str, gauss_order: Optional[int] = None) -> Tuple['NDArray', 'NDArray', 'NDArray', 'NDArray', 'NDArray']:
    """Compute face shape functions and geometric data for surface integrals.
    
    Generates shape function values, gradients, quadrature data, and geometric
    information for element faces needed for boundary/surface integral computations.

    Parameters
    ----------
    ele_type : str
        Element type identifier (e.g., 'HEX8', 'TET4', 'QUAD4')
    gauss_order : int, optional
        Gaussian quadrature order for faces. If None, uses element-specific default

    Returns
    -------
    face_shape_vals : np.ndarray
        Shape function values at face quadrature points.
        Shape: (num_faces, num_face_quads, num_nodes)
    face_shape_grads_ref : np.ndarray
        Shape function gradients at face quadrature points in reference coordinates.
        Shape: (num_faces, num_face_quads, num_nodes, dim)
    face_weights : np.ndarray
        Quadrature weights for face integration (includes Jacobian scaling).
        Shape: (num_faces, num_face_quads)
    face_normals : np.ndarray
        Outward normal vectors for each face.
        Shape: (num_faces, dim)
    face_inds : np.ndarray
        Local node indices defining each face.
        Shape: (num_faces, num_face_vertices)
        
    Examples
    --------
    >>> vals, grads, weights, normals, inds = get_face_shape_vals_and_grads('HEX8')
    >>> print(vals.shape)    # (6, 4, 8) for HEX8: 6 faces, 4 quad points each
    >>> print(normals.shape) # (6, 3) for 6 face normals in 3D
    """
    element_family, basix_ele, basix_face_ele, gauss_order_default, degree, re_order = get_elements(ele_type)

    if gauss_order is None:
        gauss_order = gauss_order_default

    points, weights = basix.make_quadrature(basix_face_ele, gauss_order)

    map_degree = 1
    lagrange_map = basix.create_element(basix.ElementFamily.P, basix_face_ele, map_degree)
    values = lagrange_map.tabulate(0, points)[0, :, :, 0]
    vertices = basix.geometry(basix_ele)
    dim = len(vertices[0])
    facets = basix.cell.sub_entity_connectivity(basix_ele)[dim - 1]
    # Map face points
    # Reference: https://docs.fenicsproject.org/basix/main/python/demo/demo_facet_integral.py.html
    face_quad_points = []
    face_inds = []
    face_weights = []
    for f, facet in enumerate(facets):
        mapped_points = []
        for i in range(len(points)):
            vals = values[i]
            mapped_point = onp.sum(vertices[facet[0]] * vals[:, None], axis=0)
            mapped_points.append(mapped_point)
        face_quad_points.append(mapped_points)
        face_inds.append(facet[0])
        jacobian = basix.cell.facet_jacobians(basix_ele)[f]
        if dim == 2:
            size_jacobian = onp.linalg.norm(jacobian)
        else:
            size_jacobian = onp.linalg.norm(onp.cross(jacobian[:, 0], jacobian[:, 1]))
        face_weights.append(weights*size_jacobian)
    face_quad_points = onp.stack(face_quad_points)
    face_weights = onp.stack(face_weights)

    face_normals = basix.cell.facet_outward_normals(basix_ele)
    face_inds = onp.array(face_inds)
    face_inds = reorder_inds(face_inds, re_order)
    num_faces, num_face_quads, dim = face_quad_points.shape
    element = basix.create_element(element_family, basix_ele, degree)
    vals_and_grads = element.tabulate(1, face_quad_points.reshape(-1, dim))[:, :, re_order, :]
    face_shape_vals = vals_and_grads[0, :, :, 0].reshape(num_faces, num_face_quads, -1)
    face_shape_grads_ref = vals_and_grads[1:, :, :, 0].reshape(dim, num_faces, num_face_quads, -1)
    face_shape_grads_ref = onp.transpose(face_shape_grads_ref, axes=(1, 2, 3, 0))
    return face_shape_vals, face_shape_grads_ref, face_weights, face_normals, face_inds
