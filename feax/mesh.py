"""
Mesh management and generation utilities for FEAX finite element framework.

This module provides the Mesh class for managing finite element meshes and utility
functions for mesh generation, validation, and format conversion.
"""

import os
from typing import Tuple, Callable, Optional, TYPE_CHECKING
import numpy as onp
import meshio

from feax.basis import get_face_shape_vals_and_grads

import jax
import jax.numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Mesh():
    """Finite element mesh manager.
    
    This class manages mesh data including node coordinates, element connectivity,
    and element type information. It provides methods for querying mesh properties
    and analyzing boundary conditions.

    Parameters
    ----------
    points : NDArray
        Node coordinates with shape (num_total_nodes, dim)
    cells : NDArray
        Element connectivity with shape (num_cells, num_nodes_per_element)
    ele_type : str, optional
        Element type identifier (default: 'TET4')
        
    Attributes
    ----------
    points : NDArray
        Node coordinates with shape (num_total_nodes, dim)
    cells : NDArray
        Element connectivity with shape (num_cells, num_nodes_per_element)  
    ele_type : str
        Element type identifier
        
    Notes
    -----
    The element connectivity array should follow the standard node ordering
    conventions for each element type.
    """
    def __init__(self, points: 'NDArray', cells: 'NDArray', ele_type: str = 'TET4') -> None:
        # TODO (Very important for debugging purpose!): Assert that cells must have correct orders
        self.points = points
        self.cells = cells
        self.ele_type = ele_type

    def count_selected_faces(self, location_fn: Callable[[np.ndarray], bool]) -> int:
        """Count faces that satisfy a location function.
        
        This method is useful for setting up distributed load conditions by
        identifying boundary faces that meet specified geometric criteria.

        Parameters
        ----------
        location_fn : Callable[[np.ndarray], bool]
            Function that takes face centroid coordinates and returns True
            if the face is on the desired boundary

        Returns
        -------
        face_count : int
            Number of faces satisfying the location function
            
        Notes
        -----
        This method uses vectorized operations for efficient face selection
        and works with all supported element types.
        """
        _, _, _, _, face_inds = get_face_shape_vals_and_grads(self.ele_type)
        cell_points = onp.take(self.points, self.cells, axis=0)
        cell_face_points = onp.take(cell_points, face_inds, axis=1)

        vmap_location_fn = jax.vmap(location_fn)

        def on_boundary(cell_points):
            boundary_flag = vmap_location_fn(cell_points)
            return onp.all(boundary_flag)

        vvmap_on_boundary = jax.vmap(jax.vmap(on_boundary))
        boundary_flags = vvmap_on_boundary(cell_face_points)
        boundary_inds = onp.argwhere(boundary_flags)
        return boundary_inds.shape[0]


def check_mesh_TET4(points: 'NDArray', cells: 'NDArray') -> np.ndarray:
    """Check the node ordering of TET4 elements by computing signed volumes.
    
    This function computes the signed volume of each tetrahedral element to verify
    proper node ordering. Negative volumes indicate inverted elements.

    Parameters
    ----------
    points : NDArray
        Node coordinates with shape (num_nodes, 3)
    cells : NDArray  
        Element connectivity with shape (num_elements, 4)

    Returns
    -------
    qualities : np.ndarray
        Signed volumes for each element. Positive values indicate proper ordering,
        negative values indicate inverted elements
        
    Notes
    -----
    The quality metric is computed as the scalar triple product of edge vectors
    from the first node to the other three nodes.
    """
    def quality(pts):
        p1, p2, p3, p4 = pts
        v1 = p2 - p1
        v2 = p3 - p1
        v12 = np.cross(v1, v2)
        v3 = p4 - p1
        return np.dot(v12, v3)
    qlts = jax.vmap(quality)(points[cells])
    return qlts

def get_meshio_cell_type(ele_type: str) -> str:
    """Convert FEAX element type to meshio-compatible cell type string.
    
    This function maps FEAX element type identifiers to the corresponding
    cell type names used by the meshio library for file I/O operations.

    Parameters
    ----------
    ele_type : str
        FEAX element type identifier (e.g., 'TET4', 'HEX8', 'TRI3', 'QUAD4')

    Returns
    -------
    cell_type : str
        Meshio-compatible cell type name
        
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
    """
    if ele_type == 'TET4':
        cell_type = 'tetra'
    elif ele_type == 'TET10':
        cell_type = 'tetra10'
    elif ele_type == 'HEX8':
        cell_type = 'hexahedron'
    elif ele_type == 'HEX27':
        cell_type = 'hexahedron27'
    elif  ele_type == 'HEX20':
        cell_type = 'hexahedron20'
    elif ele_type == 'TRI3':
        cell_type = 'triangle'
    elif ele_type == 'TRI6':
        cell_type = 'triangle6'
    elif ele_type == 'QUAD4':
        cell_type = 'quad'
    elif ele_type == 'QUAD8':
        cell_type = 'quad8'
    else:
        raise NotImplementedError
    return cell_type


def rectangle_mesh(Nx: int, Ny: int, domain_x: float, domain_y: float) -> Mesh:
    """Generate structured QUAD4 mesh for rectangular domain.
    
    Creates a structured quadrilateral mesh over a rectangular domain with
    uniform spacing in both directions.

    Parameters
    ----------
    Nx : int
        Number of elements along x-axis
    Ny : int
        Number of elements along y-axis
    domain_x : float
        Length of domain along x-axis
    domain_y : float
        Length of domain along y-axis
        
    Returns
    -------
    mesh : Mesh
        Structured rectangular mesh with QUAD4 elements
        
    Notes
    -----
    The mesh spans from (0, 0) to (domain_x, domain_y) with uniform element spacing.
    Node numbering follows standard conventions with elements oriented counter-clockwise.
    """
    dim = 2
    x = onp.linspace(0, domain_x, Nx + 1)
    y = onp.linspace(0, domain_y, Ny + 1)
    xv, yv = onp.meshgrid(x, y, indexing='ij')
    points_xy = onp.stack((xv, yv), axis=dim)
    points = points_xy.reshape(-1, dim)
    points_inds = onp.arange(len(points))
    points_inds_xy = points_inds.reshape(Nx + 1, Ny + 1)
    inds1 = points_inds_xy[:-1, :-1]
    inds2 = points_inds_xy[1:, :-1]
    inds3 = points_inds_xy[1:, 1:]
    inds4 = points_inds_xy[:-1, 1:]
    cells = onp.stack((inds1, inds2, inds3, inds4), axis=dim).reshape(-1, 4)
    mesh = meshio.Mesh(points=points, cells={'quad': cells})
    return Mesh(mesh.points, mesh.cells_dict['quad'], ele_type="QUAD4")


def box_mesh(Nx: int, Ny: int, Nz: int, domain_x: float, domain_y: float, domain_z: float) -> Mesh:
    """Generate structured HEX8 mesh for box domain.
    
    Creates a structured hexahedral mesh over a box domain with uniform
    spacing in all three directions.

    Parameters
    ----------
    Nx : int
        Number of elements along x-axis
    Ny : int
        Number of elements along y-axis
    Nz : int
        Number of elements along z-axis
    domain_x : float
        Length of domain along x-axis
    domain_y : float
        Length of domain along y-axis
    domain_z : float
        Length of domain along z-axis
        
    Returns
    -------
    mesh : Mesh
        Structured box mesh with HEX8 elements
        
    Notes
    -----
    The mesh spans from (0, 0, 0) to (domain_x, domain_y, domain_z) with uniform 
    element spacing. Node numbering follows standard hexahedral conventions.
    """
    dim = 3
    x = onp.linspace(0, domain_x, Nx + 1)
    y = onp.linspace(0, domain_y, Ny + 1)
    z = onp.linspace(0, domain_z, Nz + 1)
    xv, yv, zv = onp.meshgrid(x, y, z, indexing='ij')
    points_xyz = onp.stack((xv, yv, zv), axis=dim)
    points = points_xyz.reshape(-1, dim)
    points_inds = onp.arange(len(points))
    points_inds_xyz = points_inds.reshape(Nx + 1, Ny + 1, Nz + 1)
    inds1 = points_inds_xyz[:-1, :-1, :-1]
    inds2 = points_inds_xyz[1:, :-1, :-1]
    inds3 = points_inds_xyz[1:, 1:, :-1]
    inds4 = points_inds_xyz[:-1, 1:, :-1]
    inds5 = points_inds_xyz[:-1, :-1, 1:]
    inds6 = points_inds_xyz[1:, :-1, 1:]
    inds7 = points_inds_xyz[1:, 1:, 1:]
    inds8 = points_inds_xyz[:-1, 1:, 1:]
    cells = onp.stack((inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8),
                      axis=dim).reshape(-1, 8)
    mesh = meshio.Mesh(points=points, cells={'hexahedron': cells})
    return Mesh(mesh.points, mesh.cells_dict['hexahedron'], ele_type="HEX8")