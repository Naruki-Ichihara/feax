"""
Mesh management and generation utilities for FEAX finite element framework.

This module provides the Mesh class for managing finite element meshes and utility
functions for mesh generation, validation, and format conversion.
"""

import os
from typing import Tuple, Callable, Optional, Union, TYPE_CHECKING
import numpy as onp
import meshio

from feax.basis import get_face_shape_vals_and_grads

import jax
import jax.numpy as np

try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False

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

    @staticmethod
    def from_gmsh(gmsh_mesh: meshio.Mesh, element_type: Optional[str] = None) -> 'Mesh':
        """Convert meshio.Mesh (from Gmsh) to FEAX Mesh.

        This static method converts a meshio.Mesh object (typically from reading
        a Gmsh file or using meshio's mesh generation) to a FEAX Mesh object.

        Parameters
        ----------
        gmsh_mesh : meshio.Mesh
            Meshio mesh object containing points and cells
        element_type : str, optional
            FEAX element type to use. If None, automatically detects from gmsh_mesh.
            Supported: 'TET4', 'TET10', 'HEX8', 'HEX20', 'HEX27', 'TRI3', 'TRI6', 'QUAD4', 'QUAD8'

        Returns
        -------
        mesh : Mesh
            FEAX Mesh object

        Raises
        ------
        ValueError
            If element_type is not found in gmsh_mesh or is unsupported

        Examples
        --------
        Read Gmsh .msh file and convert:
        >>> import meshio
        >>> gmsh_mesh = meshio.read("mesh.msh")
        >>> mesh = Mesh.from_gmsh(gmsh_mesh)

        Convert meshio mesh with specific element type:
        >>> mesh = Mesh.from_gmsh(gmsh_mesh, element_type='HEX8')

        Notes
        -----
        - The method automatically maps meshio cell types to FEAX element types
        - Only volume elements (3D) and surface elements (2D) are supported
        - If multiple element types exist, specify element_type to select one
        """
        # Mapping from meshio cell types to FEAX element types
        meshio_to_feax = {
            'tetra': 'TET4',
            'tetra10': 'TET10',
            'hexahedron': 'HEX8',
            'hexahedron20': 'HEX20',
            'hexahedron27': 'HEX27',
            'triangle': 'TRI3',
            'triangle6': 'TRI6',
            'quad': 'QUAD4',
            'quad8': 'QUAD8',
        }

        # Get available cell types in the mesh
        available_types = {}
        for cell_block in gmsh_mesh.cells:
            cell_type = cell_block.type
            if cell_type in meshio_to_feax:
                feax_type = meshio_to_feax[cell_type]
                available_types[feax_type] = cell_block.data

        if not available_types:
            raise ValueError(
                f"No supported element types found in gmsh_mesh. "
                f"Available types: {[c.type for c in gmsh_mesh.cells]}"
            )

        # Determine element type to use
        if element_type is None:
            # Auto-select: prefer 3D elements, then 2D
            priority = ['HEX8', 'HEX20', 'HEX27', 'TET4', 'TET10', 'QUAD4', 'QUAD8', 'TRI3', 'TRI6']
            for pref_type in priority:
                if pref_type in available_types:
                    element_type = pref_type
                    break
            if element_type is None:
                element_type = list(available_types.keys())[0]

        if element_type not in available_types:
            raise ValueError(
                f"Element type '{element_type}' not found in gmsh_mesh. "
                f"Available types: {list(available_types.keys())}"
            )

        # Extract points and cells
        points = gmsh_mesh.points
        cells = available_types[element_type]

        return Mesh(points, cells, ele_type=element_type)

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



def box_mesh(
    size: Union[float, Tuple[float, float, float]],
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    mesh_size: float = 0.1,
    element_type: str = 'HEX8',
    recombine: bool = True
) -> Mesh:
    """Generate structured or unstructured mesh for box domain using Gmsh.

    Creates high-quality meshes using Gmsh, supporting both hexahedral (HEX8)
    and tetrahedral (TET4) elements with structured or unstructured meshing.

    Parameters
    ----------
    size : float or tuple of 3 floats
        If float: creates cube with side length = size
        If tuple: (length_x, length_y, length_z) for rectangular box
    origin : tuple of 3 floats, optional
        Origin point (x0, y0, z0) of the box. Default is (0, 0, 0)
    mesh_size : float, optional
        Target element size. Smaller values create finer meshes. Default is 0.1
    element_type : str, optional
        'HEX8' for hexahedral elements (default) or 'TET4' for tetrahedral
    recombine : bool, optional
        If True and element_type='HEX8', use structured recombination algorithm.
        Default is True for better quality hexahedral meshes.

    Returns
    -------
    mesh : Mesh
        Mesh with HEX8 or TET4 elements

    Raises
    ------
    ImportError
        If gmsh is not installed
    ValueError
        If element_type is not 'HEX8' or 'TET4'

    Examples
    --------
    Create structured HEX8 mesh:
    >>> mesh = box_mesh_gmsh(1.0, mesh_size=0.1, element_type='HEX8')

    Create unstructured TET4 mesh:
    >>> mesh = box_mesh_gmsh((2.0, 1.0, 0.5), mesh_size=0.05, element_type='TET4')

    Notes
    -----
    - Gmsh provides superior mesh quality compared to simple structured meshes
    - HEX8 meshes are preferred for most applications (better accuracy per DOF)
    - TET4 meshes are more flexible for complex geometries
    - For simple boxes with uniform elements, use box_mesh() for speed
    """
    if not GMSH_AVAILABLE:
        raise ImportError("gmsh is not installed. Install with: pip install gmsh")

    if element_type not in ['HEX8', 'TET4']:
        raise ValueError(f"element_type must be 'HEX8' or 'TET4', got {element_type}")

    # Parse size argument
    if isinstance(size, (int, float)):
        lx = ly = lz = float(size)
    else:
        lx, ly, lz = size

    x0, y0, z0 = origin

    # Initialize Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Suppress terminal output

    try:
        gmsh.model.add("box")

        # Create box geometry
        box_tag = gmsh.model.occ.addBox(x0, y0, z0, lx, ly, lz)
        gmsh.model.occ.synchronize()

        # Set mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

        if element_type == 'HEX8' and recombine:
            # Use transfinite algorithm for structured HEX mesh
            # Get all surfaces and curves
            surfaces = gmsh.model.getEntities(2)
            curves = gmsh.model.getEntities(1)

            # Calculate number of divisions based on mesh_size
            nx = max(2, int(lx / mesh_size) + 1)
            ny = max(2, int(ly / mesh_size) + 1)
            nz = max(2, int(lz / mesh_size) + 1)

            # Set transfinite curves - assign appropriate number of points based on curve direction
            for curve in curves:
                _, curve_tag = curve
                # Get curve endpoints to determine its direction
                bounds = gmsh.model.getBoundingBox(1, curve_tag)
                dx = abs(bounds[3] - bounds[0])
                dy = abs(bounds[4] - bounds[1])
                dz = abs(bounds[5] - bounds[2])

                # Determine curve direction and set appropriate number of points
                if dx > dy and dx > dz:  # X-direction curve
                    num_points = nx
                elif dy > dx and dy > dz:  # Y-direction curve
                    num_points = ny
                else:  # Z-direction curve
                    num_points = nz

                gmsh.model.mesh.setTransfiniteCurve(curve_tag, num_points)

            # Set transfinite surfaces
            for surface in surfaces:
                _, surface_tag = surface
                gmsh.model.mesh.setTransfiniteSurface(surface_tag)
                gmsh.model.mesh.setRecombine(2, surface_tag)

            # Set transfinite volume
            gmsh.model.mesh.setTransfiniteVolume(box_tag)
            gmsh.model.mesh.setRecombine(3, box_tag)

        elif element_type == 'HEX8' and not recombine:
            # Unstructured HEX mesh via recombination
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay for tets, then recombine

        # Generate 3D mesh
        gmsh.model.mesh.generate(3)

        # Get mesh data
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        points = node_coords.reshape(-1, 3)

        # Get elements
        if element_type == 'HEX8':
            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3, -1)
            # Find hexahedral elements (type 5 in Gmsh)
            hex_idx = None
            for i, etype in enumerate(elem_types):
                if etype == 5:  # Hexahedron
                    hex_idx = i
                    break
            if hex_idx is None:
                raise RuntimeError("No hexahedral elements found in mesh")
            cells = elem_node_tags[hex_idx].reshape(-1, 8) - 1  # Gmsh uses 1-based indexing
            ele_type_out = 'HEX8'
        else:  # TET4
            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3, -1)
            # Find tetrahedral elements (type 4 in Gmsh)
            tet_idx = None
            for i, etype in enumerate(elem_types):
                if etype == 4:  # Tetrahedron
                    tet_idx = i
                    break
            if tet_idx is None:
                raise RuntimeError("No tetrahedral elements found in mesh")
            cells = elem_node_tags[tet_idx].reshape(-1, 4) - 1
            ele_type_out = 'TET4'

        # Reindex nodes to be contiguous from 0
        unique_nodes = onp.unique(cells.flatten())
        node_map = onp.full(len(points), -1, dtype=onp.int32)
        node_map[unique_nodes] = onp.arange(len(unique_nodes))
        cells_reindexed = node_map[cells]
        points_filtered = points[unique_nodes]

        return Mesh(points_filtered, cells_reindexed, ele_type=ele_type_out)

    finally:
        gmsh.finalize()


def rectangle_mesh(
    Nx: int,
    Ny: int,
    domain_x: float = 1.0,
    domain_y: float = 1.0,
    origin: Tuple[float, float] = (0.0, 0.0)
) -> Mesh:
    """Generate structured 2D rectangular mesh with QUAD4 elements.

    Creates a simple structured quadrilateral mesh for rectangular domains.
    This is a lightweight alternative to Gmsh for simple 2D problems.

    Parameters
    ----------
    Nx : int
        Number of elements in x-direction
    Ny : int
        Number of elements in y-direction
    domain_x : float, optional
        Length of domain in x-direction. Default is 1.0
    domain_y : float, optional
        Length of domain in y-direction. Default is 1.0
    origin : tuple of 2 floats, optional
        Origin point (x0, y0) of the rectangle. Default is (0, 0)

    Returns
    -------
    mesh : Mesh
        Mesh with QUAD4 elements

    Examples
    --------
    Create 32x32 mesh on unit square:
    >>> mesh = rectangle_mesh(Nx=32, Ny=32, domain_x=1.0, domain_y=1.0)

    Notes
    -----
    - Generates (Nx+1) × (Ny+1) nodes
    - Generates Nx × Ny QUAD4 elements
    - Node ordering follows standard QUAD4 convention
    """
    x0, y0 = origin

    # Create structured grid of nodes
    x = onp.linspace(x0, x0 + domain_x, Nx + 1)
    y = onp.linspace(y0, y0 + domain_y, Ny + 1)
    xv, yv = onp.meshgrid(x, y, indexing='ij')

    # Create points array (num_nodes, 2) - embed in 2D space
    points = onp.stack([xv.reshape(-1), yv.reshape(-1)], axis=1)

    # Create connectivity for QUAD4 elements
    # Node numbering: 0--1
    #                 |  |
    #                 3--2
    cells = []
    for i in range(Nx):
        for j in range(Ny):
            # Node indices in the grid
            n0 = i * (Ny + 1) + j
            n1 = (i + 1) * (Ny + 1) + j
            n2 = (i + 1) * (Ny + 1) + (j + 1)
            n3 = i * (Ny + 1) + (j + 1)
            cells.append([n0, n1, n2, n3])

    cells = onp.array(cells, dtype=onp.int32)

    return Mesh(points, cells, ele_type='QUAD4')


def sphere_mesh(
    radius: float,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    mesh_size: float = 0.1,
    element_type: str = 'TET4'
) -> Mesh:
    """Generate mesh for sphere using Gmsh.

    Creates a tetrahedral or hexahedral mesh for a spherical domain.
    Note: Hexahedral meshing of spheres is challenging and may produce
    lower quality elements.

    Parameters
    ----------
    radius : float
        Radius of the sphere
    center : tuple of 3 floats, optional
        Center point (x, y, z) of the sphere. Default is (0, 0, 0)
    mesh_size : float, optional
        Target element size. Default is 0.1
    element_type : str, optional
        'TET4' for tetrahedral (default, recommended) or 'HEX8' for hexahedral

    Returns
    -------
    mesh : Mesh
        Mesh with TET4 or HEX8 elements

    Raises
    ------
    ImportError
        If gmsh is not installed

    Examples
    --------
    Create TET4 sphere mesh:
    >>> mesh = sphere_mesh_gmsh(1.0, mesh_size=0.1)

    Notes
    -----
    - TET4 is strongly recommended for spheres (better quality)
    - HEX8 meshes for spheres may have distorted elements
    """
    if not GMSH_AVAILABLE:
        raise ImportError("gmsh is not installed. Install with: pip install gmsh")

    if element_type not in ['HEX8', 'TET4']:
        raise ValueError(f"element_type must be 'HEX8' or 'TET4', got {element_type}")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.model.add("sphere")

        # Create sphere geometry
        x0, y0, z0 = center
        sphere_tag = gmsh.model.occ.addSphere(x0, y0, z0, radius)
        gmsh.model.occ.synchronize()

        # Set mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

        if element_type == 'HEX8':
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)

        # Generate mesh
        gmsh.model.mesh.generate(3)

        # Get mesh data
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        points = node_coords.reshape(-1, 3)

        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3, -1)

        if element_type == 'HEX8':
            hex_idx = None
            for i, etype in enumerate(elem_types):
                if etype == 5:
                    hex_idx = i
                    break
            if hex_idx is None:
                raise RuntimeError("No hexahedral elements found. Try element_type='TET4'")
            cells = elem_node_tags[hex_idx].reshape(-1, 8) - 1
            ele_type_out = 'HEX8'
        else:
            tet_idx = None
            for i, etype in enumerate(elem_types):
                if etype == 4:
                    tet_idx = i
                    break
            if tet_idx is None:
                raise RuntimeError("No tetrahedral elements found")
            cells = elem_node_tags[tet_idx].reshape(-1, 4) - 1
            ele_type_out = 'TET4'

        # Reindex nodes
        unique_nodes = onp.unique(cells.flatten())
        node_map = onp.full(len(points), -1, dtype=onp.int32)
        node_map[unique_nodes] = onp.arange(len(unique_nodes))
        cells_reindexed = node_map[cells]
        points_filtered = points[unique_nodes]

        return Mesh(points_filtered, cells_reindexed, ele_type=ele_type_out)

    finally:
        gmsh.finalize()


def cylinder_mesh(
    radius: float,
    height: float,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    mesh_size: float = 0.1,
    element_type: str = 'TET4'
) -> Mesh:
    """Generate mesh for cylinder using Gmsh.

    Creates a tetrahedral or hexahedral mesh for a cylindrical domain.

    Parameters
    ----------
    radius : float
        Radius of the cylinder
    height : float
        Height of the cylinder along the axis
    center : tuple of 3 floats, optional
        Center point (x, y, z) of the cylinder base. Default is (0, 0, 0)
    axis : tuple of 3 floats, optional
        Direction vector of cylinder axis. Default is (0, 0, 1) (z-axis)
    mesh_size : float, optional
        Target element size. Default is 0.1
    element_type : str, optional
        'TET4' for tetrahedral (default) or 'HEX8' for hexahedral

    Returns
    -------
    mesh : Mesh
        Mesh with TET4 or HEX8 elements

    Raises
    ------
    ImportError
        If gmsh is not installed

    Examples
    --------
    Create TET4 cylinder mesh:
    >>> mesh = cylinder_mesh_gmsh(0.5, 2.0, mesh_size=0.1)

    Create HEX8 cylinder mesh:
    >>> mesh = cylinder_mesh_gmsh(0.5, 2.0, mesh_size=0.1, element_type='HEX8')
    """
    if not GMSH_AVAILABLE:
        raise ImportError("gmsh is not installed. Install with: pip install gmsh")

    if element_type not in ['HEX8', 'TET4']:
        raise ValueError(f"element_type must be 'HEX8' or 'TET4', got {element_type}")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.model.add("cylinder")

        # Normalize axis
        axis_array = onp.array(axis)
        axis_norm = axis_array / onp.linalg.norm(axis_array)

        # For simplicity, create cylinder along z-axis then rotate if needed
        # Gmsh's addCylinder works best with z-aligned cylinders
        x0, y0, z0 = center

        # Create cylinder along z-axis
        cylinder_tag = gmsh.model.occ.addCylinder(x0, y0, z0, 0, 0, height, radius)

        # If axis is not z-aligned, rotate the cylinder
        if not onp.allclose(axis_norm, [0, 0, 1]):
            # Calculate rotation to align z-axis with desired axis
            z_axis = onp.array([0, 0, 1])
            rotation_axis = onp.cross(z_axis, axis_norm)
            if onp.linalg.norm(rotation_axis) > 1e-10:
                rotation_axis = rotation_axis / onp.linalg.norm(rotation_axis)
                angle = onp.arccos(onp.dot(z_axis, axis_norm))
                gmsh.model.occ.rotate([(3, cylinder_tag)], x0, y0, z0,
                                     rotation_axis[0], rotation_axis[1], rotation_axis[2], angle)

        gmsh.model.occ.synchronize()

        # Set mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

        if element_type == 'HEX8':
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)

        # Generate mesh
        gmsh.model.mesh.generate(3)

        # Get mesh data
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        points = node_coords.reshape(-1, 3)

        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3, -1)

        if element_type == 'HEX8':
            hex_idx = None
            for i, etype in enumerate(elem_types):
                if etype == 5:
                    hex_idx = i
                    break
            if hex_idx is None:
                raise RuntimeError("No hexahedral elements found. Try element_type='TET4'")
            cells = elem_node_tags[hex_idx].reshape(-1, 8) - 1
            ele_type_out = 'HEX8'
        else:
            tet_idx = None
            for i, etype in enumerate(elem_types):
                if etype == 4:
                    tet_idx = i
                    break
            if tet_idx is None:
                raise RuntimeError("No tetrahedral elements found")
            cells = elem_node_tags[tet_idx].reshape(-1, 4) - 1
            ele_type_out = 'TET4'

        # Reindex nodes
        unique_nodes = onp.unique(cells.flatten())
        node_map = onp.full(len(points), -1, dtype=onp.int32)
        node_map[unique_nodes] = onp.arange(len(unique_nodes))
        cells_reindexed = node_map[cells]
        points_filtered = points[unique_nodes]

        return Mesh(points_filtered, cells_reindexed, ele_type=ele_type_out)

    finally:
        gmsh.finalize()