"""
Mesh management and generation utilities for FEAX finite element framework.

This module provides the Mesh class for managing finite element meshes and utility
functions for mesh generation, validation, and format conversion.
"""

from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as np
import numpy as onp
from jax import Array

from feax.basis import get_face_shape_vals_and_grads


class Mesh():
    """Finite element mesh manager.
    
    This class manages mesh data including node coordinates, element connectivity,
    and element type information. It provides methods for querying mesh properties
    and analyzing boundary conditions.

    Parameters
    ----------
    points : Array
        Node coordinates with shape (num_total_nodes, dim)
    cells : Array
        Element connectivity with shape (num_cells, num_nodes_per_element)
    ele_type : str, optional
        Element type identifier (default: 'TET4')

    Attributes
    ----------
    points : Array
        Node coordinates with shape (num_total_nodes, dim)
    cells : Array
        Element connectivity with shape (num_cells, num_nodes_per_element)
    ele_type : str
        Element type identifier

    Notes
    -----
    The element connectivity array should follow the standard node ordering
    conventions for each element type.
    """
    def __init__(self, points: Array, cells: Array, ele_type: str = 'TET4') -> None:
        # TODO (Very important for debugging purpose!): Assert that cells must have correct orders
        self.points = points
        self.cells = cells
        self.ele_type = ele_type

    @staticmethod
    def from_gmsh(gmsh_mesh: 'object', element_type: Optional[str] = None) -> 'Mesh':
        """Convert a meshio-style mesh (e.g. from Gmsh) to a FEAX Mesh.

        Accepts any object exposing meshio's ``.points`` array and ``.cells``
        list of cell blocks (each with ``.type`` and ``.data``) — for example a
        ``meshio.Mesh`` returned by ``meshio.read(...)``. ``meshio`` is not a
        FEAX dependency; install it separately if you need to read mesh files.

        Parameters
        ----------
        gmsh_mesh : meshio.Mesh (or compatible)
            Mesh object containing points and cells
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
            'quad9': 'QUAD9',
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


def check_mesh_TET4(points: Array, cells: Array) -> np.ndarray:
    """Check the node ordering of TET4 elements by computing signed volumes.

    This function computes the signed volume of each tetrahedral element to verify
    proper node ordering. Negative volumes indicate inverted elements.

    Parameters
    ----------
    points : Array
        Node coordinates with shape (num_nodes, 3)
    cells : Array
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
    elif ele_type == 'QUAD9':
        cell_type = 'quad9'
    else:
        raise NotImplementedError
    return cell_type



# Local hexahedron corners (meshio/VTK ordering) split into 6 tetrahedra that
# all share the 0-6 body diagonal (Freudenthal / Kuhn decomposition). Each row
# is a right-handed (positive-volume) tet given the corner layout produced by
# _structured_hex_grid.
_HEX_TO_TET6 = onp.array([
    [0, 1, 2, 6],
    [0, 2, 3, 6],
    [0, 3, 7, 6],
    [0, 7, 4, 6],
    [0, 4, 5, 6],
    [0, 5, 1, 6],
], dtype=onp.int32)


def _tetrahedralize_surface(surface, mesh_size: float) -> Tuple[onp.ndarray, onp.ndarray]:
    """Fill a closed triangulated surface with TET4 elements via TetGen.

    Parameters
    ----------
    surface : pyvista.PolyData
        Closed, watertight triangle surface mesh.
    mesh_size : float
        Target edge length; controls the TetGen maximum tetrahedron volume.

    Returns
    -------
    points : np.ndarray, shape (n_nodes, 3)
    cells : np.ndarray, shape (n_cells, 4), int32
    """
    try:
        import tetgen
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "sphere_mesh/cylinder_mesh require the 'tetgen' package. "
            "Install it with `pip install tetgen`."
        ) from exc

    tet = tetgen.TetGen(surface.triangulate())
    # Roughly one tet per (mesh_size)^3 of volume; the /6 keeps element edge
    # lengths close to mesh_size.
    max_volume = (mesh_size ** 3) / 6.0
    tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5, maxvolume=max_volume)
    grid = tet.grid
    points = onp.asarray(grid.points, dtype=onp.float64)
    # VTK_TETRA == 10
    cells = onp.asarray(grid.cells_dict[10], dtype=onp.int32)
    return points, cells


def _structured_hex_grid(
    origin: Tuple[float, float, float],
    lengths: Tuple[float, float, float],
    n_pts: Tuple[int, int, int],
) -> Tuple[onp.ndarray, onp.ndarray]:
    """Build a structured HEX8 grid (points + connectivity) with pure numpy.

    Node ``(i, j, k)`` has global id ``(i * nyp + j) * nzp + k`` (k fastest),
    and each cell's 8 corners are emitted in meshio/VTK hexahedron order.

    Returns
    -------
    points : np.ndarray, shape (nxp*nyp*nzp, 3)
    hexes : np.ndarray, shape (num_cells, 8), int32
    """
    x0, y0, z0 = origin
    lx, ly, lz = lengths
    nxp, nyp, nzp = n_pts

    xp = onp.linspace(x0, x0 + lx, nxp)
    yp = onp.linspace(y0, y0 + ly, nyp)
    zp = onp.linspace(z0, z0 + lz, nzp)
    X, Y, Z = onp.meshgrid(xp, yp, zp, indexing='ij')
    points = onp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # Element base-corner indices.
    ei, ej, ek = onp.meshgrid(
        onp.arange(nxp - 1), onp.arange(nyp - 1), onp.arange(nzp - 1),
        indexing='ij',
    )
    ei, ej, ek = ei.ravel(), ej.ravel(), ek.ravel()

    def nid(di, dj, dk):
        return ((ei + di) * nyp + (ej + dj)) * nzp + (ek + dk)

    hexes = onp.stack([
        nid(0, 0, 0), nid(1, 0, 0), nid(1, 1, 0), nid(0, 1, 0),
        nid(0, 0, 1), nid(1, 0, 1), nid(1, 1, 1), nid(0, 1, 1),
    ], axis=1).astype(onp.int32)

    return points, hexes


# HEX20 (serendipity) node offsets on a half-element fine grid (spacing = 1),
# in meshio/VTK ``hexahedron20`` order: 8 corners, then 12 edge midpoints
# (bottom edges 0-1,1-2,2-3,3-0; top edges 4-5,5-6,6-7,7-4; verticals
# 0-4,1-5,2-6,3-7). No face/body centres — that is what makes it serendipity.
_HEX20_OFFSETS = onp.array([
    # corners
    (0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0),
    (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2),
    # bottom edges
    (1, 0, 0), (2, 1, 0), (1, 2, 0), (0, 1, 0),
    # top edges
    (1, 0, 2), (2, 1, 2), (1, 2, 2), (0, 1, 2),
    # vertical edges
    (0, 0, 1), (2, 0, 1), (2, 2, 1), (0, 2, 1),
], dtype=onp.int64)


def _structured_hex20_grid(
    origin: Tuple[float, float, float],
    lengths: Tuple[float, float, float],
    n_elem: Tuple[int, int, int],
) -> Tuple[onp.ndarray, onp.ndarray]:
    """Build a structured HEX20 (serendipity) grid with pure numpy.

    Uses a half-element fine grid ``(2*nex+1, 2*ney+1, 2*nez+1)`` and keeps
    only nodes with at most one odd fine index (corners + edge midpoints),
    dropping face centres and body centres. Cells are emitted in meshio/VTK
    ``hexahedron20`` order.

    Returns
    -------
    points : np.ndarray, shape (n_nodes, 3)
    cells : np.ndarray, shape (num_cells, 20), int32
    """
    x0, y0, z0 = origin
    lx, ly, lz = lengths
    nex, ney, nez = n_elem
    nfx, nfy, nfz = 2 * nex + 1, 2 * ney + 1, 2 * nez + 1

    xf = onp.linspace(x0, x0 + lx, nfx)
    yf = onp.linspace(y0, y0 + ly, nfy)
    zf = onp.linspace(z0, z0 + lz, nfz)
    Xf, Yf, Zf = onp.meshgrid(xf, yf, zf, indexing='ij')
    all_points = onp.stack([Xf.ravel(), Yf.ravel(), Zf.ravel()], axis=1)

    # Keep corners (0 odd) and edge midpoints (1 odd); drop face/body centres.
    I, J, K = onp.meshgrid(
        onp.arange(nfx), onp.arange(nfy), onp.arange(nfz), indexing='ij',
    )
    n_odd = (I % 2) + (J % 2) + (K % 2)
    used = (n_odd <= 1).ravel()
    old_to_new = onp.full(nfx * nfy * nfz, -1, dtype=onp.int64)
    old_to_new[used] = onp.arange(int(used.sum()))
    points = all_points[used]

    ei, ej, ek = onp.meshgrid(
        onp.arange(nex), onp.arange(ney), onp.arange(nez), indexing='ij',
    )
    ei, ej, ek = ei.ravel(), ej.ravel(), ek.ravel()
    base = onp.stack([2 * ei, 2 * ej, 2 * ek], axis=1)  # (n_cells, 3)

    # Fine-grid ids for all 20 local nodes: (n_cells, 20)
    fine = base[:, None, :] + _HEX20_OFFSETS[None, :, :]
    gids = (fine[:, :, 0] * nfy + fine[:, :, 1]) * nfz + fine[:, :, 2]
    cells = old_to_new[gids].astype(onp.int32)

    return points, cells


def box_mesh(
    size: Union[float, Tuple[float, float, float]],
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    mesh_size: float = 0.1,
    element_type: str = 'HEX8',
    recombine: bool = True
) -> Mesh:
    """Generate a structured mesh for a box domain (pure numpy, no Gmsh).

    Creates a structured tensor-product grid of hexahedral (HEX8) elements, or
    a conforming tetrahedral (TET4) mesh obtained by splitting each hexahedron
    into 6 tetrahedra. No external mesh generator is required.

    Parameters
    ----------
    size : float or tuple of 3 floats
        If float: creates cube with side length = size
        If tuple: (length_x, length_y, length_z) for rectangular box
    origin : tuple of 3 floats, optional
        Origin point (x0, y0, z0) of the box. Default is (0, 0, 0)
    mesh_size : float, optional
        Target element size. Smaller values create finer meshes. Default is 0.1.
        The number of elements along an edge of length L is ``int(L / mesh_size)``
        (at least one).
    element_type : str, optional
        'HEX8' (default) or 'HEX20' for hexahedral elements, or 'TET4' for
        tetrahedral.
    recombine : bool, optional
        Deprecated / no-op. Kept for backward compatibility; the box mesh is
        always a structured grid now.

    Returns
    -------
    mesh : Mesh
        Mesh with HEX8 or TET4 elements

    Raises
    ------
    ValueError
        If element_type is not 'HEX8' or 'TET4'

    Examples
    --------
    Create structured HEX8 mesh:
    >>> mesh = box_mesh(1.0, mesh_size=0.1, element_type='HEX8')

    Create TET4 mesh:
    >>> mesh = box_mesh((2.0, 1.0, 0.5), mesh_size=0.05, element_type='TET4')

    Notes
    -----
    - HEX8 nodes/cells follow the meshio/VTK hexahedron ordering.
    - TET4 uses a Freudenthal (6-tet) split of each hexahedron; the result is
      conforming because every cell of the structured grid is a pure translate.
    """
    del recombine  # no-op, kept for backward-compatible signature

    if element_type not in ['HEX8', 'HEX20', 'TET4']:
        raise ValueError(
            f"element_type must be 'HEX8', 'HEX20' or 'TET4', got {element_type}"
        )

    # Parse size argument
    if isinstance(size, (int, float)):
        lx = ly = lz = float(size)
    else:
        lx, ly, lz = size

    x0, y0, z0 = origin

    # Number of nodes per direction. Matches the previous Gmsh transfinite
    # behaviour: n_pts = max(2, int(L / mesh_size) + 1) => int(L/mesh_size)
    # elements along each edge (at least one).
    nxp = max(2, int(lx / mesh_size) + 1)
    nyp = max(2, int(ly / mesh_size) + 1)
    nzp = max(2, int(lz / mesh_size) + 1)

    if element_type == 'HEX20':
        # Serendipity quadratic box: nex = nxp - 1 elements per edge.
        points, cells = _structured_hex20_grid(
            (x0, y0, z0), (lx, ly, lz), (nxp - 1, nyp - 1, nzp - 1)
        )
        return Mesh(points, cells, ele_type='HEX20')

    points, hexes = _structured_hex_grid(
        (x0, y0, z0), (lx, ly, lz), (nxp, nyp, nzp)
    )

    if element_type == 'HEX8':
        return Mesh(points, hexes, ele_type='HEX8')

    # TET4: split every hexahedron into 6 tetrahedra sharing the 0-6 body
    # diagonal. On a structured grid every cell is a pure translate, so the
    # induced face diagonals match across shared faces and the tet mesh is
    # conforming.
    cells = hexes[:, _HEX_TO_TET6].reshape(-1, 4)
    return Mesh(points, cells, ele_type='TET4')


def rectangle_mesh(
    Nx: int,
    Ny: int,
    domain_x: float = 1.0,
    domain_y: float = 1.0,
    origin: Tuple[float, float] = (0.0, 0.0),
    ele_type: str = 'QUAD4',
) -> Mesh:
    """Generate structured 2D rectangular mesh with QUAD4 or QUAD8 elements.

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
    ele_type : str, optional
        Element type: 'QUAD4' (bilinear) or 'QUAD8' (serendipity quadratic).
        Default is 'QUAD4'.

    Returns
    -------
    mesh : Mesh
        Mesh with the specified element type.
    """
    if ele_type not in ('QUAD4', 'QUAD8', 'QUAD9'):
        raise ValueError(f"rectangle_mesh supports 'QUAD4', 'QUAD8', or 'QUAD9', got '{ele_type}'")

    x0, y0 = origin

    if ele_type == 'QUAD4':
        # Create structured grid of nodes
        x = onp.linspace(x0, x0 + domain_x, Nx + 1)
        y = onp.linspace(y0, y0 + domain_y, Ny + 1)
        xv, yv = onp.meshgrid(x, y, indexing='ij')
        points = onp.stack([xv.reshape(-1), yv.reshape(-1)], axis=1)

        cells = []
        for i in range(Nx):
            for j in range(Ny):
                n0 = i * (Ny + 1) + j
                n1 = (i + 1) * (Ny + 1) + j
                n2 = (i + 1) * (Ny + 1) + (j + 1)
                n3 = i * (Ny + 1) + (j + 1)
                cells.append([n0, n1, n2, n3])

        cells = onp.array(cells, dtype=onp.int32)
        return Mesh(points, cells, ele_type='QUAD4')

    if ele_type == 'QUAD9':
        return _rectangle_mesh_quad9(Nx, Ny, domain_x, domain_y, x0, y0)

    # ── QUAD8: serendipity (8-node) quadrilateral ──
    # Fine grid: (2*Nx+1) x (2*Ny+1) nodes, then select the 8-node pattern.
    # Corner nodes live on even-even grid indices; midside nodes on mixed.
    nx_fine = 2 * Nx + 1
    ny_fine = 2 * Ny + 1
    x_fine = onp.linspace(x0, x0 + domain_x, nx_fine)
    y_fine = onp.linspace(y0, y0 + domain_y, ny_fine)
    xv, yv = onp.meshgrid(x_fine, y_fine, indexing='ij')
    all_points = onp.stack([xv.reshape(-1), yv.reshape(-1)], axis=1)

    def gid(i, j):
        return i * ny_fine + j

    # Identify which fine-grid nodes are actually used (corners + midsides)
    used = onp.zeros(nx_fine * ny_fine, dtype=bool)
    cells = []
    for i in range(Nx):
        for j in range(Ny):
            ci, cj = 2 * i, 2 * j  # fine-grid origin of this element
            # Corners: (ci,cj), (ci+2,cj), (ci+2,cj+2), (ci,cj+2)
            # Midsides: (ci+1,cj), (ci+2,cj+1), (ci+1,cj+2), (ci,cj+1)
            # Meshio/Abaqus QUAD8 ordering: 4 corners then 4 midsides
            nodes = [
                gid(ci, cj), gid(ci + 2, cj),
                gid(ci + 2, cj + 2), gid(ci, cj + 2),
                gid(ci + 1, cj), gid(ci + 2, cj + 1),
                gid(ci + 1, cj + 2), gid(ci, cj + 1),
            ]
            cells.append(nodes)
            for n in nodes:
                used[n] = True

    # Compact: renumber used nodes
    old_to_new = onp.full(nx_fine * ny_fine, -1, dtype=onp.int32)
    old_to_new[used] = onp.arange(used.sum(), dtype=onp.int32)
    points = all_points[used]
    cells = old_to_new[onp.array(cells, dtype=onp.int32)]

    return Mesh(points, cells, ele_type='QUAD8')


def _rectangle_mesh_quad9(
    Nx: int, Ny: int, domain_x: float, domain_y: float,
    x0: float, y0: float,
) -> Mesh:
    """Generate structured 2D rectangular mesh with QUAD9 (bi-quadratic Lagrange) elements."""
    nx_fine = 2 * Nx + 1
    ny_fine = 2 * Ny + 1
    x_fine = onp.linspace(x0, x0 + domain_x, nx_fine)
    y_fine = onp.linspace(y0, y0 + domain_y, ny_fine)
    xv, yv = onp.meshgrid(x_fine, y_fine, indexing='ij')
    all_points = onp.stack([xv.reshape(-1), yv.reshape(-1)], axis=1)

    def gid(i, j):
        return i * ny_fine + j

    used = onp.zeros(nx_fine * ny_fine, dtype=bool)
    cells = []
    for i in range(Nx):
        for j in range(Ny):
            ci, cj = 2 * i, 2 * j
            # Meshio/Abaqus QUAD9: 4 corners + 4 midsides + 1 center
            nodes = [
                gid(ci, cj), gid(ci + 2, cj),
                gid(ci + 2, cj + 2), gid(ci, cj + 2),
                gid(ci + 1, cj), gid(ci + 2, cj + 1),
                gid(ci + 1, cj + 2), gid(ci, cj + 1),
                gid(ci + 1, cj + 1),
            ]
            cells.append(nodes)
            for n in nodes:
                used[n] = True

    old_to_new = onp.full(nx_fine * ny_fine, -1, dtype=onp.int32)
    old_to_new[used] = onp.arange(used.sum(), dtype=onp.int32)
    points = all_points[used]
    cells = old_to_new[onp.array(cells, dtype=onp.int32)]

    return Mesh(points, cells, ele_type='QUAD9')


def sphere_mesh(
    radius: float,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    mesh_size: float = 0.1,
    element_type: str = 'TET4'
) -> Mesh:
    """Generate a tetrahedral mesh for a sphere (PyVista surface + TetGen).

    A triangulated sphere surface is generated with PyVista and filled with
    TET4 elements by TetGen. No Gmsh dependency.

    Parameters
    ----------
    radius : float
        Radius of the sphere
    center : tuple of 3 floats, optional
        Center point (x, y, z) of the sphere. Default is (0, 0, 0)
    mesh_size : float, optional
        Target element size. Default is 0.1
    element_type : str, optional
        Only 'TET4' is supported.

    Returns
    -------
    mesh : Mesh
        Mesh with TET4 elements

    Raises
    ------
    ImportError
        If the 'tetgen' package is not installed
    ValueError
        If element_type is not 'TET4'

    Examples
    --------
    Create TET4 sphere mesh:
    >>> mesh = sphere_mesh(1.0, mesh_size=0.1)
    """

    if element_type != 'TET4':
        raise ValueError(
            f"sphere_mesh only supports element_type='TET4', got {element_type}. "
            f"Hexahedral meshing of curved geometry is no longer supported."
        )

    import pyvista as pv

    # Surface resolution from the target edge length along a great circle.
    res = max(8, int(2.0 * onp.pi * radius / mesh_size))
    surface = pv.Sphere(
        radius=radius, center=center,
        theta_resolution=res, phi_resolution=res,
    )
    points, cells = _tetrahedralize_surface(surface, mesh_size)
    return Mesh(points, cells, ele_type='TET4')


def cylinder_mesh(
    radius: float,
    height: float,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    mesh_size: float = 0.1,
    element_type: str = 'TET4'
) -> Mesh:
    """Generate a tetrahedral mesh for a cylinder (PyVista surface + TetGen).

    A closed cylinder surface is generated with PyVista and filled with TET4
    elements by TetGen. No Gmsh dependency.

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
        Only 'TET4' is supported.

    Returns
    -------
    mesh : Mesh
        Mesh with TET4 elements

    Raises
    ------
    ImportError
        If the 'tetgen' package is not installed
    ValueError
        If element_type is not 'TET4'

    Examples
    --------
    Create TET4 cylinder mesh:
    >>> mesh = cylinder_mesh(0.5, 2.0, mesh_size=0.1)
    """

    if element_type != 'TET4':
        raise ValueError(
            f"cylinder_mesh only supports element_type='TET4', got {element_type}. "
            f"Hexahedral meshing of curved geometry is no longer supported."
        )

    import pyvista as pv

    axis_norm = onp.asarray(axis, dtype=onp.float64)
    axis_norm = axis_norm / onp.linalg.norm(axis_norm)

    # PyVista's Cylinder is centred on its midpoint; the FEAX convention treats
    # `center` as the base, so shift by half the height along the axis.
    base = onp.asarray(center, dtype=onp.float64)
    pv_center = base + 0.5 * height * axis_norm

    res = max(8, int(2.0 * onp.pi * radius / mesh_size))
    surface = pv.Cylinder(
        center=pv_center, direction=axis_norm,
        radius=radius, height=height, resolution=res, capping=True,
    )
    points, cells = _tetrahedralize_surface(surface, mesh_size)
    return Mesh(points, cells, ele_type='TET4')


# ── Generic pure-numpy uniform mesh refinement ─────────────────────────────

# Per-element node count and topological dimension.
_ELE_INFO = {
    'LINE2':  (2, 1),
    'TRI3':   (3, 2),
    'QUAD4':  (4, 2),
    'TET4':   (4, 3),
    'HEX8':   (8, 3),
    'PRISM6': (6, 3),
    'TRI6':   (6, 2),
    'QUAD9':  (9, 2),
    'TET10':  (10, 3),
    'HEX27':  (27, 3),
    'QUAD8':  (8, 2),
    'HEX20':  (20, 3),
}


# Uniform-subdivision templates. Each template is a list of child cells; each
# child cell is a list of *node descriptors*. A descriptor is a tuple of local
# corner indices of the parent element: the new node sits at the average of
# those corners, and its identity (for sharing across parent elements) is the
# sorted set of the corresponding global corner ids. Thus a 1-tuple is a corner
# (shared trivially), a 2-tuple an edge midpoint (shared by all elements on that
# edge), a 4-tuple a quad-face centre (shared by the two elements on that face),
# and the full-corner tuple a cell centre (never shared).

_TRI3_TEMPLATE = [
    [(0,),   (0, 1), (2, 0)],
    [(0, 1), (1,),   (1, 2)],
    [(2, 0), (1, 2), (2,)],
    [(0, 1), (1, 2), (2, 0)],
]

_QUAD4_TEMPLATE = [
    [(0,),          (0, 1),         (0, 1, 2, 3), (3, 0)],
    [(0, 1),        (1,),           (1, 2),       (0, 1, 2, 3)],
    [(0, 1, 2, 3),  (1, 2),         (2,),         (2, 3)],
    [(3, 0),        (0, 1, 2, 3),   (2, 3),       (3,)],
]

# TET4 → 8 sub-tets (Bey subdivision). Inner octahedron split along the
# (m02, m13) diagonal. Orientation is fixed up numerically in _refine_once.
_TET4_TEMPLATE = [
    [(0,),   (0, 1), (0, 2), (0, 3)],
    [(0, 1), (1,),   (1, 2), (1, 3)],
    [(0, 2), (1, 2), (2,),   (2, 3)],
    [(0, 3), (1, 3), (2, 3), (3,)],
    [(0, 2), (1, 3), (0, 1), (0, 3)],
    [(0, 2), (1, 3), (0, 3), (2, 3)],
    [(0, 2), (1, 3), (2, 3), (1, 2)],
    [(0, 2), (1, 3), (1, 2), (0, 1)],
]

# HEX8 corner <-> unit-cube coordinate maps (meshio/VTK ordering).
_HEX_ID_TO_COORD = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
]
_HEX_COORD_TO_ID = {c: i for i, c in enumerate(_HEX_ID_TO_COORD)}


def _build_hex8_template():
    """8 sub-hexes of a HEX8 via trilinear (2x2x2) subdivision."""

    def descriptor(a, b, c):
        # Fine-grid coord (a,b,c) in {0,1,2} -> parent corners it averages.
        xs = [0] if a == 0 else ([1] if a == 2 else [0, 1])
        ys = [0] if b == 0 else ([1] if b == 2 else [0, 1])
        zs = [0] if c == 0 else ([1] if c == 2 else [0, 1])
        return tuple(_HEX_COORD_TO_ID[(x, y, z)] for x in xs for y in ys for z in zs)

    template = []
    for ok in (0, 1):
        for oj in (0, 1):
            for oi in (0, 1):
                sub = []
                for (ux, uy, uz) in _HEX_ID_TO_COORD:  # meshio corner order
                    sub.append(descriptor(oi + ux, oj + uy, ok + uz))
                template.append(sub)
    return template


_HEX8_TEMPLATE = _build_hex8_template()

_REFINE_TEMPLATES = {
    'TRI3':  _TRI3_TEMPLATE,
    'QUAD4': _QUAD4_TEMPLATE,
    'TET4':  _TET4_TEMPLATE,
    'HEX8':  _HEX8_TEMPLATE,
}


def _refine_once(points: onp.ndarray, cells: onp.ndarray, ele_type: str):
    """Apply one level of uniform subdivision (pure numpy)."""
    template = _REFINE_TEMPLATES[ele_type]

    # Unique descriptors across the template drive node creation.
    node_index = {}          # key -> new global node id
    new_points = []

    def get_node(cell_corners, desc):
        gids = cell_corners[list(desc)]
        key = tuple(sorted(int(g) for g in gids))
        idx = node_index.get(key)
        if idx is None:
            idx = len(new_points)
            node_index[key] = idx
            new_points.append(points[list(gids)].mean(axis=0))
        return idx

    new_cells = []
    for cell in cells:
        for child in template:
            new_cells.append([get_node(cell, desc) for desc in child])

    new_points = onp.asarray(new_points, dtype=points.dtype)
    new_cells = onp.asarray(new_cells, dtype=onp.int32)

    # Ensure positive-volume TET4 (interior diagonal choice may invert some).
    if ele_type == 'TET4':
        p = new_points[new_cells]
        vol = onp.einsum(
            'ij,ij->i',
            onp.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
            p[:, 3] - p[:, 0],
        )
        flip = vol < 0
        new_cells[flip] = new_cells[flip][:, [0, 1, 3, 2]]

    return new_points, new_cells


def refine_mesh(mesh: Mesh, levels: int = 1) -> Mesh:
    """Uniformly refine a mesh by edge subdivision (pure numpy, no Gmsh).

    Each refinement level subdivides every cell into smaller cells of
    the *same type*:

    * ``QUAD4`` → 4 sub-quads        (4 edge midpoints + cell centre)
    * ``TRI3``  → 4 sub-triangles    (3 edge midpoints)
    * ``HEX8``  → 8 sub-hexes        (trilinear 2x2x2 subdivision)
    * ``TET4``  → 8 sub-tets         (Bey subdivision)

    Edge midpoints (and quad-face centres) are shared between adjacent
    cells, so the result is conforming. Node counts grow as ``≈ 4ⁿ``
    (2-D) / ``≈ 8ⁿ`` (3-D) per level — call multiple times only if you
    really need it (memory grows quickly).

    Works on *any* mesh of a supported type — structured or unstructured,
    generated by :func:`box_mesh`, :func:`rectangle_mesh`,
    :func:`Mesh.from_gmsh`, or read from an external file.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.  ``mesh.points`` must be ``(n_nodes, 2)`` or
        ``(n_nodes, 3)``; ``mesh.cells`` must be ``(n_cells,
        n_nodes_per_cell)`` matching ``mesh.ele_type``.
    levels : int, default 1
        Number of refinement levels.  ``0`` returns ``mesh`` unchanged.

    Returns
    -------
    Mesh
        Refined mesh with the same ``ele_type`` as the input.

    Examples
    --------
    >>> coarse = rectangle_mesh(Nx=10, Ny=10, ele_type='QUAD4')
    >>> fine = refine_mesh(coarse, levels=2)   # 16x more cells
    >>> hex_coarse = box_mesh(1.0, mesh_size=0.2, element_type='HEX8')
    >>> hex_fine = refine_mesh(hex_coarse, levels=1)   # 8x more cells

    Notes
    -----
    * Only the linear element types ``TRI3``, ``QUAD4``, ``TET4`` and
      ``HEX8`` are supported. Refine the linear mesh first, then promote
      to a higher order if needed.
    """
    if levels < 0:
        raise ValueError(f"levels must be >= 0, got {levels}")
    if levels == 0:
        return mesh

    ele_type = mesh.ele_type
    if ele_type not in _REFINE_TEMPLATES:
        raise NotImplementedError(
            f"refine_mesh: unsupported ele_type '{ele_type}'. "
            f"Supported: {sorted(_REFINE_TEMPLATES)}"
        )

    n_nodes_per_cell, _ = _ELE_INFO[ele_type]

    pts = onp.asarray(mesh.points, dtype=onp.float64)
    cells = onp.asarray(mesh.cells, dtype=onp.int64)

    if pts.ndim != 2:
        raise ValueError(f"mesh.points must be 2-D, got shape {pts.shape}")
    if pts.shape[1] not in (2, 3):
        raise ValueError(
            f"mesh.points dim must be 2 or 3, got {pts.shape[1]}"
        )
    if cells.shape[1] != n_nodes_per_cell:
        raise ValueError(
            f"mesh.cells has {cells.shape[1]} nodes/cell but {ele_type} "
            f"expects {n_nodes_per_cell}"
        )

    for _ in range(levels):
        pts, cells = _refine_once(pts, cells, ele_type)

    return Mesh(pts, cells, ele_type=ele_type)
