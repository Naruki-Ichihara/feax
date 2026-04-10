"""Surface and volume mesh extraction from topology optimization results.

Extracts smooth iso-surfaces from node-based density fields using VTK's
contour filter on the unstructured FE mesh. Unlike grid-based marching
cubes, this interpolates along element edges and produces smooth surfaces
that respect the FE shape function continuity.

Requires ``pyvista`` (``pip install pyvista``).

Example
-------
>>> result = gene.optimizer.run(...)
>>> surface = extract_surface(result.rho_filtered, result.mesh, threshold=0.5)
>>> surface.save("optimized.stl")
>>>
>>> # With Gmsh volume remeshing
>>> fe_mesh = extract_volume_mesh(result.rho_filtered, result.mesh, mesh_size=0.05)
"""

from __future__ import annotations

import sys
from typing import Optional, Tuple

import numpy as onp

# Suppress PyVista's PolyData.__del__ cleanup warning (harmless GC race)
_original_unraisablehook = sys.unraisablehook


def _quiet_unraisablehook(unraisable):
    if unraisable.object and 'PolyData' in type(unraisable.object).__name__:
        return
    _original_unraisablehook(unraisable)


sys.unraisablehook = _quiet_unraisablehook


def _to_pyvista(mesh, density: onp.ndarray):
    """Convert a feax Mesh + node density to a PyVista UnstructuredGrid."""
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError(
            "mesh_extraction requires pyvista. "
            "Install with: pip install pyvista"
        )

    points = onp.asarray(mesh.points, dtype=onp.float64)
    cells = onp.asarray(mesh.cells)
    ele_type = mesh.ele_type

    # Map feax element types to VTK cell types
    _vtk_cell_types = {
        'TET4': 10, 'TET10': 24,
        'HEX8': 12, 'HEX20': 25, 'HEX27': 29,
        'TRI3': 5, 'TRI6': 22,
        'QUAD4': 9, 'QUAD8': 23, 'QUAD9': 28,
    }
    if ele_type not in _vtk_cell_types:
        raise ValueError(f"Unsupported element type: {ele_type}")

    vtk_type = _vtk_cell_types[ele_type]
    n_nodes_per_cell = cells.shape[1]
    n_cells = cells.shape[0]

    # Build VTK cell array: [n_nodes, node0, node1, ...]
    vtk_cells = onp.hstack([
        onp.full((n_cells, 1), n_nodes_per_cell, dtype=cells.dtype),
        cells,
    ]).ravel()
    cell_types = onp.full(n_cells, vtk_type, dtype=onp.uint8)

    grid = pv.UnstructuredGrid(vtk_cells, cell_types, points)
    grid['density'] = onp.asarray(density).ravel()
    return grid


def extract_surface(
    density: onp.ndarray,
    mesh,
    threshold: float = 0.5,
    smooth_iterations: int = 0,
) -> 'pyvista.PolyData':
    """Extract a closed, manifold surface mesh from a density field.

    Uses VTK's ``clip_scalar`` to keep the solid region (density >= threshold),
    then extracts the outer surface. The result is a closed (manifold) triangle
    mesh where:

    - **Interior faces** are smooth iso-surfaces interpolated along element edges
    - **Boundary faces** are the domain boundary where the solid meets the box edge

    Parameters
    ----------
    density : ndarray, shape (num_nodes,)
        Node-based density field (e.g. ``result.rho_filtered``).
    mesh : feax.Mesh
        The finite element mesh.
    threshold : float
        Iso-density level defining the solid boundary (default 0.5).
    smooth_iterations : int
        Laplacian smoothing passes on the extracted surface (default 0).

    Returns
    -------
    pyvista.PolyData
        Closed triangle surface mesh.

    Example
    -------
    >>> surface = extract_surface(result.rho_filtered, result.mesh)
    >>> surface.save("optimized.stl")
    >>> print(f"Manifold: {surface.is_manifold}")
    """
    grid = _to_pyvista(mesh, density)
    clipped = grid.clip_scalar(scalars='density', value=threshold, invert=False)
    surface = clipped.extract_surface(algorithm=None)

    if smooth_iterations > 0:
        surface = surface.smooth(n_iter=smooth_iterations)

    return surface


def extract_volume_mesh(
    density: onp.ndarray,
    mesh,
    threshold: float = 0.5,
    mesh_size: Optional[float] = None,
    element_type: str = 'TET4',
) -> 'feax.Mesh':
    """Extract a clean volume mesh from a density field via surface remeshing.

    Pipeline: density field → VTK iso-surface → STL → Gmsh volume mesh → feax Mesh.

    Parameters
    ----------
    density : ndarray, shape (num_nodes,)
        Node-based density field.
    mesh : feax.Mesh
        The finite element mesh.
    threshold : float
        Iso-density level (default 0.5).
    mesh_size : float, optional
        Target element size for Gmsh. Defaults to average edge length of the
        input mesh.
    element_type : str
        ``'TET4'`` (default) or ``'TET10'``.

    Returns
    -------
    feax.Mesh
        Volumetric finite element mesh of the solid region.
    """
    import tempfile
    import os

    try:
        import gmsh
    except ImportError:
        raise ImportError(
            "extract_volume_mesh requires gmsh. "
            "Install with: pip install gmsh"
        )
    import meshio
    from feax.mesh import Mesh

    # Step 1: extract closed surface
    surface = extract_surface(density, mesh, threshold=threshold)

    if surface.n_points < 4:
        raise RuntimeError(
            f"Surface extraction produced only {surface.n_points} points. "
            "Check the density field and threshold."
        )

    # Step 2: write temporary STL
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        stl_path = f.name
    surface.save(stl_path)

    # Step 3: default mesh size from input mesh
    if mesh_size is None:
        points = onp.asarray(mesh.points)
        cells = onp.asarray(mesh.cells)
        edge_vecs = points[cells[:, 1]] - points[cells[:, 0]]
        mesh_size = float(onp.mean(onp.linalg.norm(edge_vecs, axis=1)))

    # Step 4: Gmsh volume meshing
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.merge(stl_path)

    gmsh.model.mesh.createTopology()
    gmsh.model.mesh.classifySurfaces(
        40 * 3.14159265 / 180,  # angle
        True,                    # boundary
        True,                    # forceParametrizablePatches
    )
    gmsh.model.mesh.createGeometry()

    surfaces = gmsh.model.getEntities(2)
    if not surfaces:
        gmsh.finalize()
        os.unlink(stl_path)
        raise RuntimeError(
            "Gmsh could not create surfaces from STL. "
            "Try adjusting the threshold."
        )

    loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
    gmsh.model.geo.addVolume([loop])
    gmsh.model.geo.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    if element_type == 'TET10':
        gmsh.option.setNumber("Mesh.ElementOrder", 2)

    gmsh.model.mesh.generate(3)

    with tempfile.NamedTemporaryFile(suffix='.msh', delete=False) as f:
        msh_path = f.name
    gmsh.write(msh_path)
    gmsh.finalize()

    # Step 5: convert to feax Mesh
    msh_data = meshio.read(msh_path)
    result_mesh = Mesh.from_gmsh(msh_data, element_type=element_type)

    os.unlink(stl_path)
    os.unlink(msh_path)

    print(f"Volume mesh: {result_mesh.points.shape[0]} nodes, "
          f"{result_mesh.cells.shape[0]} elements ({element_type})")

    return result_mesh


def from_opt_result(
    result,
    threshold: float = 0.5,
) -> 'pyvista.PolyData':
    """Extract surface mesh from an OptResult.

    Convenience wrapper around :func:`extract_surface` that uses
    ``result.rho_filtered`` and ``result.mesh``.

    Parameters
    ----------
    result : OptResult
        Output of ``feax.gene.optimizer.run()``.
    threshold : float
        Iso-density level (default 0.5).

    Returns
    -------
    pyvista.PolyData
    """
    return extract_surface(result.rho_filtered, result.mesh, threshold=threshold)
