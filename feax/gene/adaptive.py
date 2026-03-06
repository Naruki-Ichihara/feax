"""Adaptive remeshing for topology optimization using Gmsh size callback.

Regenerates TET4 meshes with element sizes controlled by density field:
small elements near solid regions, large elements in void regions.

Two entry points:

- ``adaptive_mesh`` — general-purpose; accepts any geometry
  (STEP/BREP file or a Gmsh model-builder callable).
- ``adaptive_box_mesh`` — convenience wrapper for axis-aligned boxes
  with optional periodic face meshing (for unit-cell homogenisation).
"""

import numpy as onp
from scipy.spatial import cKDTree
from typing import Callable, Union, Tuple, Optional

import gmsh

from feax.mesh import Mesh


# ---------------------------------------------------------------------------
# Cross-mesh density interpolation
# ---------------------------------------------------------------------------

def interpolate_field(values_old, points_old, points_new, clip=None):
    """Transfer a node-based scalar field from one mesh to another.

    Uses Delaunay-based linear interpolation for accuracy, with
    nearest-neighbor fallback for points outside the old convex hull.

    Parameters
    ----------
    values_old : ndarray, shape (num_old_nodes,)
        Field values on the old mesh (density, temperature, etc.).
    points_old : ndarray, shape (num_old_nodes, 3)
        Node coordinates of the old mesh.
    points_new : ndarray, shape (num_new_nodes, 3)
        Node coordinates of the new mesh.
    clip : (float, float), optional
        If given, clamp output to ``(min, max)``.

    Returns
    -------
    values_new : ndarray, shape (num_new_nodes,)
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    values_old = onp.array(values_old)
    points_old = onp.array(points_old)
    points_new = onp.array(points_new)

    interp_lin = LinearNDInterpolator(points_old, values_old)
    values_new = interp_lin(points_new)

    nan_mask = onp.isnan(values_new)
    if nan_mask.any():
        interp_nn = NearestNDInterpolator(points_old, values_old)
        values_new[nan_mask] = interp_nn(points_new[nan_mask])

    if clip is not None:
        values_new = onp.clip(values_new, clip[0], clip[1])

    return values_new


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_size_callback(density_field, old_mesh, h_min, h_max):
    """Build a Gmsh size callback driven by a density field.

    Returns ``(callback, is_uniform)``.  If the field is None or
    constant, ``is_uniform`` is True and the callback returns ``h_max``
    everywhere.
    """
    if density_field is not None and old_mesh is not None:
        _density = onp.array(density_field).clip(0, 1)
        _tree = cKDTree(onp.array(old_mesh.points))
        _uniform = float(_density.max()) - float(_density.min()) < 1e-8
    else:
        _density = None
        _tree = None
        _uniform = True

    def size_cb(dim, tag, x, y, z, lc):
        if _uniform:
            return h_max
        _, idx = _tree.query([x, y, z])
        return h_max - _density[idx] * (h_max - h_min)

    return size_cb, _uniform


def _extract_tet4():
    """Extract TET4 mesh from the current Gmsh model.

    Returns a ``Mesh`` with contiguously-indexed nodes.
    """
    _, node_coords, _ = gmsh.model.mesh.getNodes()
    points = node_coords.reshape(-1, 3)

    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(3, -1)
    tet_idx = None
    for i, etype in enumerate(elem_types):
        if etype == 4:  # 4 = TET4
            tet_idx = i
            break
    if tet_idx is None:
        raise RuntimeError("No TET4 elements found in generated mesh")
    cells = elem_node_tags[tet_idx].reshape(-1, 4) - 1

    # Reindex to contiguous node IDs
    unique_nodes = onp.unique(cells.flatten())
    nmap = onp.full(len(points), -1, dtype=onp.int32)
    nmap[unique_nodes] = onp.arange(len(unique_nodes))
    return Mesh(points[unique_nodes], nmap[cells], ele_type='TET4')


def _apply_size_options():
    """Disable Gmsh's built-in size heuristics so the callback is sole
    authority on element size."""
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1e22)


# ---------------------------------------------------------------------------
# Periodic face helper (box-specific)
# ---------------------------------------------------------------------------

def _identify_periodic_face_pairs(x0, y0, z0, lx, ly, lz):
    """Identify opposite face pairs from Gmsh surfaces and set periodicity."""
    surfaces = gmsh.model.getEntities(2)
    eps = 1e-6

    face_pairs = {}  # axis -> {'master': tag, 'slave': tag}

    for _, tag in surfaces:
        bb = gmsh.model.getBoundingBox(2, tag)
        xmin, ymin, zmin, xmax, ymax, zmax = bb

        if abs(xmax - xmin) < eps:
            if abs(xmin - x0) < eps:
                face_pairs.setdefault(0, {})['master'] = tag
            elif abs(xmin - (x0 + lx)) < eps:
                face_pairs.setdefault(0, {})['slave'] = tag
        elif abs(ymax - ymin) < eps:
            if abs(ymin - y0) < eps:
                face_pairs.setdefault(1, {})['master'] = tag
            elif abs(ymin - (y0 + ly)) < eps:
                face_pairs.setdefault(1, {})['slave'] = tag
        elif abs(zmax - zmin) < eps:
            if abs(zmin - z0) < eps:
                face_pairs.setdefault(2, {})['master'] = tag
            elif abs(zmin - (z0 + lz)) < eps:
                face_pairs.setdefault(2, {})['slave'] = tag

    translations = [lx, ly, lz]
    for axis, pair in face_pairs.items():
        if 'master' not in pair or 'slave' not in pair:
            continue
        affine = [0.0] * 16
        affine[0] = affine[5] = affine[10] = affine[15] = 1.0
        affine[3 + axis * 4] = translations[axis]
        gmsh.model.mesh.setPeriodic(
            2, [pair['slave']], [pair['master']], affine
        )


# ---------------------------------------------------------------------------
# General-purpose adaptive meshing
# ---------------------------------------------------------------------------

def adaptive_mesh(
    geometry: Union[str, Callable[[], None]],
    density_field: Optional[onp.ndarray] = None,
    old_mesh: Optional[Mesh] = None,
    h_min: float = 0.02,
    h_max: float = 0.15,
) -> Mesh:
    """Generate adaptive TET4 mesh for an arbitrary geometry.

    Element sizes are driven by a density field: high-density regions
    receive fine elements (``h_min``), low-density regions receive
    coarse elements (``h_max``).

    Parameters
    ----------
    geometry : str or callable
        How to define the geometry in Gmsh:

        - **str** — path to a CAD file (STEP, BREP, IGES, …).
          Gmsh imports it via ``gmsh.model.occ.importShapes``.
        - **callable** — ``() -> None`` that builds the Gmsh model
          programmatically.  Called after ``gmsh.initialize()``; must
          leave the model synchronised (``occ.synchronize()``).
    density_field : ndarray, shape (num_old_nodes,), optional
        Filtered density on the old mesh (pre-Heaviside).
        If None, a uniform mesh at ``h_max`` is generated.
    old_mesh : Mesh, optional
        Previous mesh (needed to map density_field to new mesh points).
    h_min : float
        Minimum element size (solid regions).
    h_max : float
        Maximum element size (void regions / uniform).

    Returns
    -------
    Mesh
        New TET4 mesh with adaptive element sizes.

    Examples::

        # From a STEP file
        mesh = adaptive_mesh("bracket.step", rho, old_mesh,
                             h_min=0.5, h_max=3.0)

        # From a Gmsh model builder
        def my_geometry():
            gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 10, 2.0)
            gmsh.model.occ.synchronize()

        mesh = adaptive_mesh(my_geometry, h_max=1.0)
    """
    size_cb, _ = _make_size_callback(density_field, old_mesh, h_min, h_max)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("adaptive")

        if isinstance(geometry, str):
            gmsh.model.occ.importShapes(geometry)
            gmsh.model.occ.synchronize()
        else:
            geometry()

        gmsh.model.mesh.setSizeCallback(size_cb)
        _apply_size_options()

        gmsh.model.mesh.generate(3)
        return _extract_tet4()
    finally:
        gmsh.finalize()


# ---------------------------------------------------------------------------
# Box-mesh convenience wrapper (with periodic support)
# ---------------------------------------------------------------------------

def adaptive_box_mesh(
    size: Union[float, Tuple[float, float, float]],
    density_field: Optional[onp.ndarray] = None,
    old_mesh: Optional[Mesh] = None,
    h_min: float = 0.02,
    h_max: float = 0.15,
    target_nodes: Optional[int] = None,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    periodic: bool = False,
    max_retries: int = 3,
) -> Mesh:
    """Generate adaptive TET4 box mesh (with optional periodic faces).

    Convenience wrapper around :func:`adaptive_mesh` for axis-aligned
    boxes.  Adds periodic face meshing for unit-cell homogenisation and
    optional node-count tuning.

    Parameters
    ----------
    size : float or (float, float, float)
        Domain dimensions (cube side length, or ``(lx, ly, lz)``).
    density_field : ndarray, shape (num_old_nodes,), optional
        Filtered density on the old mesh (pre-Heaviside).
    old_mesh : Mesh, optional
        Previous mesh.
    h_min : float
        Minimum element size.
    h_max : float
        Maximum element size.
    target_nodes : int, optional
        If set, iteratively scale h_min/h_max to approximately match
        this node count (within ±30 %).
    origin : (float, float, float)
        Domain origin.
    periodic : bool
        If True, enforce periodic face meshing via Gmsh.
    max_retries : int
        Maximum mesh generation attempts for node-count tuning.

    Returns
    -------
    Mesh
        New TET4 mesh.
    """
    if isinstance(size, (int, float)):
        lx = ly = lz = float(size)
    else:
        lx, ly, lz = size
    x0, y0, z0 = origin

    def _generate(h_lo, h_hi):
        size_cb, _ = _make_size_callback(density_field, old_mesh, h_lo, h_hi)

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            gmsh.model.add("adaptive_box")
            gmsh.model.occ.addBox(x0, y0, z0, lx, ly, lz)
            gmsh.model.occ.synchronize()

            gmsh.model.mesh.setSizeCallback(size_cb)
            _apply_size_options()

            if periodic:
                _identify_periodic_face_pairs(x0, y0, z0, lx, ly, lz)

            gmsh.model.mesh.generate(3)
            return _extract_tet4()
        finally:
            gmsh.finalize()

    cur_h_min, cur_h_max = h_min, h_max
    mesh_out = _generate(cur_h_min, cur_h_max)

    if target_nodes is not None:
        for _ in range(max_retries):
            n = mesh_out.points.shape[0]
            if 0.7 < (n / target_nodes) < 1.3:
                break
            ratio = (n / target_nodes) ** (1.0 / 3.0)
            cur_h_min *= ratio
            cur_h_max *= ratio
            mesh_out = _generate(cur_h_min, cur_h_max)
            print(f"    Retune: h=[{cur_h_min:.4f}, {cur_h_max:.4f}] "
                  f"-> {mesh_out.points.shape[0]} nodes")

    return mesh_out
