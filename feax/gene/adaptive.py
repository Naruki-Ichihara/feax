"""Adaptive remeshing for topology optimization using Gmsh size callback.

Regenerates TET4 meshes with element sizes controlled by a refinement
field: high-value regions receive fine elements, low-value regions
receive coarse elements.  The field can be any node-based scalar in
[0, 1] — filtered density, density-gradient magnitude, etc.

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
# Cross-mesh field interpolation (TET4)
# ---------------------------------------------------------------------------

def interpolate_field(values_old, mesh, points_new, clip=None):
    """Transfer a node-based scalar field to a new point set via TET4
    shape functions (barycentric coordinates).

    For each query point, finds the containing tetrahedron and
    interpolates using barycentric weights.  Points outside the mesh
    fall back to nearest-node value.

    Parameters
    ----------
    values_old : ndarray, shape (num_old_nodes,)
        Field values on the old mesh (density, temperature, etc.).
    mesh : Mesh
        Old TET4 mesh (``cells`` must have 4 columns).
    points_new : ndarray, shape (num_new_nodes, 3)
        Node coordinates of the new mesh.
    clip : (float, float), optional
        If given, clamp output to ``(min, max)``.

    Returns
    -------
    values_new : ndarray, shape (num_new_nodes,)

    Raises
    ------
    ValueError
        If the mesh is not TET4 (cells must have exactly 4 nodes).
    """
    values_old = onp.array(values_old)
    points_old = onp.array(mesh.points)
    points_new = onp.array(points_new)
    cells = onp.array(mesh.cells)

    if cells.shape[1] != 4:
        raise ValueError(
            f"interpolate_field requires TET4 mesh (4 nodes per element), "
            f"got {cells.shape[1]}")

    n_elem = len(cells)
    n_query = len(points_new)

    # Precompute inverse transforms for all elements
    # For TET4: p = v0 + T @ lambda_{1,2,3}
    v0 = points_old[cells[:, 0]]                 # (E, 3)
    T = onp.stack([
        points_old[cells[:, 1]] - v0,
        points_old[cells[:, 2]] - v0,
        points_old[cells[:, 3]] - v0,
    ], axis=-1)                                  # (E, 3, 3)
    inv_T = onp.linalg.inv(T)                   # (E, 3, 3)

    # Element centroids for candidate lookup
    centroids = points_old[cells].mean(axis=1)   # (E, 3)
    tree = cKDTree(centroids)

    n_candidates = min(8, n_elem)
    _, cand_all = tree.query(points_new, k=n_candidates)  # (Q, K)
    if n_candidates == 1:
        cand_all = cand_all[:, None]

    # Try all candidates in vectorised batches
    values_new = onp.full(n_query, onp.nan)

    for k in range(n_candidates):
        mask = onp.isnan(values_new)
        if not mask.any():
            break

        idx = onp.where(mask)[0]
        ei = cand_all[idx, k]                    # element indices

        # Barycentric coords: lam = inv_T[ei] @ (pt - v0[ei])
        diff = points_new[idx] - v0[ei]          # (N, 3)
        lam = onp.einsum('nij,nj->ni', inv_T[ei], diff)  # (N, 3)
        lam0 = 1.0 - lam.sum(axis=1)            # (N,)

        # Containment check
        inside = (lam0 >= -1e-8) & onp.all(lam >= -1e-8, axis=1)
        hit = idx[inside]
        hit_ei = ei[inside]
        hit_lam = lam[inside]
        hit_lam0 = lam0[inside]

        # Interpolate: N0*v0 + N1*v1 + N2*v2 + N3*v3
        node_vals = values_old[cells[hit_ei]]    # (H, 4)
        values_new[hit] = (hit_lam0 * node_vals[:, 0]
                           + hit_lam[:, 0] * node_vals[:, 1]
                           + hit_lam[:, 1] * node_vals[:, 2]
                           + hit_lam[:, 2] * node_vals[:, 3])

    # Nearest-node fallback for remaining points
    remaining = onp.isnan(values_new)
    if remaining.any():
        node_tree = cKDTree(points_old)
        _, ni = node_tree.query(points_new[remaining])
        values_new[remaining] = values_old[ni]

    if clip is not None:
        values_new = onp.clip(values_new, clip[0], clip[1])

    return values_new


# ---------------------------------------------------------------------------
# Refinement field helpers
# ---------------------------------------------------------------------------

def gradient_refinement(rho, mesh):
    """Compute normalised gradient magnitude as a refinement field.

    Uses TET4 shape functions (constant gradient per element), then
    averages to nodes and normalises to [0, 1].

    Parameters
    ----------
    rho : ndarray, shape (num_nodes,)
        Node-based scalar field (e.g. filtered density).
    mesh : Mesh
        TET4 mesh (``cells`` must have 4 columns).

    Returns
    -------
    ndarray, shape (num_nodes,)
        Normalised gradient magnitude in [0, 1], suitable for use as
        ``refinement_field`` in :func:`adaptive_mesh`.

    Raises
    ------
    ValueError
        If the mesh is not TET4 (cells must have exactly 4 nodes).
    """
    pts = onp.array(mesh.points)
    cells = onp.array(mesh.cells)
    rho = onp.array(rho)

    if cells.shape[1] != 4:
        raise ValueError(
            f"gradient_refinement requires TET4 mesh (4 nodes per element), "
            f"got {cells.shape[1]}")

    # Element vertex coords: (E, 4, 3)
    v = pts[cells]
    # Jacobian: edges from node 0 -> (E, 3, 3)
    J = v[:, 1:, :] - v[:, 0:1, :]
    inv_J = onp.linalg.inv(J)

    # TET4 shape function gradients in reference coords:
    #   N0 = 1-xi-eta-zeta,  N1 = xi,  N2 = eta,  N3 = zeta
    rho_e = rho[cells]  # (E, 4)
    drho_ref = rho_e[:, 1:] - rho_e[:, 0:1]  # (E, 3)

    # Physical gradient: grad(rho) = inv(J)^T @ drho_ref
    grad_rho = onp.einsum('eij,ej->ei', inv_J.transpose(0, 2, 1), drho_ref)
    grad_mag_elem = onp.linalg.norm(grad_rho, axis=1)  # (E,)

    # Scatter to nodes (average of surrounding elements)
    grad_mag_node = onp.zeros(len(pts))
    count = onp.zeros(len(pts))
    for i in range(cells.shape[1]):
        onp.add.at(grad_mag_node, cells[:, i], grad_mag_elem)
        onp.add.at(count, cells[:, i], 1.0)
    count = onp.maximum(count, 1.0)
    grad_mag_node /= count

    # Normalise to [0, 1]
    gmax = grad_mag_node.max()
    if gmax > 1e-12:
        grad_mag_node /= gmax

    return grad_mag_node


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_size_callback(refinement_field, old_mesh, h_min, h_max):
    """Build a Gmsh size callback driven by a refinement field.

    The field should be a node-based scalar in [0, 1].  Values near 1
    produce elements of size ``h_min``; values near 0 produce ``h_max``.

    Returns ``(callback, is_uniform)``.  If the field is None or
    constant, ``is_uniform`` is True and the callback returns ``h_max``
    everywhere.
    """
    if refinement_field is not None and old_mesh is not None:
        _field = onp.array(refinement_field).clip(0, 1)
        _tree = cKDTree(onp.array(old_mesh.points))
        _uniform = float(_field.max()) - float(_field.min()) < 1e-8
    else:
        _field = None
        _tree = None
        _uniform = True

    def size_cb(dim, tag, x, y, z, lc):
        if _uniform:
            return h_max
        _, idx = _tree.query([x, y, z])
        return h_max - _field[idx] * (h_max - h_min)

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
    refinement_field: Optional[onp.ndarray] = None,
    old_mesh: Optional[Mesh] = None,
    h_min: float = 0.02,
    h_max: float = 0.15,
) -> Mesh:
    """Generate adaptive TET4 mesh for an arbitrary geometry.

    Element sizes are driven by a refinement field (any node-based
    scalar in [0, 1]): high values produce fine elements (``h_min``),
    low values produce coarse elements (``h_max``).  Typical choices
    include filtered density, density-gradient magnitude, or an
    indicator function.

    Parameters
    ----------
    geometry : str or callable
        How to define the geometry in Gmsh:

        - **str** — path to a CAD file (STEP, BREP, IGES, …).
          Gmsh imports it via ``gmsh.model.occ.importShapes``.
        - **callable** — ``() -> None`` that builds the Gmsh model
          programmatically.  Called after ``gmsh.initialize()``; must
          leave the model synchronised (``occ.synchronize()``).
    refinement_field : ndarray, shape (num_old_nodes,), optional
        Node-based scalar in [0, 1] on the old mesh.  Values are
        clamped to [0, 1] internally.
        If None, a uniform mesh at ``h_max`` is generated.
    old_mesh : Mesh, optional
        Previous mesh (needed to map the field to new mesh points).
    h_min : float
        Minimum element size (where field ≈ 1).
    h_max : float
        Maximum element size (where field ≈ 0, or uniform).

    Returns
    -------
    Mesh
        New TET4 mesh with adaptive element sizes.

    Examples:

    ```python
    # From a STEP file — refine by density
    mesh = adaptive_mesh("bracket.step", rho_filtered, old_mesh,
                         h_min=0.5, h_max=3.0)

    # Refine by density-gradient magnitude
    grad_mag = compute_gradient_magnitude(rho_filtered, old_mesh)
    grad_norm = grad_mag / grad_mag.max()  # normalise to [0, 1]
    mesh = adaptive_mesh(geometry, grad_norm, old_mesh,
                         h_min=0.5, h_max=3.0)
    ```
    """
    size_cb, _ = _make_size_callback(refinement_field, old_mesh, h_min, h_max)

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
    refinement_field: Optional[onp.ndarray] = None,
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
    refinement_field : ndarray, shape (num_old_nodes,), optional
        Node-based scalar in [0, 1] on the old mesh (see
        :func:`adaptive_mesh` for details).
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
        size_cb, _ = _make_size_callback(refinement_field, old_mesh, h_lo, h_hi)

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
