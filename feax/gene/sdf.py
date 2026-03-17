"""Signed distance field conversion for topology optimization results.

Converts node-based density fields from topology optimization into
queryable signed distance fields (SDFs) and exports surface meshes
via marching cubes.

The interface follows jaxcad conventions: an SDF is a callable
``f(p) -> distance`` where negative values indicate the interior.

Supports all element types in feax (TET4, TET10, HEX8, HEX20,
TRI3, TRI6, QUAD4, QUAD8) via basix shape functions and Newton
inverse mapping.

Example
-------
>>> result = gene.optimizer.run(...)
>>> field = DensityField(result.rho_filtered, result.mesh, threshold=0.5)
>>> verts, faces = field.to_mesh(resolution=80)
>>> field.to_stl("optimized.stl", resolution=80)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import basix
import meshio
import numpy as onp
from scipy.spatial import cKDTree

from feax.basis import get_elements


# ---------------------------------------------------------------------------
# Abstract base — mirrors jaxcad.SDF
# ---------------------------------------------------------------------------

class SDF(ABC):
    """Abstract signed distance function.

    Convention: ``f(p) < 0`` inside, ``f(p) = 0`` on surface,
    ``f(p) > 0`` outside.
    """

    @abstractmethod
    def __call__(self, p: onp.ndarray) -> float:
        """Evaluate signed distance at a single point *p*."""

    def evaluate(self, points: onp.ndarray) -> onp.ndarray:
        """Vectorised evaluation on an array of points ``(N, dim) -> (N,)``."""
        return onp.array([self(p) for p in points])

    # -- CSG operators (jaxcad-compatible) -----------------------------------

    def __or__(self, other: SDF) -> SDF:
        """Union: ``self | other``."""
        return Union(self, other)

    def __and__(self, other: SDF) -> SDF:
        """Intersection: ``self & other``."""
        return Intersection(self, other)

    def __sub__(self, other: SDF) -> SDF:
        """Difference: ``self - other``."""
        return Difference(self, other)

    # -- Mesh extraction -----------------------------------------------------

    def to_mesh(
        self,
        resolution: int = 80,
        bounds: Optional[Tuple[onp.ndarray, onp.ndarray]] = None,
        level: float = 0.0,
        padding: float = 0.0,
        watertight: bool = False,
    ) -> Tuple[onp.ndarray, onp.ndarray]:
        """Extract a triangle surface mesh via marching cubes.

        Parameters
        ----------
        resolution : int
            Number of grid points per axis.
        bounds : (lower, upper), optional
            Axis-aligned bounding box as two arrays.
            Defaults to the bounding box stored on the SDF (if any).
        level : float
            Iso-level for marching cubes (default 0.0 = surface).
        padding : float
            Extra space added around bounds in each direction.
        watertight : bool
            If ``True``, post-process the mesh with trimesh to
            guarantee a watertight (manifold, closed) surface.
            Merges duplicate vertices, removes degenerate faces,
            fills holes, and fixes normals.  Requires ``trimesh``.

        Returns
        -------
        vertices : ndarray, shape (V, 3)
        faces : ndarray, shape (F, 3)   — triangle connectivity (int)
        """
        try:
            from skimage.measure import marching_cubes
        except ImportError:
            raise ImportError(
                "to_mesh() requires scikit-image. "
                "Install with: pip install scikit-image"
            )

        if bounds is None:
            bounds = self._default_bounds()
        lo = onp.asarray(bounds[0], dtype=onp.float64) - padding
        hi = onp.asarray(bounds[1], dtype=onp.float64) + padding
        size = hi - lo

        # Build regular grid
        axes = [onp.linspace(lo[d], hi[d], resolution) for d in range(3)]
        X, Y, Z = onp.meshgrid(*axes, indexing='ij')
        pts = onp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        # Evaluate SDF on grid
        vol = self.evaluate(pts).reshape(resolution, resolution, resolution)

        spacing = tuple(size / (resolution - 1))
        verts, faces, _, _ = marching_cubes(vol, level=level, spacing=spacing)
        verts += lo  # shift back to world coordinates
        faces = faces.astype(onp.int32)

        if watertight:
            verts, faces = _repair_watertight(verts, faces)

        return verts, faces

    def to_stl(
        self,
        path: str,
        resolution: int = 80,
        bounds: Optional[Tuple[onp.ndarray, onp.ndarray]] = None,
        level: float = 0.0,
        padding: float = 0.0,
        watertight: bool = False,
    ) -> None:
        """Export the iso-surface as an STL file.

        Parameters are forwarded to :meth:`to_mesh`.
        """
        verts, faces = self.to_mesh(
            resolution=resolution, bounds=bounds, level=level,
            padding=padding, watertight=watertight,
        )
        mesh = meshio.Mesh(points=verts, cells=[("triangle", faces)])
        mesh.write(path)
        print(f"Saved STL: {path}  ({len(verts)} vertices, {len(faces)} faces)")

    # -- Internal helpers ----------------------------------------------------

    def _default_bounds(self):
        raise NotImplementedError(
            "bounds must be supplied explicitly for this SDF type."
        )


# ---------------------------------------------------------------------------
# Watertight mesh repair
# ---------------------------------------------------------------------------

def _repair_watertight(verts, faces):
    """Post-process a triangle mesh to guarantee watertight (manifold) output.

    Pipeline:
    1. Merge duplicate / near-duplicate vertices
    2. Remove degenerate (zero-area) faces
    3. Fill holes
    4. Fix face winding for consistent outward normals
    5. Split into connected components and keep only the largest

    Requires the ``trimesh`` package.

    Returns
    -------
    verts : ndarray (V, 3)
    faces : ndarray (F, 3)  int32
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "watertight=True requires trimesh. "
            "Install with: pip install trimesh"
        )

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # 1. Merge close vertices and remove degenerate faces
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())

    # 2. Fill holes
    trimesh.repair.fill_holes(mesh)

    # 3. Fix winding / normals
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_inversion(mesh)

    # 4. If multiple components, keep the largest
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        largest = max(components, key=lambda c: c.area)
        # Re-apply hole filling on the largest component
        trimesh.repair.fill_holes(largest)
        trimesh.repair.fix_normals(largest)
        mesh = largest

    if mesh.is_watertight:
        print(f"  Watertight: OK  (Euler χ={mesh.euler_number})")
    else:
        print(f"  Watertight: repair incomplete — "
              f"{mesh.euler_number=}, "
              f"holes may remain.  Try increasing resolution.")

    return onp.array(mesh.vertices), onp.array(mesh.faces, dtype=onp.int32)


# ---------------------------------------------------------------------------
# CSG operations
# ---------------------------------------------------------------------------

class Union(SDF):
    """Boolean union of two SDFs (``min``)."""

    def __init__(self, a: SDF, b: SDF):
        self.a, self.b = a, b

    def __call__(self, p):
        return min(self.a(p), self.b(p))

    def evaluate(self, points):
        return onp.minimum(self.a.evaluate(points), self.b.evaluate(points))

    def _default_bounds(self):
        la, ha = self.a._default_bounds()
        lb, hb = self.b._default_bounds()
        return onp.minimum(la, lb), onp.maximum(ha, hb)


class Intersection(SDF):
    """Boolean intersection of two SDFs (``max``)."""

    def __init__(self, a: SDF, b: SDF):
        self.a, self.b = a, b

    def __call__(self, p):
        return max(self.a(p), self.b(p))

    def evaluate(self, points):
        return onp.maximum(self.a.evaluate(points), self.b.evaluate(points))

    def _default_bounds(self):
        la, ha = self.a._default_bounds()
        lb, hb = self.b._default_bounds()
        return onp.maximum(la, lb), onp.minimum(ha, hb)


class Difference(SDF):
    """Boolean difference ``a - b``  (i.e. ``max(a, -b)``)."""

    def __init__(self, a: SDF, b: SDF):
        self.a, self.b = a, b

    def __call__(self, p):
        return max(self.a(p), -self.b(p))

    def evaluate(self, points):
        return onp.maximum(self.a.evaluate(points), -self.b.evaluate(points))

    def _default_bounds(self):
        return self.a._default_bounds()


# ---------------------------------------------------------------------------
# Reference element containment check
# ---------------------------------------------------------------------------

_SIMPLEX_TYPES = {basix.CellType.tetrahedron, basix.CellType.triangle}
_TENSOR_TYPES = {basix.CellType.hexahedron, basix.CellType.quadrilateral}


def _inside_ref_element(xi, cell_type, tol=1e-6):
    """Check if reference coordinates lie inside the basix reference element.

    Basix uses [0, 1] reference domains:
    - Simplex (tet/tri): all xi >= 0 and sum(xi) <= 1
    - Tensor  (hex/quad): all 0 <= xi <= 1
    """
    if cell_type in _SIMPLEX_TYPES:
        return onp.all(xi >= -tol) and xi.sum() <= 1.0 + tol
    elif cell_type in _TENSOR_TYPES:
        return onp.all(xi >= -tol) and onp.all(xi <= 1.0 + tol)
    else:
        raise ValueError(f"Unsupported cell type: {cell_type}")


def _inside_ref_element_batch(xi, cell_type, tol=1e-6):
    """Vectorised containment check.  ``xi`` has shape ``(N, dim)``."""
    if cell_type in _SIMPLEX_TYPES:
        return onp.all(xi >= -tol, axis=1) & (xi.sum(axis=1) <= 1.0 + tol)
    elif cell_type in _TENSOR_TYPES:
        return onp.all(xi >= -tol, axis=1) & onp.all(xi <= 1.0 + tol, axis=1)
    else:
        raise ValueError(f"Unsupported cell type: {cell_type}")


# ---------------------------------------------------------------------------
# DensityField — density-to-SDF adapter (all element types)
# ---------------------------------------------------------------------------

class DensityField(SDF):
    """Convert a node-based density field on an FE mesh into an SDF.

    The signed distance is approximated as ``threshold - density(p)``:

    * density > threshold  →  SDF < 0  (inside / solid)
    * density < threshold  →  SDF > 0  (outside / void)
    * density = threshold  →  SDF = 0  (surface)

    Supports all element types available in feax (TET4, TET10, HEX8,
    HEX20, TRI3, TRI6, QUAD4, QUAD8).  Uses basix shape functions
    for interpolation and Newton iteration for inverse mapping from
    physical to reference coordinates.

    For linear simplex elements (TET4, TRI3) a fast direct
    barycentric path is used instead of Newton iteration.

    Parameters
    ----------
    rho : ndarray, shape (num_nodes,)
        Node-based density (filtered / projected).
    mesh : feax.Mesh
        Finite element mesh.
    threshold : float
        Iso-density level that defines the solid boundary.
    close_boundary : bool
        If ``True`` (default), points outside the FE mesh are
        treated as void (density = 0) so that the marching-cubes
        surface closes at the domain boundary.  If ``False``, the
        nearest-node density is used instead (legacy behaviour).
    """

    def __init__(
        self,
        rho: onp.ndarray,
        mesh,
        threshold: float = 0.5,
        close_boundary: bool = True,
    ):
        rho = onp.asarray(rho).ravel()
        points = onp.asarray(mesh.points, dtype=onp.float64)
        cells = onp.asarray(mesh.cells)
        ele_type = mesh.ele_type

        self._rho = rho
        self._points = points
        self._cells = cells
        self._close_boundary = close_boundary
        self._ele_type = ele_type
        self._threshold = threshold
        self._dim = points.shape[1]

        # Bounding box
        self._lo = points.min(axis=0)
        self._hi = points.max(axis=0)

        # --- basix element info ---
        family, basix_ele, _, _, degree, re_order = get_elements(ele_type)
        self._basix_ele = basix_ele
        self._re_order = onp.array(re_order)
        self._element = basix.create_element(family, basix_ele, degree)
        self._ref_centroid = basix.geometry(basix_ele).mean(axis=0)

        # Fast path flag: TET4 / TRI3 have linear maps → direct solve
        self._linear_simplex = (
            ele_type in ('TET4', 'TRI3') and degree == 1
        )

        # --- Precompute per-element data ---
        # Node coordinates per element: (E, nodes_per_elem, dim)
        self._elem_nodes = points[cells[:, re_order]]

        if self._linear_simplex:
            # Direct barycentric: x = v0 + T @ lam
            v0 = self._elem_nodes[:, 0]  # (E, dim)
            T = self._elem_nodes[:, 1:] - v0[:, None, :]  # (E, dim, dim)
            # T has shape (E, n_simplex_verts, dim), rearrange to (E, dim, dim)
            T = onp.transpose(T, (0, 2, 1))  # (E, dim, n)
            self._v0 = v0
            self._inv_T = onp.linalg.inv(T)

        # Newton iteration settings
        self._newton_max_iter = 1 if self._linear_simplex else 20
        self._newton_tol = 1e-10

        # --- Spatial search structures ---
        self._centroids = self._elem_nodes.mean(axis=1)  # (E, dim)
        self._centroid_tree = cKDTree(self._centroids)
        self._node_tree = cKDTree(points)
        self._n_candidates = min(16, len(cells))

    # -- Single-point evaluation ---------------------------------------------

    def __call__(self, p: onp.ndarray) -> float:
        p = onp.asarray(p, dtype=onp.float64)
        density = self._interpolate_single(p)
        return float(self._threshold - density)

    def _interpolate_single(self, p: onp.ndarray) -> float:
        """Interpolate density at a single point."""
        _, cands = self._centroid_tree.query(p, k=self._n_candidates)
        if self._n_candidates == 1:
            cands = [cands]

        for ei in cands:
            xi = self._inverse_map_single(p, ei)
            if xi is not None and _inside_ref_element(xi, self._basix_ele):
                # Shape functions in basix order, density in basix order
                N = self._shape_vals_at(xi)
                node_vals = self._rho[self._cells[ei][self._re_order]]
                return float(N @ node_vals)

        # Point is outside the mesh
        if self._close_boundary:
            return 0.0  # void → SDF = threshold > 0 → surface closes
        _, ni = self._node_tree.query(p)
        return float(self._rho[ni])

    def _inverse_map_single(self, p, ei):
        """Map physical point *p* to reference coords for element *ei*.

        Returns reference coordinates or ``None`` if Newton diverges.
        Uses ``self._elem_nodes`` (basix node order) and raw basix
        shape functions (no re-ordering needed — both are basix order).
        """
        elem_pts = self._elem_nodes[ei]  # (nodes_per_elem, dim)

        if self._linear_simplex:
            diff = p - self._v0[ei]
            return self._inv_T[ei] @ diff

        # Newton iteration: find xi s.t. x(xi) = p
        xi = self._ref_centroid.copy()
        for _ in range(self._newton_max_iter):
            tab = self._element.tabulate(1, xi.reshape(1, -1))
            N = tab[0, 0, :, 0]              # (nodes,)  basix order
            dN = tab[1:, 0, :, 0]             # (dim, nodes)  basix order

            x_current = N @ elem_pts          # (dim,)
            J = dN @ elem_pts                 # (dim, dim)

            residual = x_current - p
            res_norm = onp.linalg.norm(residual)
            if not onp.isfinite(res_norm) or res_norm > 1e10:
                return None  # diverged
            if res_norm < self._newton_tol:
                return xi

            try:
                delta = onp.linalg.solve(J, residual)
            except onp.linalg.LinAlgError:
                return None
            xi = xi - delta

        # Accept if close enough after all iterations
        if onp.isfinite(res_norm) and res_norm < 1e-6:
            return xi
        return None

    def _shape_vals_at(self, xi):
        """Evaluate shape function values at a single reference point.

        Returns array of shape ``(nodes_per_elem,)`` in **basix order**.
        """
        if self._linear_simplex:
            return onp.concatenate([[1.0 - xi.sum()], xi])
        tab = self._element.tabulate(0, xi.reshape(1, -1))
        return tab[0, 0, :, 0]  # basix order

    # -- Vectorised evaluation -----------------------------------------------

    def evaluate(self, points: onp.ndarray) -> onp.ndarray:
        """Evaluate SDF at many points ``(N, dim) -> (N,)``.

        Uses batched Newton inverse mapping with basix shape functions.
        For TET4/TRI3 (linear simplex), falls back to the fast direct
        barycentric path.
        """
        points = onp.asarray(points, dtype=onp.float64)
        n_query = len(points)
        dim = self._dim

        _, cand_all = self._centroid_tree.query(
            points, k=self._n_candidates,
        )
        if self._n_candidates == 1:
            cand_all = cand_all[:, None]

        densities = onp.full(n_query, onp.nan)

        for k in range(self._n_candidates):
            mask = onp.isnan(densities)
            if not mask.any():
                break

            idx = onp.where(mask)[0]
            ei = cand_all[idx, k]

            # --- Batch inverse map ---
            xi_batch = self._inverse_map_batch(points[idx], ei)
            if xi_batch is None:
                continue

            # --- Containment check ---
            inside = _inside_ref_element_batch(xi_batch, self._basix_ele)
            hit = idx[inside]
            hit_ei = ei[inside]
            hit_xi = xi_batch[inside]

            if len(hit) == 0:
                continue

            # --- Batch shape function evaluation & interpolation ---
            N_batch = self._shape_vals_batch(hit_xi)  # (H, nodes) basix order
            # Gather density values in basix order: (H, nodes)
            node_vals = self._rho[self._cells[hit_ei][:, self._re_order]]
            densities[hit] = onp.einsum('hj,hj->h', N_batch, node_vals)

        # Points outside the mesh
        remaining = onp.isnan(densities)
        if remaining.any():
            if self._close_boundary:
                densities[remaining] = 0.0  # void → surface closes
            else:
                _, ni = self._node_tree.query(points[remaining])
                densities[remaining] = self._rho[ni]

        return self._threshold - densities

    def _inverse_map_batch(self, pts, ei):
        """Batch inverse mapping for ``pts`` (N, dim) into elements ``ei``.

        Returns reference coordinates ``(N, dim)`` or None.
        Uses basix-ordered node coordinates and raw (basix-ordered)
        shape functions — no re-ordering needed since both match.
        """
        elem_pts = self._elem_nodes[ei]  # (N, nodes, dim) basix order

        if self._linear_simplex:
            diff = pts - self._v0[ei]
            return onp.einsum('nij,nj->ni', self._inv_T[ei], diff)

        # Batched Newton iteration
        N_pts = len(pts)
        xi = onp.ascontiguousarray(
            onp.tile(self._ref_centroid, (N_pts, 1)), dtype=onp.float64,
        )  # (N, dim)

        for _ in range(self._newton_max_iter):
            # tabulate at all xi simultaneously — basix order
            tab = self._element.tabulate(1, xi)
            # tab shape: (1+dim, N, nodes, 1)
            N_vals = tab[0, :, :, 0]    # (N, nodes) basix order
            dN = tab[1:, :, :, 0]        # (dim, N, nodes) basix order

            # x(xi) = sum_j N_j * x_j  for each query point
            x_current = onp.einsum('nj,njk->nk', N_vals, elem_pts)  # (N, dim)
            # Jacobian: J[n,d,k] = sum_j dN[d,n,j] * elem_pts[n,j,k]
            J = onp.einsum('dnj,njk->ndk', dN, elem_pts)  # (N, dim, dim)

            residual = x_current - pts  # (N, dim)
            max_res = onp.max(onp.linalg.norm(residual, axis=1))
            if max_res < self._newton_tol:
                break

            try:
                # solve expects (N, dim, dim) @ (N, dim, 1) → (N, dim, 1)
                delta = onp.linalg.solve(J, residual[..., None])[..., 0]
            except onp.linalg.LinAlgError:
                return None
            xi = onp.ascontiguousarray(xi - delta)

        return xi

    def _shape_vals_batch(self, xi_batch):
        """Evaluate shape functions at multiple reference points.

        Parameters
        ----------
        xi_batch : ndarray, shape (N, dim)

        Returns
        -------
        N : ndarray, shape (N, nodes_per_elem)  — **basix order**
        """
        if self._linear_simplex:
            lam0 = 1.0 - xi_batch.sum(axis=1, keepdims=True)  # (N, 1)
            return onp.concatenate([lam0, xi_batch], axis=1)

        tab = self._element.tabulate(0, xi_batch)
        return tab[0, :, :, 0]  # basix order

    # -- Bounds --------------------------------------------------------------

    def _default_bounds(self):
        return self._lo.copy(), self._hi.copy()

    @property
    def bounds(self) -> Tuple[onp.ndarray, onp.ndarray]:
        """Axis-aligned bounding box ``(lower, upper)`` of the FE mesh."""
        return self._lo.copy(), self._hi.copy()


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def from_opt_result(result, threshold: float = 0.5) -> DensityField:
    """Create a :class:`DensityField` from an :class:`~feax.gene.optimizer.OptResult`.

    Uses the filtered density (``result.rho_filtered``) and the final
    mesh (``result.mesh``).

    Parameters
    ----------
    result : OptResult
        Output of ``feax.gene.optimizer.run()``.
    threshold : float
        Iso-density level (default 0.5).

    Returns
    -------
    DensityField
    """
    return DensityField(result.rho_filtered, result.mesh, threshold=threshold)


_MESHIO_TO_FEAX = {
    'tetra': 'TET4',
    'tetra10': 'TET10',
    'hexahedron': 'HEX8',
    'hexahedron20': 'HEX20',
    'triangle': 'TRI3',
    'triangle6': 'TRI6',
    'quad': 'QUAD4',
    'quad8': 'QUAD8',
}


def from_vtk(
    path: str,
    field_name: str = "density",
    threshold: float = 0.5,
) -> DensityField:
    """Create a :class:`DensityField` from a VTK/VTU file.

    Reads a mesh file (typically ``.vtu`` saved by
    ``feax.utils.save_sol``) containing a node-based scalar field
    and wraps it as an SDF.

    Supports all element types that feax supports (TET4, TET10,
    HEX8, HEX20, TRI3, TRI6, QUAD4, QUAD8).

    Parameters
    ----------
    path : str
        Path to the VTK/VTU file.
    field_name : str
        Name of the point-data array to use as density (default
        ``"density"``).
    threshold : float
        Iso-density level (default 0.5).

    Returns
    -------
    DensityField

    Raises
    ------
    KeyError
        If *field_name* is not found in the file's point data.

    Example
    -------
    >>> field = from_vtk("output/final.vtu")
    >>> field.to_stl("result.stl", resolution=100)
    """
    data = meshio.read(path)

    if field_name not in data.point_data:
        available = list(data.point_data.keys())
        raise KeyError(
            f"Point data '{field_name}' not found in {path}. "
            f"Available fields: {available}"
        )

    rho = data.point_data[field_name].ravel()
    points = data.points

    # Find supported cell block
    cells = None
    ele_type = None
    for block in data.cells:
        if block.type in _MESHIO_TO_FEAX:
            cells = block.data
            ele_type = _MESHIO_TO_FEAX[block.type]
            break

    if cells is None:
        cell_types = [block.type for block in data.cells]
        raise ValueError(
            f"No supported cell type found in {path}. "
            f"Available: {cell_types}. "
            f"Supported: {list(_MESHIO_TO_FEAX.keys())}"
        )

    from feax.mesh import Mesh
    mesh = Mesh(points, cells, ele_type=ele_type)

    return DensityField(rho, mesh, threshold=threshold)
