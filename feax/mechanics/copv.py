"""Mesh and model generation for double-wall composite pressure vessels (COPV).

A *double-wall* (sandwich) COPV is an axisymmetric tank whose wall is a radial
stack

.. code-block:: text

    [ inner CFRP shell ] [ solid fill ] [ outer CFRP shell ]

This module builds the full-ring structured ``HEX8`` mesh of the **axial half
model** (equator at ``z = 0``, one polar dome at the top) and the companion
data needed to drive a :mod:`feax.mechanics.layered_solid_element` analysis:

* :class:`DoubleWallCopvGeometry` — geometry + discretization parameters.
* :func:`create_double_wall_copv_mesh` — the structured mesh, radial *zone*
  labels (inner shell / fill / outer shell) and node-id bookkeeping, returned
  as a :class:`DoubleWallCopvMesh`.
* :func:`create_winding_ply_stiffness` — per-cell, per-ply stiffness with the
  Clairaut (geodesic) winding orientation on the shells and an isotropic fill.
* surface helpers (:meth:`DoubleWallCopvMesh.outer_surface_location_fn`,
  :meth:`DoubleWallCopvMesh.rim_location_fn`) and DoF helpers for the reference
  static solve / buckling boundary conditions.

The wall is described by a reference meridian on the **inner** surface (a
cylinder of radius ``a`` and length ``Lc`` capped by a spherical dome down to a
polar opening of radius ``r_p``); every radial layer is offset outward along
the local meridian normal, so the dome thickness follows the mould line.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as onp

import feax as fe

from .layered_solid_element import (
    _tabulate_reference, rotate_stiffness_3d, rotate_cte_3d,
)

# Radial zone labels (per cell).
ZONE_INNER_SHELL = 0
ZONE_FILL = 1
ZONE_OUTER_SHELL = 2


@dataclass(frozen=True)
class DoubleWallCopvGeometry:
    """Geometry and discretization of a double-wall (sandwich) COPV half model.

    Lengths are in consistent units (mm in the bundled examples). The reference
    surface is the **inner** mould line: a cylinder of radius ``a`` and length
    ``cyl_len`` above the equator, closed by a spherical dome down to a polar
    opening of radius ``polar_radius``.

    Parameters
    ----------
    a : float
        Inner-tank inner radius (the reference surface radius).
    shell_t : float
        Thickness of each CFRP shell (inner and outer are equal).
    gap : float
        Radial thickness of the solid fill between the two shells.
    cyl_len : float
        Cylinder length above the equator (the dome sits on top of it).
    polar_radius : float
        Polar-opening radius at the reference (inner) surface; sets the maximum
        winding latitude via Clairaut's relation.
    n_circum : int
        Circumferential divisions of the full ring (max wavenumber ~``n_circum/4``).
    n_merid_cyl, n_merid_dome : int
        Meridional element counts along the cylinder and the dome.
    n_fill : int
        Radial elements through the solid fill (each shell is a single radial
        element, so the wall has ``n_fill + 2`` radial elements total).
    axis : tuple of float
        Tank axis (unit vector); default global ``+z``.
    """

    a: float = 96.0
    shell_t: float = 4.0
    gap: float = 16.0
    cyl_len: float = 150.0
    polar_radius: float = 25.0
    n_circum: int = 42
    n_merid_cyl: int = 24
    n_merid_dome: int = 4
    n_fill: int = 4
    axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    @property
    def center_z(self) -> float:
        """Axial coordinate of the dome's sphere center (top of the cylinder)."""
        return self.cyl_len

    @property
    def d_layers(self) -> onp.ndarray:
        """Radial offsets (from the inner surface) of every layer interface.

        ``[0, t]`` (inner shell) + ``n_fill`` fill stations + ``t+gap+t`` (outer
        shell); length ``n_fill + 3`` → ``n_fill + 2`` radial elements.
        """
        t, gap, nfill = self.shell_t, self.gap, self.n_fill
        return onp.array([0.0, t]
                         + [t + gap * i / nfill for i in range(1, nfill + 1)]
                         + [t + gap + t], dtype=onp.float64)

    @property
    def n_layers(self) -> int:
        return self.n_fill + 3

    @property
    def n_rad_el(self) -> int:
        return self.n_fill + 2


@dataclass
class DoubleWallCopvMesh:
    """Result of :func:`create_double_wall_copv_mesh`.

    Bundles the :class:`feax.Mesh`, per-cell radial ``zone`` labels and the
    structured node-id bookkeeping used to assemble materials, surfaces and
    boundary conditions.

    Attributes
    ----------
    mesh : feax.Mesh
        Full-ring ``HEX8`` mesh of the half model.
    cell_zone : (n_cells,) int ndarray
        Radial zone of each cell: :data:`ZONE_INNER_SHELL`, :data:`ZONE_FILL`
        or :data:`ZONE_OUTER_SHELL`.
    geom : DoubleWallCopvGeometry
        The generating geometry.
    n_merid : int
        Number of meridional element rows (``= n_merid_cyl + n_merid_dome``).
    meridian : tuple of ndarray
        ``(r, z, n_r, n_z)`` reference-meridian stations and normals.
    """

    mesh: fe.Mesh
    cell_zone: onp.ndarray
    geom: DoubleWallCopvGeometry
    n_merid: int
    meridian: Tuple[onp.ndarray, onp.ndarray, onp.ndarray, onp.ndarray]

    # -- structured indexing --------------------------------------------------
    def node_id(self, i: int, j: int, kk: int) -> int:
        """Node id at meridian row ``i``, circumferential ``j`` (wraps), layer ``kk``."""
        nc, nl = self.geom.n_circum, self.geom.n_layers
        return (i * nc + (j % nc)) * nl + kk

    @property
    def points(self) -> onp.ndarray:
        return onp.asarray(self.mesh.points)

    @property
    def cells(self) -> onp.ndarray:
        return onp.asarray(self.mesh.cells)

    @property
    def cell_nodes(self) -> jnp.ndarray:
        """``(n_cells, 8, 3)`` physical node coordinates per cell."""
        return jnp.asarray(self.points[self.cells])

    # -- node-id sets ---------------------------------------------------------
    def equator_node_ids(self) -> onp.ndarray:
        """All nodes on the equator ring (``i = 0``), every column and layer."""
        nc, nl = self.geom.n_circum, self.geom.n_layers
        return onp.array([self.node_id(0, j, kk) for j in range(nc) for kk in range(nl)])

    def rim_inner_node_ids(self) -> onp.ndarray:
        """Inner two layers of the polar-rim ring (top meridian row ``i = n_merid``)."""
        nc = self.geom.n_circum
        return onp.array([self.node_id(self.n_merid, j, kk)
                          for j in range(nc) for kk in (0, 1)])

    def outer_surface_node_ids(self) -> onp.ndarray:
        """All nodes on the outermost mould line (last layer)."""
        nc, nl = self.geom.n_circum, self.geom.n_layers
        return onp.array([self.node_id(i, j, nl - 1)
                          for i in range(self.n_merid + 1) for j in range(nc)])

    def inner_surface_node_ids(self) -> onp.ndarray:
        """All nodes on the innermost mould line (first layer, the tank cavity)."""
        nc = self.geom.n_circum
        return onp.array([self.node_id(i, j, 0)
                          for i in range(self.n_merid + 1) for j in range(nc)])

    def rim_inner_radius(self) -> float:
        """Minimum cylindrical radius of the inner rim nodes (polar-boss radius)."""
        pts = self.points[self.rim_inner_node_ids()]
        return float(onp.linalg.norm(pts[:, :2], axis=1).min())

    # -- cell masks -----------------------------------------------------------
    def dome_cell_mask(self) -> onp.ndarray:
        """Boolean ``(n_cells,)`` mask: ``True`` for cells on the polar dome.

        Cells are emitted meridian-row-major (row ``i``, column ``j``, radial
        ``ke``), so a cell's meridian row is ``index // (n_circum · n_rad_el)``;
        rows ``>= n_merid_cyl`` are the dome.
        """
        nc, nre = self.geom.n_circum, self.geom.n_rad_el
        rows = onp.arange(self.cells.shape[0]) // (nc * nre)
        return rows >= self.geom.n_merid_cyl

    # -- boundary-condition DoFs ---------------------------------------------
    def equator_axial_dofs(self) -> onp.ndarray:
        """``u_z`` DoFs of the equator ring (symmetry-plane constraint)."""
        return self.equator_node_ids() * 3 + 2

    def rigid_inplane_pin_dofs(self) -> onp.ndarray:
        """Three DoFs that remove the in-plane rigid modes (x,y at A; y at B).

        ``A`` and ``B`` are two diametrically opposite equator inner-surface
        nodes; pinning ``A_x, A_y, B_y`` removes the two in-plane translations
        and the rotation about the tank axis.
        """
        nc = self.geom.n_circum
        a = self.node_id(0, 0, 0)
        b = self.node_id(0, nc // 2, 0)
        return onp.array([a * 3 + 0, a * 3 + 1, b * 3 + 1])

    def buckling_free_dofs(self, num_total_dofs: int) -> onp.ndarray:
        """Free DoFs for the buckling eigenproblem.

        Equator ``u_z = 0`` plus the three in-plane pins, removed from the full
        ``num_total_dofs`` set. The result is the ``free_dofs`` argument of
        :func:`feax.create_linear_buckling_solver`.
        """
        constrained = set(self.equator_axial_dofs().tolist())
        constrained.update(self.rigid_inplane_pin_dofs().tolist())
        return onp.array(sorted(set(range(num_total_dofs)) - constrained))

    # -- surface location functions ------------------------------------------
    def outer_surface_location_fn(self, tol: float = 1e-4) -> Callable:
        """``location_fn(point) -> bool`` selecting the outer mould-line surface."""
        coords = jnp.asarray(self.points[self.outer_surface_node_ids()])

        def fn(point):
            return jnp.min(jnp.linalg.norm(coords - point, axis=1)) < tol

        return fn

    def inner_surface_location_fn(self, tol: float = 1e-4) -> Callable:
        """``location_fn(point) -> bool`` selecting the inner (cavity) mould-line surface."""
        coords = jnp.asarray(self.points[self.inner_surface_node_ids()])

        def fn(point):
            return jnp.min(jnp.linalg.norm(coords - point, axis=1)) < tol

        return fn

    def rim_location_fn(self, tol: float = 1e-4) -> Callable:
        """``location_fn(point) -> bool`` selecting the inner polar-rim surface."""
        coords = jnp.asarray(self.points[self.rim_inner_node_ids()])

        def fn(point):
            return jnp.min(jnp.linalg.norm(coords - point, axis=1)) < tol

        return fn


def build_meridian(geom: DoubleWallCopvGeometry
                   ) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray, onp.ndarray]:
    """Reference (inner-surface) meridian: cylinder + spherical dome.

    Returns ``(r, z, n_r, n_z)`` arrays of length ``n_merid + 1`` giving the
    radius, axial coordinate and the (outward) meridian-normal components at
    each meridional station. Radial layers are offset along ``(n_r, n_z)``.
    """
    a, Lc, r_p = geom.a, geom.cyl_len, geom.polar_radius
    r_list, z_list, nr_list, nz_list = [], [], [], []
    for i in range(geom.n_merid_cyl + 1):                 # cylinder (normal = +r)
        r_list.append(a); z_list.append(Lc * i / geom.n_merid_cyl)
        nr_list.append(1.0); nz_list.append(0.0)
    lam_max = onp.arccos(r_p / a)                          # Clairaut latitude cap
    for i in range(1, geom.n_merid_dome + 1):             # spherical dome
        lam = lam_max * i / geom.n_merid_dome
        r_list.append(a * onp.cos(lam)); z_list.append(geom.center_z + a * onp.sin(lam))
        nr_list.append(onp.cos(lam)); nz_list.append(onp.sin(lam))
    return (onp.array(r_list), onp.array(z_list),
            onp.array(nr_list), onp.array(nz_list))


def create_double_wall_copv_mesh(geom: DoubleWallCopvGeometry = DoubleWallCopvGeometry(),
                                 check_jacobian: bool = True) -> DoubleWallCopvMesh:
    """Build the full-ring ``HEX8`` mesh of a double-wall COPV half model.

    Parameters
    ----------
    geom : DoubleWallCopvGeometry
        Geometry and discretization (defaults to the bundled example tank).
    check_jacobian : bool
        Assert all cell Jacobian determinants are positive (no inverted HEX8).

    Returns
    -------
    DoubleWallCopvMesh
        Mesh, per-cell radial zone labels and structured node-id helpers.
    """
    nc, nl = geom.n_circum, geom.n_layers
    n_rad_el = geom.n_rad_el
    inner_shell, outer_shell = 0, n_rad_el - 1
    d_layers = geom.d_layers
    r_ref, z_ref, n_r, n_z = build_meridian(geom)
    n_merid = r_ref.shape[0] - 1
    axis = onp.asarray(geom.axis, dtype=onp.float64)
    if not onp.allclose(axis, [0.0, 0.0, 1.0]):
        raise NotImplementedError(
            "mesh generation currently assumes the tank axis is global +z")

    def nid(i, j, kk):
        return (i * nc + (j % nc)) * nl + kk

    points = onp.zeros(((n_merid + 1) * nc * nl, 3))
    for i in range(n_merid + 1):
        for j in range(nc):
            theta = 2.0 * onp.pi * j / nc
            ct, st = onp.cos(theta), onp.sin(theta)
            for kk, d in enumerate(d_layers):
                r = r_ref[i] + d * n_r[i]
                z = z_ref[i] + d * n_z[i]
                points[nid(i, j, kk)] = [r * ct, r * st, z]

    cells, zone = [], []
    for i in range(n_merid):
        for j in range(nc):
            for ke in range(n_rad_el):
                cells.append([
                    nid(i, j, ke),     nid(i, j + 1, ke),     nid(i + 1, j + 1, ke),     nid(i + 1, j, ke),
                    nid(i, j, ke + 1), nid(i, j + 1, ke + 1), nid(i + 1, j + 1, ke + 1), nid(i + 1, j, ke + 1),
                ])
                zone.append(ZONE_INNER_SHELL if ke == inner_shell
                            else ZONE_OUTER_SHELL if ke == outer_shell else ZONE_FILL)

    mesh = fe.Mesh(jnp.asarray(points),
                   jnp.asarray(onp.array(cells, dtype=onp.int64)), "HEX8")

    if check_jacobian:
        dNdxi_c = onp.asarray(_tabulate_reference("HEX8", onp.array([[0.5, 0.5, 0.5]])))
        detJ = onp.linalg.det(onp.einsum("cai,qaI->cqiI", points[onp.array(cells)],
                                         dNdxi_c))[:, 0]
        if not onp.all(detJ > 0):
            raise ValueError(f"Negative HEX8 Jacobian (min {detJ.min():.3e}); "
                             "check the geometry / discretization.")

    return DoubleWallCopvMesh(mesh=mesh, cell_zone=onp.array(zone), geom=geom,
                              n_merid=n_merid, meridian=(r_ref, z_ref, n_r, n_z))


def _cell_triad(cell_nodes, axis):
    """Local orthonormal triad of a wall cell: ``(t_merid, t_hoop, e_normal, r)``.

    ``e_normal`` is the radial (stacking) direction, ``t_hoop`` the circumferential
    direction, ``t_merid`` the meridional direction; ``r`` is the cylindrical radius.
    """
    e3 = jnp.mean(cell_nodes[4:8], 0) - jnp.mean(cell_nodes[0:4], 0)
    e3 = e3 / jnp.linalg.norm(e3)
    c = jnp.mean(cell_nodes, 0)
    r_vec = c - (c @ axis) * axis
    e_r = r_vec / jnp.linalg.norm(r_vec)
    t_h = jnp.cross(axis, e_r); t_h = t_h - (t_h @ e3) * e3
    t_h = t_h / jnp.linalg.norm(t_h)
    t_m = jnp.cross(t_h, e3)
    return t_m, t_h, e3, jnp.linalg.norm(r_vec)


def _cell_ply_frames(cell_nodes, is_dome, axis, r_p, layup, drop_dome_hoop):
    """Per-ply ply-axis frames ``R = [f1, f2, e3]`` (columns) for one cell.

    Single source of truth for the winding orientation used by both
    :func:`create_winding_ply_stiffness` and :func:`cell_fiber_directions`.
    Returns ``(n_ply, 3, 3)`` rotation matrices; column 0 is the primary fiber
    direction ``f1``, column 2 the stacking normal ``e3``.
    """
    t_m, t_h, e3, r_cyl = _cell_triad(cell_nodes, axis)
    alpha = jnp.arcsin(jnp.clip(r_p / r_cyl, 0.0, 1.0))

    def helical_axes(ang):
        f1 = jnp.cos(ang) * t_m + jnp.sin(ang) * t_h
        f2 = -jnp.sin(ang) * t_m + jnp.cos(ang) * t_h
        return f1, f2

    frames = []
    hoop_count = 0
    for kind, sign in layup:
        if kind == "hoop":
            f1, f2 = t_h, -t_m                            # cylinder: circumferential
            if drop_dome_hoop:                            # dome: helical continuation
                hsign = 1.0 if hoop_count % 2 == 0 else -1.0
                f1d, f2d = helical_axes(hsign * alpha)
                f1 = jnp.where(is_dome, f1d, f1)
                f2 = jnp.where(is_dome, f2d, f2)
            hoop_count += 1
        else:
            f1, f2 = helical_axes(sign * alpha)
        frames.append(jnp.stack([f1, f2, e3], axis=1))
    return jnp.stack(frames, 0)


def create_winding_ply_stiffness(copv_mesh: DoubleWallCopvMesh,
                                 ply_C: jnp.ndarray,
                                 layup: Sequence[Tuple[str, float]],
                                 fill_C: jnp.ndarray = None,
                                 drop_dome_hoop: bool = True) -> jnp.ndarray:
    """Per-cell, per-ply stiffness with Clairaut winding on the shells.

    For every shell cell each ply is rotated into the wall triad: a ``"hoop"``
    ply aligns with the circumferential direction, a ``"helical"`` ply is wound
    at the geodesic angle ``α = arcsin(r_p / r)`` (Clairaut's relation) about the
    meridian, with ``sign`` giving the ``±`` winding. Fill cells (if ``fill_C``
    is given) take the isotropic ``fill_C`` for every ply.

    Hoop windings physically exist only on the cylinder (real filament winding
    terminates the hoop band at the cylinder–dome tangent line). With
    ``drop_dome_hoop=True`` (default), ``"hoop"`` plies on the **dome** are
    therefore re-oriented as helical continuations at the local geodesic angle
    instead of circumferentially — the ply count and wall thickness are fixed by
    the mesh, so the hoop reinforcement is removed without changing the geometry.
    Successive hoop plies alternate ``±α`` to keep the dome laminate balanced.

    Parameters
    ----------
    copv_mesh : DoubleWallCopvMesh
        The generated mesh (provides cell geometry and zone labels).
    ply_C : (3,3,3,3) array
        Single-ply (unidirectional) stiffness in ply axes.
    layup : sequence of ``(kind, sign)``
        Per-ply stacking, ``kind`` in ``{"helical", "hoop"}`` and ``sign = ±1``
        the winding orientation (ignored for hoop).
    fill_C : (3,3,3,3) array, optional
        Isotropic fill stiffness. If ``None`` every cell is treated as a shell.
    drop_dome_hoop : bool
        Re-orient hoop plies on the dome as helical continuations (default
        ``True``). Set ``False`` to keep hoop windings circumferential everywhere.

    Returns
    -------
    C_cell_ply : (n_cells, n_ply, 3, 3, 3, 3) array
        Rotated stiffness per cell and ply, ready for
        :func:`feax.mechanics.create_oriented_layered_solid`.
    """
    geom = copv_mesh.geom
    axis = jnp.asarray(geom.axis)
    r_p = geom.polar_radius
    n_ply = len(layup)

    def cell_shell(cell_nodes, is_dome):
        frames = _cell_ply_frames(cell_nodes, is_dome, axis, r_p, layup, drop_dome_hoop)
        return jax.vmap(lambda R: rotate_stiffness_3d(ply_C, R))(frames)

    is_dome = jnp.asarray(copv_mesh.dome_cell_mask())
    C_shell = jax.vmap(cell_shell)(copv_mesh.cell_nodes, is_dome)
    if fill_C is None:
        return C_shell
    C_fill_plies = jnp.broadcast_to(fill_C, (n_ply, 3, 3, 3, 3))
    is_shell = jnp.asarray(copv_mesh.cell_zone != ZONE_FILL)[:, None, None, None, None, None]
    return jnp.where(is_shell, C_shell, C_fill_plies[None])


def create_winding_cte(copv_mesh: DoubleWallCopvMesh,
                       ply_alpha: jnp.ndarray,
                       layup: Sequence[Tuple[str, float]],
                       fill_alpha: jnp.ndarray = None,
                       drop_dome_hoop: bool = True) -> jnp.ndarray:
    """Per-cell, per-ply thermal-expansion (CTE) tensor with Clairaut winding.

    Thermal analogue of :func:`create_winding_ply_stiffness`: each shell ply's CTE
    is rotated into the **same** wall triad / winding frame used for the stiffness,
    so the anisotropic contraction (small ``α₁`` along the fibre, large ``α₂``
    transverse) follows the winding direction. Fill cells (if ``fill_alpha`` is
    given) take the isotropic ``fill_alpha`` for every ply.

    The result feeds the layered-solid thermal eigenstrain: expand it to quadrature
    points and multiply by the temperature change ``ΔT`` to get ``ε_th = α·ΔT``
    (see :func:`feax.mechanics.expand_cte_to_quad`).

    Parameters
    ----------
    copv_mesh : DoubleWallCopvMesh
        The generated mesh (provides cell geometry and zone labels).
    ply_alpha : (3, 3) array
        Single-ply CTE tensor in ply (material) axes, fibre = 1-axis, e.g.
        ``transverse_isotropic_cte_3d(alpha_1, alpha_2)``.
    layup : sequence of ``(kind, sign)``
        Per-ply stacking, identical to :func:`create_winding_ply_stiffness` so the
        CTE and stiffness frames stay in lock-step.
    fill_alpha : (3, 3) array, optional
        Isotropic fill CTE. If ``None`` every cell is treated as a shell.
    drop_dome_hoop : bool
        Re-orient hoop plies on the dome as helical continuations (default
        ``True``); must match the value used for the stiffness.

    Returns
    -------
    cte_cell_ply : (n_cells, n_ply, 3, 3) array
        Rotated CTE per cell and ply, in **global** axes.
    """
    geom = copv_mesh.geom
    axis = jnp.asarray(geom.axis)
    r_p = geom.polar_radius
    n_ply = len(layup)

    def cell_shell(cell_nodes, is_dome):
        frames = _cell_ply_frames(cell_nodes, is_dome, axis, r_p, layup, drop_dome_hoop)
        return jax.vmap(lambda R: rotate_cte_3d(ply_alpha, R))(frames)

    is_dome = jnp.asarray(copv_mesh.dome_cell_mask())
    A_shell = jax.vmap(cell_shell)(copv_mesh.cell_nodes, is_dome)
    if fill_alpha is None:
        return A_shell
    A_fill_plies = jnp.broadcast_to(fill_alpha, (n_ply, 3, 3))
    is_shell = jnp.asarray(copv_mesh.cell_zone != ZONE_FILL)[:, None, None, None]
    return jnp.where(is_shell, A_shell, A_fill_plies[None])


def cell_fiber_directions(copv_mesh: DoubleWallCopvMesh,
                          layup: Sequence[Tuple[str, float]],
                          drop_dome_hoop: bool = True,
                          shell_only: bool = True) -> onp.ndarray:
    """Per-cell, per-ply primary fiber direction in **global** coordinates.

    Uses the same orientation as :func:`create_winding_ply_stiffness` (so the
    vectors visualise the laminate actually assembled). Fill cells carry no
    fibers and are zeroed when ``shell_only`` (the default).

    Returns
    -------
    fiber : (n_cells, n_ply, 3) ndarray
        Unit fiber direction ``f1`` of each ply, per cell (zero on fill cells).
    """
    geom = copv_mesh.geom
    axis = jnp.asarray(geom.axis)
    is_dome = jnp.asarray(copv_mesh.dome_cell_mask())

    def cell_f1(cell_nodes, dome):
        frames = _cell_ply_frames(cell_nodes, dome, axis, geom.polar_radius,
                                  layup, drop_dome_hoop)
        return frames[:, :, 0]                            # f1 = column 0, (n_ply, 3)

    fiber = onp.asarray(jax.vmap(cell_f1)(copv_mesh.cell_nodes, is_dome))
    if shell_only:
        fiber = fiber * (copv_mesh.cell_zone != ZONE_FILL)[:, None, None]
    return fiber


def cell_winding_angle_deg(copv_mesh: DoubleWallCopvMesh) -> onp.ndarray:
    """Per-cell helical winding angle ``α = arcsin(r_p / r)`` in degrees (Clairaut)."""
    axis = jnp.asarray(copv_mesh.geom.axis)
    r_cyl = jax.vmap(lambda c: _cell_triad(c, axis)[3])(copv_mesh.cell_nodes)
    return onp.degrees(onp.arcsin(onp.clip(copv_mesh.geom.polar_radius
                                           / onp.asarray(r_cyl), 0.0, 1.0)))


def cell_triads(copv_mesh: DoubleWallCopvMesh
                ) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """Per-cell wall triad ``(t_meridian, t_hoop, e_normal)``, each ``(n_cells, 3)``.

    The same geometric directions used for the winding orientation — handy for
    projecting per-cell stresses (e.g. the hoop prestress ``t_hoop·σ·t_hoop``).
    """
    axis = jnp.asarray(copv_mesh.geom.axis)
    tri = jax.vmap(lambda c: _cell_triad(c, axis))(copv_mesh.cell_nodes)
    return onp.asarray(tri[0]), onp.asarray(tri[1]), onp.asarray(tri[2])


def save_orientation_vtk(copv_mesh: DoubleWallCopvMesh,
                         sol_file: str,
                         layup: Sequence[Tuple[str, float]],
                         drop_dome_hoop: bool = True) -> None:
    """Write a VTU visualising the fiber orientation of the COPV laminate.

    The output carries, as **cell** data (view as glyphs in ParaView):

    * ``fiber_ply{k}`` — primary fiber direction of ply ``k`` (global coords);
    * ``dir_meridian`` / ``dir_hoop`` / ``dir_normal`` — the wall triad;
    * ``winding_angle_deg`` — the helical Clairaut angle;
    * ``zone`` (0/1/2 inner-shell/fill/outer-shell) and ``is_dome`` (0/1).

    In ParaView: open the ``.vtu``, ``Glyph`` filter, orient by e.g.
    ``fiber_ply0`` to see the winding (steepening over the dome, hoop on the
    cylinder), and colour by ``winding_angle_deg`` or ``zone``.

    Parameters
    ----------
    copv_mesh : DoubleWallCopvMesh
    sol_file : str
        Output path (``.vtu``).
    layup, drop_dome_hoop
        As in :func:`create_winding_ply_stiffness`.
    """
    axis = jnp.asarray(copv_mesh.geom.axis)
    triad = jax.vmap(lambda c: _cell_triad(c, axis))(copv_mesh.cell_nodes)
    t_m, t_h, e3 = (onp.asarray(triad[0]), onp.asarray(triad[1]), onp.asarray(triad[2]))
    fiber = cell_fiber_directions(copv_mesh, layup, drop_dome_hoop=drop_dome_hoop)

    cell_infos = [(f"fiber_ply{k}", fiber[:, k, :]) for k in range(len(layup))]
    cell_infos += [
        ("dir_meridian", t_m), ("dir_hoop", t_h), ("dir_normal", e3),
        ("winding_angle_deg", cell_winding_angle_deg(copv_mesh)),
        ("zone", copv_mesh.cell_zone.astype(onp.float64)),
        ("is_dome", copv_mesh.dome_cell_mask().astype(onp.float64)),
    ]
    fe.utils.save_sol(mesh=copv_mesh.mesh, sol_file=sol_file, cell_infos=cell_infos)


__all__ = [
    "ZONE_INNER_SHELL",
    "ZONE_FILL",
    "ZONE_OUTER_SHELL",
    "DoubleWallCopvGeometry",
    "DoubleWallCopvMesh",
    "build_meridian",
    "create_double_wall_copv_mesh",
    "create_winding_ply_stiffness",
    "create_winding_cte",
    "cell_fiber_directions",
    "cell_winding_angle_deg",
    "cell_triads",
    "save_orientation_vtk",
]
