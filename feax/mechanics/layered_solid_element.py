"""Layered solid (3D) element with exact per-ply integration.

This module provides a small-strain, linear, fully-anisotropic 3D continuum
element for laminated composites.  An arbitrary number of plies can live
inside a *single* element: each ply is integrated with its own Gauss rule so
the through-thickness integral

    K = Σ_k ∫_{z_k}^{z_{k+1}} Bᵀ C_k B dz

is evaluated **exactly** regardless of ply count.  The implementation lives
in ``get_universal_kernel`` — feax handles global assembly and autodiff
linearisation.

Scope
-----
* small-strain linear elasticity, ``σ_ij = C_ijkl ε_kl``
* full 3D anisotropic stiffness, arbitrary number of plies per element
* ply stacking along an arbitrary (default global-z) normal, each ply
  rotated about that normal by its own angle

Not included (see :mod:`feax.mechanics.shell` for the FSDT plate):

* no ANS / EAS — thin laminates with HEX8 still exhibit shear / thickness
  locking in the in-plane directions.
* no geometric nonlinearity.

Kinematics & constitutive law
------------------------------

.. code-block:: text

    ε   = sym(∇u)              (3×3 small strain)
    σ_k = C_k : (ε − ε_th,k)   (per-ply stress; ε_th = α·ΔT, optional thermal
                                eigenstrain — zero unless thermal coupling is on)
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as np
import numpy as onp
import basix
import jax.flatten_util

import feax as fe
from feax.basis import get_elements
from feax.internal_vars import InternalVars


# ---------------------------------------------------------------------------
# Voigt <-> 4th-order tensor conversion (3D)
# ---------------------------------------------------------------------------

# Voigt index map: (i, j) -> Voigt slot, with
# 11->0, 22->1, 33->2, 23->3, 13->4, 12->5.
_VOIGT = onp.array([[0, 5, 4],
                    [5, 1, 3],
                    [4, 3, 2]])
# Representative (i, j) pair for each Voigt slot.
_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def voigt_to_tensor(C6: np.ndarray) -> np.ndarray:
    """Expand a ``(6, 6)`` Voigt stiffness to the ``(3,3,3,3)`` tensor.

    Uses the convention ``σ_ij = C_ijkl ε_kl`` (tensor strains) together
    with engineering-shear Voigt order ``(11, 22, 33, 23, 13, 12)``. The
    expansion ``C_ijkl = C6[v(i,j), v(k,l)]`` reproduces both the
    engineering shear factors and the minor symmetries automatically.
    """
    idx = np.asarray(_VOIGT)
    return C6[idx[:, :, None, None], idx[None, None, :, :]]


def tensor_to_voigt(C: np.ndarray) -> np.ndarray:
    """Contract a ``(3,3,3,3)`` stiffness tensor to its ``(6, 6)`` Voigt form."""
    rows = []
    for (i, j) in _VOIGT_PAIRS:
        rows.append(np.array([C[i, j, k, l] for (k, l) in _VOIGT_PAIRS]))
    return np.stack(rows, axis=0)


# ---------------------------------------------------------------------------
# 3D stiffness builders (material axes; fibre direction = local 1-axis)
# ---------------------------------------------------------------------------

def isotropic_stiffness_3d(E: float, nu: float) -> np.ndarray:
    """Isotropic 3D stiffness tensor ``C_ijkl`` of shape ``(3,3,3,3)``.

    .. code-block:: text

        σ_ij = λ δ_ij tr(ε) + 2 μ ε_ij
        λ = E ν / ((1+ν)(1-2ν)),   μ = E / (2(1+ν))
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    delta = np.eye(3)
    return (
        lam * np.einsum("ij,kl->ijkl", delta, delta)
        + mu * (
            np.einsum("ik,jl->ijkl", delta, delta)
            + np.einsum("il,jk->ijkl", delta, delta)
        )
    )


def orthotropic_stiffness_3d(
    E1: float, E2: float, E3: float,
    G12: float, G13: float, G23: float,
    nu12: float, nu13: float, nu23: float,
) -> np.ndarray:
    """Orthotropic 3D stiffness tensor in *material* axes (fibre = 1-axis).

    Built by inverting the engineering compliance matrix

    .. code-block:: text

        S = [[ 1/E1,    -ν12/E1, -ν13/E1, 0,     0,     0    ],
             [-ν12/E1,   1/E2,   -ν23/E2, 0,     0,     0    ],
             [-ν13/E1,  -ν23/E2,  1/E3,   0,     0,     0    ],
             [ 0,        0,        0,      1/G23, 0,     0    ],
             [ 0,        0,        0,      0,     1/G13, 0    ],
             [ 0,        0,        0,      0,     0,     1/G12]]

    (symmetric, using ``ν21/E2 = ν12/E1`` etc.) with Voigt shear order
    ``(23, 13, 12)``, then expanded to the 4th-order tensor.
    """
    S = onp.zeros((6, 6))
    S[0, 0] = 1.0 / E1
    S[1, 1] = 1.0 / E2
    S[2, 2] = 1.0 / E3
    S[0, 1] = S[1, 0] = -nu12 / E1
    S[0, 2] = S[2, 0] = -nu13 / E1
    S[1, 2] = S[2, 1] = -nu23 / E2
    S[3, 3] = 1.0 / G23
    S[4, 4] = 1.0 / G13
    S[5, 5] = 1.0 / G12
    C6 = onp.linalg.inv(S)
    return voigt_to_tensor(np.asarray(C6))


def transverse_isotropic_stiffness_3d(
    E1: float, E2: float, G12: float, nu12: float, nu23: float,
) -> np.ndarray:
    """Transversely-isotropic 3D stiffness for a unidirectional lamina.

    The fibre direction is the 1-axis and the 2-3 plane is isotropic, so

    .. code-block:: text

        E3  = E2,   G13 = G12,   ν13 = ν12,
        G23 = E2 / (2 (1 + ν23)).

    A convenience wrapper around :func:`orthotropic_stiffness_3d` for the
    common single-ply UD-composite case.
    """
    G23 = E2 / (2.0 * (1.0 + nu23))
    return orthotropic_stiffness_3d(
        E1=E1, E2=E2, E3=E2,
        G12=G12, G13=G12, G23=G23,
        nu12=nu12, nu13=nu12, nu23=nu23,
    )


# ---------------------------------------------------------------------------
# Thermal-expansion (CTE) tensors — material axes (fibre = local 1-axis)
# ---------------------------------------------------------------------------
# The CTE α is a symmetric 2nd-order tensor; the thermal eigenstrain at a quad
# point is ``ε_th = α · ΔT`` and the constitutive law becomes
# ``σ = C : (ε − ε_th)``. These builders return α in *material* axes; rotate
# them into global axes with :func:`rotate_cte_3d` (same frame as the stiffness).

def isotropic_cte_3d(alpha: float) -> np.ndarray:
    """Isotropic thermal-expansion tensor ``α_ij = α δ_ij`` of shape ``(3, 3)``."""
    return alpha * np.eye(3)


def transverse_isotropic_cte_3d(alpha_1: float, alpha_2: float) -> np.ndarray:
    """CTE tensor of a UD lamina in material axes (fibre = 1-axis).

    Transverse isotropy in the 2-3 plane gives ``α = diag(α₁, α₂, α₂)``. For CFRP
    the fibre-direction ``α₁`` is small (often slightly negative) while the
    transverse ``α₂`` is large — the source of large cool-down eigenstrains and
    inter-ply stresses in cryogenic laminates.
    """
    return np.diag(np.array([alpha_1, alpha_2, alpha_2], dtype=np.float64))


# ---------------------------------------------------------------------------
# Rotation of a 4th-order stiffness tensor
# ---------------------------------------------------------------------------

def rotation_matrix_axis(theta: float, axis: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Right-handed rotation matrix of angle ``theta`` (rad) about ``axis``.

    Uses Rodrigues' formula ``R = I + sinθ K + (1−cosθ) K²`` with ``K`` the
    skew-symmetric matrix of the unit ``axis``.
    """
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    K = np.array([
        [0.0, -a[2], a[1]],
        [a[2], 0.0, -a[0]],
        [-a[1], a[0], 0.0],
    ])
    c, s = np.cos(theta), np.sin(theta)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def rotate_stiffness_3d(C: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate a ``(3,3,3,3)`` stiffness tensor by rotation matrix ``R``.

    .. math:: \\bar C_{ijkl} = R_{ip} R_{jq} R_{kr} R_{ls} C_{pqrs}.
    """
    return np.einsum("ip,jq,kr,ls,pqrs->ijkl", R, R, R, R, C)


def rotate_cte_3d(alpha: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate a ``(3, 3)`` symmetric CTE tensor by rotation matrix ``R``.

    .. math:: \\bar\\alpha_{ij} = R_{ip} R_{jq} \\alpha_{pq}.

    Use the **same** ``R`` as :func:`rotate_stiffness_3d` so the ply's stiffness
    and thermal expansion are expressed in a consistent (global) frame.
    """
    return np.einsum("ip,jq,pq->ij", R, R, alpha)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _broadcast_plies(ply_C: np.ndarray, n_ply: int) -> np.ndarray:
    """Return a ``(n_ply, 3,3,3,3)`` per-ply stiffness array."""
    ply_C = np.asarray(ply_C)
    if ply_C.ndim == 4:
        ply_C = np.broadcast_to(ply_C, (n_ply, 3, 3, 3, 3))
    elif ply_C.shape[0] != n_ply:
        raise ValueError(
            f"ply_C has {ply_C.shape[0]} plies but {n_ply} angles were given"
        )
    return ply_C


def _gauss1d_unit(n: int) -> Tuple[onp.ndarray, onp.ndarray]:
    """Gauss–Legendre points/weights on the unit interval ``[0, 1]``."""
    x, w = onp.polynomial.legendre.leggauss(n)  # on [-1, 1], Σw = 2
    return 0.5 * (x + 1.0), 0.5 * w             # on [0, 1], Σw = 1


def layered_reference_quadrature(
    ply_thicknesses: Sequence[float],
    n_inplane: int = 2,
    n_thick_per_ply: int = 2,
) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """Per-ply through-thickness quadrature on the reference hex ``[0,1]³``.

    The in-plane directions (ξ, η) use an ``n_inplane`` Gauss rule; the
    thickness direction (ζ) is split at the ply interfaces and each ply gets
    its own ``n_thick_per_ply`` Gauss rule with weights scaled by the ply's
    ζ-extent. The total weight sums to 1 (the reference-cell volume), matching
    basix's convention.

    Returns
    -------
    points : (nq, 3) array       Reference quadrature points in ``[0,1]³``.
    weights : (nq,) array        Quadrature weights (Σ = 1).
    ply_of_quad : (nq,) int array Ply index owning each quadrature point.
    """
    thick = onp.asarray(ply_thicknesses, dtype=onp.float64)
    n_ply = thick.shape[0]
    zeta_if = onp.concatenate([[0.0], onp.cumsum(thick) / onp.sum(thick)])

    xi, wxi = _gauss1d_unit(n_inplane)
    eta, weta = _gauss1d_unit(n_inplane)
    gz, wgz = _gauss1d_unit(n_thick_per_ply)

    pts, wts, ply_of = [], [], []
    for k in range(n_ply):
        a, b = zeta_if[k], zeta_if[k + 1]
        z_k = a + (b - a) * gz          # map ply Gauss points into [a, b]
        w_k = (b - a) * wgz             # scale weights by ply ζ-extent
        for iz in range(z_k.shape[0]):
            for ix in range(xi.shape[0]):
                for iy in range(eta.shape[0]):
                    pts.append([xi[ix], eta[iy], z_k[iz]])
                    wts.append(wxi[ix] * weta[iy] * w_k[iz])
                    ply_of.append(k)
    return (onp.asarray(pts), onp.asarray(wts),
            onp.asarray(ply_of, dtype=onp.int64))


def _tabulate_reference(ele_type: str, ref_points: onp.ndarray) -> np.ndarray:
    """Reference-space shape-function gradients ``dN_a/dξ_I`` at ``ref_points``.

    Mirrors :func:`feax.basis.get_shape_vals_and_grads` but tabulates at the
    given points (in feax node ordering). Returns ``(nq, num_nodes, dim)``.
    """
    family, basix_ele, _, _, degree, re_order = get_elements(ele_type)
    element = basix.create_element(family, basix_ele, degree)
    pts = onp.ascontiguousarray(ref_points, dtype=onp.float64)
    tab = element.tabulate(1, pts)[:, :, re_order, :]   # (deriv, q, node, val)
    dNdxi = onp.transpose(tab[1:, :, :, 0], axes=(1, 2, 0))  # (q, node, dim)
    return np.asarray(dNdxi)


def _tabulate_reference_vals(ele_type: str, ref_points: onp.ndarray) -> np.ndarray:
    """Reference-space shape-function **values** ``N_a`` at ``ref_points``.

    Companion to :func:`_tabulate_reference` (which returns gradients). Returns
    ``(nq, num_nodes)`` in feax node ordering.
    """
    family, basix_ele, _, _, degree, re_order = get_elements(ele_type)
    element = basix.create_element(family, basix_ele, degree)
    pts = onp.ascontiguousarray(ref_points, dtype=onp.float64)
    tab = element.tabulate(0, pts)[:, :, re_order, :]   # (1, q, node, val)
    return np.asarray(tab[0, :, :, 0])                  # (q, node)


def interpolate_nodal_to_layered_quad(
    nodal_cells: np.ndarray,
    ply_thicknesses: Sequence[float],
    *,
    ele_type: str = "HEX8",
    n_inplane: int = 2,
    n_thick_per_ply: int = 2,
) -> np.ndarray:
    """Interpolate a per-cell nodal field to the layered-solid quadrature points.

    Maps a scalar nodal field gathered per cell — e.g. a solved temperature
    ``T[cells]`` of shape ``(num_cells, n_nodes)`` — onto the **same** per-ply
    through-thickness quadrature as :func:`create_oriented_layered_solid`, giving
    ``(num_cells, nq)``. Combine with :func:`expand_cte_to_quad` to form the
    thermal eigenstrain at each quad point::

        T_quad   = interpolate_nodal_to_layered_quad(T[cells], ply_thicknesses)
        cte_quad = expand_cte_to_quad(cte_cell_ply, ply_thicknesses)
        eps_th   = cte_quad * (T_quad - T_ref)[..., None, None]
    """
    thick = onp.asarray(ply_thicknesses, dtype=onp.float64)
    ref_pts, _, _ = layered_reference_quadrature(
        thick, n_inplane=n_inplane, n_thick_per_ply=n_thick_per_ply)
    N = _tabulate_reference_vals(ele_type, ref_pts)     # (nq, n_nodes)
    return np.einsum("qa,ca->cq", N, np.asarray(nodal_cells))


class LayeredSolid(fe.Problem):
    """3D composite solid with exact per-ply (layer-wise) through-thickness
    integration — many plies inside a single element.

    Displacement-only continuum element (``vec=3``, ``dim=3``) implemented as
    a custom element via :meth:`get_universal_kernel`. Build it with
    :func:`create_layered_solid`, which precomputes the per-ply quadrature and
    rotated stiffness and supplies per-cell node coordinates as the single
    volume internal variable.

    ``additional_info = (dNdxi_ref, weights, C_quad)`` where

    * ``dNdxi_ref`` : ``(nq, num_nodes, 3)`` reference shape gradients,
    * ``weights``   : ``(nq,)`` reference quadrature weights,
    * ``C_quad``    : ``(nq, 3,3,3,3)`` rotated stiffness at each quad point.
    """

    def custom_init(self, dNdxi_ref, weights, C_quad):
        self._dNdxi_ref = dNdxi_ref
        self._weights = weights
        self._C_quad = C_quad

    def get_universal_kernel(self) -> Callable:
        dNdxi = self._dNdxi_ref      # (nq, n_nodes, 3)
        w = self._weights            # (nq,)
        C_quad = self._C_quad        # (nq, 3,3,3,3)
        unflatten = self.unflatten_fn_dof

        def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads,
                   cell_JxW, cell_v_grads_JxW, cell_nodes):
            # cell_nodes: (n_nodes, 3) physical coords (per-cell internal var)
            cell_sol = unflatten(cell_sol_flat)[0]          # (n_nodes, vec)

            # Isoparametric Jacobian J_iI = Σ_a x_ai dN_a/dξ_I  at each quad pt
            J = np.einsum("ai,qaI->qiI", cell_nodes, dNdxi)  # (nq, 3, 3)
            Jinv = np.linalg.inv(J)
            detJ = np.linalg.det(J)
            # Physical shape gradients dN_a/dx_i = dN_a/dξ_I (J⁻¹)_Ii
            dNdx = np.einsum("qaI,qIi->qai", dNdxi, Jinv)    # (nq, n_nodes, 3)

            # Small strain and stress at each quad point
            grad_u = np.einsum("ai,qaj->qij", cell_sol, dNdx)   # (nq, 3, 3)
            eps = 0.5 * (grad_u + np.transpose(grad_u, (0, 2, 1)))
            sigma = np.einsum("qijkl,qkl->qij", C_quad, eps)    # (nq, 3, 3)

            # Internal-force residual R_ai = Σ_q (w·detJ) σ_ij dN_a/dx_j
            JxW = w * detJ                                       # (nq,)
            R = np.einsum("q,qij,qaj->ai", JxW, sigma, dNdx)    # (n_nodes, vec)
            return jax.flatten_util.ravel_pytree(R)[0]

        return kernel


def create_layered_solid(
    mesh,
    ply_C: np.ndarray,
    ply_angles: Sequence[float],
    ply_thicknesses: Sequence[float],
    *,
    ele_type: str = "HEX8",
    n_inplane: int = 2,
    n_thick_per_ply: int = 2,
    location_fns: Optional[Iterable[Callable]] = None,
    normal: Sequence[float] = (0.0, 0.0, 1.0),
) -> Tuple[LayeredSolid, InternalVars]:
    """Build a :class:`LayeredSolid` with exact per-ply through-thickness
    integration (many plies in one element).

    One element is assumed to span the whole laminate thickness, with the
    laminate stacked along the element's local ζ-axis (the last spatial axis
    for a structured :func:`feax.mesh.box_mesh`). Each ply is integrated with
    its own ``n_thick_per_ply`` Gauss rule, so an *n*-ply laminate is captured
    exactly by a single layer of elements regardless of ply count.

    Parameters
    ----------
    mesh : feax.Mesh
        3D mesh, one element through the laminate thickness.
    ply_C : array
        Lamina stiffness in material axes, ``(3,3,3,3)`` (broadcast) or
        ``(n_ply, 3,3,3,3)``.
    ply_angles : sequence of float
        Per-ply rotation angles (radians) about ``normal``, bottom → top.
    ply_thicknesses : sequence of float
        Per-ply thicknesses, bottom → top.
    ele_type : str
        Element type (only ``"HEX8"`` is supported).
    n_inplane : int
        In-plane Gauss points per direction (default 2).
    n_thick_per_ply : int
        Through-thickness Gauss points *per ply* (default 2).
    location_fns : iterable of callables, optional
        Boundary-location predicates for surface integrals.
    normal : sequence of float
        Physical stacking direction used to rotate each ply (default ``z``).

    Returns
    -------
    problem : LayeredSolid
    internal_vars : feax.InternalVars
        Holds per-cell node coordinates ``(num_cells, num_nodes, 3)``.
    """
    if ele_type != "HEX8":
        raise NotImplementedError("create_layered_solid currently supports HEX8 only")

    angles = onp.asarray(ply_angles, dtype=onp.float64)
    thick = onp.asarray(ply_thicknesses, dtype=onp.float64)
    n_ply = angles.shape[0]
    if thick.shape[0] != n_ply:
        raise ValueError("ply_angles and ply_thicknesses must have equal length")

    # Per-ply rotated stiffness (material axes -> laminate axes about normal).
    ply_C = _broadcast_plies(ply_C, n_ply)
    nrm = np.asarray(normal, dtype=np.float64)
    nrm = nrm / np.linalg.norm(nrm)
    C_rot = jax.vmap(
        lambda C_k, th: rotate_stiffness_3d(C_k, rotation_matrix_axis(th, nrm))
    )(ply_C, np.asarray(angles))                            # (n_ply,3,3,3,3)

    # Per-ply reference quadrature + shape gradients + per-quad stiffness.
    ref_pts, weights, ply_of = layered_reference_quadrature(
        thick, n_inplane=n_inplane, n_thick_per_ply=n_thick_per_ply)
    dNdxi = _tabulate_reference(ele_type, ref_pts)          # (nq, n_nodes, 3)
    C_quad = C_rot[np.asarray(ply_of)]                      # (nq, 3,3,3,3)

    location_fns = tuple(location_fns) if location_fns is not None else None
    problem = LayeredSolid(
        mesh, vec=3, dim=3, ele_type=ele_type, location_fns=location_fns,
        additional_info=(dNdxi, np.asarray(weights), C_quad),
    )
    cell_nodes = np.asarray(mesh.points)[np.asarray(mesh.cells)]
    internal_vars = InternalVars(volume_vars=(cell_nodes,))
    return problem, internal_vars


class OrientedLayeredSolid(fe.Problem):
    """Layered composite solid whose per-ply stiffness varies **from element to
    element** — for curved laminates where the stacking normal and the fibre
    orientation follow the geometry (e.g. a filament-wound pressure vessel).

    Identical in spirit to :class:`LayeredSolid` (exact per-ply through-thickness
    integration, many plies in one element), but the rotated stiffness at each
    quadrature point is supplied **per cell** as a volume internal variable rather
    than shared across the mesh. Build it with :func:`create_oriented_layered_solid`.

    Volume internal variables (in order): ``(cell_nodes, C_quad)`` — or, when the
    element is built with ``with_thermal=True``, ``(cell_nodes, C_quad, eps_th)`` —
    where

    * ``cell_nodes`` : ``(num_cells, num_nodes, 3)`` physical node coordinates,
    * ``C_quad``     : ``(num_cells, nq, 3,3,3,3)`` rotated stiffness per quad point,
    * ``eps_th``     : ``(num_cells, nq, 3, 3)`` thermal eigenstrain per quad point
      (``= α·ΔT`` in global axes). Only present when ``with_thermal=True``; the
      stress is then ``σ = C : (ε − ε_th)``, which adds a constant cool-down
      thermal load to the residual.

    ``additional_info = (dNdxi_ref, weights, with_thermal)`` carries the reference
    shape gradients ``(nq, num_nodes, 3)``, quadrature weights ``(nq,)`` and the
    thermal-coupling flag.
    """

    def custom_init(self, dNdxi_ref, weights, with_thermal=False):
        self._dNdxi_ref = dNdxi_ref
        self._weights = weights
        self._with_thermal = with_thermal

    def get_universal_kernel(self) -> Callable:
        dNdxi = self._dNdxi_ref      # (nq, n_nodes, 3)
        w = self._weights            # (nq,)
        unflatten = self.unflatten_fn_dof

        def _kinematics(cell_sol_flat, cell_nodes):
            cell_sol = unflatten(cell_sol_flat)[0]          # (n_nodes, vec)
            J = np.einsum("ai,qaI->qiI", cell_nodes, dNdxi)  # (nq, 3, 3)
            detJ = np.linalg.det(J)
            dNdx = np.einsum("qaI,qIi->qai", dNdxi, np.linalg.inv(J))  # (nq, n_nodes, 3)
            grad_u = np.einsum("ai,qaj->qij", cell_sol, dNdx)   # (nq, 3, 3)
            eps = 0.5 * (grad_u + np.transpose(grad_u, (0, 2, 1)))
            return eps, dNdx, detJ

        def _residual(sigma, dNdx, detJ):
            JxW = w * detJ
            R = np.einsum("q,qij,qaj->ai", JxW, sigma, dNdx)
            return jax.flatten_util.ravel_pytree(R)[0]

        if self._with_thermal:
            def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads,
                       cell_JxW, cell_v_grads_JxW, cell_nodes, cell_C_quad, cell_eps_th):
                # cell_eps_th: (nq, 3, 3) thermal eigenstrain α·ΔT in global axes
                eps, dNdx, detJ = _kinematics(cell_sol_flat, cell_nodes)
                sigma = np.einsum("qijkl,qkl->qij", cell_C_quad, eps - cell_eps_th)
                return _residual(sigma, dNdx, detJ)
        else:
            def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads,
                       cell_JxW, cell_v_grads_JxW, cell_nodes, cell_C_quad):
                # cell_nodes: (n_nodes, 3); cell_C_quad: (nq, 3,3,3,3)
                eps, dNdx, detJ = _kinematics(cell_sol_flat, cell_nodes)
                sigma = np.einsum("qijkl,qkl->qij", cell_C_quad, eps)  # (nq, 3, 3)
                return _residual(sigma, dNdx, detJ)

        return kernel


def expand_cte_to_quad(
    cte_cell_ply: np.ndarray,
    ply_thicknesses: Sequence[float],
    *,
    n_inplane: int = 2,
    n_thick_per_ply: int = 2,
) -> np.ndarray:
    """Per-cell, per-ply CTE (global axes) -> per-quad CTE ``(num_cells, nq, 3, 3)``.

    Uses the **same** per-ply through-thickness quadrature as
    :func:`create_oriented_layered_solid`, so the result lines up with ``C_quad``.
    Multiply by the temperature change to get the thermal eigenstrain to feed the
    element / :func:`layered_quad_stress`::

        cte_quad = expand_cte_to_quad(cte_cell_ply, ply_thicknesses)
        eps_th   = cte_quad * dT                       # (num_cells, nq, 3, 3)

    ``dT`` may be a scalar, a per-cell array (broadcast as ``dT[:, None, None, None]``)
    or any field — keep it separate so the temperature can be a load/design input.
    """
    thick = onp.asarray(ply_thicknesses, dtype=onp.float64)
    _, _, ply_of = layered_reference_quadrature(
        thick, n_inplane=n_inplane, n_thick_per_ply=n_thick_per_ply)
    return np.asarray(cte_cell_ply)[:, np.asarray(ply_of)]


def create_oriented_layered_solid(
    mesh,
    C_cell_ply: np.ndarray,
    ply_thicknesses: Sequence[float],
    *,
    ele_type: str = "HEX8",
    n_inplane: int = 2,
    n_thick_per_ply: int = 2,
    location_fns: Optional[Iterable[Callable]] = None,
    problem_class: type = OrientedLayeredSolid,
    with_thermal: bool = False,
    cte_cell_ply: Optional[np.ndarray] = None,
) -> Tuple[OrientedLayeredSolid, InternalVars]:
    """Build an :class:`OrientedLayeredSolid` from *already-rotated*, per-cell,
    per-ply stiffness tensors (global axes).

    The orientation logic (per-element local triad, winding angle, ply rotation) is
    left to the caller — this keeps the element general. The caller supplies, for
    every cell and ply, the lamina stiffness expressed in **global** axes; this
    function only handles the exact per-ply through-thickness quadrature and packs
    the per-quad stiffness as an internal variable.

    Parameters
    ----------
    mesh : feax.Mesh
        3D mesh, one element through the laminate thickness, laminate stacked along
        the element's local ζ-axis (last reference axis, i.e. nodes 0-3 = inner
        thickness face, nodes 4-7 = outer face for a HEX8).
    C_cell_ply : array
        ``(num_cells, n_ply, 3,3,3,3)`` rotated lamina stiffness in global axes,
        bottom → top through the thickness.
    ply_thicknesses : sequence of float
        Per-ply thicknesses, bottom → top (length ``n_ply``).
    ele_type, n_inplane, n_thick_per_ply, location_fns
        As in :func:`create_layered_solid`.
    with_thermal : bool
        Enable thermal coupling. When ``True`` the element uses
        ``σ = C : (ε − ε_th)`` and expects a **third** volume internal variable,
        the per-quad thermal eigenstrain ``eps_th`` ``(num_cells, nq, 3, 3)``.
        The returned ``internal_vars`` carry a default ``eps_th`` (zero, or
        ``cte_quad · 1`` from ``cte_cell_ply``); recompute it per solve as
        ``cte_quad * dT`` and rebuild :class:`InternalVars` with it.
    cte_cell_ply : array, optional
        ``(num_cells, n_ply, 3, 3)`` per-ply CTE in **global** axes (rotate with
        :func:`rotate_cte_3d` using the same frame as the stiffness). Only used
        when ``with_thermal=True`` to seed the default ``eps_th`` at ``ΔT = 1``;
        if omitted the default ``eps_th`` is zero.

    Returns
    -------
    problem : OrientedLayeredSolid
    internal_vars : feax.InternalVars
        ``volume_vars = (cell_nodes, C_quad)``, or
        ``(cell_nodes, C_quad, eps_th)`` when ``with_thermal=True``.
    """
    if ele_type != "HEX8":
        raise NotImplementedError(
            "create_oriented_layered_solid currently supports HEX8 only"
        )

    C_cell_ply = np.asarray(C_cell_ply)
    num_cells, n_ply = C_cell_ply.shape[0], C_cell_ply.shape[1]
    thick = onp.asarray(ply_thicknesses, dtype=onp.float64)
    if thick.shape[0] != n_ply:
        raise ValueError(
            f"ply_thicknesses has {thick.shape[0]} entries but C_cell_ply has "
            f"{n_ply} plies"
        )

    ref_pts, weights, ply_of = layered_reference_quadrature(
        thick, n_inplane=n_inplane, n_thick_per_ply=n_thick_per_ply)
    dNdxi = _tabulate_reference(ele_type, ref_pts)          # (nq, n_nodes, 3)
    nq = ref_pts.shape[0]

    # Expand per-ply stiffness to per-quad: (num_cells, nq, 3,3,3,3)
    C_quad = C_cell_ply[:, np.asarray(ply_of)]

    location_fns = tuple(location_fns) if location_fns is not None else None
    problem = problem_class(
        mesh, vec=3, dim=3, ele_type=ele_type, location_fns=location_fns,
        additional_info=(dNdxi, np.asarray(weights), with_thermal),
    )
    cell_nodes = np.asarray(mesh.points)[np.asarray(mesh.cells)]
    if with_thermal:
        if cte_cell_ply is not None:
            cte = np.asarray(cte_cell_ply)
            if cte.shape[:2] != (num_cells, n_ply):
                raise ValueError(
                    f"cte_cell_ply must be (num_cells, n_ply, 3, 3) = "
                    f"({num_cells}, {n_ply}, 3, 3); got {tuple(cte.shape)}"
                )
            eps_th = cte[:, np.asarray(ply_of)]            # ΔT = 1 seed
        else:
            eps_th = np.zeros((num_cells, nq, 3, 3))
        internal_vars = InternalVars(volume_vars=(cell_nodes, C_quad, eps_th))
    else:
        internal_vars = InternalVars(volume_vars=(cell_nodes, C_quad))
    return problem, internal_vars


class GeometricStiffnessSolid(fe.Problem):
    """Initial-stress (geometric) stiffness element for linear buckling.

    Given a prestress field ``σ₀`` (from a reference static solution), this element's
    residual is the gradient of the geometric energy

    .. code-block:: text

        E_g(u) = ∫ σ₀_ij · ½ · (∂u_k/∂x_i)(∂u_k/∂x_j) dV,

    so its (constant) Jacobian is the geometric stiffness matrix ``K_g``. The linear
    buckling problem is then the generalized eigenproblem ``(K + λ K_g) φ = 0`` where
    ``K`` is the material tangent stiffness; the lowest ``λ`` scales the reference load
    to the critical load.

    Volume internal variables: ``(cell_nodes, sigma0)`` with
    ``sigma0`` of shape ``(num_cells, nq, 3, 3)``. ``additional_info = (dNdxi_ref,
    weights)``. Assemble ``K_g`` by calling :func:`feax.assembler.get_jacobian` at ``u = 0``
    (the residual is linear in ``u``, so the Jacobian is exact and constant).
    """

    def custom_init(self, dNdxi_ref, weights):
        self._dNdxi_ref = dNdxi_ref
        self._weights = weights

    def get_universal_kernel(self) -> Callable:
        dNdxi = self._dNdxi_ref
        w = self._weights
        unflatten = self.unflatten_fn_dof

        def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads,
                   cell_JxW, cell_v_grads_JxW, cell_nodes, cell_sigma0):
            cell_sol = unflatten(cell_sol_flat)[0]              # (n_nodes, 3)
            J = np.einsum("ai,qaI->qiI", cell_nodes, dNdxi)
            Jinv = np.linalg.inv(J)
            detJ = np.linalg.det(J)
            dNdx = np.einsum("qaI,qIi->qai", dNdxi, Jinv)       # (nq, n_nodes, 3)

            # grad_u[q, c, j] = ∂u_c/∂x_j
            grad_u = np.einsum("ac,qaj->qcj", cell_sol, dNdx)
            s = np.einsum("qij,qcj->qci", cell_sigma0, grad_u)  # σ₀_ij ∂u_c/∂x_j
            JxW = w * detJ
            R = np.einsum("q,qai,qci->ac", JxW, dNdx, s)        # (n_nodes, 3)
            return jax.flatten_util.ravel_pytree(R)[0]

        return kernel


def create_layered_solid_geometric_stiffness(
    mesh,
    cell_sigma0: np.ndarray,
    ply_thicknesses: Sequence[float],
    *,
    ele_type: str = "HEX8",
    n_inplane: int = 2,
    n_thick_per_ply: int = 2,
) -> Tuple[GeometricStiffnessSolid, InternalVars]:
    """Build a :class:`GeometricStiffnessSolid` from a per-quad prestress field.

    Uses the **same** per-ply through-thickness quadrature as
    :func:`create_oriented_layered_solid`, so ``cell_sigma0`` must be evaluated at those
    quadrature points (e.g. via :func:`layered_quad_stress` on the reference solution).

    Parameters
    ----------
    mesh : feax.Mesh
    cell_sigma0 : array
        ``(num_cells, nq, 3, 3)`` Cauchy prestress at each quadrature point.
    ply_thicknesses : sequence of float
        Per-ply thicknesses (defines the through-thickness quadrature split).
    ele_type, n_inplane, n_thick_per_ply
        Must match the material model used for the reference solution.

    Returns
    -------
    problem : GeometricStiffnessSolid
    internal_vars : feax.InternalVars
        ``volume_vars = (cell_nodes, sigma0)``.
    """
    if ele_type != "HEX8":
        raise NotImplementedError("create_layered_solid_geometric_stiffness supports HEX8 only")
    thick = onp.asarray(ply_thicknesses, dtype=onp.float64)
    ref_pts, weights, _ = layered_reference_quadrature(
        thick, n_inplane=n_inplane, n_thick_per_ply=n_thick_per_ply)
    dNdxi = _tabulate_reference(ele_type, ref_pts)
    problem = GeometricStiffnessSolid(
        mesh, vec=3, dim=3, ele_type=ele_type,
        additional_info=(dNdxi, np.asarray(weights)),
    )
    cell_nodes = np.asarray(mesh.points)[np.asarray(mesh.cells)]
    internal_vars = InternalVars(volume_vars=(cell_nodes, np.asarray(cell_sigma0)))
    return problem, internal_vars


def layered_quad_stress(
    dNdxi_ref: np.ndarray,
    cell_nodes: np.ndarray,
    u_cells: np.ndarray,
    C_quad: np.ndarray,
    eps_th_quad: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-cell, per-quad Cauchy stress for a layered solid.

    With no thermal field this is the purely-mechanical ``σ = C : ε(u)``; passing
    ``eps_th_quad`` gives ``σ = C : (ε(u) − ε_th)`` so the prestress fed to the
    geometric-stiffness / buckling solve includes the cool-down thermal stress
    (the same constitutive law as the ``with_thermal=True`` element).

    Parameters
    ----------
    dNdxi_ref : array
        ``(nq, n_nodes, 3)`` reference shape gradients (``problem._dNdxi_ref``).
    cell_nodes : array
        ``(num_cells, n_nodes, 3)`` physical node coordinates.
    u_cells : array
        ``(num_cells, n_nodes, 3)`` nodal displacements gathered per cell.
    C_quad : array
        ``(num_cells, nq, 3,3,3,3)`` per-quad stiffness (``internal_vars`` from
        :func:`create_oriented_layered_solid`).
    eps_th_quad : array, optional
        ``(num_cells, nq, 3, 3)`` thermal eigenstrain ``α·ΔT`` in global axes
        (e.g. ``expand_cte_to_quad(...) * dT``). ``None`` → no thermal term.

    Returns
    -------
    sigma : array
        ``(num_cells, nq, 3, 3)`` stress at each quadrature point.
    """
    def _eps(nodes, u):
        J = np.einsum("ai,qaI->qiI", nodes, dNdxi_ref)
        dNdx = np.einsum("qaI,qIi->qai", dNdxi_ref, np.linalg.inv(J))
        grad_u = np.einsum("ai,qaj->qij", u, dNdx)
        return 0.5 * (grad_u + np.transpose(grad_u, (0, 2, 1)))

    if eps_th_quad is None:
        def one_cell(nodes, u, C):
            return np.einsum("qijkl,qkl->qij", C, _eps(nodes, u))
        return jax.vmap(one_cell)(cell_nodes, u_cells, C_quad)

    def one_cell_th(nodes, u, C, eth):
        return np.einsum("qijkl,qkl->qij", C, _eps(nodes, u) - eth)
    return jax.vmap(one_cell_th)(cell_nodes, u_cells, C_quad, eps_th_quad)


__all__ = [
    "voigt_to_tensor",
    "tensor_to_voigt",
    "isotropic_stiffness_3d",
    "orthotropic_stiffness_3d",
    "transverse_isotropic_stiffness_3d",
    "isotropic_cte_3d",
    "transverse_isotropic_cte_3d",
    "rotation_matrix_axis",
    "rotate_stiffness_3d",
    "rotate_cte_3d",
    "layered_reference_quadrature",
    "LayeredSolid",
    "create_layered_solid",
    "OrientedLayeredSolid",
    "create_oriented_layered_solid",
    "expand_cte_to_quad",
    "interpolate_nodal_to_layered_quad",
    "GeometricStiffnessSolid",
    "create_layered_solid_geometric_stiffness",
    "layered_quad_stress",
]
