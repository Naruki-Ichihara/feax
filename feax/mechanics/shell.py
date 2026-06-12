"""Mindlin/Reissner (FSDT) flat plate elements.

This module provides building blocks for first-order shear-deformation
plate theory (Mindlin/Reissner) on a flat 2D mesh embedded in the
``z = 0`` plane. Curved-shell (curvilinear basis, geometric stiffness,
drilling DOF) extensions are not implemented yet.

Variable layout
---------------
The plate is treated as a *two-variable* problem so that the transverse
shear coupling ``γ_α = ∂w/∂x_α + θ_α`` can be expressed as a mixed
"gradient + value" term in feax's :meth:`~feax.Problem.get_weak_form`:

* variable 0 — translations  ``(u, v, w)``         (vec = 3)
* variable 1 — section rotations ``(θ_x, θ_y)``    (vec = 2)

Both variables share the same 2D mesh.

Kinematics (FSDT, flat plate)
-----------------------------

.. code-block:: text

    ε_αβ  = sym(∇u_in)         (membrane strain,  2×2)
    κ_αβ  = sym(∇θ)            (curvature,        2×2)
    γ_α   = ∂w/∂x_α + θ_α      (transverse shear, 2)

Constitutive law (homogeneous symmetric plate, no membrane-bending coupling):

.. code-block:: text

    N = A : ε        (membrane stress resultant,  2×2)
    M = D : κ        (bending moment,             2×2)
    Q = G_s · γ      (transverse shear force,     2)

with thickness-integrated stiffness

.. code-block:: text

    A   = h        · C_in                  (2,2,2,2)
    D   = h^3 / 12 · C_in                  (2,2,2,2)
    G_s = κ_s h    · G_transverse          (2,2)

where ``κ_s = 5/6`` is the standard shear-correction factor.

Notes
-----
Low-order interpolations (QUAD4 with full integration) suffer from
transverse-shear locking on thin plates. A reduced/MITC integration
scheme is not yet implemented; for the time being prefer moderately
thick plates (``h / L ≳ 0.05``) or use higher-order elements.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple, Union

import jax
import jax.numpy as np

import feax as fe


# ---------------------------------------------------------------------------
# In-plane stiffness builders (4th-order, plane stress)
# ---------------------------------------------------------------------------

def isotropic_in_plane_stiffness(E: float, nu: float) -> np.ndarray:
    """Plane-stress isotropic stiffness tensor ``C_ijkl`` of shape ``(2,2,2,2)``.

    .. code-block:: text

        σ_ij = λ* δ_ij tr(ε) + 2 μ ε_ij     (plane stress)
        λ*   = E ν / (1 - ν^2)
        μ    = E / (2 (1 + ν))
    """
    lam = E * nu / (1.0 - nu * nu)
    mu = E / (2.0 * (1.0 + nu))
    delta = np.eye(2)
    C = (
        lam * np.einsum("ij,kl->ijkl", delta, delta)
        + mu * (
            np.einsum("ik,jl->ijkl", delta, delta)
            + np.einsum("il,jk->ijkl", delta, delta)
        )
    )
    return C


def orthotropic_in_plane_stiffness(
    E1: float, E2: float, G12: float, nu12: float
) -> np.ndarray:
    """Plane-stress orthotropic stiffness tensor in *material* axes.

    The fibre direction is the local 1-axis. The Voigt matrix is

    .. code-block:: text

        Q = 1 / (1 - ν12 ν21) · [[ E1,        ν21 E1,  0       ],
                                  [ ν12 E2,   E2,      0       ],
                                  [ 0,        0,       G12·(1-ν12 ν21) ]]

    (with ``ν21 = ν12 E2 / E1``) and is converted to the 4th-order tensor
    ``C_ijkl`` with the Voigt convention σ_voigt = (σ11, σ22, σ12).
    """
    nu21 = nu12 * E2 / E1
    m = 1.0 / (1.0 - nu12 * nu21)
    Q11 = m * E1
    Q22 = m * E2
    Q12 = m * nu21 * E1
    Q66 = G12

    # Build via Voigt: σ_ij ε_ij = σ_v^T Q ε_v (with shear factor 2)
    # We construct C_ijkl directly to make einsum-friendly.
    C = np.zeros((2, 2, 2, 2))
    C = C.at[0, 0, 0, 0].set(Q11)
    C = C.at[1, 1, 1, 1].set(Q22)
    C = C.at[0, 0, 1, 1].set(Q12)
    C = C.at[1, 1, 0, 0].set(Q12)
    # shear: σ_12 = Q66 (2 ε_12), so C_1212 = C_1221 = C_2112 = C_2121 = Q66
    C = C.at[0, 1, 0, 1].set(Q66)
    C = C.at[0, 1, 1, 0].set(Q66)
    C = C.at[1, 0, 0, 1].set(Q66)
    C = C.at[1, 0, 1, 0].set(Q66)
    return C


# ---------------------------------------------------------------------------
# Plate stiffness (thickness integrals)
# ---------------------------------------------------------------------------

def plate_stiffness(
    C_in: np.ndarray,
    h: float,
    G_transverse: Union[float, np.ndarray],
    kappa_s: float = 5.0 / 6.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(A, D, G_s)`` for a homogeneous symmetric flat plate.

    Parameters
    ----------
    C_in : (2,2,2,2) array
        In-plane plane-stress stiffness tensor.
    h : float
        Plate thickness.
    G_transverse : float or (2,2) array
        Transverse shear modulus. A scalar is interpreted as isotropic
        (``G·I``); a ``(2,2)`` array is used directly (e.g.
        ``diag(G13, G23)`` for an orthotropic lamina).
    kappa_s : float
        Shear correction factor (default ``5/6``).

    Returns
    -------
    A : (2,2,2,2) array        Membrane stiffness.
    D : (2,2,2,2) array        Bending stiffness.
    G_s : (2,2) array          Transverse-shear stiffness.
    """
    A = h * C_in
    D = (h ** 3 / 12.0) * C_in
    G_arr = np.asarray(G_transverse)
    if G_arr.ndim == 0:
        G_s = kappa_s * h * G_arr * np.eye(2)
    else:
        G_s = kappa_s * h * G_arr
    return A, D, G_s


# ---------------------------------------------------------------------------
# Laminate (CLT) helpers — analogue of fenics-shells/common/laminates.py
# ---------------------------------------------------------------------------

def z_layer_coordinates(thicknesses: np.ndarray) -> np.ndarray:
    """Interface z-coordinates of an n-layer laminate, midplane at ``z = 0``.

    Layers are ordered from bottom (``k = 0``) to top (``k = n − 1``).
    The returned array has length ``n + 1`` with ``z[0] = -h/2`` and
    ``z[n] = +h/2`` (where ``h = sum(thicknesses)``).
    """
    thicknesses = np.asarray(thicknesses, dtype=np.float64)
    h_total = np.sum(thicknesses)
    return np.concatenate([
        np.array([-h_total / 2.0]),
        -h_total / 2.0 + np.cumsum(thicknesses),
    ])


def rotate_in_plane_stiffness(C_in: np.ndarray, theta: float) -> np.ndarray:
    """Rotate a 4th-order in-plane stiffness tensor by ``theta`` (radians).

    .. math::

        \\bar C_{ijkl} = R_{ip}\\, R_{jq}\\, R_{kr}\\, R_{ls}\\, C_{pqrs},
        \\qquad R(\\theta) = \\begin{pmatrix} c & -s \\\\ s & c \\end{pmatrix}.

    This is the tensor-form equivalent of the Voigt 3×3 transformation
    used in Reddy 1997, §2.3.7.
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return np.einsum("ip,jq,kr,ls,pqrs->ijkl", R, R, R, R, C_in)


def rotate_shear_stiffness(G_shear: np.ndarray, theta: float) -> np.ndarray:
    """Rotate a (2,2) transverse-shear stiffness tensor by ``theta`` (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R @ G_shear @ R.T


def laminate_stiffness(
    C_in: np.ndarray,
    G_shear: np.ndarray,
    thetas: np.ndarray,
    thicknesses: np.ndarray,
    *,
    kappa_s: float = 5.0 / 6.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ``(A, B, D, G_s)`` for an n-layer laminate (CLT integrals).

    For each layer ``k`` with material stiffness rotated by ``thetas[k]``
    (in addition to the lamina's own material orientation):

    .. code-block:: text

        A   = Σ_k Q̄_k         (z_{k+1} − z_k)
        B   = (1/2) Σ_k Q̄_k   (z_{k+1}² − z_k²)
        D   = (1/3) Σ_k Q̄_k   (z_{k+1}³ − z_k³)
        G_s = κ_s · Σ_k Q̄_s,k (z_{k+1} − z_k)

    where ``z_k`` are the interface coordinates from
    :func:`z_layer_coordinates` (midplane at z = 0). This matches Reddy
    1997 §1.3.71 (in-plane) and §3.4.18 (transverse shear).

    Parameters
    ----------
    C_in : array
        In-plane stiffness in lamina material axes. Either a single
        ``(2, 2, 2, 2)`` tensor (broadcast to every layer) or a
        ``(n_layers, 2, 2, 2, 2)`` array of per-layer materials.
    G_shear : array
        Transverse-shear stiffness, ``(2, 2)`` for a single material or
        ``(n_layers, 2, 2)`` per layer. Use ``G * np.eye(2)`` for an
        isotropic lamina or ``np.diag([G13, G23])`` for an orthotropic
        one.
    thetas : array, shape ``(n_layers,)``
        Layer rotation angles in radians (additional to the lamina's
        own material axes).
    thicknesses : array, shape ``(n_layers,)``
        Layer thicknesses, ordered bottom → top.
    kappa_s : float
        Shear correction factor (default ``5/6``).

    Returns
    -------
    A   : (2, 2, 2, 2)   Membrane stiffness.
    B   : (2, 2, 2, 2)   Membrane–bending coupling. Vanishes for
                         midplane-symmetric laminates such as
                         ``[θ / -θ]_s`` or ``[0 / 90]_s``.
    D   : (2, 2, 2, 2)   Bending stiffness.
    G_s : (2, 2)         Transverse-shear stiffness with κ_s applied.
    """
    thetas = np.asarray(thetas, dtype=np.float64)
    thicknesses = np.asarray(thicknesses, dtype=np.float64)
    n = thetas.shape[0]
    if thicknesses.shape[0] != n:
        raise ValueError("thetas and thicknesses must have the same length")

    C_in = np.asarray(C_in)
    G_shear = np.asarray(G_shear)
    if C_in.ndim == 4:
        C_in = np.repeat(C_in[None], n, axis=0)
    if G_shear.ndim == 2:
        G_shear = np.repeat(G_shear[None], n, axis=0)

    z = z_layer_coordinates(thicknesses)  # (n+1,)
    z_lo, z_hi = z[:-1], z[1:]

    def _per_layer(C_k, G_k, theta_k, zk, zk1):
        Cr = rotate_in_plane_stiffness(C_k, theta_k)
        Gr = rotate_shear_stiffness(G_k, theta_k)
        return (
            Cr * (zk1 - zk),
            0.5 * Cr * (zk1 ** 2 - zk ** 2),
            (1.0 / 3.0) * Cr * (zk1 ** 3 - zk ** 3),
            Gr * (zk1 - zk),
        )

    a_k, b_k, d_k, g_k = jax.vmap(_per_layer)(
        C_in, G_shear, thetas, z_lo, z_hi
    )
    A = np.sum(a_k, axis=0)
    B = np.sum(b_k, axis=0)
    D = np.sum(d_k, axis=0)
    G_s = kappa_s * np.sum(g_k, axis=0)
    return A, B, D, G_s


# ---------------------------------------------------------------------------
# Thermal expansion (anisotropic)
# ---------------------------------------------------------------------------

def thermal_expansion_from_orientation(
    a2: np.ndarray,
    alpha_fibre: float,
    alpha_transverse: float,
) -> np.ndarray:
    """Build a 2D thermal-expansion tensor from an orientation tensor.

    Rank-2 closure analogue of the libertas Advani–Tucker stiffness:

    .. math::

        \\alpha(a_2) = \\alpha_t \\, I + (\\alpha_f - \\alpha_t)\\, a_2.

    Limits:

    * ``a_2 = e_1 ⊗ e_1`` (full alignment) → ``α = diag(α_f, α_t)``
    * ``a_2 = (1/2)·I`` (uniform random) → ``α = ((α_f+α_t)/2)·I``

    The expansion tensor field rotates and decays naturally as ``a_2``
    varies in space, making it suitable for libertas-style continuous
    fibre-orientation optimisation with thermal loading.
    """
    return alpha_transverse * np.eye(2) + (alpha_fibre - alpha_transverse) * a2


def rotate_thermal_expansion(alpha: np.ndarray, theta: float) -> np.ndarray:
    """Rotate a ``(2, 2)`` thermal-expansion tensor by ``theta`` (radians).

    .. math:: \\bar\\alpha = R\\,\\alpha\\,R^T.
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R @ alpha @ R.T


def laminate_thermal_loads(
    C_in: np.ndarray,
    alpha: np.ndarray,
    thetas: np.ndarray,
    thicknesses: np.ndarray,
    *,
    dT_avg: float = 1.0,
    dT_grad: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Thermal force/moment resultants ``(N_T, M_T)`` for an n-layer laminate.

    For a through-thickness temperature distribution
    ``ΔT(z) = dT_avg + z · dT_grad``,

    .. code-block:: text

        N_T = Σ_k (C̄_k : α̅_k) · ∫_{z_k}^{z_{k+1}} ΔT(z) dz
        M_T = Σ_k (C̄_k : α̅_k) · ∫_{z_k}^{z_{k+1}} z · ΔT(z) dz

    where ``C̄_k`` and ``α̅_k`` are the layer stiffness and CTE rotated
    by ``thetas[k]`` to laminate axes (Reddy 1997, §1.4).

    For a midplane-symmetric laminate at uniform ``ΔT``, ``M_T = 0``.

    Parameters
    ----------
    C_in : array
        In-plane stiffness, ``(2, 2, 2, 2)`` (broadcast to all layers)
        or ``(n_layers, 2, 2, 2, 2)``.
    alpha : array
        Lamina CTE in material axes, ``(2, 2)`` (broadcast) or
        ``(n_layers, 2, 2)``. Build with
        :func:`thermal_expansion_from_orientation` or
        ``np.diag([alpha_1, alpha_2])``.
    thetas : ``(n_layers,)``
        Layer rotation angles (radians).
    thicknesses : ``(n_layers,)``
        Layer thicknesses, ordered bottom → top.
    dT_avg, dT_grad : float
        Coefficients of ``ΔT(z) = dT_avg + z · dT_grad``.

    Returns
    -------
    N_T : ``(2, 2)`` array — membrane thermal force resultant.
    M_T : ``(2, 2)`` array — bending thermal moment resultant.
    """
    thetas = np.asarray(thetas, dtype=np.float64)
    thicknesses = np.asarray(thicknesses, dtype=np.float64)
    n = thetas.shape[0]
    if thicknesses.shape[0] != n:
        raise ValueError("thetas and thicknesses must have the same length")

    C_in = np.asarray(C_in)
    alpha = np.asarray(alpha)
    if C_in.ndim == 4:
        C_in = np.repeat(C_in[None], n, axis=0)
    if alpha.ndim == 2:
        alpha = np.repeat(alpha[None], n, axis=0)

    z = z_layer_coordinates(thicknesses)
    z_lo, z_hi = z[:-1], z[1:]

    def _per_layer(C_k, alpha_k, theta_k, zk, zk1):
        C_rot = rotate_in_plane_stiffness(C_k, theta_k)
        a_rot = rotate_thermal_expansion(alpha_k, theta_k)
        # σ per unit ΔT in laminate axes
        sigma_per_dT = np.einsum("ijkl,kl->ij", C_rot, a_rot)
        I_T = dT_avg * (zk1 - zk) + 0.5 * dT_grad * (zk1**2 - zk**2)
        I_zT = (
            0.5 * dT_avg * (zk1**2 - zk**2)
            + (1.0 / 3.0) * dT_grad * (zk1**3 - zk**3)
        )
        return sigma_per_dT * I_T, sigma_per_dT * I_zT

    Ns, Ms = jax.vmap(_per_layer)(C_in, alpha, thetas, z_lo, z_hi)
    return np.sum(Ns, axis=0), np.sum(Ms, axis=0)


# ---------------------------------------------------------------------------
# Mindlin kinematics & resultants
# ---------------------------------------------------------------------------

def mindlin_strains(
    grad_uvw: np.ndarray,
    grad_theta: np.ndarray,
    theta: np.ndarray,
    *,
    nonlinear: str = "linear",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute membrane strain, curvature, and transverse shear strain.

    Parameters
    ----------
    grad_uvw : (3, 2) array
        Spatial gradient of ``(u, v, w)``.
    grad_theta : (2, 2) array
        Spatial gradient of ``(θ_x, θ_y)``.
    theta : (2,) array
        Section rotations ``(θ_x, θ_y)``.
    nonlinear : {"linear", "von_karman"}
        ``"linear"`` (default) uses ``ε = sym(∇u_in)``.
        ``"von_karman"`` adds the moderate-rotation membrane term
        ``½ ∇w ⊗ ∇w``, capturing membrane stretching due to bending
        for rotations up to ~10–15 degrees. The curvature ``κ`` and
        transverse shear ``γ`` remain linear in their arguments.

    Returns
    -------
    eps   : (2, 2) — membrane strain
    kappa : (2, 2) — sym(∇θ)           (curvature)
    gamma : (2,)   — ∇w + θ            (transverse shear)
    """
    u_grad = grad_uvw[0:2, :]
    grad_w = grad_uvw[2, :]

    eps = 0.5 * (u_grad + u_grad.T)
    if nonlinear == "von_karman":
        eps = eps + 0.5 * np.outer(grad_w, grad_w)
    elif nonlinear != "linear":
        raise ValueError(
            f"nonlinear must be 'linear' or 'von_karman', got {nonlinear!r}"
        )

    kappa = 0.5 * (grad_theta + grad_theta.T)
    gamma = grad_w + theta
    return eps, kappa, gamma


def mindlin_resultants(
    eps: np.ndarray,
    kappa: np.ndarray,
    gamma: np.ndarray,
    A: np.ndarray,
    D: np.ndarray,
    G_s: np.ndarray,
    *,
    B: Optional[np.ndarray] = None,
    N_T: Optional[np.ndarray] = None,
    M_T: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear constitutive law for an FSDT plate / laminate.

    Without optional terms:

    .. code-block:: text

        N = A : ε,    M = D : κ,    Q = G_s · γ.

    With ``B`` (membrane–bending coupling, e.g. asymmetric laminate):

    .. code-block:: text

        N = A : ε + B : κ,    M = B : ε + D : κ.

    With ``N_T`` and/or ``M_T`` (thermal pre-force / pre-moment from
    :func:`laminate_thermal_loads`, or computed locally from a quad-point
    eigenstrain ``α(x)·ΔT``), they are subtracted from the elastic
    resultants:

    .. code-block:: text

        N ← N − N_T,    M ← M − M_T.
    """
    N = np.einsum("ijkl,kl->ij", A, eps)
    M = np.einsum("ijkl,kl->ij", D, kappa)
    if B is not None:
        N = N + np.einsum("ijkl,kl->ij", B, kappa)
        M = M + np.einsum("ijkl,kl->ij", B, eps)
    if N_T is not None:
        N = N - N_T
    if M_T is not None:
        M = M - M_T
    Q = G_s @ gamma
    return N, M, Q


# ---------------------------------------------------------------------------
# Weak-form factory
# ---------------------------------------------------------------------------

def mindlin_weak_form(
    A: np.ndarray,
    D: np.ndarray,
    G_s: np.ndarray,
    *,
    B: Optional[np.ndarray] = None,
    N_T: Optional[np.ndarray] = None,
    M_T: Optional[np.ndarray] = None,
    nonlinear: str = "linear",
) -> Callable:
    """Return a quad-point weak form for the FSDT (Mindlin) plate.

    The callable is meant to be returned by
    :meth:`feax.Problem.get_weak_form` for a two-variable problem with
    layout ``var0 = (u, v, w)``, ``var1 = (θ_x, θ_y)``.

    Per quadrature point (with test variations ``δ(u,v,w), δθ``):

    .. code-block:: text

        δ(u,v) :  ∫ N_αβ ∂_β δu_α dΩ          → grad_term[0][0:2, :] = N
        δw    :  ∫ q_β   ∂_β δw   dΩ          → grad_term[0][2,   :] = q
        δθ    :  ∫ M_αβ ∂_β δθ_α + Q_α δθ_α dΩ → grad_term[1] = M, mass_term[1] = Q

    where ``q = Q`` for ``nonlinear="linear"`` and
    ``q = Q + N · ∇w`` for ``nonlinear="von_karman"`` — the latter is
    the membrane–bending coupling term arising from the variation of
    ``½ ∇w ⊗ ∇w``.

    Parameters
    ----------
    A, D : (2,2,2,2) arrays      Membrane and bending stiffness.
    G_s  : (2,2) array           Transverse shear stiffness.
    nonlinear : {"linear", "von_karman"}
        Strain measure (see :func:`mindlin_strains`). ``"von_karman"``
        makes the residual cubic in ``w`` and requires Newton iteration
        (``linear=False`` when calling :func:`feax.create_solver`).

    Returns
    -------
    weak_form : callable
        ``(vals, grads, x, *internal_vars) -> (mass_terms, grad_terms)``.
    """

    def weak_form(vals, grads, x, *internal_vars):
        uvw = vals[0]                       # (3,)
        theta = vals[1]                     # (2,)
        grad_uvw = grads[0]                 # (3, 2)
        grad_theta = grads[1]               # (2, 2)

        eps, kappa, gamma = mindlin_strains(
            grad_uvw, grad_theta, theta, nonlinear=nonlinear
        )
        N, M, Q = mindlin_resultants(
            eps, kappa, gamma, A, D, G_s, B=B, N_T=N_T, M_T=M_T,
        )

        # δw integrand: linear shear Q plus, for vK, the membrane-bending
        # coupling N · ∇w arising from the variation of ½ ∇w ⊗ ∇w.
        if nonlinear == "von_karman":
            grad_w = grad_uvw[2, :]
            q_w = Q + N @ grad_w
        else:
            q_w = Q

        # var 0: (u, v, w) — pack [N; q_w] into a (3, 2) gradient term.
        grad0 = np.concatenate([N, q_w[None, :]], axis=0)
        mass0 = np.zeros(3)

        # var 1: (θ_x, θ_y) — bending and shear coupling.
        grad1 = M
        mass1 = Q

        return ([mass0, mass1], [grad0, grad1])

    return weak_form


# ---------------------------------------------------------------------------
# Problem subclass and factory
# ---------------------------------------------------------------------------

class MindlinPlate(fe.Problem):
    """Two-variable FSDT plate problem (translations + rotations).

    Use :func:`make_mindlin_plate` to construct one cleanly from a single
    mesh; this class itself is a vanilla :class:`feax.Problem` subclass
    that expects ``additional_info=(A, D, G_s, nonlinear)``.
    """

    def custom_init(
        self,
        A: np.ndarray,
        D: np.ndarray,
        G_s: np.ndarray,
        nonlinear: str = "linear",
        B: Optional[np.ndarray] = None,
        N_T: Optional[np.ndarray] = None,
        M_T: Optional[np.ndarray] = None,
    ):
        self.A = A
        self.D = D
        self.G_s = G_s
        self.nonlinear = nonlinear
        self.B = B
        self.N_T = N_T
        self.M_T = M_T

    def get_weak_form(self):
        return mindlin_weak_form(
            self.A, self.D, self.G_s,
            B=self.B, N_T=self.N_T, M_T=self.M_T,
            nonlinear=self.nonlinear,
        )


def make_mindlin_plate(
    mesh,
    A: np.ndarray,
    D: np.ndarray,
    G_s: np.ndarray,
    ele_type: str = "QUAD4",
    location_fns: Optional[Iterable[Callable]] = None,
    gauss_order: Optional[int] = None,
    nonlinear: str = "linear",
    B: Optional[np.ndarray] = None,
    N_T: Optional[np.ndarray] = None,
    M_T: Optional[np.ndarray] = None,
) -> MindlinPlate:
    """Construct a :class:`MindlinPlate` from a single 2D mesh.

    Variable layout is fixed: ``var0 = (u, v, w)`` (``vec=3``) and
    ``var1 = (θ_x, θ_y)`` (``vec=2``), both on the same mesh.

    Parameters
    ----------
    mesh : feax.Mesh
        Single 2D mesh in the ``z = 0`` plane (only x, y coordinates used).
    A, D, G_s : arrays
        Plate stiffness from :func:`plate_stiffness`.
    ele_type : str
        Element type for both variables (default ``"QUAD4"``).
    location_fns : iterable of callables, optional
        Boundary-location predicates for surface integrals.
    gauss_order : int, optional
        Gaussian quadrature order; ``None`` lets feax choose per element.
    nonlinear : {"linear", "von_karman"}
        Strain measure. ``"von_karman"`` requires a Newton solve
        (``linear=False`` in :func:`feax.create_solver`).
    B : (2,2,2,2) array, optional
        Membrane–bending coupling stiffness for an asymmetric laminate.
        Defaults to ``None`` (uncoupled, symmetric plate).
    N_T, M_T : (2, 2) arrays, optional
        Thermal force / moment resultants from
        :func:`laminate_thermal_loads` (or another eigenstrain source).
        Subtracted from the constitutive resultants.
    """
    location_fns = tuple(location_fns) if location_fns is not None else None
    if gauss_order is None:
        gorder = None
    else:
        gorder = [gauss_order, gauss_order]
    return MindlinPlate(
        mesh=[mesh, mesh],
        vec=[3, 2],
        dim=2,
        ele_type=[ele_type, ele_type],
        gauss_order=gorder,
        location_fns=location_fns,
        additional_info=(A, D, G_s, nonlinear, B, N_T, M_T),
    )


__all__ = [
    "isotropic_in_plane_stiffness",
    "orthotropic_in_plane_stiffness",
    "plate_stiffness",
    "z_layer_coordinates",
    "rotate_in_plane_stiffness",
    "rotate_shear_stiffness",
    "laminate_stiffness",
    "thermal_expansion_from_orientation",
    "rotate_thermal_expansion",
    "laminate_thermal_loads",
    "mindlin_strains",
    "mindlin_resultants",
    "mindlin_weak_form",
    "MindlinPlate",
    "make_mindlin_plate",
]
