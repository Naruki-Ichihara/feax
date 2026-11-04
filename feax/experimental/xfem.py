"""XFEM (eXtended FEM) on a fixed mesh — experimental.

An interface described by a nodal level set ``phi`` (zero iso-line = the
interface) is embedded in a mesh that does not conform to it.  The cells the
interface crosses ("cut" cells) get *enriched*: the nodes of those cells carry
extra DOFs ``a_j`` and the displacement is

.. code-block:: text

    u(x) = Σ_j N_j(x) u_j  +  Σ_{j enriched} N_j(x) ψ_j(x) a_j

with an enrichment function ``ψ_j`` chosen for the physics of the interface
(:class:`ShiftedHeaviside` for a displacement jump, :class:`ShiftedAbs` /
:class:`ModifiedAbs` for a strain jump, ...).  Cut cells are integrated with a
sub-triangulation that follows the interface, so no quadrature point straddles
the discontinuity.

Building blocks
---------------

* :class:`Enrichment` and the defaults :class:`ShiftedHeaviside`,
  :class:`Heaviside`, :class:`ShiftedAbs`, :class:`ModifiedAbs` — pluggable
  ``ψ_j(x)``, ``∇ψ_j(x)``.
* :class:`XFEMSpace` — host-side preprocessing: clamps ``phi``, finds the cut
  cells and the enriched nodes, builds the *enrichment mesh* (enriched nodes +
  one phantom node, so the enriched DOFs cost exactly ``vec × n_enriched``
  DOFs, not ``vec × n_nodes``) and the padded sub-cell quadrature.
* :class:`XFEMProblem` — a two-variable :class:`~feax.Problem` (variable 0:
  standard displacement; variable 1: enriched DOFs on a DOF-only space,
  ``with_jxw=False``) whose universal kernel integrates a user constitutive
  law ``sigma = material(u_grad, phi, x, *extra)`` on the cut quadrature.

The standard feax machinery (Dirichlet BCs, direct / Krylov solvers,
``TracedStructure``, autodiff tangents and adjoints) applies unchanged.

Usage
-----

.. code-block:: python

    from feax.experimental.xfem import XFEMSpace, ShiftedAbs, xfem_problem

    phi = mesh.points[:, 0] - 0.73                    # interface at x = 0.73
    space = XFEMSpace(mesh, phi, ShiftedAbs(), vec=2, ele_type="QUAD4")

    def material(u_grad, phi, x):                     # two-phase, nu = 0
        E = np.where(phi < 0, E_left, E_right)
        eps = 0.5 * (u_grad + u_grad.T)
        return E * eps

    problem = xfem_problem(space, material)
    bc = space.create_bc(problem, fe.DirichletBCConfig([...]))   # adds the phantom rows
    solver = fe.create_solver(problem, bc, linear=True)
    sol = solver(space.traced_params(), fe.zero_like_initial_guess(problem, bc))
    u_nodes = space.nodal_displacement(problem, sol)
    x_q, u_q, phi_q = space.quad_fields(problem, sol)  # both sides of the interface

Scope of this first version: 2D (``TRI3``, ``QUAD4``), one level set, one
enrichment per node (``Enrichment.n_funcs == 1``; the DOF layout already
allows more, e.g. crack-tip branch functions), bulk terms only (an interface
cohesive law is the natural next addition: the cut segments are available from
the same preprocessing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as np
import numpy as onp

from feax.DCboundary import DirichletBC, DirichletBCConfig
from feax.mesh import Mesh
from feax.problem import Problem
from feax.traced_params import TracedParams


# ---------------------------------------------------------------------------
# Enrichment functions
# ---------------------------------------------------------------------------

def _sgn(p):
    """Sign without the zero: phi = 0 is counted as the positive side."""
    return np.where(p >= 0.0, 1.0, -1.0)


class Enrichment:
    """Enrichment function ``ψ_j(x)`` of the partition-of-unity ansatz.

    Subclasses implement :meth:`__call__` returning, for the ``nq`` quadrature
    points of one cell and its ``nn`` nodes, the value ``psi[q, a, f]`` and the
    physical gradient ``gpsi[q, a, f, :]`` of the ``f``-th enrichment function
    attached to node ``a`` (``f < n_funcs``).  :meth:`at_nodes` gives the value
    ``ψ_j(x_j)`` at the node itself (zero for *shifted* enrichments), used to
    recover nodal displacements.

    Parameters passed to ``__call__``:

    * ``phi_q``  (nq,)        level set at the quadrature points
    * ``gphi_q`` (nq, dim)    its physical gradient
    * ``phi_n``  (nn,)        nodal level-set values of the cell
    * ``N``      (nq, nn)     shape functions
    * ``dNdx``   (nq, nn, dim) their physical gradients
    """
    n_funcs: int = 1
    name: str = "enrichment"

    def __call__(self, phi_q, gphi_q, phi_n, N, dNdx):
        raise NotImplementedError

    def at_nodes(self, phi_n):
        """``ψ_j(x_j)`` for an array of nodal level-set values (zero if shifted)."""
        return np.zeros(phi_n.shape + (self.n_funcs,))


class ShiftedHeaviside(Enrichment):
    """Strong discontinuity (displacement jump), shifted so ``ψ_j(x_j) = 0``:

    ``ψ_j(x) = sign(phi(x)) − sign(phi_j)``.  The jump across the interface is
    ``2 a_j``; nodal displacements stay ``u_j``; blending cells are enrichment-
    free automatically.  This is the enrichment of NLXP3D's CUT elements.
    """
    name = "shifted_heaviside"

    def __call__(self, phi_q, gphi_q, phi_n, N, dNdx):
        psi = _sgn(phi_q)[:, None] - _sgn(phi_n)[None, :]
        return psi[:, :, None], np.zeros(psi.shape + (1, gphi_q.shape[-1]))


class Heaviside(Enrichment):
    """Strong discontinuity, classical unshifted form ``ψ_j(x) = sign(phi(x))``.

    Nodal displacements are ``u_j + sign(phi_j) a_j`` (see :meth:`at_nodes`).
    """
    name = "heaviside"

    def __call__(self, phi_q, gphi_q, phi_n, N, dNdx):
        psi = np.broadcast_to(_sgn(phi_q)[:, None], (phi_q.shape[0], phi_n.shape[0]))
        return psi[:, :, None], np.zeros(psi.shape + (1, gphi_q.shape[-1]))

    def at_nodes(self, phi_n):
        return _sgn(phi_n)[:, None]


class ShiftedAbs(Enrichment):
    """Weak discontinuity (strain jump, continuous displacement), shifted:

    ``ψ_j(x) = |phi(x)| − |phi_j|``,  ``∇ψ_j = sign(phi) ∇phi``.

    Note the shift only zeroes ``ψ_j`` *at* node j: on a blending cell (one
    that holds enriched nodes but is not cut) ``|phi|`` is linear, so ``ψ_j``
    is linear there and the enriched ansatz carries spurious quadratic terms.
    This is the classic blending-element error of the abs enrichment: a
    kinked piecewise-linear field is reproduced only approximately (the
    error vanishes with refinement).  Prefer :class:`ModifiedAbs`, which is
    blending-free and exact for such fields.
    """
    name = "shifted_abs"

    def __call__(self, phi_q, gphi_q, phi_n, N, dNdx):
        psi = np.abs(phi_q)[:, None] - np.abs(phi_n)[None, :]
        gpsi = (_sgn(phi_q)[:, None] * gphi_q)[:, None, :]               # (nq, 1, dim)
        gpsi = np.broadcast_to(gpsi, (phi_q.shape[0], phi_n.shape[0], gphi_q.shape[-1]))
        return psi[:, :, None], gpsi[:, :, None, :]


class ModifiedAbs(Enrichment):
    """Weak discontinuity, the blending-free ridge function of Moës et al. (2003):

    ``ψ(x) = Σ_i N_i(x) |phi_i| − |Σ_i N_i(x) phi_i|`` (the same function for
    every node of the cell).  It vanishes identically on cells the interface
    does not cross, so blending cells need no special treatment.
    """
    name = "modified_abs"

    def __call__(self, phi_q, gphi_q, phi_n, N, dNdx):
        ridge = N @ np.abs(phi_n) - np.abs(phi_q)                         # (nq,)
        gridge = np.einsum("qai,a->qi", dNdx, np.abs(phi_n)) - _sgn(phi_q)[:, None] * gphi_q
        nq, nn = N.shape
        psi = np.broadcast_to(ridge[:, None], (nq, nn))
        gpsi = np.broadcast_to(gridge[:, None, :], (nq, nn, gphi_q.shape[-1]))
        return psi[:, :, None], gpsi[:, :, None, :]


DEFAULT_ENRICHMENTS = {
    "shifted_heaviside": ShiftedHeaviside,
    "heaviside": Heaviside,
    "shifted_abs": ShiftedAbs,
    "modified_abs": ModifiedAbs,
}


# ---------------------------------------------------------------------------
# Cohesive zone on the interface: traction-separation laws
# ---------------------------------------------------------------------------

def _blend(x, x0, half_width):
    """Smoothstep from 0 (x < x0 − w) to 1 (x > x0 + w); ``half_width = 0``
    gives the sharp switch ``x >= x0``."""
    if half_width <= 0.0:
        return np.where(x >= x0, 1.0, 0.0)
    t = np.clip((x - (x0 - half_width)) / (2.0 * half_width), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _secant(phi_load, delta_max):
    """Secant stiffness ``k = t_L(δmax)/δmax`` and the loading potential at
    ``δmax`` (``t_L = dφ_L/dδ``; ``½ K``-type limit for a virgin chord)."""
    dm = np.maximum(delta_max, 1e-12)
    flat = np.reshape(dm, (-1,))                                  # works for any rank, scalars included
    t_max = np.reshape(jax.vmap(jax.grad(phi_load))(flat), np.shape(dm))
    return t_max / dm, phi_load(dm), dm


def _with_unloading(phi_load, delta, delta_max, half_width):
    """Irreversibility: unloading from ``δmax`` on the secant to the origin,
    ``t = (t_L(δmax)/δmax) δ``, i.e. the potential
    ``φ_L(δmax) + ½ k_sec (δ² − δmax²)`` — continuous in value AND traction
    at the switch (a secant built from ``φ_L(δmax)/δmax²`` instead would jump
    in traction once past the peak and break Newton).

    The curvature still jumps at ``δmax``; ``half_width > 0`` blends the
    switch over an absolute band ``δmax ± half_width`` for chords that have
    opened beyond the band (barely opened chords stay on the loading branch).
    """
    k_sec, phi_max, dm = _secant(phi_load, delta_max)
    phi_un = phi_max + 0.5 * k_sec * (delta ** 2 - dm ** 2)
    if half_width > 0.0:
        s_load = np.where(delta_max > 2.0 * half_width, _blend(delta, delta_max, half_width), 1.0)
    else:
        s_load = np.where(delta >= delta_max - 1e-15, 1.0, 0.0)
    return s_load * phi_load(delta) + (1.0 - s_load) * phi_un


def bilinear_tsl(delta, delta_max, *, K, sigma_c, Gamma, smooth=0.0):
    """Bilinear traction-separation potential with irreversibility (NLXP3D /
    Camanho style), as a function of the effective opening ``delta``.

    .. code-block:: text

        t(δ) = K δ                               δ < δ0 = σc / K      (elastic)
        t(δ) = σc (δf − δ) / (δf − δ0)           δ0 ≤ δ < δf          (softening)
        t(δ) = 0                                 δ ≥ δf = 2Γ / σc     (failed)

    ``φ = ∫ t dδ`` is returned; unloading from ``delta_max`` is a secant to
    the origin (``φ = φ(δmax)/δmax² · δ²``).  ``K`` is the initial (penalty)
    stiffness, ``sigma_c`` the strength, ``Gamma`` the fracture energy;
    ``δf > δ0`` is required (``2ΓK > σc²``).  ``smooth > 0`` blends the kinks
    of the law (at ``δ0``, ``δf`` and the loading/unloading switch) over the
    absolute band ``± smooth δ0`` (continuous tangent); the default is the
    exact piecewise law.  For robust load stepping prefer the secant mode of
    :class:`CohesiveZone` (``secant=True``) over smoothing.
    """
    d0 = sigma_c / K
    df = 2.0 * Gamma / sigma_c
    hw = smooth * d0

    def phi_load(d):
        elastic = 0.5 * K * d ** 2
        soft = Gamma - 0.5 * sigma_c * (df - d) ** 2 / (df - d0)
        s0, sf = _blend(d, d0, hw), _blend(d, df, hw)
        return (1.0 - s0) * elastic + s0 * ((1.0 - sf) * soft + sf * Gamma)

    return _with_unloading(phi_load, delta, delta_max, hw)


def exponential_tsl(delta, delta_max, *, sigma_c, Gamma, smooth=0.0):
    """Xu–Needleman exponential potential ``φ = Γ[1 − (1 + δ/δc) e^{−δ/δc}]``,
    ``δc = Γ/(e σc)``, with secant unloading from ``delta_max`` (the loading
    law of :func:`feax.mechanics.cohesive.exponential_potential`; the
    loading/unloading switch is blended over ``± smooth δc``, see
    :func:`bilinear_tsl`)."""
    delta_c = Gamma / (np.e * sigma_c)

    def phi_load(d):
        return Gamma * (1.0 - (1.0 + d / delta_c) * np.exp(-d / delta_c))

    return _with_unloading(phi_load, delta, delta_max, smooth * delta_c)


class CohesiveZone:
    """Cohesive zone model integrated on the cut chord of the enriched cells.

    The displacement jump ``[[u]] = u⁺ − u⁻`` across the interface (``u±`` the
    limits from ``phi ≷ 0``) is split into the normal opening ``δn = [[u]]·n``
    (``n = ∇phi/|∇phi|``, pointing to ``phi > 0``) and the sliding ``δt``, and
    the effective opening ``δ = sqrt(⟨δn⟩₊² + β² δt²)`` drives the potential
    ``tsl(δ, δmax, **params)`` (see :func:`bilinear_tsl`,
    :func:`exponential_tsl`, or any ``feax.mechanics.cohesive`` potential).
    ``contact_penalty`` adds ``½ kp ⟨−δn⟩₊²`` against interpenetration.

    The interface energy is added to the element kernel and its tangent comes
    from autodiff, so the tangent stays symmetric.  Irreversibility needs the
    history ``delta_max`` (max effective opening so far, per interface Gauss
    point): pass it through :meth:`XFEMSpace.traced_params` and update it
    after each converged load step with :meth:`XFEMSpace.update_history`.

    Parameters
    ----------
    tsl : callable  ``tsl(delta, delta_max, **params) -> energy per unit length``
    beta : float    mode-mixity weight of the sliding (default 1.0)
    contact_penalty : float   penalty stiffness for ``δn < 0`` (default 0: none)
    n_gauss : int   Gauss points per chord (default 2)
    on : str        ``"interface"`` (default, the ``psi < 0`` cells) or ``"cut"``
                    (every sub-divided cell, i.e. including the tip cell)
    viscosity : float   viscous regularisation (see :meth:`energy_density`)
    secant : bool   use the secant potential ``k_sec(δmax) δ²`` with the damage
                    frozen during a solve (explicit damage update); the
                    history is advanced between solves and the solve repeated
                    until it is consistent (:meth:`XFEMSpace.solve_step`).
                    Robust where the implicit loading/unloading switch makes
                    Newton stall (many chords at ``δ = δmax`` at a step start).
    **params        forwarded to ``tsl``
    """

    def __init__(self, tsl: Callable, *, beta: float = 1.0, contact_penalty: float = 0.0,
                 n_gauss: int = 2, on: str = "interface", viscosity: float = 0.0,
                 secant: bool = False, **params):
        if on not in ("interface", "cut"):
            raise ValueError("on must be 'interface' or 'cut'")
        self.tsl = tsl
        self.beta = float(beta)
        self.contact_penalty = float(contact_penalty)
        self.n_gauss = int(n_gauss)
        self.on = on
        self.viscosity = float(viscosity)
        self.secant = bool(secant)
        self.params = params

    def secant_potential(self, delta, delta_max):
        """``φ_L(δmax) + ½ k_sec (δ² − δmax²)`` with ``k_sec = t_L(δmax)/δmax``
        (``K`` for a virgin chord): the cohesive potential with the damage
        frozen at the history ``delta_max`` (same traction as the loading law
        at ``δmax``).  Quadratic in the opening, so a Newton step on it is
        essentially linear; the history is advanced *between* solves
        (explicit damage update, see :meth:`XFEMSpace.solve_step`)."""
        dm = np.maximum(delta_max, 1e-12)
        phi_load = lambda d: self.tsl(d, d, **self.params)        # loading branch (δ = δmax)
        k_sec, phi_max, dm = _secant(phi_load, dm)
        return phi_max + 0.5 * k_sec * (delta ** 2 - dm ** 2)

    def opening(self, jump, normal):
        """``(delta_eff, delta_n, delta_t)`` from jumps (g, 2) and normals (g, 2)."""
        dn = np.sum(jump * normal, axis=-1)
        # squared sliding (no norm: its gradient at a closed interface is NaN)
        dt_sq = np.sum((jump - dn[:, None] * normal) ** 2, axis=-1)
        d = np.sqrt(np.maximum(dn, 0.0) ** 2 + self.beta ** 2 * dt_sq + 1e-30)
        return d, dn, np.sqrt(dt_sq + 1e-30)

    def energy_density(self, jump, normal, delta_max, scale=1.0, delta_prev=None):
        """Cohesive energy per unit length at the Gauss points.

        ``scale`` multiplies the traction-separation potential (strength,
        stiffness and toughness together, so ``δ0`` and ``δf`` are
        unchanged) — a per-chord imperfection that lets damage localise
        instead of bifurcating when many identical interfaces reach their
        peak simultaneously.  ``delta_prev`` (effective opening at the previous
        converged step) activates the viscous regularisation
        ``½ η (δ − δ_prev)²`` per step (``viscosity`` = η, a traction per unit
        opening increment): it adds a positive stiffness ``η`` to every chord,
        which keeps Newton from cycling between the loading and the secant
        (unloading) branch when many chords sit at their peak, and
        regularises snap-backs; keep ``η Δδ`` small against ``σc``.
        """
        d, dn, _ = self.opening(jump, normal)
        if self.secant:
            e = scale * self.secant_potential(d, delta_max)
        else:
            e = scale * self.tsl(d, delta_max, **self.params)
        if self.contact_penalty > 0.0:
            e = e + 0.5 * self.contact_penalty * np.minimum(dn, 0.0) ** 2
        if self.viscosity > 0.0 and delta_prev is not None:
            e = e + 0.5 * self.viscosity * (d - delta_prev) ** 2
        return e

    def traction(self, delta, delta_max):
        """Effective traction ``t = dφ/dδ`` (vectorised over Gauss points)."""
        return jax.vmap(jax.grad(lambda d, dm: self.tsl(d, dm, **self.params)))(delta, delta_max)


# ---------------------------------------------------------------------------
# Reference shape functions (evaluated at arbitrary reference points)
# ---------------------------------------------------------------------------

def _shape_tri3(xi):
    """Unit triangle, nodes (0,0), (1,0), (0,1).  xi: (nq, 2)."""
    r, s = xi[:, 0], xi[:, 1]
    N = np.stack([1.0 - r - s, r, s], axis=1)
    dN = np.broadcast_to(np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]), (xi.shape[0], 3, 2))
    return N, dN


def _shape_quad4(xi):
    """Square [-1,1]^2, nodes (-1,-1), (1,-1), (1,1), (-1,1) (meshio order)."""
    r, s = xi[:, 0], xi[:, 1]
    sr = np.array([-1.0, 1.0, 1.0, -1.0])
    ss = np.array([-1.0, -1.0, 1.0, 1.0])
    N = 0.25 * (1.0 + sr[None, :] * r[:, None]) * (1.0 + ss[None, :] * s[:, None])
    dNdr = 0.25 * sr[None, :] * (1.0 + ss[None, :] * s[:, None])
    dNds = 0.25 * ss[None, :] * (1.0 + sr[None, :] * r[:, None])
    return N, np.stack([dNdr, dNds], axis=-1)


_SHAPE_FNS = {"TRI3": _shape_tri3, "QUAD4": _shape_quad4}

# Reference vertices and the triangulation used for the cut quadrature.
_REF_TRI3 = onp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
_REF_QUAD4 = onp.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


def _triangle_rule(npts: int) -> Tuple[onp.ndarray, onp.ndarray]:
    """Quadrature on the unit triangle, weights summing to its area 1/2."""
    if npts == 1:
        return onp.array([[1 / 3, 1 / 3]]), onp.array([0.5])
    if npts == 3:
        return (onp.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]]),
                onp.array([1 / 6, 1 / 6, 1 / 6]))
    if npts == 6:   # Dunavant degree 4
        a, b = 0.445948490915965, 0.091576213509771
        wa, wb = 0.223381589678011, 0.109951743655322
        pts = onp.array([[a, a], [1 - 2 * a, a], [a, 1 - 2 * a],
                         [b, b], [1 - 2 * b, b], [b, 1 - 2 * b]])
        return pts, 0.5 * onp.array([wa, wa, wa, wb, wb, wb])
    raise ValueError("tri_rule must be 1, 3 or 6 points")


def _cut_triangle(v: onp.ndarray, p: onp.ndarray) -> list:
    """Split triangle ``v`` (3, 2) by the zero line of the linear field ``p`` (3,).

    Returns the list of sub-triangles (each (3, 2)); ``[v]`` if uncut.  ``p``
    must have no exact zeros (the caller clamps them).
    """
    s = onp.sign(p)
    if s[0] == s[1] == s[2]:
        return [v]
    # the lone node is the one whose sign differs from the other two
    i0 = int([i for i in range(3) if s[(i + 1) % 3] == s[(i + 2) % 3]][0])
    i1, i2 = (i0 + 1) % 3, (i0 + 2) % 3
    t1 = p[i0] / (p[i0] - p[i1])
    t2 = p[i0] / (p[i0] - p[i2])
    P1 = v[i0] + t1 * (v[i1] - v[i0])
    P2 = v[i0] + t2 * (v[i2] - v[i0])
    return [onp.array([v[i0], P1, P2]),
            onp.array([P1, v[i1], v[i2]]),
            onp.array([P1, v[i2], P2])]


def _sub_triangle_points(tri: onp.ndarray, rule_pts, rule_w):
    """Map a unit-triangle rule onto sub-triangle ``tri`` (3, 2) in parent coords."""
    e1, e2 = tri[1] - tri[0], tri[2] - tri[0]
    pts = tri[0][None, :] + rule_pts[:, :1] * e1[None, :] + rule_pts[:, 1:] * e2[None, :]
    det = abs(e1[0] * e2[1] - e1[1] * e2[0])
    return pts, rule_w * det


def _cell_triangulation(ele_type: str, p: onp.ndarray, threshold: float = 1e-3) -> list:
    """Triangles (parent reference coords) + nodal values covering one cell.

    QUAD4 is split into 4 triangles about its centre, where the (bilinear)
    level set takes the mean of the nodal values; a centre value closer to
    zero than ``threshold`` times the nodal scale is nudged off zero exactly
    like the nodal clamp, so the linear sub-triangle cut never degenerates.
    """
    if ele_type == "TRI3":
        return [(_REF_TRI3, p)]
    if ele_type == "QUAD4":
        c = onp.zeros(2)
        pc = p.mean()
        eps = threshold * onp.abs(p).max()
        if abs(pc) < eps:
            pc = eps if pc >= 0 else -eps
        out = []
        for k in range(4):
            k2 = (k + 1) % 4
            out.append((onp.array([c, _REF_QUAD4[k], _REF_QUAD4[k2]]),
                        onp.array([pc, p[k], p[k2]])))
        return out
    raise NotImplementedError(f"XFEMSpace supports TRI3 and QUAD4, got {ele_type}")


_EDGES = {"TRI3": [(0, 1), (1, 2), (2, 0)], "QUAD4": [(0, 1), (1, 2), (2, 3), (3, 0)]}
_REF_VERTS = {"TRI3": _REF_TRI3, "QUAD4": _REF_QUAD4}


def _chord(ele_type: str, p: onp.ndarray):
    """End points (reference coords) of the zero line of the nodal field ``p``
    on the cell boundary, and the edge parameters.  Two points for a cell
    crossed once; a saddle (4 crossings) keeps the first and last."""
    verts = _REF_VERTS[ele_type]
    pts, ts = [], []
    for i, j in _EDGES[ele_type]:
        if onp.sign(p[i]) != onp.sign(p[j]):
            t = p[i] / (p[i] - p[j])
            pts.append(verts[i] + t * (verts[j] - verts[i]))
            ts.append(t)
    pts = onp.array(pts)
    return onp.array([pts[0], pts[-1]]), ts


def _shape_np(ele_type: str, xi) -> onp.ndarray:
    """Host (numpy) shape functions at reference points xi (m, 2) -> (m, nn).
    Used by the host-side geometry, where per-call JAX dispatch would dominate."""
    xi = onp.asarray(xi, dtype=onp.float64).reshape(-1, 2)
    r, s = xi[:, 0], xi[:, 1]
    if ele_type == "TRI3":
        return onp.stack([1.0 - r - s, r, s], axis=1)
    sr = onp.array([-1.0, 1.0, 1.0, -1.0])
    ss = onp.array([-1.0, -1.0, 1.0, 1.0])
    return 0.25 * (1.0 + sr[None, :] * r[:, None]) * (1.0 + ss[None, :] * s[:, None])


def _interp_cell_batch(ele_type: str, xi: onp.ndarray, nodal: onp.ndarray) -> onp.ndarray:
    """Row-wise interpolation: xi (m, 2), nodal (m, nn, k) -> (m, k)."""
    return onp.einsum("ma,mak->mk", _shape_np(ele_type, xi), nodal)


def _interp_cell(ele_type: str, xi: onp.ndarray, nodal: onp.ndarray) -> onp.ndarray:
    """Interpolate nodal values (nn, k) at reference points xi (m, 2) -> (m, k)."""
    return _shape_np(ele_type, xi) @ nodal


# ---------------------------------------------------------------------------
# XFEMSpace: level set -> cut cells, enriched nodes, enrichment mesh, quadrature
# ---------------------------------------------------------------------------

class XFEMSpace:
    """Host-side XFEM preprocessing for one level set on a 2D mesh.

    Parameters
    ----------
    mesh : Mesh
        The (non-conforming) background mesh; variable 0 lives on it.
    phi : array (num_nodes,)
        Nodal level set; the interface is ``phi = 0``.  Values closer to zero
        than ``threshold`` times the local scale are pushed away from zero
        (sign preserved, ``phi = 0`` goes positive) to avoid sliver sub-cells —
        the same role as NLXP3D's ``threshold``.  The clamped copy is used
        consistently for both the cut quadrature and the kernel.
    psi : array (num_nodes,), optional
        Tangential level set limiting the *extent* of the interface, as in
        NLXP3D: the interface is the part of ``phi = 0`` where ``psi < 0``.
        A cell crossed by ``phi = 0`` is an *interface cell* (its nodes get
        enriched) when ``psi < 0`` at the midpoint of the cut chord.  With a
        sign-type enrichment and no tip functions the jump does not stop at
        the chord of the last interface cell: it tapers linearly to zero
        across the next crossed cell (that cell is sub-divided for quadrature
        too, see ``cut_cells``), so the effective tip is up to one cell beyond
        ``psi = 0``.  ``None`` (default): the whole ``phi = 0`` line.
    enrichment : Enrichment
        ``ψ_j`` (see the defaults in this module).
    vec : int
        Components of the displacement (2 for 2D elasticity).
    ele_type : str
        ``"TRI3"`` or ``"QUAD4"``.
    cohesive : CohesiveZone, optional
        Cohesive zone model on the interface chord of the enriched cells
        (``iqpts`` / ``iqw``: its Gauss points in reference coords and physical
        length weights, zero for cells without an interface).  Needs a
        sign-type enrichment (a jump); with a continuous enrichment the
        interface never opens and the term vanishes.
    tri_rule : int
        Points of the triangle rule used on every sub-triangle (1, 3 or 6).
        Uncut cells are triangulated the same way (QUAD4: 4 triangles about
        the centre), so the rule is uniform across the mesh.
    threshold : float
        Relative clamp for near-zero nodal level-set values.

    Attributes
    ----------
    interface_cells : bool (num_cells,)
        Cells crossed by the interface (within its ``psi`` extent); their
        nodes are enriched.
    cut_cells : bool (num_cells,)
        Cells integrated with the interface-conforming sub-triangulation:
        every cell crossed by ``phi = 0`` that holds at least one enriched
        node (``interface_cells`` plus the tip cell(s)).
    segments : float (num_cells, 2, 2)
        Physical end points of the cut chord of each ``cut_cells`` cell
        (``nan`` elsewhere); ``segments_ref`` holds them in reference coords.
    enriched_mask : bool (num_nodes,)
    local : int (num_nodes,)
        Node of the enrichment mesh for each background node (``phantom``
        where not enriched).
    emesh : Mesh
        Enrichment mesh: enriched nodes (real coordinates) + the phantom node
        (coordinates ``nan`` so no location-based BC ever selects it).
    qpts, qw : (num_cells, nq_max, 2), (num_cells, nq_max)
        Sub-cell quadrature in parent reference coordinates (weights are
        reference-area weights; the kernel multiplies by ``|det J|``), zero-
        padded to ``nq_max``.
    """

    def __init__(self, mesh: Mesh, phi, enrichment: Enrichment, vec: int = 2,
                 ele_type: str = "TRI3", tri_rule: int = 3, threshold: float = 1e-3,
                 psi=None, cohesive: Optional[CohesiveZone] = None):
        if ele_type not in _SHAPE_FNS:
            raise NotImplementedError(f"XFEMSpace supports TRI3 and QUAD4, got {ele_type}")
        self.mesh = mesh
        self.ele_type = ele_type
        self.enrichment = enrichment
        self.vec = int(vec)
        self.dim = 2
        self.tri_rule = int(tri_rule)
        self.threshold = float(threshold)

        points = onp.asarray(mesh.points, dtype=onp.float64)[:, :2]
        cells = onp.asarray(mesh.cells, dtype=onp.int64)
        self.points = points
        self.cells = cells
        self.num_cells, self.num_nodes_per_cell = cells.shape
        self.num_nodes = points.shape[0]

        self.phi = self._clamp_phi(onp.asarray(phi, dtype=onp.float64).reshape(-1), cells)

        # --- interface cells, enriched nodes, cut (sub-divided) cells -------
        sgn = onp.sign(self.phi[cells])                                   # (C, nn), no zeros
        crossed = onp.any(sgn != sgn[:, :1], axis=1)
        self.segments_ref = onp.full((self.num_cells, 2, 2), onp.nan)
        self.segments = onp.full((self.num_cells, 2, 2), onp.nan)
        chord_psi = onp.full(self.num_cells, -1.0)
        self.psi = None if psi is None else onp.asarray(psi, dtype=onp.float64).reshape(-1)
        if self.psi is not None and self.psi.shape[0] != self.num_nodes:
            raise ValueError(f"psi has {self.psi.shape[0]} entries for {self.num_nodes} nodes")
        for c in onp.flatnonzero(crossed):
            xi_pts, t_pts = _chord(ele_type, self.phi[cells[c]])
            self.segments_ref[c] = xi_pts
            self.segments[c] = _interp_cell(ele_type, xi_pts, points[cells[c]])
            if self.psi is not None:
                chord_psi[c] = _interp_cell(ele_type, xi_pts, self.psi[cells[c]][:, None]).mean()
        self.interface_cells = crossed & (chord_psi < 0.0)
        self.enriched_mask = onp.zeros(self.num_nodes, dtype=bool)
        self.enriched_mask[onp.unique(cells[self.interface_cells])] = True
        self.num_enriched = int(self.enriched_mask.sum())
        self.cut_cells = crossed & onp.any(self.enriched_mask[cells], axis=1)
        self.segments_ref[~self.cut_cells] = onp.nan
        self.segments[~self.cut_cells] = onp.nan

        # --- enrichment mesh: enriched nodes + phantom -----------------------
        self.phantom = self.num_enriched
        self.local = onp.full(self.num_nodes, self.phantom, dtype=onp.int32)
        self.local[self.enriched_mask] = onp.arange(self.num_enriched, dtype=onp.int32)
        epoints = onp.full((self.num_enriched + 1, 2), onp.nan)
        epoints[:self.num_enriched] = points[self.enriched_mask]
        self.emesh = Mesh(epoints, self.local[cells], ele_type=ele_type)

        # --- sub-cell quadrature -------------------------------------------
        self.qpts, self.qw = self._build_quadrature()

        # --- interface quadrature for the cohesive zone --------------------
        self.cohesive = cohesive
        if cohesive is not None:
            self.iqpts, self.iqw = self._build_interface_quadrature(cohesive)
        self.nq_max = self.qpts.shape[1]

        # Pre-gathered per-cell internal variables (cell-major so feax passes
        # them through untouched, whatever the node / cell counts are).
        self._cell_X = points[cells]                                       # (C, nn, 2)
        self._cell_X_np = onp.asarray(self._cell_X)
        self._remesh_cache = None
        self._cell_phi = self.phi[cells]                                   # (C, nn)
        self._cell_mask = self.enriched_mask[cells].astype(onp.float64)    # (C, nn)

    # ------------------------------------------------------------------ build
    def _clamp_phi(self, phi, cells):
        if phi.shape[0] != self.num_nodes:
            raise ValueError(f"phi has {phi.shape[0]} entries for {self.num_nodes} nodes")
        cell_scale = onp.abs(phi[cells]).max(axis=1)                       # (C,)
        node_scale = onp.zeros(self.num_nodes)
        onp.maximum.at(node_scale, cells.reshape(-1), onp.repeat(cell_scale, cells.shape[1]))
        if not onp.all(node_scale > 0):
            raise ValueError("phi is identically zero around some node")
        eps = self.threshold * node_scale
        small = onp.abs(phi) < eps
        out = phi.copy()
        out[small] = onp.where(phi[small] >= 0, eps[small], -eps[small])
        return out

    def _build_quadrature(self):
        rule_pts, rule_w = _triangle_rule(self.tri_rule)
        per_cell = []
        for c in range(self.num_cells):
            p = self.phi[self.cells[c]]
            pts, ws = [], []
            for tri, ptri in _cell_triangulation(self.ele_type, p, self.threshold):
                subs = _cut_triangle(tri, ptri) if self.cut_cells[c] else [tri]
                for sub in subs:
                    xq, wq = _sub_triangle_points(sub, rule_pts, rule_w)
                    pts.append(xq)
                    ws.append(wq)
            per_cell.append((onp.concatenate(pts), onp.concatenate(ws)))
        nq_max = max(len(w) for _, w in per_cell)
        qpts = onp.zeros((self.num_cells, nq_max, 2))
        qw = onp.zeros((self.num_cells, nq_max))
        for c, (xq, wq) in enumerate(per_cell):
            qpts[c, :len(wq)] = xq
            qw[c, :len(wq)] = wq
        return qpts, qw

    def _build_interface_quadrature(self, cz: CohesiveZone):
        """Gauss points on the straight chord of each cohesive cell: reference
        coordinates (C, g, 2) and physical length weights (C, g)."""
        active = self.interface_cells if cz.on == "interface" else self.cut_cells
        g, w = onp.polynomial.legendre.leggauss(cz.n_gauss)       # on [-1, 1]
        iqpts = onp.zeros((self.num_cells, cz.n_gauss, 2))
        iqw = onp.zeros((self.num_cells, cz.n_gauss))
        for c in onp.flatnonzero(active):
            a, b = self.segments_ref[c]
            iqpts[c] = a[None, :] + 0.5 * (g[:, None] + 1.0) * (b - a)[None, :]
            length = onp.linalg.norm(self.segments[c, 1] - self.segments[c, 0])
            iqw[c] = 0.5 * length * w
        self.cohesive_cells = active
        return iqpts, iqw

    @property
    def num_interface_gauss(self) -> int:
        return 0 if self.cohesive is None else self.cohesive.n_gauss

    def zero_history(self):
        """Initial ``delta_max`` (num_cells, n_gauss) for the cohesive zone."""
        return np.zeros((self.num_cells, self.num_interface_gauss))

    # ------------------------------------------------------------- interface
    @property
    def vec_enriched(self) -> int:
        return self.vec * self.enrichment.n_funcs

    def problem_kwargs(self) -> dict:
        """Constructor arguments for the two-variable :class:`~feax.Problem`."""
        return dict(mesh=[self.mesh, self.emesh], vec=[self.vec, self.vec_enriched],
                    dim=self.dim, ele_type=[self.ele_type, self.ele_type],
                    with_jxw=[True, False])

    def traced_params(self, *extra, delta_max=None, cohesive_scale=None, delta_prev=None) -> TracedParams:
        """``TracedParams`` carrying the geometry, level set and quadrature.

        ``extra`` arrays are forwarded to the material as cell-level arguments:
        each must have leading dimension ``num_cells`` (or be ≥3-D, which feax
        passes through as-is) and is given to ``material`` per cell.
        With a cohesive zone, ``delta_max`` (num_cells, n_gauss) is the opening
        history (default: :meth:`zero_history`) and ``cohesive_scale``
        (num_cells,) an optional per-chord scaling of the cohesive potential
        (default 1; see :meth:`CohesiveZone.energy_density`); ``delta_prev``
        (num_cells, n_gauss) the opening at the previous converged step for
        the viscous regularisation (:meth:`current_opening`).
        """
        base = (np.asarray(self._cell_X), np.asarray(self._cell_phi), np.asarray(self._cell_mask),
                np.asarray(self.qpts), np.asarray(self.qw))
        if self.cohesive is not None:
            dm = self.zero_history() if delta_max is None else np.asarray(delta_max)
            sc = np.ones(self.num_cells) if cohesive_scale is None else np.asarray(cohesive_scale, dtype=np.float64)
            dp = self.zero_history() if delta_prev is None else np.asarray(delta_prev)
            base = base + (np.asarray(self.iqpts), np.asarray(self.iqw), dm, sc, dp)
        return TracedParams(volume_vars=base + tuple(extra))

    def interface_state(self, problem: Problem, sol, delta_max=None):
        """Cohesive-zone state at the interface Gauss points of the cohesive
        cells: dict with ``x`` (n, 2), ``delta`` (effective opening), ``delta_n``,
        ``delta_t``, ``traction`` (effective, from ``delta_max``; default
        zero history), ``cell`` (cell index), all (n_cohesive_cells * n_gauss,).
        """
        cz = self.cohesive
        if cz is None:
            raise ValueError("this space has no cohesive zone")
        cells = onp.flatnonzero(self.cohesive_cells)
        idx = np.asarray(cells)
        cs = self.cell_solution(problem, sol)[idx]
        X, phi, mask = (np.asarray(v)[idx] for v in (self._cell_X, self._cell_phi, self._cell_mask))
        ixi = np.asarray(self.iqpts)[idx]
        dm = (self.zero_history() if delta_max is None else np.asarray(delta_max))[idx]
        jump_fn = _interface_jump_fn(self, problem.unflatten_fn_dof)
        x_g, jump, normal = jax.vmap(jump_fn)(cs, X, phi, mask, ixi)      # (nc, g, ..)
        g = cz.n_gauss
        d, dn, dt = cz.opening(jump.reshape(-1, 2), normal.reshape(-1, 2))
        t = cz.traction(d, dm.reshape(-1))
        return dict(x=x_g.reshape(-1, 2), delta=d, delta_n=dn, delta_t=dt, traction=t,
                    cell=onp.repeat(cells, g))

    def current_opening(self, problem: Problem, sol):
        """Effective opening at the interface Gauss points as a (num_cells,
        n_gauss) array (zero outside the cohesive cells) — the ``delta_prev``
        of the next step for the viscous regularisation."""
        st = self.interface_state(problem, sol)
        cur = np.zeros((self.num_cells, self.cohesive.n_gauss))
        return cur.at[np.asarray(st["cell"]).reshape(-1, self.cohesive.n_gauss)[:, 0]].set(
            st["delta"].reshape(-1, self.cohesive.n_gauss))

    def update_history(self, delta_max, problem: Problem, sol):
        """``max(delta_max, current effective opening)`` (num_cells, n_gauss)."""
        return np.maximum(np.asarray(delta_max), self.current_opening(problem, sol))

    def solve_step(self, problem: Problem, solver: Callable, bc, ts, u0, delta_max, *extra,
                   cohesive_scale=None, max_outer: int = 30, tol: float = 1e-6, relax: float = 1.0):
        """One load step with an explicit damage update (secant cohesive zone).

        Solves with the history ``delta_max`` frozen, advances the history
        with the new openings (``max``), and repeats until the history no
        longer changes (relative ``tol`` on the opening scale).  ``relax < 1``
        damps the update when it oscillates (snap-through).  Returns
        ``(sol, delta_max, n_outer)``.  With ``secant=False`` cohesive zones
        the single implicit solve is returned unchanged.
        """
        dm = np.asarray(delta_max)
        sol = None
        for k in range(1, max_outer + 1):
            tp = self.traced_params(*extra, delta_max=dm, cohesive_scale=cohesive_scale, delta_prev=dm)
            sol = solver(tp, u0, bc=bc, traced_structure=ts).dofs
            if not self.cohesive.secant:
                return sol, self.update_history(dm, problem, sol), 1
            cur = self.current_opening(problem, sol)
            dm_new = np.maximum(dm, cur)
            dm_next = dm + relax * (dm_new - dm)
            change = float(np.max(np.abs(dm_next - dm)))
            scale = float(np.max(np.abs(dm_new))) + 1e-30
            dm = dm_next
            u0 = sol
            if change <= tol * scale:
                return sol, dm, k
        return sol, dm, max_outer

    def phantom_rows(self, problem: Problem) -> onp.ndarray:
        """Global DOF rows of the phantom node (always Dirichlet-fixed to 0)."""
        base = problem.offset[1] + self.phantom * self.vec_enriched
        return base + onp.arange(self.vec_enriched)

    def create_bc(self, problem: Problem, config: Optional[DirichletBCConfig] = None) -> DirichletBC:
        """Dirichlet BC from ``config`` (on variable 0 and/or 1) plus the phantom rows."""
        base = (config if config is not None else DirichletBCConfig([])).create_bc(problem)
        rows = onp.concatenate([onp.asarray(base.bc_rows), self.phantom_rows(problem)])
        vals = onp.concatenate([onp.asarray(base.bc_vals), onp.zeros(self.vec_enriched)])
        order = onp.argsort(rows, kind="stable")
        rows, vals = rows[order], vals[order]
        rows, first = onp.unique(rows, return_index=True)
        vals = vals[first]
        mask = onp.zeros(problem.num_total_dofs_all_vars, dtype=bool)
        mask[rows] = True
        return DirichletBC(bc_rows=np.asarray(rows, dtype=np.int32), bc_mask=np.asarray(mask),
                           bc_vals=np.asarray(vals), total_dofs=problem.num_total_dofs_all_vars)

    # --------------------------------------------------------- post-process
    def split(self, problem: Problem, sol):
        """``(u, a)``: standard nodal values (num_nodes, vec) and enriched values
        (num_enriched + 1, n_funcs, vec) (last row = phantom)."""
        u, a = problem.unflatten_fn_sol_list(sol)
        return u, a.reshape(-1, self.enrichment.n_funcs, self.vec)

    def nodal_displacement(self, problem: Problem, sol):
        """Total displacement at the nodes, ``u_j + ψ_j(x_j) a_j`` (num_nodes, vec)."""
        u, a = self.split(problem, sol)
        psi_n = self.enrichment.at_nodes(np.asarray(self.phi))             # (N, nf)
        a_n = a[np.asarray(self.local)] * np.asarray(self.enriched_mask)[:, None, None]
        return u + np.einsum("nf,nfi->ni", psi_n, a_n)

    def cell_solution(self, problem: Problem, sol):
        """Per-cell flat solution in the kernel's layout (num_cells, nn*(vec+vec_enriched))."""
        u, a = problem.unflatten_fn_sol_list(sol)
        cu = u[np.asarray(self.cells)].reshape(self.num_cells, -1)
        ca = a[np.asarray(self.emesh.cells)].reshape(self.num_cells, -1)
        return np.concatenate([cu, ca], axis=1)

    def interface_jump(self, problem: Problem, sol, eps: float = 1e-12):
        """Displacement jump ``u⁺ − u⁻`` across the interface at the midpoint
        of the cut chord of every ``cut_cells`` cell.

        Returns ``(x_mid, jump)``: (num_cut, 2) physical points and
        (num_cut, vec) jumps (zero for continuous enrichments).  ``u⁺`` /
        ``u⁻`` are the limits from ``phi > 0`` / ``phi < 0``, evaluated by
        calling the enrichment with ``phi = ±eps``.
        """
        cut = onp.flatnonzero(self.cut_cells)
        xi_mid = np.asarray(self.segments_ref[cut].mean(axis=1))          # (nc, 2)
        cs = self.cell_solution(problem, sol)[np.asarray(cut)]
        X = np.asarray(self._cell_X)[np.asarray(cut)]
        phi_n = np.asarray(self._cell_phi)[np.asarray(cut)]
        mask = np.asarray(self._cell_mask)[np.asarray(cut)]
        shape_fn = _SHAPE_FNS[self.ele_type]
        enrich = self.enrichment
        nn, vec, nf = self.num_nodes_per_cell, self.vec, self.enrichment.n_funcs
        unflatten = problem.unflatten_fn_dof

        def one(cs_c, X_c, phi_c, m_c, xi_c):
            _, a = unflatten(cs_c)
            a = a.reshape(nn, nf, vec)
            N, dNdxi = shape_fn(xi_c[None, :])
            dNdx = np.einsum("qaI,qIi->qai", dNdxi, np.linalg.inv(np.einsum("ai,qaI->qiI", X_c, dNdxi)))
            gphi = np.einsum("qai,a->qi", dNdx, phi_c)
            side = lambda sgn: enrich(np.full((1,), sgn * eps), gphi, phi_c, N, dNdx)[0][0] * m_c[:, None]
            dpsi = side(+1.0) - side(-1.0)                                 # (nn, nf)
            return np.einsum("a,af,afi->i", N[0], dpsi, a), N[0] @ X_c

        jump, x_mid = jax.vmap(one)(cs, X, phi_n, mask, xi_mid)
        return x_mid, jump

    # --------------------------------------------------------- crack-aware remesh
    def _remesh_topology(self):
        """Interface-conforming sub-triangulation (host, cached).

        Returns per sub-triangle: parent cell, reference vertex coords (3, 2),
        side (+1/-1 of phi at the centroid), and per vertex a flag whether it
        lies on the interface of a cut cell (phi = 0 there).
        """
        if getattr(self, "_remesh_cache", None) is not None:
            return self._remesh_cache
        parent, ref, side, on_if = [], [], [], []
        for c in range(self.num_cells):
            p = self.phi[self.cells[c]]
            for tri, ptri in _cell_triangulation(self.ele_type, p, self.threshold):
                subs = _cut_triangle(tri, ptri) if self.cut_cells[c] else [tri]
                for sub in subs:
                    lam = _interp_cell(self.ele_type, sub, p[:, None])[:, 0]          # phi at the 3 vertices
                    cen = _interp_cell(self.ele_type, sub.mean(0, keepdims=True), p[:, None])[0, 0]
                    parent.append(c)
                    ref.append(sub)
                    side.append(1.0 if cen >= 0 else -1.0)
                    on_if.append(self.cut_cells[c] & (onp.abs(lam) <= 1e-9 * (onp.abs(p).max() + 1e-300)))
        self._remesh_cache = (onp.asarray(parent), onp.asarray(ref), onp.asarray(side), onp.asarray(on_if))
        return self._remesh_cache

    def remesh(self, problem: Problem, sol, *extra, split: str = "all", delta_max=None,
               damage_threshold: float = 1.0, nudge: float = 1e-6) -> dict:
        """Crack-aware conforming remesh of the XFEM solution.

        Every cell crossed by the interface is split along it (the same
        sub-triangulation as the quadrature); vertices are shared between
        neighbouring triangles *except* across the interface, where they are
        duplicated — one copy per side — so the displacement jump becomes a
        real gap in the mesh.  ``split`` selects where the mesh is opened:

        * ``"all"``: along every interface (default);
        * ``"damaged"``: only along chords whose cohesive damage
          (``max δmax / δf`` over the chord's Gauss points, needs a cohesive
          zone and ``delta_max``) reaches ``damage_threshold`` — i.e. the
          cracks; intact interfaces stay connected (their vertices take the
          average of the two sides);
        * ``"none"``: a plain conforming refinement.

        Fields are evaluated at the vertices nudged by ``nudge`` toward the
        sub-triangle centroid (picks the correct side on the interface).

        Returns host numpy arrays: ``points`` (P, 2), ``cells`` (T, 3),
        ``u`` (P, vec), ``phi`` (P,), ``parent`` (T,), ``side`` (T,),
        ``grad_u`` (T, 2, 2) at the centroids, ``damage`` (T,) of the parent
        chord (0 where none), ``opened`` (T,) bool, and ``crack_lines``
        (L, 2) vertex pairs along the opened interface (both faces).
        Write it with :func:`feax.utils.save_sol` on ``Mesh(points, cells, "TRI3")``.
        """
        if split not in ("all", "damaged", "none"):
            raise ValueError("split must be 'all', 'damaged' or 'none'")
        parent, ref, side, on_if = self._remesh_topology()
        T = len(parent)
        # --- chord damage per cell -------------------------------------------
        damage_cell = onp.zeros(self.num_cells)
        if self.cohesive is not None and delta_max is not None:
            dm = onp.asarray(delta_max)
            d_f = None
            prm = self.cohesive.params
            if "sigma_c" in prm and "Gamma" in prm:
                # bilinear: delta_f = 2 Gamma / sigma_c; exponential (no finite delta_f):
                # 5 delta_c, where the traction has dropped below 4 % of sigma_c
                d_f = 2.0 * prm["Gamma"] / prm["sigma_c"] if "K" in prm else 5.0 * prm["Gamma"] / (onp.e * prm["sigma_c"])
            if d_f is not None:
                damage_cell = onp.where(self.cohesive_cells, onp.clip(dm.max(axis=1) / d_f, 0.0, 1.0), 0.0)
        if split == "all":
            opened_cell = self.cut_cells.copy()
        elif split == "damaged":
            opened_cell = self.cut_cells & (damage_cell >= damage_threshold)
        else:
            opened_cell = onp.zeros(self.num_cells, dtype=bool)

        # --- evaluate fields at the nudged vertices and centroids -------------
        counts = onp.bincount(parent, minlength=self.num_cells)
        smax = int(counts.max())
        slot = onp.zeros(T, dtype=int)
        order = onp.argsort(parent, kind="stable")
        starts = onp.concatenate([[0], onp.cumsum(counts)[:-1]])
        slot[order] = onp.arange(T) - starts[parent[order]]
        cent = ref.mean(axis=1, keepdims=True)
        verts = ref + nudge * (cent - ref)                                      # (T, 3, 2)
        xi = onp.zeros((self.num_cells, smax, 4, 2))
        xi[parent, slot, :3] = verts
        xi[parent, slot, 3] = cent[:, 0]
        fields = _cell_fields(self, problem.unflatten_fn_dof)
        cs = self.cell_solution(problem, sol)
        f = jax.vmap(lambda s_, X, p_, m, x: fields(s_, X, p_, m, x)[:4])
        x_q, u_q, phi_q, g_q = (onp.asarray(a) for a in f(
            cs, np.asarray(self._cell_X), np.asarray(self._cell_phi), np.asarray(self._cell_mask),
            np.asarray(xi.reshape(self.num_cells, -1, 2))))
        x_q = x_q.reshape(self.num_cells, smax, 4, 2)[parent, slot]
        u_q = u_q.reshape(self.num_cells, smax, 4, -1)[parent, slot]
        phi_q = phi_q.reshape(self.num_cells, smax, 4)[parent, slot]
        g_q = g_q.reshape(self.num_cells, smax, 4, 2, 2)[parent, slot, 3]

        # --- merge vertices: shared, except across opened interfaces ----------
        Xv = _interp_cell_batch(self.ele_type, ref.reshape(-1, 2), self._cell_X_np[parent].repeat(3, axis=0))
        scale = float(onp.abs(self.points).max()) + 1.0
        keyxy = onp.round(Xv / scale * 1e10).astype(onp.int64)
        opened_vert = (on_if & opened_cell[parent][:, None]).reshape(-1)
        side_v = onp.where(opened_vert, onp.repeat(side, 3), 0.0).astype(onp.int64)
        key = onp.stack([keyxy[:, 0], keyxy[:, 1], side_v], axis=1)
        uniq, inv = onp.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        P = len(uniq)
        cnt = onp.bincount(inv, minlength=P).astype(float)
        points = onp.zeros((P, 2)); onp.add.at(points, inv, Xv); points /= cnt[:, None]
        u = onp.zeros((P, u_q.shape[-1])); onp.add.at(u, inv, u_q[:, :3].reshape(-1, u_q.shape[-1])); u /= cnt[:, None]
        ph = onp.zeros(P); onp.add.at(ph, inv, phi_q[:, :3].reshape(-1)); ph /= cnt
        cells = inv.reshape(T, 3)
        # crack lines: edges of opened sub-triangles with both vertices on the interface
        lines = []
        ov = opened_vert.reshape(T, 3)
        for e0, e1 in ((0, 1), (1, 2), (2, 0)):
            m = ov[:, e0] & ov[:, e1]
            lines.append(onp.stack([cells[m, e0], cells[m, e1]], axis=1))
        crack_lines = onp.unique(onp.sort(onp.concatenate(lines), axis=1), axis=0) if lines else onp.zeros((0, 2), int)
        return dict(points=points, cells=cells, u=u, phi=ph, parent=parent, side=side,
                    grad_u=g_q, damage=damage_cell[parent], opened=opened_cell[parent],
                    crack_lines=crack_lines)

    def save_remesh_vtu(self, filename: str, problem: Problem, sol, material: Optional[Callable] = None,
                        crack_filename: Optional[str] = None, **kwargs) -> dict:
        """Write :meth:`remesh` with :func:`feax.utils.save_sol`: point data ``u``
        (3 components, for Warp By Vector) and ``phi``; cell data ``parent``,
        ``side``, ``damage``, ``opened`` and, if ``material(u_grad, phi, x)`` is
        given, ``stress`` (3x3, plane) at the sub-triangle centroids.  With
        ``crack_filename`` the opened interface faces are written as ``LINE2``
        cells.  ``kwargs`` go to :meth:`remesh`.  Returns the remesh dict."""
        from feax.utils import save_sol
        rm = self.remesh(problem, sol, **kwargs)
        u3 = onp.c_[rm["u"], onp.zeros(len(rm["u"]))] if rm["u"].shape[1] == 2 else rm["u"]
        cell_infos = [("parent", rm["parent"].astype(float)), ("side", rm["side"]),
                      ("damage", rm["damage"]), ("opened", rm["opened"].astype(float))]
        if material is not None:
            cen = rm["points"][rm["cells"]].mean(axis=1)
            phi_c = rm["phi"][rm["cells"]].mean(axis=1)
            sig = onp.asarray(jax.vmap(material)(np.asarray(rm["grad_u"]), np.asarray(phi_c), np.asarray(cen)))
            s3 = onp.zeros((len(sig), 3, 3)); s3[:, :2, :2] = sig
            cell_infos.append(("stress", s3))
        save_sol(Mesh(rm["points"], rm["cells"], ele_type="TRI3"), filename,
                 point_infos=[("u", u3), ("phi", rm["phi"])], cell_infos=cell_infos)
        if crack_filename is not None and len(rm["crack_lines"]):
            save_sol(Mesh(rm["points"], rm["crack_lines"], ele_type="LINE2"), crack_filename,
                     point_infos=[("u", u3)])
        return rm

    def quad_fields(self, problem: Problem, sol, grad: bool = False):
        """``(x_q, u_q, phi_q)`` at the sub-cell quadrature points, each
        (num_cells, nq_max, ...), plus the displacement gradient ``grad_u``
        (num_cells, nq_max, dim, dim) when ``grad=True``.  Padded points carry
        ``qw == 0``; use ``space.qw > 0`` to select the real ones.  Handy for
        looking at the fields (and strains / stresses) on both sides of the
        interface."""
        fields = _cell_fields(self, problem.unflatten_fn_dof)
        cs = self.cell_solution(problem, sol)
        n = 4 if grad else 3
        f = jax.vmap(lambda s, X, p, m, xi: fields(s, X, p, m, xi)[:n])
        return f(cs, np.asarray(self._cell_X), np.asarray(self._cell_phi),
                 np.asarray(self._cell_mask), np.asarray(self.qpts))


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def _cell_fields(space: XFEMSpace, unflatten):
    """Per-cell XFEM kinematics shared by the residual kernel and post-processing.

    Returns ``fields(cell_sol_flat, X, phi, mask, xi) ->
    (x_q, u_q, phi_q, grad_u, dNdx, dM, detJ)``.
    """
    shape_fn = _SHAPE_FNS[space.ele_type]
    enrich = space.enrichment
    nn, vec, nf = space.num_nodes_per_cell, space.vec, space.enrichment.n_funcs

    def fields(cell_sol_flat, X, phi, mask, xi):
        u, a = unflatten(cell_sol_flat)                  # (nn, vec), (nn, vec*nf)
        a = a.reshape(nn, nf, vec)
        N, dNdxi = shape_fn(xi)                          # (nq, nn), (nq, nn, 2)
        J = np.einsum("ai,qaI->qiI", X, dNdxi)           # (nq, 2, 2)
        detJ = np.linalg.det(J)
        dNdx = np.einsum("qaI,qIi->qai", dNdxi, np.linalg.inv(J))
        x_q = N @ X                                      # (nq, 2)
        phi_q = N @ phi                                  # (nq,)
        gphi_q = np.einsum("qai,a->qi", dNdx, phi)       # (nq, 2)
        psi, gpsi = enrich(phi_q, gphi_q, phi, N, dNdx)  # (nq, nn, nf), (nq, nn, nf, 2)
        psi = psi * mask[None, :, None]
        gpsi = gpsi * mask[None, :, None, None]
        # enriched shape functions M = N ψ and their gradients
        M = N[:, :, None] * psi                                           # (nq, nn, nf)
        dM = dNdx[:, :, None, :] * psi[..., None] + N[:, :, None, None] * gpsi
        u_q = N @ u + np.einsum("qaf,afi->qi", M, a)
        grad_u = np.einsum("ai,qaj->qij", u, dNdx) + np.einsum("afi,qafj->qij", a, dM)
        return x_q, u_q, phi_q, grad_u, dNdx, dM, detJ

    return fields


def _interface_jump_fn(space: XFEMSpace, unflatten):
    """``jump_fn(cell_sol_flat, X, phi, mask, ixi) -> (x_g, jump, normal)`` at the
    interface Gauss points ``ixi`` (g, 2) of one cell: ``jump = u⁺ − u⁻`` with
    the enrichment evaluated at ``phi = ±eps``, ``normal = ∇phi/|∇phi|``."""
    shape_fn = _SHAPE_FNS[space.ele_type]
    enrich = space.enrichment
    nn, vec, nf = space.num_nodes_per_cell, space.vec, space.enrichment.n_funcs
    eps = 1e-12

    def jump_fn(cell_sol_flat, X, phi, mask, ixi):
        _, a = unflatten(cell_sol_flat)
        a = a.reshape(nn, nf, vec)
        N, dNdxi = shape_fn(ixi)                                          # (g, nn), (g, nn, 2)
        J = np.einsum("ai,qaI->qiI", X, dNdxi)
        dNdx = np.einsum("qaI,qIi->qai", dNdxi, np.linalg.inv(J))
        gphi = np.einsum("qai,a->qi", dNdx, phi)                          # (g, 2)
        normal = gphi / (np.linalg.norm(gphi, axis=-1, keepdims=True) + 1e-30)
        side = lambda sgn: enrich(np.full((ixi.shape[0],), sgn * eps), gphi, phi, N, dNdx)[0]
        dpsi = (side(+1.0) - side(-1.0)) * mask[None, :, None]            # (g, nn, nf)
        jump = np.einsum("qa,qaf,afi->qi", N, dpsi, a)                    # (g, 2)
        return N @ X, jump, normal

    return jump_fn


@jax.tree_util.register_pytree_node_class
@dataclass
class XFEMProblem(Problem):
    """Two-variable XFEM problem; build it with :func:`xfem_problem`.

    ``additional_info = (space, material)`` where
    ``material(u_grad, phi, x, *extra) -> sigma`` (``u_grad``: (dim, dim)
    displacement gradient at a quadrature point, ``phi``: level-set value there,
    so a two-phase law is ``np.where(phi < 0, ...)``, ``x``: its coordinates,
    ``extra``: the cell-level arrays given to :meth:`XFEMSpace.traced_params`).
    """

    def custom_init(self, space: XFEMSpace, material: Callable) -> None:
        self.space = space
        self.material = material

    def get_universal_kernel(self) -> Callable:
        space, material = self.space, self.material
        fields = _cell_fields(space, self.unflatten_fn_dof)
        cz = space.cohesive
        jump_fn = _interface_jump_fn(space, self.unflatten_fn_dof) if cz is not None else None

        def bulk(cell_sol_flat, X, phi, mask, xi, w, extra):
            x_q, _, phi_q, grad_u, dNdx, dM, detJ = fields(cell_sol_flat, X, phi, mask, xi)
            mat = jax.vmap(material, in_axes=(0, 0, 0) + (None,) * len(extra))
            sigma = mat(grad_u, phi_q, x_q, *extra)                       # (nq, 2, 2)
            JxW = w * np.abs(detJ)
            R_u = np.einsum("qij,qaj,q->ai", sigma, dNdx, JxW)            # (nn, vec)
            R_a = np.einsum("qij,qafj,q->afi", sigma, dM, JxW)            # (nn, nf, vec)
            return np.concatenate([R_u.reshape(-1), R_a.reshape(-1)])

        if cz is None:
            def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads,
                       cell_JxW, cell_v_grads_JxW, X, phi, mask, xi, w, *extra):
                return bulk(cell_sol_flat, X, phi, mask, xi, w, extra)
        else:
            def interface_energy(cell_sol_flat, X, phi, mask, ixi, iw, dmax, sc, dprev):
                _, jump, normal = jump_fn(cell_sol_flat, X, phi, mask, ixi)
                return np.sum(cz.energy_density(jump, normal, dmax, sc, dprev) * iw)

            def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads,
                       cell_JxW, cell_v_grads_JxW, X, phi, mask, xi, w, ixi, iw, dmax, sc, dprev, *extra):
                R = bulk(cell_sol_flat, X, phi, mask, xi, w, extra)
                return R + jax.grad(interface_energy)(cell_sol_flat, X, phi, mask, ixi, iw, dmax, sc, dprev)

        return kernel


def xfem_problem(space: XFEMSpace, material: Callable, **kwargs) -> XFEMProblem:
    """Create the :class:`XFEMProblem` for ``space`` with constitutive law ``material``.

    ``kwargs`` (e.g. ``matrix_view``, ``location_fns``) go to :class:`~feax.Problem`.
    """
    return XFEMProblem(**space.problem_kwargs(), additional_info=(space, material), **kwargs)


__all__ = [
    "Enrichment", "ShiftedHeaviside", "Heaviside", "ShiftedAbs", "ModifiedAbs",
    "DEFAULT_ENRICHMENTS", "CohesiveZone", "bilinear_tsl", "exponential_tsl",
    "XFEMSpace", "XFEMProblem", "xfem_problem",
]
