"""LEFM crack growth with XFEM on a fixed mesh: traced geometry, tip enrichment
and energy release rates by autodiff — experimental.

The crack is a polyline.  Everything downstream of it is a fixed-shape JAX
function, so one compilation serves every crack configuration:

.. code-block:: text

    polyline (M_max, 2), n_pts ──► phi, psi (nodal level sets)
        ──► per-cell class (uncut / Heaviside-cut / tip), nodal enrichment
            masks, interface-conforming sub-cell quadrature  (fixed nq_max)
        ──► XFEM solve  ──► strain energy  Π
    J-vector = −∂Π/∂a of a *virtual crack extension*: the nodes around the
    tip move with it (smooth bump q), so the derivative is the domain-form
    J integral of the discrete solution                        (jax.grad)

Enrichment (Moës–Dolbow–Belytschko node sets, shifted basis):

* **Heaviside** ``sign(phi(x)) − sign(phi_j)`` on the nodes of the cells the
  crack crosses completely (variable 1, 2 DOFs per node),
* **branch functions** ``F_k(r, θ) − F_k(r_j, θ_j)``,
  ``F = √r {sin θ/2, cos θ/2, sin θ/2 sin θ, cos θ/2 sin θ}``, on the nodes
  of the cell containing the tip (variable 2, 8 DOFs per node);
  ``r, θ`` come from the two level sets (``θ = atan2(phi, psi)``), their
  gradients from ``jax.jacfwd``.

Both enriched spaces live on a static *band* of candidate nodes (plus one
phantom node each, ``with_jxw=False``).  Which band nodes are actually
enriched is a traced 0/1 mask: inactive enriched DOFs are held by a small
quadratic penalty inside the element energy, so no Dirichlet row ever
changes.  The sub-cell quadrature is padded to a fixed ``nq_max`` (the tip
fan: 6 triangles × 6 points) and the three candidate rules (uncut / cut /
tip fan) are computed for every cell and selected by its class, so the
geometry is one ``vmap`` over cells.

Scope: 2D ``TRI3`` meshes, linear elasticity given as an energy density
``W(eps, x)``, displacement-controlled loading (``Π`` = strain energy, so
``J = −∂Π/∂a`` at fixed grips).  ``K_I, K_II`` follow from the J-vector
(``J_t = (K_I²+K_II²)/E'``, ``J_n = −2K_I K_II/E'``) and the growth direction
from the maximum circumferential stress criterion.

Why not differentiate with respect to the tip position directly?  The
discrete energy is not smooth at the cell scale (the enriched node sets and
the sub-cell quadrature change from cell to cell), so ``∂Π/∂x_tip`` shows a
sawtooth of ±50 % within a cell even though the one-cell secant of ``Π`` is
accurate to a few percent.  Moving the whole neighbourhood with the tip
removes that pollution (it is the discrete form of the J domain integral).

Usage
-----

.. code-block:: python

    from feax.experimental.xfem_crack import CrackSpace, crack_problem, plane_strain_energy

    space = CrackSpace(mesh, max_points=32)              # TRI3 mesh
    problem = crack_problem(space, plane_strain_energy(E, nu))
    bc = space.create_bc(problem, fe.DirichletBCConfig([...]))
    ts = fe.TracedStructure.from_problem(problem)
    solver = fe.create_solver(problem, bc, linear=True, traced_structure=ts)
    crack = space.polyline([(-0.05, 0.0), (0.3, 0.0)])  # (poly, n_pts)

    J = space.j_vector_fn(problem, solver, bc, ts, radius=4 * h)   # jitted
    J_t, J_n = J(poly, n_pts)                                     # energy release rate + J2
    theta, K_I, K_II = space.growth_direction(J, poly, n_pts, E_prime)
    poly, n_pts = space.grow(poly, n_pts, theta, da)
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

from .xfem import _REF_TRI3, _sgn, _shape_tri3, _triangle_rule

_R3_PTS, _R3_W = (np.asarray(a) for a in _triangle_rule(3))
_R6_PTS, _R6_W = (np.asarray(a) for a in _triangle_rule(6))
_REF = np.asarray(_REF_TRI3)
_EDGES = np.array([[0, 1], [1, 2], [2, 0]])
N_BRANCH = 4
DEFAULT_TIP_RINGS = (0.0, 0.1, 0.35, 1.0)   # radial grading of the tip fan toward the tip


def nq_max_for(rings) -> int:
    """Quadrature slots per cell: the tip fan (6 fan triangles, each split into
    ``1 + 2 (m-1)`` sub-triangles by ``m`` radial rings, 6-point rule) is the
    largest of the three candidate rules."""
    m = len(rings) - 1
    return 6 * (1 + 2 * (m - 1)) * 6


NQ_MAX = nq_max_for(DEFAULT_TIP_RINGS)


# ---------------------------------------------------------------------------
# Level sets from a polyline
# ---------------------------------------------------------------------------

def crack_level_sets(poly, n_pts, X):
    """Nodal level sets of a polyline crack.

    ``poly`` (M, 2) holds ``n_pts`` valid points (the rest is padding); ``X``
    (N, 2) are the node coordinates.  Returns ``phi`` (signed distance to the
    polyline, positive on the left of the crack direction; the first and last
    segments are extended to infinity so ``phi = 0`` continues straight ahead
    of the tip and behind the mouth) and ``psi`` (arc length of the nearest
    point minus the crack length: negative on the crack, positive ahead of the
    tip, where it is the distance along the last segment's direction).
    """
    M = poly.shape[0]
    a, b = poly[:-1], poly[1:]                                   # (M-1, 2)
    idx = np.arange(M - 1)
    valid = idx < n_pts - 1
    first = idx == 0
    last = idx == n_pts - 2
    d = b - a
    # safe norms: padded (zero-length) segments would otherwise put NaN into
    # the gradient through the masked branches of the np.where below
    L = np.sqrt(np.sum(d * d, axis=1) + 1e-300)
    Ls = np.where(valid & (L > 1e-150), L, 1.0)
    tdir = d / Ls[:, None]
    rel = X[:, None, :] - a[None, :, :]                          # (N, M-1, 2)
    t = np.einsum("nsi,si->ns", rel, tdir) / Ls[None, :]         # in [0, 1] on the segment
    lo = np.where(first, -np.inf, 0.0)
    hi = np.where(last, np.inf, 1.0)
    t = np.clip(t, lo[None, :], hi[None, :])
    near = a[None, :, :] + (t * Ls[None, :])[:, :, None] * tdir[None, :, :]
    dist = np.sqrt(np.sum((X[:, None, :] - near) ** 2, axis=-1) + 1e-300)
    side = _sgn(tdir[None, :, 0] * rel[:, :, 1] - tdir[None, :, 1] * rel[:, :, 0])
    dist = np.where(valid[None, :], dist, np.inf)
    k = np.argmin(dist, axis=1)                                  # nearest segment per node
    rows = np.arange(X.shape[0])
    phi = side[rows, k] * dist[rows, k]
    cum = np.concatenate([np.zeros(1), np.cumsum(np.where(valid, L, 0.0))])[:-1]
    s = cum[k] + t[rows, k] * Ls[k]
    psi = s - np.sum(np.where(valid, L, 0.0))
    return phi, psi


# ---------------------------------------------------------------------------
# Per-cell traced geometry (TRI3)
# ---------------------------------------------------------------------------

def _tri_rule_pts(tri, rule_pts, rule_w):
    """Map a unit-triangle rule onto ``tri`` (3, 2): points (m, 2), ref-area weights (m,)."""
    e1, e2 = tri[1] - tri[0], tri[2] - tri[0]
    pts = tri[0][None, :] + rule_pts[:, :1] * e1[None, :] + rule_pts[:, 1:] * e2[None, :]
    det = np.abs(e1[0] * e2[1] - e1[1] * e2[0])
    return pts, rule_w * det


def _edge_crossings(p):
    """For the 3 edges: crossed flag and the crossing parameter t (0 if not crossed)."""
    pi, pj = p[_EDGES[:, 0]], p[_EDGES[:, 1]]
    crossed = _sgn(pi) != _sgn(pj)
    t = np.where(crossed, pi / np.where(crossed, pi - pj, 1.0), 0.5)
    return crossed, t


def _cut_sub_triangles(p):
    """Split the reference triangle by the zero line of the linear field ``p``
    into 3 sub-triangles (3, 3, 2); when ``p`` has one sign the first is the
    whole triangle and the other two are degenerate (zero area)."""
    s = _sgn(p)
    lone = np.array([s[1] == s[2], s[2] == s[0], s[0] == s[1]])
    crossed = np.any(s != s[0])
    i0 = np.argmax(lone)
    i1, i2 = (i0 + 1) % 3, (i0 + 2) % 3
    v0, v1, v2 = _REF[i0], _REF[i1], _REF[i2]
    p0, p1, p2 = p[i0], p[i1], p[i2]
    t1 = p0 / np.where(crossed, p0 - p1, 1.0)
    t2 = p0 / np.where(crossed, p0 - p2, 1.0)
    P1 = v0 + t1 * (v1 - v0)
    P2 = v0 + t2 * (v2 - v0)
    cut = np.stack([np.stack([v0, P1, P2]), np.stack([P1, v1, v2]), np.stack([P1, v2, P2])])
    whole = np.stack([_REF, np.stack([v0, v0, v0]), np.stack([v0, v0, v0])])
    return np.where(crossed, cut, whole)


def _fan_sub_triangles(tip, A, B, rings):
    """Graded split of the fan triangle (tip, A, B): ``1 + 2 (m-1)`` triangles."""
    r = np.asarray(rings)
    Ai = tip[None, :] + r[:, None] * (A - tip)[None, :]
    Bi = tip[None, :] + r[:, None] * (B - tip)[None, :]
    tris = [np.stack([tip, Ai[1], Bi[1]])]
    for i in range(1, len(rings) - 1):
        tris.append(np.stack([Ai[i], Ai[i + 1], Bi[i + 1]]))
        tris.append(np.stack([Ai[i], Bi[i + 1], Bi[i]]))
    return np.stack(tris)


def _cell_geometry(X, phi, psi, tip, threshold, rings=DEFAULT_TIP_RINGS):
    """Class (0 uncut, 1 Heaviside-cut, 2 tip) and the padded quadrature of one
    TRI3 cell: reference points (nq, 2), reference-area weights (nq,), with
    ``nq = nq_max_for(rings)``."""
    NQ_MAX = nq_max_for(rings)
    # clamp near-zero nodal values (consistently with the kernel: the clamped
    # phi is what the caller stores)
    scale = np.max(np.abs(phi))
    eps = threshold * scale
    phi = np.where(np.abs(phi) < eps, np.where(phi >= 0, eps, -eps), phi)
    s = _sgn(phi)
    crossed = np.any(s != s[0])
    crossed_e, t_e = _edge_crossings(phi)
    psi_e = psi[_EDGES[:, 0]] + t_e * (psi[_EDGES[:, 1]] - psi[_EDGES[:, 0]])
    chord_psi = np.sum(np.where(crossed_e, psi_e, 0.0)) / np.maximum(np.sum(crossed_e), 1)
    # tip inside the (affine) cell?
    A = np.stack([X[1] - X[0], X[2] - X[0]], axis=1)             # columns e1, e2
    xi_tip = np.linalg.solve(A, tip - X[0])
    lam = np.array([1.0 - xi_tip[0] - xi_tip[1], xi_tip[0], xi_tip[1]])
    inside = np.all(lam >= -1e-12)
    cls = np.where(inside, 2, np.where(crossed & (chord_psi < 0.0), 1, 0))

    # (i) uncut
    pu, wu = _tri_rule_pts(_REF, _R3_PTS, _R3_W)
    q_uncut = (np.zeros((NQ_MAX, 2)).at[:3].set(pu), np.zeros(NQ_MAX).at[:3].set(wu))
    # (ii) Heaviside cut
    subs = _cut_sub_triangles(phi)
    pc, wc = jax.vmap(lambda tri: _tri_rule_pts(tri, _R3_PTS, _R3_W))(subs)
    q_cut = (np.zeros((NQ_MAX, 2)).at[:9].set(pc.reshape(9, 2)), np.zeros(NQ_MAX).at[:9].set(wc.reshape(9)))
    # (iii) tip fan: tip -> each edge, the edge split at its crossing (or midpoint)
    vi, vj = _REF[_EDGES[:, 0]], _REF[_EDGES[:, 1]]
    P = vi + t_e[:, None] * (vj - vi)                              # (3, 2)
    fan = np.concatenate([
        jax.vmap(lambda a, b: _fan_sub_triangles(xi_tip, a, b, rings))(vi, P),
        jax.vmap(lambda a, b: _fan_sub_triangles(xi_tip, a, b, rings))(P, vj),
    ]).reshape(-1, 3, 2)                                            # (6 (1+2(m-1)), 3, 2)
    pt, wt = jax.vmap(lambda tri: _tri_rule_pts(tri, _R6_PTS, _R6_W))(fan)
    q_tip = (pt.reshape(NQ_MAX, 2), wt.reshape(NQ_MAX))

    qpts = np.where(cls == 2, q_tip[0], np.where(cls == 1, q_cut[0], q_uncut[0]))
    qw = np.where(cls == 2, q_tip[1], np.where(cls == 1, q_cut[1], q_uncut[1]))
    # the sub-triangulation itself (for visualization), padded with zero-area triangles
    n_sub = fan.shape[0]
    degenerate = np.zeros((n_sub, 3, 2))
    tri_uncut = degenerate.at[0].set(_REF)
    tri_cut = degenerate.at[:3].set(subs)
    subtris = np.where(cls == 2, fan, np.where(cls == 1, tri_cut, tri_uncut))
    return cls, qpts, qw, phi, subtris


# ---------------------------------------------------------------------------
# Branch functions
# ---------------------------------------------------------------------------

def branch_functions(phi, psi):
    """``F_k(r, θ)`` of a crack tip, ``r = sqrt(phi² + psi²)``, ``θ = atan2(phi, psi)``."""
    r = np.sqrt(phi * phi + psi * psi + 1e-30)
    th = np.arctan2(phi, psi)
    sr = np.sqrt(r)
    return np.stack([sr * np.sin(th / 2), sr * np.cos(th / 2),
                     sr * np.sin(th / 2) * np.sin(th), sr * np.cos(th / 2) * np.sin(th)])


_branch_jac = jax.jacfwd(branch_functions, argnums=(0, 1))     # dF/dphi, dF/dpsi (4,) each


# ---------------------------------------------------------------------------
# CrackSpace
# ---------------------------------------------------------------------------

def plane_strain_energy(E: float, nu: float) -> Callable:
    """Isotropic linear elastic energy density ``W(eps, x)`` (plane strain)."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    def W(eps, x):
        return 0.5 * lam * np.trace(eps) ** 2 + mu * np.sum(eps * eps)

    return W


class CrackSpace:
    """XFEM crack space on a TRI3 mesh with traced (fixed-shape) geometry.

    Parameters
    ----------
    mesh : Mesh
        ``TRI3`` background mesh.
    band : bool array (num_nodes,), optional
        Candidate nodes for enrichment (default: all nodes).  Enriched DOFs
        exist on these nodes only (``2 + 8`` per band node, plus 2 phantom
        nodes); which of them are active is a traced mask.
    max_points : int
        Capacity of the crack polyline (fixed shape).
    threshold : float
        Relative clamp for near-zero nodal ``phi`` (sliver sub-cells).
    penalty : float
        Stiffness of the quadratic penalty holding inactive enriched DOFs.
    tip_rings : tuple of float
        Radial grading ``0 = r_0 < r_1 < ... < r_m = 1`` of the tip-cell fan
        quadrature (sub-triangles concentrate toward the tip, where the
        energy density behaves like 1/r).  ``nq_max`` follows from it.
    """

    def __init__(self, mesh: Mesh, band=None, max_points: int = 32,
                 threshold: float = 1e-3, penalty: float = 1.0,
                 tip_rings: Sequence[float] = DEFAULT_TIP_RINGS):
        if mesh.ele_type != "TRI3" or onp.asarray(mesh.cells).shape[1] != 3:
            raise NotImplementedError("CrackSpace supports TRI3 meshes")
        self.mesh = mesh
        self.points = onp.asarray(mesh.points, dtype=onp.float64)[:, :2]
        self.cells = onp.asarray(mesh.cells, dtype=onp.int64)
        self.num_nodes = self.points.shape[0]
        self.num_cells = self.cells.shape[0]
        self.vec = 2
        self.max_points = int(max_points)
        self.threshold = float(threshold)
        self.penalty = float(penalty)
        self.tip_rings = tuple(float(r) for r in tip_rings)
        self.nq_max = nq_max_for(self.tip_rings)

        self.band = onp.ones(self.num_nodes, dtype=bool) if band is None else onp.asarray(band, dtype=bool)
        self.num_band = int(self.band.sum())
        self.phantom = self.num_band
        self.local = onp.full(self.num_nodes, self.phantom, dtype=onp.int32)
        self.local[self.band] = onp.arange(self.num_band, dtype=onp.int32)
        epts = onp.full((self.num_band + 1, 2), onp.nan)
        epts[:self.num_band] = self.points[self.band]
        self.emesh = Mesh(epts, self.local[self.cells], ele_type="TRI3")   # shared by both spaces

        self._cell_X = np.asarray(self.points[self.cells])                  # (C, 3, 2)
        self._cell_band = np.asarray(self.band[self.cells].astype(onp.float64))
        self._cells = np.asarray(self.cells)

    # ----------------------------------------------------------- polyline
    def polyline(self, points: Sequence[Tuple[float, float]]):
        """``(poly, n_pts)`` padded to ``max_points``."""
        pts = onp.asarray(points, dtype=onp.float64)
        poly = onp.zeros((self.max_points, 2))
        poly[:len(pts)] = pts
        return np.asarray(poly), np.int32(len(pts))

    @staticmethod
    def grow(poly, n_pts, theta, da):
        """Append a segment of length ``da`` at angle ``theta`` from the tip."""
        tip = poly[n_pts - 1]
        new = tip + da * np.array([np.cos(theta), np.sin(theta)])
        return poly.at[n_pts].set(new), n_pts + 1

    # ----------------------------------------------------------- geometry
    def geometry(self, poly, n_pts, X=None):
        """Traced per-cell arrays for ``TracedParams`` (fixed shapes).  ``X``
        (num_nodes, 2) overrides the node coordinates (used by the virtual
        crack extension, which moves the nodes around the tip with it)."""
        X = np.asarray(self.points) if X is None else X
        phi, psi = crack_level_sets(poly, n_pts, X)
        tip = poly[n_pts - 1]
        cells = self._cells
        Xc = X[cells]
        geom = lambda Xcc, pc, sc: _cell_geometry(Xcc, pc, sc, tip, self.threshold, self.tip_rings)
        cls, qpts, qw, phi_c, _ = jax.vmap(geom)(Xc, phi[cells], psi[cells])
        # nodal enrichment masks (max over incident cells), restricted to the band
        flat = cells.reshape(-1)
        isT = np.zeros(self.num_nodes).at[flat].max(np.repeat((cls == 2) * 1.0, 3))
        isH = np.zeros(self.num_nodes).at[flat].max(np.repeat((cls == 1) * 1.0, 3))
        band = np.asarray(self.band * 1.0)
        maskT = isT * band
        maskH = isH * (1.0 - isT) * band
        # the clamped phi of a node may differ between cells (cell scale);
        # use the per-cell clamped values the quadrature was built with
        return (Xc, phi_c, psi[cells], maskH[cells], maskT[cells], qpts, qw)

    def traced_params(self, poly, n_pts, X=None) -> TracedParams:
        return TracedParams(volume_vars=self.geometry(poly, n_pts, X))

    def classes(self, poly, n_pts):
        """Per-cell class (0 uncut, 1 Heaviside, 2 tip) as a numpy array."""
        return onp.asarray(self._geometry_full(poly, n_pts)[0])

    def _geometry_full(self, poly, n_pts):
        X = np.asarray(self.points)
        phi, psi = crack_level_sets(poly, n_pts, X)
        tip = poly[n_pts - 1]
        geom = lambda Xc, pc, sc: _cell_geometry(Xc, pc, sc, tip, self.threshold, self.tip_rings)
        return jax.vmap(geom)(self._cell_X, phi[self._cells], psi[self._cells])

    def visualization(self, problem: Problem, sol, poly, n_pts, nudge: float = 1e-6) -> dict:
        """Sub-cell visualization mesh with the discontinuity resolved.

        Every sub-triangle of the interface-conforming sub-triangulation gets
        its own three vertices, so the displacement jump across the crack and
        the kink at the tip appear in the output instead of being smeared by
        a nodal average.  Fields are evaluated at the vertices nudged by
        ``nudge`` toward the sub-triangle centroid, which puts a vertex lying
        on the crack on the correct side.

        Returns a dict (host numpy) with ``points`` (P, 2), ``cells`` (T, 3),
        ``cell_of`` (T,) parent cell index, ``class`` (T,), ``u`` (P, 2),
        ``phi`` (P,), ``psi`` (P,) and ``eps`` (T, 2, 2) (strain at the
        sub-triangle centroid).  Padding sub-triangles are dropped.  Use it
        with :func:`feax.utils.save_sol` on ``Mesh(points, cells, "TRI3")``.
        """
        tp = self.traced_params(poly, n_pts)
        cls, _, _, _, subtris = self._geometry_full(poly, n_pts)     # subtris (C, S, 3, 2)
        C, S = subtris.shape[:2]
        cent = subtris.mean(axis=2, keepdims=True)
        verts = subtris + nudge * (cent - subtris)                  # (C, S, 3, 2)
        xi = np.concatenate([verts.reshape(C, S * 3, 2), cent.reshape(C, S, 2)], axis=1)
        kin = _cell_kinematics_fn(self, problem.unflatten_fn_dof)
        cs = self.cell_solution(problem, sol)
        X, phi, psi, mH, mT, _, _ = tp.volume_vars
        x_all, u_all, eps_all, _ = jax.vmap(lambda *a: kin(*a))(cs, X, phi, psi, mH, mT, xi)
        N_all, _ = _shape_tri3(xi.reshape(-1, 2))
        phi_v = (N_all.reshape(C, -1, 3) * phi[:, None, :]).sum(-1)
        psi_v = (N_all.reshape(C, -1, 3) * psi[:, None, :]).sum(-1)

        x_v = onp.asarray(x_all[:, :S * 3]).reshape(C, S, 3, 2)
        u_v = onp.asarray(u_all[:, :S * 3]).reshape(C, S, 3, 2)
        phi_v = onp.asarray(phi_v[:, :S * 3]).reshape(C, S, 3)
        psi_v = onp.asarray(psi_v[:, :S * 3]).reshape(C, S, 3)
        eps_c = onp.asarray(eps_all[:, S * 3:])                      # (C, S, 2, 2)
        st = onp.asarray(subtris)
        e1, e2 = st[:, :, 1] - st[:, :, 0], st[:, :, 2] - st[:, :, 0]
        area = onp.abs(e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0])
        keep = area > 1e-12                                          # (C, S)
        T = int(keep.sum())
        points = x_v[keep].reshape(T * 3, 2)
        cells = onp.arange(T * 3).reshape(T, 3)
        cell_of = onp.repeat(onp.arange(C), S).reshape(C, S)[keep]
        return dict(points=points, cells=cells, cell_of=cell_of,
                    **{"class": onp.asarray(cls)[cell_of]},
                    u=u_v[keep].reshape(T * 3, 2), phi=phi_v[keep].reshape(-1),
                    psi=psi_v[keep].reshape(-1), eps=eps_c[keep])

    @staticmethod
    def polyline_mesh(poly, n_pts) -> Mesh:
        """The crack polyline as a ``LINE2`` mesh for :func:`feax.utils.save_sol`."""
        pts = onp.asarray(poly[:int(n_pts)])
        lines = onp.stack([onp.arange(len(pts) - 1), onp.arange(1, len(pts))], axis=1)
        return Mesh(pts, lines, ele_type="LINE2")

    # ----------------------------------------------------------- problem glue
    def problem_kwargs(self) -> dict:
        return dict(mesh=[self.mesh, self.emesh, self.emesh], vec=[2, 2, 2 * N_BRANCH], dim=2,
                    ele_type=["TRI3", "TRI3", "TRI3"], with_jxw=[True, False, False])

    def phantom_rows(self, problem: Problem) -> onp.ndarray:
        rows = []
        for var, v in ((1, 2), (2, 2 * N_BRANCH)):
            rows.append(problem.offset[var] + self.phantom * v + onp.arange(v))
        return onp.concatenate(rows)

    def create_bc(self, problem: Problem, config: Optional[DirichletBCConfig] = None) -> DirichletBC:
        base = (config if config is not None else DirichletBCConfig([])).create_bc(problem)
        rows = onp.concatenate([onp.asarray(base.bc_rows), self.phantom_rows(problem)])
        vals = onp.concatenate([onp.asarray(base.bc_vals), onp.zeros(len(rows) - len(base.bc_rows))])
        order = onp.argsort(rows, kind="stable")
        rows, vals = rows[order], vals[order]
        rows, first = onp.unique(rows, return_index=True)
        vals = vals[first]
        mask = onp.zeros(problem.num_total_dofs_all_vars, dtype=bool)
        mask[rows] = True
        return DirichletBC(bc_rows=np.asarray(rows, dtype=np.int32), bc_mask=np.asarray(mask),
                           bc_vals=np.asarray(vals), total_dofs=problem.num_total_dofs_all_vars)

    # ----------------------------------------------------------- post-processing
    def nodal_displacement(self, problem: Problem, sol):
        """Total nodal displacement (shifted enrichments vanish at the nodes)."""
        return problem.unflatten_fn_sol_list(sol)[0]

    def cell_solution(self, problem: Problem, sol):
        u, a, b = problem.unflatten_fn_sol_list(sol)
        ec = np.asarray(self.emesh.cells)
        return np.concatenate([u[self._cells].reshape(self.num_cells, -1),
                               a[ec].reshape(self.num_cells, -1),
                               b[ec].reshape(self.num_cells, -1)], axis=1)

    def strain_energy(self, problem: Problem, sol, tp: TracedParams):
        """``Π`` = Σ_cells element energy (strain energy + inactive-DOF penalty)."""
        energy = _cell_energy_fn(self, problem.unflatten_fn_dof, problem.material)
        cs = self.cell_solution(problem, sol)
        return np.sum(jax.vmap(energy)(cs, *tp.volume_vars))

    def quad_fields(self, problem: Problem, sol, tp: TracedParams):
        """``(x_q, u_q, eps_q)`` at the sub-cell quadrature points (num_cells, nq_max, ...)."""
        kin = _cell_kinematics_fn(self, problem.unflatten_fn_dof)
        cs = self.cell_solution(problem, sol)
        X, phi, psi, mH, mT, xi, w = tp.volume_vars
        f = jax.vmap(lambda *args: kin(*args)[:3])
        return f(cs, X, phi, psi, mH, mT, xi)

    # ----------------------------------------------------------- fracture mechanics
    def j_vector_fn(self, problem: Problem, solver: Callable, bc: DirichletBC, ts,
                    radius: float, inner: float = 0.5) -> Callable:
        """Jitted ``J(poly, n_pts, radius=radius) -> (J_t, J_n)``: the J-vector
        (configurational force on the tip) by **virtual crack extension with
        autodiff**.  ``radius`` may be overridden per call (traced), e.g. to
        shrink the domain when the tip approaches a boundary.

        The nodes within ``radius`` of the tip are moved with the tip by the
        smooth bump ``q`` (``q = 1`` for ``r < inner * radius``, smoothstep to 0
        at ``radius``) and the strain energy is differentiated with respect to
        the extension length at zero:  ``J_e = −∂Π/∂a``.  Because the mesh and
        the sub-cell geometry around the tip translate rigidly, this is the
        domain-integral J of the discrete solution and is free of the cell-
        scale sawtooth that the pointwise derivative of the discrete energy
        with respect to the tip position shows.  ``J_t`` is along the crack
        direction at the tip (the energy release rate for straight growth),
        ``J_n`` along the left normal.  ``radius`` must keep ``q = 0`` on the
        outer boundary.  Displacement-controlled loading is assumed.
        """
        u0 = np.asarray(fe_zero_guess(problem, bc))
        X0 = np.asarray(self.points)

        def bump(r, r2):
            r1 = inner * r2
            t = np.clip((r - r1) / (r2 - r1), 0.0, 1.0)
            return 1.0 - t * t * (3.0 - 2.0 * t)

        def Pi(poly, n_pts, X):
            tp = self.traced_params(poly, n_pts, X)
            sol = solver(tp, u0, traced_structure=ts)
            return self.strain_energy(problem, sol, tp)

        def J_dir(poly, n_pts, e, r2):
            tip = poly[n_pts - 1]
            qX = bump(np.linalg.norm(X0 - tip, axis=1), r2)             # (N,)
            qP = bump(np.linalg.norm(poly - tip, axis=1), r2)           # (M,)

            def Pi_a(a):
                return Pi(poly + a * qP[:, None] * e[None, :], n_pts, X0 + a * qX[:, None] * e[None, :])

            return -jax.grad(Pi_a)(np.float64(0.0))

        def J(poly, n_pts, r2):
            tip, prev = poly[n_pts - 1], poly[n_pts - 2]
            t = tip - prev
            t = t / np.linalg.norm(t)
            nrm = np.array([-t[1], t[0]])
            return J_dir(poly, n_pts, t, r2), J_dir(poly, n_pts, nrm, r2)

        J_jit = jax.jit(J)
        state = {"warm": False}

        def J_wrapped(poly, n_pts, radius=radius):
            if not state["warm"]:
                # direct solvers (cuDSS) initialise lazily on their first call
                # and need concrete CSR data: run one eager solve first
                Pi(poly, n_pts, X0)
                state["warm"] = True
            return J_jit(poly, n_pts, np.float64(radius))

        return J_wrapped

    @staticmethod
    def sif_from_j(J_t, J_n, E_prime):
        """``(K_I, K_II)`` from the J-vector of a straight crack tip:
        ``J_t = (K_I² + K_II²)/E'``, ``J_n = −2 K_I K_II/E'`` (K_I ≥ 0 branch)."""
        S = E_prime * J_t
        P = -0.5 * E_prime * J_n                     # K_I K_II
        disc = np.sqrt(np.maximum(S * S - 4.0 * P * P, 0.0))
        KI = np.sqrt(np.maximum(0.5 * (S + disc), 0.0))
        KII = np.where(KI > 0, P / np.where(KI > 0, KI, 1.0), 0.0)
        return KI, KII

    @staticmethod
    def hoop_stress_angle(KI, KII):
        """Maximum circumferential stress kink angle (relative to the crack
        direction, positive toward the left normal)."""
        KII_s = np.where(np.abs(KII) > 1e-300, KII, 1e-300)
        th = 2.0 * np.arctan((KI - np.sqrt(KI * KI + 8.0 * KII * KII)) / (4.0 * KII_s))
        return np.where(np.abs(KII) > 1e-12 * (KI + 1e-300), th, 0.0)

    @staticmethod
    def hoop_stress_intensity(KI, KII, theta):
        """``K_θθ(θ) = cos(θ/2) [K_I cos²(θ/2) − 1.5 K_II sin θ]``: the mode-I
        intensity of the kinked crack; growth when it reaches ``K_Ic``."""
        return np.cos(theta / 2) * (KI * np.cos(theta / 2) ** 2 - 1.5 * KII * np.sin(theta))

    def growth_direction(self, J, poly, n_pts, E_prime, **j_kwargs):
        """``(theta_abs, K_I, K_II)``: absolute growth angle from the SIFs of
        the J-vector and the maximum hoop stress criterion."""
        J_t, J_n = J(poly, n_pts, **j_kwargs)
        KI, KII = self.sif_from_j(J_t, J_n, E_prime)
        tip, prev = poly[n_pts - 1], poly[n_pts - 2]
        base = float(np.arctan2(tip[1] - prev[1], tip[0] - prev[0]))
        return base + float(self.hoop_stress_angle(KI, KII)), float(KI), float(KII)


def fe_zero_guess(problem, bc):
    from feax.utils import zero_like_initial_guess
    return zero_like_initial_guess(problem, bc)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def _cell_kinematics_fn(space: CrackSpace, unflatten):
    """``kin(cell_sol_flat, X, phi, psi, mH, mT, xi) -> (x_q, u_q, eps_q, detJ)``."""
    nn, vec = 3, 2

    def kin(cell_sol_flat, X, phi, psi, mH, mT, xi):
        u, a, b = unflatten(cell_sol_flat)                      # (3,2), (3,2), (3,8)
        b = b.reshape(nn, N_BRANCH, vec)
        N, dNdxi = _shape_tri3(xi)                              # (nq,3), (nq,3,2)
        J = np.einsum("ai,qaI->qiI", X, dNdxi)
        detJ = np.linalg.det(J)
        dNdx = np.einsum("qaI,qIi->qai", dNdxi, np.linalg.inv(J))
        x_q = N @ X
        phi_q, psi_q = N @ phi, N @ psi
        gphi = np.einsum("qai,a->qi", dNdx, phi)                # (nq,2) constant per cell
        gpsi = np.einsum("qai,a->qi", dNdx, psi)
        # Heaviside (shifted)
        psiH = (_sgn(phi_q)[:, None] - _sgn(phi)[None, :]) * mH[None, :]          # (nq,3)
        # branch functions (shifted) and their physical gradients
        Fq = jax.vmap(branch_functions)(phi_q, psi_q)                              # (nq,4)
        dFphi, dFpsi = jax.vmap(_branch_jac)(phi_q, psi_q)                         # (nq,4) each
        gF = dFphi[:, :, None] * gphi[:, None, :] + dFpsi[:, :, None] * gpsi[:, None, :]   # (nq,4,2)
        Fn = jax.vmap(branch_functions)(phi, psi)                                  # (3,4)
        psiT = (Fq[:, None, :] - Fn[None, :, :]) * mT[None, :, None]               # (nq,3,4)
        gpsiT = gF[:, None, :, :] * mT[None, :, None, None]                        # (nq,3,4,2)
        # displacement and gradient
        u_q = (N @ u + np.einsum("qa,ai->qi", N * psiH, a)
               + np.einsum("qa,qak,aki->qi", N, psiT, b))
        dM_H = dNdx * psiH[:, :, None]                                             # (nq,3,2)
        dM_T = dNdx[:, :, None, :] * psiT[..., None] + N[:, :, None, None] * gpsiT  # (nq,3,4,2)
        grad_u = (np.einsum("ai,qaj->qij", u, dNdx) + np.einsum("ai,qaj->qij", a, dM_H)
                  + np.einsum("aki,qakj->qij", b, dM_T))
        eps = 0.5 * (grad_u + np.transpose(grad_u, (0, 2, 1)))
        return x_q, u_q, eps, detJ

    return kin


def _cell_energy_fn(space: CrackSpace, unflatten, W):
    kin = _cell_kinematics_fn(space, unflatten)
    kappa = space.penalty

    def energy(cell_sol_flat, X, phi, psi, mH, mT, xi, w):
        x_q, _, eps, detJ = kin(cell_sol_flat, X, phi, psi, mH, mT, xi)
        Wq = jax.vmap(W)(eps, x_q)
        u, a, b = unflatten(cell_sol_flat)
        pen = 0.5 * kappa * (np.sum((1.0 - mH)[:, None] * a * a) + np.sum((1.0 - mT)[:, None] * b * b))
        return np.sum(Wq * w * np.abs(detJ)) + pen

    return energy


@jax.tree_util.register_pytree_node_class
@dataclass
class CrackProblem(Problem):
    """Three-variable XFEM crack problem; build with :func:`crack_problem`.
    ``additional_info = (space, W)``, ``W(eps, x)`` the energy density."""

    def custom_init(self, space: CrackSpace, material: Callable) -> None:
        self.space = space
        self.material = material

    def get_universal_kernel(self) -> Callable:
        energy = _cell_energy_fn(self.space, self.unflatten_fn_dof, self.material)

        def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW,
                   cell_v_grads_JxW, X, phi, psi, mH, mT, xi, w):
            return jax.grad(energy)(cell_sol_flat, X, phi, psi, mH, mT, xi, w)

        return kernel


def crack_problem(space: CrackSpace, W: Callable, **kwargs) -> CrackProblem:
    return CrackProblem(**space.problem_kwargs(), additional_info=(space, W), **kwargs)


__all__ = ["CrackSpace", "CrackProblem", "crack_problem", "crack_level_sets",
           "branch_functions", "plane_strain_energy", "nq_max_for", "NQ_MAX", "N_BRANCH"]
