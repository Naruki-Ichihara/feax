"""Stripe patterns aligned with a direction field on a planar triangle mesh —
experimental.

A port of the core of Knöppel, Crane, Pinkall & Schröder, *Stripe Patterns on
Surfaces* (ACM TOG 2015; code: Keenan Crane's ``StripePatterns``) to flat 2D
``TRI3`` meshes.  Given a nodal *line field* (a direction defined up to sign,
e.g. the filament / fibre orientation of a 3D print) and a frequency, it
computes a phase ``α`` whose iso-lines ``α = 2πk`` are evenly spaced stripes
running **along** the field.  The stripes are the print paths; the zeros of
``sin α`` are the boundaries between neighbouring filaments.

On a flat domain the tangent spaces share one global frame, so the parallel
transport of the original algorithm is the identity and the method reduces to:

1. per edge ``ij``: the phase the stripes should advance between the two
   nodes, ``ω_ij = λ (l_ij / 2) [cos(φ_i − β_ij) + cos(φ_j − β_ij)]`` (the
   trapezoid rule of ``λ X · e_ij``; ``φ`` field angle, ``β`` edge angle,
   ``λ`` frequency).  If the two roots of the line field point in opposite
   directions the edge *crosses sheets* and ``φ_j`` is flipped by ``π``;
2. the stripe energy ``Σ_ij w_ij |ψ_i − e^{iω_ij} ψ_j|²`` (cotan weights,
   complex conjugation across sheet crossings) as a real ``2n × 2n`` matrix;
   its smallest generalized eigenvector w.r.t. the lumped mass matrix is the
   unit complex field ``ψ_i = e^{iα_i}``.  Relaxing ``|ψ| = 1`` is what lets
   the pattern place its own branch points (zeros of ``ψ``) where the field
   has singularities or the spacing cannot be kept;
3. per triangle, the three corner phases are unwrapped consistently with
   ``ω`` and the net turning number ``n`` of the phase around the triangle is
   the stripe *zero index* (a stripe starts or ends there).

Usage
-----

.. code-block:: python

    from feax.experimental.stripe import StripePattern

    sp = StripePattern.compute(mesh, angle, frequency=2 * np.pi / spacing)
    s = sp.scalar()                 # nodal cos α   (1 on stripe centre lines)
    phi = sp.level_set()            # nodal sin α   (0 on the filament boundaries) — orientable fields
    a = sp.phase(face_ids, bary)    # exact phase anywhere, branch points included
    sp.save_vtu("stripes.vtu")

``smooth_line_field`` regularises a noisy field (e.g. principal stress
directions of a topology-optimised part) before computing the pattern.

:class:`DifferentiableStripes` is the same algorithm as a JAX function
``psi(angle, frequency)`` with gradients: the matrix assembly is traced, the
eigenvector is computed by scipy inside a ``pure_callback`` and its VJP comes
from the implicit function theorem (one sparse solve of the bordered
eigen-system per backward pass).  It turns the chain *field angle → stripes →
interface level set → XFEM solve* into one differentiable pipeline, i.e. the
print path becomes an optimisation variable.

Limitations: planar ``TRI3`` meshes; the eigenvector is only differentiable
where the smallest eigenvalue is simple (after fixing the global phase, which
is a free gauge for orientable fields); the sheet-crossing and singular-face
decisions are piecewise constant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import jax
import jax.numpy as np
import numpy as onp
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from feax.mesh import Mesh


# ---------------------------------------------------------------------------
# Mesh connectivity helpers
# ---------------------------------------------------------------------------

def _fmod_pi(x):
    """Wrap to (−π, π]."""
    return (x + onp.pi) % (2 * onp.pi) - onp.pi


def _edges_of(cells: onp.ndarray):
    """Unique undirected edges (i < j) and, per face corner c, the edge id and
    orientation sign of the halfedge from corner c to corner c+1."""
    i = cells
    j = onp.roll(cells, -1, axis=1)
    a, b = onp.minimum(i, j), onp.maximum(i, j)
    key = a.astype(onp.int64) * (cells.max() + 1) + b
    uniq, inv = onp.unique(key.ravel(), return_inverse=True)
    edges = onp.stack([uniq // (cells.max() + 1), uniq % (cells.max() + 1)], axis=1)
    edge_of_corner = inv.reshape(cells.shape)
    sign_of_corner = onp.where(i < j, 1.0, -1.0)
    return edges, edge_of_corner, sign_of_corner


def _cotan_weights(points, cells, edge_of_corner, n_edges):
    """Half-cotangent weights per edge, ``w_ij = (cot α + cot β) / 2``, with the
    contribution of each triangle accumulated on its three edges."""
    w = onp.zeros(n_edges)
    P = points[cells]
    for c in range(3):
        # edge from corner c to c+1 is opposite corner c+2
        p0, p1, p2 = P[:, c], P[:, (c + 1) % 3], P[:, (c + 2) % 3]
        u, v = p0 - p2, p1 - p2
        cot = (u * v).sum(1) / onp.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        onp.add.at(w, edge_of_corner[:, c], 0.5 * cot)
    return w


def _dual_areas(points, cells):
    P = points[cells]
    area = 0.5 * onp.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                         - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    A = onp.zeros(points.shape[0])
    onp.add.at(A, cells.ravel(), onp.repeat(area / 3.0, 3))
    return A, area


def _ccw(points, cells):
    P = points[cells]
    s = (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1])
    out = cells.copy()
    out[s < 0] = cells[s < 0][:, [0, 2, 1]]
    return out


# ---------------------------------------------------------------------------
# Direction field helpers
# ---------------------------------------------------------------------------

def line_field_angle(vectors) -> onp.ndarray:
    """Angle in ``[0, π)`` of a nodal vector field (sign ignored: a line field)."""
    v = onp.asarray(vectors, dtype=onp.float64)
    return onp.arctan2(v[:, 1], v[:, 0]) % onp.pi


def field_index(points, cells, angle) -> onp.ndarray:
    """Index of the line field in every triangle (``±1/2`` singularities give
    ``±1`` for the doubled angle, i.e. the value returned is the index of the
    2-vector field ``e^{2iφ}``; nonsingular faces give 0)."""
    cells = _ccw(onp.asarray(points), onp.asarray(cells))
    phi = onp.asarray(angle)
    idx = onp.zeros(len(cells))
    for c in range(3):
        i, j = cells[:, c], cells[:, (c + 1) % 3]
        idx += _fmod_pi(2.0 * (phi[j] - phi[i]))
    return onp.rint(idx / (2 * onp.pi)).astype(int)


def smooth_line_field(mesh: Mesh, angle, t: float) -> onp.ndarray:
    """Smooth a nodal line field by one implicit step of the 2-vector
    Dirichlet flow: ``(L + M/t) z = M z0`` with ``z = e^{2iφ}`` (cotan
    Laplacian ``L``, lumped mass ``M``).  Larger ``t`` smooths more; the
    result is re-normalised and returned as an angle in ``[0, π)``."""
    points = onp.asarray(mesh.points, dtype=onp.float64)[:, :2]
    cells = _ccw(points, onp.asarray(mesh.cells))
    edges, eoc, _ = _edges_of(cells)
    w = _cotan_weights(points, cells, eoc, len(edges))
    n = points.shape[0]
    i, j = edges[:, 0], edges[:, 1]
    L = sp.coo_matrix((onp.concatenate([w, w, -w, -w]),
                       (onp.concatenate([i, j, i, j]), onp.concatenate([i, j, j, i]))), shape=(n, n)).tocsr()
    A, _ = _dual_areas(points, cells)
    M = sp.diags(A)
    z0 = onp.exp(2j * onp.asarray(angle, dtype=onp.float64))
    z = spla.spsolve((L + M / t).tocsc(), M @ z0)
    return (onp.angle(z) / 2.0) % onp.pi


# ---------------------------------------------------------------------------
# Stripe pattern
# ---------------------------------------------------------------------------

@dataclass
class StripePattern:
    """Result of :meth:`StripePattern.compute`.

    Attributes
    ----------
    mesh : Mesh
        The (CCW-oriented copy of the) input mesh.
    angle : (N,)
        The nodal line-field angle used.
    frequency : float or (N,) array
        ``λ``: stripe phase advance per unit length (spacing ``2π/λ``).
    psi : (N,) complex
        Unit complex field ``e^{iα}`` at the nodes.
    psi_raw : (N,) complex
        The eigenvector before the per-node normalisation: smooth through
        the branch points (where it vanishes), hence the better level set.
    corner_phase : (F, 3)
        Unwrapped phase at the three corners of each triangle (consistent
        within the triangle; triangles are independent "islands").
    zero_index : (F,)
        Turning number of the phase around each triangle: a stripe starts or
        ends in triangles with ``zero_index != 0``.
    field_index : (F,)
        Index of the input line field in each triangle (singularities).
    omega : (E,) and edges : (E, 2)
        Target phase advance per edge and the edge list.
    crosses_sheets : (E,) bool
        Edges where the two roots of the line field point opposite ways.
    """
    mesh: Mesh
    angle: onp.ndarray
    frequency: float
    psi: onp.ndarray
    psi_raw: onp.ndarray
    corner_phase: onp.ndarray
    zero_index: onp.ndarray
    field_index: onp.ndarray
    omega: onp.ndarray
    edges: onp.ndarray
    crosses_sheets: onp.ndarray
    eigenvalue: float

    # ------------------------------------------------------------------ compute
    @classmethod
    def compute(cls, mesh: Mesh, angle, frequency: float, shift: float = 1e-4,
                oriented: bool = False, weight=None) -> "StripePattern":
        """Compute the stripe pattern.

        Parameters
        ----------
        mesh : Mesh
            Planar ``TRI3`` mesh.
        angle : (N,) array
            Nodal field angle in radians.  By default a *line* field (only
            ``angle mod π`` matters; use :func:`line_field_angle` for a vector
            field).  With ``oriented=True`` the angle is a *direction* (mod
            2π): no sheet crossings are introduced, the phase ``α`` is
            globally consistent and ``sin α`` is single-valued — use this
            when the field has a consistent orientation (e.g. a print
            direction) and the interface level set is needed.
        frequency : float or (N,) array
            ``λ`` (radians per unit length): the stripe spacing is ``2π / λ``.
            A nodal array lets the spacing vary in space; where the number of
            stripes then has to change, the pattern inserts branch points
            (edge dislocations).
        shift : float
            Positive diagonal shift that keeps the stripe energy definite
            (does not change the eigenvector).
        oriented : bool
            Treat ``angle`` as an oriented direction field (see above).
        weight : (N,) array, optional
            Nodal multiplier of the stripe energy (edge weight = mean of its
            two nodes).  The *number* of branch points is fixed by the field
            and the frequency, their *position* by the energy: lowering the
            weight in a region (e.g. ``1 − 0.9 exp(−r²/r0²)`` around a point)
            makes it the cheapest place for a branch point and pins it there.
        """
        points = onp.asarray(mesh.points, dtype=onp.float64)[:, :2]
        cells = _ccw(points, onp.asarray(mesh.cells))
        if cells.shape[1] != 3:
            raise NotImplementedError("StripePattern supports TRI3 meshes")
        # The energy makes the phase grow ALONG the field it is given, so the
        # stripes (iso-lines of the phase) are perpendicular to it.  Rotate the
        # user's field by 90 degrees, exactly as the original wrapper does, so
        # that the stripes run along the user's direction (print paths along
        # the fibre direction).
        period = 2 * onp.pi if oriented else onp.pi
        phi_user = onp.asarray(angle, dtype=onp.float64) % period
        phi = (phi_user + onp.pi / 2) % period
        n = points.shape[0]
        edges, eoc, soc = _edges_of(cells)
        n_e = len(edges)
        w = _cotan_weights(points, cells, eoc, n_e)

        # faces with a field singularity do not contribute cotan weight (as in the original)
        fidx = field_index(points, cells, phi_user % onp.pi)
        if onp.any(fidx != 0):
            w_sing = onp.zeros(n_e)
            P = points[cells]
            for c in range(3):
                p0, p1, p2 = P[:, c], P[:, (c + 1) % 3], P[:, (c + 2) % 3]
                u, v = p0 - p2, p1 - p2
                cot = (u * v).sum(1) / onp.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
                onp.add.at(w_sing, eoc[:, c], 0.5 * cot * (fidx != 0))
            w = w - w_sing

        if weight is not None:
            wn = onp.asarray(weight, dtype=onp.float64).reshape(-1)
            w = w * 0.5 * (wn[edges[:, 0]] + wn[edges[:, 1]])

        # --- edge data: sheet crossing and target phase advance ω_ij -------------
        i, j = edges[:, 0], edges[:, 1]
        d = points[j] - points[i]
        length = onp.linalg.norm(d, axis=1)
        beta = onp.arctan2(d[:, 1], d[:, 0])
        Xi, Xj = onp.exp(1j * phi[i]), onp.exp(1j * phi[j])
        s = onp.where(onp.real(onp.conj(Xi) * Xj) > 0.0, 1.0, -1.0)
        if oriented:
            s = onp.ones_like(s)                   # fieldDegree == 1 in the original
        crosses = s < 0
        phi_j = onp.where(crosses, phi[j] + onp.pi, phi[j])
        lam_n = onp.broadcast_to(onp.asarray(frequency, dtype=onp.float64), (n,))
        omega = (length / 2.0) * (lam_n[i] * onp.cos(phi[i] - beta) + lam_n[j] * onp.cos(phi_j - beta))

        # --- stripe energy as a real 2n x 2n sparse matrix -----------------------
        a, b = w * onp.cos(omega), w * onp.sin(omega)
        rows = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        cols = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        vals = [w, w, w, w]
        same = ~crosses
        # same sheet: -[[a, b], [-b, a]] on (i,j) and its transpose on (j,i)
        # crossing : -[[a, -b], [-b, -a]] on both (symmetric)
        ij = [(-a, -b, b, -a), (-a, b, b, a)]
        ji = [(-a, b, -b, -a), (-a, b, b, a)]
        for (m00, m01, m10, m11), (r, c) in (((ij[0], (i, j)), (ji[0], (j, i)))):
            for di, dj, m in ((0, 0, m00), (0, 1, m01), (1, 0, m10), (1, 1, m11)):
                rows.append(2 * r[same] + di); cols.append(2 * c[same] + dj); vals.append(m[same])
        for (m00, m01, m10, m11), (r, c) in (((ij[1], (i, j)), (ji[1], (j, i)))):
            for di, dj, m in ((0, 0, m00), (0, 1, m01), (1, 0, m10), (1, 1, m11)):
                rows.append(2 * r[crosses] + di); cols.append(2 * c[crosses] + dj); vals.append(m[crosses])
        A = sp.coo_matrix((onp.concatenate(vals), (onp.concatenate(rows), onp.concatenate(cols))),
                          shape=(2 * n, 2 * n)).tocsr()
        A = A + shift * sp.identity(2 * n)
        areas, _ = _dual_areas(points, cells)
        M = sp.diags(onp.repeat(areas, 2))

        # --- smallest generalized eigenvector -------------------------------------
        vals_, vecs = spla.eigsh(A.tocsc(), k=1, M=M.tocsc(), sigma=0.0, which="LM")
        g = vecs[:, 0]
        psi_raw = g[0::2] + 1j * g[1::2]
        psi_raw = psi_raw / onp.sqrt(onp.mean(onp.abs(psi_raw) ** 2))     # O(1) amplitude
        psi = psi_raw / onp.abs(psi_raw)

        # --- per-triangle unwrapping (assignTextureCoordinates) ------------------
        corner = onp.zeros((len(cells), 3))
        zero_index = onp.zeros(len(cells), dtype=int)
        e_ij, e_jk, e_ki = eoc[:, 0], eoc[:, 1], eoc[:, 2]
        c_ij, c_jk, c_ki = soc[:, 0], soc[:, 1], soc[:, 2]
        psiI, psiJ, psiK = psi[cells[:, 0]], psi[cells[:, 1]], psi[cells[:, 2]]
        om_ij, om_jk, om_ki = c_ij * omega[e_ij], c_jk * omega[e_jk], c_ki * omega[e_ki]
        x_ij, x_ki = crosses[e_ij], crosses[e_ki]
        psiJ = onp.where(x_ij, onp.conj(psiJ), psiJ)
        om_jk = onp.where(x_ij, -c_jk * omega[e_jk], om_jk)
        psiK = onp.where(x_ki, onp.conj(psiK), psiK)
        om_ki = onp.where(x_ki, -c_ki * omega[e_ki], om_ki)
        om_jk = onp.where(x_ki, c_jk * omega[e_jk], om_jk)
        rij, rjk, rki = onp.exp(1j * om_ij), onp.exp(1j * om_jk), onp.exp(1j * om_ki)
        aI = onp.angle(psiI)
        aJ = aI + om_ij - onp.angle(rij * psiI / psiJ)
        aK = aJ + om_jk - onp.angle(rjk * psiJ / psiK)
        aL = aK + om_ki - onp.angle(rki * psiK / psiI)
        nz = onp.rint((aL - aI) / (2 * onp.pi))
        aJ = aJ - 2 * onp.pi * nz / 3.0
        aK = aK - 4 * onp.pi * nz / 3.0
        corner[:, 0], corner[:, 1], corner[:, 2] = aI, aJ, aK
        zero_index[:] = nz.astype(int)

        freq = float(frequency) if onp.ndim(frequency) == 0 else onp.asarray(frequency, dtype=onp.float64)
        return cls(mesh=Mesh(points, cells, ele_type="TRI3"), angle=phi_user % onp.pi, frequency=freq,
                   psi=psi, psi_raw=psi_raw, corner_phase=corner, zero_index=zero_index, field_index=fidx,
                   omega=omega, edges=edges, crosses_sheets=crosses, eigenvalue=float(vals_[0]))

    # ------------------------------------------------------------- derived fields
    @property
    def orientable(self) -> bool:
        """True when no edge crosses sheets (the line field has a global sign)."""
        return not bool(self.crosses_sheets.any())

    def nodal_phase(self) -> onp.ndarray:
        """Wrapped nodal phase ``α_i = arg ψ_i`` in (−π, π]."""
        return onp.angle(self.psi)

    def scalar(self) -> onp.ndarray:
        """Nodal ``cos α``: 1 on the stripe centre lines, −1 half-way between.
        Well defined for any line field (invariant under ``α → −α``)."""
        return onp.real(self.psi)

    def level_set(self, raw: bool = False, shift: float = 0.0) -> onp.ndarray:
        """Nodal ``sin α``: zero on the filament boundaries ``α = kπ`` with
        alternating sign (the XFEM / NLXP3D-style interface level set).
        Requires an orientable field: across a sheet crossing the sign of
        ``α`` flips and ``sin α`` is not single-valued.

        ``raw=True`` returns ``Im ψ_raw`` instead (``|ψ| sin α``): the same
        zero set, but smooth through the stripe branch points, where the
        normalised phase is undefined — use it when the pattern has
        ``zero_index != 0`` triangles and the level set feeds an interface
        model.

        ``shift`` (radians) moves the interfaces to ``α = shift + kπ``: the
        level set is ``Im(ψ e^{−i shift}) = |ψ| sin(α − shift)``.  Note the
        zero set has period π, so ``shift = π`` gives the same interfaces;
        ``shift = π/2`` moves them from the stripe centre lines (``cos α = ±1``)
        to the stripe edges (``cos α = 0``)."""
        if not self.orientable:
            raise ValueError("sin(alpha) is not single-valued: the line field has sheet crossings; "
                             "give an oriented vector field (consistent sign) or use scalar()")
        z = (self.psi_raw if raw else self.psi) * onp.exp(-1j * shift)
        return onp.imag(z)

    def phase(self, face_ids, bary) -> onp.ndarray:
        """Exact phase at points given by triangle ids (m,) and barycentric
        coordinates (m, 3) w.r.t. the CCW corners of ``self.mesh.cells``,
        including the branch-point correction of the original renderer."""
        face_ids = onp.asarray(face_ids)
        b = onp.asarray(bary, dtype=onp.float64)
        a = (self.corner_phase[face_ids] * b).sum(1)
        bx, by, bz = b[:, 0], b[:, 1], b[:, 2]
        with onp.errstate(divide="ignore", invalid="ignore"):
            log1 = onp.where((bz <= bx) & (bz <= by), (onp.pi / 3) * (1 + (by - bx) / (1 - 3 * bz)),
                   onp.where((bx <= by) & (bx <= bz), (onp.pi / 3) * (3 + (bz - by) / (1 - 3 * bx)),
                             (onp.pi / 3) * (5 + (bx - bz) / (1 - 3 * by))))
        log1 = onp.nan_to_num(log1)
        return a + log1 * self.zero_index[face_ids]

    def stripes_at(self, face_ids, bary, duty: float = 0.5) -> onp.ndarray:
        """Binary filament mask at points: 1 where ``cos α ≥ cos(π duty)``
        (a stripe of relative width ``duty`` around each centre line)."""
        return (onp.cos(self.phase(face_ids, bary)) >= onp.cos(onp.pi * duty)).astype(float)

    # ------------------------------------------------------------------ output
    def save_vtu(self, filename: str, extra_point_fields=(), extra_cell_fields=()) -> None:
        """Write the pattern with :func:`feax.utils.save_sol`: nodal ``cos_alpha``,
        ``sin_alpha`` (if orientable), ``alpha``, ``angle``, the field as a vector,
        and per-cell ``zero_index`` / ``field_index``."""
        from feax.utils import save_sol
        pts = self.mesh.points
        point_infos = [("cos_alpha", self.scalar()), ("alpha", self.nodal_phase()), ("angle", self.angle),
                       ("direction", onp.c_[onp.cos(self.angle), onp.sin(self.angle), onp.zeros(len(pts))])]
        if self.orientable:
            point_infos.append(("sin_alpha", self.level_set()))
        point_infos += list(extra_point_fields)
        cell_infos = [("zero_index", self.zero_index.astype(float)), ("field_index", self.field_index.astype(float))]
        cell_infos += list(extra_cell_fields)
        save_sol(self.mesh, filename, point_infos=point_infos, cell_infos=cell_infos)


# ---------------------------------------------------------------------------
# Differentiable version
# ---------------------------------------------------------------------------

class DifferentiableStripes:
    """Stripe pattern as a differentiable JAX function of the field and frequency.

    ``psi(angle, frequency)`` returns the unit complex nodal field ``e^{iα}``
    (same algorithm and conventions as :meth:`StripePattern.compute`), and
    ``jax.grad`` works through it:

    * the edge phase advances ``ω``, the matrix entries ``a = w cos ω``,
      ``b = w sin ω`` and the final per-node normalisation are ordinary JAX;
    * the smallest generalized eigenvector ``g`` of ``(A(a, b), M)`` is a
      ``custom_vjp`` primitive: forward with ``scipy.sparse.linalg.eigsh`` in a
      ``pure_callback``; backward by the adjoint of the eigen-equation
      ``(A − μM) g = 0, gᵀMg = 1``: solve the bordered system
      ``[[A − μM, Mg], [−(Mg)ᵀ, 0]] [λ; ν] = [ḡ; 0]`` and ``∂L/∂A_kl = −λ_k g_l``.
      For an *orientable* field the eigenvalue is double (the global phase of
      ``ψ`` is free), so the phase is gauged by making ``ψ`` real and positive
      at the node of largest amplitude and the system gets a second border
      row for that gauge.

    Decisions that are piecewise constant in the parameters (sheet crossings,
    singular faces, the gauge node) are made with zero-gradient operations.
    """

    def __init__(self, mesh: Mesh, shift: float = 1e-4, oriented: bool = False):
        points = onp.asarray(mesh.points, dtype=onp.float64)[:, :2]
        cells = _ccw(points, onp.asarray(mesh.cells))
        if cells.shape[1] != 3:
            raise NotImplementedError("DifferentiableStripes supports TRI3 meshes")
        self.oriented = bool(oriented)
        self.mesh = Mesh(points, cells, ele_type="TRI3")
        self.points, self.cells = points, cells
        self.n = points.shape[0]
        self.shift = float(shift)
        edges, eoc, _ = _edges_of(cells)
        self.edges, self.eoc = edges, eoc
        self.n_e = len(edges)
        self.w = _cotan_weights(points, cells, eoc, self.n_e)
        # per-corner cotangents for the singular-face weight removal
        P = points[cells]
        cot = onp.zeros((len(cells), 3))
        for c in range(3):
            p0, p1, p2 = P[:, c], P[:, (c + 1) % 3], P[:, (c + 2) % 3]
            u, v = p0 - p2, p1 - p2
            cot[:, c] = (u * v).sum(1) / onp.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        self._cot = np.asarray(cot)
        self._eoc = np.asarray(eoc)
        d = points[edges[:, 1]] - points[edges[:, 0]]
        self.length = np.asarray(onp.linalg.norm(d, axis=1))
        self.beta = np.asarray(onp.arctan2(d[:, 1], d[:, 0]))
        areas, _ = _dual_areas(points, cells)
        self.M = sp.diags(onp.repeat(areas, 2)).tocsc()
        self._i = np.asarray(edges[:, 0])
        self._j = np.asarray(edges[:, 1])
        self._cells = np.asarray(cells)
        self._eig = self._make_eig_primitive()

    # ----------------------------------------------------------- matrix
    def _matrix(self, a, b, w, cross):
        """scipy CSR of the stripe energy from numpy entry arrays."""
        i, j = self.edges[:, 0], self.edges[:, 1]
        same = ~cross
        rows = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        cols = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        vals = [w, w, w, w]
        ij = [(-a, -b, b, -a), (-a, b, b, a)]
        ji = [(-a, b, -b, -a), (-a, b, b, a)]
        for mask, k in ((same, 0), (cross, 1)):
            for (m00, m01, m10, m11), (r, c) in ((ij[k], (i, j)), (ji[k], (j, i))):
                for di, dj, m in ((0, 0, m00), (0, 1, m01), (1, 0, m10), (1, 1, m11)):
                    rows.append(2 * r[mask] + di); cols.append(2 * c[mask] + dj); vals.append(m[mask])
        A = sp.coo_matrix((onp.concatenate(vals), (onp.concatenate(rows), onp.concatenate(cols))),
                          shape=(2 * self.n, 2 * self.n)).tocsr()
        return A + self.shift * sp.identity(2 * self.n)

    # ----------------------------------------------------------- eigen primitive
    def _make_eig_primitive(self):
        n, n_e = self.n, self.n_e
        M = self.M

        def eig_fwd_np(a, b, w, cross_f):
            cross = onp.asarray(cross_f) > 0.5
            A = self._matrix(onp.asarray(a), onp.asarray(b), onp.asarray(w), cross)
            vals, vecs = spla.eigsh(A.tocsc(), k=2, M=M, sigma=0.0, which="LM")
            order = onp.argsort(vals)
            vals, vecs = vals[order], vecs[:, order]
            g = vecs[:, 0]
            g = g / onp.sqrt(g @ (M @ g))
            psi = g[0::2] + 1j * g[1::2]
            k = int(onp.argmax(onp.abs(psi)))
            # a (numerically) double smallest eigenvalue means the global phase of
            # psi is free (exact for orientable fields): gauge psi_k real positive
            double = (vals[1] - vals[0]) <= 1e-6 * max(abs(vals[0]), 1e-300)
            if double:
                psi = psi * onp.conj(psi[k]) / onp.abs(psi[k])
            else:
                psi = psi * (1.0 if onp.real(psi[k]) >= 0 else -1.0)
            g = onp.empty(2 * n); g[0::2], g[1::2] = onp.real(psi), onp.imag(psi)
            return g, onp.float64(k + (0.5 if double else 0.0))

        def eig_bwd_np(a, b, w, cross_f, g, k, gbar):
            cross = onp.asarray(cross_f) > 0.5
            a, b, w, g, gbar = (onp.asarray(x, dtype=onp.float64) for x in (a, b, w, g, gbar))
            A = self._matrix(a, b, w, cross)
            mu = g @ (A @ g)
            Mg = M @ g
            double = (float(k) % 1.0) > 0.25          # flag packed with the gauge node
            k = int(onp.floor(float(k)))
            if double:
                e = sp.coo_matrix(([1.0], ([2 * k + 1], [0])), shape=(2 * n, 1))
                B = sp.bmat([[A - mu * M, sp.csc_matrix(Mg[:, None]), e],
                             [sp.csc_matrix(-Mg[None, :]), None, None],
                             [e.T, None, None]]).tocsc()
                rhs = onp.concatenate([gbar, [0.0, 0.0]])
            else:
                B = sp.bmat([[A - mu * M, sp.csc_matrix(Mg[:, None])],
                             [sp.csc_matrix(-Mg[None, :]), None]]).tocsc()
                rhs = onp.concatenate([gbar, [0.0]])
            try:
                lam = spla.splu(B).solve(rhs)[:2 * n]
            except RuntimeError:
                # degenerate configuration (e.g. a nearly double eigenvalue that was
                # not detected): fall back to a tiny Tikhonov regularisation
                eps = 1e-10 * max(abs(mu), 1.0)
                lam = spla.splu((B + eps * sp.identity(B.shape[0])).tocsc()).solve(rhs)[:2 * n]
            i, j = self.edges[:, 0], self.edges[:, 1]
            l0, l1, l2, l3 = lam[2 * i], lam[2 * i + 1], lam[2 * j], lam[2 * j + 1]
            g0, g1, g2, g3 = g[2 * i], g[2 * i + 1], g[2 * j], g[2 * j + 1]
            # dL/dA_kl = -lam_k g_l, chained through the 2x2 blocks
            abar_same = l0 * g2 + l1 * g3 + l2 * g0 + l3 * g1
            bbar_same = l0 * g3 - l1 * g2 - l2 * g1 + l3 * g0
            abar_cross = l0 * g2 - l1 * g3 + l2 * g0 - l3 * g1
            bbar_cross = -(l0 * g3 + l1 * g2 + l2 * g1 + l3 * g0)
            abar = onp.where(cross, abar_cross, abar_same)
            bbar = onp.where(cross, bbar_cross, bbar_same)
            # diagonal entries w: dL/dw_e = -(lam . g) on the 4 diagonal slots
            wbar = -(l0 * g0 + l1 * g1 + l2 * g2 + l3 * g3)
            return abar, bbar, wbar

        @jax.custom_vjp
        def eig(a, b, w, cross_f):
            g, _ = jax.pure_callback(
                eig_fwd_np, (jax.ShapeDtypeStruct((2 * n,), np.float64), jax.ShapeDtypeStruct((), np.float64)),
                a, b, w, cross_f)
            return g

        def fwd(a, b, w, cross_f):
            g, k = jax.pure_callback(
                eig_fwd_np, (jax.ShapeDtypeStruct((2 * n,), np.float64), jax.ShapeDtypeStruct((), np.float64)),
                a, b, w, cross_f)
            return g, (a, b, w, cross_f, g, k)

        def bwd(res, gbar):
            a, b, w, cross_f, g, k = res
            abar, bbar, wbar = jax.pure_callback(
                eig_bwd_np, tuple(jax.ShapeDtypeStruct((n_e,), np.float64) for _ in range(3)),
                a, b, w, cross_f, g, k, gbar)
            return abar, bbar, wbar, np.zeros_like(cross_f)

        eig.defvjp(fwd, bwd)
        return eig

    # ----------------------------------------------------------- public
    def psi(self, angle, frequency):
        """Unit complex nodal field ``e^{iα}`` (N,), differentiable in both arguments."""
        period = 2 * np.pi if self.oriented else np.pi
        phi_user = np.asarray(angle, dtype=np.float64) % period
        phi = (phi_user + np.pi / 2) % period
        i, j = self._i, self._j
        cross = (np.cos(phi[i] - phi[j]) <= 0.0) & (not self.oriented)
        cross_f = jax.lax.stop_gradient(cross.astype(np.float64))
        phi_user = phi_user % np.pi
        phi_j = phi[j] + np.pi * cross_f
        omega = frequency * (self.length / 2.0) * (np.cos(phi[i] - self.beta) + np.cos(phi_j - self.beta))
        # singular faces (piecewise constant): drop their cotan contributions
        c = self._cells
        idx = sum(np.remainder(2.0 * (phi_user[c[:, (k + 1) % 3]] - phi_user[c[:, k]]) + np.pi, 2 * np.pi) - np.pi
                  for k in range(3))
        singular = jax.lax.stop_gradient((np.abs(np.rint(idx / (2 * np.pi))) > 0.5).astype(np.float64))
        w_sing = np.zeros(self.n_e).at[self._eoc.reshape(-1)].add((0.5 * self._cot * singular[:, None]).reshape(-1))
        w = np.asarray(self.w) - w_sing
        a, b = w * np.cos(omega), w * np.sin(omega)
        g = self._eig(a, b, w, cross_f)
        z = g[0::2] + 1j * g[1::2]
        return z / np.abs(z)

    def scalar(self, angle, frequency):
        """Nodal ``cos α`` (differentiable)."""
        return np.real(self.psi(angle, frequency))

    def level_set(self, angle, frequency):
        """Nodal ``sin α`` (differentiable; meaningful for orientable fields)."""
        return np.imag(self.psi(angle, frequency))


__all__ = ["StripePattern", "DifferentiableStripes", "line_field_angle", "field_index", "smooth_line_field"]
