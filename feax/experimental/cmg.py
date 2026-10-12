"""Compact geometric multigrid (cmg) on a structured narrow band — experimental.

A matrix-free, O(band) geometric-multigrid preconditioned CG (MGPCG) for
LINEAR, SYMMETRIC POSITIVE DEFINITE problems on a uniform HEX8
:class:`feax.StructuredGrid` — the memory-scalable solver for the giga-voxel
narrow-band regime, where an assembled factorization or an algebraic-MG
hierarchy would exhaust memory.

What is fixed and what is not
-----------------------------
Fixed (the reasons cmg is O(band)):

* a uniform HEX8 structured grid — connectivity and coordinates are index
  arithmetic, never materialised;
* a linear operator ``K(params) u`` that is symmetric positive definite once the
  Dirichlet DOFs are removed — CG, the Gauss-Seidel smoother and the
  "adjoint = one more forward solve" of the design gradient all rely on it.
  One-way coupled physics (e.g. temperature -> thermal strain -> displacement)
  is NOT one SPD system; solve it as two cmg instances sharing the same band
  geometry (see :func:`build_geometry`).

Free (chosen through a :class:`Material`):

* the number of DOFs per node ``vec`` (3 = elasticity, 1 = diffusion, ...);
* the constitutive law. :class:`ScalarMaterial` is the fast path — one
  reference element matrix ``KE`` times a per-cell scalar (SIMP, RAMP, ...),
  one float of state per cell. :class:`KernelMaterial` takes any *linear* feax
  ``tensor_map`` (``u_grad (vec,3) -> flux (vec,3)``) and evaluates it through
  the per-cell constitutive matrix ``D = d flux / d u_grad`` — anisotropic,
  orientation-dependent, multi-material, ... at ``(3*vec)**2`` floats per cell
  and roughly 5x the matvec cost. :meth:`KernelMaterial.from_problem` pulls the
  ``tensor_map`` (or the gradient of the energy density) out of an ordinary
  :class:`feax.Problem`, so the cmg operator is generated from the same kernel
  feax assembles from.

Conventions
-----------
Global node/DOF numbering matches ``StructuredGrid`` (node
``n = i*(ny+1)*(nz+1) + j*(nz+1) + k``, DOF ``vec*n + c``), so the returned
field is on the same band-node layout as ``NarrowBand(grid, active)``
(``node_map``). The element-local corner ordering here is internal (the 8
corners ``(dx,dy,dz)`` with ``dz`` fastest) and differs from feax's HEX8 order;
the assembled operator and the solution are identical.

Smoother: 8-colour block (``vec x vec`` node) Gauss-Seidel. Transfers:
trilinear prolongation / full-weighting restriction on the compact active set.
Coarsest level: cuDSS (via spineax, default) or a matrix-free block-Jacobi
Krylov solve — selected by the feax ``solver_options`` passed to
:meth:`NarrowBandCMG.create_solver`.

Usage::

    from feax.experimental import NarrowBandCMG, ScalarMaterial, KernelMaterial, simp

    cmg    = NarrowBandCMG(grid, fixed_pred)                # isotropic SIMP elasticity
    cmg    = NarrowBandCMG(grid, fixed_pred, material=ScalarMaterial.isotropic_conduction(
                 grid.spacing, scale_fn=simp(1e-3, 1.0, 3.0)))          # vec=1 diffusion
    cmg    = NarrowBandCMG(grid, fixed_pred, material=KernelMaterial.from_problem(
                 MyProblem, grid, vec=3, additional_info=(...)))       # any linear Problem

    levels = cmg.build(active_cells)
    b      = cmg.load_vector(levels, node_ids, comp=2, value=-1.0)
    b      = cmg.face_load_vector(levels, axis=1, side=0, traction=(0, -1.0, 0))
    solver = cmg.create_solver(levels, b)          # None -> cuDSS coarsest
    u      = solver(params)                        # differentiable (implicit vjp)
    u      = solver(params, b=other_rhs)           # ... in the design AND the rhs
    # non-zero Dirichlet values: lift them (u_p holds the values on fixed DOFs)
    u      = u_p + solver(params, b=cmg.dirichlet_rhs(levels, params, u_p))

A grid built by :meth:`feax.StructuredGrid.from_mesh` from a ``box_mesh`` shares
its node and cell numbering, so a full-grid cmg field and a feax Problem on the
box mesh use one layout (no permutation).
"""

import numpy as onp
import jax
import jax.numpy as jnp
import scipy.sparse as sp

try:                                    # optional: only the cuDSS coarse solve needs it
    from feax.solvers._cudss_compat import factorize as _factorize, solve_with as _solve_with
except ImportError:
    _factorize = _solve_with = None

# cmg's element-local corner order: (dx,dy,dz), dz fastest
_CORNERS = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]   # 8
# feax's HEX8 corner order (box_mesh / StructuredGrid), for the Problem bridge
from ..spgrid import _HEX8_OFFSETS as _FEAX_HEX8_OFFSETS_ARR
_FEAX_HEX8_OFFSETS = [tuple(int(v) for v in row) for row in _FEAX_HEX8_OFFSETS_ARR]
# permutation p with cmg_corner[a] == feax_corner[p[a]]
FEAX_TO_CMG_CORNER = onp.array([_FEAX_HEX8_OFFSETS.index(c) for c in _CORNERS])


def _as_tuple(params):
    return tuple(params) if isinstance(params, (tuple, list)) else (params,)


def _design_of(params_or_tp):
    """Band design from a bare array / tuple of arrays or a TracedParams
    (``volume_vars``: the single array if there is one, else the tuple)."""
    from ..traced_params import TracedParams
    if isinstance(params_or_tp, TracedParams):
        vv = tuple(params_or_tp.volume_vars)
        return vv[0] if len(vv) == 1 else vv
    return params_or_tp


# ---------------------------------------------------------------------------
# Trilinear HEX8 on a box cell
# ---------------------------------------------------------------------------

def hex_shape_grads(spacing=1.0):
    """Shape-function gradients of the trilinear hex on a box cell of size
    ``spacing`` (scalar or (3,)), at its 8 Gauss points.

    Returns ``(dNdx, w)``: ``dNdx`` of shape ``(8 gp, 8 nodes, 3)`` (nodes in
    cmg corner order) and the quadrature weights ``w`` of shape ``(8,)`` (sum =
    cell volume)."""
    h = onp.broadcast_to(onp.asarray(spacing, float), (3,))
    g = 0.5 / onp.sqrt(3.0)
    pts = [(a, b, c) for a in (0.5 - g, 0.5 + g) for b in (0.5 - g, 0.5 + g)
           for c in (0.5 - g, 0.5 + g)]
    corners = onp.array(_CORNERS, float)                       # (8,3)
    dNdx = onp.zeros((8, 8, 3))
    for q, xi in enumerate(pts):
        xi = onp.asarray(xi)
        # factor_i(a) = xi_i if corner_i == 1 else 1 - xi_i
        fac = onp.where(corners > 0.5, xi[None, :], 1.0 - xi[None, :])    # (8,3)
        sgn = onp.where(corners > 0.5, 1.0, -1.0)
        for i in range(3):
            others = [j for j in range(3) if j != i]
            dNdx[q, :, i] = sgn[:, i] * fac[:, others[0]] * fac[:, others[1]] / h[i]
    w = onp.full(8, onp.prod(h) / 8.0)
    return dNdx, w


def gradient_operator(spacing=1.0, vec=3):
    """``B`` of shape ``(8 gp, 3*vec, 8*vec)`` mapping the element DOF vector
    (index ``vec*a + c``: node-major, component-minor) to the flattened
    gradient ``u_grad.reshape(-1)`` (row-major ``(vec, 3)``: index ``3*c + i``),
    plus the weights ``w`` ``(8,)``."""
    dNdx, w = hex_shape_grads(spacing)
    B = onp.zeros((8, 3 * vec, 8 * vec))
    for a in range(8):
        for c in range(vec):
            for i in range(3):
                B[:, 3 * c + i, vec * a + c] = dNdx[:, a, i]
    return B, w


def constitutive_matrix(tensor_map, vec, *params):
    """``D = d flux / d u_grad`` at zero gradient, as ``(3*vec, 3*vec)`` on the
    row-major flattened gradient (JAX; differentiable in ``params``)."""
    z = jnp.zeros((vec, 3))
    J = jax.jacfwd(lambda g: tensor_map(g, *params))(z)     # (vec,3,vec,3)
    return J.reshape(3 * vec, 3 * vec)


def element_matrix(B, w, D):
    """``K_e = sum_g w_g B_g^T D B_g`` for one constitutive matrix."""
    return jnp.einsum('g,gie,ij,gjf->ef', w, B, D, B)


def isotropic_stress(nu):
    """feax-style ``tensor_map`` of unit-modulus isotropic linear elasticity."""
    lam = nu / ((1 + nu) * (1 - 2 * nu))
    mu = 1.0 / (2 * (1 + nu))

    def stress(u_grad):
        eps = 0.5 * (u_grad + u_grad.T)
        return lam * jnp.trace(eps) * jnp.eye(3) + 2.0 * mu * eps
    return stress


def make_KE_3d(nu=0.3, E=1.0, spacing=1.0):
    """``24x24`` stiffness of an isotropic linear-elastic box HEX8 cell (unit
    cube by default), in cmg corner order."""
    B, w = gradient_operator(spacing, 3)
    D = constitutive_matrix(isotropic_stress(nu), 3)
    return E * onp.asarray(element_matrix(jnp.asarray(B), jnp.asarray(w), D))


def make_KE_conduction(spacing=1.0, k=1.0):
    """``8x8`` conductivity matrix of a box HEX8 cell (``div(k grad T)``)."""
    B, w = gradient_operator(spacing, 1)
    return k * onp.asarray(element_matrix(jnp.asarray(B), jnp.asarray(w), jnp.eye(3)))


def unit_cell_mesh(grid):
    """One-cell :class:`feax.Mesh` (feax HEX8 order) with the grid's spacing."""
    from ..mesh import Mesh
    h = onp.broadcast_to(onp.asarray(grid.spacing, float), (3,))
    pts = onp.array(_FEAX_HEX8_OFFSETS, float) * h[None, :]
    return Mesh(pts, onp.arange(8, dtype=onp.int64)[None, :], ele_type='HEX8')


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def simp(Emin=1e-9, E0=1.0, penal=3.0):
    """SIMP scale ``rho -> Emin + rho**penal * (E0 - Emin)``."""
    return lambda rho: Emin + rho ** penal * (E0 - Emin)


def ramp(Emin=1e-9, E0=1.0, q=8.0):
    """RAMP scale ``rho -> Emin + (E0 - Emin) * rho / (1 + q (1 - rho))``."""
    return lambda rho: Emin + (E0 - Emin) * rho / (1.0 + q * (1.0 - rho))


class Material:
    """Constitutive interface of cmg. Everything is JAX over a leading cell axis.

    ``state0(params)``            per-cell material state on level 0 from the
                                  design parameters (an array or a tuple of
                                  arrays over the band cells);
    ``matvec(state, ue)``         ``(n, 8*vec) -> (n, 8*vec)`` element matvec;
    ``node_blocks(state)``        ``(n, 8, vec, vec)`` diagonal corner blocks;
    ``element_matrices(state)``   ``(n, 8*vec, 8*vec)`` (coarsest level only).

    ``coarsen`` (average children into the parent) and ``pad`` are generic.
    """

    vec = None

    def state0(self, params):
        raise NotImplementedError

    def matvec(self, state, ue):
        raise NotImplementedError

    def node_blocks(self, state):
        raise NotImplementedError

    def element_matrices(self, state):
        raise NotImplementedError

    def coarsen(self, state, pos, invcnt, ncoarse):
        s = jnp.zeros((ncoarse,) + state.shape[1:], state.dtype).at[pos].add(state)
        return s * invcnt.reshape((-1,) + (1,) * (state.ndim - 1))

    def pad(self, state, capacity):
        return jnp.zeros((capacity,) + state.shape[1:], state.dtype).at[:state.shape[0]].set(state)


class ScalarMaterial(Material):
    """``K_e = scale_fn(*params_e) * KE`` — one float of state per cell.

    ``KE``: the reference element matrix ``(8*vec, 8*vec)`` in cmg corner
    order; ``scale_fn``: JAX ``(*params_cell) -> scalar`` (default SIMP).
    """

    def __init__(self, KE, scale_fn=None):
        KE = onp.asarray(KE, float)
        if KE.ndim != 2 or KE.shape[0] != KE.shape[1] or KE.shape[0] % 8:
            raise ValueError(f"KE must be (8*vec, 8*vec); got {KE.shape}")
        self.vec = KE.shape[0] // 8
        self.KE = jnp.asarray(KE)
        v = self.vec
        self.KEd = jnp.asarray(onp.stack([KE[v*a:v*a+v, v*a:v*a+v] for a in range(8)]))
        self.scale_fn = scale_fn if scale_fn is not None else simp()

    @classmethod
    def isotropic_elastic(cls, spacing=1.0, nu=0.3, scale_fn=None):
        """``vec=3`` isotropic linear elasticity (unit modulus times ``scale_fn``)."""
        return cls(make_KE_3d(nu, 1.0, spacing), scale_fn)

    @classmethod
    def isotropic_conduction(cls, spacing=1.0, scale_fn=None):
        """``vec=1`` steady diffusion ``div(k grad T)`` (unit conductivity times
        ``scale_fn``)."""
        return cls(make_KE_conduction(spacing, 1.0), scale_fn)

    def state0(self, params):
        return self.scale_fn(*_as_tuple(params))

    def matvec(self, state, ue):
        return state[:, None] * (ue @ self.KE.astype(ue.dtype))

    def node_blocks(self, state):
        return state[:, None, None, None] * self.KEd.astype(state.dtype)[None]

    def element_matrices(self, state):
        return state[:, None, None] * self.KE.astype(state.dtype)[None]


class KernelMaterial(Material):
    """Any linear feax ``tensor_map`` — state is the per-cell constitutive
    matrix ``D_e = d flux / d u_grad`` of shape ``(3*vec, 3*vec)``.

    ``tensor_map(u_grad, *params_cell) -> flux`` follows feax's
    ``Problem.get_tensor_map`` convention (``u_grad``/``flux`` of shape
    ``(vec, 3)``); the design ``params`` given to the solver is a tuple of
    per-cell arrays (or one array) in the same order. Cell-constant parameters
    only (no nodal internal variables).

    Memory: ``(3*vec)**2`` floats per cell (81 for elasticity) on every level.
    Cost: about 5x the :class:`ScalarMaterial` matvec.

    ``sample_params``: per-cell parameter values used to verify at construction
    that the map is linear in ``u_grad`` and that ``D`` is symmetric (SPD is
    what cmg relies on). Pass it whenever the law is not obviously linear.
    """

    def __init__(self, tensor_map, vec, spacing=1.0, *, sample_params=None):
        self.tensor_map = tensor_map
        self.vec = int(vec)
        B, w = gradient_operator(spacing, self.vec)
        self.B, self.w = jnp.asarray(B), jnp.asarray(w)
        if sample_params is not None:
            self.check_linear(*_as_tuple(sample_params))

    @classmethod
    def from_problem(cls, problem, grid, vec=None, *, additional_info=(),
                     sample_params=None, **problem_kw):
        """The material of a :class:`feax.Problem`.

        ``problem`` is either an INSTANCE (its maps are used as they are; the
        mesh it lives on is irrelevant, ``vec`` is read from it) or a SUBCLASS,
        which is instantiated on a one-cell mesh with the grid's spacing
        (``problem(mesh, vec=vec, dim=3, ele_type='HEX8',
        additional_info=additional_info, **problem_kw)``).

        The constitutive map is ``get_tensor_map()`` when the Problem defines
        it, else the gradient of ``get_energy_density()`` with respect to the
        field gradient (``flux = d psi / d u_grad``) — so energy-based Problems
        work unchanged. Either way it must be linear in the field
        (``sample_params`` checks that)."""
        from ..problem import Problem
        if isinstance(problem, type):
            if vec is None:
                raise ValueError("vec is required when a Problem class is given")
            problem = problem(mesh=unit_cell_mesh(grid), vec=vec, dim=3,
                              ele_type='HEX8', additional_info=tuple(additional_info),
                              **problem_kw)
        elif not isinstance(problem, Problem):
            raise TypeError("problem must be a feax Problem instance or subclass")
        pvec = int(problem.vec[0] if isinstance(problem.vec, (list, tuple)) else problem.vec)
        if vec is None:
            vec = pvec
        elif int(vec) != pvec:
            raise ValueError(f"vec={vec} but the Problem has vec={pvec}")
        tm = problem.get_tensor_map()
        if tm is None:
            psi = problem.get_energy_density()
            if psi is None:
                raise ValueError(f"{type(problem).__name__} defines neither "
                                 "get_tensor_map nor get_energy_density")
            tm = jax.grad(psi, argnums=0)
        return cls(tm, vec, grid.spacing, sample_params=sample_params)

    def check_linear(self, *params_cell, rtol=1e-6):
        """Raise ``ValueError`` unless ``tensor_map`` is linear in ``u_grad``
        with a symmetric ``D`` at ``params_cell``."""
        rng = onp.random.default_rng(0)
        g1, g2 = (jnp.asarray(rng.standard_normal((self.vec, 3))) for _ in range(2))
        f = lambda g: self.tensor_map(g, *params_cell)
        scale = float(jnp.abs(f(g1)).max()) + 1e-30
        if float(jnp.abs(f(jnp.zeros_like(g1))).max()) > rtol * scale:
            raise ValueError("tensor_map(0) != 0: not linear (affine offset)")
        if float(jnp.abs(f(g1 + g2) - f(g1) - f(g2)).max()) > rtol * scale:
            raise ValueError("tensor_map is not linear in u_grad")
        if float(jnp.abs(f(2.5 * g1) - 2.5 * f(g1)).max()) > rtol * scale:
            raise ValueError("tensor_map is not homogeneous in u_grad")
        D = self.constitutive(*params_cell)
        if float(jnp.abs(D - D.T).max()) > rtol * (float(jnp.abs(D).max()) + 1e-30):
            raise ValueError("constitutive matrix is not symmetric: cmg needs an SPD operator")

    def constitutive(self, *params_cell):
        return constitutive_matrix(self.tensor_map, self.vec, *params_cell)

    def state0(self, params):
        return jax.vmap(self.constitutive)(*_as_tuple(params))       # (n, 3v, 3v)

    def matvec(self, state, ue):
        B, w = self.B.astype(ue.dtype), self.w.astype(ue.dtype)
        g = jnp.einsum('gie,ne->ngi', B, ue)                         # (n, 8gp, 3v)
        f = jnp.matmul(g, jnp.swapaxes(state, 1, 2))                 # D g at each gp (D symmetric)
        return jnp.einsum('g,gie,ngi->ne', w, B, f)

    def node_blocks(self, state):
        v = self.vec
        B, w = self.B.astype(state.dtype), self.w.astype(state.dtype)
        Ba = B.reshape(8, 3 * v, 8, v)                               # (gp, i, a, c)
        return jnp.einsum('g,giac,nij,gjad->nacd', w, Ba, state, Ba)

    def element_matrices(self, state):
        B, w = self.B.astype(state.dtype), self.w.astype(state.dtype)
        return jnp.einsum('g,gie,nij,gjf->nef', w, B, state, B)


# ---------------------------------------------------------------------------
# Band geometry (physics-independent, shareable between cmg instances)
# ---------------------------------------------------------------------------

def _cap(n, bucket=512):
    """Round ``n`` up to a multiple of ``bucket`` (padded capacity)."""
    return int(-(-max(int(n), 1) // bucket) * bucket)


def auto_levels(dims, floor=16):
    """Geometric MG depth: ceil-halve each dim (works for arbitrary, incl. odd,
    grid sizes) while the smallest dim of the next level would still be
    ``>= floor``, so every level has at least ``floor`` cells per axis.

    The default ``floor=16`` stops coarsening while the coarsest level is still
    a few thousand cells per short dimension, and leaves the rest to the
    coarsest-level solver (cuDSS factors ~1e5 DOF in milliseconds). Deeper
    hierarchies cost more sequential V-cycle work per iteration AND converge
    worse for anisotropic / high-contrast materials, whose coarse operators
    (averaged material state) degrade with every level: on a 33^3 anisotropic
    RAMP elasticity problem, 5 levels (coarsest 3^3) took 71 PCG iterations and
    2 levels (coarsest 17^3) took 16, ten times faster overall."""
    L, d = 1, list(dims)
    while (min(d) + 1) // 2 >= floor:          # coarsen only while the result stays >= floor
        d = [(x + 1) // 2 for x in d]; L += 1
    return L


def build_geometry(grid, cells0, n_levels=None):
    """Per-level band geometry (cells, nodes, corner->node map, transfers) for
    an active-cell set — everything cmg needs that does not depend on the
    physics. Pass the result to :meth:`NarrowBandCMG.build` of several cmg
    instances (e.g. a ``vec=1`` and a ``vec=3`` one) to share it."""
    L = n_levels or auto_levels((grid.nx, grid.ny, grid.nz))
    levels = []
    cells = onp.sort(onp.asarray(cells0, onp.int64))
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    for l in range(L):
        nyz, nz1 = ny * nz, nz + 1
        nn1 = (ny + 1) * (nz + 1)
        ex = cells // nyz; rem = cells % nyz; ey = rem // nz; ez = rem % nz
        cn = onp.stack([(ex + dx) * nn1 + (ey + dy) * nz1 + (ez + dz)
                        for (dx, dy, dz) in _CORNERS], axis=1)
        nodes = onp.unique(cn)
        ncid = onp.searchsorted(nodes, cn)                          # (na, 8) compact
        ni = nodes // nn1; rr = nodes % nn1; nj = rr // nz1; nk = rr % nz1
        color = (ni % 2) * 4 + (nj % 2) * 2 + (nk % 2)
        levels.append(dict(cells=cells, nodes=nodes, ncid=ncid, nnode=nodes.size,
                           color=color, ni=ni, nj=nj, nk=nk, nx=nx, ny=ny, nz=nz))
        if l < L - 1:
            # ceil-halved coarse dims -> arbitrary (incl. odd) sizes coarsen;
            # each fine cell (ex,ey,ez) maps to coarse (ex//2,ey//2,ez//2).
            cnx, cny, cnz = (nx + 1) // 2, (ny + 1) // 2, (nz + 1) // 2
            cells = onp.unique((ex // 2) * (cny * cnz) + (ey // 2) * cnz + (ez // 2))
            nx, ny, nz = cnx, cny, cnz
    for l in range(L - 1):
        _build_transfer(levels[l], levels[l + 1])
    return levels


def _build_transfer(fine, coarse):
    ni, nj, nk = fine["ni"], fine["nj"], fine["nk"]
    cnn1 = (coarse["ny"] + 1) * (coarse["nz"] + 1); cnz1 = coarse["nz"] + 1
    ncoarse = coarse["nnode"]
    ex_e, ey_e, ez_e = ni % 2 == 0, nj % 2 == 0, nk % 2 == 0
    ix = (ni // 2, ni // 2 + 1); wx = (onp.where(ex_e, 1.0, 0.5), onp.where(ex_e, 0.0, 0.5))
    iy = (nj // 2, nj // 2 + 1); wy = (onp.where(ey_e, 1.0, 0.5), onp.where(ey_e, 0.0, 0.5))
    iz = (nk // 2, nk // 2 + 1); wz = (onp.where(ez_e, 1.0, 0.5), onp.where(ez_e, 0.0, 0.5))
    nn = fine["nnode"]
    P_idx = onp.full((nn, 8), ncoarse, onp.int64); P_w = onp.zeros((nn, 8))
    k = 0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                cx, cy, cz = ix[a], iy[b], iz[c]
                w = wx[a] * wy[b] * wz[c]
                valid = (w > 0) & (cx <= coarse["nx"]) & (cy <= coarse["ny"]) & (cz <= coarse["nz"])
                gid = onp.where(valid, cx * cnn1 + cy * cnz1 + cz, -1)
                pos = onp.clip(onp.searchsorted(coarse["nodes"], gid), 0, ncoarse - 1)
                hit = valid & (coarse["nodes"][pos] == gid)
                P_idx[hit, k] = pos[hit]; P_w[hit, k] = w[hit]
                k += 1
    fine["P_idx"], fine["P_w"] = P_idx, P_w


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class NarrowBandCMG:
    """Matrix-free O(band) geometric-MG (MGPCG) solver for a structured band.

    ``grid``       : a :class:`feax.StructuredGrid`.
    ``fixed_pred`` : ``(ni,nj,nk,nx,ny,nz) -> bool`` over band nodes (by grid
                     index): ``(n,)`` fixes every component of a node,
                     ``(n, vec)`` fixes components individually.
    ``material``   : a :class:`Material` (default: isotropic SIMP elasticity
                     built from ``nu``/``Emin``/``E0``/``penal`` and the grid
                     spacing). Sets ``vec``.
    ``n_levels``   : MG levels (default from :func:`auto_levels`: coarsen while
                     the smallest dim stays >= 16; shallow is faster here).
    ``bucket``     : per-level padded capacities are rounded up to a multiple of
                     this, so a moving band reuses the compiled MGPCG while its
                     sizes stay in the same bucket.
    ``precision``  : ``"double"`` runs everything in float64. ``"mixed"`` runs
                     the multigrid PRECONDITIONER (smoother, transfers, coarse
                     Krylov) in float32 and keeps the outer CG — residual,
                     search direction, the level-0 matvec and the cuDSS coarse
                     solve — in float64, so the converged solution is a float64
                     solution to ``cg_tol``. On GPUs with slow float64 (consumer
                     / GB10 class) this makes a V-cycle several times cheaper.

    Usage (feax create_solver convention — one differentiable solver)::

        cmg    = NarrowBandCMG(grid, fixed_pred, nu=0.3, penal=3.0)
        levels = cmg.build(active_cells)             # or cmg.build(build_geometry(...))
        b      = cmg.load_vector(levels, node_ids, comp=2, value=-1.0)
        solver = cmg.create_solver(levels, b)        # None -> cuDSS coarsest
        u      = solver(params)                      # array / tuple / TracedParams
        dc     = jax.grad(lambda r: jnp.dot(b, solver(r)))(params)
    """

    def __init__(self, grid, fixed_pred, *, material=None, n_levels=None,
                 nu=0.3, Emin=1e-9, E0=1.0, penal=3.0, pre=2, post=2,
                 cg_tol=1e-8, cg_maxit=200, bucket=512, precision="double"):
        self.grid = grid
        self.bucket = int(bucket)
        if precision not in ("double", "mixed"):
            raise ValueError(f"precision must be 'double' or 'mixed', got {precision!r}")
        self.precision = precision
        self.nelx, self.nely, self.nelz = grid.nx, grid.ny, grid.nz
        self.L = n_levels or auto_levels((grid.nx, grid.ny, grid.nz))
        self.fixed_pred = fixed_pred
        if material is None:
            material = ScalarMaterial.isotropic_elastic(grid.spacing, nu, simp(Emin, E0, penal))
        self.material = material
        self.vec = int(material.vec)
        self.pre, self.post = pre, post
        self.cg_tol, self.cg_maxit = cg_tol, cg_maxit

    @staticmethod
    def _parse_coarse(solver_options):
        """Map a feax solver_options to the coarsest-level solver config.
        None / DirectSolverOptions -> cuDSS (factor-once/solve-many);
        KrylovSolverOptions -> matrix-free block-Jacobi Krylov (cg/bicgstab/
        gmres) with that tol/maxiter."""
        from feax.solvers.options import KrylovSolverOptions
        if isinstance(solver_options, KrylovSolverOptions):
            name = str(getattr(solver_options, "solver", "cg")).lower()
            if name not in ("cg", "bicgstab", "gmres"):
                name = "cg"
            return dict(cudss=False, name=name,
                        tol=float(getattr(solver_options, "tol", 1e-10)),
                        maxit=int(getattr(solver_options, "maxiter", 2000)))
        if _factorize is None:
            raise ImportError("cuDSS coarse solve needs spineax; pass a "
                              "KrylovSolverOptions for the matrix-free coarse solve")
        return dict(cudss=True, name="cudss", tol=0.0, maxit=0)

    # --- band structure ------------------------------------------------------
    def _free_dofs(self, geo):
        fp = onp.asarray(self.fixed_pred(geo["ni"], geo["nj"], geo["nk"],
                                         geo["nx"], geo["ny"], geo["nz"]))
        n, v = geo["nnode"], self.vec
        if fp.ndim <= 1:
            fixed = onp.broadcast_to(fp.reshape(-1) if fp.ndim else fp, (n,))
            fixed = onp.repeat(fixed, v)
        elif fp.shape == (n, v):
            fixed = fp.reshape(-1)
        else:
            raise ValueError(f"fixed_pred must return ({n},) or ({n},{v}); got {fp.shape}")
        return (~fixed).astype(onp.float64)

    def build(self, cells_or_geometry):
        """Per-level band structure for an active-cell set (global cell ids) or
        for a :func:`build_geometry` result (shared between instances)."""
        if isinstance(cells_or_geometry, list) and cells_or_geometry \
                and isinstance(cells_or_geometry[0], dict):
            geometry = cells_or_geometry
            if len(geometry) != self.L:
                raise ValueError(f"geometry has {len(geometry)} levels, cmg expects {self.L}")
        else:
            geometry = build_geometry(self.grid, cells_or_geometry, self.L)
        v = self.vec
        levels = []
        for geo in geometry:
            ncid = geo["ncid"]
            edof_c = onp.empty((ncid.shape[0], 8 * v), onp.int64)
            for a in range(8):
                for c in range(v):
                    edof_c[:, v * a + c] = v * ncid[:, a] + c
            levels.append({**geo, "edof_c": edof_c, "ndof": v * geo["nnode"],
                           "free": self._free_dofs(geo)})
        return levels

    # === jitted O(band) path (values on device) ===============================
    def _mv_j(self, u, lev):
        ue = u[lev["edof"]]
        ye = self.material.matvec(lev["state"], ue)
        return jnp.zeros(u.shape[0], u.dtype).at[lev["edof"].reshape(-1)].add(ye.reshape(-1)) * lev["free"]

    def _dinv_j(self, lev):
        v = self.vec
        CN = lev["color"].shape[0]
        nc = lev["edof"][:, 0::v] // v                                   # (CE, 8)
        nb = self.material.node_blocks(lev["state"])
        blk = jnp.zeros((CN, v, v), nb.dtype).at[nc].add(nb)
        F = lev["free"].reshape(CN, v).astype(nb.dtype)
        # fixed components: zero their rows/columns, unit diagonal (also every padded node)
        blk = blk * F[:, :, None] * F[:, None, :] + jnp.eye(v, dtype=nb.dtype)[None] * (1.0 - F)[:, :, None]
        return jnp.linalg.inv(blk)

    def _smooth_j(self, u, b, lev, Dinv, sweeps):
        v = self.vec
        CN = lev["color"].shape[0]
        for _ in range(sweeps):
            for c in range(8):
                r = (b - self._mv_j(u, lev)) * lev["free"]
                du = jnp.einsum("nab,nb->na", Dinv, r.reshape(CN, v))
                un = u.reshape(CN, v) + (lev["color"] == c)[:, None] * du
                u = un.reshape(-1) * lev["free"]
        return u

    def _coarse_solve_cg(self, b, lev, Dinv, coarse):
        """Matrix-free (block-Jacobi preconditioned) Krylov solve of the coarsest
        level — no cuDSS. Operator = element matvec + identity on fixed DOFs
        (matches the cuDSS coarse matrix); preconditioner = the node-block
        inverse ``Dinv``."""
        v = self.vec
        fc = lev["free"]
        CN = lev["color"].shape[0]
        A = lambda x: self._mv_j(x, lev) + (1.0 - fc) * x
        M = lambda r: jnp.einsum("nab,nb->na", Dinv, r.reshape(CN, v)).reshape(-1)
        kw = dict(tol=coarse["tol"], atol=0.0, maxiter=coarse["maxit"], M=M)
        if coarse["name"] == "bicgstab":
            x, _ = jax.scipy.sparse.linalg.bicgstab(A, b * fc, **kw)
        elif coarse["name"] == "gmres":
            x, _ = jax.scipy.sparse.linalg.gmres(A, b * fc, **kw)
        else:
            x, _ = jax.scipy.sparse.linalg.cg(A, b * fc, **kw)
        return x * fc

    def _vcycle_j(self, b, P, Dinvs, token, coarse, l):
        v = self.vec
        lev = P[l]
        if l == self.L - 1:
            fc = lev["free"]
            if coarse["cudss"]:                             # cuDSS factors in float64
                return _solve_with(token, (b * fc).astype(jnp.float64)).astype(b.dtype) * fc
            return self._coarse_solve_cg(b, lev, Dinvs[l], coarse)
        u = jnp.zeros_like(b)
        u = self._smooth_j(u, b, lev, Dinvs[l], self.pre)
        r = (b - self._mv_j(u, lev)) * lev["free"]
        clev = P[l + 1]; CNc = clev["color"].shape[0]
        rn = r.reshape(lev["color"].shape[0], v)
        rc = jnp.zeros((CNc, v), b.dtype).at[lev["Pidx"]].add(lev["Pw"][:, :, None] * rn[:, None, :])
        rc = rc.reshape(-1) * clev["free"]
        ec = self._vcycle_j(rc, P, Dinvs, token, coarse, l + 1)
        ecn = ec.reshape(CNc, v)
        uf = (lev["Pw"][:, :, None] * ecn[lev["Pidx"]]).sum(axis=1)
        u = u + uf.reshape(-1) * lev["free"]
        return self._smooth_j(u, b, lev, Dinvs[l], self.post)

    def _solve_j(self, P, Pm, b, u0, token, coarse):
        """MGPCG. ``P``: per-level structure + state in float64 (outer CG and
        the level-0 matvec); ``Pm``: the same in the preconditioner's dtype
        (float32 for ``precision="mixed"``, else ``P`` itself)."""
        Dinvs = [self._dinv_j(Pm[l]) for l in range(self.L - 1)]
        # coarsest needs a block-Jacobi preconditioner only for the iterative path
        Dinvs += [None if coarse["cudss"] else self._dinv_j(Pm[-1])]
        f0 = P[0]["free"]; b = b * f0; u = u0 * f0
        A = lambda x: self._mv_j(x, P[0])
        mdt = Pm[0]["free"].dtype

        def M(r):                                           # V-cycle in the preconditioner dtype
            return self._vcycle_j(r.astype(mdt), Pm, Dinvs, token, coarse, 0).astype(r.dtype)

        r = b - A(u); r0 = jnp.max(jnp.abs(r)) + 1e-30; tol = self.cg_tol * r0
        z = M(r); rz = jnp.vdot(r, z)

        def cond(st):
            u, r, p, rz, k = st
            return (jnp.max(jnp.abs(r)) > tol) & (k < self.cg_maxit)

        def body(st):
            u, r, p, rz, k = st
            Ap = A(p); alpha = rz / (jnp.vdot(p, Ap) + 1e-30)
            u = u + alpha * p; r = r - alpha * Ap
            z = M(r); rz_new = jnp.vdot(r, z)
            p = z + (rz_new / (rz + 1e-30)) * p
            return (u, r, p, rz_new, k + 1)

        u, r, p, rz, k = jax.lax.while_loop(cond, body, (u, r, z, rz, 0))
        return u, k, jnp.max(jnp.abs(r)) / r0

    # === convenience API: loads, sensitivity, combined solve ================
    def compact_nodes(self, levels, node_ids):
        """Global grid node ids -> compact band node indices (level 0)."""
        return onp.searchsorted(levels[0]["nodes"], onp.asarray(node_ids, onp.int64))

    def load_vector(self, levels, node_ids, comp=None, value=-1.0):
        """Nodal-load RHS on the compact band DOFs: set component ``comp`` (in
        ``range(vec)``; default: the last one — ``z`` for elasticity, the only
        one for a scalar field) to ``value`` at every global node in
        ``node_ids`` (must be in the band)."""
        v = self.vec
        comp = v - 1 if comp is None else int(comp)
        if not 0 <= comp < v:
            raise ValueError(f"comp={comp} out of range for vec={v}")
        b = onp.zeros(levels[0]["ndof"])
        cn = self.compact_nodes(levels, node_ids)
        b[v * cn + comp] = value
        return b

    def face_load_vector(self, levels, axis, side, traction):
        """Consistent nodal loads of a uniform traction on one boundary face of
        the grid (``axis`` 0/1/2, ``side`` 0 = the ``index == 0`` face, 1 = the
        ``index == n`` face), restricted to the band's cells on that face.
        ``traction`` is the force per unit area, ``(vec,)``, applied in the
        POSITIVE coordinate directions. feax's ``surface_map`` convention adds
        the map to the residual, so a feax traction ``t`` corresponds to
        ``traction = -t`` here.

        For bilinear faces under uniform traction the consistent load is one
        quarter of ``traction * face_area`` at each of the face's 4 nodes."""
        v = self.vec
        t = onp.broadcast_to(onp.asarray(traction, float), (v,))
        lev = levels[0]
        h = onp.broadcast_to(onp.asarray(self.grid.spacing, float), (3,))
        area = float(onp.prod(h) / h[axis])
        cx, cy, cz = self.grid.cell_ijk(lev["cells"])
        idx = (cx, cy, cz)[axis]
        n_axis = (lev["nx"], lev["ny"], lev["nz"])[axis]
        on_face = (idx == 0) if side == 0 else (idx == n_axis - 1)
        corners = onp.array(_CORNERS)
        face_corners = onp.nonzero(corners[:, axis] == int(side))[0]         # 4 of 8
        ncid = lev["ncid"][on_face][:, face_corners]                           # (nf, 4)
        b = onp.zeros(lev["ndof"])
        for c in range(v):
            if t[c] != 0.0:
                onp.add.at(b, v * ncid.reshape(-1) + c, 0.25 * area * t[c])
        return b

    def _matvec0(self, levels, params, u):
        """Pure-JAX level-0 matvec ``K(params) u`` on the band DOFs (fixed rows
        zeroed), differentiable in both arguments."""
        lev = levels[0]
        edof = jnp.asarray(lev["edof_c"]); free = jnp.asarray(lev["free"])
        ue = u[edof]
        ye = self.material.matvec(self.material.state0(_design_of(params)), ue)
        return jnp.zeros(lev["ndof"]).at[edof.reshape(-1)].add(ye.reshape(-1)) * free

    def dirichlet_rhs(self, levels, params, prescribed):
        """Right-hand side of the lifting for NON-ZERO Dirichlet values.

        ``prescribed`` is a band DOF vector holding the prescribed values on the
        fixed DOFs (anything on free DOFs is ignored). With ``u_0 = solver(params,
        b=dirichlet_rhs(...))`` the field is ``u = prescribed_on_fixed + u_0``:
        ``K_ff u_0 = -K_fc u_c``. Differentiable in ``params`` (the RHS depends
        on the design), so ``jax.grad`` through ``solver(params, b=...)`` is
        complete."""
        lev = levels[0]
        free = jnp.asarray(lev["free"])
        u_c = jnp.asarray(prescribed) * (1.0 - free)
        return -self._matvec0(levels, params, u_c)

    def compliance_and_dc(self, levels, params, u):
        """Analytic self-adjoint compliance and sensitivity on the band cells:
        ``c = u^T K(params) u``, ``dc = -u^T (dK/dparams) u`` (no adjoint solve).
        ``params``: array, tuple of arrays or TracedParams; ``dc`` has the same
        structure (numpy)."""
        p = _design_of(params)
        u = jnp.asarray(u)
        edof = jnp.asarray(levels[0]["edof_c"])
        ue = u[edof]

        def energy(p_):
            return jnp.sum(ue * self.material.matvec(self.material.state0(p_), ue))

        c, vjp = jax.vjp(energy, p)
        (g,) = vjp(jnp.ones_like(c))
        dc = jax.tree_util.tree_map(lambda x: -onp.asarray(x), g)
        return float(c), dc

    # === the solver: create_solver(levels, b, solver_options) ================
    # (implicit diff via custom_vjp)
    # Structure (edof/free/color/transfers/coarsen maps/coarse CSR pattern) is
    # design-independent -> built once, concrete. Only the VALUES (per-level
    # material state, the coarsest CSR data + its factorization) are JAX
    # functions of the design, so the forward MGPCG is jittable and accepts
    # tracers. We do NOT autodiff through the MGPCG / coarse factorization: u
    # solves K(p)u=b, so the adjoint is another solve K λ = ∂J/∂u (K symmetric)
    # and the design gradient is -λ^T (∂K/∂p) u, obtained by jax.vjp of a
    # pure-JAX level-0 matvec.
    def create_solver(self, levels, b=None, solver_options=None, jit=True,
                      return_solution=True):
        """Build the cmg solver for this band, matching feax's
        ``create_solver`` convention: returns a single callable
        ``solver(params, b=None) -> u`` that is differentiable (implicit-diff
        ``custom_vjp``) in the design AND in the right-hand side. ``b`` (compact
        band DOF layout) given here is the default load; a ``b`` passed to the
        call overrides it (e.g. a design-dependent :meth:`dirichlet_rhs`).
        Compliance sensitivity for the analytic/OC path is
        ``compliance_and_dc(levels, params, solver(params))``.

        ``solver_options`` selects the coarsest-level solver: ``None`` /
        ``DirectSolverOptions`` → cuDSS; ``KrylovSolverOptions`` → matrix-free
        block-Jacobi Krylov (cg/bicgstab/gmres), no cuDSS dependency. The
        geometric-MG smoother/transfers are intrinsic to cmg.

        ``jit=True`` (default) wraps the MGPCG in ``jax.jit``; the compiled
        executable is reused across bands whose padded (bucketed) shapes match.
        ``jit=False`` runs eagerly — no compilation, so a band whose shape
        changes every iteration avoids the per-iteration recompile (pair with
        ``bucket=1`` for an exact-size band). Result and gradient are identical.

        The result is a :class:`feax.Solution` (one field,
        ``(num_band_nodes, vec)``) by default; ``return_solution=False`` gives
        the raw flat band vector.
        """
        coarse = self._parse_coarse(solver_options)
        L, v, mat = self.L, self.vec, self.material
        # --- padded, design-independent per-level structure (no state) ---
        S = []
        for l, lev in enumerate(levels):
            CN, CE, DOF = (_cap(lev["nnode"], self.bucket),
                           _cap(lev["cells"].size, self.bucket),
                           v * _cap(lev["nnode"], self.bucket))
            edof = onp.zeros((CE, 8 * v), onp.int32); edof[:lev["cells"].size] = lev["edof_c"]
            free = onp.zeros(DOF); free[:lev["ndof"]] = lev["free"]
            color = onp.zeros(CN, onp.int32); color[:lev["nnode"]] = lev["color"]
            d = dict(edof=jnp.asarray(edof), free=jnp.asarray(free), color=jnp.asarray(color),
                     na=lev["cells"].size, CE=CE)
            if l < L - 1:
                Pi = onp.zeros((CN, 8), onp.int32); Pi[:lev["nnode"]] = lev["P_idx"]
                Pw = onp.zeros((CN, 8)); Pw[:lev["nnode"]] = lev["P_w"]
                d["Pidx"] = jnp.asarray(Pi); d["Pw"] = jnp.asarray(Pw)
            S.append(d)

        # --- coarsen scatter maps (state fine->coarse averaging) ---
        coarsen = []
        for l in range(1, L):
            fine, clev = levels[l - 1], levels[l]
            fnyz = fine["ny"] * fine["nz"]
            fex = fine["cells"] // fnyz; frem = fine["cells"] % fnyz
            fey = frem // fine["nz"]; fez = frem % fine["nz"]
            cnyz = clev["ny"] * clev["nz"]
            parent = (fex // 2) * cnyz + (fey // 2) * clev["nz"] + (fez // 2)
            pos = onp.searchsorted(clev["cells"], parent)            # (na_fine,)
            cnt = onp.zeros(clev["cells"].size); onp.add.at(cnt, pos, 1.0)
            coarsen.append((jnp.asarray(pos), jnp.asarray(1.0 / onp.maximum(cnt, 1)),
                            clev["cells"].size))

        # --- coarsest CSR pattern (free-masked + identity on fixed) + data map ---
        # The matrix is built at the PADDED coarsest size (identity on the padded
        # DOFs) so the cuDSS token matches the padded RHS the V-cycle hands it.
        levc = levels[-1]; edc = levc["edof_c"]; nac = edc.shape[0]
        ndc = S[-1]["free"].shape[0]
        ne = 8 * v
        e_i = onp.broadcast_to(onp.arange(nac)[:, None, None], (nac, ne, ne)).ravel()
        a_i = onp.broadcast_to(onp.arange(ne)[None, :, None], (nac, ne, ne)).ravel()
        b_i = onp.broadcast_to(onp.arange(ne)[None, None, :], (nac, ne, ne)).ravel()
        ri = edc[e_i, a_i]; rj = edc[e_i, b_i]
        freec = onp.asarray(S[-1]["free"])                       # padded: 0 beyond ndof
        keep = (freec[ri] > 0) & (freec[rj] > 0)
        e_i, a_i, b_i, ri, rj = (x[keep] for x in (e_i, a_i, b_i, ri, rj))
        fixed = onp.nonzero(freec == 0)[0]                        # fixed + padded DOFs
        # pattern = free-free element entries + a diagonal for every fixed dof
        pr = onp.concatenate([ri, fixed]); pc = onp.concatenate([rj, fixed])
        K0 = sp.coo_matrix((onp.ones(pr.size), (pr, pc)), shape=(ndc, ndc)).tocsr()
        K0.sort_indices()
        indptr, indices = K0.indptr.astype(onp.int32), K0.indices.astype(onp.int32)
        nnz = int(indices.size)
        rows = onp.repeat(onp.arange(ndc, dtype=onp.int64), onp.diff(indptr))
        csr_keys = rows * ndc + indices.astype(onp.int64)
        ent_slot = onp.searchsorted(csr_keys, ri.astype(onp.int64) * ndc + rj).astype(onp.int32)
        fix_slot = onp.searchsorted(csr_keys, fixed.astype(onp.int64) * ndc + fixed).astype(onp.int32)
        cpat = dict(indptr=jnp.asarray(indptr), indices=jnp.asarray(indices), nnz=nnz,
                    ent_e=jnp.asarray(e_i.astype(onp.int32)),
                    ent_a=jnp.asarray(a_i.astype(onp.int32)),
                    ent_b=jnp.asarray(b_i.astype(onp.int32)),
                    ent_slot=jnp.asarray(ent_slot), fix_slot=jnp.asarray(fix_slot))

        # --- level-0 matvec as a pure-JAX function of the design (for the vjp) ---
        edof0 = jnp.asarray(levels[0]["edof_c"].astype(onp.int32))      # (na0, 8v)
        free0 = jnp.asarray(levels[0]["free"]); ndof0 = levels[0]["ndof"]

        def mv0(params, u):
            ue = u[edof0]
            ye = mat.matvec(mat.state0(params), ue)
            out = jnp.zeros(ndof0).at[edof0.reshape(-1)].add(ye.reshape(-1))
            return out * free0

        def states_from_params(params):
            states = [mat.state0(params)]
            for pos, invcnt, ncoarse in coarsen:
                states.append(mat.coarsen(states[-1], pos, invcnt, ncoarse))
            return [mat.pad(states[l], S[l]["CE"]) for l in range(L)]

        def coarse_token(state_c):
            Ke = mat.element_matrices(state_c[:nac])                    # (nac, 8v, 8v)
            data = jnp.zeros(cpat["nnz"]).at[cpat["ent_slot"]].add(
                Ke[cpat["ent_e"], cpat["ent_a"], cpat["ent_b"]])
            data = data.at[cpat["fix_slot"]].add(1.0)
            return _factorize(data, cpat["indptr"], cpat["indices"], mtype_id=3, mview_id=0)

        DOF0 = S[0]["free"].shape[0]
        b_default = None if b is None else jnp.asarray(b)
        mixed = self.precision == "mixed"
        if mixed:                     # float32 copies of the float arrays of the structure
            S32 = [{k: (val.astype(jnp.float32) if isinstance(val, jax.Array)
                        and jnp.issubdtype(val.dtype, jnp.floating) else val)
                    for k, val in d.items()} for d in S]

        # Cache the jitted MGPCG on the instance, keyed by the coarse config. The
        # structure ``P`` is passed as arguments (not baked into the closure), so
        # the compiled executable is reused across bands whose padded shapes match
        # (bucketed). states_from_params / coarse_token stay eager (cheap).
        key = dict(coarse, precision=self.precision)
        if jit:
            if getattr(self, "_jsj_key", None) != key:
                _c = dict(coarse)
                self._jsj = jax.jit(lambda P, Pm, rhs_, u0_, tok_:
                                    self._solve_j(P, Pm, rhs_, u0_, tok_, _c))
                self._jsj_key = key
            _jsj = self._jsj
        else:                                               # eager: no compile, no shape reuse
            _c = dict(coarse)
            _jsj = lambda P, Pm, rhs_, u0_, tok_: self._solve_j(P, Pm, rhs_, u0_, tok_, _c)

        def raw_solve_stats(params, rhs):
            states = states_from_params(params)
            token = coarse_token(states[-1]) if coarse["cudss"] else None
            P = [{**S[l], "state": states[l]} for l in range(L)]
            if mixed:
                Pm = [{**S32[l], "state": states[l].astype(jnp.float32)} for l in range(L)]
            else:
                Pm = P
            u, k, res = _jsj(P, Pm, rhs, jnp.zeros(DOF0), token)
            return u[:ndof0], k, res

        def raw_solve(params, rhs):
            return raw_solve_stats(params, rhs)[0]

        def _pad_rhs(rhs):
            return jnp.zeros(DOF0).at[:ndof0].set(rhs)

        @jax.custom_vjp
        def solve(params, rhs):
            return raw_solve(params, _pad_rhs(rhs))

        def solve_fwd(params, rhs):
            u = raw_solve(params, _pad_rhs(rhs))
            return u, (params, u)

        def solve_bwd(res, u_bar):
            params, u = res
            lam = raw_solve(params, _pad_rhs(u_bar))          # adjoint (K symmetric)
            _, vjp = jax.vjp(lambda p: mv0(p, u), params)
            (g,) = vjp(lam)
            # d u / d rhs = K^-1  ->  rhs_bar = lam ; d u / d params = -K^-1 (dK/dp) u
            return (jax.tree_util.tree_map(jnp.negative, g), lam)

        solve.defvjp(solve_fwd, solve_bwd)

        _layout = ((levels[0]["nnode"], v),)

        def solver(params, b=None):
            """Accepts the band design as a bare array, a tuple of arrays, or a
            ``TracedParams`` (``volume_vars``), matching the feax solver
            convention ``solver(traced_params) -> u``, and optionally a
            right-hand side ``b`` (band DOF layout) overriding the default.
            Differentiable in every form (the pytree extraction is transparent
            to ``jax.grad``)."""
            rhs = b_default if b is None else jnp.asarray(b)
            if rhs is None:
                raise ValueError("no right-hand side: pass b to create_solver or to the call")
            u = solve(_design_of(params), rhs)
            if return_solution:
                from ..solution import Solution
                return Solution(u, _layout)
            return u

        def stats(params, b=None):
            """Diagnostics of one forward solve (no vjp): ``(iterations,
            relative residual)`` of the MGPCG — the way to see whether a solve
            is hitting ``cg_maxit`` or converging in a few V-cycles."""
            rhs = b_default if b is None else jnp.asarray(b)
            _, k, res = raw_solve_stats(_design_of(params), _pad_rhs(rhs))
            return int(k), float(res)

        solver.stats = stats
        return solver
