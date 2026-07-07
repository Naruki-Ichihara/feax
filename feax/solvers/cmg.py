"""Compact geometric multigrid (cmg) solver for structured narrow bands.

A matrix-free, O(band) geometric multigrid preconditioned CG (MGPCG) for
LINEAR ELASTICITY on a uniform HEX8 :class:`feax.StructuredGrid` — the
memory-scalable solver for the giga-voxel structured narrow-band regime, where
an assembled direct factorization (3D fill) or an algebraic-MG hierarchy would
exhaust memory.

Scope: uniform structured grid, linear elasticity with per-cell SIMP scaling
E = Emin + rho^p (E0-Emin). The global node/DOF numbering matches
``StructuredGrid`` (node n = i*(ny+1)*(nz+1)+j*(nz+1)+k, DOF 3n+c), so the
returned displacement is on the same band-node layout as
``NarrowBand(grid, active)`` (node_map). The element-local corner ordering here
is internal (paired with make_KE_3d) and independent of feax's HEX8 order — the
assembled operator and the solution are identical.

Smoother: 8-colour block (3x3 node) Gauss-Seidel. Transfers: trilinear
prolongation / full-weighting restriction on the compact active set. Coarsest
level: cuDSS (via spineax, default) or a matrix-free block-Jacobi Krylov solve
— selected by the feax ``solver_options`` passed to
:meth:`NarrowBandCMG.create_solver`.
"""

import numpy as onp
import jax
import jax.numpy as jnp
import scipy.sparse as sp

try:                                    # optional: only the cuDSS coarse solve needs it
    from spineax.cudss.factor_solve import factorize as _factorize, solve_with as _solve_with
except ImportError:
    _factorize = _solve_with = None

_CORNERS = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]   # 8


def _design_of(rho_or_tp):
    """Band design array from a bare array or a TracedParams (volume_vars[0])."""
    from ..traced_params import TracedParams
    if isinstance(rho_or_tp, TracedParams):
        return rho_or_tp.volume_vars[0]
    return rho_or_tp


def make_KE_3d(nu=0.3, E=1.0):
    """24x24 stiffness of a unit-cube trilinear hex; local node order = the 8
    corners (dx,dy,dz), dz fastest — matches the compact edof in this module."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = onp.array([[lam + 2*mu, lam, lam, 0, 0, 0], [lam, lam + 2*mu, lam, 0, 0, 0],
                   [lam, lam, lam + 2*mu, 0, 0, 0], [0, 0, 0, mu, 0, 0],
                   [0, 0, 0, 0, mu, 0], [0, 0, 0, 0, 0, mu]])
    corners = onp.array(_CORNERS, float)
    signs = 2 * corners - 1
    g = 1 / onp.sqrt(3)
    gp = [(a, b, c) for a in (-g, g) for b in (-g, g) for c in (-g, g)]
    KE = onp.zeros((24, 24))
    for xi, eta, ze in gp:
        dN = onp.zeros((3, 8))
        for a in range(8):
            sx, sy, sz = signs[a]
            dN[0, a] = 0.125 * sx * (1 + sy*eta) * (1 + sz*ze)
            dN[1, a] = 0.125 * sy * (1 + sx*xi) * (1 + sz*ze)
            dN[2, a] = 0.125 * sz * (1 + sx*xi) * (1 + sy*eta)
        J = dN @ corners
        detJ = onp.linalg.det(J)
        dNxyz = onp.linalg.solve(J, dN)
        B = onp.zeros((6, 24))
        B[0, 0::3] = dNxyz[0]; B[1, 1::3] = dNxyz[1]; B[2, 2::3] = dNxyz[2]
        B[3, 0::3] = dNxyz[1]; B[3, 1::3] = dNxyz[0]
        B[4, 1::3] = dNxyz[2]; B[4, 2::3] = dNxyz[1]
        B[5, 0::3] = dNxyz[2]; B[5, 2::3] = dNxyz[0]
        KE += B.T @ D @ B * detJ
    return KE


def _cap(n, bucket=512):
    import math
    return int(math.ceil(max(1, n) / bucket) * bucket)


def auto_levels(dims, floor=4):
    """Geometric MG depth: ceil-halve each dim (works for arbitrary, incl. odd,
    grid sizes) until the smallest dim reaches ``floor``."""
    L, d = 1, list(dims)
    while min(d) > floor:
        d = [(x + 1) // 2 for x in d]; L += 1
    return L


def _node_diag_blocks_3d(KE):
    return onp.stack([KE[3*a:3*a+3, 3*a:3*a+3] for a in range(8)])


class NarrowBandCMG:
    """Matrix-free O(band) geometric-MG (MGPCG) solver for a structured band.

    ``grid``       : a :class:`feax.StructuredGrid`.
    ``fixed_pred`` : ``(ni,nj,nk,nx,ny,nz) -> bool`` marking fixed (Dirichlet)
                     nodes by grid index (e.g. ``lambda ni,nj,nk,nx,ny,nz: nk==0``).
    ``n_levels``   : MG levels (default from :func:`auto_levels`).
    ``bucket``     : per-level padded capacities are rounded up to a multiple of
                     this, so a moving band reuses the compiled MGPCG while its
                     sizes stay in the same bucket (coarser bucket = fewer
                     recompiles, more padding per solve).

    Usage (feax create_solver convention — one differentiable solver)::

        cmg    = fe.NarrowBandCMG(grid, fixed_pred, nu=0.3, penal=3.0)
        levels = cmg.build(active_cells)
        b      = cmg.load_vector(levels, node_ids, comp=2, value=-1.0)
        solver = cmg.create_solver(levels, b)      # None -> cuDSS coarsest
        u      = solver(rho_cells)                 # bare array OR TracedParams
        u      = solver(fe.TracedParams(volume_vars=(rho_cells,)))   # same
        dc     = jax.grad(lambda r: jnp.dot(b, solver(r)))(rho_cells)
    """

    def __init__(self, grid, fixed_pred, *, n_levels=None, nu=0.3,
                 Emin=1e-9, E0=1.0, penal=3.0, pre=2, post=2,
                 cg_tol=1e-8, cg_maxit=200, bucket=512):
        self.bucket = int(bucket)
        self.nelx, self.nely, self.nelz = grid.nx, grid.ny, grid.nz
        self.L = n_levels or auto_levels((grid.nx, grid.ny, grid.nz))
        self.fixed_pred = fixed_pred
        self.KE = onp.asarray(make_KE_3d(nu))
        self.KEd = _node_diag_blocks_3d(self.KE)
        self.Emin, self.E0, self.penal = Emin, E0, penal
        self.pre, self.post = pre, post
        self.cg_tol, self.cg_maxit = cg_tol, cg_maxit

    @staticmethod
    def _parse_coarse(solver_options):
        """Map a feax solver_options to the coarsest-level solver config.
        None / DirectSolverOptions -> cuDSS (factor-once/solve-many);
        KrylovSolverOptions -> matrix-free block-Jacobi Krylov (cg/bicgstab/
        gmres) — no cuDSS dependency."""
        from .options import DirectSolverOptions, KrylovSolverOptions
        if solver_options is None or isinstance(solver_options, DirectSolverOptions):
            if _factorize is None:
                raise ImportError(
                    "The cuDSS coarsest-level solve requires spineax "
                    "(spineax.cudss.factor_solve). Install spineax, or pass "
                    "KrylovSolverOptions for the matrix-free coarse solve.")
            return dict(cudss=True)
        if isinstance(solver_options, KrylovSolverOptions):
            name = solver_options.solver if solver_options.solver != "auto" else "cg"
            return dict(cudss=False, name=name, tol=solver_options.tol,
                        maxit=solver_options.maxiter)
        raise ValueError("solver_options must be None, DirectSolverOptions (cuDSS), "
                         "or KrylovSolverOptions")

    def build(self, cells0):
        levels = []
        cells = onp.sort(onp.asarray(cells0, onp.int64))
        nx, ny, nz = self.nelx, self.nely, self.nelz
        for l in range(self.L):
            nyz, nz1 = ny * nz, nz + 1
            nn1 = (ny + 1) * (nz + 1)
            ex = cells // nyz; rem = cells % nyz; ey = rem // nz; ez = rem % nz
            cn = onp.stack([(ex + dx) * nn1 + (ey + dy) * nz1 + (ez + dz)
                            for (dx, dy, dz) in _CORNERS], axis=1)
            nodes = onp.unique(cn)
            ncid = onp.searchsorted(nodes, cn)
            nnode = nodes.size
            edof_c = onp.empty((cells.size, 24), onp.int64)
            for a in range(8):
                edof_c[:, 3*a] = 3 * ncid[:, a]
                edof_c[:, 3*a + 1] = 3 * ncid[:, a] + 1
                edof_c[:, 3*a + 2] = 3 * ncid[:, a] + 2
            ni = nodes // nn1; rr = nodes % nn1; nj = rr // nz1; nk = rr % nz1
            free_node = ~self.fixed_pred(ni, nj, nk, nx, ny, nz)
            free_dof = onp.repeat(free_node.astype(onp.float64), 3)
            color = (ni % 2) * 4 + (nj % 2) * 2 + (nk % 2)
            levels.append(dict(cells=cells, nodes=nodes, edof_c=edof_c, nnode=nnode,
                               ndof=3*nnode, free=free_dof, color=color,
                               ni=ni, nj=nj, nk=nk, nx=nx, ny=ny, nz=nz))
            if l < self.L - 1:
                # ceil-halved coarse dims -> arbitrary (incl. odd) sizes coarsen;
                # each fine cell (ex,ey,ez) maps to coarse (ex//2,ey//2,ez//2).
                cnx, cny, cnz = (nx + 1) // 2, (ny + 1) // 2, (nz + 1) // 2
                cells = onp.unique((ex // 2) * (cny * cnz) + (ey // 2) * cnz + (ez // 2))
                nx, ny, nz = cnx, cny, cnz
        for l in range(self.L - 1):
            self._build_transfer(levels[l], levels[l + 1])
        return levels

    def _build_transfer(self, fine, coarse):
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

    # === jitted O(band) path (values on device, fixed CSR pattern) ============
    def _mv_j(self, u, lev):
        ue = u[lev["edof"]]
        ye = lev["E"][:, None] * (ue @ self.KE)
        return jnp.zeros(u.shape[0]).at[lev["edof"].reshape(-1)].add(ye.reshape(-1)) * lev["free"]

    def _dinv_j(self, lev):
        CN = lev["color"].shape[0]
        blk = jnp.zeros((CN, 3, 3))
        nc = lev["edof"][:, 0::3] // 3
        for a in range(8):
            blk = blk.at[nc[:, a]].add(lev["E"][:, None, None] * self.KEd[a][None])
        fn = lev["free"].reshape(CN, 3)[:, 0] > 0
        blk = jnp.where(fn[:, None, None], blk, jnp.eye(3)[None])
        return jnp.linalg.inv(blk)

    def _smooth_j(self, u, b, lev, Dinv, sweeps):
        CN = lev["color"].shape[0]
        for _ in range(sweeps):
            for c in range(8):
                r = (b - self._mv_j(u, lev)) * lev["free"]
                du = jnp.einsum("nab,nb->na", Dinv, r.reshape(CN, 3))
                un = u.reshape(CN, 3) + (lev["color"] == c)[:, None] * du
                u = un.reshape(-1) * lev["free"]
        return u

    def _coarse_solve_cg(self, b, lev, Dinv, coarse):
        """Matrix-free (block-Jacobi preconditioned) Krylov solve of the coarsest
        level — no cuDSS. Operator = element matvec + identity on fixed DOFs
        (matches the cuDSS coarse matrix); preconditioner = the 3x3 node-block
        inverse ``Dinv``. ``coarse`` = the config from :meth:`_parse_coarse`."""
        fc = lev["free"]
        CN = lev["color"].shape[0]
        A = lambda x: self._mv_j(x, lev) + (1.0 - fc) * x
        M = lambda r: jnp.einsum("nab,nb->na", Dinv, r.reshape(CN, 3)).reshape(-1)
        kw = dict(tol=coarse["tol"], atol=0.0, maxiter=coarse["maxit"], M=M)
        if coarse["name"] == "bicgstab":
            x, _ = jax.scipy.sparse.linalg.bicgstab(A, b * fc, **kw)
        elif coarse["name"] == "gmres":
            x, _ = jax.scipy.sparse.linalg.gmres(A, b * fc, **kw)
        else:
            x, _ = jax.scipy.sparse.linalg.cg(A, b * fc, **kw)
        return x * fc

    def _vcycle_j(self, b, P, Dinvs, token, coarse, l):
        lev = P[l]
        if l == self.L - 1:
            fc = lev["free"]
            if coarse["cudss"]:
                return _solve_with(token, b * fc) * fc
            return self._coarse_solve_cg(b, lev, Dinvs[l], coarse)
        u = jnp.zeros_like(b)
        u = self._smooth_j(u, b, lev, Dinvs[l], self.pre)
        r = (b - self._mv_j(u, lev)) * lev["free"]
        clev = P[l + 1]; CNc = clev["color"].shape[0]
        rn = r.reshape(lev["color"].shape[0], 3)
        rc = jnp.zeros((CNc, 3)).at[lev["Pidx"]].add(lev["Pw"][:, :, None] * rn[:, None, :])
        rc = rc.reshape(-1) * clev["free"]
        ec = self._vcycle_j(rc, P, Dinvs, token, coarse, l + 1)
        ecn = ec.reshape(CNc, 3)
        uf = (lev["Pw"][:, :, None] * ecn[lev["Pidx"]]).sum(axis=1)
        u = u + uf.reshape(-1) * lev["free"]
        return self._smooth_j(u, b, lev, Dinvs[l], self.post)

    def _solve_j(self, P, b, u0, token, coarse):
        Dinvs = [self._dinv_j(P[l]) for l in range(self.L - 1)]
        # coarsest needs a block-Jacobi preconditioner only for the iterative path
        Dinvs += [None if coarse["cudss"] else self._dinv_j(P[-1])]
        f0 = P[0]["free"]; b = b * f0; u = u0 * f0
        A = lambda x: self._mv_j(x, P[0])
        r = b - A(u); r0 = jnp.max(jnp.abs(r)) + 1e-30; tol = self.cg_tol * r0
        z = self._vcycle_j(r, P, Dinvs, token, coarse, 0); rz = jnp.vdot(r, z)

        def cond(st):
            u, r, p, rz, k = st
            return (jnp.max(jnp.abs(r)) > tol) & (k < self.cg_maxit)

        def body(st):
            u, r, p, rz, k = st
            Ap = A(p); alpha = rz / (jnp.vdot(p, Ap) + 1e-30)
            u = u + alpha * p; r = r - alpha * Ap
            z = self._vcycle_j(r, P, Dinvs, token, coarse, 0); rz_new = jnp.vdot(r, z)
            p = z + (rz_new / (rz + 1e-30)) * p
            return (u, r, p, rz_new, k + 1)

        u, r, p, rz, k = jax.lax.while_loop(cond, body, (u, r, z, rz, 0))
        return u, k, jnp.max(jnp.abs(r)) / r0

    # === convenience API: loads, sensitivity, combined solve ================
    def compact_nodes(self, levels, node_ids):
        """Global grid node ids -> compact band node indices (level 0)."""
        return onp.searchsorted(levels[0]["nodes"], onp.asarray(node_ids, onp.int64))

    def load_vector(self, levels, node_ids, comp=2, value=-1.0):
        """Nodal-load RHS on the compact band DOFs: set component ``comp`` (0/1/2)
        to ``value`` at every global node in ``node_ids`` (must be in the band)."""
        b = onp.zeros(levels[0]["ndof"])
        cn = self.compact_nodes(levels, node_ids)
        b[3 * cn + comp] = value
        return b

    def compliance_and_dc(self, levels, rho_cells, u):
        """Analytic self-adjoint compliance and SIMP sensitivity on the band cells:
        c = Σ E_e uₑᵀKₑuₑ ,  dc = -p ρ^(p-1)(E0-Emin) uₑᵀKₑuₑ.
        ``rho_cells``: bare array or TracedParams (volume_vars[0])."""
        rho = onp.asarray(_design_of(rho_cells))
        E = self.Emin + rho ** self.penal * (self.E0 - self.Emin)
        ue = onp.asarray(u)[levels[0]["edof_c"]]
        ce = onp.einsum("ei,ij,ej->e", ue, self.KE, ue)
        c = float((E * ce).sum())
        dc = -self.penal * rho ** (self.penal - 1) * (self.E0 - self.Emin) * ce
        return c, dc

    # === the solver: create_solver(levels, b, solver_options) ================
    # (implicit diff via custom_vjp)
    # Structure (edof/free/color/transfers/coarsen maps/coarse CSR pattern) is
    # rho-independent -> built once, concrete. Only the VALUES (per-level E, the
    # coarsest CSR data + its factorization) are JAX functions of rho, so the
    # forward MGPCG is jittable and accepts tracers. We do NOT autodiff through
    # the MGPCG / coarse factorization: u solves K(rho)u=b, so the adjoint is
    # another solve K λ = ∂J/∂u (K symmetric) and the design gradient is
    # -λ^T (∂K/∂rho) u, obtained by jax.vjp of a pure-JAX level-0 matvec.
    def create_solver(self, levels, b, solver_options=None, jit=True,
                      return_solution=True):
        """Build the cmg solver for this band + load ``b`` (compact band DOF
        layout), matching feax's ``create_solver`` convention: returns a single
        callable ``solver(rho_cells) -> u`` that is differentiable (implicit-diff
        ``custom_vjp``). Compliance sensitivity for the analytic/OC path is
        ``compliance_and_dc(levels, rho, solver(rho))``.

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
        ``(num_band_nodes, 3)``) by default; ``return_solution=False`` gives
        the raw flat band vector.
        """
        coarse = self._parse_coarse(solver_options)
        L = self.L
        # --- padded, rho-independent per-level structure (no E) ---
        S = []
        for l, lev in enumerate(levels):
            CN, CE, DOF = (_cap(lev["nnode"], self.bucket),
                           _cap(lev["cells"].size, self.bucket),
                           3 * _cap(lev["nnode"], self.bucket))
            edof = onp.zeros((CE, 24), onp.int32); edof[:lev["cells"].size] = lev["edof_c"]
            free = onp.zeros(DOF); free[:lev["ndof"]] = lev["free"]
            color = onp.zeros(CN, onp.int32); color[:lev["nnode"]] = lev["color"]
            d = dict(edof=jnp.asarray(edof), free=jnp.asarray(free), color=jnp.asarray(color),
                     na=lev["cells"].size, CE=CE)
            if l < L - 1:
                Pi = onp.zeros((CN, 8), onp.int32); Pi[:lev["nnode"]] = lev["P_idx"]
                Pw = onp.zeros((CN, 8)); Pw[:lev["nnode"]] = lev["P_w"]
                d["Pidx"] = jnp.asarray(Pi); d["Pw"] = jnp.asarray(Pw)
            S.append(d)

        # --- coarsen scatter maps (rho fine->coarse averaging), padded ---
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
        levc = levels[-1]; ndc = levc["ndof"]; edc = levc["edof_c"]; nac = edc.shape[0]
        e_i = onp.broadcast_to(onp.arange(nac)[:, None, None], (nac, 24, 24)).ravel()
        a_i = onp.broadcast_to(onp.arange(24)[None, :, None], (nac, 24, 24)).ravel()
        b_i = onp.broadcast_to(onp.arange(24)[None, None, :], (nac, 24, 24)).ravel()
        ri = edc[e_i, a_i]; rj = edc[e_i, b_i]
        freec = levc["free"]
        keep = (freec[ri] > 0) & (freec[rj] > 0)
        e_i, a_i, b_i, ri, rj = (v[keep] for v in (e_i, a_i, b_i, ri, rj))
        fixed = onp.nonzero(freec == 0)[0]
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
                    ent_ke=jnp.asarray(self.KE[a_i, b_i]),
                    ent_slot=jnp.asarray(ent_slot), fix_slot=jnp.asarray(fix_slot))

        # --- level-0 matvec as a pure-JAX function of rho (for the design vjp) ---
        edof0 = jnp.asarray(levels[0]["edof_c"].astype(onp.int32))      # (na0, 24)
        free0 = jnp.asarray(levels[0]["free"]); ndof0 = levels[0]["ndof"]
        KE = jnp.asarray(self.KE); Emin, E0, p = self.Emin, self.E0, self.penal

        def mv0(rho_cells, u):
            E = Emin + rho_cells ** p * (E0 - Emin)
            ue = u[edof0]
            ye = E[:, None] * (ue @ KE)
            out = jnp.zeros(ndof0).at[edof0.reshape(-1)].add(ye.reshape(-1))
            return out * free0

        def es_from_rho(rho_cells):
            rhos = [rho_cells]
            for pos, invcnt, ncoarse in coarsen:
                s = jnp.zeros(ncoarse).at[pos].add(rhos[-1])
                rhos.append(s * invcnt)
            Es = []
            for l in range(L):
                E = Emin + rhos[l] ** p * (E0 - Emin)
                Ep = jnp.zeros(S[l]["CE"]).at[:E.shape[0]].set(E)
                Es.append(Ep)
            return Es

        def coarse_token(Ecoarse):
            E = Ecoarse[:nac]
            data = jnp.zeros(cpat["nnz"]).at[cpat["ent_slot"]].add(E[cpat["ent_e"]] * cpat["ent_ke"])
            data = data.at[cpat["fix_slot"]].add(1.0)
            return _factorize(data, cpat["indptr"], cpat["indices"], mtype_id=3, mview_id=0)

        DOF0 = S[0]["free"].shape[0]
        bp = jnp.zeros(DOF0).at[:levels[0]["ndof"]].set(jnp.asarray(b))

        # Cache the jitted MGPCG on the instance, keyed by the coarse config. The
        # structure ``P`` is passed as arguments (not baked into the closure), so
        # the compiled executable is reused across bands whose padded shapes match
        # (bucketed). es_from_rho / coarse_token stay eager (cheap).
        if jit:
            if getattr(self, "_jsj_key", None) != coarse:
                _c = dict(coarse)
                self._jsj = jax.jit(lambda P, rhs_, u0_, tok_:
                                    self._solve_j(P, rhs_, u0_, tok_, _c))
                self._jsj_key = dict(coarse)
            _jsj = self._jsj
        else:                                               # eager: no compile, no shape reuse
            _c = dict(coarse)
            _jsj = lambda P, rhs_, u0_, tok_: self._solve_j(P, rhs_, u0_, tok_, _c)

        def raw_solve(rho_cells, rhs):
            Es = es_from_rho(rho_cells)
            token = coarse_token(Es[-1]) if coarse["cudss"] else None
            P = [{**S[l], "E": Es[l]} for l in range(L)]
            u, _, _ = _jsj(P, rhs, jnp.zeros(DOF0), token)
            return u[:ndof0]

        @jax.custom_vjp
        def solve(rho_cells):
            return raw_solve(rho_cells, bp)

        def solve_fwd(rho_cells):
            u = raw_solve(rho_cells, bp)
            return u, (rho_cells, u)

        def solve_bwd(res, u_bar):
            rho_cells, u = res
            rhs = jnp.zeros(DOF0).at[:ndof0].set(u_bar)
            lam = raw_solve(rho_cells, rhs)                 # adjoint (K symmetric)
            _, vjp = jax.vjp(lambda r: mv0(r, u), rho_cells)
            (g,) = vjp(lam)
            return (-g,)

        solve.defvjp(solve_fwd, solve_bwd)

        _layout = ((levels[0]["nnode"], 3),)

        def solver(rho_cells):
            """Accepts the band design as a bare ``(n_active,)`` array or as a
            ``TracedParams`` (``volume_vars[0]``), matching the feax solver
            convention ``solver(traced_params) -> u``. Differentiable in both
            forms (the pytree extraction is transparent to ``jax.grad``)."""
            u = solve(_design_of(rho_cells))
            if return_solution:
                from ..solution import Solution
                return Solution(u, _layout)
            return u

        return solver
