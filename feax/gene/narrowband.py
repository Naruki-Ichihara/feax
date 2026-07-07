"""Narrow-band topology optimization on a StructuredGrid (cmg + OC).

The moving-band SIMP compliance loop for :class:`feax.NarrowBandCMG`: the
active band is re-extracted from the design every iteration, the solve is
O(band), and the coarse-to-fine multires bootstrap keeps even the exploration
phase off the full grid — the giga-voxel scaling path.

The density update is Optimality Criteria (:func:`oc_update`), deliberately:
the band's design vector changes size every iteration and the full-grid
density can reach 1e9 cells, which rules out NLopt MMA (fixed dimension,
host-memory heavy). OC is a pure-numpy O(n) bisection and pairs with the
self-adjoint compliance sensitivity (one forward solve, no adjoint). For
general objectives/constraints on a fixed mesh use the MMA driver
(:func:`feax.gene.optimizer.run`).

Usage (see ``examples/advance/cantilever3d.py``)::

    def build_fn(nx, ny, nz):
        grid = fe.StructuredGrid((nx, ny, nz))
        cmg = fe.NarrowBandCMG(grid, fixed_pred, ...)
        keep = ...                                  # bool (nx,ny,nz)
        load_fn = lambda c, lv: c.load_vector(lv, tip_nodes(grid), 2, -1.0)
        return dict(grid=grid, cmg=cmg, keep=keep, load_fn=load_fn)

    res = run_narrowband_multires(build_fn, (256, 128, 128), volfrac=0.2)
"""

import numpy as onp


# ---------------------------------------------------------------------------
# Optimality Criteria (OC) update — general, mesh-agnostic
# ---------------------------------------------------------------------------

def oc_update(rho, dc, volfrac, *, dv=1.0, move=0.2, xmin=1e-3, xmax=1.0,
              damping=0.5, bisect_tol=1e-4, l2_init=1e9):
    """Optimality-Criteria density update for a single volume constraint.

    The classic Bendsøe multiplicative update used for compliance minimisation.
    Given the (filtered) objective sensitivity ``dc`` (dc<0 for compliance), it
    moves density toward the optimality condition and bisects the Lagrange
    multiplier λ so the volume constraint ``Σ ρ·dv = volfrac·Σ dv`` is met::

        ρ_new = clip( ρ · max(0, -dc/(dv·λ))^damping , [ρ-move, ρ+move] ∩ [xmin,xmax] )

    General and mesh-agnostic: ``rho``/``dc`` are any 1-D arrays (node- or
    cell-based, narrow-band or full mesh). ``dv`` is per-element volume weight
    (scalar 1.0 = equal cells; pass cell volumes for unstructured meshes). Returns
    the updated density (numpy array). Pin passive regions by re-setting them after.
    """
    rho = onp.asarray(rho, float)
    dc = onp.asarray(dc, float)
    dv = onp.asarray(dv, float) * onp.ones_like(rho)
    target = volfrac * dv.sum()
    l1, l2 = 0.0, float(l2_init)
    xnew = rho
    while (l2 - l1) / (l1 + l2 + 1e-30) > bisect_tol:
        lmid = 0.5 * (l1 + l2)
        be = onp.maximum(0.0, -dc / (dv * lmid + 1e-30))
        xnew = onp.clip(rho * be ** damping, rho - move, rho + move)
        xnew = onp.clip(xnew, xmin, xmax)
        if float((xnew * dv).sum()) > target:
            l1 = lmid
        else:
            l2 = lmid
    return xnew


# ---------------------------------------------------------------------------
# Moving narrow-band SIMP loop (one resolution)
# ---------------------------------------------------------------------------

def run_narrowband_oc(grid, cmg, keep, volfrac, load_fn, *, rmin=1.5,
                      n_iter=40, grid_update_start=1, x_init=None, move=0.2,
                      xmin=1e-3, penal=None, threshold=1e-2, margin=2,
                      solver_options=None, jit_after=None,
                      sensitivity_filter=None, verbose=True, label="",
                      it_offset=0):
    """One-resolution moving narrow-band SIMP compliance topopt via cmg + OC.

    The active band is re-extracted every iteration from the design
    (``dilate(rho>threshold, margin) | keep``), so the cmg solve is O(band) and
    shrinks as the structure localises. Uses the self-adjoint compliance
    sensitivity (ONE forward solve, no adjoint) via
    ``cmg.compliance_and_dc``, the narrow-band sensitivity filter, and
    :func:`oc_update`. This is the reusable core of the moving-band loop (see
    :func:`run_narrowband_multires` for the coarse-grid bootstrap that chains it
    across resolutions).

    Parameters
    ----------
    grid : StructuredGrid
    cmg : NarrowBandCMG          already constructed for ``grid``.
    keep : bool ndarray (nx,ny,nz)   always-active cells (near loads/supports).
    volfrac : float              volume target.
    load_fn : callable           ``load_fn(cmg, levels) -> b`` (compact band DOFs),
                                 e.g. ``lambda c, lv: c.load_vector(lv, [n], 2, -1.)``.
    x_init : flat ndarray, optional   initial density (num_cells,). Default uniform.
    grid_update_start : int      iters of full-grid exploration before banding
                                 (0 = band from the first iter, for a seeded level).
    solver_options : feax solver_options   coarsest-level solver (None -> cuDSS).
    jit_after : int or None      iteration index from which cmg switches to
                                 ``jit=True`` (eager before). The band typically
                                 stops moving by then, so the jitted executable is
                                 reused across the tail (pair with ``bucket>1`` to
                                 absorb small size flicker). ``None`` (default)
                                 stays eager the whole level — safest for a band
                                 that keeps moving (no per-iter recompile).

    Returns ``(rho_flat, history)`` — ``rho_flat`` shape ``(num_cells,)``.
    """
    from scipy.ndimage import binary_dilation
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    NC = grid.num_cells
    keep = onp.asarray(keep, bool)
    if sensitivity_filter is None:
        from feax.gene.filters import create_sensitivity_filter
        sensitivity_filter = create_sensitivity_filter(grid, rmin)

    rho = (onp.full(NC, volfrac) if x_init is None
           else onp.asarray(x_init, float).reshape(-1).copy())
    active_cells = levels = solver = None
    prev_active, jit_on = None, False
    history = []
    for it in range(n_iter):
        if it < grid_update_start:
            want = onp.ones((nx, ny, nz), bool)
        else:
            want = binary_dilation(rho.reshape(nx, ny, nz) > threshold,
                                   iterations=margin) | keep
        active_cells = onp.nonzero(want.reshape(-1))[0].astype(onp.int64)

        # Primitive schedule: eager until iter ``jit_after`` (band still moving ->
        # no recompile), jit from then on (band settled -> reuse the executable).
        use_jit = jit_after is not None and it >= jit_after
        changed = (prev_active is None or active_cells.size != prev_active.size
                   or not onp.array_equal(active_cells, prev_active))
        if changed or (use_jit and not jit_on):
            levels = cmg.build(active_cells)
            b = load_fn(cmg, levels)
            solver = cmg.create_solver(levels, b, solver_options=solver_options,
                                       jit=use_jit, return_solution=False)
            jit_on = use_jit
        prev_active = active_cells

        rho_a = rho[active_cells]
        u = onp.asarray(solver(rho_a))
        compliance, dc_a = cmg.compliance_and_dc(levels, rho_a, u)
        dc_f_a = sensitivity_filter(active_cells, rho_a, dc_a)
        dc_f = onp.zeros(NC); dc_f[active_cells] = dc_f_a
        rho = oc_update(rho, dc_f, volfrac, move=move, xmin=xmin)

        af = active_cells.size / NC
        history.append((it, float(compliance), af, float(rho.mean())))
        if verbose:
            print(f"  {label}it {it_offset + it:3d} {'J' if jit_on else 'e'} "
                  f"c={compliance:11.4e}  active={af*100:5.1f}%  "
                  f"vol={rho.mean():.3f}")
    return rho, history


# ---------------------------------------------------------------------------
# Coarse-to-fine multires bootstrap
# ---------------------------------------------------------------------------

def _res_ladder(dims, n_levels, factor, floor=8):
    """Resolution ladder coarse->fine: [dims/factor^(L-1), ..., dims/factor, dims].

    Rounds each coarse dim to at least ``floor`` and stops early if a level can
    get no coarser (so any dims/n_levels combo is safe).
    """
    ladder = [tuple(int(d) for d in dims)]
    for _ in range(n_levels - 1):
        coarse = tuple(max(floor, int(round(d / factor))) for d in ladder[0])
        if coarse == ladder[0]:
            break
        ladder.insert(0, coarse)
    return ladder


def _upsample(rho_grid, target_shape):
    """Trilinear/bilinear upsample a density grid to ``target_shape`` (any ratio,
    incl. odd), clipped to [0, 1]. Exact shape enforced (crop / edge-pad)."""
    from scipy.ndimage import zoom
    fac = [t / s for t, s in zip(target_shape, rho_grid.shape)]
    out = zoom(rho_grid, fac, order=1, mode="nearest")
    sl = tuple(slice(0, t) for t in target_shape)
    out = out[sl]
    pad = [(0, t - o) for t, o in zip(target_shape, out.shape)]
    if any(p[1] > 0 for p in pad):
        out = onp.pad(out, pad, mode="edge")
    return onp.clip(out, 0.0, 1.0)


def run_narrowband_multires(build_fn, dims, volfrac, *, n_levels=2,
                            coarse_factor=2, rmin=1.5, n_iter=40,
                            coarse_iter=None, grid_update_start=1, floor=8,
                            verbose=True, **oc_kw):
    """Coarse-grid bootstrap for the narrow-band cmg path (BandJax's multires).

    The full-grid EXPLORATION phase (whole domain active) is what sets the memory
    peak at high resolution. Here it runs only on a COARSE grid (cheap); each
    finer level is SEEDED by upsampling the previous density and runs with
    ``grid_update_start=0`` so it NEVER solves the full grid — its peak is the
    band, not the domain. This moves the resolution ceiling from O(domain) to
    O(band), the key to giga-voxel scaling.

    Parameters
    ----------
    build_fn : callable
        ``build_fn(nx, ny, nz) -> dict(grid=, cmg=, keep=, load_fn=)`` — builds a
        :class:`StructuredGrid`, its :class:`NarrowBandCMG`, the always-active
        ``keep`` mask, and ``load_fn(cmg, levels) -> b`` at ANY resolution. Loads
        and supports must be placed by PHYSICAL location so every level is the
        same problem discretised differently.
    dims : (nx, ny, nz)          finest resolution.
    n_levels : int               number of resolutions (2 = one coarse + fine).
    coarse_iter : int, optional  iters per non-final level (default max(10, n_iter//2)).
    **oc_kw                      forwarded to :func:`run_narrowband_oc`
                                 (move, xmin, threshold, margin, solver_options,
                                 stable_after, ...).

    Returns ``{"x": rho_grid, "levels": ladder, "history": [per-level ...]}``
    where ``rho_grid`` is the finest density reshaped to ``dims``.
    """
    ladder = _res_ladder(dims, n_levels, coarse_factor, floor)
    if coarse_iter is None:
        coarse_iter = max(10, n_iter // 2)

    rho_grid = None
    per_level = []
    it_offset = 0                       # continuous iteration counter across levels
    for lvl, d in enumerate(ladder):
        finest = (lvl == len(ladder) - 1)
        parts = build_fn(*d)
        grid, cmg, keep, load_fn = (parts["grid"], parts["cmg"],
                                    parts["keep"], parts["load_fn"])
        if lvl == 0:
            x_init, gus, niter, tag = None, grid_update_start, coarse_iter, "explore"
        else:
            x_init = _upsample(rho_grid, d).reshape(-1)   # seed band from coarser design
            gus = 0                                        # no full-grid solve here
            niter = n_iter if finest else coarse_iter
            tag = "refine"
        if verbose:
            ndof = 3 * (d[0] + 1) * (d[1] + 1) * (d[2] + 1)
            print(f"[multires] L{lvl} {d}  DOF={ndof:,}  {tag}  iters={niter}  "
                  f"gus={gus}")
        rho_flat, hist = run_narrowband_oc(
            grid, cmg, keep, volfrac, load_fn, rmin=rmin, n_iter=niter,
            grid_update_start=gus, x_init=x_init, verbose=verbose,
            label=f"L{lvl} ", it_offset=it_offset, **oc_kw)
        rho_grid = rho_flat.reshape(d)
        per_level.append({"level": lvl, "dims": d, "history": hist})
        it_offset += niter
    return {"x": rho_grid, "levels": ladder, "history": per_level}
