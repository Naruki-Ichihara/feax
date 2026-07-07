---
sidebar_label: narrowband
title: feax.gene.narrowband
---

Narrow-band topology optimization on a StructuredGrid (cmg + OC).

The moving-band SIMP compliance loop for :class:`feax.NarrowBandCMG`: the
active band is re-extracted from the design every iteration, the solve is
O(band), and the coarse-to-fine multires bootstrap keeps even the exploration
phase off the full grid — the giga-voxel scaling path.

The density update is Optimality Criteria (:func:`oc_update`), deliberately:
the band&#x27;s design vector changes size every iteration and the full-grid
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

#### oc\_update

```python
def oc_update(rho,
              dc,
              volfrac,
              *,
              dv=1.0,
              move=0.2,
              xmin=1e-3,
              xmax=1.0,
              damping=0.5,
              bisect_tol=1e-4,
              l2_init=1e9)
```

Optimality-Criteria density update for a single volume constraint.

The classic Bendsøe multiplicative update used for compliance minimisation.
Given the (filtered) objective sensitivity ``dc`` (dc&lt;0 for compliance), it
moves density toward the optimality condition and bisects the Lagrange
multiplier λ so the volume constraint ``Σ ρ·dv = volfrac·Σ dv`` is met::

    ρ_new = clip( ρ · max(0, -dc/(dv·λ))^damping , [ρ-move, ρ+move] ∩ [xmin,xmax] )

General and mesh-agnostic: ``rho``/``dc`` are any 1-D arrays (node- or
cell-based, narrow-band or full mesh). ``dv`` is per-element volume weight
(scalar 1.0 = equal cells; pass cell volumes for unstructured meshes). Returns
the updated density (numpy array). Pin passive regions by re-setting them after.

#### run\_narrowband\_oc

```python
def run_narrowband_oc(grid,
                      cmg,
                      keep,
                      volfrac,
                      load_fn,
                      *,
                      rmin=1.5,
                      n_iter=40,
                      grid_update_start=1,
                      x_init=None,
                      move=0.2,
                      xmin=1e-3,
                      penal=None,
                      threshold=1e-2,
                      margin=2,
                      solver_options=None,
                      jit_after=None,
                      sensitivity_filter=None,
                      verbose=True,
                      label="",
                      it_offset=0)
```

One-resolution moving narrow-band SIMP compliance topopt via cmg + OC.

The active band is re-extracted every iteration from the design
(``dilate(rho&gt;threshold, margin) | keep``), so the cmg solve is O(band) and
shrinks as the structure localises. Uses the self-adjoint compliance
sensitivity (ONE forward solve, no adjoint) via
``cmg.compliance_and_dc``, the narrow-band sensitivity filter, and
:func:`oc_update`. This is the reusable core of the moving-band loop (see
:func:`run_narrowband_multires` for the coarse-grid bootstrap that chains it
across resolutions).

Parameters
----------
- **grid** (*StructuredGrid*)
- **cmg** (*NarrowBandCMG          already constructed for ``grid``.*)
- **keep** (*bool ndarray (nx,ny,nz)   always-active cells (near loads/supports).*)
- **volfrac** (*float              volume target.*)
- **load_fn** (*callable           ``load_fn(cmg, levels) -&gt; b`` (compact band DOFs),*): e.g. ``lambda c, lv: c.load_vector(lv, [n], 2, -1.)``.
- **x_init** (*flat ndarray, optional   initial density (num_cells,). Default uniform.*)
- **grid_update_start** (*int      iters of full-grid exploration before banding*): (0 = band from the first iter, for a seeded level).
- **solver_options** (*feax solver_options   coarsest-level solver (None -&gt; cuDSS).*)
- **jit_after** (*int or None      iteration index from which cmg switches to*): ``jit=True`` (eager before). The band typically                          stops moving by then, so the jitted executable is                          reused across the tail (pair with ``bucket&gt;1`` to                          absorb small size flicker). ``None`` (default)                          stays eager the whole level — safest for a band                          that keeps moving (no per-iter recompile).


#### run\_narrowband\_multires

```python
def run_narrowband_multires(build_fn,
                            dims,
                            volfrac,
                            *,
                            n_levels=2,
                            coarse_factor=2,
                            rmin=1.5,
                            n_iter=40,
                            coarse_iter=None,
                            grid_update_start=1,
                            floor=8,
                            verbose=True,
                            **oc_kw)
```

Coarse-grid bootstrap for the narrow-band cmg path (BandJax&#x27;s multires).

The full-grid EXPLORATION phase (whole domain active) is what sets the memory
peak at high resolution. Here it runs only on a COARSE grid (cheap); each
finer level is SEEDED by upsampling the previous density and runs with
``grid_update_start=0`` so it NEVER solves the full grid — its peak is the
band, not the domain. This moves the resolution ceiling from O(domain) to
O(band), the key to giga-voxel scaling.

Parameters
----------
- **build_fn** (*callable*): ``build_fn(nx, ny, nz) -&gt; dict(grid=, cmg=, keep=, load_fn=)`` — builds a :class:`StructuredGrid`, its :class:`NarrowBandCMG`, the always-active ``keep`` mask, and ``load_fn(cmg, levels) -&gt; b`` at ANY resolution. Loads and supports must be placed by PHYSICAL location so every level is the same problem discretised differently.
- **dims** (*(nx, ny, nz)          finest resolution.*)
- **n_levels** (*int               number of resolutions (2 = one coarse + fine).*)
- **coarse_iter** (*int, optional  iters per non-final level (default max(10, n_iter//2)).*)

