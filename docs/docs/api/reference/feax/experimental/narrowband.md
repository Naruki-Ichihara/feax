---
sidebar_label: narrowband
title: feax.experimental.narrowband
---

Narrow-band topology optimization on a StructuredGrid (cmg + OC) — experimental.

The moving-band SIMP compliance loop for
:class:`feax.experimental.NarrowBandCMG`: the active band is re-extracted from
the design every iteration, the solve is O(band), and the coarse-to-fine
multires bootstrap keeps even the exploration phase off the full grid — the
giga-voxel scaling path.

The density update is Optimality Criteria (:func:`oc_update`): a pure-numpy
O(n) bisection that pairs with the self-adjoint compliance sensitivity (one
forward solve, no adjoint) and needs no state across the moving band. It is
valid for compliance minimisation under a single volume constraint only.
:func:`run_narrowband_mma` is the same moving-band loop driven by the pure-JAX
MMA (:mod:`feax.experimental.mma`) with the optimizer state (design, previous
iterates, asymptotes) kept on the FULL grid, so the band may change size freely;
select it in :func:`run_narrowband_multires` with ``update=&quot;mma&quot;``. For general
objectives/constraints on a fixed mesh use :func:`feax.gene.optimizer.run`.

Usage (see ``examples/advance/high_reso_topopt.py``)::

    from feax.experimental import NarrowBandCMG, run_narrowband_multires

    def build_fn(nx, ny, nz):
        grid = fe.StructuredGrid((nx, ny, nz))
        cmg = NarrowBandCMG(grid, fixed_pred, ...)
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
- **cmg** (*NarrowBandCMG          already constructed for ``grid`` (any*): :class:`~feax.experimental.cmg.Material` whose                          design is a single per-cell density array).
- **keep** (*bool ndarray (nx,ny,nz)   always-active cells (near loads/supports).*)
- **volfrac** (*float              volume target.*)
- **load_fn** (*callable           ``load_fn(cmg, levels) -&gt; b`` (compact band DOFs),*): e.g. ``lambda c, lv: c.load_vector(lv, [n], 2, -1.)``.
- **x_init** (*flat ndarray, optional   initial density (num_cells,). Default uniform.*)
- **grid_update_start** (*int      iters of full-grid exploration before banding*): (0 = band from the first iter, for a seeded level).
- **solver_options** (*feax solver_options   coarsest-level solver (None -&gt; cuDSS).*)
- **jit_after** (*int or None      iteration index from which cmg switches to*): ``jit=True`` (eager before). The band typically                          stops moving by then, so the jitted executable is                          reused across the tail (pair with ``bucket&gt;1`` to                          absorb small size flicker). ``None`` (default)                          stays eager the whole level — safest for a band                          that keeps moving (no per-iter recompile).


#### run\_narrowband\_mma

```python
def run_narrowband_mma(grid,
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
                       it_offset=0,
                       mma_options=None)
```

Moving narrow-band SIMP compliance topopt via cmg + MMA.

Same band extraction, solve and sensitivity as :func:`run_narrowband_oc`
(``dilate(rho&gt;threshold, margin) | keep`` every iteration, one self-adjoint
forward solve, the narrow-band sensitivity filter), but the density update
is Svanberg&#x27;s MMA (:class:`feax.experimental.mma.MMA`, plain variant).

The optimizer state lives on the FULL grid: the design ``x``, the two
previous iterates and the asymptotes are ``(num_cells,)`` arrays, so the
band can change size every iteration without any state remapping. Cells
outside the band get a zero objective gradient and a positive volume
gradient, so MMA drives them to ``xmin`` — the same behaviour as OC. The
cost is a few full-grid vectors (about ``12 * 8 * num_cells`` bytes in
float64; pass ``mma_options=dict(dtype=jnp.float32)`` to halve it) and
O(num_cells) elementwise work per iteration, both on the JAX device; the
solve stays O(band).

Objective and constraint are scaled for MMA&#x27;s elastic penalty: ``f0 =
c / c_first``, ``g = mean(x) / volfrac - 1``. ``n_iter`` counts solves, as
in :func:`run_narrowband_oc` (``n_iter - 1`` MMA updates). ``move`` is the
MMA move limit as a fraction of ``xmax - xmin``. Extra keyword arguments for
:class:``7 (``asyinit``, ``c``, ``dtype``, ...) go
in ``mma_options``.

Returns ``(rho_flat, history)`` with the same layout as
:func:`run_narrowband_oc`.

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
                            update="oc",
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
- **build_fn** (*callable*): ``build_fn(nx, ny, nz) -&gt; dict(grid=, cmg=, keep=, load_fn=)`` — builds a :class:`StructuredGrid`, its :class:`~feax.experimental.NarrowBandCMG`, the always-active ``keep`` mask, and ``load_fn(cmg, levels) -&gt; b`` at ANY resolution. Loads and supports must be placed by PHYSICAL location so every level is the same problem discretised differently.
- **dims** (*(nx, ny, nz)          finest resolution.*)
- **n_levels** (*int               number of resolutions (2 = one coarse + fine).*)
- **coarse_iter** (*int, optional  iters per non-final level (default max(10, n_iter//2)).*)
- **update** (*&quot;oc&quot; | &quot;mma&quot;        density update per level: :func:``0*): or :func:``1 (MMA state is                          re-initialised at every level; only the density                          is carried across resolutions).


#### create\_sensitivity\_filter

```python
def create_sensitivity_filter(grid, rmin: float)
```

Narrow-band-native sensitivity filter (88-line ``ft=1``) for a StructuredGrid.

Unlike the Helmholtz / density filters above (node-based, mesh-based, and not
designed for a cell-based moving band), this one is CELL-based and O(band): for
each active cell it averages the sensitivity over its in-radius ACTIVE
neighbours using cone weights, addressing neighbours by grid arithmetic +
``searchsorted`` into the active set — no full-grid array, so it follows the
moving band. Intended for the analytic-sensitivity + OC workflow (it modifies
the sensitivity heuristically; it is not a differentiable design transform —
use Helmholtz/density for ``jax.grad`` design filtering).

Returns ``filt(active_cells, rho_active, dc_active) -&gt; dc_filtered`` (all arrays
over the active band cells), computing

    dc_filt_e = Σ_n w_en ρ_n dc_n / ( Hs_e · max(1e-3, ρ_e) )

with Hs_e summed over all IN-GRID neighbours (active or void).

