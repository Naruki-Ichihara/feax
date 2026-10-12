# Narrow-Band & Giga-Voxel Topology Optimization

This tutorial demonstrates narrow-band topology optimization on implicit structured grids using FEAX's `StructuredGrid`, `NarrowBand`, and the matrix-free geometric multigrid solver `NarrowBandCMG` (in `feax.experimental`: usable, but its interface may still change). By solving only where material exists and never materializing the background domain, the memory peak scales with the **band** instead of the **domain** — the path to giga-voxel resolutions.

## Overview

Dense 3D topology optimization hits a memory wall long before it hits a resolution wall: the design array, the mesh, and the linear system all scale with the full domain, yet the converged structure typically occupies only a few percent of it. FEAX attacks both sides of this:

1. **Implicit domain** — `StructuredGrid` is a uniform HEX8 grid whose connectivity and coordinates are index arithmetic. The full grid is never stored, so a $10^9$-cell background domain costs O(1) memory.
2. **Sparse design** — `SparseDesign` stores per-cell values only on the cells that carry them (12 B/cell), keyed by global cell id.
3. **Narrow-band solve** — `NarrowBand` extracts the sub-mesh spanned by the active cells; the linear system has DOFs only for the nodes the band touches. If the excluded cells are void (stiffness → 0), the band solve reproduces the full solve.
4. **O(band) solver** — `NarrowBandCMG` is a matrix-free geometric multigrid PCG (MGPCG) that operates directly on the structured band, differentiable via implicit-diff `custom_vjp`.
5. **Moving-band OC loop + multires bootstrap** — `feax.gene` re-extracts the band from the design every iteration and chains resolutions coarse-to-fine, so even the exploration phase never solves the full grid.

## Structured Grids and Sparse Designs

### StructuredGrid

`StructuredGrid` is FEAX's second domain representation, next to the explicit `Mesh`. Connectivity and coordinates are computed on the fly:

```python
import numpy as onp
import feax as fe

# 256^3 = 16.8M cells — the grid object itself costs O(1) memory
grid = fe.StructuredGrid((256, 256, 256), spacing=(1.0, 1.0, 1.0),
                         origin=(0.0, 0.0, 0.0))

grid.num_cells                      # 16_777_216
grid.cell_to_nodes([0, 1, 2])       # (3, 8) HEX8 connectivity, index arithmetic
grid.cell_centroids([0, 1, 2])      # (3, 3) coordinates, no stored arrays
```

The node ordering of `cell_to_nodes` matches FEAX's `box_mesh` HEX8 convention, so any extracted band is a valid FEAX `Mesh`.

Any mesh-defined geometry embeds into an enclosing grid: `StructuredGrid.fit` places a grid over the bounding box of arbitrary points, and `voxelize_mesh` marks the cells an unstructured mesh occupies:

```python
# Background grid over an arbitrary geometry's bounding box
grid = fe.StructuredGrid.fit(mesh.points, h=0.5, pad_cells=2, align=8)

# Active cells = grid cells the source mesh occupies (O(points), never O(grid))
active = fe.voxelize_mesh(grid, mesh, subsamples=2)
```

**Key `fit` arguments:**
- **`h`** — target cell size (dims derived, `pad_cells` empty margin cells added per side), *or* **`dims`** — cell counts (spacing derived)
- **`align=8`** — rounds each dim up to a multiple of 8 so geometric multigrid can coarsen; the extra cells are empty

Active sets can also come from a predicate on centroids (an SDF, for example) or from grid-index queries:

```python
active = grid.cells_where(lambda c: sdf(c) < 0.0)   # O(num_cells) — small grids only
top_nodes = grid.nodes_where(lambda I, J, K: K == grid.nz)   # nodes by grid index
```

:::note
`voxelize` / `voxelize_mesh` are O(sample points) and never touch the full grid — prefer them over `cells_where` when the grid is large and the active set is small.
:::

### SparseDesign

`SparseDesign` is the companion storage for extreme resolution — per-cell design values stored only where they exist:

```python
design = fe.SparseDesign.uniform(active, 0.3)        # 12 B per stored cell

rho_band = design.gather(active, default=0.0)        # values on any query id set
design = design.updated(active, rho_new)             # write band values back (new store)

# Active band = dilate({value > threshold}, margin) ∪ keep
active = design.band_cells(grid, 1e-2, margin=2, keep_ids=keep_cells)

# Bridge to the container every FEAX solver accepts
tp = design.traced_params(active)                    # TracedParams(volume_vars=(rho,))
```

`traced_params` / `updated` are inverses: gather the stored design onto the current band for the solve, then write the optimizer's update back into the store.

## Narrow-Band Solves with a Standard FEAX Solver

`NarrowBand` extracts the explicit sub-mesh spanned by a set of active cells. The `domain` argument is **either** an unstructured `Mesh` (any element type) **or** an implicit `StructuredGrid` — both yield the same kind of band:

```python
import jax.numpy as np

active = onp.nonzero(rho > 1e-2)[0]           # any user policy
band = fe.NarrowBand(mesh, active)            # or fe.NarrowBand(grid, active)

band.mesh              # sub-mesh (renumbered nodes) — build a Problem on this
band.active_cells      # band cell -> global cell
band.node_map          # band node -> global node
```

Everything downstream is stock FEAX. Coordinate-based `location_fns` and Dirichlet BC predicates transfer unchanged to the sub-mesh, since the band nodes keep their physical coordinates:

```python
class Elasticity(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E):
            nu = 0.3
            mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lam * np.trace(eps) * np.eye(3) + 2 * mu * eps
        return stress

problem = Elasticity(band.mesh, vec=3, dim=3, ele_type='HEX8')
bc = fe.DirichletBCConfig([
    fe.DirichletBCSpec(lambda p: np.isclose(p[0], 0.0), "all", 0.0),
    fe.DirichletBCSpec(lambda p: np.isclose(p[0], 4.0), "x", 0.05),
]).create_bc(problem)

tp_band = band.gather_params(tp_full)         # full-domain TracedParams -> band
solver = fe.create_solver(problem, bc, solver_options=fe.DirectSolverOptions(),
                          linear=True, traced_params=tp_band)
u_band = solver(tp_band, fe.zero_like_initial_guess(problem, bc))

u_full = band.scatter_sol(u_band, vec=3)      # back to the full domain (zeros outside)
```

**Key properties:**
- **Band solve = full solve** — when the excluded cells are void (stiffness → 0), the band displacement matches the full-domain solve to O(void stiffness). In general it is the solve on the sub-domain with a free surface on the band boundary.
- **`gather_params` / `scatter_params`** — map `TracedParams` volume vars between full domain and band, dispatching per-cell arrays by `active_cells` and per-node arrays by `node_map`. Pure gathers, so `jax.grad` flows through.
- **`gather_sol` / `scatter_sol`, `gather_cells` / `scatter_cells`** — the same maps for solutions and cell fields.
- `jax.grad` works as usual via FEAX's implicit-diff adjoint — the band solver is an ordinary `create_solver` solver.

:::note Surface variables
`gather_params` only maps `volume_vars`. The band's boundary faces differ from the full mesh's, so `surface_vars` must be rebuilt on the band `Problem`.
:::

### Building the band from a field

For unstructured meshes, `from_field` derives the active set from any scalar field:

```python
band = fe.NarrowBand.from_field(mesh, rho, 1e-2, margin=1,
                                keep_cells=loaded_cells, cell_field=True)
```

- **`margin`** — node-adjacency dilation rings around the seed (`margin=1` is the usual narrow-band safety layer)
- **`keep_cells`** — cells always kept active (near loads/supports, so the band never disconnects the load path)

For a `StructuredGrid`, use `SparseDesign.band_cells` or `StructuredGrid.cells_where` instead.

### Moving bands: SupersetBand

Re-extracting the band changes array shapes, which forces JAX recompilation. `SupersetBand` amortizes this for moving bands: it holds a *superset* (the active band dilated by `margin` rings) fixed while the true band moves inside it, and only re-extracts when the band (dilated by `guard`) reaches the superset boundary:

```python
mgr = fe.SupersetBand(mesh, margin=2, guard=1)
for it in range(n_iter):
    active = onp.nonzero(rho > threshold)[0]        # any policy
    if mgr.needs_reextract(active):
        band = mgr.reextract(active)                # rare recompile point
        solver, run = build_solver(band.mesh)       # user physics/solver
    sol = run(mgr.map_cells(rho))                   # warm solve, same shapes
    ...                                             # update rho
```

Inactive-but-in-superset cells are simply low-stiffness in `traced_params`, so every solve reuses one compiled solver with any `solver_options`.

## Matrix-Free Geometric Multigrid: NarrowBandCMG

For very large structured bands, even an assembled sparse matrix (and certainly a direct factorization with 3D fill) exhausts memory. `NarrowBandCMG` is a matrix-free, O(band) geometric multigrid preconditioned CG for **linear, symmetric positive definite** problems on a uniform HEX8 `StructuredGrid`. The default material is isotropic linear elasticity with per-cell SIMP scaling $E = E_{\min} + \rho^p (E_0 - E_{\min})$:

```python
from feax.experimental import NarrowBandCMG

grid = fe.StructuredGrid((128, 64, 64))
fixed_pred = lambda ni, nj, nk, nx, ny, nz: ni == 0     # clamp the x=0 face (grid indices)

cmg = NarrowBandCMG(grid, fixed_pred, nu=0.3, E0=1.0, Emin=1e-9, penal=3.0,
                    cg_tol=1e-8, cg_maxit=200, bucket=512)
levels = cmg.build(active_cells)                        # O(band) MG hierarchy

load_node = grid.node_id(128, 32, 32)                   # must lie in the band
b = cmg.load_vector(levels, [load_node], comp=2, value=-1.0)

solver = cmg.create_solver(levels, b)                   # None -> cuDSS coarsest level
sol = solver(rho_cells)                                 # (n_active,) array OR TracedParams
u = sol.field(0)                                        # (num_band_nodes, 3)
```

The solver follows FEAX's `create_solver` convention: one callable `solver(rho_cells) -> u`, returning a `fe.Solution` by default (`return_solution=False` gives the raw flat band vector). It accepts the band design as a bare array or as `fe.TracedParams(volume_vars=(rho_cells,))` — e.g. straight from `SparseDesign.traced_params`.

`fixed_pred` may also return a `(n_nodes, vec)` boolean mask to fix individual components (e.g. a roller support).

**Components:** 8-colour block (`vec`×`vec` node) Gauss–Seidel smoother, trilinear prolongation / full-weighting restriction on the compact active set, and a coarsest-level solve selected by standard FEAX `solver_options`:

```python
# cuDSS direct factorization at the coarsest level (default; needs spineax)
solver = cmg.create_solver(levels, b, solver_options=fe.DirectSolverOptions())

# Matrix-free block-Jacobi Krylov coarse solve — no cuDSS dependency
solver = cmg.create_solver(levels, b,
    solver_options=fe.KrylovSolverOptions(solver="cg", tol=1e-8, maxiter=400))
```

### Materials and DOFs per node

What cmg keeps fixed is the uniform HEX8 grid and the SPD operator. Everything else is a `Material`:

- **`ScalarMaterial(KE, scale_fn)`** — the fast path: one reference element matrix times a per-cell scalar, one float of state per cell. `ScalarMaterial.isotropic_elastic(spacing, nu, scale_fn)` (`vec=3`) and `ScalarMaterial.isotropic_conduction(spacing, scale_fn)` (`vec=1`) build the reference matrix from the grid spacing; `simp(...)` / `ramp(...)` are the usual scale functions.
- **`KernelMaterial(tensor_map, vec, spacing)`** — any *linear* feax `tensor_map` (`u_grad (vec,3) -> flux (vec,3)`), evaluated through the per-cell constitutive matrix $D = \partial\,\mathrm{flux}/\partial\,\nabla u$: anisotropic and orientation-dependent laws, multi-material, ... at $(3\,\mathrm{vec})^2$ floats per cell and roughly 5× the matvec cost. `KernelMaterial.from_problem(ProblemClass, grid, vec, additional_info=...)` takes the map straight out of an ordinary `fe.Problem` subclass, so cmg solves the same operator feax assembles (`sample_params=` verifies linearity and symmetry at construction).

```python
from feax.experimental import NarrowBandCMG, ScalarMaterial, KernelMaterial, simp

# steady conduction with SIMP conductivity (one DOF per node)
thermal = NarrowBandCMG(grid, fixed_pred, material=ScalarMaterial.isotropic_conduction(
    grid.spacing, scale_fn=simp(1e-3, 1.0, 3.0)))

# anisotropic elasticity from a feax Problem: tensor_map(u_grad, rho, theta)
elastic = NarrowBandCMG(grid, fixed_pred, material=KernelMaterial.from_problem(
    AnisotropicElasticity, grid, vec=3, sample_params=(0.5, 0.0)))
u = elastic.create_solver(elastic.build(active), b)((rho_cells, theta_cells))
```

The design given to the solver is whatever the material expects — one array (`ScalarMaterial`, or a `KernelMaterial` with one parameter) or a tuple of per-cell arrays (`fe.TracedParams(volume_vars=...)` works for both).

One-way coupled physics (temperature → thermal strain → displacement) is *not* one SPD system, so it is solved as two cmg instances that share the band geometry: `geo = build_geometry(grid, active)` once, then `thermal.build(geo)` and `elastic.build(geo)`.

### From a box mesh Problem to cmg

`StructuredGrid` uses exactly the node numbering, cell numbering and HEX8 corner order of `fe.mesh.box_mesh`, and `StructuredGrid.from_mesh(mesh)` recovers the grid of a box mesh. A full-grid cmg field therefore has the same layout as a feax solution on that mesh: response functions, filters and output code written for the feax Problem work on the cmg result unchanged. Loads and non-zero Dirichlet values have grid-side helpers, and the solver is differentiable in the right-hand side as well as the design:

```python
grid = fe.StructuredGrid.from_mesh(mesh)                  # the box mesh's grid
mat = KernelMaterial.from_problem(problem, grid, sample_params=(0.5, 0.0))
#   `problem` may be an instance (its tensor_map, or the gradient of its
#   get_energy_density, is used as is) or a Problem subclass

# uniform traction on the y = 0 face. feax adds a surface_map t to the residual
# (it acts along -t), so a feax traction t is `traction=-t` here.
b = cmg.face_load_vector(levels, axis=1, side=0, traction=(0.0, -1.0, 0.0))
solver = cmg.create_solver(levels, b)
u = solver(params)                                         # default rhs
u = solver(params, b=other_rhs)                            # rhs at call time

# non-zero Dirichlet values (T = 1 on the fixed top nodes): lift them
T_p = jnp.where(levels[0]["nj"] == grid.ny, 1.0, 0.0)
T = T_p + solver(params, b=cmg.dirichlet_rhs(levels, params, T_p))
#   dirichlet_rhs depends on the design, and jax.grad through it is complete
```

`thermax/examples/thermal_isolation_3d_aniso_uncoupled.py` runs this way with `SOLVER = "cmg"`: anisotropic conduction (`vec=1`) and orientation-averaged elasticity (`vec=3`) from the same thermax Problems, sharing one band geometry, inside a jitted GCMMA driver.

### Differentiability

The solver is differentiable via an implicit-diff `custom_vjp`: FEAX does **not** autodiff through the MGPCG iterations. Since $K(\rho)\,u = b$ with symmetric $K$, the adjoint is another MGPCG solve and the design gradient is $-\lambda^\top (\partial K/\partial \rho)\, u$ via a pure-JAX level-0 matvec:

```python
import jax
import jax.numpy as jnp

solver = cmg.create_solver(levels, b, return_solution=False)
dc = jax.grad(lambda r: jnp.dot(jnp.asarray(b), solver(r)))(rho_cells)
```

For the compliance/OC workflow there is an even cheaper analytic path — the problem is self-adjoint ($\lambda = u$), so **one forward solve** gives both objective and sensitivity:

```python
u = onp.asarray(solver(rho_cells))
c, dc = cmg.compliance_and_dc(levels, rho_cells, u)
# c  = uᵀ K(ρ) u = Σ uₑᵀ Kₑ(ρₑ) uₑ
# dc = -uₑᵀ (∂Kₑ/∂ρₑ) uₑ   (per band cell, any Material; tuple designs give a tuple)
```

### Precision and hierarchy depth

Consumer-class GPUs (a GB10 included) run float64 many times slower than float32, and the V-cycle is where cmg spends its time. `precision="mixed"` runs the multigrid *preconditioner* (smoother, transfers, coarse Krylov) in float32 while the outer CG, the level-0 matvec and the cuDSS coarse solve stay in float64, so the converged result is a float64 solution to `cg_tol` (results and gradients agree with `"double"` to 1e-12). On a 33³ anisotropic elasticity problem one solve drops from 2.7 s to 0.55 s.

`auto_levels` keeps every level at least 16 cells wide by default and leaves the rest to the coarsest-level solver: deeper hierarchies cost more sequential work per V-cycle and converge worse for anisotropic or high-contrast materials (on the same problem, coarsening to 3³ took 71 PCG iterations, stopping at 17³ took 16). `solver.stats(params)` returns the iteration count and residual of one forward solve, the first thing to look at when a solve seems slow.

### Buckets and the jit schedule

`create_solver(..., jit=True)` (default) compiles the MGPCG; the compiled executable is cached on the `cmg` instance and reused across bands whose **padded** shapes match. The `bucket` parameter rounds every per-level capacity up to a multiple (default 512), so a moving band stays in the same compilation bucket while its size flickers. `jit=False` runs eagerly — no compilation at all, which is what you want while the band changes shape every iteration (pair with `bucket=1` for an exact-size band). Result and gradient are identical either way.

## The Moving-Band OC Loop

`feax.experimental.narrowband` provides the moving-band SIMP compliance loop for `NarrowBandCMG`. Each iteration:

1. Re-extract the band from the design: `dilate(rho > threshold, margin) ∪ keep`
2. If the band changed, rebuild the MG hierarchy and solver (eager — no recompile)
3. One forward solve, then `compliance_and_dc` (self-adjoint — no adjoint solve)
4. Filter the sensitivity with the O(band) narrow-band sensitivity filter (`feax.experimental.create_sensitivity_filter(grid, rmin)`)
5. Update the density with the Optimality Criteria bisection `oc_update`

```python
from feax.experimental import run_narrowband_oc

rho, history = run_narrowband_oc(
    grid, cmg, keep, volfrac=0.2, load_fn=load_fn,
    rmin=1.5, n_iter=40, grid_update_start=1,
    threshold=1e-2, margin=2, move=0.2,
    solver_options=fe.DirectSolverOptions(), jit_after=20)
```

- **`keep`** — boolean `(nx, ny, nz)` mask of always-active cells (supports and loads)
- **`load_fn(cmg, levels) -> b`** — builds the RHS on the current band, e.g. `lambda c, lv: c.load_vector(lv, [node], comp=2, value=-1.0)`
- **`grid_update_start`** — iterations of full-grid exploration before banding starts (`0` = band from the first iteration, for a seeded level)
- **`jit_after`** — iteration from which cmg switches to `jit=True`; eager before. The band typically stops moving by then, so the jitted executable is reused across the tail.

:::note Why OC?
The band's design vector changes size every iteration, so the update rule must not carry state tied to the band's dimension. `oc_update` is a pure-numpy O(n) bisection on the volume Lagrange multiplier, needs no state, and pairs naturally with the self-adjoint compliance sensitivity — but it is valid only for compliance minimisation under a single volume constraint. For general objectives/constraints on a fixed mesh, use the MMA driver `feax.gene.optimizer.run()` (see [Adaptive Topology Optimization](adaptive_topology_optimization.md)). A band-aware MMA update built on the pure-JAX `feax.experimental.mma` (optimizer state kept on the full grid) is the planned generalisation of this loop.
:::

`oc_update` itself is general and mesh-agnostic — any 1-D `rho`/`dc` (band or full mesh), with `dv` as the per-element volume weight:

```python
from feax.experimental import oc_update
rho_new = oc_update(rho, dc_filtered, volfrac, move=0.2, xmin=1e-3)
```

## Giga-Voxel Multires Workflow

The remaining bottleneck of a single-resolution loop is the **exploration phase**: with the whole grid active, the first iterations set the memory peak at O(domain). `run_narrowband_multires` removes it with a coarse-to-fine bootstrap:

1. Run full-grid exploration only on a **coarse** grid (cheap)
2. Seed each finer level by trilinearly upsampling the coarser density
3. Run every finer level with `grid_update_start=0`, so it **never** solves the full grid — its peak is the band, not the domain

This moves the resolution ceiling from O(domain) to O(band) — the giga-voxel enabler. Condensed from `examples/advance/high_reso_topopt.py` (a 1200×300×300 cantilever, 108M cells / 327M DOF):

```python
import numpy as onp
import feax as fe
from feax.experimental import NarrowBandCMG, run_narrowband_multires

NELX, NELY, NELZ = 1200, 300, 300           # finest resolution
VOLFRAC = 0.01

def build_fn(nx, ny, nz):
    """Build grid + cmg + keep mask + load at ANY resolution.
    Loads/supports are placed by PHYSICAL location so every level is the
    same problem discretised differently."""
    grid = fe.StructuredGrid((nx, ny, nz))
    fixed_pred = lambda ni, nj, nk, gx, gy, gz: ni == 0        # clamp x=0 face
    cmg = NarrowBandCMG(grid, fixed_pred, nu=0.3, penal=3.0,
                        Emin=1e-9, E0=1.0,
                        cg_tol=1e-7, cg_maxit=300, bucket=256)
    load_node = grid.node_id(nx, ny // 2, nz // 2)             # -z load at x=L centre
    keep = onp.zeros((nx, ny, nz), bool)
    keep[0, :, :] = True                                       # support face
    keep[nx - 1, ny // 2 - 1: ny // 2 + 1,
         nz // 2 - 1: nz // 2 + 1] = True                      # around the load
    load_fn = lambda c, lv: c.load_vector(lv, [load_node], comp=2, value=-1.0)
    return dict(grid=grid, cmg=cmg, keep=keep, load_fn=load_fn)

res = run_narrowband_multires(
    build_fn, (NELX, NELY, NELZ), VOLFRAC,
    n_levels=3, coarse_factor=2,                # full-grid explore only at 300x75x75
    rmin=1.5, n_iter=40, coarse_iter=20,
    grid_update_start=4,                        # full-grid iters, coarsest level only
    threshold=1e-2, margin=2, move=0.2, xmin=1e-3,
    solver_options=fe.DirectSolverOptions(solver="cudss"),
    jit_after=10, verbose=True)

rho = res["x"]              # (NELX, NELY, NELZ) final density grid
```

`build_fn(nx, ny, nz)` must return `dict(grid=, cmg=, keep=, load_fn=)` at any resolution — the key contract is that loads and supports are placed by *physical* location (here via grid-index predicates that scale with the dims), so every level is the same problem discretised differently. The result contains the finest density (`res["x"]`), the resolution ladder (`res["levels"]`), and per-level convergence history (`res["history"]`).

## Practical Notes

- **`margin` vs. void pockets on small grids** — the band margin adds low-density dilation cells around the structure. On coarse grids a large `margin` can enclose void pockets that are technically "active" and slow the MG solve; on fine grids `margin=2` is a good default. The margin is also what lets the structure *grow* — with `margin=0` the band can never expand beyond the current material.
- **The `keep` mask must preserve the load path** — cells near loads and supports must always stay active. If the OC update drives the density near a load below `threshold`, the band would otherwise disconnect the load from the structure and the solve becomes singular (a load node that is not in the band cannot even receive its force).
- **Recompile amortization (`jit_after`, `bucket`)** — a band that moves every iteration would trigger a recompile per iteration under `jax.jit`. The schedule in `run_narrowband_oc` is deliberately simple: eager while the band moves (no compilation), `jit=True` from iteration `jit_after` once the band has settled; a coarser `bucket` absorbs small size flicker in the jitted tail at the cost of padding.
- **MG-friendly dims** — use `StructuredGrid.fit(..., align=8)` (or pick dims divisible by powers of two) so `auto_levels` can build a deep hierarchy; arbitrary (including odd) sizes still coarsen via ceil-halving, but aligned dims coarsen cleanly.
- **cuDSS vs. Krylov coarse solve** — the default cuDSS coarsest-level factorization (via spineax) is factor-once/solve-many and fastest on GPU; pass `fe.KrylovSolverOptions(...)` for a fully matrix-free fallback without the spineax/cuDSS dependency.
- **Recovering a mesh** — `fe.NarrowBand(grid, active)` materializes the band as an explicit HEX8 `Mesh` sharing the cmg node ordering (`band.node_map` equals `levels[0]["nodes"]`), so the cmg displacement maps onto it directly for VTU export or verification against a FEAX-assembled operator (see `examples/basic/lattice_spgrid_cmg.py`).

## Summary

**Key concepts:**
- **`fe.StructuredGrid`** — implicit uniform HEX8 grid; connectivity/coordinates are index arithmetic (O(1) memory); `fit` + `voxelize_mesh` embed arbitrary geometry
- **`fe.SparseDesign`** — sparse per-cell design storage keyed by global cell id; `band_cells`, `traced_params`, `updated`
- **`fe.NarrowBand`** — active-cell sub-mesh over a `Mesh` *or* `StructuredGrid`; gather/scatter for params and solutions; coordinate-based BCs transfer; works with any standard FEAX solver
- **`fe.SupersetBand`** — fixed-superset manager that amortizes recompiles for moving bands
- **`feax.experimental.NarrowBandCMG`** — matrix-free O(band) geometric-MG (MGPCG) solver for structured bands; any linear SPD `Material` (`ScalarMaterial`, `KernelMaterial`) and any `vec`; differentiable (`custom_vjp`); coarse level via `DirectSolverOptions` (cuDSS) or `KrylovSolverOptions`
- **`feax.experimental.oc_update` / `run_narrowband_oc` / `run_narrowband_multires`** — moving-band SIMP loop with self-adjoint sensitivity (one forward solve per iteration) and the coarse-to-fine giga-voxel bootstrap

**Workflow:**
1. Define the domain as a `StructuredGrid` (or `fit` one around a geometry)
2. Choose the active set (design threshold, voxelization, SDF, ...)
3. Solve on the band — `NarrowBand` + any FEAX solver, or `NarrowBandCMG` for structured linear problems at scale
4. For topology optimization, write `build_fn(nx, ny, nz)` and call `run_narrowband_multires`

## Further Reading

- [Adaptive Topology Optimization](adaptive_topology_optimization.md) - MMA-based topopt on unstructured meshes
- `examples/advance/high_reso_topopt.py` - The flagship: multires bootstrap + moving band + OC + cuDSS coarse solve
- `examples/basic/lattice_spgrid_cmg.py` - Arbitrary lattice → spgrid → NarrowBandCMG → recovered mesh
- `examples/advance/narrowband_lattice_homogenization.py` - Narrow band with periodic boundary conditions
- [API: spgrid](../api/reference/feax/spgrid.md) - `StructuredGrid`, `SparseDesign`, `voxelize_mesh`
- [API: narrowband](../api/reference/feax/narrowband.md) - `NarrowBand`, `SupersetBand`
- [API: experimental.cmg](../api/reference/feax/experimental/cmg.md) - `NarrowBandCMG`, `ScalarMaterial`, `KernelMaterial`, `build_geometry`
- [API: experimental.narrowband](../api/reference/feax/experimental/narrowband.md) - `oc_update`, `run_narrowband_oc`, `run_narrowband_multires`, `create_sensitivity_filter`
