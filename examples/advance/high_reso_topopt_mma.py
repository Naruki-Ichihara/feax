"""3D cantilever narrow-band SIMP topopt — multires bootstrap + cmg + MMA.

MMA variant of ``high_reso_topopt.py``: identical band extraction, cmg solve
and self-adjoint sensitivity, but the density update is the pure-JAX MMA
(``feax.experimental.mma``) with its state (design, previous iterates,
asymptotes) on the FULL grid, so the moving band never has to remap optimizer
state. The full-grid state costs about 12 vectors of num_cells floats on the
device; the solve stays O(band).

Coarse-grid bootstrap (BandJax's multires): the full-grid EXPLORATION phase (whole
domain active — what sets the memory peak) runs only on a COARSE grid; each finer
level is SEEDED by upsampling the coarser design and runs with grid_update_start=0,
so the fine level NEVER solves the full grid — its peak is the band, not the domain.
That moves the resolution ceiling from O(domain) to O(band).

At every level the band is a TRUE moving narrow band (re-extracted each iter as
dilate(rho>thr) | keep) solved by the O(band) matrix-free cmg (MGPCG), with the
self-adjoint compliance sensitivity (one forward solve, no adjoint), the narrow-band
sensitivity filter, and gene's OC update. cmg runs EAGER while the band moves (no
recompile) and switches to jit once the band settles.

Problem: x=0 face clamped, unit -z load at the centre of the x=L face.
Outputs: cantilever3d_mma.npy / _history.csv / .vti / _slice.png
"""
import os
import time
import resource

import numpy as onp

import feax as fe
from feax.solvers.options import DirectSolverOptions
import jax.numpy as jnp  # noqa: F401  (for MMA_OPTIONS dtype overrides)

from feax.experimental import NarrowBandCMG, run_narrowband_multires

NELX, NELY, NELZ = 1200, 300, 300            # finest resolution
VOLFRAC, PENAL, RMIN = 0.1, 3.0, 1.5
Emin, E0, XMIN, MOVE = 1e-9, 1.0, 1e-3, 0.2
N_LEVELS, COARSE_FACTOR = 3, 2              # 3 -> full-grid explore only at 63x32x32 (safe on GB10)
N_ITER, COARSE_ITER = 10, 10               # iters at fine / at each coarse level
GRID_UPDATE_START = 4                       # full-grid exploration iters (coarsest only)
THRESH, MARGIN = 1e-2, 2
BUCKET = 256                                # pad band size to a bucket so the jit tail reuses
JIT_AFTER = 10                              # eager for the first 10 iters/level, then jit=True
# Full-grid MMA state: ~12 vectors of NELX*NELY*NELZ floats on the device
# (~10 GB in float64 at 108M cells). dtype=jnp.float32 halves it, but the
# subproblem's barrier (epsimin=1e-7) then sits at float32 precision -> raise
# epsimin to ~1e-5 if you do.
MMA_OPTIONS = dict(asyinit=0.5, c=1000.0)

# cuDSS direct factorization at the coarsest MG level (factor-once/solve-many)
coarse_opts = DirectSolverOptions(solver="cudss")


def build_fn(nx, ny, nz):
    """Build the StructuredGrid + cmg + keep mask + load at ANY resolution.
    Loads/supports are placed by PHYSICAL location so every level is the same
    problem discretised differently (clamp x=0 face, unit -z load at x=L centre)."""
    grid = fe.StructuredGrid((nx, ny, nz), spacing=(1.0, 1.0, 1.0))
    fixed_pred = lambda ni, nj, nk, gx, gy, gz: ni == 0           # clamp x=0 face
    cmg = NarrowBandCMG(grid, fixed_pred, nu=0.3, penal=PENAL, Emin=Emin, E0=E0,
                        cg_tol=1e-7, cg_maxit=300, bucket=BUCKET)
    load_node = grid.node_id(nx, ny // 2, nz // 2)               # -z load at x=L centre
    keep = onp.zeros((nx, ny, nz), bool)
    keep[0, :, :] = True                                         # support face ex=0
    for ey in (ny // 2 - 1, ny // 2):
        for ez in (nz // 2 - 1, nz // 2):
            keep[nx - 1, ey, ez] = True                          # around the load
    load_fn = lambda c, lv: c.load_vector(lv, [load_node], comp=2, value=-1.0)
    return dict(grid=grid, cmg=cmg, keep=keep, load_fn=load_fn)


ndof = 3 * (NELX + 1) * (NELY + 1) * (NELZ + 1)
print(f"3D cantilever {NELX}x{NELY}x{NELZ}  cells={NELX*NELY*NELZ:,}  DOF={ndof:,}  "
      f"volfrac={VOLFRAC}  solver=cmg  update=mma  multires={N_LEVELS} levels")
t0 = time.perf_counter()
res = run_narrowband_multires(
    build_fn, (NELX, NELY, NELZ), VOLFRAC, n_levels=N_LEVELS,
    coarse_factor=COARSE_FACTOR, rmin=RMIN, n_iter=N_ITER, coarse_iter=COARSE_ITER,
    grid_update_start=GRID_UPDATE_START, move=MOVE, xmin=XMIN,
    threshold=THRESH, margin=MARGIN, solver_options=coarse_opts,
    jit_after=JIT_AFTER, update="mma", mma_options=MMA_OPTIONS, verbose=True)
dt = time.perf_counter() - t0

x = onp.asarray(res["x"])                                        # (NELX, NELY, NELZ)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
outdir = os.path.dirname(os.path.abspath(__file__))

# flatten per-level history to (level, iter, compliance, active_frac, volume)
flat = [(lv["level"], it, c, af, v)
        for lv in res["history"] for (it, c, af, v) in lv["history"]]
onp.save(os.path.join(outdir, "cantilever3d_mma.npy"), x)
with open(os.path.join(outdir, "cantilever3d_mma_history.csv"), "w") as fh:
    fh.write("level,iter,compliance,active_frac,volume\n")
    for lv, it, c, af, v in flat:
        fh.write(f"{lv},{it},{c:.6e},{af:.4f},{v:.4f}\n")

lv, it, c, af, v = flat[-1]
print("\n=== result (multires bootstrap + moving narrow band + MMA) ===")
print(f"final compliance : {c:.4e}")
print(f"active band      : {af*100:.1f}%   solid fraction {(x>0.5).mean()*100:.1f}% (target {VOLFRAC*100:.0f}%)")
print(f"wall time        : {dt:.1f}s   peak RSS {peak:.1f} GB")

try:
    import pyvista as pv
    g = pv.ImageData(dimensions=(onp.array(x.shape) + 1))
    g.cell_data["density"] = x.flatten(order="F")
    g.save(os.path.join(outdir, "cantilever3d_mma.vti"))
    print("wrote cantilever3d_mma.npy / _history.csv / cantilever3d_mma.vti")
except Exception as e:
    print(f"(no vti: {e}) wrote cantilever3d_mma.npy / _history.csv")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.imshow(-x[:, :, NELZ // 2].T, cmap="gray", origin="lower", vmin=-1, vmax=0)
    plt.title(f"cantilever3d (MMA) z={NELZ//2}  c={c:.3e}  active={af*100:.0f}%")
    plt.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cantilever3d_mma_slice.png"), dpi=120)
    print("wrote cantilever3d_mma_slice.png")
except Exception as e:
    print(f"(no slice png: {e})")
