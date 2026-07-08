"""Scaling benchmark: linear-elasticity cantilever solved across a DOF sweep.

The problem is a FIXED-geometry cantilever (a physical box of dimensions
``--dims``, clamped at x=0, tip traction at x=Lx). Only the mesh resolution
changes, so every DOF point is the SAME physical problem refined — the sweep is
scale-invariant and the per-DOF numbers are directly comparable.

For each target DOF the solver wall time is measured in three execution modes:

* ``eager``    — op-by-op dispatch (no top-level jit)
* ``jit``      — ``jax.jit`` (compile time reported separately from steady state)
* ``vmap-lhs`` — ``jax.jit(jax.vmap(...))`` batching the MATERIAL: the left-hand-side
                 matrix ``A`` varies per case (B factorizations)
* ``vmap-rhs`` — batching the LOAD: the right-hand side ``b`` varies with ``A`` fixed
                 (factor-once / solve-many; direct solver only)

Results (with the auto-detected device name) are appended to ``bench.csv``, so
repeated runs on different machines / solvers accumulate in one file.

Run:
    python bench_linear_elasticity.py --dofs 20000 100000 300000
    python bench_linear_elasticity.py --solver direct --modes eager jit
    python bench_linear_elasticity.py --dofs 50000 --batch 16 --modes vmap-rhs
    python bench_linear_elasticity.py --no-x64 --output my_run.csv --tag laptop

Solvers: ``krylov`` (matrix-free CG, scales in memory — default), ``direct``
(cuDSS/host sparse), ``amg`` (rigid-body near-null-space; the ``vmap`` modes are
skipped — its host setup cannot be vmapped).
"""
import argparse
import csv
import os
import platform as _platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from time import perf_counter


# ---------------------------------------------------------------------------
# CLI (parsed BEFORE importing feax/jax so env flags take effect)
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Linear-elasticity cantilever scaling benchmark (eager/jit/vmap).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dofs", type=int, nargs="+",
                   default=[20_000, 60_000, 150_000, 300_000],
                   help="Target DOF counts (mesh is refined to hit each).")
    p.add_argument("--dims", type=float, nargs=3, metavar=("LX", "LY", "LZ"),
                   default=[8.0, 2.0, 2.0],
                   help="Fixed physical cantilever dimensions (scale-invariant).")
    p.add_argument("--solver", choices=["krylov", "direct", "amg"], default="krylov")
    p.add_argument("--modes", nargs="+",
                   choices=["eager", "jit", "vmap-lhs", "vmap-rhs"],
                   default=["eager", "jit", "vmap-lhs", "vmap-rhs"],
                   help="vmap-lhs: batch the material E (left-hand-side matrix A "
                        "varies -> B factorizations). vmap-rhs: batch the "
                        "load/traction (right-hand side b varies, A fixed -> "
                        "factor once, multi-RHS solve).")
    p.add_argument("--batch", type=int, default=8, help="vmap batch size.")
    p.add_argument("--repeats", type=int, default=5, help="Timed repeats per point.")
    p.add_argument("--warmup", type=int, default=2, help="Untimed warmup runs.")
    p.add_argument("--tol", type=float, default=1e-8, help="Krylov relative tol.")
    p.add_argument("--maxiter", type=int, default=2000, help="Krylov max iterations.")
    p.add_argument("--output", default=None,
                   help="CSV path (default: bench.csv next to this script).")
    p.add_argument("--tag", default="", help="Free-form label written to every row.")
    p.add_argument("--device-name", default=None,
                   help="Override the auto-detected device name in the CSV.")
    p.add_argument("--x64", dest="x64", action="store_true", default=True,
                   help="float64 (default).")
    p.add_argument("--no-x64", dest="x64", action="store_false",
                   help="float32.")
    p.add_argument("--preallocate", action="store_true",
                   help="Re-enable XLA GPU memory preallocation.")
    p.add_argument("--cpu", action="store_true",
                   help="Force the JAX CPU backend (JAX_PLATFORMS=cpu). Note: the "
                        "direct solver falls back to a host sparse solver — no "
                        "cuDSS, so vmap-rhs has no factor-once benefit on CPU.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Geometry: pick the mesh resolution whose DOF count is closest to a target,
# keeping the physical dimensions fixed (scale-invariant refinement).
# ---------------------------------------------------------------------------
def resolution_for_dof(target, dims):
    """Return ``(nx, ny, nz, dof)`` closest to ``target`` for fixed ``dims``.

    Cells stay ~cubic: ``ny/nz`` track ``nx`` by the ``dims`` aspect ratio, so
    the physical box is unchanged and only the cell size shrinks. HEX8, vec=3,
    so ``dof = 3 (nx+1)(ny+1)(nz+1)``.
    """
    lx, ly, lz = dims

    def dof_of(nx):
        s = nx / lx
        ny = max(1, round(ly * s))
        nz = max(1, round(lz * s))
        return 3 * (nx + 1) * (ny + 1) * (nz + 1), ny, nz

    hi = 4
    while dof_of(hi)[0] < target:
        hi *= 2
    lo = 1
    while lo < hi:                                  # smallest nx with dof >= target
        mid = (lo + hi) // 2
        if dof_of(mid)[0] < target:
            lo = mid + 1
        else:
            hi = mid
    cand = [lo - 1, lo] if lo > 1 else [lo]
    nx = min(cand, key=lambda c: abs(dof_of(c)[0] - target))
    dof, ny, nz = dof_of(nx)
    return nx, ny, nz, dof


# ---------------------------------------------------------------------------
# Device / environment capture
# ---------------------------------------------------------------------------
def _cpu_name():
    """Best-effort human-readable CPU model name (lscpu -> /proc/cpuinfo -> arch)."""
    try:
        out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if line.lower().startswith("model name"):
                n = line.split(":", 1)[1].strip()
                if n:
                    return n
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):        # x86
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return _platform.processor() or _platform.machine() or "cpu"


def device_info(jax, override):
    dev = jax.devices()[0]
    plat = dev.platform
    name = override or getattr(dev, "device_kind", None) or str(dev)
    if not override and plat == "gpu" and name in ("", "gpu", "cuda"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True, stderr=subprocess.DEVNULL)
            name = out.strip().splitlines()[0].strip()
        except Exception:
            name = f"{_platform.machine()}-gpu"
    if not override and plat == "cpu":
        name = _cpu_name()
    return plat, name


def timed(fn, block, warmup, repeats):
    """Run ``fn`` (warmup untimed, then ``repeats`` timed). Returns a list of s."""
    for _ in range(warmup):
        block(fn())
    out = []
    for _ in range(repeats):
        t0 = perf_counter()
        r = fn()
        block(r)
        out.append(perf_counter() - t0)
    return out


CSV_COLUMNS = [
    "timestamp", "device", "platform", "dtype", "jax_version", "feax_version",
    "solver", "mode", "dof", "nx", "ny", "nz", "batch",
    "warmup", "repeats", "compile_s", "mean_s", "std_s", "min_s",
    "throughput_dofps", "amortized_s", "tag", "note",
]


def append_row(out_path, row):
    """Append one result row immediately (header written only when creating the
    file), and fsync so a later SIGKILL — e.g. the host-memory watchdog killing
    a crash-prone jit/vmap compile — cannot lose already-completed DOF points."""
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    new_file = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    with open(out_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def main(argv=None):
    args = parse_args(argv)

    # Env flags must be set before importing jax/feax.
    if args.cpu:                                  # force the JAX CPU backend
        os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["FEAX_X64"] = "1" if args.x64 else "0"
    if args.preallocate:
        os.environ["FEAX_PREALLOCATE"] = "1"
    # vmap-lhs batches the MATRIX -> B live factorizations at once; size the spineax
    # factor cache above B so it does not evict tokens ("unknown/evicted token").
    os.environ.setdefault("SPINEAX_FACTOR_CACHE", str(max(16, args.batch + 4)))

    import jax
    import jax.numpy as jnp
    import feax as fe

    try:
        from importlib.metadata import version as _pkg_version
        feax_version = _pkg_version("feax")
    except Exception:
        feax_version = getattr(fe, "__version__", "?")

    plat, device = device_info(jax, args.device_name)
    dtype = "float64" if args.x64 else "float32"
    out_path = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "bench.csv")

    print(f"device={device} ({plat})  dtype={dtype}  solver={args.solver}  "
          f"jax={jax.__version__}  feax={feax_version}")
    print(f"dims={tuple(args.dims)} (fixed)  modes={args.modes}  "
          f"batch={args.batch}  output={out_path}\n")

    def block(x):                                  # solvers return a flat array
        return jax.block_until_ready(getattr(x, "dofs", x))

    def solver_options():
        if args.solver == "direct":
            # reuse_factorization=True lets vmap-rhs (load-batched, matrix fixed)
            # factor once and multi-RHS solve (factor-once / solve-many).
            return fe.DirectSolverOptions(reuse_factorization=True)
        if args.solver == "amg":
            return fe.AMGSolverOptions(near_nullspace="rigid_body", verbose=False)
        return fe.KrylovSolverOptions(solver="cg", tol=args.tol,
                                      atol=args.tol * 1e-3, maxiter=args.maxiter,
                                      use_jacobi_preconditioner=True)

    nu = 0.3

    class LinearElasticity(fe.Problem):
        def get_tensor_map(self):
            def stress(u_grad, E):
                mu = E / (2. * (1. + nu))
                lam = E * nu / ((1. + nu) * (1. - 2. * nu))
                eps = 0.5 * (u_grad + u_grad.T)
                return lam * jnp.trace(eps) * jnp.eye(self.dim) + 2. * mu * eps
            return stress

        def get_surface_maps(self):
            def surface_map(u, x, traction_mag):
                return jnp.array([0., 0., traction_mag])
            return [surface_map]

    def build(nx, ny, nz):
        lx, ly, lz = args.dims
        spacing = (lx / nx, ly / ny, lz / nz)
        mesh = fe.StructuredGrid((nx, ny, nz), spacing=spacing).to_mesh()
        left = lambda p: jnp.isclose(p[0], 0., 1e-5)
        right = lambda p: jnp.isclose(p[0], lx, 1e-5 + 0.5 * spacing[0])
        problem = LinearElasticity(mesh, vec=3, dim=3, ele_type="HEX8",
                                   location_fns=[right])
        bc = fe.DirichletBCConfig([
            fe.DirichletBCSpec(location=left, component="all", value=0.),
        ]).create_bc(problem)
        E = fe.TracedParams.create_node_var(problem, 70e3)
        traction = fe.TracedParams.create_uniform_surface_var(problem, 1e-3)
        surf = [(traction,)]
        tp = fe.TracedParams(volume_vars=(E,), surface_vars=surf)
        ts = fe.TracedStructure.from_problem(problem)
        solver = fe.create_solver(problem, bc, solver_options=solver_options(),
                                  linear=True, traced_params=tp,
                                  traced_structure=ts, return_solution=False)
        initial = fe.zero_like_initial_guess(problem, bc)
        return solver, tp, initial, ts, E, surf

    rows = []

    def record(mode, dof, res, batch, compile_s, times, note=""):
        mean = statistics.mean(times) if times else float("nan")
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        mn = min(times) if times else float("nan")
        thru = (dof * batch / mean) if times and mean > 0 else float("nan")
        amort = (mean / batch) if times else float("nan")
        row = dict(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            device=device, platform=plat, dtype=dtype, jax_version=jax.__version__,
            feax_version=feax_version, solver=args.solver, mode=mode, dof=dof,
            nx=res[0], ny=res[1], nz=res[2], batch=batch, warmup=args.warmup,
            repeats=args.repeats, compile_s=compile_s, mean_s=mean, std_s=std,
            min_s=mn, throughput_dofps=thru, amortized_s=amort, tag=args.tag, note=note)
        rows.append(row)
        append_row(out_path, row)          # persist immediately (survives SIGKILL)
        tag = f"  ({note})" if note else ""
        cs = f"compile={compile_s:7.3f}s" if compile_s == compile_s else "compile=   -   "
        print(f"  {mode:5s} dof={dof:>9,} nx={res[0]:>4} b={batch:>2}  "
              f"{cs}  mean={mean:8.4f}s  min={mn:8.4f}s  "
              f"{thru/1e6:7.2f} MDOF/s{tag}")

    for target in args.dofs:
        nx, ny, nz, dof = resolution_for_dof(target, args.dims)
        res = (nx, ny, nz)
        print(f"target={target:,} -> nx={nx} ny={ny} nz={nz}  dof={dof:,}")
        try:
            solver, tp, initial, ts, E, surf = build(nx, ny, nz)
        except Exception as e:                     # OOM at build / assembly
            print(f"  build failed: {type(e).__name__}: {e}")
            record("build", dof, res, 1, float("nan"), [], note=type(e).__name__)
            jax.clear_caches()
            continue

        if "eager" in args.modes:
            try:
                fn = lambda: solver(tp, initial, traced_structure=ts)
                record("eager", dof, res, 1, float("nan"),
                       timed(fn, block, args.warmup, args.repeats))
            except Exception as e:
                record("eager", dof, res, 1, float("nan"), [], note=type(e).__name__)

        if "jit" in args.modes:
            try:
                jfn = jax.jit(lambda t, ig: solver(t, ig, traced_structure=ts))
                t0 = perf_counter(); block(jfn(tp, initial))
                compile_s = perf_counter() - t0
                record("jit", dof, res, 1, compile_s,
                       timed(lambda: jfn(tp, initial), block, args.warmup, args.repeats))
            except Exception as e:
                record("jit", dof, res, 1, float("nan"), [], note=type(e).__name__)

        # vmap-lhs: batch the material E (left-hand-side matrix A varies per
        #           element -> B distinct cuDSS factorizations). vmap-rhs: batch
        #           the load/traction (right-hand side b varies, A FIXED -> ONE
        #           factorization + a single multi-RHS solve, factor-once/solve-many).
        vmap_kinds = []
        if "vmap-lhs" in args.modes:
            vmap_kinds.append("lhs")
        if "vmap-rhs" in args.modes:
            vmap_kinds.append("rhs")
        for kind in vmap_kinds:
            label = f"vmap-{kind}"
            if args.solver == "amg":
                record(label, dof, res, args.batch, float("nan"), [],
                       note="skipped: AMG host setup is not vmappable")
                print(f"  {label} skipped (AMG host setup is not vmappable)")
                continue
            try:
                B = args.batch
                scales = jnp.linspace(0.8, 1.2, B)
                if kind == "lhs":
                    E_batch = E[None, :] * scales[:, None]         # (B, num_nodes)

                    def solve_one(E_field):
                        tpi = fe.TracedParams(volume_vars=(E_field,), surface_vars=surf)
                        return solver(tpi, initial, traced_structure=ts)
                    batched_arg = E_batch
                else:                                              # kind == "rhs"
                    traction = surf[0][0]                          # the load leaf
                    tr_batch = traction[None] * scales.reshape((B,) + (1,) * traction.ndim)

                    def solve_one(tr_field):
                        tpi = fe.TracedParams(volume_vars=(E,),
                                              surface_vars=[(tr_field,)])
                        return solver(tpi, initial, traced_structure=ts)
                    batched_arg = tr_batch

                vfn = jax.jit(jax.vmap(solve_one))
                t0 = perf_counter(); block(vfn(batched_arg))
                compile_s = perf_counter() - t0
                record(label, dof, res, B, compile_s,
                       timed(lambda: vfn(batched_arg), block, args.warmup, args.repeats))
            except Exception as e:
                record(label, dof, res, args.batch, float("nan"), [],
                       note=type(e).__name__)

        jax.clear_caches()
        print()

    # Rows were already persisted incrementally by record()/append_row above,
    # so nothing is written here — this is just the run summary.
    print(f"Wrote {len(rows)} row(s) to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
