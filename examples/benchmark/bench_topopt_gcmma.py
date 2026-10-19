"""Benchmark: isotropic 3-D topology optimization driven by the in-house GCMMA.

Based on ``examples/advance/anisotropic_topology_optimization_3d.py`` with the
anisotropy removed: the same cantilever box, boundary conditions, Heaviside
projection and volume constraint, but a fully isotropic SIMP material, a single
nodal density field, and the lightweight distance-weighted density filter (a
sparse matrix-vector product, no PDE solve) instead of the Helmholtz filter.
NLopt's LD_MMA is replaced by :class:`feax.experimental.mma.GCMMA`, the
pure-JAX GCMMA whose loops are all traced.

Physical setup (same as the base example):

    * Box ``[0, L] x [0, W] x [0, H]`` = ``100 x 5 x 20``, TET4 ``box_mesh``
    * Left face ``x = 0`` clamped
    * Traction ``-z`` on the lower quarter of the right face (``x = L, z < H/4``)
    * SIMP ``E(rho) = E0 * (eps + (1 - eps) * rho**p)``, ``nu`` constant

Optimization problem in Svanberg's form (scaled so the multiplier stays far
below GCMMA's elastic penalty ``c = 1000``):

    minimise   c(rho) / c(rho0)
    subject to V(rho_f) / V* - 1 <= 0,      0 <= rho <= 1

Execution modes, each over a fixed number of outer iterations (no early stop,
so every mode does the same work):

* ``run``   -- host loop over a jitted ``GCMMA.step`` (one compiled step called
               ``--iters`` times; the inner-iteration count is read back every
               step, as a logging driver would).
* ``solve`` -- ``GCMMA.solve_with_history`` under ``jax.jit``: the whole
               optimization, init included, is ONE XLA program.

Both modes compile ahead of time (``jit(f).lower(...).compile()``), so
``compile_s`` contains no execution. Everything mesh-sized the optimization
needs at run time -- the TracedStructure (with the AMG hierarchy for
``--solver amg``), the initial guess, the filter's sparse weight matrix and the
nodal volume weights -- travels as ONE pytree jit ARGUMENT, never as a closure, so none of it is baked
into the executables. ``const_mb`` reports what the traced program still
captures, as a check that this holds (it should stay well under 1 MB and not
grow with the mesh).

Results are appended to ``bench_topopt.csv`` next to this script (git-ignored).

Run:
    python bench_topopt_gcmma.py
    python bench_topopt_gcmma.py --mesh-sizes 2 1.5 1 --solver amg --iters 50
    python bench_topopt_gcmma.py --modes solve --iters 100 --save-dir out_topopt
"""
import argparse
import csv
import os
import statistics
import sys
from datetime import datetime, timezone
from time import perf_counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench_linear_elasticity import device_info  # noqa: E402  (imports no jax at module level)


# ---------------------------------------------------------------------------
# CLI (parsed BEFORE importing feax/jax so env flags take effect)
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Isotropic 3-D topology optimization with the in-house GCMMA.")
    p.add_argument("--mesh-sizes", type=float, nargs="+", default=[2.0, 1.5, 1.0],
                   help="TET4 box_mesh element sizes (smaller = more DOF).")
    p.add_argument("--solver", choices=["direct", "amg", "krylov"], default="direct")
    p.add_argument("--modes", nargs="+", choices=["run", "solve"], default=["run", "solve"])
    p.add_argument("--iters", type=int, default=30, help="Outer GCMMA iterations (fixed).")
    p.add_argument("--max-inner", type=int, default=15, help="GCMMA inner-loop cap.")
    p.add_argument("--repeats", type=int, default=1, help="Timed repeats of a whole optimization.")
    p.add_argument("--vf", type=float, default=0.4, help="Target volume fraction.")
    p.add_argument("--penal", type=float, default=3.0, help="SIMP penalty p.")
    p.add_argument("--filter-radius", type=float, default=3.0,
                   help="Density filter radius (cone weights), in model units.")
    p.add_argument("--beta", type=float, default=4.0, help="Heaviside projection sharpness.")
    p.add_argument("--tol", type=float, default=1e-8, help="Krylov / AMG relative tol.")
    p.add_argument("--maxiter", type=int, default=2000, help="Krylov max iterations.")
    p.add_argument("--save-dir", default=None,
                   help="Write the final projected density of each run as VTU here.")
    p.add_argument("--output", default=None,
                   help="CSV path (default: bench_topopt.csv next to this script).")
    p.add_argument("--tag", default="", help="Free-form label written to every row.")
    p.add_argument("--device-name", default=None, help="Override the detected device name.")
    p.add_argument("--x64", dest="x64", action="store_true", default=True, help="float64 (default).")
    p.add_argument("--no-x64", dest="x64", action="store_false", help="float32.")
    p.add_argument("--preallocate", action="store_true", help="Re-enable XLA GPU preallocation.")
    p.add_argument("--cpu", action="store_true", help="Force the JAX CPU backend.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Fixed physics (= anisotropic_topology_optimization_3d.py, isotropic material)
# ---------------------------------------------------------------------------
L, W, H = 100.0, 5.0, 20.0
E0 = 70.0e3
NU = 0.30
SIMP_EPSILON = 1e-3
TRACTION_MAG = 1.0
TOL = 1e-3

CSV_COLUMNS = [
    "timestamp", "device", "platform", "dtype", "jax_version", "feax_version",
    "solver", "mode", "mesh_size", "n_nodes", "n_cells", "dof", "n_design",
    "iters", "repeats", "compile_s", "total_s", "std_s", "per_iter_s",
    "evals", "mean_inner", "objective", "compliance", "volume", "kkt_norm",
    "const_mb", "tag", "note",
]


def append_row(out_path, row):
    """Append one row (header only for a new file) and fsync it, so a later
    crash or OOM kill cannot lose finished points."""
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
    if args.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["FEAX_X64"] = "1" if args.x64 else "0"
    if args.preallocate:
        os.environ["FEAX_PREALLOCATE"] = "1"

    import jax
    import jax.numpy as jnp
    import feax as fe
    import feax.gene as gene
    from feax.experimental.mma import GCMMA
    from feax.gene.filters import _compute_density_filter_weights_sparse

    try:
        from importlib.metadata import version as _pkg_version
        feax_version = _pkg_version("feax")
    except Exception:
        feax_version = getattr(fe, "__version__", "?")

    plat, device = device_info(jax, args.device_name)
    dtype = "float64" if args.x64 else "float32"
    out_path = args.output or os.path.join(HERE, "bench_topopt.csv")
    print(f"device={device} ({plat})  dtype={dtype}  solver={args.solver}  "
          f"jax={jax.__version__}  feax={feax_version}")
    print(f"box={L:g}x{W:g}x{H:g}  vf={args.vf}  p={args.penal}  r={args.filter_radius}  "
          f"beta={args.beta}  iters={args.iters}  modes={args.modes}  output={out_path}\n")

    def block(x):
        return jax.block_until_ready(x)

    def aot(jitted, *call_args):
        """Trace + compile ahead of time; returns (compile_s, executable)."""
        t0 = perf_counter()
        compiled = jitted.lower(*call_args).compile()
        return perf_counter() - t0, compiled

    def solver_options():
        if args.solver == "direct":
            # reuse_factorization: the adjoint reuses the forward factors (SPD).
            return fe.DirectSolverOptions(reuse_factorization=True)
        if args.solver == "amg":
            return fe.AMGSolverOptions(near_nullspace="rigid_body", tol=args.tol)
        return fe.KrylovSolverOptions(solver="cg", tol=args.tol, atol=args.tol * 1e-3,
                                      maxiter=args.maxiter, use_jacobi_preconditioner=True)

    penal = float(args.penal)

    class IsotropicSIMP(fe.Problem):
        """Linear isotropic elasticity with a SIMP-penalised nodal density."""

        def get_tensor_map(self):
            def stress(u_grad, rho):
                E = E0 * (SIMP_EPSILON + (1.0 - SIMP_EPSILON) * rho ** penal)
                mu = E / (2.0 * (1.0 + NU))
                lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
                eps = 0.5 * (u_grad + u_grad.T)
                return lam * jnp.trace(eps) * jnp.eye(self.dim) + 2.0 * mu * eps
            return stress

        def get_surface_maps(self):
            def traction(u, x, *a):
                return jnp.array([0.0, 0.0, -TRACTION_MAG])
            return [traction]

    left_face = lambda pt: jnp.isclose(pt[0], 0.0, atol=TOL)
    load_patch = lambda pt: jnp.isclose(pt[0], L, atol=TOL) & (pt[2] < H / 4.0)

    def const_mb(fn, *call_args):
        """Bytes of closure constants captured when tracing fn(*call_args)."""
        jaxpr = jax.make_jaxpr(fn)(*call_args)
        return sum(getattr(c, "nbytes", 0) for c in jaxpr.consts) / 1e6

    def build(mesh_size):
        mesh = fe.mesh.box_mesh((L, W, H), mesh_size=mesh_size, element_type="TET4")
        problem = IsotropicSIMP(mesh=mesh, vec=3, dim=3, ele_type=mesh.ele_type,
                                location_fns=(load_patch,))
        bc = fe.DirichletBCConfig([
            fe.DirichletBCSpec(location=left_face, component="all", value=0.0),
        ]).create_bc(problem)
        ts = fe.TracedStructure.from_problem(problem)
        n_nodes = mesh.points.shape[0]
        sample_tp = fe.TracedParams(
            volume_vars=(fe.TracedParams.create_node_var(problem, args.vf),),
            surface_vars=())
        solver = fe.create_solver(problem, bc, solver_options=solver_options(),
                                  linear=True, traced_params=sample_tp, traced_structure=ts)
        # AMG: carry the hierarchy in ts so it is a jit argument, not a constant.
        ts = ts.with_preconditioner(solver.preconditioner_state)
        u0 = fe.zero_like_initial_guess(problem, bc)

        compliance_fn = gene.create_compliance_fn(problem)
        volume_fn = gene.create_volume_fn(problem)
        beta = float(args.beta)

        # Density filter rho_f = (W rho) / rowsum(W), cone weights within the radius.
        # Built with the same routine as gene.create_density_filter, but the sparse W
        # is kept as DATA (a BCOO pytree) so it can be a jit argument: the public
        # filter_fn closes over W, which would bake ~nodes x neighbours entries into
        # every executable.
        filt_W, filt_rowsum = _compute_density_filter_weights_sparse(
            mesh.points, args.filter_radius, "cone")

        # Volume fraction of a nodal density is linear in it, V(rho) = w . rho, and
        # w is exactly the gradient of gene's volume_fn. Precompute w once and pass
        # it as data: volume_fn itself closes over the cell connectivity and JxW.
        vol_w = jax.grad(volume_fn)(jnp.zeros((n_nodes,)))

        # Run-time data, passed as one pytree jit argument everywhere below.
        data = (ts, u0, (filt_W, filt_rowsum), vol_w)

        def density_filter(x, d):
            Wm, rs = d[2]
            return (Wm @ x) / (rs + 1e-12)

        def project(x, d):
            return gene.heaviside_projection(density_filter(x, d), beta=beta)

        def compliance(x, d):
            s, u_init = d[0], d[1]
            tp = fe.TracedParams(volume_vars=(project(x, d),), surface_vars=())
            return compliance_fn(solver(tp, u_init, traced_structure=s))

        def volume(x, d):
            # Constraint on the filtered, un-projected density (as in the base example).
            return jnp.dot(d[3], density_filter(x, d))

        x0 = jnp.full((n_nodes,), args.vf)
        c0 = float(jax.jit(compliance)(x0, data))        # scaling, outside any timing
        vf = float(args.vf)

        def value_and_grad(x, d):
            c, dc = jax.value_and_grad(compliance)(x, d)
            v, dv = jax.value_and_grad(volume)(x, d)
            return c / c0, dc / c0, jnp.reshape(v / vf - 1.0, (1,)), jnp.reshape(dv / vf, (1, -1))

        opt = GCMMA(value_and_grad, 0.0, 1.0, n_constraints=1, max_inner=args.max_inner)
        info = dict(mesh=mesh, n_nodes=n_nodes, n_cells=mesh.cells.shape[0],
                    dof=int(u0.shape[0]), c0=c0, filter_nnz=int(filt_W.nse))
        return opt, x0, data, project, info

    def record(mode, mesh_size, info, compile_s, times, *, evals=float("nan"),
               mean_inner=float("nan"), objective=float("nan"), compliance=float("nan"),
               volume=float("nan"), kkt=float("nan"), cmb=float("nan"), note=""):
        mean = statistics.mean(times) if times else float("nan")
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        row = dict(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            device=device, platform=plat, dtype=dtype, jax_version=jax.__version__,
            feax_version=feax_version, solver=args.solver, mode=mode, mesh_size=mesh_size,
            n_nodes=info.get("n_nodes", ""), n_cells=info.get("n_cells", ""),
            dof=info.get("dof", ""), n_design=info.get("n_nodes", ""), iters=args.iters,
            repeats=len(times), compile_s=compile_s, total_s=mean, std_s=std,
            per_iter_s=mean / args.iters if times else float("nan"), evals=evals,
            mean_inner=mean_inner, objective=objective, compliance=compliance,
            volume=volume, kkt_norm=kkt, const_mb=cmb, tag=args.tag, note=note)
        append_row(out_path, row)
        cs = f"{compile_s:7.2f}s" if compile_s == compile_s else "   -   "
        print(f"  {mode:5s} dof={row['dof']:>9,} compile={cs}  total={mean:8.3f}s  "
              f"per-iter={row['per_iter_s']:7.4f}s  evals={evals:>5}  inner={mean_inner:5.2f}  "
              f"obj={objective:.5f}  vol={volume:.4f}  const={cmb:6.2f}MB{'  (' + note + ')' if note else ''}")
        return row

    finals = {}
    for mesh_size in args.mesh_sizes:
        print(f"mesh_size={mesh_size:g}")
        try:
            opt, x0, data, project, info = build(mesh_size)
        except Exception as e:
            print(f"  build failed: {type(e).__name__}: {e}")
            record("build", mesh_size, {}, float("nan"), [], note=type(e).__name__)
            jax.clear_caches()
            continue
        print(f"  nodes={info['n_nodes']:,}  cells={info['n_cells']:,}  dof={info['dof']:,}  "
              f"filter nnz={info['filter_nnz']:,}  c0={info['c0']:.4e}")

        def summarize(state, inner_total):
            x = state.x
            return dict(
                evals=1 + args.iters + int(inner_total),       # init + one per outer + inner re-evals
                mean_inner=float(inner_total) / args.iters,
                objective=float(state.f0),
                compliance=float(state.f0) * info["c0"],
                volume=float(state.fval[0] + 1.0) * args.vf,
                kkt=float(state.kkt_norm),
            ), x

        # -- run: host loop over one compiled GCMMA step ---------------------
        if "run" in args.modes:
            try:
                init_fn = jax.jit(opt.init)
                step_fn = jax.jit(opt.step)
                c_init, init_c = aot(init_fn, x0, data)
                state = block(init_c(x0, data))
                c_step, step_c = aot(step_fn, state, data)
                cmb = const_mb(opt.step, state, data)
                times = []
                for _ in range(args.repeats):
                    t0 = perf_counter()
                    state = init_c(x0, data)
                    inner_total = 0
                    for _k in range(args.iters):
                        state = step_c(state, data)
                        inner_total += int(state.inner_iterations)   # host read-back each step
                    block(state.x)
                    times.append(perf_counter() - t0)
                stats, x_final = summarize(state, inner_total)
                record("run", mesh_size, info, c_init + c_step, times, cmb=cmb, **stats)
                finals[(mesh_size, "run")] = (stats["objective"], x_final)
            except Exception as e:
                record("run", mesh_size, info, float("nan"), [], note=type(e).__name__)
                print(f"    {type(e).__name__}: {e}")

        # -- solve: the whole optimization as one XLA program ----------------
        if "solve" in args.modes:
            try:
                n_iter = args.iters
                def whole(x, d):
                    return opt.solve_with_history(x, d, n_iter=n_iter)
                compile_s, solve_c = aot(jax.jit(whole), x0, data)
                cmb = const_mb(whole, x0, data)
                times = []
                for _ in range(args.repeats):
                    t0 = perf_counter()
                    state, hist = solve_c(x0, data)
                    block(state.x)
                    times.append(perf_counter() - t0)
                inner_total = int(jnp.sum(hist["inner_iterations"]))
                stats, x_final = summarize(state, inner_total)
                record("solve", mesh_size, info, compile_s, times, cmb=cmb, **stats)
                finals[(mesh_size, "solve")] = (stats["objective"], x_final)
            except Exception as e:
                record("solve", mesh_size, info, float("nan"), [], note=type(e).__name__)
                print(f"    {type(e).__name__}: {e}")

        if (mesh_size, "run") in finals and (mesh_size, "solve") in finals:
            fr, xr = finals[(mesh_size, "run")]
            fs, xs = finals[(mesh_size, "solve")]
            dx = float(jnp.max(jnp.abs(xr - xs)))
            print(f"  run vs solve: |obj diff| = {abs(fr - fs):.2e}  max|x diff| = {dx:.2e}")

        if args.save_dir:
            key = (mesh_size, "solve") if (mesh_size, "solve") in finals else (mesh_size, "run")
            if key in finals:
                os.makedirs(args.save_dir, exist_ok=True)
                path = os.path.join(args.save_dir, f"density_{args.solver}_h{mesh_size:g}.vtu")
                rho_p = jax.device_get(project(finals[key][1], data))
                fe.utils.save_sol(mesh=info["mesh"], sol_file=path,
                                  point_infos=[("density", rho_p)])
                print(f"  wrote {path}")

        jax.clear_caches()
        print()

    print(f"Rows appended to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
