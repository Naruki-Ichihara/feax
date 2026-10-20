"""Benchmark: isotropic 3-D topology optimization driven by the in-house GCMMA.

The structure is the cantilever of ``bench_linear_elasticity.py`` (same box,
HEX8 structured grid, DOF sweep, clamp and tip traction), so the per-solve cost
here can be read against that benchmark. Design follows
``examples/advance/anisotropic_topology_optimization_3d.py`` with the anisotropy
removed: SIMP, Heaviside projection and a volume constraint, a single nodal
density field, and the lightweight distance-weighted density filter (a sparse
matrix-vector product, no PDE solve) instead of the Helmholtz filter. NLopt's
LD_MMA is replaced by :class:`feax.experimental.mma.GCMMA`, the pure-JAX GCMMA
whose loops are all traced.

Physical setup (= bench_linear_elasticity.py):

    * Box ``[0, L] x [0, W] x [0, H]`` = ``8 x 2 x 2``, HEX8 ``StructuredGrid``
      refined to each ``--dofs`` target with ~cubic cells
    * Left face ``x = 0`` clamped
    * Uniform traction ``(0, 0, 1e-3)`` on the whole right face ``x = L``
    * SIMP ``E(rho) = E0 * (eps + (1 - eps) * rho**p)``, ``E0 = 70e3``, ``nu = 0.3``

Optimization problem in Svanberg's form (scaled so the multiplier stays far
below GCMMA's elastic penalty ``c = 1000``):

    minimise   c(rho) / c(rho0)
    subject to V(rho_f) / V* - 1 <= 0,      0 <= rho <= 1

Execution modes, each over a fixed number of outer iterations (no early stop,
so every mode does the same work):

* ``run``   -- host loop over a jitted ``GCMMA.step`` (one compiled step called
               ``--iters`` times; the inner-iteration count is read back every
               step, as a logging driver would). With ``--solver amg`` and
               ``--amg-rebuild-every N`` the AMG hierarchy is rebuilt on the host
               from the current design every N iterations (see below).
* ``solve`` -- ``GCMMA.solve_with_history`` under ``jax.jit``: the whole
               optimization, init included, is ONE XLA program.

Both modes compile ahead of time (``jit(f).lower(...).compile()``), so
``compile_s`` contains no execution. Everything mesh-sized the optimization
needs at run time -- the TracedStructure, the initial guess, the filter's sparse
weight matrix, the nodal volume weights and (for ``--solver amg``) the AMG
preconditioner ``P`` (an ``fe.AMGPreconditioner`` pytree, handed to every solve
as ``preconditioner=P``) -- travels as ONE pytree jit ARGUMENT, never as a
closure, so none of it is baked into the executables. ``const_mb`` reports what the traced program still
captures, as a check that this holds (it should stay well under 1 MB and not
grow with the mesh).

The mesh is sized from ``--dofs`` targets (the same 50k..1M sweep and
resolution rule as ``bench_linear_elasticity.py``). The filter radius
defaults to ``--filter-radius-cells`` element sizes so every mesh keeps the same
neighbourhood per node, and the filter weights are built with a KD-tree (the
all-pairs builder in ``feax.gene`` does not scale to 1M DOF).

AMG rebuild: the hierarchy built by ``fe.amg`` comes from a uniform-density
sample. As the design turns black-and-white (stiffness contrast ~1/SIMP_EPSILON)
that fixed V-cycle degrades and CG iterations grow ~10x. ``--amg-rebuild-every N``
reassembles the Jacobian at the current design every N outer iterations, rebuilds
the hierarchy with PyAMG (``P = P.rebuild(J)``) and swaps the new preconditioner
into the data pytree. Because the hierarchy is a jit argument the step does not
recompile unless its level shapes change. ``rebuild_s`` (assembly + PyAMG) and ``recompile_s`` are included in
``total_s`` and also reported separately. ``solve`` mode cannot rebuild (the
whole optimization is one XLA program), so it always uses the initial hierarchy.

Results are appended to ``bench_topopt.csv`` next to this script (git-ignored).

Run:
    python bench_topopt_gcmma.py
    python bench_topopt_gcmma.py --dofs 50000 100000 --solver amg --iters 50
    python bench_topopt_gcmma.py --solver amg --modes run --amg-rebuild-every 5
    python bench_topopt_gcmma.py --modes solve --iters 100 --save-dir out_topopt

The final structure of every mesh size is written as VTU (default ``out_topopt/``
next to this script; ``--no-save`` to skip): point fields ``density`` (projected),
``density_filtered``, ``design`` (raw GCMMA variable) and ``displacement`` (the
solution at the final design), plus the cell field ``density`` (mean of the
projected nodal density) for thresholding in ParaView.
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
from bench_linear_elasticity import device_info, resolution_for_dof  # noqa: E402  (no jax at import)


# ---------------------------------------------------------------------------
# CLI (parsed BEFORE importing feax/jax so env flags take effect)
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Isotropic 3-D topology optimization with the in-house GCMMA.")
    p.add_argument("--dofs", type=int, nargs="+",
                   default=[50_000, 100_000, 250_000, 500_000, 1_000_000],
                   help="Target DOF counts (same sweep as bench_linear_elasticity.py); "
                        "the grid is refined to the closest achievable count.")
    p.add_argument("--solver", choices=["direct", "amg", "krylov"], default="direct")
    p.add_argument("--modes", nargs="+", choices=["run", "solve"], default=["run", "solve"])
    p.add_argument("--iters", type=int, default=30, help="Outer GCMMA iterations (fixed).")
    p.add_argument("--max-inner", type=int, default=15, help="GCMMA inner-loop cap.")
    p.add_argument("--repeats", type=int, default=1, help="Timed repeats of a whole optimization.")
    p.add_argument("--vf", type=float, default=0.4, help="Target volume fraction.")
    p.add_argument("--penal", type=float, default=3.0, help="SIMP penalty p.")
    p.add_argument("--filter-radius", type=float, default=None,
                   help="Density filter radius (cone weights) in model units; "
                        "default --filter-radius-cells x element size.")
    p.add_argument("--filter-radius-cells", type=float, default=3.0,
                   help="Filter radius in element sizes when --filter-radius is unset.")
    p.add_argument("--beta", type=float, default=4.0, help="Heaviside projection sharpness.")
    p.add_argument("--tol", type=float, default=1e-8, help="Krylov / AMG relative tol.")
    p.add_argument("--maxiter", type=int, default=2000, help="Krylov max iterations.")
    p.add_argument("--amg-rebuild-every", type=int, default=0,
                   help="run mode + --solver amg: rebuild the AMG hierarchy from the "
                        "current design every N outer iterations (0 = never).")
    p.add_argument("--save-dir", default=os.path.join(HERE, "out_topopt"),
                   help="Write the final structure (densities + displacement) of each "
                        "mesh size as VTU here.")
    p.add_argument("--no-save", dest="save_dir", action="store_const", const=None,
                   help="Do not write VTU files.")
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
# Fixed physics (= bench_linear_elasticity.py cantilever, SIMP material)
# ---------------------------------------------------------------------------
L, W, H = 8.0, 2.0, 2.0
E0 = 70.0e3
NU = 0.30
SIMP_EPSILON = 1e-3
TRACTION_MAG = 1e-3

CSV_COLUMNS = [
    "timestamp", "device", "platform", "dtype", "jax_version", "feax_version",
    "solver", "mode", "mesh_size", "n_nodes", "n_cells", "dof", "n_design",
    "iters", "repeats", "compile_s", "total_s", "std_s", "per_iter_s",
    "evals", "mean_inner", "objective", "compliance", "volume", "kkt_norm",
    "const_mb", "filter_radius", "filter_nnz", "rebuild_every", "n_rebuilds",
    "rebuild_s", "recompile_s", "tag", "note",
]


def filter_weights(points, radius):
    """Cone-weight density filter ``W_ij = max(0, r - |x_i - x_j|)`` as a BCOO plus
    its row sums: the same weights as ``feax.gene``'s sparse builder, found with a
    KD-tree instead of all pairs."""
    import numpy as onp
    import jax.numpy as jnp
    from jax.experimental import sparse as jsparse
    from scipy.spatial import cKDTree

    pts = onp.asarray(points)
    tree = cKDTree(pts)
    D = tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")
    off = D.row != D.col                       # the diagonal (d = 0) is added explicitly
    n = pts.shape[0]
    rows = onp.concatenate([D.row[off], onp.arange(n)])
    cols = onp.concatenate([D.col[off], onp.arange(n)])
    w = onp.concatenate([onp.maximum(0.0, radius - D.data[off]), onp.full(n, radius)])
    Wm = jsparse.BCOO((jnp.asarray(w), jnp.asarray(onp.stack([rows, cols], 1), dtype=jnp.int32)),
                      shape=(n, n))
    rowsum = jnp.asarray(onp.bincount(rows, weights=w, minlength=n))
    return Wm, rowsum


def append_row(out_path, row):
    """Append one row (header only for a new file) and fsync it, so a later
    crash or OOM kill cannot lose finished points."""
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    new_file = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    if not new_file:
        with open(out_path, newline="") as fh:
            header = next(csv.reader(fh), [])
        if header != CSV_COLUMNS:
            raise ValueError(f"{out_path} has an older column layout; move it aside "
                             "or pass --output.")
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
    from feax.assembler import create_J_bc_csr_parametric

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
    r_desc = (f"r={args.filter_radius}" if args.filter_radius is not None
              else f"r={args.filter_radius_cells:g}h")
    print(f"box={L:g}x{W:g}x{H:g}  vf={args.vf}  p={args.penal}  {r_desc}  "
          f"beta={args.beta}  iters={args.iters}  modes={args.modes}  "
          f"amg_rebuild_every={args.amg_rebuild_every}  output={out_path}\n")

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
            # Outer CG of the AMG-preconditioned solve (the hierarchy itself is
            # built by fe.amg in build()).
            return fe.KrylovSolverOptions(solver="cg", tol=args.tol, atol=1e-10, maxiter=500)
        return fe.KrylovSolverOptions(solver="cg", tol=args.tol, atol=args.tol * 1e-3,
                                      maxiter=args.maxiter)

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
                return jnp.array([0.0, 0.0, TRACTION_MAG])
            return [traction]

    def const_mb(fn, *call_args):
        """Bytes of closure constants captured when tracing fn(*call_args)."""
        jaxpr = jax.make_jaxpr(fn)(*call_args)
        return sum(getattr(c, "nbytes", 0) for c in jaxpr.consts) / 1e6

    def build(res):
        # Same grid, clamp and load faces as bench_linear_elasticity.build().
        nx, ny, nz = res
        spacing = (L / nx, W / ny, H / nz)
        mesh = fe.StructuredGrid((nx, ny, nz), spacing=spacing).to_mesh()
        left = lambda pt: jnp.isclose(pt[0], 0.0, 1e-5)
        right = lambda pt: jnp.isclose(pt[0], L, 1e-5 + 0.5 * spacing[0])
        problem = IsotropicSIMP(mesh=mesh, vec=3, dim=3, ele_type="HEX8",
                                location_fns=(right,))
        bc = fe.DirichletBCConfig([
            fe.DirichletBCSpec(location=left, component="all", value=0.0),
        ]).create_bc(problem)
        mesh_size = spacing[0]
        ts = fe.TracedStructure.from_problem(problem)
        n_nodes = mesh.points.shape[0]
        sample_tp = fe.TracedParams(
            volume_vars=(fe.TracedParams.create_node_var(problem, args.vf),),
            surface_vars=())
        opts = solver_options()
        # AMG: the hierarchy P (built once from the uniform-density sample) is a
        # pytree carried in the run-time data below, so it is a jit argument, not
        # a constant. None for the other solvers (krylov uses fe.jacobi()).
        if args.solver == "amg":
            P = fe.amg(problem, bc, sample_tp, fe.AMGOptions(near_nullspace="rigid_body"),
                       traced_structure=ts)
            pre = P
        else:
            P = None
            pre = fe.jacobi() if args.solver == "krylov" else None
        solver = fe.create_solver(problem, bc, solver_options=opts, preconditioner=pre,
                                  linear=True, traced_params=sample_tp, traced_structure=ts)
        u0 = fe.zero_like_initial_guess(problem, bc)

        compliance_fn = gene.create_compliance_fn(problem)
        volume_fn = gene.create_volume_fn(problem)
        beta = float(args.beta)

        # Density filter rho_f = (W rho) / rowsum(W), cone weights within the radius.
        # Same weights as gene.create_density_filter, but the sparse W is kept as
        # DATA (a BCOO pytree) so it can be a jit argument: the public filter_fn
        # closes over W, which would bake ~nodes x neighbours entries into every
        # executable.
        radius = (args.filter_radius if args.filter_radius is not None
                  else args.filter_radius_cells * mesh_size)
        filt_W, filt_rowsum = filter_weights(mesh.points, radius)

        # Volume fraction of a nodal density is linear in it, V(rho) = w . rho, and
        # w is exactly the gradient of gene's volume_fn. Precompute w once and pass
        # it as data: volume_fn itself closes over the cell connectivity and JxW.
        vol_w = jax.grad(volume_fn)(jnp.zeros((n_nodes,)))

        # Run-time data, passed as one pytree jit argument everywhere below.
        # d[4] is the AMG preconditioner (None unless --solver amg).
        data = (ts, u0, (filt_W, filt_rowsum), vol_w, P)

        def density_filter(x, d):
            Wm, rs = d[2]
            return (Wm @ x) / (rs + 1e-12)

        def project(x, d):
            return gene.heaviside_projection(density_filter(x, d), beta=beta)

        def compliance(x, d):
            s, u_init, pre_d = d[0], d[1], d[4]
            tp = fe.TracedParams(volume_vars=(project(x, d),), surface_vars=())
            return compliance_fn(solver(tp, u_init, traced_structure=s, preconditioner=pre_d))

        def displacement(x, d):
            s, u_init, pre_d = d[0], d[1], d[4]
            tp = fe.TracedParams(volume_vars=(project(x, d),), surface_vars=())
            return jnp.asarray(solver(tp, u_init, traced_structure=s, preconditioner=pre_d))

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
                    dof=int(u0.shape[0]), c0=c0, filter_nnz=int(filt_W.nse),
                    filter_radius=radius)
        fields = dict(filtered=density_filter, projected=project, displacement=displacement)

        # AMG rebuild: assemble the BC-applied Jacobian at the current design and
        # rebuild the hierarchy from it on the host (P.rebuild), then swap the new
        # preconditioner into the data pytree. The forward and adjoint solves
        # share one preconditioner, so both pick up the new hierarchy.
        J_csr = create_J_bc_csr_parametric(problem, symmetric=True)
        jacobian = jax.jit(lambda x, d: J_csr(
            d[1], fe.TracedParams(volume_vars=(project(x, d),), surface_vars=()), bc, d[0]))

        def rebuild_amg(x, d):
            P_new = d[4].rebuild(jacobian(x, d))
            return tuple(d[:4]) + (P_new,)

        info["rebuild"] = rebuild_amg if args.solver == "amg" else None

        return opt, x0, data, fields, info

    def record(mode, mesh_size, info, compile_s, times, *, evals=float("nan"),
               mean_inner=float("nan"), objective=float("nan"), compliance=float("nan"),
               volume=float("nan"), kkt=float("nan"), cmb=float("nan"),
               rebuild_every=0, n_rebuilds=0, rebuild_s=0.0, recompile_s=0.0, note=""):
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
            volume=volume, kkt_norm=kkt, const_mb=cmb,
            filter_radius=info.get("filter_radius", ""), filter_nnz=info.get("filter_nnz", ""),
            rebuild_every=rebuild_every, n_rebuilds=n_rebuilds, rebuild_s=rebuild_s,
            recompile_s=recompile_s, tag=args.tag, note=note)
        append_row(out_path, row)
        cs = f"{compile_s:7.2f}s" if compile_s == compile_s else "   -   "
        dof_s = f"{row['dof']:>9,}" if row["dof"] != "" else f"{'-':>9}"
        print(f"  {mode:5s} dof={dof_s} compile={cs}  total={mean:8.3f}s  "
              f"per-iter={row['per_iter_s']:7.4f}s  evals={evals:>5}  inner={mean_inner:5.2f}  "
              f"obj={objective:.5f}  vol={volume:.4f}  const={cmb:6.2f}MB"
              + (f"  rebuilds={n_rebuilds} ({rebuild_s:.1f}s, recompile {recompile_s:.1f}s)"
                 if n_rebuilds else "")
              + (f"  ({note})" if note else ""))
        return row

    finals = {}
    for target in args.dofs:
        nx, ny, nz, _ = resolution_for_dof(target, (L, W, H))
        mesh_size = L / nx
        print(f"target={target:,} -> nx={nx} ny={ny} nz={nz}  h={mesh_size:.4g}")
        try:
            opt, x0, data, fields, info = build((nx, ny, nz))
        except Exception as e:
            print(f"  build failed: {type(e).__name__}: {e}")
            record("build", mesh_size, {}, float("nan"), [], note=type(e).__name__)
            jax.clear_caches()
            continue
        print(f"  nodes={info['n_nodes']:,}  cells={info['n_cells']:,}  dof={info['dof']:,}  "
              f"filter r={info['filter_radius']:.3g} nnz={info['filter_nnz']:,}  "
              f"c0={info['c0']:.4e}")

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

                # One executable per data layout: a rebuilt AMG hierarchy only forces
                # a recompile when its level shapes change.
                def layout(d):
                    leaves, tdef = jax.tree_util.tree_flatten(d)
                    return tdef, tuple((l.shape, l.dtype) for l in leaves)
                step_execs = {layout(data): step_c}

                every = args.amg_rebuild_every if info["rebuild"] is not None else 0
                times = []
                for _ in range(args.repeats):
                    t0 = perf_counter()
                    d = data
                    state = init_c(x0, d)
                    inner_total = 0
                    n_rebuilds, rebuild_s, recompile_s = 0, 0.0, 0.0
                    for k in range(args.iters):
                        if every and k and k % every == 0:
                            t1 = perf_counter()
                            d = info["rebuild"](state.x, d)
                            rebuild_s += perf_counter() - t1
                            n_rebuilds += 1
                            key = layout(d)
                            if key not in step_execs:
                                c, step_execs[key] = aot(step_fn, state, d)
                                recompile_s += c
                        state = step_execs[layout(d)](state, d)
                        inner_total += int(state.inner_iterations)   # host read-back each step
                    block(state.x)
                    times.append(perf_counter() - t0)
                stats, x_final = summarize(state, inner_total)
                record("run", mesh_size, info, c_init + c_step, times, cmb=cmb,
                       rebuild_every=every, n_rebuilds=n_rebuilds, rebuild_s=rebuild_s,
                       recompile_s=recompile_s, **stats)
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
                rb = f"_rb{args.amg_rebuild_every}" if key[1] == "run" and \
                    args.solver == "amg" and args.amg_rebuild_every else ""
                path = os.path.join(args.save_dir,
                                    f"topopt_{args.solver}_dof{info['dof']}_{key[1]}{rb}.vtu")
                x_final = finals[key][1]
                rho_f = jax.device_get(jax.jit(fields["filtered"])(x_final, data))
                rho_p = jax.device_get(jax.jit(fields["projected"])(x_final, data))
                u = jax.device_get(jax.jit(fields["displacement"])(x_final, data))
                cells = info["mesh"].cells
                fe.utils.save_sol(
                    mesh=info["mesh"], sol_file=path,
                    point_infos=[("density", rho_p), ("density_filtered", rho_f),
                                 ("design", jax.device_get(x_final)),
                                 ("displacement", u.reshape(info["n_nodes"], 3))],
                    cell_infos=[("density", rho_p[cells].mean(axis=1))])
                print(f"  wrote {path}")

        jax.clear_caches()
        print()

    print(f"Rows appended to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
