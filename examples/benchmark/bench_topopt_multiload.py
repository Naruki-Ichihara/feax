"""Benchmark: multi-load-case 3-D topology optimization, all cases in ONE solve.

Same cantilever and optimization as ``bench_topopt_gcmma.py`` (the 8 x 2 x 2 HEX8
box of ``bench_linear_elasticity.py``, SIMP, distance filter, Heaviside
projection, volume constraint, in-house GCMMA), but the structure has to carry
``K`` load cases at once:

    minimise   (1/K) sum_k c_k(rho) / c_mean(rho0)
    subject to V(rho_f) / V* - 1 <= 0,      0 <= rho <= 1

Load case ``k`` is a uniform traction on a small PATCH of the top face ``z = H``
at the free end: ``x in [L - 0.2, L]``, ``y in [W/2 - 0.1, W/2 + 0.1]`` (0.2 x 0.2,
centred in y). It points in the y-z plane (perpendicular to the beam axis) at
angle ``theta_k = 180 * k / K`` degrees from vertical DOWN ``-z``:
``t_k = (F / A) * (0, sin theta_k, -cos theta_k)``, with ``A`` the patch area
actually covered by boundary faces and the resultant ``F = 4e-3`` (the force the
whole tip face carried in ``bench_topopt_gcmma.py``). ``K = 1`` is a single
downward point-like load. (Compliance is unchanged by ``t -> -t``, so half a turn
covers every direction.)

Why it is cheap: the stiffness matrix depends on the design only, not on the
load. The K forward solves run as ``jax.vmap`` over the load (vmap-rhs), so the
direct solver assembles and factorizes ONCE per design and back-substitutes K
right-hand sides. Compliance is self-adjoint, so the gradient needs no adjoint
solves: ``dc_k/drho = -u_k^T (dK/drho) u_k``, summed over the cases in chunks of
``--grad-chunk`` (autodiff through the solver would instead batch that
element-level product over all K cases at once: 65 GB for K = 400 at 500k DOF).
The cost per optimization iteration therefore grows only mildly with K
(500k DOF on A100: 5.6 s per evaluation at K = 1, 7.6 s at K = 400).

Compliance of case k is ``F_k . u_k``. The load vector is linear in the traction,
so ``F_k = sum_j t_kj F_j`` with the three unit-traction load vectors ``F_j``
assembled once up front and passed as data.

As in ``bench_topopt_gcmma.py`` everything mesh-sized (TracedStructure, filter
weights, volume weights, unit load vectors, the traction batch) is ONE pytree jit
argument, so nothing is baked into the executables (``const_mb`` checks it).
The optimization runs as a host loop over a compiled ``GCMMA.step``.

Results are appended to ``bench_topopt_multiload.csv`` next to this script, and
the final structure of each K is written as VTU (``out_topopt_multiload/``):
point fields ``density`` (projected), ``density_filtered``, ``design`` and the
displacement of up to ``--vtu-cases`` load cases (``u_k<k>_<deg>deg``), plus the
cell field ``density``.

Run:
    python bench_topopt_multiload.py                                # 500k DOF, K = 400, 20 iterations
    python bench_topopt_multiload.py --n-loads 1 4 16 64            # sweep K on the same structure
"""
import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from time import perf_counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench_linear_elasticity import device_info, resolution_for_dof  # noqa: E402  (no jax at import)
from bench_topopt_gcmma import filter_weights  # noqa: E402  (no jax at import)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Multi-load-case topology optimization (direct solver, vmap over loads).")
    p.add_argument("--dofs", type=int, default=500_000,
                   help="Target DOF count (grid rule of bench_linear_elasticity.py).")
    p.add_argument("--n-loads", type=int, nargs="+", default=[400],
                   help="Numbers of load cases K to run (one optimization each).")
    p.add_argument("--iters", type=int, default=20, help="Outer GCMMA iterations (fixed).")
    p.add_argument("--max-inner", type=int, default=15, help="GCMMA inner-loop cap.")
    p.add_argument("--vf", type=float, default=0.4, help="Target volume fraction.")
    p.add_argument("--penal", type=float, default=3.0, help="SIMP penalty p.")
    p.add_argument("--filter-radius-cells", type=float, default=3.0,
                   help="Density filter radius in element sizes.")
    p.add_argument("--beta", type=float, default=4.0, help="Heaviside projection sharpness.")
    p.add_argument("--save-dir", default=os.path.join(HERE, "out_topopt_multiload"),
                   help="Write the final structure of each K as VTU here.")
    p.add_argument("--no-save", dest="save_dir", action="store_const", const=None,
                   help="Do not write VTU files.")
    p.add_argument("--grad-chunk", type=int, default=25,
                   help="Load cases per chunk in the sensitivity sum (bounds its memory).")
    p.add_argument("--vtu-cases", type=int, default=8,
                   help="Write the displacement of at most this many (evenly spaced) cases.")
    p.add_argument("--output", default=None,
                   help="CSV path (default: bench_topopt_multiload.csv next to this script).")
    p.add_argument("--tag", default="", help="Free-form label written to every row.")
    p.add_argument("--device-name", default=None, help="Override the detected device name.")
    return p.parse_args(argv)


L, W, H = 8.0, 2.0, 2.0
E0 = 70.0e3
NU = 0.30
SIMP_EPSILON = 1e-3
TOTAL_FORCE = 4e-3          # load resultant per case (= 1e-3 x the 2 x 2 tip face)
PATCH = 0.2                 # loaded patch on the top face: PATCH (x) x PATCH (y)

CSV_COLUMNS = [
    "timestamp", "device", "jax_version", "feax_version", "dof", "n_nodes", "n_cells",
    "n_loads", "iters", "compile_s", "total_s", "per_iter_s", "evals", "per_eval_s",
    "mean_inner", "objective", "mean_compliance", "max_compliance", "volume", "kkt_norm",
    "const_mb", "tag", "note",
]


def append_row(out_path, row):
    """Append one row (header only for a new file) and fsync it."""
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
    os.environ["FEAX_X64"] = "1"
    # One live factorization per design is all the optimization needs; a deeper
    # spineax factor cache only pins GPU memory with stale factors.
    os.environ.setdefault("SPINEAX_FACTOR_CACHE", "2")

    import jax
    import jax.numpy as jnp
    import numpy as onp
    import feax as fe
    import feax.gene as gene
    from feax.assembler import create_res_bc_parametric
    from feax.experimental.mma import GCMMA

    try:
        from importlib.metadata import version as _pkg_version
        feax_version = _pkg_version("feax")
    except Exception:
        feax_version = getattr(fe, "__version__", "?")

    _, device = device_info(jax, args.device_name)
    out_path = args.output or os.path.join(HERE, "bench_topopt_multiload.csv")
    print(f"device={device}  jax={jax.__version__}  feax={feax_version}  solver=direct")
    print(f"box={L:g}x{W:g}x{H:g}  vf={args.vf}  p={args.penal}  "
          f"r={args.filter_radius_cells:g}h  beta={args.beta}  iters={args.iters}  "
          f"n_loads={args.n_loads}  output={out_path}\n")

    penal = float(args.penal)

    class IsotropicSIMP(fe.Problem):
        """Linear isotropic elasticity, SIMP nodal density, vector tip traction."""

        def get_tensor_map(self):
            def stress(u_grad, rho):
                E = E0 * (SIMP_EPSILON + (1.0 - SIMP_EPSILON) * rho ** penal)
                mu = E / (2.0 * (1.0 + NU))
                lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
                eps = 0.5 * (u_grad + u_grad.T)
                return lam * jnp.trace(eps) * jnp.eye(self.dim) + 2.0 * mu * eps
            return stress

        def get_surface_maps(self):
            # Three surface variables = the traction vector, so each load case
            # can point in its own direction. feax adds the surface map to the
            # residual with a + sign, so the map returns MINUS the applied traction.
            def traction(u, x, tx, ty, tz):
                return jnp.array([tx, ty, tz])
            return [traction]

    # ---- one structure, shared by every K --------------------------------
    nx, ny, nz, _ = resolution_for_dof(args.dofs, (L, W, H))
    spacing = (L / nx, W / ny, H / nz)
    mesh = fe.StructuredGrid((nx, ny, nz), spacing=spacing).to_mesh()
    left = lambda pt: jnp.isclose(pt[0], 0.0, 1e-5)
    # Loaded patch: top face z = H, last PATCH of the span, PATCH wide centred in y.
    patch = lambda pt: (jnp.isclose(pt[2], H, 1e-5)
                        & (pt[0] >= L - PATCH - 1e-6)
                        & (jnp.abs(pt[1] - 0.5 * W) <= 0.5 * PATCH + 1e-6))
    problem = IsotropicSIMP(mesh=mesh, vec=3, dim=3, ele_type="HEX8", location_fns=(patch,))
    bc = fe.DirichletBCConfig([
        fe.DirichletBCSpec(location=left, component="all", value=0.0),
    ]).create_bc(problem)
    ts = fe.TracedStructure.from_problem(problem, free_scratch=False)
    n_nodes = mesh.points.shape[0]
    unit = fe.TracedParams.create_uniform_surface_var(problem, 1.0)       # (faces, quads)
    rho_vf = fe.TracedParams.create_node_var(problem, args.vf)

    def tparams(rho, tx, ty, tz):
        return fe.TracedParams(volume_vars=(rho,), surface_vars=[(tx, ty, tz)])

    solver = fe.create_solver(
        problem, bc, solver_options=fe.DirectSolverOptions(reuse_factorization=True),
        linear=True, traced_params=tparams(rho_vf, 0 * unit, 0 * unit, unit),
        traced_structure=ts, return_solution=False)
    u0 = fe.zero_like_initial_guess(problem, bc)

    # Load vectors F_j = -residual(u = 0) for the surface-map value e_j (rows of
    # Dirichlet DOFs are 0 there), i.e. for the applied traction -e_j. The load
    # does not depend on the design.
    res_bc = create_res_bc_parametric(problem)
    res_fn = jax.jit(res_bc)
    F_unit = jnp.stack([
        -res_fn(u0, tparams(rho_vf, *(unit * float(j == k) for k in range(3))), bc, ts)
        for j in range(3)])                                                # (3, ndof)
    patch_area = -float(jnp.sum(F_unit[2]))      # resultant of applied traction -e_z = area
    assert patch_area > 0, "load patch selected no boundary faces"
    traction_mag = TOTAL_FORCE / patch_area

    radius = args.filter_radius_cells * spacing[0]
    filt_W, filt_rowsum = filter_weights(mesh.points, radius)
    vol_w = jax.grad(gene.create_volume_fn(problem))(jnp.zeros((n_nodes,)))
    beta, vf = float(args.beta), float(args.vf)
    dof = int(u0.shape[0])
    print(f"nx={nx} ny={ny} nz={nz}  nodes={n_nodes:,}  cells={mesh.cells.shape[0]:,}  "
          f"dof={dof:,}  filter r={radius:.3g} nnz={int(filt_W.nse):,}")
    print(f"load patch: {len(problem.boundary_inds_list[0])} faces, area={patch_area:.4g}  "
          f"traction={traction_mag:.4g} (resultant {TOTAL_FORCE:g})\n")

    def density_filter(x, d):
        Wm, rs = d["filter"]
        return (Wm @ x) / (rs + 1e-12)

    def project(x, d):
        return gene.heaviside_projection(density_filter(x, d), beta=beta)

    def solve_rho(rho, d):
        """Displacements of all K load cases, (K, ndof): ONE vmap over the load,
        so the direct solver factorizes once and back-substitutes K RHS."""
        tx, ty, tz = d["loads"]
        one = lambda a, b, c: solver(tparams(rho, a, b, c), d["u0"], traced_structure=d["ts"])
        return jax.vmap(one)(tx, ty, tz)

    def solve_cases(x, d):
        return solve_rho(project(x, d), d)

    def compliances_rho(rho, d):
        U = solve_rho(rho, d)
        F = d["dirs"] @ d["F_unit"]                                       # (K, ndof)
        return jnp.sum(F * U, axis=1), U                                  # c_k = F_k . u_k

    def compliances(x, d):
        return compliances_rho(project(x, d), d)[0]

    # Mean compliance with a hand-written sensitivity. Compliance is self-adjoint,
    # so dc_k/drho = -u_k^T (dK/drho) u_k = -d/drho [u_k . res(u_k; rho)] at fixed
    # u_k: no adjoint solves at all. Autodiff through the solver would instead
    # evaluate that element-level product for all K cases in one batch (~160 MB
    # per case at 500k DOF -> 65 GB for K = 400); here the K cases are summed in
    # chunks of --grad-chunk with lax.map, so memory is bounded by one chunk.
    chunk = max(1, args.grad_chunk)

    @jax.custom_vjp
    def mean_compliance(rho, d):
        return jnp.mean(compliances_rho(rho, d)[0])

    def mean_compliance_fwd(rho, d):
        c, U = compliances_rho(rho, d)
        return jnp.mean(c), (rho, U, d)

    def mean_compliance_bwd(res, g):
        rho, U, d = res
        n = U.shape[0]
        pad = (-n) % chunk                       # zero displacements add nothing
        Uc = jnp.pad(U, ((0, pad), (0, 0))).reshape(-1, chunk, U.shape[1])
        zero = jnp.zeros_like(d["loads"][0][0])  # traction does not depend on rho

        def energy(r, u):                        # u . res(u; r) = u^T K(r) u (minus a rho-free term)
            return jnp.dot(u, res_bc(u, tparams(r, zero, zero, zero), bc, d["ts"]))

        def chunk_grad(Uk):
            return jax.vmap(jax.grad(energy), in_axes=(None, 0))(rho, Uk).sum(0)

        drho = jax.lax.map(chunk_grad, Uc).sum(0)
        return -g * drho / n, None                # d is data, not a design variable

    mean_compliance.defvjp(mean_compliance_fwd, mean_compliance_bwd)

    def volume(x, d):
        return jnp.dot(d["vol_w"], density_filter(x, d))

    def const_mb(fn, *call_args):
        jaxpr = jax.make_jaxpr(fn)(*call_args)
        return sum(getattr(c, "nbytes", 0) for c in jaxpr.consts) / 1e6

    def block(x):
        return jax.block_until_ready(x)

    def aot(jitted, *call_args):
        t0 = perf_counter()
        compiled = jitted.lower(*call_args).compile()
        return perf_counter() - t0, compiled

    for K in args.n_loads:
        theta = onp.pi * onp.arange(K) / K                                # from -z, in [0, pi)
        t_applied = onp.stack([onp.zeros(K), onp.sin(theta), -onp.cos(theta)], 1) * traction_mag
        dirs = -t_applied                         # surface-map values (see get_surface_maps)
        loads = tuple(jnp.asarray(dirs[:, j])[:, None, None] * unit[None] for j in range(3))
        data = dict(ts=ts, u0=u0, filter=(filt_W, filt_rowsum), vol_w=vol_w,
                    F_unit=F_unit, dirs=jnp.asarray(dirs), loads=loads)
        print(f"K={K}  load angles from -z (deg): "
              f"{onp.round(onp.degrees(theta), 1).tolist() if K <= 16 else f'0, {180 / K:g}, ... {180 - 180 / K:g}'}")
        x0 = jnp.full((n_nodes,), vf)
        try:
            c_init = jax.jit(compliances)(x0, data)
            c0 = float(jnp.mean(c_init))                                  # scaling, untimed

            def value_and_grad(x, d):
                f, df = jax.value_and_grad(lambda y: mean_compliance(project(y, d), d) / c0)(x)
                v, dv = jax.value_and_grad(volume)(x, d)
                return f, df, jnp.reshape(v / vf - 1.0, (1,)), jnp.reshape(dv / vf, (1, -1))

            opt = GCMMA(value_and_grad, 0.0, 1.0, n_constraints=1, max_inner=args.max_inner)
            c_i, init_c = aot(jax.jit(opt.init), x0, data)
            state = block(init_c(x0, data))
            c_s, step_c = aot(jax.jit(opt.step), state, data)
            cmb = const_mb(opt.step, state, data)

            t0 = perf_counter()
            state = init_c(x0, data)
            inner_total = 0
            for k in range(args.iters):
                state = step_c(state, data)
                inner_total += int(state.inner_iterations)
                if (k + 1) % 10 == 0:
                    print(f"    iter {k + 1:3d}  obj={float(state.f0):.5f}  "
                          f"vol={float(state.fval[0] + 1.0) * vf:.4f}  "
                          f"t={perf_counter() - t0:7.1f}s")
            block(state.x)
            total = perf_counter() - t0

            c_fin = jax.jit(compliances)(state.x, data)
            evals = 1 + args.iters + inner_total
            row = dict(
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                device=device, jax_version=jax.__version__, feax_version=feax_version,
                dof=dof, n_nodes=n_nodes, n_cells=mesh.cells.shape[0], n_loads=K,
                iters=args.iters, compile_s=c_i + c_s, total_s=total,
                per_iter_s=total / args.iters, evals=evals, per_eval_s=total / evals,
                mean_inner=inner_total / args.iters, objective=float(state.f0),
                mean_compliance=float(jnp.mean(c_fin)), max_compliance=float(jnp.max(c_fin)),
                volume=float(state.fval[0] + 1.0) * vf, kkt_norm=float(state.kkt_norm),
                const_mb=cmb, tag=args.tag, note="")
            append_row(out_path, row)
            print(f"  K={K:>3}  compile={c_i + c_s:6.2f}s  total={total:8.2f}s  "
                  f"per-iter={total / args.iters:7.3f}s  per-eval={total / evals:7.3f}s  "
                  f"evals={evals}  obj={row['objective']:.5f}  vol={row['volume']:.4f}  "
                  f"const={cmb:.2f}MB")

            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                path = os.path.join(args.save_dir, f"topopt_multiload_dof{dof}_K{K}.vtu")
                x_f = state.x
                rho_p = onp.asarray(jax.jit(project)(x_f, data))
                rho_f = onp.asarray(jax.jit(density_filter)(x_f, data))
                U = onp.asarray(jax.jit(solve_cases)(x_f, data))
                pick = sorted(set(onp.linspace(0, K - 1, min(K, args.vtu_cases)).round().astype(int)))
                fe.utils.save_sol(
                    mesh=mesh, sol_file=path,
                    point_infos=[("density", rho_p), ("density_filtered", rho_f),
                                 ("design", onp.asarray(x_f))]
                    + [(f"u_k{i:02d}_{onp.degrees(theta[i]):05.1f}deg", U[i].reshape(n_nodes, 3))
                       for i in pick],
                    cell_infos=[("density", rho_p[onp.asarray(mesh.cells)].mean(axis=1))])
                print(f"  wrote {path}")
        except Exception as e:
            print(f"  K={K} failed: {type(e).__name__}: {str(e).splitlines()[0][:300]}")
            append_row(out_path, dict(
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                device=device, jax_version=jax.__version__, feax_version=feax_version,
                dof=dof, n_nodes=n_nodes, n_cells=mesh.cells.shape[0], n_loads=K,
                iters=args.iters, tag=args.tag, note=type(e).__name__))
        jax.clear_caches()
        print()

    print(f"Rows appended to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
