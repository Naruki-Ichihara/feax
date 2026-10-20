"""Benchmark: B independent topology optimizations, one shared factorization per step.

``bench_topopt_multiload.py`` optimizes ONE design for many load cases, so all
solves share K(rho). Here every load case gets its OWN design ``rho_i`` (B
independent topologies), so the matrices differ per case -- the ``vmap-lhs``
situation, where the direct solver would factorize B matrices per evaluation.

Both modes batch the one linear solver with ``fe.vmap`` over the per-case
modulus fields and loads (an lhs batch: the modulus changes the operator).
Shared mode factorizes once per evaluation --

    E_i = SIMP(rho_i)                    per-case nodal modulus
    K_0 = K(mean_i E_i) = mean_i K_i     factorized ONCE (fe.shared_factor)
    K_i u_i = f_i                        batched CG preconditioned by K_0^{-1}

(``bench_shared_factor_pcg.py`` measures that solver alone.) Direct mode
(``options=DirectSolverOptions(), chunk=1``) solves each case with its own
factorization, one after another, as the reference.

Compliance is self-adjoint, so the gradient needs no adjoint solves:
``dc_i/drho_i = -u_i^T (dK/drho_i) u_i``.

The B designs are stacked into one GCMMA variable vector: the objective is the
sum of the per-case normalized compliances and there is one volume constraint
per case, so the problem separates into B independent optimizations (only
GCMMA's inner-loop acceptance test is taken over the sum).

Cantilever of ``bench_linear_elasticity.py`` (8 x 2 x 2 HEX8 grid, clamp at
x = 0). Case ``k`` loads the 0.2 x 0.2 top-face patch at the free end (as in
``bench_topopt_multiload.py``), rotated ``180 k / B`` degrees from down in the
y-z plane. The final designs are written as VTU (``out_topopt_batched/``), rows
append to ``bench_topopt_batched.csv``.

Run:
    python bench_topopt_batched_shared.py                       # 100k DOF, B = 10, both modes
    python bench_topopt_batched_shared.py --modes shared --dofs 500000
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
        description="B independent topology optimizations with one shared factorization.")
    p.add_argument("--dofs", type=int, default=100_000, help="Target DOF count.")
    p.add_argument("--batch", type=int, default=10, help="Number of independent designs B.")
    p.add_argument("--modes", nargs="+", choices=["shared", "direct"],
                   default=["shared", "direct"],
                   help="shared: one factorization of mean_i K_i + batched PCG; "
                        "direct: one factorization per case (reference).")
    p.add_argument("--iters", type=int, default=20, help="Outer GCMMA iterations (fixed).")
    p.add_argument("--max-inner", type=int, default=15, help="GCMMA inner-loop cap.")
    p.add_argument("--vf", type=float, default=0.4, help="Target volume fraction per design.")
    p.add_argument("--penal", type=float, default=3.0, help="SIMP penalty p.")
    p.add_argument("--filter-radius-cells", type=float, default=3.0,
                   help="Density filter radius in element sizes.")
    p.add_argument("--beta", type=float, default=4.0, help="Heaviside projection sharpness.")
    p.add_argument("--tol", type=float, default=1e-8, help="PCG relative residual tolerance.")
    p.add_argument("--maxiter", type=int, default=1000, help="PCG iteration cap.")
    p.add_argument("--save-dir", default=os.path.join(HERE, "out_topopt_batched"),
                   help="Write the B final designs as VTU here.")
    p.add_argument("--no-save", dest="save_dir", action="store_const", const=None)
    p.add_argument("--output", default=None,
                   help="CSV path (default: bench_topopt_batched.csv next to this script).")
    p.add_argument("--tag", default="", help="Free-form label written to every row.")
    p.add_argument("--device-name", default=None, help="Override the detected device name.")
    return p.parse_args(argv)


L, W, H = 8.0, 2.0, 2.0
E0 = 70.0e3
NU = 0.30
SIMP_EPSILON = 1e-3
TOTAL_FORCE = 4e-3
PATCH = 0.2

CSV_COLUMNS = [
    "timestamp", "device", "jax_version", "mode", "dof", "batch", "iters", "compile_s",
    "total_s", "per_iter_s", "evals", "per_eval_s", "pcg_iters_mean", "pcg_iters_max",
    "objective", "compliances", "volumes", "tag", "note",
]


def append_row(out_path, row):
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
    os.environ.setdefault("SPINEAX_FACTOR_CACHE", "2")

    import jax
    import jax.numpy as jnp
    import numpy as onp
    import feax as fe
    import feax.gene as gene
    from feax.assembler import create_res_bc_parametric
    from feax.experimental.mma import GCMMA

    _, device = device_info(jax, args.device_name)
    out_path = args.output or os.path.join(HERE, "bench_topopt_batched.csv")
    B = args.batch
    print(f"device={device}  jax={jax.__version__}  B={B}  modes={args.modes}  "
          f"iters={args.iters}  output={out_path}")

    class LinearElasticity(fe.Problem):
        """Linear elasticity with a nodal Young's modulus field and a vector traction."""

        def get_tensor_map(self):
            def stress(u_grad, E):
                mu = E / (2.0 * (1.0 + NU))
                lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
                eps = 0.5 * (u_grad + u_grad.T)
                return lam * jnp.trace(eps) * jnp.eye(self.dim) + 2.0 * mu * eps
            return stress

        def get_surface_maps(self):
            # feax adds the surface map to the residual with a + sign, so the map
            # returns MINUS the applied traction.
            def traction(u, x, tx, ty, tz):
                return jnp.array([tx, ty, tz])
            return [traction]

    nx, ny, nz, _ = resolution_for_dof(args.dofs, (L, W, H))
    spacing = (L / nx, W / ny, H / nz)
    mesh = fe.StructuredGrid((nx, ny, nz), spacing=spacing).to_mesh()
    patch = lambda pt: (jnp.isclose(pt[2], H, 1e-5)
                        & (pt[0] >= L - PATCH - 1e-6)
                        & (jnp.abs(pt[1] - 0.5 * W) <= 0.5 * PATCH + 1e-6))
    problem = LinearElasticity(mesh=mesh, vec=3, dim=3, ele_type="HEX8", location_fns=(patch,))
    bc = fe.DirichletBCConfig([fe.DirichletBCSpec(
        location=lambda pt: jnp.isclose(pt[0], 0.0, 1e-5),
        component="all", value=0.0)]).create_bc(problem)
    ts = fe.TracedStructure.from_problem(problem, free_scratch=False)
    u0 = fe.zero_like_initial_guess(problem, bc)
    n_nodes, n = mesh.points.shape[0], u0.shape[0]
    unit = fe.TracedParams.create_uniform_surface_var(problem, 1.0)
    zero = 0.0 * unit
    E_vf = fe.TracedParams.create_node_var(problem, E0)

    def tparams(E, tx=zero, ty=zero, tz=zero):
        return fe.TracedParams(volume_vars=(E,), surface_vars=[(tx, ty, tz)])

    # Per-case load vectors (design-independent): F_j for surface-map value e_j.
    res_bc = create_res_bc_parametric(problem)
    res_fn = jax.jit(res_bc)
    F_unit = jnp.stack([-res_fn(u0, tparams(E_vf, *(unit * float(j == k) for k in range(3))), bc, ts)
                        for j in range(3)])
    area = -float(jnp.sum(F_unit[2]))
    theta = onp.pi * onp.arange(B) / B
    t_applied = onp.stack([onp.zeros(B), onp.sin(theta), -onp.cos(theta)], 1) * TOTAL_FORCE / area
    map_vals = jnp.asarray(-t_applied)                                   # (B, 3)
    F = map_vals @ F_unit                                                # (B, n)
    loads = tuple(map_vals[:, j][:, None, None] * unit[None] for j in range(3))

    # The filter / volume operators.
    radius = args.filter_radius_cells * spacing[0]
    filt_W, filt_rowsum = filter_weights(mesh.points, radius)
    vol_w = jax.grad(gene.create_volume_fn(problem))(jnp.zeros((n_nodes,)))
    vf, beta, penal = float(args.vf), float(args.beta), float(args.penal)
    print(f"nx={nx} ny={ny} nz={nz}  nodes={n_nodes:,}  dof={n:,}  design vars={B * n_nodes:,}  "
          f"load angles from -z: {onp.round(onp.degrees(theta), 1).tolist()}\n")

    data = dict(ts=ts, u0=u0, F=F, loads=loads, filter=(filt_W, filt_rowsum), vol_w=vol_w)

    def project(X, d):                                                   # (B, n_nodes)
        Wm, rs = d["filter"]
        rho_f = (Wm @ X.T).T / (rs + 1e-12)
        return gene.heaviside_projection(rho_f, beta=beta), rho_f

    def simp(rho):
        return E0 * (SIMP_EPSILON + (1.0 - SIMP_EPSILON) * rho ** penal)

    # ---------------- solvers: (B, n_nodes) moduli -> (B, n) displacements ----
    # One linear solver, batched by fe.vmap over (E_i, loads_i); the modulus
    # changes the operator, so both modes are lhs batches of the same solver.
    direct_solver = fe.create_solver(
        problem, bc, solver_options=fe.DirectSolverOptions(), linear=True,
        traced_params=tparams(E_vf, *(l[0] for l in loads)), traced_structure=ts,
        return_solution=False)
    batch_axes = fe.TracedParams(volume_vars=(0,), surface_vars=[(0, 0, 0)])

    # shared: one factorization of K(mean E) = mean_i K_i, batched PCG for the K_i
    shared_batched = fe.vmap(
        direct_solver, in_axes=batch_axes, batch="lhs", return_info=True,
        options=fe.KrylovSolverOptions(solver="cg", tol=args.tol, atol=0.0, maxiter=args.maxiter),
        preconditioner=fe.shared_factor())
    # direct: each case factorized and solved on its own (chunk=1: one at a time)
    direct_batched = fe.vmap(
        direct_solver, in_axes=batch_axes, batch="lhs", return_info=True,
        options=fe.DirectSolverOptions(), chunk=1)

    def make_solve(batched):
        def solve(Es, d):
            U, info = batched(tparams(Es, *d["loads"]), d["u0"], d["ts"])
            return U, info.iterations
        return solve

    solve_shared, solve_direct = make_solve(shared_batched), make_solve(direct_batched)

    # ---------------- compliance with the self-adjoint sensitivity -------------
    def make_compliances(solve):
        @jax.custom_vjp
        def comp(Es, d):
            U, its = solve(Es, d)
            return jnp.sum(d["F"] * U, 1), its

        def fwd(Es, d):
            U, its = solve(Es, d)
            return (jnp.sum(d["F"] * U, 1), its), (Es, U, d)

        def bwd(res, g):
            Es, U, d = res
            gc, _ = g

            def energy(E, u):                                            # u^T K(E) u
                return jnp.dot(u, res_bc(u, tparams(E), bc, d["ts"]))

            dE = jax.vmap(jax.grad(energy))(Es, U)                       # (B, n_nodes)
            return -gc[:, None] * dE, None

        comp.defvjp(fwd, bwd)
        return comp

    def block(x):
        return jax.block_until_ready(x)

    def aot(jitted, *a):
        t0 = perf_counter()
        c = jitted.lower(*a).compile()
        return perf_counter() - t0, c

    X0 = jnp.full((B, n_nodes), vf)
    for mode in args.modes:
        comp = make_compliances(solve_shared if mode == "shared" else solve_direct)
        print(f"[{mode}]")
        row = dict(timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   device=device, jax_version=jax.__version__, mode=mode, dof=n, batch=B,
                   iters=args.iters, tag=args.tag)
        try:
            c0 = block(jax.jit(lambda X, d: comp(simp(project(X, d)[0]), d)[0])(X0, data))

            def value_and_grad(x, d):
                def obj(x):
                    c, its = comp(simp(project(x.reshape(B, -1), d)[0]), d)
                    return jnp.sum(c / c0), its
                (f, its), df = jax.value_and_grad(obj, has_aux=True)(x)

                def vols(x):
                    return (project(x.reshape(B, -1), d)[1] @ d["vol_w"]) / vf - 1.0
                g, dg = vols(x), jax.jacrev(vols)(x)                     # (B,), (B, B n)
                return f, df, g, dg, its

            opt = GCMMA(value_and_grad, 0.0, 1.0, n_constraints=B, max_inner=args.max_inner,
                        has_aux=True)
            x0 = X0.reshape(-1)
            c_i, init_c = aot(jax.jit(opt.init), x0, data)
            state = block(init_c(x0, data))
            c_s, step_c = aot(jax.jit(opt.step), state, data)

            t0 = perf_counter()
            state = init_c(x0, data)
            inner, pcg = 0, []
            for k in range(args.iters):
                state = step_c(state, data)
                inner += int(state.inner_iterations)
                pcg.append(onp.asarray(state.aux))
                if (k + 1) % 5 == 0:
                    msg = f"    iter {k + 1:3d}  obj={float(state.f0):.5f}  " \
                          f"vol=[{float(state.fval.min() + 1) * vf:.3f}, {float(state.fval.max() + 1) * vf:.3f}]"
                    if mode == "shared":
                        msg += f"  PCG iters (this step) mean={pcg[-1].mean():.0f} max={pcg[-1].max()}"
                    print(msg + f"  t={perf_counter() - t0:7.1f}s")
            block(state.x)
            total = perf_counter() - t0
            evals = 1 + args.iters + inner
            Xf = state.x.reshape(B, -1)
            cf = onp.asarray(jax.jit(lambda X, d: comp(simp(project(X, d)[0]), d)[0])(Xf, data))
            vols = onp.asarray(project(Xf, data)[1] @ vol_w) / 1.0
            pcg = onp.concatenate(pcg) if mode == "shared" else onp.zeros(1)
            row.update(compile_s=c_i + c_s, total_s=total, per_iter_s=total / args.iters,
                       evals=evals, per_eval_s=total / evals,
                       pcg_iters_mean=float(pcg.mean()) if mode == "shared" else "",
                       pcg_iters_max=int(pcg.max()) if mode == "shared" else "",
                       objective=float(state.f0),
                       compliances=" ".join(f"{c:.4e}" for c in cf),
                       volumes=" ".join(f"{v:.4f}" for v in vols), note="")
            print(f"  [{mode}] total={total:8.1f}s  per-eval={total / evals:6.2f}s  evals={evals}  "
                  f"obj={float(state.f0):.5f}"
                  + (f"  PCG iters mean={pcg.mean():.1f} max={pcg.max()}" if mode == "shared" else ""))
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                rho_p = onp.asarray(project(Xf, data)[0])
                path = os.path.join(args.save_dir, f"topopt_batched_{mode}_dof{n}_B{B}.vtu")
                cells = onp.asarray(mesh.cells)
                fe.utils.save_sol(
                    mesh=mesh, sol_file=path,
                    point_infos=[(f"density_k{i}_{onp.degrees(theta[i]):05.1f}deg", rho_p[i])
                                 for i in range(B)],
                    cell_infos=[(f"density_k{i}", rho_p[i][cells].mean(1)) for i in range(B)])
                print(f"  wrote {path}")
        except Exception as e:
            print(f"  [{mode}] failed: {type(e).__name__}: {str(e).splitlines()[0][:300]}")
            row["note"] = type(e).__name__
        append_row(out_path, row)
        jax.clear_caches()
        print()

    print(f"Rows appended to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
