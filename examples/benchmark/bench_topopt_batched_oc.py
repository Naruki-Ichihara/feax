"""B independent topology optimizations at scale: shared factorizations + OC updates.

Same problem as ``bench_topopt_batched_shared.py`` -- one design per load case,
the 0.2 x 0.2 top-face patch at the free end loaded at ``180 k / B`` degrees from
down in the y-z plane -- but built to reach B = 100 at 500k DOF:

* Solves: the B matrices differ (one design each), so they are solved with
  batched CG preconditioned by a SHARED factorization. Cases are processed in
  groups of ``--case-chunk`` (only one group's ``K_i`` is stored at a time).
  ``--k0 global`` factorizes ``K(mean_i E_i)`` once per evaluation for all B
  cases; ``--k0 group`` factorizes the mean of each group instead (B / chunk
  factorizations, but neighbouring load angles give similar designs, so the
  preconditioner stays close and CG needs far fewer iterations).
* Sensitivities: compliance is self-adjoint, ``dc_i/drho_i = -u_i^T dK u_i``.
* Updates: optimality criteria (OC) with a per-case bisection on the volume
  multiplier, vectorized over the B designs (GCMMA on one stacked vector would
  need a dense (B, B * nodes) constraint Jacobian: 13 GB at B = 100, 500k DOF).

Density filter (cone, ``--filter-radius-cells``), Heaviside projection, SIMP,
volume fraction on the filtered density -- as in the other topopt benchmarks.

Timing goes to CSV: one row per iteration (``bench_topopt_batched_oc_iters.csv``:
wall time of the evaluation and of the update, PCG iteration counts, mean/max
compliance ratio, volume range) and one summary row per run
(``bench_topopt_batched_oc.csv``). The final B designs are written as one VTU
(cell field ``density_kNN`` per case).

Run:
    python bench_topopt_batched_oc.py                                  # 500k DOF, B = 100
    python bench_topopt_batched_oc.py --dofs 20000 --batch 20 --iters 5
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
        description="B independent topology optimizations: shared factorizations + OC.")
    p.add_argument("--dofs", type=int, default=500_000, help="Target DOF count.")
    p.add_argument("--batch", type=int, default=100, help="Number of independent designs B.")
    p.add_argument("--case-chunk", type=int, default=10,
                   help="Cases assembled and solved together (one group's K_i in memory).")
    p.add_argument("--k0", choices=["global", "group"], default="group",
                   help="Preconditioner: factorize the mean matrix of all B cases once per "
                        "evaluation (global) or of each group (group).")
    p.add_argument("--iters", type=int, default=10, help="OC iterations.")
    p.add_argument("--move", type=float, default=0.2, help="OC move limit.")
    p.add_argument("--vf", type=float, default=0.4, help="Target volume fraction per design.")
    p.add_argument("--penal", type=float, default=3.0, help="SIMP penalty p.")
    p.add_argument("--filter-radius-cells", type=float, default=3.0,
                   help="Density filter radius in element sizes.")
    p.add_argument("--beta", type=float, default=4.0, help="Heaviside projection sharpness.")
    p.add_argument("--tol", type=float, default=1e-6, help="PCG relative residual tolerance.")
    p.add_argument("--maxiter", type=int, default=2000, help="PCG iteration cap.")
    p.add_argument("--save-dir", default=os.path.join(HERE, "out_topopt_batched"),
                   help="Write the B final designs as VTU here.")
    p.add_argument("--no-save", dest="save_dir", action="store_const", const=None)
    p.add_argument("--output", default=None,
                   help="Summary CSV (default: bench_topopt_batched_oc.csv next to this "
                        "script); per-iteration rows go to <name>_iters.csv.")
    p.add_argument("--tag", default="", help="Free-form label written to every row.")
    p.add_argument("--device-name", default=None, help="Override the detected device name.")
    return p.parse_args(argv)


L, W, H = 8.0, 2.0, 2.0
E0 = 70.0e3
NU = 0.30
SIMP_EPSILON = 1e-3
TOTAL_FORCE = 4e-3
PATCH = 0.2

SUMMARY_COLUMNS = [
    "timestamp", "device", "jax_version", "dof", "batch", "case_chunk", "k0", "iters",
    "setup_s", "compile_s", "total_s", "per_iter_s", "eval_s_mean", "update_s_mean",
    "pcg_iters_mean", "pcg_iters_max", "c_ratio_mean", "c_ratio_max", "vol_min", "vol_max",
    "tag", "note",
]
ITER_COLUMNS = [
    "timestamp", "dof", "batch", "k0", "iter", "eval_s", "update_s", "elapsed_s",
    "pcg_iters_mean", "pcg_iters_max", "c_ratio_mean", "c_ratio_max", "vol_min", "vol_max",
    "tag",
]


def append_row(path, columns, row):
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        if new_file:
            w.writeheader()
        w.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def main(argv=None):
    args = parse_args(argv)
    os.environ["FEAX_X64"] = "1"
    os.environ.setdefault("SPINEAX_FACTOR_CACHE", "2")
    t_setup0 = perf_counter()

    import jax
    import jax.numpy as jnp
    import numpy as onp
    import feax as fe
    import feax.gene as gene
    from feax.assembler import create_J_bc_csr_parametric, create_res_bc_parametric
    from feax.solvers._cudss_compat import factorize, solve_with

    _, device = device_info(jax, args.device_name)
    out_path = args.output or os.path.join(HERE, "bench_topopt_batched_oc.csv")
    iter_path = os.path.splitext(out_path)[0] + "_iters.csv"
    B = args.batch
    C = min(max(1, args.case_chunk), B)
    G = -(-B // C)                                                      # groups
    print(f"device={device}  jax={jax.__version__}  B={B}  chunk={C} ({G} groups)  "
          f"k0={args.k0}  iters={args.iters}\n  summary -> {out_path}\n  iters   -> {iter_path}")

    class LinearElasticity(fe.Problem):
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

    def tparams(E, tx=zero, ty=zero, tz=zero):
        return fe.TracedParams(volume_vars=(E,), surface_vars=[(tx, ty, tz)])

    E_ref = fe.TracedParams.create_node_var(problem, E0)
    res_bc = create_res_bc_parametric(problem)
    res_fn = jax.jit(res_bc)
    F_unit = jnp.stack([-res_fn(u0, tparams(E_ref, *(unit * float(j == k) for k in range(3))), bc, ts)
                        for j in range(3)])
    area = -float(jnp.sum(F_unit[2]))
    theta = onp.pi * onp.arange(B) / B
    t_applied = onp.stack([onp.zeros(B), onp.sin(theta), -onp.cos(theta)], 1) * TOTAL_FORCE / area
    F = jnp.asarray(-t_applied) @ F_unit                                # (B, n) load vectors

    J_fn = create_J_bc_csr_parametric(problem, symmetric=True)
    J_s = jax.jit(J_fn)(u0, tparams(E_ref), bc, ts)
    indptr, indices = J_s.indptr, J_s.indices
    rows = jnp.repeat(jnp.arange(n), jnp.diff(indptr), total_repeat_length=indices.shape[0])
    del J_s
    radius = args.filter_radius_cells * spacing[0]
    filt_W, filt_rowsum = filter_weights(mesh.points, radius)
    vol_w = jax.grad(gene.create_volume_fn(problem))(jnp.zeros((n_nodes,)))
    # Volume of the filtered density is linear in x: V(x) = (vol_w / rowsum) . (W x),
    # so dV/dx = W^T (vol_w / rowsum), the same for every case.
    dV = filt_W.T @ (vol_w / (filt_rowsum + 1e-12))
    vf, beta, penal = float(args.vf), float(args.beta), float(args.penal)
    print(f"nx={nx} ny={ny} nz={nz}  nodes={n_nodes:,}  dof={n:,}  "
          f"design vars={B * n_nodes:,}  filter nnz={int(filt_W.nse):,}")

    # Padding the last group repeats its final case; padded results are dropped.
    idx = onp.arange(G * C).clip(max=B - 1)
    data = dict(ts=ts, u0=u0, F=F, filter=(filt_W, filt_rowsum), vol_w=vol_w, dV=dV,
                indptr=indptr, indices=indices, rows=rows)

    def filt(X, d):                                                     # (B, n_nodes)
        Wm, rs = d["filter"]
        return (Wm @ X.T).T / (rs + 1e-12)

    def simp(rho):
        return E0 * (SIMP_EPSILON + (1.0 - SIMP_EPSILON) * rho ** penal)

    # ---------- batched PCG for one group, preconditioned by a factor token ------
    def pcg_group(datas, Fg, token, d):
        M = jax.vmap(lambda r: solve_with(token, r))
        spmv = jax.vmap(lambda a, x: jax.ops.segment_sum(
            a * x[d["indices"]], d["rows"], num_segments=n))
        bnorm = jnp.linalg.norm(Fg, axis=1)
        Z = M(Fg)
        state = (jnp.zeros_like(Fg), Fg, Z, jnp.sum(Fg * Z, 1),
                 jnp.zeros(Fg.shape[0], bool), jnp.zeros(Fg.shape[0], jnp.int32), 0)

        def cond(s):
            return (s[6] < args.maxiter) & ~jnp.all(s[4])

        def body(s):
            X, R, P, rz, done, its, k = s
            AP = spmv(datas, P)
            alpha = jnp.where(done, 0.0, rz / jnp.sum(P * AP, 1))
            X = X + alpha[:, None] * P
            R = R - alpha[:, None] * AP
            now = jnp.linalg.norm(R, axis=1) <= args.tol * bnorm
            its = jnp.where(done, its, k + 1)
            Z = M(R)
            rz_new = jnp.sum(R * Z, 1)
            fin = done | now
            beta_ = jnp.where(fin, 0.0, rz_new / rz)
            P = jnp.where(fin[:, None], P, Z + beta_[:, None] * P)
            return X, R, P, jnp.where(fin, rz, rz_new), fin, its, k + 1

        U, _, _, _, _, its, _ = jax.lax.while_loop(cond, body, state)
        return U, its

    def assemble(E, d):
        return J_fn(d["u0"], tparams(E), bc, d["ts"]).data

    def fact(E_mean, d):
        return factorize(assemble(E_mean, d), d["indptr"], d["indices"], mtype_id=3, mview_id=0)

    def solve_all(Es, d):
        """(B, n_nodes) moduli -> (B, n) displacements, PCG iteration counts."""
        Eg = Es[idx].reshape(G, C, -1)
        Fg = d["F"][idx].reshape(G, C, -1)
        token = fact(Es.mean(0), d) if args.k0 == "global" else None

        def group(a):
            E_c, F_c = a
            tok = token if args.k0 == "global" else fact(E_c.mean(0), d)
            datas = jax.lax.map(lambda E: assemble(E, d), E_c)
            return pcg_group(datas, F_c, tok, d)

        U, its = jax.lax.map(group, (Eg, Fg))
        return U.reshape(G * C, -1)[:B], its.reshape(-1)[:B]

    @jax.custom_vjp
    def compliance(Es, d):
        U, its = solve_all(Es, d)
        return jnp.sum(d["F"] * U, 1), its

    def compliance_fwd(Es, d):
        U, its = solve_all(Es, d)
        return (jnp.sum(d["F"] * U, 1), its), (Es, U, d)

    def compliance_bwd(res, g):
        Es, U, d = res
        gc, _ = g

        def energy(E, u):                                               # u^T K(E) u
            return jnp.dot(u, res_bc(u, tparams(E), bc, d["ts"]))

        def group(a):
            E_c, U_c = a
            return jax.vmap(jax.grad(energy))(E_c, U_c)

        dE = jax.lax.map(group, (Es[idx].reshape(G, C, -1),
                                 jnp.concatenate([U, U[-1:].repeat(G * C - B, 0)])
                                 .reshape(G, C, -1))).reshape(G * C, -1)[:B]
        return -gc[:, None] * dE, None

    compliance.defvjp(compliance_fwd, compliance_bwd)

    @jax.jit
    def evaluate(X, d):
        """Compliances, their gradients w.r.t. the designs, PCG iterations."""
        def obj(X):
            rho = gene.heaviside_projection(filt(X, d), beta=beta)
            c, its = compliance(simp(rho), d)
            return jnp.sum(c), (c, its)
        (_, (c, its)), dc = jax.value_and_grad(obj, has_aux=True)(X)
        return c, dc, its

    @jax.jit
    def oc_update(X, dc, d):
        """Optimality criteria with a per-case bisection on the volume multiplier."""
        move = args.move
        Be = jnp.maximum(-dc, 1e-30) / d["dV"][None, :]                 # (B, n_nodes)

        def x_new(lam):
            return jnp.clip(X * jnp.sqrt(Be / lam[:, None]),
                            jnp.maximum(0.0, X - move), jnp.minimum(1.0, X + move))

        def vol(Xn):
            return filt(Xn, d) @ d["vol_w"]

        def bisect(_, lo_hi):
            lo, hi = lo_hi
            mid = jnp.sqrt(lo * hi)
            over = vol(x_new(mid)) > vf
            return jnp.where(over, mid, lo), jnp.where(over, hi, mid)

        lo, hi = jax.lax.fori_loop(0, 60, bisect, (jnp.full(B, 1e-20), jnp.full(B, 1e20)))
        Xn = x_new(jnp.sqrt(lo * hi))
        return Xn, vol(Xn)

    def block(x):
        return jax.block_until_ready(x)

    X = jnp.full((B, n_nodes), vf)
    setup_s = perf_counter() - t_setup0
    summary = dict(timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   device=device, jax_version=jax.__version__, dof=n, batch=B, case_chunk=C,
                   k0=args.k0, iters=args.iters, tag=args.tag)
    try:
        t0 = perf_counter()
        c_init, dc, its = block(evaluate(X, data))                      # compile + first eval
        block(oc_update(X, dc, data))
        compile_s = perf_counter() - t0
        c0 = onp.asarray(c_init)
        print(f"setup {setup_s:.1f}s, compile + first evaluation {compile_s:.1f}s  "
              f"(PCG iters mean {onp.mean(its):.1f}, max {onp.max(its)})\n")

        t_run = perf_counter()
        ev_t, up_t, pcg_all = [], [], []
        for k in range(args.iters):
            t1 = perf_counter()
            c, dc, its = block(evaluate(X, data))
            t2 = perf_counter()
            X, vols = block(oc_update(X, dc, data))
            t3 = perf_counter()
            its, ratio, vols = onp.asarray(its), onp.asarray(c) / c0, onp.asarray(vols)
            ev_t.append(t2 - t1); up_t.append(t3 - t2); pcg_all.append(its)
            append_row(iter_path, ITER_COLUMNS, dict(
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"), dof=n,
                batch=B, k0=args.k0, iter=k + 1, eval_s=t2 - t1, update_s=t3 - t2,
                elapsed_s=t3 - t_run, pcg_iters_mean=float(its.mean()),
                pcg_iters_max=int(its.max()), c_ratio_mean=float(ratio.mean()),
                c_ratio_max=float(ratio.max()), vol_min=float(vols.min()),
                vol_max=float(vols.max()), tag=args.tag))
            print(f"  iter {k + 1:3d}  eval {t2 - t1:7.2f}s  update {t3 - t2:5.2f}s  "
                  f"PCG mean {its.mean():6.1f} max {its.max():5d}  c/c0 mean {ratio.mean():.4f}  "
                  f"vol [{vols.min():.3f}, {vols.max():.3f}]  t={t3 - t_run:8.1f}s")
        total = perf_counter() - t_run
        c_fin = onp.asarray(block(evaluate(X, data))[0]) / c0
        pcg_all = onp.concatenate(pcg_all)
        summary.update(setup_s=setup_s, compile_s=compile_s, total_s=total,
                       per_iter_s=total / args.iters, eval_s_mean=float(onp.mean(ev_t)),
                       update_s_mean=float(onp.mean(up_t)),
                       pcg_iters_mean=float(pcg_all.mean()), pcg_iters_max=int(pcg_all.max()),
                       c_ratio_mean=float(c_fin.mean()), c_ratio_max=float(c_fin.max()),
                       vol_min=float(vols.min()), vol_max=float(vols.max()), note="")
        print(f"\ntotal {total:.1f}s for {args.iters} iterations ({total / args.iters:.1f}s/iter, "
              f"{total / args.iters / B:.3f}s per design per iteration)")

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            rho = onp.asarray(gene.heaviside_projection(filt(X, data), beta=beta))
            cells = onp.asarray(mesh.cells)
            path = os.path.join(args.save_dir, f"topopt_batched_oc_dof{n}_B{B}.vtu")
            fe.utils.save_sol(
                mesh=mesh, sol_file=path,
                cell_infos=[(f"density_k{i:02d}_{onp.degrees(theta[i]):05.1f}deg",
                             rho[i][cells].mean(1)) for i in range(B)])
            print(f"wrote {path}")
    except Exception as e:
        print(f"failed: {type(e).__name__}: {str(e).splitlines()[0][:300]}")
        summary["note"] = type(e).__name__
    append_row(out_path, SUMMARY_COLUMNS, summary)
    print(f"summary row appended to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
