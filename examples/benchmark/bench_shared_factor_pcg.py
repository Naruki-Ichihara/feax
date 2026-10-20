"""Benchmark: many DIFFERENT stiffness matrices solved with ONE shared factorization.

``vmap-lhs`` (``bench_linear_elasticity.py``) batches cases whose matrix differs
(material, design), so the direct solver factorizes every case: no speed-up over
solving them one by one, and B factorizations must fit in GPU memory at once
(B = 10 already fails at 254k DOF on A100).

Here the cases are solved with ``fe.vmap`` of the linear solver. The batched
parameter is the modulus field, which changes the operator, so ``fe.vmap``
takes its lhs path: the B matrices are assembled on their shared pattern and
solved together by batched CG preconditioned by ``fe.shared_factor`` --

    K_0 = K(E_bar)                      factorized ONCE (E_bar = mean of the case fields)
    K_i u_i = f   for i = 1..B          preconditioned CG with M = K_0^{-1}

Every CG iteration is one batched SpMV with the B assembled ``K_i`` plus one
batched back-substitution with the ``K_0`` factors (the cheap multi-RHS solve of
``vmap-rhs``). When ``K_i`` is close to ``K_0`` CG converges in a few iterations,
so the cost per case approaches that of ``vmap-rhs`` while only one factorization
is ever held in memory. ``--case-chunk`` (``chunk=``) solves the cases in groups,
one factorization of the group mean per group.

Cases: the cantilever of ``bench_linear_elasticity.py`` (8 x 2 x 2 HEX8 grid,
clamp at x = 0, tip traction), with a random log-normal Young's modulus field per
case, ``E_i(x) = E0 * exp(sigma * g_i(x) - sigma^2 / 2)``, ``g_i`` a smooth
unit-variance Gaussian field (random Fourier features, correlation length
``--corr-len``). ``--sigma`` sets the spread (0.3 -> E varies by roughly +-30 %).

Per DOF target it reports the wall time of the batched solve (``total_s``: the
whole ``fe.vmap`` call), the standalone times of its two set-up pieces
(``assemble_s``: assembling the B matrices, ``factor_s``: factorizing K_0) and
the remainder (``pcg_s = total_s - assemble_s - factor_s``, the CG iterations),
the CG iteration counts, and the wall time of a plain direct solve of the first
``--ref-cases`` cases, one by one (the ``vmap-lhs`` cost per case) with the
relative error of the PCG solution against them. Rows append to
``bench_shared_factor.csv``.

Run:
    python bench_shared_factor_pcg.py
    python bench_shared_factor_pcg.py --dofs 500000 --batch 50 --sigma 0.5
    python bench_shared_factor_pcg.py --dofs 500000 --batch 100 --case-chunk 10
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Different stiffness matrices, one shared factorization (PCG).")
    p.add_argument("--dofs", type=int, nargs="+", default=[100_000, 250_000, 500_000],
                   help="Target DOF counts (grid rule of bench_linear_elasticity.py).")
    p.add_argument("--batch", type=int, default=10, help="Number of cases B.")
    p.add_argument("--sigma", type=float, default=0.3,
                   help="Log-std of the Young's modulus field (case-to-case spread).")
    p.add_argument("--corr-len", type=float, default=0.5,
                   help="Correlation length of the modulus field (model units).")
    p.add_argument("--n-features", type=int, default=64,
                   help="Random Fourier features per field.")
    p.add_argument("--tol", type=float, default=1e-8, help="PCG relative residual tolerance.")
    p.add_argument("--maxiter", type=int, default=500, help="PCG iteration cap.")
    p.add_argument("--precond", choices=["shared", "jacobi", "none"], default="shared",
                   help="CG preconditioner (fe.vmap preconditioner=): the shared K_0 "
                        "factorization fe.shared_factor() (default), the per-case diagonal "
                        "fe.jacobi(), or none. jacobi/none factorize nothing.")
    p.add_argument("--case-chunk", type=int, default=0,
                   help="Solve the cases in groups of this size (fe.vmap chunk=), one "
                        "group after another with one factorization of the group mean "
                        "each (0 = all at once). Bounds the memory of the stored K_i.")
    p.add_argument("--ref-cases", type=int, default=3,
                   help="Cases also solved by a plain direct solve (timing + error).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None,
                   help="CSV path (default: bench_shared_factor.csv next to this script).")
    p.add_argument("--tag", default="", help="Free-form label written to every row.")
    p.add_argument("--device-name", default=None, help="Override the detected device name.")
    return p.parse_args(argv)


L, W, H = 8.0, 2.0, 2.0
E0 = 70.0e3
NU = 0.30
TRACTION = 1e-3

CSV_COLUMNS = [
    "timestamp", "device", "jax_version", "dof", "batch", "sigma", "corr_len",
    "assemble_s", "factor_s", "pcg_s", "total_s", "per_case_s",
    "iters_mean", "iters_max", "ref_per_case_s", "speedup_vs_direct",
    "max_rel_err", "precond", "tag", "note",
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
    # The reference direct solves run one after another; a deep factor cache would
    # keep every one of them alive on the GPU.
    os.environ.setdefault("SPINEAX_FACTOR_CACHE", "1")

    import jax
    import jax.numpy as jnp
    import numpy as onp
    import feax as fe

    _, device = device_info(jax, args.device_name)
    out_path = args.output or os.path.join(HERE, "bench_shared_factor.csv")
    print(f"device={device}  jax={jax.__version__}  B={args.batch}  sigma={args.sigma}  "
          f"corr_len={args.corr_len}  tol={args.tol}  output={out_path}\n")

    class LinearElasticity(fe.Problem):
        def get_tensor_map(self):
            def stress(u_grad, E):
                mu = E / (2.0 * (1.0 + NU))
                lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
                eps = 0.5 * (u_grad + u_grad.T)
                return lam * jnp.trace(eps) * jnp.eye(self.dim) + 2.0 * mu * eps
            return stress

        def get_surface_maps(self):
            return [lambda u, x, t: jnp.array([0.0, 0.0, t])]

    def block(x):
        return jax.block_until_ready(x)

    def timed(fn, *a):
        t0 = perf_counter()
        out = block(fn(*a))
        return perf_counter() - t0, out

    for target in args.dofs:
        nx, ny, nz, dof = resolution_for_dof(target, (L, W, H))
        spacing = (L / nx, W / ny, H / nz)
        print(f"target={target:,} -> nx={nx} ny={ny} nz={nz}  dof={dof:,}")
        row = dict(timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   device=device, jax_version=jax.__version__, dof=dof, batch=args.batch,
                   sigma=args.sigma, corr_len=args.corr_len, tag=args.tag)
        try:
            mesh = fe.StructuredGrid((nx, ny, nz), spacing=spacing).to_mesh()
            problem = LinearElasticity(
                mesh, vec=3, dim=3, ele_type="HEX8",
                location_fns=[lambda p: jnp.isclose(p[0], L, 1e-5 + 0.5 * spacing[0])])
            bc = fe.DirichletBCConfig([fe.DirichletBCSpec(
                location=lambda p: jnp.isclose(p[0], 0.0, 1e-5),
                component="all", value=0.0)]).create_bc(problem)
            ts = fe.TracedStructure.from_problem(problem, free_scratch=False)
            u0 = fe.zero_like_initial_guess(problem, bc)
            trac = fe.TracedParams.create_uniform_surface_var(problem, TRACTION)
            tp = lambda E: fe.TracedParams(volume_vars=(E,), surface_vars=[(trac,)])
            n = u0.shape[0]

            # B log-normal modulus fields from random Fourier features.
            pts = jnp.asarray(mesh.points)
            key = jax.random.PRNGKey(args.seed)
            kw, kp = jax.random.split(key)
            omega = jax.random.normal(kw, (args.batch, args.n_features, 3)) / args.corr_len
            phase = jax.random.uniform(kp, (args.batch, args.n_features), maxval=2 * jnp.pi)
            # One field at a time: all B at once would hold a (B, n_nodes, features)
            # temporary (8.5 GB for B = 100 at 500k DOF).
            def field(wp):
                w, ph = wp
                g = jnp.sqrt(2.0 / args.n_features) * jnp.cos(pts @ w.T + ph).sum(-1)
                return E0 * jnp.exp(args.sigma * g - 0.5 * args.sigma ** 2)
            Es = jax.jit(lambda o, p: jax.lax.map(field, (o, p)))(omega, phase)  # (B, n_nodes)
            E_bar = Es.mean(0)

            # The solver, batched over the modulus field by fe.vmap (lhs path).
            solver = fe.create_solver(problem, bc, solver_options=fe.DirectSolverOptions(),
                                      linear=True, traced_params=tp(E_bar),
                                      traced_structure=ts, return_solution=False)
            precond = {"shared": fe.shared_factor(), "jacobi": fe.jacobi(),
                       "none": lambda op: (lambda R: R)}[args.precond]
            C = args.case_chunk if 0 < args.case_chunk < args.batch else None
            batched = fe.vmap(
                solver, in_axes=fe.TracedParams(volume_vars=(0,), surface_vars=[(None,)]),
                options=fe.KrylovSolverOptions(solver="cg", tol=args.tol, atol=0.0,
                                               maxiter=args.maxiter),
                preconditioner=precond, chunk=C, return_info=True)
            assert batched.kind == "lhs", batched
            solve = jax.jit(lambda Es: batched(tp(Es), u0, ts))
            block(solve(Es))                                                # compile + warm
            t_total, (X, info) = timed(solve, Es)
            its, done = onp.asarray(info.iterations), onp.asarray(info.converged)
            if not done.all():
                print(f"  WARNING: {int((~done).sum())} case(s) hit --maxiter")

            # The two set-up pieces of that call, timed on their own: assembling
            # the B matrices and factorizing K_0 = K(E_bar) (shared only).
            assemble = jax.jit(lambda Es: jax.vmap(
                lambda E: solver.assemble(u0, tp(E), traced_structure=ts).data)(Es))
            E_group = Es if C is None else Es[:C]
            block(assemble(E_group))                                        # compile + warm
            t_asm, values = timed(assemble, E_group)                        # one group
            t_asm *= 1 if C is None else -(-args.batch // C)                # all groups
            if args.precond == "shared":
                from feax.solvers._cudss_compat import factorize
                indptr, indices = solver.assemble(u0, tp(E_bar)).indptr, solver.assemble(u0, tp(E_bar)).indices
                fact = jax.jit(lambda d: factorize(d, indptr, indices, mtype_id=3, mview_id=0))
                block(fact(values.mean(0)))
                t_fac, _ = timed(fact, values.mean(0))
                t_fac *= 1 if C is None else -(-args.batch // C)
            else:
                t_fac = 0.0
            del values
            t_pcg = max(t_total - t_asm - t_fac, 0.0)
            groups = -(-args.batch // (C or args.batch))

            # Reference: plain direct solve of the first few cases, one by one.
            direct = jax.jit(lambda E: solver(tp(E), u0, traced_structure=ts))
            block(direct(Es[0]))                                            # compile + warm
            n_ref = min(args.ref_cases, args.batch)
            ref_t, errs = [], []
            for i in range(n_ref):
                t_i, u_i = timed(direct, Es[i])
                ref_t.append(t_i)
                errs.append(float(jnp.linalg.norm(X[i] - u_i) / jnp.linalg.norm(u_i)))

            total = t_total
            ref_case = float(onp.mean(ref_t))
            row.update(assemble_s=t_asm, factor_s=t_fac, pcg_s=t_pcg, total_s=total,
                       per_case_s=total / args.batch, iters_mean=float(its.mean()),
                       iters_max=int(its.max()), ref_per_case_s=ref_case,
                       speedup_vs_direct=ref_case / (total / args.batch),
                       max_rel_err=max(errs), precond=args.precond, note="")
            print(f"  [{args.precond}] {args.batch} cases in {groups} group(s) of {C or args.batch}:  "
                  f"fe.vmap solve: {t_total:7.3f}s  = assemble K_i {t_asm:6.3f}s + "
                  f"factor K_0 {t_fac:6.3f}s + PCG {t_pcg:6.3f}s  "
                  f"(iters mean {its.mean():.1f}, max {its.max()})")
            print(f"  {args.precond}: {total / args.batch:.4f}s/case   direct one by one: "
                  f"{ref_case:.4f}s/case   speed-up x{ref_case / (total / args.batch):.1f}   "
                  f"max rel.err vs direct = {max(errs):.1e}")
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {str(e).splitlines()[0][:300]}")
            row["note"] = type(e).__name__
        append_row(out_path, row)
        jax.clear_caches()
        print()

    print(f"Rows appended to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
