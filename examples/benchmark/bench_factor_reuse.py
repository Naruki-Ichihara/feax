"""Factor-once / solve-many, isolated at the linear-algebra level.

Why not reuse ``bench_rhs_reuse.py``? Going through the feax *solver* re-assembles
AND re-factorizes the stiffness inside every per-batch-element solve, and its CSR
Jacobian assembly reads ``traced_params.surface_vars`` (assembler.py `_get_J_csr`
adds a surface-traction Jacobian term). So batching the traction through the
solver does NOT hoist the factorization out of the vmap — you still pay B
factorizations and see no speedup.

This benchmark bypasses that: it assembles the BC-applied CSR stiffness ``A``
ONCE, then talks to ``spineax.cudss.factor_solve`` directly to compare, for a
batch of ``B`` load cases (RHS ``b``):

* ``reuse``    — factorize ``A`` ONCE, then a single multi-RHS cuDSS SOLVE for all
                 B right-hand sides (``jax.vmap`` over ``b`` with an unbatched
                 factor token -> the factor-once / solve-many fast path).
* ``refactor`` — B distinct matrices (``A`` scaled per case), factorized
                 independently (``jax.vmap`` over the matrix -> ``lax.map``). This
                 is what you pay when the matrix genuinely changes per case
                 (nonlinear Newton, per-design topopt, material sweep).

Both do B solves; the only difference is 1 vs B factorizations. Because
factorization is superlinear and dominates at scale, ``reuse``'s per-solve cost
should fall far below ``refactor`` and the gap should GROW with DOF — the payoff
that ``bench_rhs_reuse.py`` could not expose.

Run:
    python bench_factor_reuse.py --dofs 50000 100000 200000 350000 500000 --batch 10
"""
import argparse
import os
import statistics
import sys
from datetime import datetime, timezone
from time import perf_counter

from bench_linear_elasticity import (
    resolution_for_dof, device_info, timed, CSV_COLUMNS, append_row)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Isolated factor-once/solve-many (reuse vs refactor).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dofs", type=int, nargs="+",
                   default=[50_000, 100_000, 200_000, 350_000, 500_000])
    p.add_argument("--dims", type=float, nargs=3, metavar=("LX", "LY", "LZ"),
                   default=[8.0, 2.0, 2.0])
    p.add_argument("--modes", nargs="+", choices=["reuse", "refactor"],
                   default=["reuse", "refactor"])
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--output", default=None)
    p.add_argument("--tag", default="factor_reuse")
    p.add_argument("--device-name", default=None)
    p.add_argument("--x64", dest="x64", action="store_true", default=True)
    p.add_argument("--no-x64", dest="x64", action="store_false")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ["FEAX_X64"] = "1" if args.x64 else "0"
    # The 'refactor' baseline holds B live factorizations at once; the spineax
    # factor cache (default capacity 8) would evict earlier tokens before they
    # are solved ("unknown/evicted factorization token"). Size it above B.
    os.environ.setdefault("SPINEAX_FACTOR_CACHE", str(max(16, args.batch + 4)))

    import jax
    import jax.numpy as jnp
    import feax as fe
    from feax.assembler import create_J_bc_csr_parametric, create_res_bc_parametric
    from spineax.cudss.factor_solve import factorize, solve_with

    MTYPE_SYM, MVIEW_FULL = 1, 0        # cuDSS: symmetric operator, full storage

    try:
        from importlib.metadata import version as _pkg_version
        feax_version = _pkg_version("feax")
    except Exception:
        feax_version = getattr(fe, "__version__", "?")

    plat, device = device_info(jax, args.device_name)
    dtype = "float64" if args.x64 else "float32"
    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results_factor_reuse.csv")

    print(f"device={device} ({plat})  dtype={dtype}  jax={jax.__version__}  "
          f"feax={feax_version}")
    print(f"modes={args.modes}  batch={args.batch}  output={out_path}\n")

    def block(x):
        return jax.block_until_ready(x)

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
        ts = fe.TracedStructure.from_problem(problem)
        sol0 = fe.zero_like_initial_guess(problem, bc)
        Jfn = create_J_bc_csr_parametric(problem, symmetric=True)
        resfn = create_res_bc_parametric(problem)
        return problem, bc, ts, E, traction, sol0, Jfn, resfn

    def record(mode, dof, res, batch, compile_s, times, note=""):
        mean = statistics.mean(times) if times else float("nan")
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        mn = min(times) if times else float("nan")
        thru = (dof * batch / mean) if times and mean > 0 else float("nan")
        amort = (mean / batch) if times else float("nan")
        row = dict(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            device=device, platform=plat, dtype=dtype, jax_version=jax.__version__,
            feax_version=feax_version, solver="direct", mode=mode, dof=dof,
            nx=res[0], ny=res[1], nz=res[2], batch=batch, warmup=args.warmup,
            repeats=args.repeats, compile_s=compile_s, mean_s=mean, std_s=std,
            min_s=mn, throughput_dofps=thru, amortized_s=amort, tag=args.tag, note=note)
        append_row(out_path, row)
        cs = f"compile={compile_s:7.3f}s" if compile_s == compile_s else "compile=   -   "
        print(f"  {mode:9s} dof={dof:>9,} b={batch:>2}  {cs}  "
              f"mean={mean:8.4f}s  amort={amort:8.4f}s/solve  {thru/1e6:7.3f} MDOF/s"
              + (f"  ({note})" if note else ""))

    B = args.batch
    for target in args.dofs:
        nx, ny, nz, dof = resolution_for_dof(target, args.dims)
        res = (nx, ny, nz)
        print(f"target={target:,} -> nx={nx} ny={ny} nz={nz}  dof={dof:,}")
        try:
            problem, bc, ts, E, traction, sol0, Jfn, resfn = build(nx, ny, nz)
        except Exception as e:
            print(f"  build failed: {type(e).__name__}: {e}")
            record("build", dof, res, 1, float("nan"), [], note=type(e).__name__)
            jax.clear_caches()
            continue

        # Assemble A ONCE, and B load-case RHS vectors (cheap: no factorization).
        scales = jnp.linspace(0.5, 1.5, B)
        tp0 = fe.TracedParams(volume_vars=(E,), surface_vars=[(traction,)])
        Acsr = Jfn(sol0, tp0, bc, ts)
        Adata = Acsr.data
        indptr = Acsr.indptr.astype(jnp.int32)
        indices = Acsr.indices.astype(jnp.int32)

        def rhs_of(scale):
            tp = fe.TracedParams(volume_vars=(E,),
                                 surface_vars=[(traction * scale,)])
            return -resfn(sol0, tp, bc, ts)
        b_batch = jax.jit(jax.vmap(rhs_of))(scales)          # (B, ndof)
        block(b_batch)

        # Correctness sanity: reuse-solve one case vs a feax direct solve.
        if target == args.dofs[0]:
            tok0 = factorize(Adata, indptr, indices,
                             mtype_id=MTYPE_SYM, mview_id=MVIEW_FULL)
            x0 = solve_with(tok0, b_batch[0])
            r0 = float(jnp.linalg.norm(Acsr @ x0 - b_batch[0]) /
                       (jnp.linalg.norm(b_batch[0]) + 1e-30))
            print(f"  [sanity] ||A x - b|| / ||b|| = {r0:.2e}")

        if "reuse" in args.modes:
            try:
                def reuse_solve(Adata, b_batch):
                    token = factorize(Adata, indptr, indices,
                                      mtype_id=MTYPE_SYM, mview_id=MVIEW_FULL)
                    return jax.vmap(lambda b: solve_with(token, b))(b_batch)
                rfn = jax.jit(reuse_solve)
                t0 = perf_counter(); block(rfn(Adata, b_batch))
                record("reuse", dof, res, B, perf_counter() - t0,
                       timed(lambda: rfn(Adata, b_batch), block,
                             args.warmup, args.repeats))
            except Exception as e:
                record("reuse", dof, res, B, float("nan"), [], note=type(e).__name__)

        if "refactor" in args.modes:
            try:
                def refactor_solve(Adata, scales, b_batch):
                    def one(scale, b):
                        tok = factorize(Adata * scale, indptr, indices,
                                        mtype_id=MTYPE_SYM, mview_id=MVIEW_FULL)
                        return solve_with(tok, b)
                    return jax.vmap(one)(scales, b_batch)
                ffn = jax.jit(refactor_solve)
                t0 = perf_counter(); block(ffn(Adata, scales, b_batch))
                record("refactor", dof, res, B, perf_counter() - t0,
                       timed(lambda: ffn(Adata, scales, b_batch), block,
                             args.warmup, args.repeats))
            except Exception as e:
                record("refactor", dof, res, B, float("nan"), [], note=type(e).__name__)

        jax.clear_caches()
        print()

    print(f"Done -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
