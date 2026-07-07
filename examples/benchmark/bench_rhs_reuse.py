"""Factor-once / solve-many benchmark: batch the LOAD (RHS), reuse the factors.

This is the controlled counterpart to ``bench_linear_elasticity.py``'s vmap. Both
vmap a batch of ``B`` linear-elasticity solves of the SAME physical cantilever;
the only difference is *what* is batched:

* ``vmap_mat`` — batch the material ``E`` (as in bench_linear_elasticity.py).
  The stiffness ``A`` differs per batch element, so cuDSS factorizes ``B`` times
  (``spineax`` custom_vmap -> ``jax.lax.map``). Speedup over serial DECAYS with
  DOF because the (superlinear) factorization is not shared.

* ``vmap_rhs`` — batch the traction (RHS ``b``), keep ``E``/geometry FIXED. The
  stiffness ``A`` is identical across the batch, so JAX hoists the factorize out
  of the vmap: ONE cuDSS factorization + a single multi-RHS SOLVE for all ``B``
  loads (the factor-once / solve-many fast path). This needs
  ``DirectSolverOptions(reuse_factorization=True)`` with a SYMMETRIC/SPD operator
  so the solve routes through ``spineax.cudss.factor_solve``.

Because ``vmap_rhs`` pays for the factorization only once, its per-solve cost
should stay far below ``vmap_mat`` and the gap should GROW with DOF.

Both modes are jitted ``jax.vmap`` and timed per DOF; rows append (incrementally,
survives a watchdog SIGKILL) to a plot_bench-compatible CSV.

Run:
    python bench_rhs_reuse.py --dofs 50000 100000 200000 350000 500000 --batch 10
    python bench_rhs_reuse.py --modes vmap_rhs --dofs 500000
"""
import argparse
import os
import statistics
import sys
from datetime import datetime, timezone
from time import perf_counter

# Reuse the pure helpers from the sibling benchmark (no jax/feax at its import).
from bench_linear_elasticity import (
    resolution_for_dof, device_info, timed, CSV_COLUMNS, append_row)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Factor-once/solve-many (RHS-batched) vs material-batched vmap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dofs", type=int, nargs="+",
                   default=[50_000, 100_000, 200_000, 350_000, 500_000])
    p.add_argument("--dims", type=float, nargs=3, metavar=("LX", "LY", "LZ"),
                   default=[8.0, 2.0, 2.0])
    p.add_argument("--modes", nargs="+", choices=["vmap_rhs", "vmap_mat"],
                   default=["vmap_rhs", "vmap_mat"])
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--output", default=None)
    p.add_argument("--tag", default="rhs_reuse")
    p.add_argument("--device-name", default=None)
    p.add_argument("--x64", dest="x64", action="store_true", default=True)
    p.add_argument("--no-x64", dest="x64", action="store_false")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ["FEAX_X64"] = "1" if args.x64 else "0"

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
    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results_rhs_reuse.csv")

    print(f"device={device} ({plat})  dtype={dtype}  jax={jax.__version__}  "
          f"feax={feax_version}")
    print(f"dims={tuple(args.dims)}  modes={args.modes}  batch={args.batch}  "
          f"output={out_path}\n")

    def block(x):
        return jax.block_until_ready(getattr(x, "dofs", x))

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
        tp_sample = fe.TracedParams(volume_vars=(E,), surface_vars=[(traction,)])
        ts = fe.TracedStructure.from_problem(problem)
        # reuse_factorization=True + SYMMETRIC operator => forward/adjoint AND the
        # RHS-batched vmap route through spineax.cudss.factor_solve (factor once).
        # traced_params sample lets the solver="auto" probe detect symmetry.
        solver = fe.create_solver(
            problem, bc,
            solver_options=fe.DirectSolverOptions(reuse_factorization=True),
            linear=True, traced_params=tp_sample, traced_structure=ts,
            return_solution=False)
        initial = fe.zero_like_initial_guess(problem, bc)
        return solver, E, traction, ts, initial

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
            solver, E, traction, ts, initial = build(nx, ny, nz)
        except Exception as e:
            print(f"  build failed: {type(e).__name__}: {e}")
            record("build", dof, res, 1, float("nan"), [], note=type(e).__name__)
            jax.clear_caches()
            continue

        scales = jnp.linspace(0.5, 1.5, B)

        if "vmap_rhs" in args.modes:
            try:
                # A FIXED, b varies: scale the traction field per batch element
                # (traction may be N-D, e.g. (n_faces, n_qp)); broadcast generally.
                tr_batch = traction[None] * scales.reshape((B,) + (1,) * traction.ndim)

                def solve_rhs(tr_field):
                    tp = fe.TracedParams(volume_vars=(E,),
                                         surface_vars=[(tr_field,)])
                    return solver(tp, initial, traced_structure=ts)

                vfn = jax.jit(jax.vmap(solve_rhs))
                t0 = perf_counter(); block(vfn(tr_batch))
                record("vmap_rhs", dof, res, B, perf_counter() - t0,
                       timed(lambda: vfn(tr_batch), block, args.warmup, args.repeats))
            except Exception as e:
                if os.environ.get("FEAX_BENCH_DEBUG"):
                    import traceback; traceback.print_exc()
                record("vmap_rhs", dof, res, B, float("nan"), [], note=type(e).__name__)

        if "vmap_mat" in args.modes:
            try:
                E_batch = E[None, :] * scales[:, None]             # A varies per elem

                def solve_mat(E_field):
                    tp = fe.TracedParams(volume_vars=(E_field,),
                                         surface_vars=[(traction,)])
                    return solver(tp, initial, traced_structure=ts)

                vfn = jax.jit(jax.vmap(solve_mat))
                t0 = perf_counter(); block(vfn(E_batch))
                record("vmap_mat", dof, res, B, perf_counter() - t0,
                       timed(lambda: vfn(E_batch), block, args.warmup, args.repeats))
            except Exception as e:
                record("vmap_mat", dof, res, B, float("nan"), [], note=type(e).__name__)

        jax.clear_caches()
        print()

    print(f"Done -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
