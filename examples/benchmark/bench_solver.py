"""Solver-throughput benchmark based on examples/basic/linear_elasticity.py.

Cantilever box (cross-section 20 x 20, HEX8, mesh_size=1), left face fixed,
traction on the right face, solved with feax.create_solver(linear=True,
DirectSolverOptions) — exactly the setting of the base example. The beam
length L is scaled to sweep five DoF levels; L=400 reproduces the example
(~530k DoFs, the current maximum).

Each (size, mode) is timed as the full solve pipeline per call: residual/
Jacobian assembly + direct factorization + solve. Modes:

  jit    jax.jit-compiled solve function (compile time reported separately)
  eager  the same callable without jax.jit

Usage:
    python bench_solver.py [--lengths 25 50 100 200 400] [--repeats 5]
                           [--out solver_results.csv]
"""

import argparse
import csv
import os
import time

import numpy as onp

import jax
import jax.numpy as np

import feax as fe

ELASTIC_MODULI = 70e3
POISSON_RATIO = 0.3
TRACTION = 1e-3
TOL = 1e-5
W = 20
H = 20


class LinearElasticity(fe.problem.Problem):
    def get_energy_density(self):
        def psi(u_grad, *args):
            E = ELASTIC_MODULI
            nu = POISSON_RATIO
            mu = E / (2.0 * (1.0 + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return 0.5 * lmbda * np.trace(epsilon) ** 2 + mu * np.sum(epsilon * epsilon)

        return psi

    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return np.array([0.0, 0.0, traction_mag])

        return [surface_map]


def build(L):
    mesh = fe.mesh.box_mesh((L, W, H), mesh_size=1)
    left = lambda point: np.isclose(point[0], 0.0, TOL)
    right = lambda point: np.isclose(point[0], L, TOL)

    problem = LinearElasticity(mesh, vec=3, dim=3, location_fns=[right])
    bc_config = fe.DCboundary.DirichletBCConfig(
        [fe.DCboundary.DirichletBCSpec(location=left, component="all", value=0.0)]
    )
    bc = bc_config.create_bc(problem)

    traction_array = fe.InternalVars.create_uniform_surface_var(problem, TRACTION)
    internal_vars = fe.InternalVars(volume_vars=(), surface_vars=[(traction_array,)])

    solver = fe.create_solver(
        problem,
        bc,
        solver_options=fe.DirectSolverOptions(),
        linear=True,
        internal_vars=internal_vars,
    )
    initial = fe.zero_like_initial_guess(problem, bc)
    n_dofs = problem.num_total_dofs_all_vars
    n_cells = problem.num_cells
    return solver, internal_vars, initial, n_dofs, n_cells


def median_time(fn, repeats):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(onp.median(ts)), ts


def peak_mem_gb():
    stats = jax.local_devices()[0].memory_stats()
    if stats and "peak_bytes_in_use" in stats:
        return stats["peak_bytes_in_use"] / 1e9
    return float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lengths", nargs="+", type=int, default=[25, 50, 100, 200, 400])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", default="solver_results.csv")
    args = p.parse_args()

    device = jax.devices()[0].device_kind
    dtype = "f64" if jax.config.jax_enable_x64 else "f32"

    fieldnames = [
        "L", "n_dofs", "n_cells", "mode", "time_s", "throughput_mdofs",
        "first_call_s", "peak_mem_gb", "dtype", "device",
    ]
    write_header = not os.path.exists(args.out)
    out = open(args.out, "a", newline="")
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    for L in args.lengths:
        solver, internal_vars, initial, n_dofs, n_cells = build(L)

        def solve_eager():
            return solver(internal_vars, initial)

        solve_jit = jax.jit(lambda iv: solver(iv, initial))

        for mode, fn in [("jit", lambda: solve_jit(internal_vars)), ("eager", solve_eager)]:
            t0 = time.perf_counter()
            sol = jax.block_until_ready(fn())  # compile (jit) / cache warmup (eager)
            first_call_s = time.perf_counter() - t0
            assert bool(np.all(np.isfinite(sol))), "solve produced non-finite values"

            t_med, _ = median_time(lambda: jax.block_until_ready(fn()), args.repeats)
            row = {
                "L": L,
                "n_dofs": n_dofs,
                "n_cells": n_cells,
                "mode": mode,
                "time_s": f"{t_med:.6g}",
                "throughput_mdofs": f"{n_dofs / t_med / 1e6:.4g}",
                "first_call_s": f"{first_call_s:.4g}",
                "peak_mem_gb": f"{peak_mem_gb():.3g}",
                "dtype": dtype,
                "device": device,
            }
            writer.writerow(row)
            out.flush()
            print(
                f"[L={L} {mode:5s}] n_dofs={n_dofs:,} solve={t_med*1e3:.1f} ms "
                f"({n_dofs / t_med / 1e6:.2f} MDoF/s) first_call={first_call_s:.2f} s"
            )

    out.close()


if __name__ == "__main__":
    main()
