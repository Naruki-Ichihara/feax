"""Robust 2D topology optimization of a 4-point-bending beam under NumPyro-sampled,
per-load ASYMMETRIC load-angle uncertainty — a showcase of FEAX's cuDSS ``vmap``
factor-once / solve-many path composed with a probabilistic load model.

Why this problem plays to FEAX's strengths
-------------------------------------------
The design ``rho`` is a SINGLE shared field, so the stiffness ``K(rho)`` — and
therefore its cuDSS factorization — is the SAME for every load realization. Only
the right-hand side (the two load directions) varies. FEAX detects that the
surface tractions are ``u``-independent (dead loads -> zero surface Jacobian) and,
with ``reuse_factorization=True``, a ``jax.vmap`` over the load hoists the single
factorization out of the batch: **factor once, solve many RHS** (and the adjoint
reuses the same factors, since the operator is symmetric). This is exactly the
``vmap-rhs`` mode of ``examples/benchmark`` — the regime where the direct solver
amortizes its dominant cost across the whole batch.

Probabilistic load model (NumPyro) — asymmetric, different per load
-------------------------------------------------------------------
A 4-point-bending beam carries TWO loads on top, at the third-points ``x = LX/3``
and ``x = 2 LX/3``. The tilt angle of each is uncertain, and the two follow
DIFFERENT, skewed (asymmetric) distributions, drawn once with NumPyro:

    theta_A = A_LO + (A_HI - A_LO) * Beta(2, 5)     # skewed toward A_LO
    theta_B = B_LO + (B_HI - B_LO) * Beta(5, 2)     # skewed toward B_HI

(A Beta with a != b gives a genuinely asymmetric marginal on an interval.) Each
sampled pair ``(theta_A^i, theta_B^i)`` becomes a traction pair

    t = TRACTION_MAG * (sin theta, -cos theta)

applied on the two top patches. Only the RHS changes across the ``NUM_CASES``
realizations, so all solves share one factorization.

Robust objective
----------------
    J(rho) = mean_i C_i(rho) + KAPPA * std_i C_i(rho)

the mean compliance PLUS a penalty on its spread across the sampled load
distribution, subject to a volume constraint. ``KAPPA = 0`` recovers the ordinary
(mean-only) design; larger ``KAPPA`` buys robustness to how the loads tilt. The
samples are drawn once (sample-average approximation), so gradients w.r.t. ``rho``
flow through the fixed realizations.

Full beam (no symmetry)
-----------------------
The asymmetric loads break left-right symmetry, so the FULL beam is modelled:
``x in [0, LX]``, simply supported by a pin at the bottom-left corner
(``u_x = u_y = 0``) and a roller at the bottom-right (``u_y = 0``).

Runs on GPU (cuDSS).
"""
from __future__ import annotations

import os

os.environ.setdefault("FEAX_X64", "1")          # float64 for accurate gradients
# The per-case vmap keeps 1 live cuDSS factorization; a small cache suffices.
os.environ.setdefault("SPINEAX_FACTOR_CACHE", "8")

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as onp
import nlopt
import numpyro.distributions as dist

import feax as fe
import feax.gene as gene


# ── Configuration ───────────────────────────────────────────────────────────
LX, LY = 6.0, 1.0        # full beam span x height (span:height = 6:1)
NX, NY = 600, 100        # (601 x 101) nodes = 60701, DOF = 121402, 60000 cells

E0 = 1.0                 # solid Young's modulus
EMIN = 1e-3              # void stiffness (moderate contrast keeps cuDSS well-behaved)
NU = 0.3
SIMP_PENALTY = 3.0
HEAVISIDE_BETA = 4.0     # projection sharpness (fixed, no continuation for clarity)
FILTER_RADIUS = 2.5 * (LX / NX)

TRACTION_MAG = 1.0
NUM_CASES = 100          # load realizations (the RHS batch)
SAMPLE_SEED = 0

# Per-load asymmetric angle distributions (degrees): theta = lo + (hi-lo)*Beta(a,b).
# Load A (left third-point): skewed toward negative tilt.
LOAD_A_ANGLE = dict(lo=-40.0, hi=5.0, a=2.0, b=5.0)
# Load B (right third-point): skewed toward positive tilt, different shape.
LOAD_B_ANGLE = dict(lo=-5.0, hi=40.0, a=5.0, b=2.0)

TARGET_VF = 0.1          # volume-fraction constraint
KAPPA = 10.0              # robustness weight on the compliance spread (std)
MAX_ITER = 60

OUTPUT_DIR = Path(__file__).with_name("output_robust_topology_4point_bending_k10")


# ── Geometry / boundary predicates (full model) ─────────────────────────────
DX = LX / NX
XTOL = 1e-5 * max(1.0, LX)
YTOL = 1e-5 * max(1.0, LY)
SUPPORT_HALF_WIDTH = 2.5 * DX                    # bottom-corner support patches
LOAD_HALF_BAND = 1.5 * DX                        # each top load patch ~3 elements
LOAD_A_X = LX / 3.0                              # 4-point bending: two third-points
LOAD_B_X = 2.0 * LX / 3.0

# Simply supported: pin at bottom-left, roller at bottom-right.
PIN = lambda pt: (pt[1] < YTOL) & (pt[0] < SUPPORT_HALF_WIDTH)              # u_x=u_y=0
ROLLER = lambda pt: (pt[1] < YTOL) & (pt[0] > LX - SUPPORT_HALF_WIDTH)      # u_y=0
# Two fixed load patches on the top edge; only the traction DIRECTION varies.
LOAD_A = lambda pt: jnp.isclose(pt[1], LY, atol=YTOL) & (jnp.abs(pt[0] - LOAD_A_X) <= LOAD_HALF_BAND)
LOAD_B = lambda pt: jnp.isclose(pt[1], LY, atol=YTOL) & (jnp.abs(pt[0] - LOAD_B_X) <= LOAD_HALF_BAND)


def const_field_fn(value: float):
    """A surface field that is a constant ``value`` everywhere on the patch."""
    return lambda pt: value + 0.0 * pt[0]        # 0*pt[0] keeps the vmap shape


def sample_angles(spec, key):
    """NumPyro-sample ``NUM_CASES`` angles (radians) from lo + (hi-lo)*Beta(a,b)."""
    u = dist.Beta(spec["a"], spec["b"]).sample(key, (NUM_CASES,))
    deg = spec["lo"] + (spec["hi"] - spec["lo"]) * u
    return onp.deg2rad(onp.asarray(deg))


# ── SIMP linear-elasticity problem (2D) ─────────────────────────────────────
class SIMPElasticity2D(fe.Problem):
    def custom_init(self, E0, Emin, nu, p):
        self.E0, self.Emin, self.nu, self.p = E0, Emin, nu, p

    def get_tensor_map(self):
        def stress(u_grad, rho):
            E = self.Emin + (self.E0 - self.Emin) * rho ** self.p
            mu = E / (2.0 * (1.0 + self.nu))
            lam = E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lam * jnp.trace(eps) * jnp.eye(self.dim) + 2.0 * mu * eps
        return stress

    def get_surface_maps(self):
        def surface_map(u, x, tx, ty):               # dead load: independent of u
            return jnp.array([tx, ty])               # tilted traction (tx, ty)
        return [surface_map, surface_map]            # one per load patch (A, B)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mesh = fe.mesh.rectangle_mesh(Nx=NX, Ny=NY, domain_x=LX, domain_y=LY,
                                  ele_type="QUAD4")
    problem = SIMPElasticity2D(mesh=mesh, vec=2, dim=2, ele_type="QUAD4",
                               location_fns=(LOAD_A, LOAD_B),         # surface 0=A, 1=B
                               additional_info=(E0, EMIN, NU, SIMP_PENALTY))

    bc = fe.DirichletBCConfig([
        fe.DirichletBCSpec(location=PIN,    component="all", value=0.0),
        fe.DirichletBCSpec(location=ROLLER, component="y",   value=0.0),
    ]).create_bc(problem)

    ts = fe.TracedStructure.from_problem(problem)
    n_nodes = mesh.points.shape[0]
    print(f"mesh: {n_nodes} nodes, DOF={problem.num_total_dofs_all_vars}, "
          f"{mesh.cells.shape[0]} cells | load cases: {NUM_CASES}")

    # ── Sample the two asymmetric load-angle distributions with NumPyro ──
    kA, kB = jax.random.split(jax.random.PRNGKey(SAMPLE_SEED))
    thetaA = sample_angles(LOAD_A_ANGLE, kA)                          # (NUM_CASES,)
    thetaB = sample_angles(LOAD_B_ANGLE, kB)
    print(f"load A angle[deg]: mean={onp.rad2deg(thetaA).mean():+6.2f} "
          f"std={onp.rad2deg(thetaA).std():5.2f} | "
          f"load B angle[deg]: mean={onp.rad2deg(thetaB).mean():+6.2f} "
          f"std={onp.rad2deg(thetaB).std():5.2f}")

    def mkvar(value, surface_index):
        return fe.TracedParams.create_spatially_varying_surface_var(
            problem, const_field_fn(float(value)), surface_index=surface_index)

    # Per case: ((txA, tyA) on surface 0, (txB, tyB) on surface 1).
    case_surface_vars = [
        ((mkvar(TRACTION_MAG * onp.sin(thetaA[i]), 0),
          mkvar(-TRACTION_MAG * onp.cos(thetaA[i]), 0)),
         (mkvar(TRACTION_MAG * onp.sin(thetaB[i]), 1),
          mkvar(-TRACTION_MAG * onp.cos(thetaB[i]), 1)))
        for i in range(NUM_CASES)
    ]
    batched_surface_vars = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0), *case_surface_vars)

    # ── Solver: cuDSS with factor-once / solve-many (reuse across RHS + adjoint) ──
    solver_options = fe.DirectSolverOptions(solver="cudss", reuse_factorization=True)
    sample_tp = fe.TracedParams(volume_vars=(jnp.full(n_nodes, TARGET_VF),),
                                surface_vars=case_surface_vars[0])
    solver = fe.create_solver(problem, bc=bc, solver_options=solver_options,
                              adjoint_solver_options=solver_options, linear=True,
                              traced_params=sample_tp, traced_structure=ts)
    initial_guess = fe.zero_like_initial_guess(problem, bc)

    compliance_fn = gene.create_dynamic_compliance_fn(problem)
    volume_fn = gene.create_volume_fn(problem)
    filter_fn = gene.create_helmholtz_filter(mesh, radius=FILTER_RADIUS)

    def compliances(x_flat):
        """Per-realization compliance for the shared design (factor once, solve many)."""
        rho_bar = gene.heaviside_projection(filter_fn(x_flat), beta=HEAVISIDE_BETA)

        def solve_case(surface_vars):
            tp = fe.TracedParams(volume_vars=(rho_bar,), surface_vars=surface_vars)
            sol = solver(tp, initial_guess, traced_structure=ts)
            return compliance_fn(sol, surface_vars)

        return jax.vmap(solve_case)(batched_surface_vars)          # (NUM_CASES,)

    def objective(x_flat):
        c = compliances(x_flat)
        return jnp.mean(c) + KAPPA * jnp.std(c)                    # robust: mean + κ·std

    def volume(x_flat):
        return volume_fn(filter_fn(x_flat))                       # scalar vol. fraction

    obj_and_grad = jax.jit(jax.value_and_grad(objective))
    vol_and_grad = jax.jit(jax.value_and_grad(volume))
    stats = jax.jit(lambda x: (jnp.mean(compliances(x)), jnp.std(compliances(x))))

    # ── MMA loop ───────────────────────────────────────────────────────────
    history = {"iter": [], "objective": [], "mean_C": [], "std_C": [], "vol": []}
    it = [0]

    def nlopt_obj(x, grad):
        xj = jnp.asarray(x)
        val, g = obj_and_grad(xj)
        if grad.size > 0:
            grad[:] = onp.asarray(g)
        mC, sC = stats(xj)
        v = float(volume(xj))
        it[0] += 1
        history["iter"].append(it[0]); history["objective"].append(float(val))
        history["mean_C"].append(float(mC)); history["std_C"].append(float(sC))
        history["vol"].append(v)
        print(f"iter {it[0]:3d} | J={float(val):.4e}  mean(C)={float(mC):.4e}  "
              f"std(C)={float(sC):.4e}  vol={v:.3f}")
        return float(val)

    def nlopt_vol(x, grad):
        xj = jnp.asarray(x)
        v, g = vol_and_grad(xj)
        if grad.size > 0:
            grad[:] = onp.asarray(g)
        return float(v) - TARGET_VF

    opt = nlopt.opt(nlopt.LD_MMA, n_nodes)
    opt.set_lower_bounds(0.0); opt.set_upper_bounds(1.0)
    opt.set_min_objective(nlopt_obj)
    opt.add_inequality_constraint(nlopt_vol, 1e-8)
    opt.set_maxeval(MAX_ITER)
    opt.set_ftol_rel(1e-5)

    x0 = onp.full(n_nodes, TARGET_VF)
    xopt = opt.optimize(x0)
    print(f"\ndone. final J = {opt.last_optimum_value():.6e}")

    # ── Output: final density + history ──────────────────────────────────────
    rho_final = onp.asarray(
        gene.heaviside_projection(filter_fn(jnp.asarray(xopt)), beta=HEAVISIDE_BETA))
    fe.utils.save_sol(mesh, str(OUTPUT_DIR / "density_robust.vtu"),
                      point_infos=[("density", rho_final)])
    with open(OUTPUT_DIR / "history.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(history.keys())
        for row in zip(*history.values()):
            w.writerow(row)
    print(f"saved density + history to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
