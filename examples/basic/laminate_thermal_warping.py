"""
Thermal warping of an antisymmetric ±45° CFRP laminate (pure twist).

A 2-layer antisymmetric angle-ply laminate is cooled uniformly:

    Layer 1 (bottom, ``z < 0``): CFRP, orthotropic, fibre at -45°
    Layer 2 (top,    ``z > 0``): CFRP, orthotropic, fibre at +45°

Mechanism
---------
For a ±45° antisymmetric stack, both layers contract isotropically in
the laminate (x, y) axes (``α_xx = α_yy``) so the diagonal entries of
``M_T`` cancel between layers — giving ``M_T_xx = M_T_yy = 0``.
However the off-diagonal shear thermal expansion ``α_xy`` flips sign
between -45° and +45°, and combined with opposite z-moment arms the
contributions **add** to give a large ``M_T_xy``.

The laminate therefore has

    M_T = [[0, +·], [+·, 0]]

i.e. **pure twist, no bending**. The plate buckles into a hyperbolic-
paraboloid (saddle) shape with a strong twist about its longitudinal
axis.

Boundary conditions
-------------------
Clamped along ``x = 0``: removes all rigid-body modes while letting
the rest of the plate warp freely under the thermal moment.

Geometric nonlinearity & solver
-------------------------------
The plate is long enough (``L_X = 100 mm``) that the linear-theory tip
twist is ~20°, past the small-rotation regime. We solve with the **von
Kármán** moderate-rotation extension (``½ ∇w ⊗ ∇w`` membrane term) and
use :class:`feax.ImplicitPipeline` to **load-step** the cooling: at
each pseudo-time step the thermal load is ramped from 0 → ΔT, and
Newton is warm-started from the previous step's solution. Many small
load increments are essential — the cubic vK membrane term has a
small radius of convergence for Newton when starting cold from the
flat state, so a single shot at the full thermal load diverges. The
membrane stretching that develops as the plate twists "stiffens" the
response, giving a converged finite-rotation answer below the linear
prediction.

The pseudo-time variable ``t ∈ [0, 1]`` represents the load fraction.
``TracedParams.volume_vars[0]`` carries this fraction as a node-based
scalar, and the weak form scales the thermal pre-force / pre-moment
``(N_T, M_T)`` by it.
"""
from __future__ import annotations

import math
from pathlib import Path

import jax.numpy as jnp
import numpy as onp

import feax as fe
from feax.mechanics.shell import (
    orthotropic_in_plane_stiffness,
    laminate_stiffness,
    laminate_thermal_loads,
    mindlin_strains,
    mindlin_resultants,
)
from feax.solvers.time_solver import ImplicitPipeline, TimeConfig, run


# ── Geometry ────────────────────────────────────────────────────────────────
# L_X = 100 mm: linear theory would predict ~20° tip twist — past the
# small-rotation regime. vK + load stepping captures the membrane
# stiffening that develops as the plate twists.
L_X, L_Y = 100.0e-3, 50.0e-3   # m
N_X, N_Y = 40, 20

# ── CFRP lamina (orthotropic) ───────────────────────────────────────────────
T_LAYER = 0.5e-3        # m, per layer (total laminate ≈ 1.0 mm)
E1, E2 = 140.0e9, 10.0e9
G12, NU12 = 5.0e9, 0.30
G13, G23 = 5.0e9, 3.0e9
ALPHA_FIBRE = -0.5e-6
ALPHA_TRANS = 30.0e-6
THETA_BOTTOM = -math.pi / 4.0   # -45°
THETA_TOP    = +math.pi / 4.0   # +45°

# ── Thermal load ────────────────────────────────────────────────────────────
DELTA_T = -150.0        # K  (cure ~175 °C → room ~25 °C)

# ── Load-stepping (pseudo-time) ─────────────────────────────────────────────
N_LOAD_STEPS = 1       # number of load increments (more = more robust)

OUTPUT_DIR = Path(__file__).with_name("output_laminate_thermal_warping")


# ── Problem with a load-fraction internal variable ──────────────────────────

class RampedThermalPlate(fe.Problem):
    """Mindlin/von-Kármán plate with a load-fraction internal variable.

    ``volume_vars[0]``: node-based scalar ``λ ∈ [0, 1]``. The thermal
    pre-force/moment ``(N_T, M_T)`` are scaled by ``λ`` at each
    quadrature point, allowing Newton to be load-stepped from the
    undeformed state to full cooling.
    """

    def custom_init(self, A, D, G_s, B, N_T_full, M_T_full, nonlinear="von_karman"):
        self.A = A
        self.D = D
        self.G_s = G_s
        self.B = B
        self.N_T_full = N_T_full
        self.M_T_full = M_T_full
        self.nonlinear = nonlinear

    def get_weak_form(self):
        A, D, G_s, B = self.A, self.D, self.G_s, self.B
        N_T_full, M_T_full = self.N_T_full, self.M_T_full
        nonlinear = self.nonlinear

        def weak_form(vals, grads, x, lam):
            uvw   = vals[0];   theta = vals[1]
            grad_uvw = grads[0]; grad_theta = grads[1]

            eps, kappa, gamma = mindlin_strains(
                grad_uvw, grad_theta, theta, nonlinear=nonlinear,
            )
            N, M, Q = mindlin_resultants(
                eps, kappa, gamma, A, D, G_s,
                B=B, N_T=lam * N_T_full, M_T=lam * M_T_full,
            )

            if nonlinear == "von_karman":
                grad_w = grad_uvw[2, :]
                q_w = Q + N @ grad_w
            else:
                q_w = Q

            grad0 = jnp.concatenate([N, q_w[None, :]], axis=0)
            mass0 = jnp.zeros(3)
            grad1 = M
            mass1 = Q
            return ([mass0, mass1], [grad0, grad1])

        return weak_form


# ── Pipeline ────────────────────────────────────────────────────────────────

class ThermalRampPipeline(ImplicitPipeline):

    def build(self, mesh):
        self.mesh = mesh

        # Single CFRP material reused for both layers (only orientation differs).
        C_cfrp = orthotropic_in_plane_stiffness(E1, E2, G12, NU12)
        G_cfrp = jnp.diag(jnp.array([G13, G23]))
        alpha_cfrp = jnp.diag(jnp.array([ALPHA_FIBRE, ALPHA_TRANS]))

        C_layers = jnp.stack([C_cfrp, C_cfrp])
        G_layers = jnp.stack([G_cfrp, G_cfrp])
        alpha_layers = jnp.stack([alpha_cfrp, alpha_cfrp])
        thetas = jnp.array([THETA_BOTTOM, THETA_TOP])
        thicks = jnp.array([T_LAYER, T_LAYER])

        A, B, D, G_s = laminate_stiffness(C_layers, G_layers, thetas, thicks)
        N_T, M_T = laminate_thermal_loads(
            C_layers, alpha_layers, thetas, thicks,
            dT_avg=DELTA_T, dT_grad=0.0,
        )

        print(f"Laminate: CFRP@{math.degrees(THETA_BOTTOM):+.0f}° / "
              f"CFRP@{math.degrees(THETA_TOP):+.0f}°  "
              f"({T_LAYER*1e3:g} + {T_LAYER*1e3:g} mm)")
        print(f"  ||B||_max = {float(jnp.max(jnp.abs(B))):.3e}")
        print(f"  M_T_xx = {float(M_T[0,0]):+.3e}, M_T_xy = {float(M_T[0,1]):+.3e}")

        self.problem = RampedThermalPlate(
            mesh=[mesh, mesh], vec=[3, 2], dim=2,
            ele_type=["QUAD4", "QUAD4"],
            additional_info=(A, D, G_s, B, N_T, M_T, "von_karman"),
        )

        TOL = 1e-6 * max(L_X, L_Y)
        left_edge = lambda pt: jnp.isclose(pt[0], 0.0, atol=TOL)
        self.bc = fe.DirichletBCConfig([
            fe.DirichletBCSpec(location=left_edge, component="all", value=0.0,
                               variable_index=0),
            fe.DirichletBCSpec(location=left_edge, component="all", value=0.0,
                               variable_index=1),
        ]).create_bc(self.problem)

        self._n_nodes = mesh.points.shape[0]

        self.ts = fe.TracedStructure.from_problem(self.problem)

        sample_iv = fe.TracedParams(
            volume_vars=(jnp.zeros(self._n_nodes),),
        )
        # Use umfpack (general LU) rather than the auto-selected cholmod:
        # the vK tangent stiffness can become indefinite under twisting
        # (the geometric-stiffness term ``N · ∇w`` flips sign with N), and
        # a Cholesky factorisation produces spurious Newton steps on a
        # non-SPD matrix, which the line search then rejects (alpha → 0,
        # residual stalls).
        self.solver = fe.create_solver(
            self.problem, bc=self.bc,
            solver_options=fe.DirectSolverOptions(solver="umfpack", verbose=True),
            adjoint_solver_options=fe.DirectSolverOptions(solver="umfpack"),
            newton_options=fe.NewtonOptions(
                tol=1e-6, rel_tol=1e-8, max_iter=30,
            ),
            linear=False,
            traced_params=sample_iv,
            traced_structure=self.ts,
        )

    def step(self, state, t, dt):
        # Override the base ImplicitPipeline.step to thread the
        # TracedStructure into the solver call (the problem's host-side
        # scratch maps were freed by TracedStructure.from_problem above).
        import jax
        if self.pseudo_time:
            state = jax.lax.stop_gradient(state)
        tp = self.update_vars(state, t, dt)
        return self.solver(tp, state, traced_structure=self.ts)

    def initial_state(self):
        return fe.zero_like_initial_guess(self.problem, self.bc)

    def update_vars(self, state, t, dt):
        # Pseudo-time t goes from 0 to 1 — t + dt is the load fraction at
        # the *end* of this step (backward-Euler convention).
        lam = t + dt
        return fe.TracedParams(
            volume_vars=(jnp.full(self._n_nodes, lam),),
        )

    def save(self, state, step, t, output_dir):
        sol_list = self.problem.unflatten_fn_sol_list(state)
        uvw = onp.asarray(sol_list[0])
        rotation = onp.asarray(sol_list[1])
        fe.utils.save_sol(
            self.mesh,
            str(Path(output_dir) / f"warping_{step:03d}.vtu"),
            point_infos=[("displacement", uvw), ("rotation", rotation)],
        )

    def monitor(self, state, step, t):
        sol_list = self.problem.unflatten_fn_sol_list(state)
        uvw = onp.asarray(sol_list[0])
        pts = onp.asarray(self.mesh.points)
        TOL = 1e-6 * L_X
        tip = onp.isclose(pts[:, 0], L_X, atol=TOL)
        y = pts[tip, 1]; w = uvw[tip, 2]
        o = onp.argsort(y); y, w = y[o], w[o]
        if len(y) > 1:
            slope = float(onp.polyfit(y, w, 1)[0])
        else:
            slope = 0.0
        return {
            "lam": float(t),
            "twist_deg": math.degrees(slope),
            "max_w_mm": float(abs(uvw[:, 2]).max()) * 1e3,
        }


# ── Run ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh = fe.mesh.rectangle_mesh(
        Nx=N_X, Ny=N_Y, domain_x=L_X, domain_y=L_Y, ele_type="QUAD4",
    )
    print(f"Mesh: {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} cells")

    dt = 1.0 / N_LOAD_STEPS
    config = TimeConfig(dt=dt, t_end=1.0, save_every=5, print_every=1)
    result = run(
        ThermalRampPipeline(), mesh, config, output_dir=str(OUTPUT_DIR),
    )
    print(f"\nSaved snapshots and history.csv to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
