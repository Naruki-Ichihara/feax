"""
Topology optimization: stiffness maximisation vs thermal insulation.

Physics:
  - Mechanics: force-controlled compression + thermal stress from solved T field
  - Thermal:   steady-state conduction, bottom T=T_cold, top T=T_hot

Energy densities:
  - Mechanics:  psi = 1/2 [lam*(tr(e_m))^2 + 2*mu*(e_m:e_m)]
                where e_m = sym(grad u) - alpha*(T - T_ref)*I
  - Thermal:    psi = 1/2 * kappa * |grad T|^2

Objective:  min  w_mech * compliance + w_therm * thermal_compliance

BCs:
  - Mechanics: bottom z=0 fixed, top z=L traction in -z
  - Thermal:   bottom T=T_cold, top T=T_hot
"""

import os

import jax.numpy as np
import numpy as onp

import feax as fe
import feax.gene as gene
from feax.gene.optimizer import Pipeline, constraint, Continuation, run

# ── Parameters ──────────────────────────────────────────────────────────────

L = 1.0

E0 = 1.0
nu_val = 0.3
E_eps = E0 * 1e-3
p_mech = 3

kappa0 = 1.0
kappa_eps = kappa0 * 1e-3
p_therm = 3

# Thermal stress
alpha_cte = 12e-6       # CTE [1/K]
T_ref = 293.            # reference (stress-free) temperature [K]
T_cold = 20.            # bottom plate: liquid hydrogen, 20 K (-253 C)
T_hot = 293.            # top plate: room temperature, 293 K (20 C)

traction_z = 0.1        # tensile traction on top face (inner wall pulls outward)

w_mech = 1.0
w_therm = 1.0

vol_frac = 0.3
mesh_n = 20
filter_radius = 0.15
tol = 1e-6


# ── Problem definitions ────────────────────────────────────────────────────

class MechanicalProblem(fe.problem.Problem):
    """Linear elasticity + SIMP + thermal stress from solved T field."""
    def get_energy_density(self):
        def psi(u_grad, rho, T):
            E = E_eps + (E0 - E_eps) * rho ** p_mech
            mu = E / (2. * (1. + nu_val))
            lmbda = E * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))
            eps = 0.5 * (u_grad + u_grad.T)
            eps_th = alpha_cte * (T - T_ref) * np.eye(3)
            eps_mech = eps - eps_th
            return 0.5 * lmbda * np.trace(eps_mech)**2 + mu * np.sum(eps_mech * eps_mech)
        return psi

    def get_surface_maps(self):
        return [lambda u, x, *a: np.array([0., 0., traction_z])]


class ThermalProblem(fe.problem.Problem):
    """Steady-state heat conduction + SIMP."""
    def get_energy_density(self):
        def psi(grad_T, rho):
            kappa = kappa_eps + (kappa0 - kappa_eps) * rho ** p_therm
            return 0.5 * kappa * np.sum(grad_T * grad_T)
        return psi


# ── Pipeline ────────────────────────────────────────────────────────────────

class ThermomechOpt(Pipeline):
    def build(self, mesh):
        bottom = lambda pt: np.isclose(pt[2], 0., atol=tol)
        top = lambda pt: np.isclose(pt[2], L, atol=tol)

        # ── Problems ────────────────────────────────────────────────────
        prob_mech = MechanicalProblem(
            mesh=mesh, vec=3, dim=3, ele_type='HEX8', location_fns=[top])
        prob_therm = ThermalProblem(
            mesh=mesh, vec=1, dim=3, ele_type='HEX8', location_fns=[])

        # ── Dirichlet BCs ──────────────────────────────────────────────
        bc_mech = fe.DCboundary.DirichletBCConfig([
            fe.DCboundary.DirichletBCSpec(location=bottom, component="all", value=0.),
        ]).create_bc(prob_mech)

        bc_therm = fe.DCboundary.DirichletBCConfig([
            fe.DCboundary.DirichletBCSpec(location=bottom, component="all", value=T_cold),
            fe.DCboundary.DirichletBCSpec(location=top, component="all", value=T_hot),
        ]).create_bc(prob_therm)

        # ── Solvers ────────────────────────────────────────────────────
        solver_opts = fe.DirectSolverOptions()

        # Mechanical: volume_vars = (rho, T)
        sample_iv_mech = fe.InternalVars(
            volume_vars=(
                fe.InternalVars.create_node_var(prob_mech, 0.5),
                fe.InternalVars.create_node_var(prob_mech, T_ref),
            ), surface_vars=())
        self._solver_mech = fe.create_solver(
            prob_mech, bc_mech, solver_options=solver_opts,
            adjoint_solver_options=solver_opts,
            iter_num=1, internal_vars=sample_iv_mech)

        # Thermal: volume_vars = (rho,)
        sample_iv_therm = fe.InternalVars(
            volume_vars=(fe.InternalVars.create_node_var(prob_therm, 0.5),),
            surface_vars=())
        self._solver_therm = fe.create_solver(
            prob_therm, bc_therm, solver_options=solver_opts,
            adjoint_solver_options=solver_opts,
            iter_num=1, internal_vars=sample_iv_therm)

        # ── Filter ────────────────────────────────────────────────────
        self._filter_fn = gene.create_density_filter(mesh, radius=filter_radius)

        # ── Response functions ─────────────────────────────────────────
        self._energy_mech = fe.create_energy_fn(prob_mech)
        self._compliance_fn = gene.create_compliance_fn(prob_mech)
        self._energy_therm = fe.create_energy_fn(prob_therm)
        self._volume_fn = gene.create_volume_fn(prob_mech)

        self._init_mech = fe.zero_like_initial_guess(prob_mech, bc_mech)
        self._init_therm = fe.zero_like_initial_guess(prob_therm, bc_therm)
        self._num_nodes = mesh.points.shape[0]

        # ── Reference energies for normalisation ───────────────────────
        rho_ones = np.ones(self._num_nodes)

        iv_therm_ref = fe.InternalVars(volume_vars=(rho_ones,), surface_vars=())
        sol_t_ref = self._solver_therm(iv_therm_ref, self._init_therm)

        iv_mech_ref = fe.InternalVars(
            volume_vars=(rho_ones, sol_t_ref), surface_vars=())
        sol_m_ref = self._solver_mech(iv_mech_ref, self._init_mech)

        self._W_mech_0 = abs(float(self._compliance_fn(sol_m_ref)))
        self._W_therm_0 = abs(float(self._energy_therm(sol_t_ref, iv_therm_ref)))
        print(f"  Reference: W_mech_0={self._W_mech_0:.4e}, "
              f"W_therm_0={self._W_therm_0:.4e}")

    def _phys_density(self, rho, beta):
        return gene.heaviside_projection(self._filter_fn(rho), beta=beta)

    def objective(self, rho, beta=1.0):
        rho_p = self._phys_density(rho, beta)

        # Step 1: solve thermal → T field (node-based)
        iv_therm = fe.InternalVars(volume_vars=(rho_p,), surface_vars=())
        sol_therm = self._solver_therm(iv_therm, self._init_therm)

        # Step 2: solve mechanics with T-dependent thermal strain
        iv_mech = fe.InternalVars(
            volume_vars=(rho_p, sol_therm), surface_vars=())
        sol_mech = self._solver_mech(iv_mech, self._init_mech)

        # Objectives
        W_mech = self._compliance_fn(sol_mech) / self._W_mech_0
        W_therm = self._energy_therm(sol_therm, iv_therm) / self._W_therm_0

        return w_mech * W_mech + w_therm * W_therm

    @constraint(target=vol_frac, type='eq', tol=0.001)
    def volume(self, rho, beta=1.0):
        return self._volume_fn(self._phys_density(rho, beta))

    def filter(self, rho):
        return self._filter_fn(rho)


# ── Mesh & Run ─────────────────────────────────────────────────────────────

print("Generating mesh ...")
mesh = fe.mesh.box_mesh(size=L, mesh_size=L / mesh_n, element_type='HEX8')
print(f"  {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} elements")

output_dir = "output_thermomech_opt"

rho0 = onp.full(mesh.points.shape[0], vol_frac)

result = run(
    pipeline=ThermomechOpt(),
    mesh=mesh,
    rho_init=rho0,
    max_iter=300,
    continuations={
        "beta": Continuation(initial=1.0, final=8.0, update_every=50,
                             multiply_by=1.0, add=1.0),
    },
    output_dir=output_dir,
    save_every=5,
)

# ── Post-processing ────────────────────────────────────────────────────────

print(f"\n  Final objective: {result.final_objective:.4e}")
print(f"  Volume fraction: {result.final_constraints.get('volume', 0.):.4f}")

try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(result.history['iteration'], result.history['objective'], 'b-', lw=1.5)
    axes[0].set(xlabel='Iteration', ylabel='Objective', title='Convergence')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(result.history['iteration'], result.history['volume'], 'g-', lw=1.5)
    axes[1].axhline(y=vol_frac, color='r', ls='--', label=f'Target = {vol_frac}')
    axes[1].set(xlabel='Iteration', ylabel='Volume Fraction', title='Volume')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "history.png"), dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir}/history.png")
except ImportError:
    pass

print(f"\nFinal VTU: {output_dir}/final.vtu")
print("Done.")
