"""
Topology optimization: stiffness maximisation vs thermal insulation.

Physics:
  - Mechanics: force-controlled tension + thermal stress from solved T field
  - Thermal:   steady-state conduction, bottom T=T_cold, top T=T_hot

Energy densities:
  - Mechanics:  psi = 1/2 [lam*(tr(e_m))^2 + 2*mu*(e_m:e_m)]
                where e_m = sym(grad u) - alpha*(T - T_ref)*I
  - Thermal:    psi = 1/2 * kappa * |grad T|^2

Objective:  min  w_mech * compliance + w_therm * thermal_compliance

BCs:
  - Mechanics: bottom z=0 fixed, top z=L traction in +z
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

traction_z = 0.001        # tensile traction on top face (+z direction)

w_mech = 1.0
w_therm = 1.0

vol_frac = 0.3
mesh_n = 30
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
        # Bottom: z-fixed (free in-plane), corner pins for rigid body
        origin = lambda pt: (np.isclose(pt[0], 0., atol=tol)
                             & np.isclose(pt[1], 0., atol=tol)
                             & np.isclose(pt[2], 0., atol=tol))
        corner_x = lambda pt: (np.isclose(pt[0], L, atol=tol)
                               & np.isclose(pt[1], 0., atol=tol)
                               & np.isclose(pt[2], 0., atol=tol))
        bc_mech = fe.DCboundary.DirichletBCConfig([
            fe.DCboundary.DirichletBCSpec(location=bottom, component="z", value=0.),
            fe.DCboundary.DirichletBCSpec(location=origin, component="x", value=0.),
            fe.DCboundary.DirichletBCSpec(location=origin, component="y", value=0.),
            fe.DCboundary.DirichletBCSpec(location=corner_x, component="y", value=0.),
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
        "beta": Continuation(initial=1.0, final=4.0, update_every=50,
                             step=1.0),
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

# ── Surface & volume mesh extraction ───────────────────────────────────────

print("\nExtracting surface mesh ...")
surface = gene.extract_surface(result.rho_filtered, result.mesh, threshold=0.5)
stl_path = os.path.join(output_dir, "optimized.stl")
surface.save(stl_path)
print(f"  Saved: {stl_path}  ({surface.n_points} vertices, {surface.n_cells} faces, "
      f"manifold={surface.is_manifold})")

print("\nRemeshing optimised geometry ...")
remesh = gene.extract_volume_mesh(
    result.rho_filtered, result.mesh, threshold=0.5, mesh_size=L / (mesh_n))

# ── Re-solve on the clean mesh ─────────────────────────────────────────────

bottom_r = lambda pt: np.isclose(pt[2], 0., atol=tol)
top_r = lambda pt: np.isclose(pt[2], L, atol=tol)

# Thermal
prob_therm_r = ThermalProblem(
    mesh=remesh, vec=1, dim=3, ele_type='TET4', location_fns=[])
bc_therm_r = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(location=bottom_r, component="all", value=T_cold),
    fe.DCboundary.DirichletBCSpec(location=top_r, component="all", value=T_hot),
]).create_bc(prob_therm_r)

# Full density (solid structure after remesh)
rho_solid = np.ones(remesh.points.shape[0])
iv_therm_r = fe.InternalVars(volume_vars=(rho_solid,))

solver_opts_r = fe.DirectSolverOptions()
solver_therm_r = fe.create_solver(
    prob_therm_r, bc_therm_r, solver_options=solver_opts_r,
    adjoint_solver_options=solver_opts_r,
    iter_num=1, internal_vars=iv_therm_r)

init_therm_r = fe.zero_like_initial_guess(prob_therm_r, bc_therm_r)
sol_therm_r = solver_therm_r(iv_therm_r, init_therm_r)
T_field_r = onp.array(sol_therm_r).reshape(-1)
print(f"  T range: {T_field_r.min():.1f} K .. {T_field_r.max():.1f} K")

# Mechanical (with solved T field) — same BCs as optimisation
prob_mech_r = MechanicalProblem(
    mesh=remesh, vec=3, dim=3, ele_type='TET4', location_fns=[top_r])
import numpy as _onp
pts_r = _onp.asarray(remesh.points)
bottom_idx = _onp.where(_onp.isclose(pts_r[:, 2], 0., atol=tol))[0]
pin1_idx = bottom_idx[_onp.argmin(pts_r[bottom_idx, 0])]
pin2_idx = bottom_idx[_onp.argmax(pts_r[bottom_idx, 0])]
pin1_x, pin1_y, pin1_z = float(pts_r[pin1_idx, 0]), float(pts_r[pin1_idx, 1]), float(pts_r[pin1_idx, 2])
pin2_x, pin2_y, pin2_z = float(pts_r[pin2_idx, 0]), float(pts_r[pin2_idx, 1]), float(pts_r[pin2_idx, 2])
pin1_fn = lambda pt: (np.isclose(pt[0], pin1_x, atol=tol)
                      & np.isclose(pt[1], pin1_y, atol=tol)
                      & np.isclose(pt[2], pin1_z, atol=tol))
pin2_fn = lambda pt: (np.isclose(pt[0], pin2_x, atol=tol)
                      & np.isclose(pt[1], pin2_y, atol=tol)
                      & np.isclose(pt[2], pin2_z, atol=tol))
bc_mech_r = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(location=bottom_r, component="z", value=0.),
    fe.DCboundary.DirichletBCSpec(location=pin1_fn, component="x", value=0.),
    fe.DCboundary.DirichletBCSpec(location=pin1_fn, component="y", value=0.),
    fe.DCboundary.DirichletBCSpec(location=pin2_fn, component="y", value=0.),
]).create_bc(prob_mech_r)

iv_mech_r = fe.InternalVars(
    volume_vars=(rho_solid, np.array(T_field_r)))
solver_mech_r = fe.create_solver(
    prob_mech_r, bc_mech_r, solver_options=solver_opts_r,
    adjoint_solver_options=solver_opts_r,
    iter_num=1, internal_vars=iv_mech_r)

init_mech_r = fe.zero_like_initial_guess(prob_mech_r, bc_mech_r)
sol_mech_r = solver_mech_r(iv_mech_r, init_mech_r)

# ── Compute stress & strain at nodes ───────────────────────────────────────

print("  Computing stress and strain fields ...")

num_nodes_r = remesh.points.shape[0]
u_r = onp.array(sol_mech_r).reshape(num_nodes_r, 3)

fe0 = prob_mech_r.fes[0]
shape_grads = onp.array(fe0.shape_grads)   # (C, Q, N, 3)
shape_vals = onp.array(fe0.shape_vals)     # (Q, N)
cells_r = onp.array(fe0.cells)

# Displacement gradient at centroids (average over quad points)
u_cell = u_r[cells_r]                                          # (C, N, 3)
u_grads = onp.einsum('cqnd,cnv->cqvd', shape_grads, u_cell)   # (C, Q, 3, 3)
u_grad_avg = u_grads.mean(axis=1)                              # (C, 3, 3)

# Temperature at centroids
T_cell = T_field_r[cells_r]                                    # (C, N)
T_avg = onp.einsum('qn,cn->c', onp.array(shape_vals), T_cell) / shape_vals.shape[0]

# Strain and stress per element
eps_all = 0.5 * (u_grad_avg + onp.transpose(u_grad_avg, (0, 2, 1)))
eps_th_all = alpha_cte * (T_avg - T_ref)
eps_mech_all = eps_all - eps_th_all[:, None, None] * onp.eye(3)[None, :, :]

lmbda_solid = E0 * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))
mu_solid = E0 / (2. * (1. + nu_val))
sigma_all = (lmbda_solid * onp.trace(eps_mech_all, axis1=1, axis2=2)[:, None, None]
             * onp.eye(3)[None, :, :] + 2 * mu_solid * eps_mech_all)

# Von Mises stress
s_dev = sigma_all - onp.trace(sigma_all, axis1=1, axis2=2)[:, None, None] / 3 * onp.eye(3)
von_mises = onp.sqrt(1.5 * onp.einsum('cij,cij->c', s_dev, s_dev))

print(f"  Von Mises: min={von_mises.min():.4e}, max={von_mises.max():.4e}")

# ── Save results ───────────────────────────────────────────────────────────

result_vtu = os.path.join(output_dir, "remeshed_result.vtu")
fe.utils.save_sol(
    mesh=remesh,
    sol_file=result_vtu,
    point_infos=[
        ("displacement", onp.array(u_r)),
        ("temperature", T_field_r),
    ],
    cell_infos=[
        ("von_mises", von_mises),
        ("stress", sigma_all.reshape(-1, 9)),
        ("strain", eps_all.reshape(-1, 9)),
        ("strain_mechanical", eps_mech_all.reshape(-1, 9)),
    ],
)
print(f"\nSaved: {result_vtu}")
print(f"Final VTU: {output_dir}/final.vtu")
print("Done.")
