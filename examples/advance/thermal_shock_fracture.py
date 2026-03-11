"""Thermal shock fracture using phase-field method.

Reproduces the ceramic thermal shock experiment:
A hot ceramic strip is suddenly cooled at the bottom boundary,
creating thermal stresses that drive crack initiation and propagation.

Three coupled sub-problems solved in staggered scheme:
  1. Thermal: transient heat diffusion (backward Euler)
  2. Phase-field: AT2 damage evolution driven by elastic strain energy
  3. Mechanics: linear elasticity with thermal strain and damage degradation

Each sub-problem uses the single-variable API:
  - Thermal: get_tensor_map() + get_mass_map()
  - Phase-field: get_tensor_map() + get_mass_map()
  - Mechanics: get_energy_density()

Reference: https://smec-ethz.github.io/tatva-docs/examples/thermal_shock_fracture/
"""

import os

import jax
import jax.numpy as np

import feax as fe
from feax.assembler import Operator

# ============================================================
# Material parameters (ceramic)
# ============================================================
E = 340e3          # Young's modulus [N/mm²]
nu = 0.2           # Poisson's ratio
alpha_th = 8e-6    # Thermal expansion [1/K]
kappa_th = 300.0   # Thermal conductivity
Gc = 0.042         # Fracture toughness [N/mm]
l0 = 0.2           # Phase-field length scale [mm]
eta = 1e-6         # Residual stiffness (prevents full degradation)

# Derived elastic constants
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# Simulation parameters
T0 = 500.0         # Initial temperature [°C]
T_ref = T0         # Reference temperature (stress-free state)
T_bath = 0.0       # Bath temperature [°C]
dt = 1e-5          # Time step
n_steps = 300      # Number of time steps

# ============================================================
# Geometry and mesh
# ============================================================
Lx, Ly = 50.0, 4.9       # Domain size [mm]
Nx, Ny = 250, 25          # Elements (h ≈ l0; use l0/2 for production)
mesh = fe.mesh.rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
num_nodes = mesh.points.shape[0]

# Boundary locations
bottom = lambda pt: np.isclose(pt[1], 0.0, atol=1e-5)
left = lambda pt: np.isclose(pt[0], 0.0, atol=1e-5)


# ============================================================
# Sub-problem 1: Thermal diffusion
# ============================================================
class ThermalProblem(fe.problem.Problem):
    """Transient heat equation: ∂T/∂t = κ∇²T"""

    def get_tensor_map(self):
        def flux(grad_T, *args):
            return kappa_th * grad_T
        return flux

    def get_mass_map(self):
        def mass(T, x, T_old):
            return (T - T_old) / dt
        return mass


thermal_problem = ThermalProblem(mesh, vec=1, dim=2, ele_type='QUAD4')

thermal_bc = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(bottom, 0, T_bath),
]).create_bc(thermal_problem)

thermal_solver = fe.create_solver(
    thermal_problem, thermal_bc,
    solver_options=fe.IterativeSolverOptions(solver='cg'),
    iter_num=1,
    internal_vars=fe.InternalVars(volume_vars=(np.full(num_nodes, T0),)),
)


# ============================================================
# Sub-problem 2: Phase-field damage (AT2 model)
# ============================================================
class PhaseFieldProblem(fe.problem.Problem):
    """AT2 phase-field fracture.

    Minimization of:
      Ψ(d) = ∫ [g(d)·H + Gc/(2l₀)d² + Gc·l₀/2|∇d|²] dΩ

    Weak form (∂Ψ/∂d = 0):
      ∫ [-2(1-d)H + Gc/l₀·d]v dΩ + ∫ Gc·l₀·∇d·∇v dΩ = 0
    """

    def get_tensor_map(self):
        def flux(grad_d, H):
            return Gc * l0 * grad_d
        return flux

    def get_mass_map(self):
        def mass(d, x, H):
            return np.array([-2 * (1 - d[0]) * H + Gc / l0 * d[0]])
        return mass


pf_problem = PhaseFieldProblem(mesh, vec=1, dim=2, ele_type='QUAD4')

# No Dirichlet BC (natural zero-flux BC: ∇d·n = 0)
pf_bc = fe.DCboundary.DirichletBCConfig([]).create_bc(pf_problem)

# Initial history field H = 0 (quad-point based)
num_cells = pf_problem.num_cells
num_quads = pf_problem.fes[0].num_quads
H_field = np.zeros((num_cells, num_quads))

pf_solver = fe.create_solver(
    pf_problem, pf_bc,
    solver_options=fe.IterativeSolverOptions(solver='cg'),
    iter_num=1,  # Linear in d when H is fixed
    internal_vars=fe.InternalVars(volume_vars=(H_field,)),
)


# ============================================================
# Sub-problem 3: Mechanics with damage and thermal strain
# ============================================================
class MechProblem(fe.problem.Problem):
    """Linear elasticity with thermal contraction and damage.

    Energy density:
      ψ(∇u, d, T) = g(d) · [½λ(tr ε_el)² + μ(ε_el:ε_el)]
      where ε_el = sym(∇u) - α(T - T_ref)I
            g(d) = (1-d)² + η
    """

    def get_energy_density(self):
        def psi(u_grad, d_val, T_val):
            eps = 0.5 * (u_grad + u_grad.T)
            eps_el = eps - alpha_th * (T_val - T_ref) * np.eye(2)
            g = (1 - d_val)**2 + eta
            return g * (0.5 * lmbda * np.trace(eps_el)**2
                        + mu * np.sum(eps_el * eps_el))
        return psi


mech_problem = MechProblem(mesh, vec=2, dim=2, ele_type='QUAD4')

mech_bc = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(bottom, 'y', 0.0),
    fe.DCboundary.DirichletBCSpec(left, 'x', 0.0),
]).create_bc(mech_problem)

mech_solver = fe.create_solver(
    mech_problem, mech_bc,
    solver_options=fe.IterativeSolverOptions(solver='cg'),
    iter_num=1,
    internal_vars=fe.InternalVars(volume_vars=(
        np.zeros(num_nodes),        # d
        np.full(num_nodes, T0),     # T
    )),
)


# ============================================================
# History field update
# ============================================================
def compute_elastic_energy_at_quads(u_sol, T_sol):
    """Compute undegraded elastic strain energy at all quadrature points.

    Returns array of shape (num_cells, num_quads) for history field update.
    """
    fe0 = mech_problem.fes[0]
    cells = fe0.cells                    # (num_cells, num_nodes)
    shape_grads = fe0.shape_grads        # (num_cells, nq, nn, dim)
    shape_vals = fe0.shape_vals          # (nq, nn)

    cell_u = u_sol[cells]                # (num_cells, nn, 2)
    cell_T = T_sol[cells, 0]             # (num_cells, nn)

    def per_cell(cu, ct, sg):
        # ∇u at quad points: (nq, 2, 2)
        grad_u = np.sum(cu[None, :, :, None] * sg[:, :, None, :], axis=1)
        # T at quad points: (nq,)
        T_q = np.dot(shape_vals, ct)

        def psi_at_q(gu, tq):
            eps = 0.5 * (gu + gu.T)
            eps_el = eps - alpha_th * (tq - T_ref) * np.eye(2)
            return 0.5 * lmbda * np.trace(eps_el)**2 + mu * np.sum(eps_el * eps_el)

        return jax.vmap(psi_at_q)(grad_u, T_q)  # (nq,)

    return jax.vmap(per_cell)(cell_u, cell_T, shape_grads)  # (num_cells, nq)


# ============================================================
# Output
# ============================================================
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)


def save_step(T_sol, d_sol, u_sol, step):
    vtk_path = os.path.join(data_dir, f'vtk/fracture_{step:04d}.vtu')
    fe.utils.save_sol(
        mesh=mesh,
        sol_file=vtk_path,
        point_infos=[
            ("displacement", u_sol),
            ("temperature", T_sol),
            ("damage", d_sol),
        ]
    )


# ============================================================
# Initial conditions
# ============================================================
T_sol = np.full((num_nodes, 1), T0)     # Temperature
d_sol = np.zeros((num_nodes, 1))         # Damage
u_sol = np.zeros((num_nodes, 2))         # Displacement
T_initial = fe.zero_like_initial_guess(thermal_problem, thermal_bc)
d_initial = fe.zero_like_initial_guess(pf_problem, pf_bc)
u_initial = fe.zero_like_initial_guess(mech_problem, mech_bc)

# Set initial temperature (T0 everywhere, except BC nodes)
T_flat = np.full(thermal_problem.num_total_dofs_all_vars, T0)
T_flat = T_flat.at[thermal_bc.bc_rows].set(thermal_bc.bc_vals)

d_flat = np.zeros(pf_problem.num_total_dofs_all_vars)
u_flat = np.zeros(mech_problem.num_total_dofs_all_vars)


# ============================================================
# Staggered time-stepping loop
# ============================================================
print(f"Thermal shock fracture: {Nx}x{Ny} mesh, dt={dt}, {n_steps} steps")
print(f"  Material: E={E}, nu={nu}, Gc={Gc}, l0={l0}")
print(f"  Thermal: T0={T0}°C, T_bath={T_bath}°C, kappa={kappa_th}")
save_step(T_sol, d_sol, u_sol, 0)

for step in range(1, n_steps + 1):
    T_old = T_sol[:, 0]  # (num_nodes,)

    # --- Step 1: Solve thermal ---
    thermal_iv = fe.InternalVars(volume_vars=(T_old,))
    T_flat = thermal_solver(thermal_iv, T_flat)
    T_sol = thermal_problem.unflatten_fn_sol_list(T_flat)[0]

    # --- Step 2: Update history field ---
    psi_el = compute_elastic_energy_at_quads(u_sol, T_sol)
    H_field = np.maximum(H_field, psi_el)

    # --- Step 3: Solve phase-field ---
    pf_iv = fe.InternalVars(volume_vars=(H_field,))
    d_flat = pf_solver(pf_iv, d_flat)
    d_sol = pf_problem.unflatten_fn_sol_list(d_flat)[0]
    # Clamp damage to [0, 1]
    d_sol = np.clip(d_sol, 0.0, 1.0)
    d_flat = d_sol[:, 0]

    # --- Step 4: Solve mechanics ---
    mech_iv = fe.InternalVars(volume_vars=(d_sol[:, 0], T_sol[:, 0]))
    u_flat = mech_solver(mech_iv, u_flat)
    u_sol = mech_problem.unflatten_fn_sol_list(u_flat)[0]

    if step % 10 == 0:
        d_max = float(d_sol.max())
        T_min = float(T_sol.min())
        T_max = float(T_sol.max())
        print(f"  step {step:4d}: d_max={d_max:.4f}, T=[{T_min:.1f}, {T_max:.1f}]°C")
        save_step(T_sol, d_sol, u_sol, step)

print("Done.")
save_step(T_sol, d_sol, u_sol, n_steps)
