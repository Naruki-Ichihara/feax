"""Laser metal additive manufacturing: thermal-mechanical coupling.

Reproduces the JAX-FEM laser scan tutorial using only FEAX:
  https://github.com/tianjuxue/jax-fem/tree/main/demos/laser_scan

Physics:
  1. Transient heat equation with laser heating, convection, radiation (linearised)
  2. J2 elasto-plasticity with temperature-dependent properties and phase tracking
     (POWDER → LIQUID → SOLID based on liquidus temperature)

Staggered scheme:
  - Solve T at every time step  (linear, iter_num=1)
  - Solve u every ``MECH_INTERVAL`` thermal steps (nonlinear Newton)

Internal variables for mechanics (quad-point tensors):
  sigmas_old   : (num_cells, num_quads, 3, 3)  Cauchy stress
  epsilons_old : (num_cells, num_quads, 3, 3)  total strain
  dT           : (num_cells, num_quads, 1)      temperature increment since last mech solve
  phase        : (num_cells, num_quads, 1)      0=POWDER  1=LIQUID  2=SOLID
"""

import os

import jax
import jax.numpy as np

import feax as fe

# ============================================================
# Material parameters  (Inconel 625, SI units)
# ============================================================
Cp         = 588.       # heat capacity          [J/(kg·K)]
rho        = 8440.      # density                [kg/m³]
k_cond     = 15.        # thermal conductivity   [W/(m·K)]
Tl         = 1623.      # liquidus temperature   [K]
h_conv     = 100.       # convection coefficient [W/(m²·K)]
eta_abs    = 0.25       # laser absorptivity
SB         = 5.67e-8    # Stefan-Boltzmann       [W/(m²·K⁴)]
emissivity = 0.3
T0         = 300.       # ambient temperature    [K]

POWDER = 0
LIQUID = 1
SOLID  = 2

# Laser parameters
vel = 0.5               # scanning velocity [m/s]
rb  = 0.05e-3           # beam radius       [m]
P   = 50.               # laser power       [W]

# ============================================================
# Geometry and mesh
# ============================================================
Nx, Ny, Nz = 50, 20, 5
Lx, Ly, Lz = 0.5e-3, 0.2e-3, 0.05e-3
mesh_size   = Lx / Nx          # ≈ 1e-5 m

mesh = fe.mesh.box_mesh((Lx, Ly, Lz), mesh_size=mesh_size, element_type='HEX8')
num_nodes = mesh.points.shape[0]

# Boundary location functions
def top(point):
    return np.isclose(point[2], Lz, atol=1e-9)

def bottom(point):
    return np.isclose(point[2], 0., atol=1e-9)

def walls(point):
    return (np.isclose(point[0], 0.,  atol=1e-9) |
            np.isclose(point[0], Lx,  atol=1e-9) |
            np.isclose(point[1], 0.,  atol=1e-9) |
            np.isclose(point[1], Ly,  atol=1e-9))

# ============================================================
# Interpolation helpers
# ============================================================

def dof_to_face_quad(problem, sol_nodes, surface_index=0):
    """Interpolate a scalar nodal field to face quadrature points.

    Parameters
    ----------
    sol_nodes : array (num_nodes,)

    Returns
    -------
    array (num_faces, num_face_quads)
    """
    fe0          = problem.fes[0]
    b_inds       = problem.boundary_inds_list[surface_index]
    cell_nodes   = fe0.cells[b_inds[:, 0]]                    # (F, nn)
    face_nodal   = sol_nodes[cell_nodes]                       # (F, nn)
    fsv          = fe0.face_shape_vals[b_inds[:, 1]]           # (F, fq, nn)
    return np.einsum('fqn,fn->fq', fsv, face_nodal)            # (F, fq)


def dof_to_vol_quad(problem, sol_nodes):
    """Interpolate a scalar nodal field to volume quadrature points.

    Parameters
    ----------
    sol_nodes : array (num_nodes,)

    Returns
    -------
    array (num_cells, num_quads)
    """
    fe0        = problem.fes[0]
    cell_nod   = sol_nodes[fe0.cells]                          # (C, nn)
    return np.einsum('qn,cn->cq', fe0.shape_vals, cell_nod)    # (C, nq)


# ============================================================
# Sub-problem 1: Transient thermal
# ============================================================
# dt is defined below at solve time; referenced via closure
dt = None   # will be set before first solve call


class ThermalProblem(fe.problem.Problem):
    """Heat equation:  ρ Cp ∂T/∂t = ∇·(k ∇T)  +  Neumann BCs."""

    def get_tensor_map(self):
        def flux(grad_T, *_):
            return k_cond * grad_T
        return flux

    def get_mass_map(self):
        def mass(T, x, T_old):
            return rho * Cp * (T - T_old) / dt
        return mass

    def get_surface_maps(self):
        def top_flux(u, point, old_T, laser_x, laser_y, switch):
            """Laser heating + linearised convection + radiation on top face."""
            d2      = (point[0] - laser_x)**2 + (point[1] - laser_y)**2
            q_laser = 2*eta_abs*P / (np.pi*rb**2) * np.exp(-2*d2/rb**2) * switch
            q_conv  = h_conv * (T0 - old_T)
            q_rad   = SB * emissivity * (T0**4 - old_T**4)
            return -np.array([q_conv + q_rad + q_laser])

        def wall_flux(u, point, old_T):
            """Linearised convection + radiation on side walls."""
            q_conv = h_conv * (T0 - old_T)
            q_rad  = SB * emissivity * (T0**4 - old_T**4)
            return -np.array([q_conv + q_rad])

        return [top_flux, wall_flux]


thermal_problem = ThermalProblem(
    mesh, vec=1, dim=3, ele_type='HEX8', location_fns=[top, walls])

thermal_bc = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(bottom, 0, T0),
]).create_bc(thermal_problem)

# Dimensions for surface variable arrays
_num_top_faces  = len(thermal_problem.boundary_inds_list[0])
_num_wall_faces = len(thermal_problem.boundary_inds_list[1])
_num_face_quads = thermal_problem.fes[0].num_face_quads


def make_thermal_iv(T_old_nodes, laser_center, switch):
    """Build InternalVars for the thermal problem at the current time step.

    Parameters
    ----------
    T_old_nodes  : (num_nodes,)   temperature at previous step
    laser_center : (3,)           current laser position (x, y, z)
    switch       : float          1 = laser ON, 0 = OFF
    """
    # Volume: T_old for mass (transient) term
    vol = (T_old_nodes,)

    # Surface top: old_T + laser position + switch (broadcast to face quads)
    old_T_top   = dof_to_face_quad(thermal_problem, T_old_nodes, surface_index=0)
    laser_x_top = np.full((_num_top_faces, _num_face_quads), laser_center[0])
    laser_y_top = np.full((_num_top_faces, _num_face_quads), laser_center[1])
    switch_top  = np.full((_num_top_faces, _num_face_quads), switch)

    # Surface walls: old_T only
    old_T_walls = dof_to_face_quad(thermal_problem, T_old_nodes, surface_index=1)

    return fe.InternalVars(
        volume_vars=vol,
        surface_vars=[
            (old_T_top, laser_x_top, laser_y_top, switch_top),
            (old_T_walls,),
        ],
    )


# Build solver with a representative InternalVars for auto-detection
_init_thermal_iv = make_thermal_iv(
    np.full(num_nodes, T0),
    np.array([Lx * 0.25, Ly * 0.5, Lz]),
    0.,
)
thermal_solver = fe.create_solver(
    thermal_problem, thermal_bc,
    solver_options=fe.IterativeSolverOptions(solver='cg'),
    iter_num=1,
    internal_vars=_init_thermal_iv,
)


# ============================================================
# Sub-problem 2: Elasto-plasticity
# ============================================================

class PlasticityProblem(fe.problem.Problem):
    """J2 elasto-plasticity with temperature-dependent properties.

    get_tensor_map returns the stress-return mapping; JAX AD provides the
    consistent tangent automatically.

    Volume internal variables (quad-point tensors):
      sigma_old  : (3, 3)  Cauchy stress at previous step
      epsilon_old: (3, 3)  total strain at previous step
      dT_q       : (1,)    temperature increment (cumulative since last mech solve)
      phase_q    : (1,)    phase flag
    """

    def get_tensor_map(self):

        def safe_sqrt(x):
            return np.where(x > 0., np.sqrt(x), 0.)

        def safe_divide(x, y):
            return np.where(y == 0., 0., x / y)

        def stress_return_map(u_grad, sigma_old, epsilon_old, dT_q, phase_q):
            E0       = 70.e9
            sig0     = 250.e6
            alpha_V0 = 1e-5
            nu       = 0.3

            dT_val    = dT_q[0]
            phase_val = phase_q[0]

            alpha_V = np.where(phase_val == SOLID, alpha_V0, 0.)
            E       = np.where(phase_val == SOLID, E0, 1e-2 * E0)
            mu      = E / (2. * (1. + nu))
            lmbda   = E * nu / ((1 + nu) * (1 - 2*nu))

            epsilon_crt   = 0.5 * (u_grad + u_grad.T)
            epsilon_inc   = epsilon_crt - epsilon_old
            epsilon_inc_T = alpha_V * dT_val * np.eye(3)

            sigma_trial = (lmbda * np.trace(epsilon_inc - epsilon_inc_T) * np.eye(3)
                           + 2*mu * (epsilon_inc - epsilon_inc_T)
                           + sigma_old)

            s_dev    = sigma_trial - (1./3.) * np.trace(sigma_trial) * np.eye(3)
            s_norm   = safe_sqrt(1.5 * np.sum(s_dev * s_dev))
            f_yield  = s_norm - sig0
            f_plus   = np.where(f_yield > 0., f_yield, 0.)
            sigma    = sigma_trial - safe_divide(f_plus * s_dev, s_norm)
            return sigma

        return stress_return_map


mech_problem = PlasticityProblem(mesh, vec=3, dim=3, ele_type='HEX8')

mech_bc = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(bottom, 0, 0.),
    fe.DCboundary.DirichletBCSpec(bottom, 1, 0.),
    fe.DCboundary.DirichletBCSpec(bottom, 2, 0.),
]).create_bc(mech_problem)

# Initial quad-point state
_nc = mech_problem.num_cells
_nq = mech_problem.fes[0].num_quads

sigmas_old   = np.zeros((_nc, _nq, 3, 3))
epsilons_old = np.zeros((_nc, _nq, 3, 3))
dT_quad      = np.zeros((_nc, _nq, 1))
phase_quad   = np.full((_nc, _nq, 1), POWDER, dtype=np.int32)

_init_mech_iv = fe.InternalVars(
    volume_vars=(sigmas_old, epsilons_old, dT_quad, phase_quad))

mech_solver = fe.create_solver(
    mech_problem, mech_bc,
    solver_options=fe.IterativeSolverOptions(solver='cg'),
    internal_vars=_init_mech_iv,
)


# ============================================================
# Post-solve state updates
# ============================================================

def compute_u_grads(sol_u):
    """Displacement gradients at all quadrature points.

    Parameters
    ----------
    sol_u : (num_nodes, 3)

    Returns
    -------
    (num_cells, num_quads, 3, 3)
    """
    fe0    = mech_problem.fes[0]
    cell_u = sol_u[fe0.cells]                            # (C, nn, 3)
    # shape_grads: (C, nq, nn, 3)
    return np.einsum('cnv,cqnd->cqvd', cell_u, fe0.shape_grads)  # (C, nq, 3, 3)


def update_stress_strain(sol_u, mech_params):
    """Recompute stress/strain at quad points from updated displacement.

    Returns new params tuple and (f_plus, stress_xx) for diagnostics.
    """
    sigmas_o, epsilons_o, dT, phase = mech_params
    u_grads = compute_u_grads(sol_u)   # (C, nq, 3, 3)

    def srm_full(ug, so, eo, dt_q, ph_q):
        E0       = 70.e9
        sig0     = 250.e6
        alpha_V0 = 1e-5
        nu       = 0.3
        dT_val   = dt_q[0]
        ph_val   = ph_q[0]
        alpha_V  = np.where(ph_val == SOLID, alpha_V0, 0.)
        E        = np.where(ph_val == SOLID, E0, 1e-2 * E0)
        mu       = E / (2.*(1. + nu))
        lmbda    = E * nu / ((1 + nu)*(1 - 2*nu))
        eps_crt  = 0.5 * (ug + ug.T)
        eps_inc  = eps_crt - eo
        eps_T    = alpha_V * dT_val * np.eye(3)
        s_trial  = lmbda*np.trace(eps_inc - eps_T)*np.eye(3) + 2*mu*(eps_inc - eps_T) + so
        s_dev    = s_trial - (1./3.)*np.trace(s_trial)*np.eye(3)
        s_norm   = np.where(1.5*np.sum(s_dev*s_dev) > 0.,
                            np.sqrt(1.5*np.sum(s_dev*s_dev)), 0.)
        f_yield  = s_norm - sig0
        f_plus   = np.where(f_yield > 0., f_yield, 0.)
        sigma    = s_trial - np.where(s_norm > 0., f_plus/s_norm, 0.) * s_dev
        return sigma, eps_crt, f_plus, sigma[0, 0]

    vfn = jax.vmap(jax.vmap(srm_full))
    sigmas_new, epsilons_new, f_plus, stress_xx = vfn(
        u_grads, sigmas_o, epsilons_o, dT, phase)

    new_params = (sigmas_new, epsilons_new, dT, phase)
    return new_params, (f_plus, stress_xx)


def update_dT_and_phase(dT_nodes, T_new_nodes, mech_params):
    """Update temperature increment and phase at quad points.

    Parameters
    ----------
    dT_nodes    : (num_nodes,)  T_new - T_old_for_mech
    T_new_nodes : (num_nodes,)  current temperature
    """
    sigmas, epsilons, _, phase = mech_params

    dT_q  = dof_to_vol_quad(mech_problem, dT_nodes)[:, :, np.newaxis]   # (C, nq, 1)
    T_q   = dof_to_vol_quad(mech_problem, T_new_nodes)[:, :, np.newaxis] # (C, nq, 1)

    p2l      = (phase == POWDER) & (T_q > Tl)
    l2s      = (phase == LIQUID) & (T_q < Tl)
    new_phase = np.where(p2l, LIQUID, phase)
    new_phase = np.where(l2s, SOLID, new_phase)

    return (sigmas, epsilons, dT_q, new_phase)


# ============================================================
# Output
# ============================================================
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)


def save_step(T_nodes, u_nodes, f_plus_cq, stress_xx_cq, phase_cq, step):
    vtk_path = os.path.join(data_dir, f'vtk/am_{step:05d}.vtu')
    fe.utils.save_sol(
        mesh=mesh,
        sol_file=vtk_path,
        point_infos=[
            ('displacement', u_nodes),
            ('temperature',  T_nodes[:, np.newaxis]),
        ],
        cell_infos=[
            ('f_plus',    np.mean(f_plus_cq,    axis=1)),   # (num_cells,)
            ('stress_xx', np.mean(stress_xx_cq, axis=1)),   # (num_cells,)
            ('phase',     np.max(phase_cq[:, :, 0].astype(float), axis=1)),  # (num_cells,)
        ],
    )


# ============================================================
# Time stepping
# ============================================================
dt             = 2e-6           # time step [s]
laser_on_t     = 0.5 * Lx / vel
simulation_t   = 2. * laser_on_t
MECH_INTERVAL  = 10            # solve mechanics every this many thermal steps

ts = np.arange(0., simulation_t, dt)

# Initial conditions
T_nodes        = np.full(num_nodes, T0)     # temperature field
u_nodes        = np.zeros((num_nodes, 3))   # displacement field
T_flat         = np.full(thermal_problem.num_total_dofs_all_vars, T0)
T_flat         = T_flat.at[thermal_bc.bc_rows].set(thermal_bc.bc_vals)
u_flat         = np.zeros(mech_problem.num_total_dofs_all_vars)

# T used as reference for thermal-strain increment in mechanics
T_nodes_for_mech = T_nodes.copy()

# Initial mechanics state
mech_params   = (sigmas_old, epsilons_old, dT_quad, phase_quad)
f_plus_diag   = np.zeros((_nc, _nq))
stress_xx_diag = np.zeros((_nc, _nq))

print(f"Laser AM: {Nx}×{Ny}×{Nz} mesh ({num_nodes} nodes), dt={dt:.2e}, {len(ts)-1} steps")
print(f"  Laser ON until t={laser_on_t:.4f} s,  total={simulation_t:.4f} s")
save_step(T_nodes, u_nodes, f_plus_diag, stress_xx_diag, phase_quad, 0)

for i, t_cur in enumerate(ts[1:], start=1):
    laser_center = np.array([Lx*0.25 + vel*t_cur, Ly/2., Lz])
    switch       = float(t_cur < laser_on_t)

    # ---- Step 1: Thermal solve ----------------------------------------
    thermal_iv = make_thermal_iv(T_nodes, laser_center, switch)
    T_flat     = thermal_solver(thermal_iv, T_flat)
    T_nodes    = thermal_problem.unflatten_fn_sol_list(T_flat)[0][:, 0]

    # ---- Step 2: Mechanics solve (every MECH_INTERVAL steps) ----------
    if i % MECH_INTERVAL == 0:
        dT_nodes     = T_nodes - T_nodes_for_mech
        mech_params  = update_dT_and_phase(dT_nodes, T_nodes, mech_params)

        mech_iv      = fe.InternalVars(volume_vars=mech_params)
        u_flat       = mech_solver(mech_iv, u_flat)
        u_nodes      = mech_problem.unflatten_fn_sol_list(u_flat)[0]

        mech_params, (f_plus_diag, stress_xx_diag) = update_stress_strain(
            u_nodes, mech_params)

        T_nodes_for_mech = T_nodes

        print(f"  step {i:5d}/{len(ts)-1}  "
              f"t={t_cur:.4e}  laser={'ON' if switch else 'OFF'}  "
              f"laser_x={laser_center[0]:.4f}  "
              f"max f_plus={float(f_plus_diag.max()):.3e}  "
              f"max σ_xx={float(stress_xx_diag.max()):.3e}")

        save_step(T_nodes, u_nodes, f_plus_diag, stress_xx_diag, mech_params[3], i)

print("Done.")
save_step(T_nodes, u_nodes, f_plus_diag, stress_xx_diag, mech_params[3], len(ts) - 1)
