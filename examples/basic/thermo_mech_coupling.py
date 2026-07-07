"""One-way thermo-mechanical coupling on a BCC lattice (forward analysis).

A row of three BCC unit cells is embedded in a box mesh as a nodal density
field (feax.flat.graph). The analysis is staggered:

  1. Thermal:    div( k(rho) grad T ) = 0      hot left face, cold right face
  2. Mechanical: div( sigma(u, rho, T) ) = 0   thermal strain alpha (T - T_ref)

The solved temperature is passed to the mechanical problem as a nodal
TracedParams variable via ``TracedParams.node_var_from_solution`` — the
standard feax bridge for staggered multi-physics. Both solves are linear and
run through the same differentiable ``create_solver`` machinery, so the whole
chain remains ``jax.grad``-able end to end (not used here: this is the
forward analysis).

Outputs: thermo_mech_coupling.vtu (density, temperature, displacement).
"""
import os

import numpy as onp

import jax.numpy as np

import feax as fe
import feax.flat as flat

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_CELLS = 3                 # BCC unit cells along x
L = 1.0                     # unit cell edge length
MESH_SIZE = 0.1             # background hex size
RADIUS = 0.15               # strut radius
RHO_VOID = 1e-2             # void density floor (keeps K well-conditioned)

E0, NU, PENAL = 70e3, 0.3, 3.0        # Young's modulus (solid), Poisson, SIMP
K0 = 1.0                              # thermal conductivity (solid)
ALPHA, T_REF = 1e-4, 0.0              # expansion coefficient, reference temp
T_HOT, T_COLD = 100.0, 0.0

# ---------------------------------------------------------------------------
# Background mesh + BCC lattice density (nodal field)
# ---------------------------------------------------------------------------
mesh = fe.mesh.box_mesh((N_CELLS * L, L, L), mesh_size=MESH_SIZE)
print(f"Mesh: {len(mesh.points)} nodes, {len(mesh.cells)} HEX8 cells")

corners = onp.array([[i, j, k]
                     for i in range(N_CELLS + 1)
                     for j in (0.0, L) for k in (0.0, L)], dtype=float)
centers = onp.array([[i + 0.5, 0.5 * L, 0.5 * L] for i in range(N_CELLS)])
nodes = np.asarray(onp.vstack([corners, centers]))

corner_id = {(i, j, k): 4 * i + 2 * j + k
             for i in range(N_CELLS + 1) for j in (0, 1) for k in (0, 1)}
edges = np.asarray([[corner_id[(i + di, j, k)], len(corners) + i]
                    for i in range(N_CELLS)
                    for di in (0, 1) for j in (0, 1) for k in (0, 1)])

lattice_fn = flat.graph.create_lattice_function(nodes, edges, radius=RADIUS)

# ---------------------------------------------------------------------------
# 1) Thermal conduction on the lattice
# ---------------------------------------------------------------------------
class Thermal(fe.Problem):
    def get_tensor_map(self):
        def flux(grad_T, rho):
            return K0 * rho * grad_T
        return flux


thermal = Thermal(mesh, vec=1, dim=3, ele_type="HEX8")
rho = flat.graph.create_lattice_density_field_nodal(
    thermal, lattice_fn, density_solid=1.0, density_void=RHO_VOID)
print(f"Lattice volume fraction: {float(np.mean(rho)):.3f}")

bc_th = fe.DirichletBCConfig([
    fe.DirichletBCSpec(lambda p: np.isclose(p[0], 0.0), "all", T_HOT),
    fe.DirichletBCSpec(lambda p: np.isclose(p[0], N_CELLS * L), "all", T_COLD),
]).create_bc(thermal)

tp_th = fe.TracedParams(volume_vars=(rho,))
solve_T = fe.create_solver(
    thermal, bc_th,
    solver_options=fe.KrylovSolverOptions(solver="cg", tol=1e-10, atol=1e-12,
                                          use_jacobi_preconditioner=True),
    linear=True)
sol_T = solve_T(tp_th, fe.zero_like_initial_guess(thermal, bc_th))
T_nodes = fe.TracedParams.node_var_from_solution(thermal, sol_T)
print(f"Thermal solve: T in [{float(np.min(T_nodes)):.2f}, "
      f"{float(np.max(T_nodes)):.2f}]")

# ---------------------------------------------------------------------------
# 2) Thermo-elasticity: thermal strain from the solved temperature
# ---------------------------------------------------------------------------
class ThermoElastic(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, rho, T):
            E = E0 * rho ** PENAL
            mu = E / (2.0 * (1.0 + NU))
            lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
            eps = 0.5 * (u_grad + u_grad.T) - ALPHA * (T - T_REF) * np.eye(3)
            return lam * np.trace(eps) * np.eye(3) + 2.0 * mu * eps
        return stress


mech = ThermoElastic(mesh, vec=3, dim=3, ele_type="HEX8")
bc_me = fe.DirichletBCConfig([
    fe.DirichletBCSpec(lambda p: np.isclose(p[0], 0.0), "all", 0.0),
]).create_bc(mech)

# the bridge: solved nodal T rides along rho as a second nodal variable
tp_me = fe.TracedParams(volume_vars=(rho, T_nodes))
solve_u = fe.create_solver(
    mech, bc_me,
    solver_options=fe.KrylovSolverOptions(solver="cg", tol=1e-10, atol=1e-12,
                                          use_jacobi_preconditioner=True),
    linear=True)
sol_u = solve_u(tp_me, fe.zero_like_initial_guess(mech, bc_me))

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
u = onp.asarray(sol_u.field(0))                       # (num_nodes, 3)
pts = onp.asarray(mesh.points)
tip = onp.isclose(pts[:, 0], N_CELLS * L)
solid = onp.asarray(rho) > 0.5
print(f"Free-end mean axial expansion : {u[tip, 0].mean():.4e}")
print(f"Max |u| (solid struts)        : "
      f"{onp.linalg.norm(u[solid], axis=1).max():.4e}")

outdir = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(outdir, "thermo_mech_coupling.vtu")
fe.utils.save_sol(mesh, out, point_infos=[
    ("density", onp.asarray(rho)),
    ("temperature", onp.asarray(T_nodes)),
    ("displacement", u),
])
print(f"Saved: {out}")
