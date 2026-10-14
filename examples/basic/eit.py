"""EIT forward problem with pure-Neumann boundary data.

This example is inspired by the interface-identification problem in
arXiv:2503.22872v3, Section 4.1. Here we focus only on its underlying EIT
forward model:

a known interface partitions ``Omega`` into ``Omega_in`` and
``Omega_out`` with piecewise-constant conductivity ``kappa``:

    -div(kappa * grad(y)) = 0             in Omega
    kappa * dy/dn = g                     on boundary(Omega)

The weak FE solution automatically satisfies continuity of potential and flux
across the interface. A compatible boundary input ``g`` is applied on opposite
sides of a square, and the additive constant is fixed by a zero-mean gauge.
"""
import os

import jax
import jax.numpy as np

import feax as fe


class Conductivity(fe.Problem):
    def get_tensor_map(self):
        return lambda u_grad, kappa: kappa * u_grad

    def get_surface_maps(self):
        def current(u, x, value):
            return value * np.ones_like(u)
        return [current, current]


# ── Mesh and material ────────────────────────────────────────────────────────
mesh = fe.mesh.rectangle_mesh(
    Nx=40, Ny=40, domain_x=1.0, domain_y=1.0,
    origin=(-0.5, -0.5), ele_type="QUAD4",
)
left = lambda point: np.isclose(point[0], -0.5, atol=1e-12)
right = lambda point: np.isclose(point[0], 0.5, atol=1e-12)
problem = Conductivity(mesh, vec=1, dim=2, ele_type="QUAD4",
                       location_fns=[left, right])

cell_centers = np.asarray(mesh.points)[np.asarray(mesh.cells)].mean(axis=1)
omega_in = np.sum(cell_centers**2, axis=1) <= 0.2**2
kappa = np.where(omega_in, 2.0, 0.5)

# Physical outward flux g is -1 on the left, +1 on the right, and zero on the
# insulating top/bottom boundaries. Its boundary integral is zero. FEAX's
# residual surface term has the opposite sign to g.
boundary_flux = (-1.0, 1.0)
surface_vars = [
    (fe.TracedParams.create_uniform_surface_var(problem, -g, surface_index=index),)
    for index, g in enumerate(boundary_flux)
]
params = fe.TracedParams(volume_vars=(kappa,), surface_vars=surface_vars)
bc = fe.DirichletBCConfig([]).create_bc(problem)

# ── Pure-Neumann solve ───────────────────────────────────────────────────────
ts = fe.TracedStructure.from_problem(problem)
solver = fe.create_solver(
    problem, bc,
    linear=True,
    solver_options=fe.KrylovSolverOptions(solver="cg", tol=1e-10),
    nullspace=fe.NullspaceConstraint.constant_mean_zero(),
    traced_params=params,
    traced_structure=ts,
)
initial = fe.zero_like_initial_guess(problem, bc)
solution = solver(params, initial, traced_structure=ts)
potential = solution.field(0)[:, 0]

left_nodes = jax.vmap(left)(np.asarray(mesh.points))
right_nodes = jax.vmap(right)(np.asarray(mesh.points))
voltage = np.mean(potential[right_nodes]) - np.mean(potential[left_nodes])
print(f"electrode voltage difference: {float(voltage):.6f}")

# The forward map is differentiable with respect to material parameters.
def electrode_voltage(kappa_in):
    varied_kappa = np.where(omega_in, kappa_in, 0.5)
    varied = fe.TracedParams(volume_vars=(varied_kappa,), surface_vars=surface_vars)
    u = solver(varied, initial, traced_structure=ts).field(0)[:, 0]
    return np.mean(u[right_nodes]) - np.mean(u[left_nodes])

sensitivity = jax.grad(electrode_voltage)(2.0)
print(f"d(voltage)/d(kappa_in): {float(sensitivity):.6f}")

# ── Save for visualization ───────────────────────────────────────────────────
data_dir = os.path.join(os.path.dirname(__file__), "data", "vtk")
os.makedirs(data_dir, exist_ok=True)
fe.utils.save_sol(
    mesh,
    os.path.join(data_dir, "eit.vtu"),
    cell_infos=[("conductivity", kappa)],
    point_infos=[("potential", potential)],
)
