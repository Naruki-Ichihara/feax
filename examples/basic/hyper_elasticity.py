"""
Hyperelastic solver example using automatic differentiation.
Demonstrates solving nonlinear hyperelasticity problems with Neo-Hookean material model.
A torsional surface traction is applied to the right face (load-controlled twist).
"""

import os

import jax
import jax.numpy as np

import feax as fe

# Box geometry
Lx, Ly, Lz = 5., 1., 1.
mesh_size   = 0.1

# Cross-section centroid of the right face (used in torsional traction)
y_c = Ly / 2.
z_c = Lz / 2.

# Torsional traction magnitude
T = 20.


class HyperElasticityFeax(fe.problem.Problem):
    def get_tensor_map(self):
        def psi(F):
            E = 100.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)
        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress

    def get_surface_maps(self):
        def traction_map(u_grad, surface_quad_point, traction_magnitude):
            # Torsional traction about x-axis: tangential in yz-plane
            y = surface_quad_point[1]
            z = surface_quad_point[2]
            return np.array([0., -traction_magnitude * (z - z_c), traction_magnitude * (y - y_c)])
        return [traction_map]


mesh = fe.mesh.box_mesh((Lx, Ly, Lz), mesh_size=mesh_size)

# Boundary locations
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)

# Fix left face (clamped)
bc_config = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(location=left, component='all', value=0.)
])

feax_problem = HyperElasticityFeax(mesh, vec=3, dim=3, location_fns=[right])

traction_surface = fe.internal_vars.InternalVars.create_uniform_surface_var(feax_problem, T)
internal_vars = fe.internal_vars.InternalVars(
    volume_vars=[],
    surface_vars=[(traction_surface,)]
)

bc = bc_config.create_bc(feax_problem)

solver_options = fe.solver.DirectSolverOptions(verbose=True)
newton_options = fe.NewtonOptions(internal_jit=True)
solver = fe.solver.create_solver(
    feax_problem,
    bc,
    solver_options,
    internal_vars=internal_vars,
    newton_options=newton_options,
)

def solve_fn(iv):
    sol = solver(iv, fe.utils.zero_like_initial_guess(feax_problem, bc))
    return sol

sol = solve_fn(internal_vars)
sol_unflat = feax_problem.unflatten_fn_sol_list(sol)
displacement = sol_unflat[0]

# Save solution
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)
vtk_path = os.path.join(data_dir, 'vtk/u_hyper_elast.vtu')

fe.utils.save_sol(
    mesh=mesh,
    sol_file=vtk_path,
    point_infos=[("displacement", displacement)])
