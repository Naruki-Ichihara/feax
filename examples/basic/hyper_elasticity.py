"""
Hyperelastic solver example using automatic differentiation.
Demonstrates solving nonlinear hyperelasticity problems with Neo-Hookean material model.
"""

import feax as fe
import jax
import jax.numpy as np
import os


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
    
mesh = fe.mesh.box_mesh((1, 1, 1), mesh_size=0.1)

# Define boundary locations.
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], 1, atol=1e-5)

def zero_dirichlet_val(point):
    return 0.

def dirichlet_val_x2(point):
    return (0.5 + (point[1] - 0.5) * np.cos(np.pi / 3.) -
            (point[2] - 0.5) * np.sin(np.pi / 3.) - point[1]) / 2.

def dirichlet_val_x3(point):
    return (0.5 + (point[1] - 0.5) * np.sin(np.pi / 3.) +
            (point[2] - 0.5) * np.cos(np.pi / 3.) - point[2]) / 2.

# Create boundary conditions using dataclass approach
bc_config = fe.DCboundary.DirichletBCConfig([
    # Left boundary - fix all components to zero
    fe.DCboundary.DirichletBCSpec(location=left, component='x', value=zero_dirichlet_val),
    fe.DCboundary.DirichletBCSpec(location=left, component='y', value=dirichlet_val_x2),
    fe.DCboundary.DirichletBCSpec(location=left, component='z', value=dirichlet_val_x3),
    # Right boundary - fix all components to zero  
    fe.DCboundary.DirichletBCSpec(location=right, component='all', value=zero_dirichlet_val)
])

feax_problem = HyperElasticityFeax(mesh,
                          vec=3,
                          dim=3)

internal_vars = fe.internal_vars.InternalVars()

bc = bc_config.create_bc(feax_problem)

solver_options = fe.solver.SolverOptions(tol=1e-8, linear_solver="bicgstab", verbose=True)
solver = fe.solver.create_solver(feax_problem, bc, solver_options)

def solve_fn(internal_vars):
    sol = solver(internal_vars, fe.utils.zero_like_initial_guess(feax_problem, bc))
    return sol

print("Solving...")
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