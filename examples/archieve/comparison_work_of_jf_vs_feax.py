"""
Comparison between JAX-FEM and FEAX implementations.
Demonstrates equivalent hyperelasticity solver implementations using both frameworks.
"""

# Import some useful modules.
import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import matplotlib.pyplot as plt

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, box_mesh_gmsh

E = 1e6
nu = 0.3

# Define constitutive relationship.
class HyperElasticity(Problem):
    def get_tensor_map(self):
        def psi(F, rho):
            E_red = E * rho
            mu = E_red/(2.*(1. + nu))
            kappa = E_red/(3.*(1. - 2.*nu))
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2.
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, rho):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F, rho)
            return P
        return first_PK_stress

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0., 0., 1e3])

        return [surface_map]

    def set_params(self, params):
        rho = params[0]
        self.internal_vars = [rho]

# Specify mesh-related information (first-order hexahedron element).
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly, Lz = 1., 1., 1.
meshio_mesh = box_mesh_gmsh(Nx=5, Ny=5, Nz=5, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir, ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

# Define Dirichlet boundary values.
def get_dirichlet_bottom(scale):
    def dirichlet_bottom(point):
        z_disp = scale*Lz
        return z_disp
    return dirichlet_bottom

def zero_dirichlet_val(point):
    return 0.

# Define boundary locations.
def bottom(point):
    return np.isclose(point[2], 0., atol=1e-5)

def top(point):
    return np.isclose(point[2], Lz, atol=1e-5)

dirichlet_bc_info = [[bottom]*3, [0, 1, 2], [zero_dirichlet_val]*2 + [get_dirichlet_bottom(0.1)]]
location_fns = [top]

problem = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)
rho = 0.5*np.ones((problem.fes[0].num_cells, problem.fes[0].num_quads))
params = [rho]

fwd_pred = ad_wrapper(problem) 
sol_list_jf = fwd_pred(params)

print(sol_list_jf)

sol_unflat_jf = problem.unflatten_fn_sol_list(sol_list_jf[0])
displacement_jf = sol_unflat_jf[0]
# Save solution
from feax.utils import save_sol
save_sol(
    mesh=mesh,
    sol_file="/workspace/solution_jf.vtk",
    point_infos=[("displacement", displacement_jf)])

def test_fn(sol_list):
    return np.sum(sol_list[0]**2)

def composed_fn(params):
    return test_fn(fwd_pred(params))

val = test_fn(sol_list_jf)

h = 1e-3 # small perturbation

rho_plus = rho.at[0, 0].set((1 + h)*rho[0, 0])
params_rho = [rho_plus]
drho_fd_00 = (composed_fn(params_rho) - val)/(h*rho[0, 0])

# Derivative obtained by automatic differentiation
drho = jax.grad(composed_fn)(params)

# Comparison
print(f"\nDerivative comparison between automatic differentiation (AD) and finite difference (FD)")
print(f"\ndrho[0, 0] = {drho[0][0, 0]}, drho_fd_00 = {drho_fd_00}")

from feax import Problem as feaxProblem
from feax import InternalVars, DirichletBC, SolverOptions
from feax import create_solver

class HyperElasticityFeax(feaxProblem):
    def get_tensor_map(self):
        def psi(F, rho):
            E_red = E * rho
            mu = E_red / (2. * (1. + nu))
            kappa = E_red / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy
        P_fn = jax.grad(psi)
        def first_PK_stress(u_grad, rho):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F, rho)
            return P

        return first_PK_stress
    
    def get_surface_maps(self):
        def surface_map(u, x, internal_vars_surface=None):
            return np.array([0., 0., 1e3])

        return [surface_map]

problem_feax = HyperElasticityFeax(mesh, vec=3, dim=3, location_fns=[top])
bc = DirichletBC.from_bc_info(problem_feax, dirichlet_bc_info)
rho_array = InternalVars.create_uniform_volume_var(problem_feax, 0.5)
internal_vars = InternalVars(volume_vars=(rho_array,))

solver_option = SolverOptions(tol=1e-8, linear_solver="bicgstab")
solver_fn = create_solver(problem_feax, bc, solver_option)

sol_list_feax = solver_fn(internal_vars)
sol_unflat = problem_feax.unflatten_fn_sol_list(sol_list_feax)
displacement = sol_unflat[0]
# Save solution
from feax.utils import save_sol
save_sol(
    mesh=mesh,
    sol_file="/workspace/solution.vtk",
    point_infos=[("displacement", displacement)])

def test_fn_feax(sol_vec):
    return np.sum(sol_vec**2)

def composed_fn_feax(internal_vars):
    sol_vec = solver_fn(internal_vars)
    return test_fn_feax(sol_vec)

val_feax = test_fn(sol_unflat)

print(f"sol vals are: jaxfem:{val}, feax:{val_feax}")

grad_sol = jax.grad(composed_fn_feax)(internal_vars)
drho_feax = grad_sol.volume_vars

print(f"\nDerivative comparison between automatic differentiation (AD) feax vs jf")
print(f"\ndrho feax [0, 0] = {drho_feax[0][0, 0]}, drho jf [0, 0] = {drho[0][0, 0]}")