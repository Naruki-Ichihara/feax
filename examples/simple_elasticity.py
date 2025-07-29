"""
Simple example: newton_solve inside @jax.jit
"""

import jax
import jax.numpy as np
from feax.problem import Problem as FeaxProblem
from feax.mesh import Mesh, box_mesh_gmsh
from feax.assembler import get_J, get_res
from feax.DCboundary import DirichletBC, apply_boundary_to_J, apply_boundary_to_res
from feax.solver import newton_solve, SolverOptions

# Problem setup
E = 70e3
nu = 0.3

class ElasticityProblem(FeaxProblem):
    def get_tensor_map(self):
        def stress(u_grad, internal_vars=None):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress
    
    def get_surface_maps(self):
        def surface_map(u, x, internal_vars=None):
            return np.array([0., 0., 1.])
        return [surface_map]

# Create mesh and problem
meshio_mesh = box_mesh_gmsh(50, 50, 50, 1., 1., 1., data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], 1, atol=1e-5)

# Fix boundary conditions for a proper elasticity problem
# Left boundary: fix all displacements to 0 (full constraint)  
# Right boundary: apply x-displacement of 0.1 (tension)
def zero_disp(point):
    return 0.0

def tension_disp(point):
    return 0.1

# Constrain left boundary completely, apply tension on right boundary x-direction
dirichlet_bc_info = [[left] * 3 + [right], [0, 1, 2, 0], 
                     [zero_disp, zero_disp, zero_disp, tension_disp]]

problem = ElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    dirichlet_bc_info=dirichlet_bc_info, location_fns=[right]
)

bc = DirichletBC.from_problem(problem)

initial_sol = np.zeros(problem.num_total_dofs_all_vars)
initial_sol = initial_sol.at[bc.bc_rows].set(bc.bc_vals)

def solve_jit(initial_sol):
    def J_bc_func(sol_flat):
        sol_unflat = problem.unflatten_fn_sol_list(sol_flat)
        J = get_J(problem, sol_unflat)
        return apply_boundary_to_J(bc, J)
    
    def res_bc_func(sol_flat):
        sol_unflat = problem.unflatten_fn_sol_list(sol_flat)
        res = get_res(problem, sol_unflat)
        res_flat = jax.flatten_util.ravel_pytree(res)[0]
        return apply_boundary_to_res(bc, res_flat, sol_flat)
    
    solver_options = SolverOptions(
        tol=1e-8,
        linear_solver="cg",
        x0_strategy="zeros"
    )
    
    return newton_solve(J_bc_func, res_bc_func, initial_sol, solver_options)

# Solve
print("solve..")
solution = solve_jit(initial_sol)
print(f"Max displacement: {np.max(np.abs(solution)):.6f}")

# Check the displacement solution
sol_unflat = problem.unflatten_fn_sol_list(solution)
displacement = sol_unflat[0]
# Save solution
from feax.utils import save_sol
save_sol(
    mesh=mesh,
    sol_file="/workspace/solution.vtk",
    point_infos=[("displacement", displacement)])