"""
Simple example: newton_solve inside @jax.jit using clean feax API
"""

import jax
import jax.numpy as np
from feax import Problem, InternalVars, get_J, get_res, create_J_bc_function, create_res_bc_function
from feax import Mesh, DirichletBC, newton_solve, SolverOptions
from feax.mesh import box_mesh_gmsh

# Problem setup
E = 70e3
nu = 0.3

class ElasticityProblem(Problem):
    def get_tensor_map(self):
        def stress(u_grad, E_quad):
            mu = E_quad / (2. * (1. + nu))
            lmbda = E_quad * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress
    
    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return np.array([0., 0., traction_mag])
        return [surface_map]

# Create mesh and problem
meshio_mesh = box_mesh_gmsh(5, 5, 5, 1., 1., 1., data_dir='/tmp', ele_type='HEX8')
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

# Create clean Problem (NO internal_vars!)
problem = ElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    dirichlet_bc_info=dirichlet_bc_info, location_fns=[right]
)

# Create InternalVars separately
E_array = InternalVars.create_uniform_volume_var(problem, E)
traction_array = InternalVars.create_uniform_surface_var(problem, 1.0)

internal_vars = InternalVars(
    volume_vars=(E_array,),
    surface_vars=[(traction_array,)]
)

# Create boundary conditions
bc = DirichletBC.from_problem(problem)
initial_sol = np.zeros(problem.num_total_dofs_all_vars)
initial_sol = initial_sol.at[bc.bc_rows].set(bc.bc_vals)

def solve_jit(initial_sol):
    # Create functions using the clean API
    J_bc_func = create_J_bc_function(problem, bc, internal_vars)
    res_bc_func = create_res_bc_function(problem, bc, internal_vars)
    
    solver_options = SolverOptions(
        tol=1e-8,
        linear_solver="cg",
        x0_strategy="zeros"
    )
    
    return newton_solve(J_bc_func, res_bc_func, initial_sol, solver_options)

# Solve
print("solve..")
solution, _, _, _ = solve_jit(initial_sol)
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