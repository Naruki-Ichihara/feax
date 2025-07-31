"""
Simple example using internal_vars with vmap for batch processing using clean feax API
"""

import jax
import jax.numpy as np
from feax import Problem, InternalVars, get_J, get_res, create_J_bc_function, create_res_bc_function
from feax import Mesh, DirichletBC, newton_solve, SolverOptions
from feax.mesh import box_mesh_gmsh

# Poisson's ratio (constant)
nu = 0.3
batch_size = 10

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

# Create mesh
meshio_mesh = box_mesh_gmsh(10, 10, 10, 1., 1., 1., data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

# Boundary conditions
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], 1, atol=1e-5)

dirichlet_bc_info = [[left] * 3, [0, 1, 2], [lambda x: 0., lambda x: 0., lambda x: 0.]]

# Create clean Problem (NO internal_vars!)
problem = ElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    dirichlet_bc_info=dirichlet_bc_info, location_fns=[right]
)

bc = DirichletBC.from_problem(problem)
initial_sol = np.zeros(problem.num_total_dofs_all_vars)
initial_sol = initial_sol.at[bc.bc_rows].set(bc.bc_vals)

# Solve function for single E value using clean API
def solve_single_E(E_scalar):
    # Create E array for all quadrature points
    E_array = InternalVars.create_uniform_volume_var(problem, E_scalar)
    traction_array = InternalVars.create_uniform_surface_var(problem, 1e3)
    
    # Create InternalVars separately
    internal_vars = InternalVars(
        volume_vars=(E_array,),
        surface_vars=[(traction_array,)]
    )
    
    # Create functions using the clean API
    J_bc_func = create_J_bc_function(problem, bc, internal_vars)
    res_bc_func = create_res_bc_function(problem, bc, internal_vars)
    
    solver_options = SolverOptions(tol=1e-8, linear_solver="cg")
    sol, _, _, _ = newton_solve(J_bc_func, res_bc_func, initial_sol, solver_options)
    
    # Return max displacement
    sol_list = problem.unflatten_fn_sol_list(sol)
    return np.max(np.abs(sol_list[0]))

# Create E values: 10 values from 50000 to 100000
E_values = np.linspace(50000, 100000, batch_size)

# Solve for all E values using vmap
print("Solving for E values using vmap...")
max_displacements = jax.vmap(solve_single_E)(E_values)

# Print results
print("\nResults:")
for E, max_disp in zip(E_values, max_displacements):
    print(f"E = {E:8.0f}: max displacement = {max_disp:.6f}")