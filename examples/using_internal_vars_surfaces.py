"""
Simple example using internal_vars_surfaces with vmap for batch processing using clean feax API
This example varies the surface traction magnitude
"""

import jax
import jax.numpy as np
from feax import Problem, InternalVars, get_J, get_res, create_J_bc_function, create_res_bc_function
from feax import Mesh, DirichletBC, newton_solve, SolverOptions
from feax.mesh import box_mesh_gmsh

# Material constants
E = 70e3  # Young's modulus
nu = 0.3  # Poisson's ratio

class ElasticityProblemWithSurfaceVars(Problem):
    def get_tensor_map(self):
        def stress(u_grad, E_quad):
            mu = E_quad / (2. * (1. + nu))
            lmbda = E_quad * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress
    
    def get_surface_maps(self):
        def surface_map(u, x, traction_magnitude):
            """Apply traction with varying magnitude in z-direction.
            
            Args:
                u: Displacement at surface quadrature point
                x: Position of surface quadrature point
                traction_magnitude: Magnitude of traction at this quadrature point
            """
            return np.array([0., 0., traction_magnitude])
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
problem_base = ElasticityProblemWithSurfaceVars(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    dirichlet_bc_info=dirichlet_bc_info, location_fns=[right]
)

bc = DirichletBC.from_problem(problem_base)
initial_sol = np.zeros(problem_base.num_total_dofs_all_vars)
initial_sol = initial_sol.at[bc.bc_rows].set(bc.bc_vals)

# Get the number of surface quadrature points
num_surface_faces = len(problem_base.boundary_inds_list[0])
num_face_quads = problem_base.fes[0].face_shape_vals.shape[1]

print(f"Number of surface faces on right boundary: {num_surface_faces}")
print(f"Number of quadrature points per face: {num_face_quads}")

# Solve function for single traction value using clean API
def solve_single_traction(traction_scalar):
    # Create traction array for all surface quadrature points
    # Shape: (num_selected_faces, num_face_quads)
    traction_values = np.full((num_surface_faces, num_face_quads), traction_scalar)
    
    # Create InternalVars with volume and surface variables
    E_array = InternalVars.create_uniform_volume_var(problem_base, E)
    
    internal_vars = InternalVars(
        volume_vars=(E_array,),
        surface_vars=[(traction_values,)]  # List with one tuple for one surface
    )
    
    # Create functions using the clean API
    J_bc_func = create_J_bc_function(problem_base, bc, internal_vars)
    res_bc_func = create_res_bc_function(problem_base, bc, internal_vars)
    
    solver_options = SolverOptions(tol=1e-8, linear_solver="cg")
    sol, _, _, _ = newton_solve(J_bc_func, res_bc_func, initial_sol, solver_options)
    
    # Return max displacement
    sol_list = problem_base.unflatten_fn_sol_list(sol)
    return np.max(np.abs(sol_list[0]))

# Create traction values: 10 values from 500 to 5000
traction_values = np.linspace(500, 5000, 10)

# Solve for all traction values using vmap
print("\nSolving for traction values using vmap...")
max_displacements = jax.vmap(solve_single_traction)(traction_values)

# Print results
print("\nResults:")
for traction, max_disp in zip(traction_values, max_displacements):
    print(f"Traction = {traction:8.1f}: max displacement = {max_disp:.6f}")

# Example with spatially varying traction
print("\nExample with spatially varying traction:")

# Get surface quadrature points
surface_quad_points = problem_base.physical_surface_quad_points[0]  # (num_faces, num_face_quads, dim)

# Create traction that varies with height (z-coordinate)
# Higher z gets higher traction
z_coords = surface_quad_points[:, :, 2]  # z-coordinates
traction_varying = 1000.0 + 4000.0 * z_coords  # Varies from 1000 to 5000

# Create InternalVars with spatially varying traction
E_array_varying = InternalVars.create_uniform_volume_var(problem_base, E)

internal_vars_varying = InternalVars(
    volume_vars=(E_array_varying,),
    surface_vars=[(traction_varying,)]
)

# Create functions using the clean API
J_bc_func_varying = create_J_bc_function(problem_base, bc, internal_vars_varying)
res_bc_func_varying = create_res_bc_function(problem_base, bc, internal_vars_varying)

solver_options = SolverOptions(tol=1e-8, linear_solver="cg")
sol_varying, _, _, _ = newton_solve(J_bc_func_varying, res_bc_func_varying, initial_sol, solver_options)

sol_list_varying = problem_base.unflatten_fn_sol_list(sol_varying)
max_disp_varying = np.max(np.abs(sol_list_varying[0]))

print(f"Spatially varying traction (1000 to 5000 based on z):")
print(f"  Traction range: [{np.min(traction_varying):.1f}, {np.max(traction_varying):.1f}]")
print(f"  Max displacement = {max_disp_varying:.6f}")

# Save the varying traction solution
from feax.utils import save_sol
save_sol(
    mesh=mesh,
    sol_file="/workspace/solution_varying_traction.vtk",
    point_infos=[("displacement", sol_list_varying[0])]
)
print("\nSolution with varying traction saved to /workspace/solution_varying_traction.vtk")