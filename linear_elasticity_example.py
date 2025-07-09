import jax
import jax.numpy as np
import jax.scipy as scipy
import numpy as onp

from feax.problem import Problem
from feax.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh
from feax.boundary_conditions import apply_bc, prepare_bc_info, FixedBC, create_boundary_functions
from feax.utils import save_as_vtk

# Material properties
E = 70e3
nu = 0.3
mu = E / (2. * (1. + nu))

class LinearElasticity(Problem):
    def custom_init(self):
        # No internal parameters needed - lambda and mu are used as constants
        pass
    
    def get_tensor_map(self):
        def stress(u_grad, internal_vars):
            epsilon = 0.5 * (u_grad + u_grad.T)
            # Access lambda from internal_vars if provided
            lmbda_val = internal_vars[0]  # This is a scalar for this quad point
            sigma = lmbda_val * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
            return sigma
        return stress

    def get_surface_maps(self):
        def surface_map(u, x, internal_vars):
            # internal_vars would be a list of arrays, each with shape (num_face_quads, ...)
            # For this example, we don't use internal_vars but they would be available here
            return np.array([0., 0., 100.])
        return [surface_map]
    
    # Example of how get_mass_map would work (if needed):
    # def get_mass_map(self):
    #     def mass(u, x, internal_vars=None):
    #         # u: (vec,) - solution values at quadrature point
    #         # x: (dim,) - physical coordinates of quadrature point  
    #         # internal_vars: list of arrays, each with shape (num_quads, ...)
    #         # For this example, we don't use internal_vars but they would be available here
    #         return density * u  # example mass term
    #     return mass

# Create mesh
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly, Lz = 10., 2., 2.
Nx, Ny, Nz = 25, 5, 5

import tempfile
import os
data_dir = tempfile.mkdtemp()
meshio_mesh = box_mesh_gmsh(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir, ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

# Boundary conditions - ultra simple!
boundary_fns = create_boundary_functions(Lx, Ly, Lz)
location_fns = [boundary_fns['right']]

# Define boundary conditions
DirichletBCs = FixedBC(boundary_fns['left'], components=[0, 1, 2])

# Prepare boundary condition data (NOT JIT-compatible due to mesh dependency)
bc_data = prepare_bc_info(mesh, DirichletBCs, vec_dim=3)

# Create problem
problem = LinearElasticity(mesh, vec=3, dim=3, ele_type=ele_type, location_fns=location_fns)

# Initial solution guess
sol_list = np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))

# Example of creating internal variables (optional)
# For this linear elasticity example, we don't need internal variables,
# but here's how you would create them:
# num_quads = problem.fes[0].JxW.shape[1]  # number of quadrature points per cell
# num_cells = problem.num_cells
# internal_var1 = np.zeros((num_cells * num_quads, 3))  # example: 3 scalar values per quad
# internal_var2 = np.ones((num_cells * num_quads, 2, 2))  # example: 2x2 tensor per quad
# internal_vars = [internal_var1, internal_var2]

# Get the functional state (with optional internal variables)
# state = problem.get_functional_state(internal_vars)  # with internal variables
val = E * nu / ((1 + nu) * (1 - 2 * nu))
lmbdas = np.full((problem.num_cells*8, ), val) 
state = problem.get_functional_state(internal_vars=[lmbdas])  # pass as list of arrays

# Get sparse system
from jax.experimental import sparse

@jax.jit
def solve(state, sol_list):
    # Use the pure functional approach
    from feax.problem import get_sparse_system
    A, b = get_sparse_system(state, jax.flatten_util.ravel_pytree(sol_list)[0])
    A_bc, b_bc = apply_bc(A, b, bc_data)
    x_sol, _ = scipy.sparse.linalg.cg(A_bc, b_bc, maxiter=1000)
    sol_final = state.unflatten_fn_sol_list(x_sol)
    return sol_final

# Example of vmapped solving for multiple parameter values
def single_solve(lmbda_values):
    """Solve for a single set of lambda values across all quadrature points"""
    # Create internal_vars for this lambda value
    internal_vars = [lmbda_values]  # List of arrays
    
    # Create state with these internal variables
    state_with_internal = problem.get_functional_state(internal_vars=internal_vars)
    
    # Solve
    sol_final = solve(state_with_internal, sol_list)
    
    # Return some key result (e.g., max displacement)
    displacement_magnitude = np.linalg.norm(sol_final[0], axis=1)
    max_displacement = np.max(displacement_magnitude)
    return max_displacement

# Test with different lambda values
print("Testing vmapped solve with different material parameters...")
num_cases = 5
base_lambda = val
lambda_variations = np.linspace(0.5 * base_lambda, 1.5 * base_lambda, num_cases)

# Create lambda arrays for each case (each has shape num_cells * num_quads)
lambda_arrays = np.array([np.full((problem.num_cells * 8,), lam_val) for lam_val in lambda_variations])

# Use vmap to solve all cases in parallel
vmapped_solve = jax.vmap(single_solve)
max_displacements = vmapped_solve(lambda_arrays)

print(f"Lambda variations: {lambda_variations}")
print(f"Max displacements: {max_displacements}")
print(f"Relative displacement changes: {max_displacements / max_displacements[2] - 1}")  # normalized to middle case

# Original single solve for comparison
sol_final = solve(state, sol_list)

# Calculate and show max displacement
displacement_magnitude = np.linalg.norm(sol_final[0], axis=1)
max_displacement = np.max(displacement_magnitude)
max_disp_node = np.argmax(displacement_magnitude)
max_disp_coords = mesh.points[max_disp_node]

print(f"Maximum displacement: {max_displacement:.6f}")
print(f"Location: node {max_disp_node} at coordinates {max_disp_coords}")
print(f"Displacement vector: {sol_final[0][max_disp_node]}")

# Calculate stress for visualization
print("Computing stress for VTK output...")
u_grad = problem.fes[0].sol_to_grad(sol_final[0])  # (num_cells, num_quads, vec, dim)
epsilon = 0.5 * (u_grad + u_grad.transpose(0, 1, 3, 2))  # (num_cells, num_quads, vec, dim)
lmbda = val  # Use the same lambda value as in internal vars
sigma = lmbda * np.trace(epsilon, axis1=2, axis2=3)[:, :, None, None] * np.eye(problem.dim) + 2 * mu * epsilon

# Calculate von Mises stress
# First average stress over quadrature points
cells_JxW = problem.JxW[:, 0, :]  # (num_cells, num_quads)
sigma_avg = np.sum(sigma * cells_JxW[:, :, None, None], axis=1) / np.sum(cells_JxW, axis=1)[:, None, None]

# Von Mises stress calculation
s_dev = sigma_avg - (1/3) * np.trace(sigma_avg, axis1=1, axis2=2)[:, None, None] * np.eye(3)
vm_stress = np.sqrt(3/2 * np.sum(s_dev * s_dev, axis=(1, 2)))

# Save VTK output
vtk_path = 'outputs/linear_elasticity_results.vtu'
save_as_vtk(
    mesh=mesh,
    sol_file=vtk_path,
    point_infos=[
        ('displacement', sol_final[0]),
        ('displacement_magnitude', displacement_magnitude)
    ],
    cell_infos=[
        ('von_mises_stress', vm_stress)
    ]
)

print(f"Results saved to {vtk_path}")
print(f"Max von Mises stress: {np.max(vm_stress):.2f}")
print(f"Min von Mises stress: {np.min(vm_stress):.2f}")