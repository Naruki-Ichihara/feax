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
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

class LinearElasticity(Problem):
    def custom_init(self):
        # Internal variables should be shaped as (num_cells, num_quads)
        num_quads = 8  # HEX8 element has 8 quadrature points
        self.internal_vars = (
            np.full((self.num_cells, num_quads), lmbda), 
            np.full((self.num_cells, num_quads), mu)
        )
    
    def get_tensor_map(self):
        def stress(u_grad, lmbda_state, mu_state):
            epsilon = 0.5 * (u_grad + u_grad.T)
            sigma = lmbda_state * np.trace(epsilon) * np.eye(self.dim) + 2 * mu_state * epsilon
            return sigma
        return stress

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0., 0., 100.])
        return [surface_map]

# Create mesh
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly, Lz = 10., 2., 2.
Nx, Ny, Nz = 25, 5, 55

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

# Get the functional state 
state = problem.get_functional_state()

# Get sparse system
from jax.experimental import sparse

@jax.jit
def solve(state, sol_list):
    # API Design Note: Conceptually, state should contain ONLY internal_vars
    # In the future, problem.assemble_sparse_system should be updated to:
    # 1. Take internal_vars directly instead of full state
    # 2. Only use state.internal_vars from the state object
    A, b = problem.assemble_sparse_system(state, sol_list)
    A_bc, b_bc = apply_bc(A, b, bc_data)
    x_sol, _ = scipy.sparse.linalg.cg(A_bc, b_bc, maxiter=1000)
    sol_final = state.unflatten_fn_sol_list(x_sol)
    return sol_final

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
sigma = lmbda * np.trace(epsilon, axis1=2, axis2=3)[:, :, None, None] * np.eye(problem.dim) + 2 * mu * epsilon

# Calculate von Mises stress
# First average stress over quadrature points
cells_JxW = problem.JxW[:, 0, :]  # (num_cells, num_quads)
sigma_avg = np.sum(sigma * cells_JxW[:, :, None, None], axis=1) / np.sum(cells_JxW, axis=1)[:, None, None]

# Von Mises stress calculation
s_dev = sigma_avg - (1/3) * np.trace(sigma_avg, axis1=1, axis2=2)[:, None, None] * np.eye(3)
vm_stress = np.sqrt(3/2 * np.sum(s_dev * s_dev, axis=(1, 2)))

# Save VTK output
vtk_path = 'linear_elasticity_results.vtu'
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