import jax
import jax.numpy as np
import jax.scipy as scipy
import numpy as onp
import time

from feax.problem import Problem
from feax.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh
from feax.boundary_conditions import apply_bc, prepare_bc_info, FixedBC, create_boundary_functions

# Material properties
E = 70e3
nu = 0.3
mu = E / (2. * (1. + nu))

class LinearElasticity(Problem):
    def custom_init(self):
        pass
    
    def get_tensor_map(self):
        def stress(u_grad, internal_vars):
            epsilon = 0.5 * (u_grad + u_grad.T)
            # Use spatially varying lambda from internal_vars
            lmbda_val = internal_vars[0]  # scalar for this quad point
            sigma = lmbda_val * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
            return sigma
        return stress

    def get_surface_maps(self):
        def surface_map(u, x, internal_vars):
            return np.array([0., 0., 100.])
        return [surface_map]

# Create mesh (smaller for faster vmapped solve)
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly, Lz = 10., 2., 2.
Nx, Ny, Nz = 15, 3, 3  # Smaller mesh for faster computation

import tempfile
import os
data_dir = tempfile.mkdtemp()
meshio_mesh = box_mesh_gmsh(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir, ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

# Boundary conditions
boundary_fns = create_boundary_functions(Lx, Ly, Lz)
location_fns = [boundary_fns['right']]
DirichletBCs = FixedBC(boundary_fns['left'], components=[0, 1, 2])
bc_data = prepare_bc_info(mesh, DirichletBCs, vec_dim=3)

# Create problem
problem = LinearElasticity(mesh, vec=3, dim=3, ele_type=ele_type, location_fns=location_fns)

# Initial solution guess
sol_list = np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))

@jax.jit
def solve(state, sol_list):
    from feax.problem import get_sparse_system
    A, b = get_sparse_system(state, jax.flatten_util.ravel_pytree(sol_list)[0])
    A_bc, b_bc = apply_bc(A, b, bc_data)
    x_sol, _ = scipy.sparse.linalg.cg(A_bc, b_bc, maxiter=1000)
    sol_final = state.unflatten_fn_sol_list(x_sol)
    return sol_final

def single_solve(lmbda_values):
    """Solve for a single set of lambda values"""
    internal_vars = [lmbda_values]
    state_with_internal = problem.get_functional_state(internal_vars=internal_vars)
    sol_final = solve(state_with_internal, sol_list)
    
    # Return multiple outputs: max displacement and total elastic energy
    displacement_magnitude = np.linalg.norm(sol_final[0], axis=1)
    max_displacement = np.max(displacement_magnitude)
    
    # Compute elastic energy (simplified)
    total_energy = 0.5 * np.sum(displacement_magnitude**2)
    
    return max_displacement, total_energy

print(f"Problem size: {problem.num_cells} cells, {problem.fes[0].num_total_nodes * 3} DOFs")
print(f"Quadrature points per cell: {problem.fes[0].JxW.shape[1]}")

# Parameter study: varying lambda spatially and across cases
print("\n=== Vmapped Parameter Study ===")

# Base lambda value
base_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))

# Case 1: Uniform lambda variations
print("1. Uniform lambda variations:")
num_cases = 10
lambda_variations = np.linspace(0.5 * base_lambda, 2.0 * base_lambda, num_cases)
lambda_arrays_uniform = np.array([np.full((problem.num_cells * 8,), lam_val) for lam_val in lambda_variations])

start_time = time.time()
vmapped_solve = jax.vmap(single_solve, out_axes=(0, 0))  # vmap over both outputs
max_disps_uniform, energies_uniform = vmapped_solve(lambda_arrays_uniform)
uniform_time = time.time() - start_time

print(f"   Solved {num_cases} cases in {uniform_time:.3f} seconds")
print(f"   Lambda range: {lambda_variations[0]:.0f} to {lambda_variations[-1]:.0f}")
print(f"   Max displacement range: {np.min(max_disps_uniform):.4f} to {np.max(max_disps_uniform):.4f}")
print(f"   Energy range: {np.min(energies_uniform):.2e} to {np.max(energies_uniform):.2e}")

# Case 2: Spatially varying lambda (gradient from left to right)
print("\n2. Spatially varying lambda (gradient material):")
num_cases = 5
gradient_strengths = np.linspace(0.5, 2.0, num_cases)

# Create spatially varying lambda arrays
x_coords = mesh.points[:, 0]  # x-coordinates of nodes
cell_centers_x = np.mean(x_coords[mesh.cells], axis=1)  # x-coordinate of cell centers

lambda_arrays_gradient = []
for strength in gradient_strengths:
    # Linear variation from left (strength * base_lambda) to right (base_lambda)
    lambda_per_cell = base_lambda * (1.0 + (strength - 1.0) * (1.0 - cell_centers_x / Lx))
    # Expand to all quadrature points
    lambda_per_quad = np.repeat(lambda_per_cell, 8)  # 8 quads per cell
    lambda_arrays_gradient.append(lambda_per_quad)

lambda_arrays_gradient = np.array(lambda_arrays_gradient)

start_time = time.time()
max_disps_gradient, energies_gradient = vmapped_solve(lambda_arrays_gradient)
gradient_time = time.time() - start_time

print(f"   Solved {num_cases} gradient cases in {gradient_time:.3f} seconds")
print(f"   Gradient strengths: {gradient_strengths}")
print(f"   Max displacement range: {np.min(max_disps_gradient):.4f} to {np.max(max_disps_gradient):.4f}")
print(f"   Energy range: {np.min(energies_gradient):.2e} to {np.max(energies_gradient):.2e}")

# Case 3: Random material distributions
print("\n3. Random material distributions:")
num_cases = 8
onp.random.seed(42)  # Use numpy random instead of jax.numpy

lambda_arrays_random = []
for i in range(num_cases):
    # Random lambda values per cell (log-normal distribution)
    random_factors = onp.random.lognormal(0, 0.3, problem.num_cells)  # std=0.3 in log space
    lambda_per_cell = base_lambda * random_factors
    lambda_per_quad = np.repeat(lambda_per_cell, 8)
    lambda_arrays_random.append(lambda_per_quad)

lambda_arrays_random = np.array(lambda_arrays_random)

start_time = time.time()
max_disps_random, energies_random = vmapped_solve(lambda_arrays_random)
random_time = time.time() - start_time

print(f"   Solved {num_cases} random cases in {random_time:.3f} seconds")
print(f"   Max displacement range: {np.min(max_disps_random):.4f} to {np.max(max_disps_random):.4f}")
print(f"   Energy range: {np.min(energies_random):.2e} to {np.max(energies_random):.2e}")

# Performance comparison
print(f"\n=== Performance Summary ===")
total_cases = num_cases * 2 + 5 + 8  # uniform + gradient + random
total_time = uniform_time + gradient_time + random_time
print(f"Total: {total_cases} cases solved in {total_time:.3f} seconds")
print(f"Average: {total_time/total_cases:.4f} seconds per case")
print(f"Throughput: {total_cases/total_time:.1f} cases per second")

print(f"\nThis demonstrates the power of vmapped solving:")
print(f"- Parameter studies with different material distributions")
print(f"- Uncertainty quantification with random material properties") 
print(f"- Optimization with many design evaluations")
print(f"- All leveraging JAX's efficient vectorization!")