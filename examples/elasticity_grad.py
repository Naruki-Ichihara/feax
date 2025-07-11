import jax
import jax.numpy as np
import jax.scipy as scipy
import matplotlib.pyplot as plt

from feax.problem import Problem
from feax.generate_mesh import box_mesh_gmsh, Mesh
from feax.boundary_conditions import apply_bc, prepare_bc_info, FixedBC, create_boundary_functions

nu = 0.3

class LinearElasticity(Problem):
    def get_tensor_map(self):
        def stress(u_grad, internal_vars):
            E_val = internal_vars[0]
            mu = E_val / (2. * (1. + nu))
            lmbda = E_val * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress

    def get_surface_maps(self):
        def surface_map(u, x, internal_vars):
            return np.array([0., 0., 100.])
        return [surface_map]

# Setup
print("Setting up problem...")
meshio_mesh = box_mesh_gmsh(Nx=10, Ny=2, Nz=2, Lx=10., Ly=2., Lz=2., 
                           data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])
boundary_fns = create_boundary_functions(10., 2., 2.)
bc_data = prepare_bc_info(mesh, FixedBC(boundary_fns['left'], components=[0, 1, 2]), vec_dim=3)
problem = LinearElasticity(mesh, vec=3, dim=3, ele_type='HEX8', location_fns=[boundary_fns['right']])
sol_list = np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))

@jax.jit
def solve(E_value):
    """Direct solve function that takes E_value as input"""
    from feax.problem import get_sparse_system
    
    # Create internal vars directly with E_value
    E_array = np.full((problem.num_cells * 8,), E_value)
    state = problem.get_functional_state(internal_vars=[E_array])
    
    A, b = get_sparse_system(state, jax.flatten_util.ravel_pytree(sol_list)[0])
    A_bc, b_bc = apply_bc(A, b, bc_data)
    # Convert sparse matrix to dense for direct solver
    A_dense = A_bc.todense()
    x_sol = scipy.linalg.solve(A_dense, b_bc)
    sol_final = state.unflatten_fn_sol_list(x_sol)
    return np.max(np.linalg.norm(sol_final[0], axis=1))

def max_displacement(E_value):
    """Function that maps E value to max displacement"""
    return solve(E_value)

# Create derivative function using JAX autodiff
d_disp_dE = jax.grad(max_displacement)

# Test multiple E values using vmap with jit
batch_size = 100
E_values = np.linspace(10e3, 100e3, batch_size)

# Create vmapped and jitted functions for batch computation
batch_displacement = jax.vmap(max_displacement)
batch_gradient = jax.vmap(d_disp_dE)

# JIT compile the batched functions
batch_displacement_jit = jax.jit(batch_displacement)
batch_gradient_jit = jax.jit(batch_gradient)

print("Computing displacements and gradients with vmap+jit...")
import time

# Warm up JIT compilation
_ = batch_displacement_jit(E_values[:1])
_ = batch_gradient_jit(E_values[:1])

# Time the batched computation
start_time = time.time()
displacements = batch_displacement_jit(E_values)
gradients = batch_gradient_jit(E_values)
batch_time = time.time() - start_time

print(f"Batched computation time: {batch_time:.4f} seconds")

# Compare with sequential computation
print("\nTiming comparison with sequential computation...")
start_time = time.time()
seq_displacements = []
seq_gradients = []
for E_val in E_values:
    seq_displacements.append(max_displacement(E_val))
    seq_gradients.append(d_disp_dE(E_val))
seq_time = time.time() - start_time

print(f"Sequential computation time: {seq_time:.4f} seconds")
print(f"Speedup: {seq_time/batch_time:.2f}x")

# Print results
for i, E_val in enumerate(E_values):
    print(f"E={E_val:.0f}: u={displacements[i]:.6f}, du/dE={gradients[i]:.6e}")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot E vs displacement
ax1.plot(E_values, displacements, 'b-o', linewidth=2, markersize=8)
ax1.set_xlabel('Young\'s Modulus E [Pa]')
ax1.set_ylabel('Max Displacement [m]')
ax1.set_title('E vs Displacement')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_yscale('log')

# Plot E vs gradient (absolute value for log scale)
ax2.plot(E_values, np.abs(np.array(gradients)), 'r-s', linewidth=2, markersize=8)
ax2.set_xlabel('Young\'s Modulus E [Pa]')
ax2.set_ylabel('|du/dE| [m/Pa]')
ax2.set_title('Gradient: |du/dE|')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('elasticity_gradient.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nGradient computation successful! Plot saved as 'elasticity_gradient.png'")