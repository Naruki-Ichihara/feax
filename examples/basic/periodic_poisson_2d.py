"""FEAX equivalent of the reference problem in /workspace/ref.py

This example replicates the JAX-FEM reference problem using FEAX:
- 2D Poisson equation on unit square
- Periodic boundary conditions (left-right)  
- Dirichlet boundary conditions (top-bottom = 0)
- Mixed source term: x*sin(5πy) + exp(-((x-0.5)² + (y-0.5)²)/0.02)
- Parameter θ scaling the diffusion tensor
"""

import jax
import jax.numpy as np
import numpy as onp
from feax import Problem, SolverOptions, InternalVars
from feax import create_solver
from feax.mesh import rectangle_mesh
from feax.lattice_toolkit.pbc import PeriodicPairing, prolongation_matrix
from feax import DirichletBCSpec, DirichletBCConfig
from feax.utils import save_sol

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

class PoissonParametric(Problem):
    """Parametric 2D Poisson problem equivalent to ref.py."""
    
    def get_tensor_map(self):
        """Diffusion tensor scaled by parameter θ."""
        def tensor_map(u_grad, theta):
            # Diffusion coefficient is θ
            return theta * u_grad
        return tensor_map
    
    def get_mass_map(self):
        """Source term: x*sin(5πy) + exp(-((x-0.5)² + (y-0.5)²)/0.02)"""
        def mass_map(u, x, theta):
            # Mixed source term from ref.py
            dx = x[0] - 0.5
            dy = x[1] - 0.5
            val = x[0]*np.sin(5.0*np.pi*x[1]) + 1.0*np.exp(-(dx*dx + dy*dy)/0.02)
            return np.array([-val])  # Negative to match ref.py sign convention
        return mass_map

def create_periodic_bc_2d(mesh):
    """Create periodic boundary conditions for left-right boundaries."""
    
    def left_boundary(point):
        return np.isclose(point[0], 0.0, atol=1e-5)
    
    def right_boundary(point):
        return np.isclose(point[0], 1.0, atol=1e-5)
    
    def mapping_x(point_A):
        """Map left boundary to right boundary."""
        return np.array([point_A[0] + 1.0, point_A[1]])
    
    # Create periodic pairing for x-direction (component 0)
    periodic_pairing = PeriodicPairing(
        location_master=left_boundary,
        location_slave=right_boundary, 
        mapping=mapping_x,
        vec=0  # x-component
    )
    
    return [periodic_pairing]


# 1. Create 2D mesh (unit square)
print("Creating mesh...")
Nx, Ny = 32, 32  
Lx, Ly = 1.0, 1.0
mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    
print(f"Mesh: {len(mesh.points)} nodes, {len(mesh.cells)} elements")
ele_type = 'QUAD4'
    
problem = PoissonParametric(
    mesh=mesh,
    vec=1,
    dim=2,
    ele_type=ele_type,
    location_fns=[]
)
    
# 2. Setup periodic boundary conditions (left-right)
print("Setting up periodic boundary conditions...")
periodic_pairings = create_periodic_bc_2d(mesh)
P = prolongation_matrix(periodic_pairings, mesh, vec=1)
    
print(f"Prolongation matrix: {P.shape}")
print(f"DOF reduction: {P.shape[0]} -> {P.shape[1]} ({P.shape[1]/P.shape[0]:.1%})")
    
# 3. Setup Dirichlet boundary conditions (top-bottom = 0)
print("Setting up Dirichlet boundary conditions...")
    
def bottom_boundary(point):
    return np.isclose(point[1], 0.0, atol=1e-5)
    
def top_boundary(point):
    return np.isclose(point[1], 1.0, atol=1e-5)
    
def zero_value(point):
    return 0.0
    
# Create Dirichlet BC config
bc_config = DirichletBCConfig([
    DirichletBCSpec(bottom_boundary, 0, zero_value),  # Bottom = 0
    DirichletBCSpec(top_boundary, 0, zero_value),     # Top = 0  
])
    
bc = bc_config.create_bc(problem)
print(f"Dirichlet BCs: {len(bc.bc_rows)} constrained DOFs")

# 4. Use integrated reduced solver
from feax import create_solver, SolverOptions

theta = 1.0
theta_array = InternalVars.create_uniform_volume_var(problem, theta)
internal_vars = InternalVars(volume_vars=(theta_array,))

# Create reduced solver with P matrix for periodic constraints
solver_options = SolverOptions(tol=1e-8, linear_solver="cg")
solver = create_solver(problem, bc, solver_options=solver_options, iter_num=1, P=P)

print(f"Solving with θ = {theta}...")
# Initial guess in full space (required by solver API)
initial_guess = np.zeros(problem.num_total_dofs_all_vars)

# Solve with automatic periodic constraint handling
sol_full = solver(internal_vars, initial_guess)
print(f"Solution norm: {np.linalg.norm(sol_full):.6f}")

# Test automatic differentiation accuracy
print("\nTesting AD accuracy...")

def loss_fn(theta_val):
    """Loss function: sum of squared solution values."""
    theta_arr = InternalVars.create_uniform_volume_var(problem, theta_val)
    iv = InternalVars(volume_vars=(theta_arr,))
    sol = solver(iv, initial_guess)
    return np.sum(sol**2)

# Compute analytical gradient using AD
grad_fn = jax.grad(loss_fn)
grad_ad = grad_fn(theta)
print(f"AD gradient at θ={theta}: {grad_ad:.6f}")

# Verify with finite differences
eps = 1e-6
loss_plus = loss_fn(theta + eps)
loss_minus = loss_fn(theta - eps)
grad_fd = (loss_plus - loss_minus) / (2 * eps)
print(f"FD gradient at θ={theta}: {grad_fd:.6f}")

# Check relative error
rel_error = abs(grad_ad - grad_fd) / abs(grad_fd)
print(f"Relative error: {rel_error:.2e}")
print(f"AD accuracy: {'✓ PASS' if rel_error < 1e-4 else '✗ FAIL'}")

# Test JIT compatibility
print("\nTesting JIT compatibility...")
import time

# Create JIT-compiled solver
@jax.jit
def jit_solve(internal_vars):
    return solver(internal_vars, initial_guess)

@jax.jit  
def jit_loss_and_grad(theta_val):
    theta_arr = InternalVars.create_uniform_volume_var(problem, theta_val)
    iv = InternalVars(volume_vars=(theta_arr,))
    sol = solver(iv, initial_guess)
    loss = np.sum(sol**2)
    return loss, jax.grad(lambda t: loss_fn(t))(theta_val)

# First call (compilation time)
print("Compiling JIT functions...")
start_time = time.time()
sol_jit = jit_solve(internal_vars)
compile_time = time.time() - start_time

# Second call (execution time)  
start_time = time.time()
sol_jit2 = jit_solve(internal_vars)
exec_time = time.time() - start_time

# Verify JIT correctness
jit_error = np.linalg.norm(sol_jit - sol_full)
print(f"JIT compilation time: {compile_time:.3f}s")
print(f"JIT execution time: {exec_time:.6f}s")
print(f"JIT solution error: {jit_error:.2e}")
print(f"JIT compatibility: {'✓ PASS' if jit_error < 1e-12 else '✗ FAIL'}")

# Save solution
save_sol(
    mesh=mesh,
    sol_file="ref_equivalent_feax_solution.vtk",
    point_infos=[("u", sol_full.reshape(-1, 1))]
)