"""FEAX equivalent of the reference problem in /workspace/ref.py

This example replicates the JAX-FEM reference problem using FEAX:
- 2D Poisson equation on unit square
- Periodic boundary conditions (left-right)  
- Dirichlet boundary conditions (top-bottom = 0)
- Mixed source term: x*sin(5πy) + exp(-((x-0.5)² + (y-0.5)²)/0.02)
- Parameter θ scaling the diffusion tensor
"""

import feax as fe
import feax.flat as flat
import jax
import jax.numpy as np

class PoissonParametric(fe.problem.Problem):
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
    periodic_pairing = flat.pbc.PeriodicPairing(
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
mesh = fe.mesh.rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    
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
P = flat.pbc.prolongation_matrix(periodic_pairings, mesh, vec=1)
    
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
bc_config = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(bottom_boundary, 0, zero_value),  # Bottom = 0
    fe.DCboundary.DirichletBCSpec(top_boundary, 0, zero_value),     # Top = 0  
])
    
bc = bc_config.create_bc(problem)
print(f"Dirichlet BCs: {len(bc.bc_rows)} constrained DOFs")

# 4. Use integrated reduced solver
from feax import create_solver, SolverOptions

theta = 1.0
theta_array = fe.internal_vars.InternalVars.create_uniform_volume_var(problem, theta)
internal_vars = fe.internal_vars.InternalVars(volume_vars=(theta_array,))

# Create reduced solver with P matrix for periodic constraints
solver_options = fe.solver.SolverOptions(tol=1e-8, linear_solver="cg")
solver = fe.solver.create_solver(problem, bc, solver_options=solver_options, iter_num=1, P=P)

print(f"Solving with θ = {theta}...")
# Initial guess in full space (required by solver API)
initial_guess = np.zeros(problem.num_total_dofs_all_vars)

# Solve with automatic periodic constraint handling
sol_full = solver(internal_vars, initial_guess)
print(f"Solution norm: {np.linalg.norm(sol_full):.6f}")

# Save solution
fe.utils.save_sol(
    mesh=mesh,
    sol_file="ref_equivalent_feax_solution.vtk",
    point_infos=[("u", sol_full.reshape(-1, 1))]
)