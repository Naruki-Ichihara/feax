"""
2D Poisson equation solver using FEAX - Equivalent to JAX-FEM ref_poisson.py

Solves:
    -div(∇u) = b    in Ω = [0,1] × [0,1]
    
where:
    b = -10*exp(-((x-0.5)² + (y-0.5)²)/0.02)  (Gaussian source)
    
Boundary conditions:
    u = 0 on left and right boundaries (Dirichlet)
    ∂u/∂n = sin(5x) on bottom and top boundaries (Neumann)
"""

import jax.numpy as np
from feax import Problem, InternalVars, create_solver
from feax import Mesh, SolverOptions, zero_like_initial_guess
from feax import DirichletBCSpec, DirichletBCConfig
from feax.mesh import rectangle_mesh
from feax.utils import save_sol
import os


class Poisson(Problem):
    """2D Poisson problem implementation matching JAX-FEM example."""
    
    def get_tensor_map(self):
        """Identity tensor for standard Poisson equation."""
        # JAX-FEM solves -div.f(u_grad) = b, with f being identity
        # FEAX expects the function to accept internal variables, even if not used
        def tensor_map(u_grad, *args):
            return u_grad
        return tensor_map
    
    def get_mass_map(self):
        """Define the source term b."""
        def mass_map(u, x, *args):
            # Source term: b = -10*exp(-((x-0.5)² + (y-0.5)²)/0.02)
            val = -10.0 * np.exp(-(np.power(x[0] - 0.5, 2) + np.power(x[1] - 0.5, 2)) / 0.02)
            return np.array([val])
        return mass_map
    
    def get_surface_maps(self):
        """Define Neumann boundary conditions."""
        def surface_map(u, x, *args):
            # Neumann BC: ∂u/∂n = sin(5x)
            return -np.array([np.sin(5.0 * x[0])])
        
        # Return list with one function per boundary location
        return [surface_map, surface_map]  # bottom, top


# Define boundary locations
def left(point):
    return np.isclose(point[0], 0.0, atol=1e-5)

def right(point):
    return np.isclose(point[0], 1.0, atol=1e-5)

def bottom(point):
    return np.isclose(point[1], 0.0, atol=1e-5)

def top(point):
    return np.isclose(point[1], 1.0, atol=1e-5)


# Define Dirichlet boundary values
def dirichlet_val_left(point):
    return 0.0

def dirichlet_val_right(point):
    return 0.0


ele_type = 'QUAD4'
Lx, Ly = 1.0, 1.0
Nx, Ny = 32, 32
    
# Create mesh
print("Creating 2D mesh...")
meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'])
    
print(f"Mesh info: {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} elements")
    
# Create boundary conditions using dataclass approach
bc_config = DirichletBCConfig([
    # Left boundary: u = 0
    DirichletBCSpec(location=left, component=0, value=dirichlet_val_left),
    # Right boundary: u = 0  
    DirichletBCSpec(location=right, component=0, value=dirichlet_val_right)
])
    
# Define Neumann boundary locations
location_fns = [bottom, top]
    
# Create problem instance
problem = Poisson(
    mesh=mesh,
    vec=1,  # Scalar field
    dim=2,  # 2D problem
    ele_type=ele_type,
    location_fns=location_fns
)
    
# Create boundary conditions
bc = bc_config.create_bc(problem)
    
# Create internal variables
internal_vars = InternalVars()
    
# Create solver
solver_options = SolverOptions(
    tol=1e-10,
    linear_solver="cg"  # Default linear solver
)
solver = create_solver(problem, bc, solver_options, iter_num=1)
    
# Solve
print("\nSolving Poisson equation...")
solution = solver(internal_vars, zero_like_initial_guess(problem, bc))
    
print(f"Solution shape: {solution.shape}")
print(f"Expected shape: {problem.num_total_dofs_all_vars}")
    
# Extract solution at nodes
sol_list = problem.unflatten_fn_sol_list(solution)
u = sol_list[0]  # Shape: (num_nodes, vec) = (num_nodes, 1)
    
# Save solution to VTK file - matching JAX-FEM output format
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)
vtk_path = os.path.join(data_dir, 'vtk/u.vtu')
    
# FEAX save_sol expects mesh as first argument
save_sol(mesh, vtk_path, point_infos=[("u", u)])