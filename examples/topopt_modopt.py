"""
Linear elasticity topology optimization example with SIMP-based material interpolation.
Uses MDMM (Modified Differential Multiplier Method) optimizer with optax.

Features:
- Universal compliance function that handles any number of surface loads
- JIT-compiled response functions for performance
- Helmholtz filtering for regularization
- Real-time timing information for each optimization iteration
"""

import jax
import jax.numpy as np
from feax import Problem, InternalVars, create_solver
from feax import Mesh, SolverOptions, zero_like_initial_guess
from feax import DirichletBCSpec, DirichletBCConfig
from feax.mesh import box_mesh
from feax.utils import save_sol
from feax.topopt_toolkit import create_compliance_fn, create_volume_fn
import os
import modopt as mo
jax.config.update("jax_enable_x64", True)  # Use 64-bit precision for better accuracy

# Problem setup
E0 = 70e3
E_eps = 1e-1
x_init = 0.5
target = 0.2
T = 1e3
nu = 0.3
p = 3

class ElasticityProblem(Problem):
    def get_tensor_map(self):
        def stress(u_grad, rho):
            E = (E0 - E_eps) * rho**p + E_eps
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress
    
    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return np.array([0., 0., traction_mag])
        return [surface_map]

# Create mesh and problem
mesh = box_mesh(40, 20, 20, 2., 1., 1.)

def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right_corner(point):
    return np.where(np.logical_and(
                        np.logical_and(np.isclose(point[0], 2, atol=1e-5), 
                            np.logical_and(point[2] > 0, point[2] < 0.2)),
                        np.logical_and(point[1] > 0.8, point[1] < 1.1)), 
                    True, False)

# Create boundary conditions using the new dataclass approach
bc_config = DirichletBCConfig([
    # Fix left boundary completely (all components to zero)
    DirichletBCSpec(
        location=left,
        component='all',  # Fix x, y, z components
        value=0.0
    )
])

problem = ElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    location_fns=[right_corner])

# Create InternalVars for other variables
traction_array = InternalVars.create_uniform_surface_var(problem, T)

bc = bc_config.create_bc(problem)
solver_option = SolverOptions(tol=1e-8, linear_solver="cg", use_jacobi_preconditioner=True)
solver = create_solver(problem, bc, solver_option, iter_num=1)
initial_guess = zero_like_initial_guess(problem, bc)

compute_compliance = create_compliance_fn(problem, surface_load_params=T)
compute_volume = create_volume_fn(problem)

# Create Helmholtz filter function (to be applied inside objective/constraint)
from feax.topopt_toolkit import create_helmholtz_filter
helmholtz_filter = create_helmholtz_filter(problem, radius=0.02)

# Define objective function (minimize compliance) - works with unconstrained x values
def objective(x_quad):
    rho_filtered = helmholtz_filter(x_quad)
    internal_vars = InternalVars(
        volume_vars=(rho_filtered,),
        surface_vars=[(traction_array,)]
    )
    sol = solver(internal_vars, initial_guess)
    return compute_compliance(sol)

# Define constraint function (volume constraint: target - V/V0 >= 0) - works with unconstrained x values
def volume_constraint(x_quad):\
    # Apply Helmholtz filter to density field before volume computation
    rho_filtered = helmholtz_filter(x_quad)
    return target - compute_volume(rho_filtered)

import numpy as onp

# Create design variables - one per element (cell) instead of per quadrature point
num_elements = mesh.cells.shape[0]
x_cell = onp.full(num_elements, x_init)  # One design variable per element
xl_cell = onp.zeros(num_elements)        # Lower bounds per element
xu_cell = onp.ones(num_elements)         # Upper bounds per element

print(f"Number of elements: {num_elements}")
print(f"Design variables reduced from quad points to: {num_elements} (one per element)")

# Design variables are already 1D arrays (one per element)
x_quad_0 = x_cell
xl = xl_cell
xu = xu_cell

print(f"Design variable shape: {x_quad_0.shape}")

# Function to expand cell-based design variables to volume format for FEAX
def expand_cell_to_volume(x_cell_1d):
    """Expand cell-based design variables to volume format expected by FEAX"""
    # Get the number of quadrature points per element from the problem
    num_quads = problem.fes[0].num_quads
    num_cells = len(x_cell_1d)
    
    # Expand each cell value to all quadrature points within that cell
    x_expanded = onp.repeat(x_cell_1d[:, None], num_quads, axis=1)
    
    # Return in the format expected by FEAX (num_cells, num_quads)
    return x_expanded

# Wrapper functions that handle cell-based design variables
def objective_1d(x_1d):
    """Objective wrapper that converts cell-based design variables to FEAX volume format"""
    x_volume = expand_cell_to_volume(x_1d)
    return objective(x_volume)

def constraint_1d(x_1d):
    """Constraint wrapper that converts cell-based design variables to FEAX volume format"""
    x_volume = expand_cell_to_volume(x_1d)
    constraint_value = volume_constraint(x_volume)
    # ModOpt requires constraints to be 1D arrays with shape (nc,)
    return np.array([constraint_value])

prob = mo.JaxProblem(x_quad_0, nc=1, jax_obj=objective_1d, jax_con=constraint_1d,
                     xl=xl, xu=xu,
                     cl=0, cu=np.inf, 
                     name='topopt_jax', order=1)

from modopt import IPOPT

# Configure optimizer with recording enabled
optimizer = IPOPT(prob, solver_options={'max_iter': 40, 'tol': 1e-4})

print("\n" + "="*60)
print("PROCESSING: Optimization is progress")
print("="*60)
results = optimizer.solve()

# Print results of optimization
optimizer.print_results()

# Post-processing: Save final solution using FEAX save_sol
print("\n" + "="*60)
print("POST-PROCESSING: Saving Final Solution")
print("="*60)

# Get final design variables from results and convert to FEAX volume format
x_final_1d = results['x']
x_final_volume = expand_cell_to_volume(x_final_1d)

# Apply Helmholtz filter to get final density field
rho_final = helmholtz_filter(x_final_volume)

# Create output directory
save_dir = "./data"
os.makedirs(save_dir, exist_ok=True)

# Save density field using FEAX save_sol
density_file = os.path.join(save_dir, "final_density.vtu")
save_sol(mesh, density_file, cell_infos=[("density", rho_final)])

# Compute and save final structural solution
internal_vars_final = InternalVars(
    volume_vars=(rho_final,),
    surface_vars=[(traction_array,)]
)
sol_final = solver(internal_vars_final, initial_guess)

# Save displacement field - reshape to (n_nodes, 3) for 3D visualization
n_nodes = mesh.points.shape[0]
displacement_reshaped = sol_final.reshape((n_nodes, 3))
displacement_file = os.path.join(save_dir, "final_displacement.vtu")
save_sol(mesh, displacement_file, point_infos=[("displacement", displacement_reshaped)])

# Compute final metrics
final_compliance = float(compute_compliance(sol_final))
final_volume = float(compute_volume(rho_final))

print(f"Final Results:")
print(f"  Compliance: {final_compliance:.6e}")
print(f"  Volume fraction: {final_volume:.3f} (target: {target:.3f})")
print(f"  Volume constraint: {target - final_volume:.6e} (should be ≥ 0)")

print(f"\nOutput files saved to: {save_dir}")
print(f"  Density field:     {density_file}")
print(f"  Displacement field: {displacement_file}")