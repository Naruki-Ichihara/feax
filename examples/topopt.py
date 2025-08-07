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
from feax.mesh import box_mesh_gmsh
from feax.utils import save_sol
from feax.topopt_toolkit import create_compliance_fn, create_volume_fn, mdmm
import optax
import os
import time

# Problem setup
E0 = 70e3
E_eps = 1e-1
x_init = 0.0
target = 0.3
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
        # Note: For multiple surfaces, return a list like:
        # return [surface_map1, surface_map2, ...]
        # The universal compliance function will handle all surfaces automatically

# Create mesh and problem
meshio_mesh = box_mesh_gmsh(40, 20, 20, 2., 1., 1., data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right_corner(point):
    return np.where(np.logical_and(
                        np.logical_and(np.isclose(point[0], 2, atol=1e-5), 
                            np.logical_and(point[2] > 0, point[2] < 0.2)),
                        np.logical_and(point[1] > 0.40, point[1] < 0.6)), 
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

# Create cell-based design variables (one per element instead of per quadrature point)
# For sigmoid transformation, we work with unconstrained variables
# Initialize x such that sigmoid(x) = rho_0
# sigmoid(x) = 1/(1 + exp(-scale*x)) = rho_0
# => x = log(rho_0/(1-rho_0)) / scale
scale = 0.2
num_elements = mesh.cells.shape[0]
x_cell_0 = np.full(num_elements, x_init)

# Function to expand cell-based design variables to volume format for FEAX
def expand_cell_to_volume(x_cell):
    """Expand cell-based design variables to volume format expected by FEAX"""
    # Get the number of quadrature points per element from the problem
    num_quads = problem.fes[0].num_quads
    num_cells = len(x_cell)
    
    # Expand each cell value to all quadrature points within that cell
    x_expanded = np.repeat(x_cell[:, None], num_quads, axis=1)
    
    # Return in the format expected by FEAX (num_cells, num_quads)
    return x_expanded

# Convert to volume format for initial use
x_quad_0 = expand_cell_to_volume(x_cell_0)

print(f"Number of elements: {num_elements}")
print(f"Design variables reduced from quadrature points to: {num_elements} (one per element)")

# Helper functions for sigmoid transformation
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-scale * x))

# Create InternalVars for other variables
traction_array = InternalVars.create_uniform_surface_var(problem, T)

bc = bc_config.create_bc(problem)
solver_option = SolverOptions(tol=1e-8, linear_solver="cg", use_jacobi_preconditioner=True)
solver = create_solver(problem, bc, solver_option, iter_num=1)
initial_guess = zero_like_initial_guess(problem, bc)

# Create response functions using the universal API
# The compliance function now automatically handles all surfaces defined in the problem
# and applies the traction magnitude T to the surface map(s)
compute_compliance = create_compliance_fn(problem, surface_load_params=T)
compute_volume = create_volume_fn(problem)

# Create Helmholtz filter function (to be applied inside objective/constraint)
from feax.topopt_toolkit import create_helmholtz_filter
helmholtz_filter = create_helmholtz_filter(problem, radius=0.02)

# Define objective function (minimize compliance) - works with unconstrained x values
def objective(x_quad):
    rho_quad = sigmoid(x_quad)
    # Apply Helmholtz filter to density field before physics computation
    rho_filtered = helmholtz_filter(rho_quad)
    internal_vars = InternalVars(
        volume_vars=(rho_filtered,),
        surface_vars=[(traction_array,)]
    )
    sol = solver(internal_vars, initial_guess)
    return compute_compliance(sol)

# Define constraint function (volume constraint: target - V/V0 >= 0) - works with unconstrained x values
def volume_constraint(x_quad):
    rho_quad = sigmoid(x_quad)
    # Apply Helmholtz filter to density field before volume computation
    rho_filtered = helmholtz_filter(rho_quad)
    return target - compute_volume(rho_filtered)

# Modified objective and constraint functions to work with cell-based variables
def objective_cell(x_cell):
    """Objective function that works with cell-based design variables"""
    x_quad = expand_cell_to_volume(x_cell)
    return objective(x_quad)

def volume_constraint_cell(x_cell):
    """Volume constraint function that works with cell-based design variables"""
    x_quad = expand_cell_to_volume(x_cell)
    return volume_constraint(x_quad)

# Create MDMM inequality constraint using cell-based functions
volume_ineq = mdmm.ineq(volume_constraint_cell, damping=10, weight=5)
design_vars = {'x': x_cell_0}  # Use cell-based variables
constraint_params = volume_ineq.init(design_vars['x'])

@jax.jit
def mdmm_loss(design_vars, constraint_params):
    x = design_vars['x']
    obj = objective_cell(x)
    constraint_loss, constraint_val = volume_ineq.loss(constraint_params, x)
    total_loss = obj + constraint_loss
    return total_loss, (obj, constraint_val)

learning_rate = 0.1
optimizer = optax.chain(
    optax.adam(learning_rate),
    mdmm.optax_prepare_update()
)

# Initialize optimizer state
opt_state = optimizer.init({'design': design_vars, 'constraint': constraint_params})

# Create data directory
data_dir = os.path.join(os.path.dirname(__file__), 'data2')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)
vtk_path = os.path.join(data_dir, 'vtk/topopt_result.vtu')

# Optimization loop with growth curve visualization
num_iterations = 100
history = {'objective': [], 'volume': [], 'constraint_violation': [], 'iteration_time': []}

print("Starting topology optimization with MDMM...")
print(f"Target volume fraction: {target}")

# Import matplotlib for visualization
import matplotlib.pyplot as plt

# Track total optimization time
total_start_time = time.time()

for iteration in range(num_iterations):
    # Track iteration start time
    iter_start_time = time.time()
    
    # Compute gradients
    grad_start_time = time.time()
    (loss_val, aux), grads = jax.value_and_grad(mdmm_loss, argnums=(0, 1), has_aux=True)(
        design_vars, constraint_params
    )
    grad_time = time.time() - grad_start_time
    
    obj_val, constraint_val = aux
    
    # Update parameters - pass current params for proper box projection
    update_start_time = time.time()
    updates, opt_state = optimizer.update(
        {'design': grads[0], 'constraint': grads[1]}, 
        opt_state,
        {'design': design_vars, 'constraint': constraint_params}
    )
    
    # Apply updates
    design_vars = optax.apply_updates(design_vars, updates['design'])
    constraint_params = optax.apply_updates(constraint_params, updates['constraint'])


    update_time = time.time() - update_start_time
    
    # Record history
    history_start_time = time.time()
    current_x_cell = design_vars['x']
    current_x_quad = expand_cell_to_volume(current_x_cell)
    current_rho_quad = sigmoid(current_x_quad)
    current_rho_filtered = helmholtz_filter(current_rho_quad)
    current_volume = compute_volume(current_rho_filtered)
    history['objective'].append(float(obj_val))
    history['volume'].append(float(current_volume))
    history['constraint_violation'].append(float(constraint_val))
    history_time = time.time() - history_start_time
    
    # Calculate total iteration time
    iter_total_time = time.time() - iter_start_time
    history['iteration_time'].append(iter_total_time)
    
    # Print progress with timing information
    if iteration % 1 == 0:
        print(f"Iteration {iteration:3d}: Compliance = {obj_val:10.4f}, "
              f"Volume = {current_volume:6.4f}, Constraint = {constraint_val:8.4e}, "
              f"Time = {iter_total_time:6.4f}s (grad: {grad_time:5.3f}s, update: {update_time:5.3f}s)")
    
    # Save solution every 20 iterations
    if iteration % 2 == 0:
        current_x_cell = design_vars['x']
        current_x_quad = expand_cell_to_volume(current_x_cell)
        current_rho_quad = sigmoid(current_x_quad)
        current_rho_filtered = helmholtz_filter(current_rho_quad)
        internal_vars = InternalVars(
            volume_vars=(current_rho_filtered,),
            surface_vars=[(traction_array,)]
        )
        sol = solver(internal_vars, initial_guess)
        sol_unflat = problem.unflatten_fn_sol_list(sol)
        displacement = sol_unflat[0]
        
        iteration_vtk_path = os.path.join(data_dir, f'vtk/topopt_iter_{iteration:03d}.vtu')
        save_sol(
            mesh=mesh,
            sol_file=iteration_vtk_path,
            point_infos=[("displacement", displacement)],
            cell_infos=[("density", current_rho_filtered[:, 0])]
        )

# Calculate total optimization time
total_opt_time = time.time() - total_start_time

print("\nOptimization completed!")
print(f"Final compliance: {history['objective'][-1]:.4f}")
print(f"Final volume fraction: {history['volume'][-1]:.4f}")
print(f"\nTiming Statistics:")
print(f"Total optimization time: {total_opt_time:.2f}s")
iteration_times = np.array(history['iteration_time'])
print(f"Average time per iteration: {np.mean(iteration_times):.4f}s")
print(f"Min iteration time: {np.min(iteration_times):.4f}s")
print(f"Max iteration time: {np.max(iteration_times):.4f}s")

# Create growth curve visualization with timing information
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
iterations = range(len(history['objective']))

# Plot 1: Compliance (objective) over iterations
ax1.plot(iterations, history['objective'], 'b-', linewidth=2, label='Compliance')
ax1.set_ylabel('Compliance', fontsize=12)
ax1.set_xlabel('Iteration', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_title('Topology Optimization Results', fontsize=14, fontweight='bold')

# Plot 2: Volume fraction over iterations
ax2.plot(iterations, history['volume'], 'g-', linewidth=2, label='Volume Fraction')
ax2.axhline(y=target, color='r', linestyle='--', alpha=0.7, label=f'Target ({target})')
ax2.set_ylabel('Volume Fraction', fontsize=12)
ax2.set_xlabel('Iteration', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Constraint violation over iterations
ax3.plot(iterations, np.abs(np.array(history['constraint_violation'])), 'r-', linewidth=2, label='|Constraint Violation|')
ax3.set_ylabel('|Constraint Violation|', fontsize=12)
ax3.set_xlabel('Iteration', fontsize=12)
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Iteration time over iterations
ax4.plot(iterations, history['iteration_time'], 'purple', linewidth=2, label='Iteration Time')
ax4.axhline(y=np.mean(iteration_times), color='orange', linestyle='--', alpha=0.7, 
            label=f'Average ({np.mean(iteration_times):.3f}s)')
ax4.set_ylabel('Time per Iteration (s)', fontsize=12)
ax4.set_xlabel('Iteration', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_title(f'Total Time: {total_opt_time:.1f}s', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'topopt_growth_curves.png'), dpi=150, bbox_inches='tight')
print(f"\nGrowth curve plot saved to {os.path.join(data_dir, 'topopt_growth_curves.png')}")

# Save final solution
final_x_cell = design_vars['x']
final_x_quad = expand_cell_to_volume(final_x_cell)
final_rho_quad = sigmoid(final_x_quad)
final_rho_filtered = helmholtz_filter(final_rho_quad)
internal_vars = InternalVars(
    volume_vars=(final_rho_filtered,),
    surface_vars=[(traction_array,)]
)
sol = solver(internal_vars, initial_guess)
sol_unflat = problem.unflatten_fn_sol_list(sol)
displacement = sol_unflat[0]

final_vtk_path = os.path.join(data_dir, 'vtk/topopt_final.vtu')
save_sol(
    mesh=mesh,
    sol_file=final_vtk_path,
    point_infos=[("displacement", displacement)],
    cell_infos=[("density", final_rho_filtered[:, 0])]
)

print(f"Results saved to {final_vtk_path}")
print(f"Intermediate results saved every 20 iterations in {os.path.join(data_dir, 'vtk/')}")