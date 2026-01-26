"""
3D Topology Optimization using MDMM (Modified Differential Multiplier Method).

Minimizes compliance subject to volume constraint for a cantilever beam
with density-based material interpolation (SIMP).

Uses optax for optimization with MDMM for constraint handling.
"""

import feax as fe
import feax.gene as gene
from feax.gene.responses import create_compliance_fn, create_volume_fn
import jax
import jax.numpy as np
import os

# Material properties
elastic_moduli = 70e3
poisson_ratio = 0.3
penalty = 3  # SIMP penalization parameter
traction_mag = 1.0

# Optimization parameters
radius = 3 # Helmholtz filter radius (smaller = sharper boundaries)
target_fraction = 0.4  # Target volume fraction

# Geometry
L = 100  # Length
W = 4    # Width
H = 20   # Height
box_size = (L, W, H)
mesh = fe.mesh.box_mesh(box_size, mesh_size=2)
tol = 1e-3  # Boundary tolerance
    
# Locations
left = lambda point: np.isclose(point[0], 0., tol)
right = lambda point: np.isclose(point[0], L, tol) & (point[2] < H/4)

# Problem definition with SIMP material interpolation
E0 = elastic_moduli
nu = poisson_ratio
E_eps = E0 * 1e-6  # Minimum stiffness (avoid singularity)

class LinearElasticity(fe.problem.Problem):
    """Linear elasticity with density-dependent material properties (SIMP)."""

    def get_tensor_map(self):
        def stress(u_grad, rho):
            # SIMP interpolation: E(rho) = E_eps + (E0 - E_eps) * rho^p
            E = E_eps + (E0 - E_eps) * rho**penalty
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress

    def get_surface_maps(self):
        def surface_map(u, x, *args):
            return np.array([0., 0., -traction_mag])  # Downward traction
        return [surface_map]
    
problem = LinearElasticity(mesh, vec=3, dim=3, location_fns=[right])

# Boundary
left_fix = fe.DCboundary.DirichletBCSpec(
    location=left,
    component="all",
    value=0.
)
bc_config = fe.DCboundary.DirichletBCConfig([left_fix])
bc = bc_config.create_bc(problem)

# Solver configuration
# Forward solver: bicgstab with Jacobi preconditioning for better convergence
solver_options = fe.solver.SolverOptions(
    linear_solver="cudss_solver",
    linear_solver_tol=1e-10,
    linear_solver_atol=1e-12,
    use_jacobi_preconditioner=True,
    linear_solver_maxiter=10000,
)

# Adjoint solver: relaxed tolerance for gradient computation
adjoint_solver_options = fe.solver.SolverOptions(
    linear_solver="cudss_solver",
    tol=1e-6,
    linear_solver_tol=1e-8,
    linear_solver_atol=1e-10,
    use_jacobi_preconditioner=True,
    linear_solver_maxiter=10000,
)

solver = fe.solver.create_solver(
    problem, bc, solver_options,
    adjoint_solver_options=adjoint_solver_options,
    iter_num=1
)
initial = fe.utils.zero_like_initial_guess(problem, bc)

# Responses
compliance_fn = create_compliance_fn(problem)
volume_fn = create_volume_fn(problem)

# Create filter function once (outside of differentiated function)
# Filter uses node-based input and output
filter_fn = gene.filters.create_density_filter(mesh, radius)

def solve_forward(rho):
    """Compute compliance for given node-based density field."""
    rho_filtered = filter_fn(rho)
    internal_vars = fe.internal_vars.InternalVars(volume_vars=(rho_filtered,), surface_vars=())
    sol = solver(internal_vars, initial)
    compliance = compliance_fn(sol)
    return compliance

def evaluate_volume(rho):
    """Compute volume fraction for given node-based density field."""
    rho_filtered = filter_fn(rho)
    return volume_fn(rho_filtered)

# Initialize node-based density field
num_nodes = mesh.points.shape[0]
rho_init = fe.internal_vars.InternalVars.create_node_var(problem, target_fraction)

# ============================================================
# Optax + MDMM Optimization
# ============================================================
import optax
from feax.gene import mdmm
import numpy as onp
import matplotlib.pyplot as plt
import csv

# History tracking
history = {
    'iteration': [],
    'compliance': [],
    'volume': [],
    'constraint_violation': [],
    'lagrange_multiplier': [],
}

# Output directory
output_dir = "output_mdmm"
os.makedirs(output_dir, exist_ok=True)

# Optimization settings
max_iter = 1000
print_every = 1
save_every = 10
learning_rate = 0.05

# Define objective function (minimize compliance)
def objective_fn(params):
    """Objective: minimize compliance."""
    rho = params['rho']
    rho_filtered = filter_fn(rho)
    internal_vars = fe.internal_vars.InternalVars(volume_vars=(rho_filtered,), surface_vars=())
    sol = solver(internal_vars, initial)
    compliance = compliance_fn(sol)
    return compliance

# Define volume constraint: volume - target <= 0
# For MDMM inequality constraint h(x) >= 0, we need: target - volume >= 0
def volume_constraint_fn(params):
    """Constraint function: target - volume (should be >= 0)."""
    rho = params['rho']
    rho_filtered = filter_fn(rho)
    vol = volume_fn(rho_filtered)
    return target_fraction - vol  # >= 0 for MDMM ineq constraint

# Create MDMM constraint
constraint = mdmm.ineq(
    volume_constraint_fn,
    damping=10.0,  # Damping parameter (controls oscillation)
    weight=100.0,  # Weight relative to objective
)

# Initialize parameters
params = {
    'rho': rho_init,
}

# Initialize constraint parameters (Lagrange multipliers and slack variables)
constraint_params = constraint.init(params)

# Combine all parameters for optimization
all_params = {
    'rho': params['rho'],
    'constraint_params': constraint_params,
}

# Define combined loss function (objective + constraint)
def loss_fn(all_params):
    """Combined loss: objective + constraint penalty."""
    params = {'rho': all_params['rho']}

    # Compute objective
    obj = objective_fn(params)

    # Compute constraint loss
    constraint_loss, constraint_value = constraint.loss(
        all_params['constraint_params'],
        params
    )

    # Total loss
    total_loss = obj + constraint_loss

    return total_loss, (obj, constraint_value)

# JIT compile loss and gradient
loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

# Setup optimizer with MDMM
optimizer = optax.chain(
    optax.adam(learning_rate),
    mdmm.optax_prepare_update(),  # Converts gradient descent to descent-ascent for Lagrange multipliers
)

opt_state = optimizer.init(all_params)

# Box constraints projection (clip rho to [0.001, 1.0])
def project_params(params):
    """Project parameters to feasible set."""
    return {
        'rho': np.clip(params['rho'], 0.001, 1.0),
        'constraint_params': params['constraint_params'],  # No projection on constraint params
    }

# Optimization loop
print("Starting topology optimization with Optax + MDMM...")
print(f"Number of design variables: {num_nodes}")
print(f"Target volume fraction: {target_fraction}")
print(f"Learning rate: {learning_rate}")
print(f"Max iterations: {max_iter}")
print("-" * 70)

for iteration in range(max_iter):
    # Compute loss and gradients
    (loss_val, (obj_val, constraint_val)), grads = loss_and_grad(all_params)

    # Apply optimizer update
    updates, opt_state = optimizer.update(grads, opt_state, all_params)
    all_params = optax.apply_updates(all_params, updates)

    # Project to feasible set
    all_params = project_params(all_params)

    # Extract current values
    rho_current = all_params['rho']
    rho_filtered = filter_fn(rho_current)
    vol_current = float(volume_fn(rho_filtered))

    # Extract Lagrange multiplier value
    lambda_val = float(np.mean(np.abs(all_params['constraint_params']['lambda'].value)))

    # Record history
    history['iteration'].append(iteration + 1)
    history['compliance'].append(float(obj_val))
    history['volume'].append(vol_current)
    history['constraint_violation'].append(float(constraint_val))
    history['lagrange_multiplier'].append(lambda_val)

    # Print progress
    if (iteration + 1) % print_every == 0:
        print(f"Iter {iteration+1:4d}: "
              f"compliance = {float(obj_val):.4e}, "
              f"volume = {vol_current:.4f}, "
              f"constraint = {float(constraint_val):.4e}, "
              f"λ = {lambda_val:.4e}, "
              f"rho=[{float(np.min(rho_filtered)):.3f}, {float(np.max(rho_filtered)):.3f}]")

    # Save intermediate results
    if (iteration + 1) % save_every == 0:
        fe.utils.save_sol(
            mesh,
            f"{output_dir}/topology_opt_iter_{iteration+1:04d}.vtu",
            point_infos=[("density", onp.array(rho_filtered))]
        )

print("-" * 70)
print("Optimization finished!")
print(f"Final compliance: {float(obj_val):.4e}")
print(f"Final volume fraction: {vol_current:.4f}")
print(f"Target volume fraction: {target_fraction:.4f}")
print(f"Constraint violation: {float(constraint_val):.4e}")

# Save final result
rho_opt = all_params['rho']
rho_filtered_opt = filter_fn(rho_opt)
fe.utils.save_sol(
    mesh,
    f"{output_dir}/topology_opt_final.vtu",
    point_infos=[("density", onp.array(rho_filtered_opt))]
)
print(f"Final result saved to {output_dir}/topology_opt_final.vtu")

# Plot optimization history
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Compliance history
axes[0, 0].plot(history['iteration'], history['compliance'], 'b-', linewidth=1.5)
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Compliance')
axes[0, 0].set_title('Compliance History')
axes[0, 0].grid(True, alpha=0.3)

# Volume fraction history
axes[0, 1].plot(history['iteration'], history['volume'], 'g-', linewidth=1.5, label='Actual')
axes[0, 1].axhline(y=target_fraction, color='r', linestyle='--', linewidth=1.5, label=f'Target = {target_fraction}')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('Volume Fraction')
axes[0, 1].set_title('Volume Fraction History')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Constraint violation history
axes[1, 0].plot(history['iteration'], history['constraint_violation'], 'orange', linewidth=1.5)
axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('Constraint Violation')
axes[1, 0].set_title('Constraint Violation (target - volume)')
axes[1, 0].grid(True, alpha=0.3)

# Lagrange multiplier history
axes[1, 1].plot(history['iteration'], history['lagrange_multiplier'], 'purple', linewidth=1.5)
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('Lagrange Multiplier (mean |λ|)')
axes[1, 1].set_title('Lagrange Multiplier Evolution')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/optimization_history.png", dpi=150)
plt.close()
print(f"History plot saved to {output_dir}/optimization_history.png")

# Save history as CSV
with open(f"{output_dir}/optimization_history.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['iteration', 'compliance', 'volume', 'constraint_violation', 'lagrange_multiplier'])
    for i in range(len(history['iteration'])):
        writer.writerow([
            history['iteration'][i],
            history['compliance'][i],
            history['volume'][i],
            history['constraint_violation'][i],
            history['lagrange_multiplier'][i]
        ])
print(f"History data saved to {output_dir}/optimization_history.csv")
