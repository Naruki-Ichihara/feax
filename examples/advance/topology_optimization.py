"""
3D Topology Optimization using MMA (Method of Moving Asymptotes).

Minimizes compliance subject to volume constraint for a cantilever beam
with density-based material interpolation (SIMP).
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
W = 20    # Width
H = 20   # Height
box_size = (L, W, H)
mesh = fe.mesh.box_mesh(box_size, mesh_size=0.5)
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
# Forward solver: CUDSS with Jacobi preconditioning for better convergence
solver_options = fe.solver.SolverOptions(
    linear_solver="cudss",
    linear_solver_tol=1e-10,
    linear_solver_atol=1e-12,
    use_jacobi_preconditioner=True,
    linear_solver_maxiter=10000,
)

# Adjoint solver: relaxed tolerance for gradient computation
adjoint_solver_options = fe.solver.SolverOptions(
    linear_solver="cudss",
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
# NLopt MMA Optimization
# ============================================================
import nlopt
import numpy as onp
import matplotlib.pyplot as plt
import csv

# JIT compile forward and gradient functions
forward_jit = jax.jit(solve_forward)
volume_jit = jax.jit(evaluate_volume)
grad_compliance_jit = jax.jit(jax.grad(solve_forward))
grad_volume_jit = jax.jit(jax.grad(evaluate_volume))

# History tracking
history = {
    'iteration': [],
    'compliance': [],
    'volume': [],
}

# Output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Optimization settings
max_iter = 100
print_every = 1
save_every = 1

iteration_count = [0]  # Mutable counter for nested function

def objective(x, grad):
    """NLopt objective: minimize compliance."""
    rho = np.array(x)

    # Compute objective and gradient
    obj_val = float(forward_jit(rho))
    grad[:] = onp.array(grad_compliance_jit(rho))

    # Record and print progress
    iteration_count[0] += 1
    vol = float(volume_jit(rho))
    history['iteration'].append(iteration_count[0])
    history['compliance'].append(obj_val)
    history['volume'].append(vol)

    if iteration_count[0] % print_every == 0:
        rho_filtered = filter_fn(rho)
        print(f"Iter {iteration_count[0]:4d}: compliance = {obj_val:.4e}, volume = {vol:.4f}, "
              f"rho=[{float(np.min(rho_filtered)):.3f}, {float(np.max(rho_filtered)):.3f}]")

    # Save intermediate results
    if iteration_count[0] % save_every == 0:
        rho_filtered = filter_fn(rho)
        fe.utils.save_sol(
            mesh,
            f"{output_dir}/topology_opt_iter_{iteration_count[0]:04d}.vtu",
            point_infos=[("density", rho_filtered)]
        )

    return obj_val

def volume_constraint(x, grad):
    """NLopt constraint: volume - target <= 0."""
    rho = np.array(x)
    vol = float(volume_jit(rho))

    grad[:] = onp.array(grad_volume_jit(rho))

    return vol - target_fraction

# Setup NLopt MMA optimizer
opt = nlopt.opt(nlopt.LD_MMA, num_nodes)
opt.set_lower_bounds(0.001)  # Small lower bound (filter smooths to avoid ill-conditioning)
opt.set_upper_bounds(1.0)
opt.set_min_objective(objective)
opt.add_inequality_constraint(volume_constraint, 1e-8)
opt.set_maxeval(max_iter)

# Run optimization
print("Starting topology optimization with NLopt MMA...")
print(f"Number of design variables: {num_nodes}")
print(f"Target volume fraction: {target_fraction}")
print("-" * 60)

x0 = onp.array(rho_init)
try:
    x_opt = opt.optimize(x0)
    opt_val = opt.last_optimum_value()
except nlopt.RoundoffLimited:
    print("Optimization stopped due to roundoff errors (converged)")
    x_opt = x0  # Use last state
    opt_val = float(forward_jit(np.array(x_opt)))

print("-" * 60)
print("Optimization finished!")
print(f"Final compliance: {opt_val:.4e}")
print(f"Final volume fraction: {float(volume_jit(np.array(x_opt))):.4f}")

# Save final result
rho_opt = np.array(x_opt)
rho_filtered_opt = filter_fn(rho_opt)
fe.utils.save_sol(
    mesh,
    f"{output_dir}/topology_opt_final.vtu",
    point_infos=[("density", onp.array(rho_filtered_opt))]
)
print(f"Final result saved to {output_dir}/topology_opt_final.vtu")

# Plot optimization history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history['iteration'], history['compliance'], 'b-', linewidth=1.5)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Compliance')
axes[0].set_title('Compliance History')
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['iteration'], history['volume'], 'g-', linewidth=1.5)
axes[1].axhline(y=target_fraction, color='r', linestyle='--', label=f'Target = {target_fraction}')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Volume Fraction')
axes[1].set_title('Volume Fraction History')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/optimization_history.png", dpi=150)
plt.close()
print(f"History plot saved to {output_dir}/optimization_history.png")

# Save history as CSV
with open(f"{output_dir}/optimization_history.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['iteration', 'compliance', 'volume'])
    for i in range(len(history['iteration'])):
        writer.writerow([history['iteration'][i], history['compliance'][i], history['volume'][i]])
print(f"History data saved to {output_dir}/optimization_history.csv")
