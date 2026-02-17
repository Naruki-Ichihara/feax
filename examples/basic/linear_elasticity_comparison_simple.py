"""
Simple comparison of FULL vs UPPER triangular matrix storage for linear elasticity.
Based on the original linear_elasticity.py example.
"""
import feax as fe
from feax.problem import MatrixView
import jax.numpy as jnp
import numpy as np
import time

print("=" * 80)
print("Linear Elasticity: FULL vs UPPER Matrix Storage Comparison")
print("=" * 80)

# Problem parameters (same as linear_elasticity.py)
elastic_moduli = 70e3
poisson_ratio = 0.3
traction = 1e-3
tol = 1e-5

# Define mesh
L = 100
W = 10
H = 10
box_size = (L, W, H)
mesh = fe.mesh.box_mesh(box_size, mesh_size=1)

print(f"\nMesh: {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} elements")

# Locations
left = lambda point: jnp.isclose(point[0], 0., tol)
right = lambda point: jnp.isclose(point[0], L, tol)

# Define problem
E = elastic_moduli
nu = poisson_ratio

class LinearElasticity(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, *args):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * jnp.trace(epsilon) * jnp.eye(self.dim) + 2 * mu * epsilon
        return stress

    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return jnp.array([0., 0., traction_mag])
        return [surface_map]

# Boundary conditions (shared)
left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
bc_config = fe.DirichletBCConfig([left_fix])

# Internal variables (shared)
traction_array_full = fe.InternalVars.create_uniform_surface_var(
    fe.Problem(mesh, vec=3, dim=3, location_fns=[right]), traction
)
internal_vars = fe.InternalVars(volume_vars=(), surface_vars=[(traction_array_full,)])

# ============================================================================
# Test 1: FULL matrix
# ============================================================================
print("\n" + "=" * 80)
print("Test 1: FULL Matrix Storage")
print("=" * 80)

problem_full = LinearElasticity(mesh, vec=3, dim=3, location_fns=[right])
bc_full = bc_config.create_bc(problem_full)
solver_full = fe.create_solver(problem_full, bc_full, iter_num=1)
initial_full = fe.zero_like_initial_guess(problem_full, bc_full)

print("Solving...")
start = time.time()
sol_full = solver_full(internal_vars, initial_full)
time_full = time.time() - start

sol_list_full = problem_full.unflatten_fn_sol_list(sol_full)
J_info_full = fe.get_jacobian_info(problem_full, sol_list_full, internal_vars)

print(f"Jacobian NNZ: {J_info_full['nnz']:,}")
print(f"Solve time: {time_full:.4f}s")
print(f"Max displacement: {jnp.max(jnp.abs(sol_full)):.6e}")

# ============================================================================
# Test 2: UPPER triangular matrix
# ============================================================================
print("\n" + "=" * 80)
print("Test 2: UPPER Triangular Matrix Storage")
print("=" * 80)

problem_upper = LinearElasticity(
    mesh, vec=3, dim=3, location_fns=[right],
    matrix_view=MatrixView.UPPER
)
bc_upper = bc_config.create_bc(problem_upper)
solver_opts_upper = fe.SolverOptions.from_problem(problem_upper)
solver_upper = fe.create_solver(problem_upper, bc_upper, solver_options=solver_opts_upper, iter_num=1)
initial_upper = fe.zero_like_initial_guess(problem_upper, bc_upper)

print(f"Solver auto-configured:")
print(f"  matrix_type: {solver_opts_upper.cudss_options.matrix_type.name}")
print(f"  matrix_view: {solver_opts_upper.cudss_options.matrix_view.name}")

print("Solving...")
start = time.time()
sol_upper = solver_upper(internal_vars, initial_upper)
time_upper = time.time() - start

sol_list_upper = problem_upper.unflatten_fn_sol_list(sol_upper)
J_info_upper = fe.get_jacobian_info(problem_upper, sol_list_upper, internal_vars)

print(f"Jacobian NNZ: {J_info_upper['nnz']:,}")
print(f"Solve time: {time_upper:.4f}s")
print(f"Max displacement: {jnp.max(jnp.abs(sol_upper)):.6e}")

# ============================================================================
# Comparison
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

print(f"\nMemory Usage:")
print(f"  FULL:   {J_info_full['nnz']:,} entries")
print(f"  UPPER:  {J_info_upper['nnz']:,} entries")
print(f"  Reduction: {(1 - J_info_upper['nnz'] / J_info_full['nnz']) * 100:.1f}%")

print(f"\nSolution Accuracy:")
diff = jnp.linalg.norm(sol_full - sol_upper)
norm = jnp.linalg.norm(sol_full)
rel_error = diff / norm
print(f"  Difference: {diff:.3e}")
print(f"  Relative error: {rel_error:.3e}")

if rel_error < 1e-10:
    print("  ✓ Solutions are IDENTICAL!")
else:
    print(f"  ⚠ Solutions differ by {rel_error:.2e}")

print("\n" + "=" * 80)
print("✓ Test completed successfully!")
print("=" * 80)
