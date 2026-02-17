"""
Simple test to verify UPPER triangular matrix storage works correctly.
"""
import feax as fe
from feax.problem import MatrixView
import jax.numpy as jnp

# Same setup as linear_elasticity.py
elastic_moduli = 70e3
poisson_ratio = 0.3
traction = 1e-3
tol = 1e-5

L = 100
W = 10
H = 10
box_size = (L, W, H)
mesh = fe.mesh.box_mesh(box_size, mesh_size=2)  # Smaller mesh for faster testing

print(f"Mesh: {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} elements")

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

print("\n" + "="*60)
print("Test 1: FULL matrix (default)")
print("="*60)

problem_full = LinearElasticity(mesh, vec=3, dim=3, location_fns=[right])

# Boundary
left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
bc_config = fe.DirichletBCConfig([left_fix])
bc_full = bc_config.create_bc(problem_full)

# Solver
solver_full = fe.create_solver(problem_full, bc_full, iter_num=1)
initial = fe.zero_like_initial_guess(problem_full, bc_full)

# Internal vars
traction_array = fe.InternalVars.create_uniform_surface_var(problem_full, traction)
internal_vars = fe.InternalVars(volume_vars=(), surface_vars=[(traction_array,)])

print("Solving FULL...")
sol_full = solver_full(internal_vars, initial)

# Check matrix size
sol_list = problem_full.unflatten_fn_sol_list(sol_full)
J_info_full = fe.get_jacobian_info(problem_full, sol_list, internal_vars)
print(f"Jacobian NNZ: {J_info_full['nnz']:,}")
print(f"Max displacement: {jnp.max(jnp.abs(sol_full)):.6e}")

print("\n" + "="*60)
print("Test 2: UPPER triangular matrix")
print("="*60)

problem_upper = LinearElasticity(
    mesh, vec=3, dim=3, location_fns=[right],
    matrix_view=MatrixView.UPPER  # Upper triangular storage
)

bc_upper = bc_config.create_bc(problem_upper)

# Use SolverOptions.from_problem for automatic configuration
solver_opts = fe.SolverOptions.from_problem(problem_upper)
solver_upper = fe.create_solver(problem_upper, bc_upper, solver_options=solver_opts, iter_num=1)

print("Solving UPPER...")
sol_upper = solver_upper(internal_vars, initial)

# Check matrix size
sol_list_upper = problem_upper.unflatten_fn_sol_list(sol_upper)
J_info_upper = fe.get_jacobian_info(problem_upper, sol_list_upper, internal_vars)
print(f"Jacobian NNZ: {J_info_upper['nnz']:,}")
print(f"Max displacement: {jnp.max(jnp.abs(sol_upper)):.6e}")

print("\n" + "="*60)
print("Comparison")
print("="*60)
print(f"Memory reduction: {(1 - J_info_upper['nnz'] / J_info_full['nnz']) * 100:.1f}%")

# Compare solutions
diff = jnp.linalg.norm(sol_full - sol_upper)
norm = jnp.linalg.norm(sol_full)
rel_error = diff / norm

print(f"Solution difference: {diff:.3e}")
print(f"Relative error: {rel_error:.3e}")

if rel_error < 1e-10:
    print("✓ Solutions are IDENTICAL!")
else:
    print(f"⚠ Solutions differ by {rel_error:.2e}")

print("\nTest completed successfully!")
