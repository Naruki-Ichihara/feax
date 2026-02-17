"""Test to investigate UPPER matrix view adjoint issue."""
import jax
import jax.numpy as jnp
import feax as fe
from feax.problem import MatrixView

# Simple test problem
def create_test_problem(matrix_view):
    """Create a simple linear elasticity problem."""
    mesh = fe.mesh.box_mesh((10, 10, 10), mesh_size=2)
    tol = 1e-3

    class SimpleElasticity(fe.problem.Problem):
        def get_tensor_map(self):
            def stress(u_grad, rho):
                E = 70e3 * rho**3
                nu = 0.3
                mu = E / (2. * (1. + nu))
                lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
                epsilon = 0.5 * (u_grad + u_grad.T)
                return lmbda * jnp.trace(epsilon) * jnp.eye(3) + 2 * mu * epsilon
            return stress

        def get_surface_maps(self):
            def surface_map(u, x, *args):
                return jnp.array([0., 0., -1.0])
            return [surface_map]

    # Create problem with specified matrix view
    right = lambda p: jnp.isclose(p[0], 10., tol) & (p[2] < 2.5)
    problem = SimpleElasticity(mesh, vec=3, dim=3, location_fns=[right], matrix_view=matrix_view)

    # Boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DCboundary.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DCboundary.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    return problem, bc

def test_gradient_with_matrix_view(matrix_view_name):
    """Test gradient computation with specified matrix view."""
    print(f"\n{'='*60}")
    print(f"Testing with MatrixView.{matrix_view_name}")
    print(f"{'='*60}")

    matrix_view = getattr(MatrixView, matrix_view_name)
    problem, bc = create_test_problem(matrix_view)

    # Create solver (matching topology optimization setup)
    solver_opts = fe.SolverOptions.from_problem(problem)
    print(f"Solver options: linear_solver={solver_opts.linear_solver}")
    if solver_opts.cudss_options:
        print(f"  cudss matrix_type={solver_opts.cudss_options.matrix_type}")
        print(f"  cudss matrix_view={solver_opts.cudss_options.matrix_view}")

    # IMPORTANT: Use same solver_opts for both forward and adjoint (like topology_optimization.py)
    solver = fe.solver.create_solver(
        problem, bc,
        solver_options=solver_opts,
        adjoint_solver_options=solver_opts,  # This triggers the issue!
        iter_num=1
    )
    initial = fe.utils.zero_like_initial_guess(problem, bc)

    # Create test density field
    num_nodes = problem.mesh[0].points.shape[0]
    rho = fe.internal_vars.InternalVars.create_node_var(problem, 0.5)

    # Define objective: compliance
    def objective(rho_var):
        internal_vars = fe.internal_vars.InternalVars(volume_vars=(rho_var,), surface_vars=())
        sol = solver(internal_vars, initial)
        # Compliance: u^T K u
        return jnp.sum(sol * sol)

    # Compute objective and gradient
    obj_val = objective(rho)
    grad_fn = jax.grad(objective)
    grad_val = grad_fn(rho)

    print(f"\nObjective value: {float(obj_val):.6e}")
    print(f"Gradient norm: {float(jnp.linalg.norm(grad_val)):.6e}")
    print(f"Gradient min/max: [{float(jnp.min(grad_val)):.6e}, {float(jnp.max(grad_val)):.6e}]")
    print(f"Gradient mean: {float(jnp.mean(grad_val)):.6e}")

    return obj_val, grad_val

# Test with both FULL and UPPER
print("Testing gradient computation with different matrix views...")
print("If UPPER has incorrect adjoint, its gradient will be different from FULL")

obj_full, grad_full = test_gradient_with_matrix_view("FULL")
obj_upper, grad_upper = test_gradient_with_matrix_view("UPPER")

# Compare results
print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"Objective difference: {float(jnp.abs(obj_full - obj_upper)):.6e}")
print(f"Gradient difference (L2): {float(jnp.linalg.norm(grad_full - grad_upper)):.6e}")
print(f"Gradient relative difference: {float(jnp.linalg.norm(grad_full - grad_upper) / jnp.linalg.norm(grad_full)):.6e}")

# Check if gradients are significantly different
rel_diff = float(jnp.linalg.norm(grad_full - grad_upper) / jnp.linalg.norm(grad_full))
if rel_diff > 1e-6:
    print(f"\n⚠️  WARNING: Gradients differ significantly (rel_diff={rel_diff:.2e})")
    print("This indicates the UPPER matrix view has incorrect adjoint computation!")
else:
    print(f"\n✓ Gradients match (rel_diff={rel_diff:.2e})")
