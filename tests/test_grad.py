"""
Tests for automatic differentiation (grad) through solvers.

This module tests that gradients can be computed through solvers:
- JAX iterative solvers (CG, BICGSTAB, GMRES) with grad
- cuDSS direct solver with grad
- Gradient correctness and finite values
- Gradient consistency across solvers
"""

import pytest
import jax
import jax.numpy as jnp
import feax as fe
from feax.problem import MatrixView


# ============================================================================
# Environment Checks
# ============================================================================

def has_gpu():
    """Check if GPU is available."""
    try:
        devices = jax.devices('gpu')
        return len(devices) > 0
    except:
        return False


def has_cudss():
    """Check if cuDSS backend is available."""
    try:
        from feax.solver import CUDSSOptions
        return has_gpu()  # cuDSS requires GPU
    except ImportError:
        return False


# Skip decorators
requires_cuda = pytest.mark.skipif(
    not has_gpu(),
    reason="CUDA/GPU not available"
)

requires_cudss = pytest.mark.skipif(
    not has_cudss(),
    reason="cuDSS not available (requires GPU)"
)


# ============================================================================
# Gradient Tests - JAX Iterative Solvers
# ============================================================================

@pytest.mark.cpu
def test_cg_solver_grad(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that gradients can be computed through CG solver."""
    problem = linear_elasticity_problem
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Create solver with CG
    solver_opts = fe.SolverOptions(linear_solver="cg")
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Define function to differentiate: norm of solution
    def loss_fn(internal_vars):
        sol = solver(internal_vars, initial)
        return jnp.linalg.norm(sol)

    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(internal_vars)

    # Check that gradients exist and are finite
    assert grads is not None
    surface_grads = grads.surface_vars[0][0]
    assert jnp.all(jnp.isfinite(surface_grads)), "Gradients contain inf or nan"

    # Gradients should be non-trivial
    grad_norm = jnp.linalg.norm(surface_grads)
    assert grad_norm > 0, f"Gradient is trivial (norm={grad_norm})"


@pytest.mark.cpu
def test_bicgstab_solver_grad(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that gradients can be computed through BICGSTAB solver."""
    problem = linear_elasticity_problem
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Create solver with BICGSTAB
    solver_opts = fe.SolverOptions(linear_solver="bicgstab")
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Define function to differentiate
    def loss_fn(internal_vars):
        sol = solver(internal_vars, initial)
        return jnp.linalg.norm(sol)

    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(internal_vars)

    # Check gradients
    assert grads is not None
    surface_grads = grads.surface_vars[0][0]
    assert jnp.all(jnp.isfinite(surface_grads)), "Gradients contain inf or nan"

    grad_norm = jnp.linalg.norm(surface_grads)
    assert grad_norm > 0, f"Gradient is trivial (norm={grad_norm})"


@pytest.mark.cpu
def test_gmres_solver_grad(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that gradients can be computed through GMRES solver."""
    problem = linear_elasticity_problem
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Create solver with GMRES
    solver_opts = fe.SolverOptions(linear_solver="gmres")
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Define function to differentiate
    def loss_fn(internal_vars):
        sol = solver(internal_vars, initial)
        return jnp.linalg.norm(sol)

    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(internal_vars)

    # Check gradients
    assert grads is not None
    surface_grads = grads.surface_vars[0][0]
    assert jnp.all(jnp.isfinite(surface_grads)), "Gradients contain inf or nan"

    grad_norm = jnp.linalg.norm(surface_grads)
    assert grad_norm > 0, f"Gradient is trivial (norm={grad_norm})"


# ============================================================================
# Gradient Tests - cuDSS Solver
# ============================================================================

@pytest.mark.cuda
@requires_cudss
def test_cudss_solver_grad_full(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that gradients can be computed through cuDSS solver (FULL matrix)."""
    problem = linear_elasticity_problem
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Create solver with cuDSS
    solver_opts = fe.SolverOptions.from_problem(problem)
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Define function to differentiate
    def loss_fn(internal_vars):
        sol = solver(internal_vars, initial)
        return jnp.linalg.norm(sol)

    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(internal_vars)

    # Check gradients
    assert grads is not None
    surface_grads = grads.surface_vars[0][0]
    assert jnp.all(jnp.isfinite(surface_grads)), "Gradients contain inf or nan"

    grad_norm = jnp.linalg.norm(surface_grads)
    assert grad_norm > 0, f"Gradient is trivial (norm={grad_norm})"


@pytest.mark.cuda
@requires_cudss
def test_cudss_solver_grad_upper(
    linear_elasticity_problem_upper,
    internal_vars,
    material_params
):
    """Test that gradients can be computed through cuDSS solver (UPPER matrix)."""
    problem = linear_elasticity_problem_upper
    tol = material_params['tol']

    # Verify matrix view
    assert problem.matrix_view == MatrixView.UPPER

    # Create boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Create solver with cuDSS for UPPER matrix
    solver_opts = fe.SolverOptions.from_problem(problem)
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Define function to differentiate
    def loss_fn(internal_vars):
        sol = solver(internal_vars, initial)
        return jnp.linalg.norm(sol)

    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(internal_vars)

    # Check gradients
    assert grads is not None
    surface_grads = grads.surface_vars[0][0]
    assert jnp.all(jnp.isfinite(surface_grads)), "Gradients contain inf or nan"

    grad_norm = jnp.linalg.norm(surface_grads)
    assert grad_norm > 0, f"Gradient is trivial (norm={grad_norm})"


# ============================================================================
# Gradient Tests - Consistency
# ============================================================================

@pytest.mark.cpu
def test_gradient_consistency_cg_bicgstab(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that gradients from CG and BICGSTAB are similar."""
    problem = linear_elasticity_problem
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    initial = fe.zero_like_initial_guess(problem, bc)

    # Create CG solver and compute gradient
    solver_cg = fe.create_solver(
        problem, bc,
        solver_options=fe.SolverOptions(linear_solver="cg"),
        iter_num=1
    )

    def loss_cg(internal_vars):
        sol = solver_cg(internal_vars, initial)
        return jnp.linalg.norm(sol)

    grad_cg = jax.grad(loss_cg)(internal_vars)

    # Create BICGSTAB solver and compute gradient
    solver_bicgstab = fe.create_solver(
        problem, bc,
        solver_options=fe.SolverOptions(linear_solver="bicgstab"),
        iter_num=1
    )

    def loss_bicgstab(internal_vars):
        sol = solver_bicgstab(internal_vars, initial)
        return jnp.linalg.norm(sol)

    grad_bicgstab = jax.grad(loss_bicgstab)(internal_vars)

    # Compare gradients
    grad_cg_surf = grad_cg.surface_vars[0][0]
    grad_bicgstab_surf = grad_bicgstab.surface_vars[0][0]

    diff = jnp.linalg.norm(grad_cg_surf - grad_bicgstab_surf)
    norm = jnp.linalg.norm(grad_cg_surf)
    rel_diff = diff / norm

    # Gradients should be similar (within tolerance)
    assert rel_diff < 0.1, f"CG and BICGSTAB gradients differ by {rel_diff:.2e}"


# ============================================================================
# Gradient Tests - JIT Compatibility
# ============================================================================

@pytest.mark.cpu
def test_grad_jit_compatibility_cg(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that grad and JIT can be composed with CG solver."""
    problem = linear_elasticity_problem
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Create solver with CG
    solver_opts = fe.SolverOptions(linear_solver="cg")
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Define function to differentiate
    def loss_fn(internal_vars):
        sol = solver(internal_vars, initial)
        return jnp.linalg.norm(sol)

    # JIT the gradient function
    grad_fn = jax.jit(jax.grad(loss_fn))
    grads = grad_fn(internal_vars)

    # Check gradients
    assert grads is not None
    surface_grads = grads.surface_vars[0][0]
    assert jnp.all(jnp.isfinite(surface_grads)), "Gradients contain inf or nan"

    grad_norm = jnp.linalg.norm(surface_grads)
    assert grad_norm > 0, f"Gradient is trivial (norm={grad_norm})"


@pytest.mark.cpu
def test_grad_jit_compatibility_bicgstab(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that grad and JIT can be composed with BICGSTAB solver."""
    problem = linear_elasticity_problem
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Create solver with BICGSTAB
    solver_opts = fe.SolverOptions(linear_solver="bicgstab")
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Define function to differentiate
    def loss_fn(internal_vars):
        sol = solver(internal_vars, initial)
        return jnp.linalg.norm(sol)

    # JIT the gradient function
    grad_fn = jax.jit(jax.grad(loss_fn))
    grads = grad_fn(internal_vars)

    # Check gradients
    assert grads is not None
    surface_grads = grads.surface_vars[0][0]
    assert jnp.all(jnp.isfinite(surface_grads)), "Gradients contain inf or nan"

    grad_norm = jnp.linalg.norm(surface_grads)
    assert grad_norm > 0, f"Gradient is trivial (norm={grad_norm})"


@pytest.mark.cpu
def test_grad_jit_compatibility_gmres(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that grad and JIT can be composed with GMRES solver."""
    problem = linear_elasticity_problem
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Create solver with GMRES
    solver_opts = fe.SolverOptions(linear_solver="gmres")
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Define function to differentiate
    def loss_fn(internal_vars):
        sol = solver(internal_vars, initial)
        return jnp.linalg.norm(sol)

    # JIT the gradient function
    grad_fn = jax.jit(jax.grad(loss_fn))
    grads = grad_fn(internal_vars)

    # Check gradients
    assert grads is not None
    surface_grads = grads.surface_vars[0][0]
    assert jnp.all(jnp.isfinite(surface_grads)), "Gradients contain inf or nan"

    grad_norm = jnp.linalg.norm(surface_grads)
    assert grad_norm > 0, f"Gradient is trivial (norm={grad_norm})"


@pytest.mark.cuda
@requires_cudss
def test_grad_jit_compatibility_cudss(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that grad and JIT can be composed with cuDSS solver."""
    problem = linear_elasticity_problem
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Create solver with cuDSS
    solver_opts = fe.SolverOptions.from_problem(problem)
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Define function to differentiate
    def loss_fn(internal_vars):
        sol = solver(internal_vars, initial)
        return jnp.linalg.norm(sol)

    # JIT the gradient function
    grad_fn = jax.jit(jax.grad(loss_fn))
    grads = grad_fn(internal_vars)

    # Check gradients
    assert grads is not None
    surface_grads = grads.surface_vars[0][0]
    assert jnp.all(jnp.isfinite(surface_grads)), "Gradients contain inf or nan"

    grad_norm = jnp.linalg.norm(surface_grads)
    assert grad_norm > 0, f"Gradient is trivial (norm={grad_norm})"


# ============================================================================
# Gradient Tests - JIT+Grad Composition Order
# ============================================================================

@pytest.mark.cpu
def test_jit_grad_composition_order_cg(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test different composition orders of JIT and grad with CG solver."""
    problem = linear_elasticity_problem
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Create solver with CG
    solver_opts = fe.SolverOptions(linear_solver="cg")
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Define function to differentiate
    def loss_fn(internal_vars):
        sol = solver(internal_vars, initial)
        return jnp.linalg.norm(sol)

    # Test 1: jax.jit(jax.grad(loss_fn))
    grad_fn_1 = jax.jit(jax.grad(loss_fn))
    grads_1 = grad_fn_1(internal_vars)

    # Test 2: jax.grad(jax.jit(loss_fn))
    grad_fn_2 = jax.grad(jax.jit(loss_fn))
    grads_2 = grad_fn_2(internal_vars)

    # Both should produce valid gradients
    surface_grads_1 = grads_1.surface_vars[0][0]
    surface_grads_2 = grads_2.surface_vars[0][0]

    assert jnp.all(jnp.isfinite(surface_grads_1))
    assert jnp.all(jnp.isfinite(surface_grads_2))

    # Gradients should be similar (allow small numerical differences)
    diff = jnp.linalg.norm(surface_grads_1 - surface_grads_2)
    norm = jnp.linalg.norm(surface_grads_1)
    rel_diff = diff / norm

    assert rel_diff < 1e-8, f"Different composition orders produce different gradients: {rel_diff:.2e}"


@pytest.mark.cuda
@requires_cudss
def test_jit_grad_composition_order_cudss(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test different composition orders of JIT and grad with cuDSS solver."""
    problem = linear_elasticity_problem
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Create solver with cuDSS
    solver_opts = fe.SolverOptions.from_problem(problem)
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Define function to differentiate
    def loss_fn(internal_vars):
        sol = solver(internal_vars, initial)
        return jnp.linalg.norm(sol)

    # Test 1: jax.jit(jax.grad(loss_fn))
    grad_fn_1 = jax.jit(jax.grad(loss_fn))
    grads_1 = grad_fn_1(internal_vars)

    # Test 2: jax.grad(jax.jit(loss_fn))
    grad_fn_2 = jax.grad(jax.jit(loss_fn))
    grads_2 = grad_fn_2(internal_vars)

    # Both should produce valid gradients
    surface_grads_1 = grads_1.surface_vars[0][0]
    surface_grads_2 = grads_2.surface_vars[0][0]

    assert jnp.all(jnp.isfinite(surface_grads_1))
    assert jnp.all(jnp.isfinite(surface_grads_2))

    # Gradients should be similar (allow small numerical differences)
    diff = jnp.linalg.norm(surface_grads_1 - surface_grads_2)
    norm = jnp.linalg.norm(surface_grads_1)
    rel_diff = diff / norm

    assert rel_diff < 1e-8, f"Different composition orders produce different gradients: {rel_diff:.2e}"
