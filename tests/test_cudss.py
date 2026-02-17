"""
Tests for cuDSS solver with linear elasticity problem.

This module tests cuDSS direct solver:
- cuDSS with FULL matrix view
- cuDSS with UPPER matrix view (symmetric)
- Solution consistency between FULL and UPPER
- Solution consistency between cuDSS and JAX CG
- cuDSS-specific options and configurations

These tests require CUDA/GPU and are marked with @pytest.mark.cuda
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
# cuDSS Tests - Linear Elasticity
# ============================================================================

@pytest.mark.cuda
@requires_cudss
def test_cudss_full_matrix_solver(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test cuDSS solver with FULL matrix view."""
    problem = linear_elasticity_problem
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Create solver with cuDSS (default on GPU)
    solver_opts = fe.SolverOptions.from_problem(problem)
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Solve
    solution = solver(internal_vars, initial)

    # Check solution is non-trivial
    solution_norm = jnp.linalg.norm(solution)
    assert solution_norm > 0, f"Solution is trivial (norm={solution_norm})"

    # Check solution magnitude is reasonable
    assert solution_norm < 1.0, f"Solution norm too large: {solution_norm}"


@pytest.mark.cuda
@requires_cudss
def test_cudss_upper_matrix_solver(
    linear_elasticity_problem_upper,
    internal_vars,
    material_params
):
    """Test cuDSS solver with UPPER triangular matrix view."""
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

    # Solve
    solution = solver(internal_vars, initial)

    # Check solution is non-trivial
    solution_norm = jnp.linalg.norm(solution)
    assert solution_norm > 0, f"Solution is trivial (norm={solution_norm})"

    # Check solution magnitude is reasonable
    assert solution_norm < 1.0, f"Solution norm too large: {solution_norm}"


@pytest.mark.cuda
@requires_cudss
def test_cudss_full_vs_upper_consistency(
    linear_elasticity_problem,
    linear_elasticity_problem_upper,
    internal_vars,
    material_params
):
    """Test that cuDSS produces consistent solutions for FULL vs UPPER matrix views."""
    tol = material_params['tol']

    # Create boundary conditions (same for both)
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])

    # Solve with FULL matrix view
    bc_full = bc_config.create_bc(linear_elasticity_problem)
    solver_opts_full = fe.SolverOptions.from_problem(linear_elasticity_problem)
    solver_full = fe.create_solver(
        linear_elasticity_problem, bc_full,
        solver_options=solver_opts_full, iter_num=1
    )
    initial_full = fe.zero_like_initial_guess(linear_elasticity_problem, bc_full)
    sol_full = solver_full(internal_vars, initial_full)

    # Solve with UPPER matrix view
    bc_upper = bc_config.create_bc(linear_elasticity_problem_upper)
    solver_opts_upper = fe.SolverOptions.from_problem(linear_elasticity_problem_upper)
    solver_upper = fe.create_solver(
        linear_elasticity_problem_upper, bc_upper,
        solver_options=solver_opts_upper, iter_num=1
    )
    initial_upper = fe.zero_like_initial_guess(linear_elasticity_problem_upper, bc_upper)
    sol_upper = solver_upper(internal_vars, initial_upper)

    # Solutions should be very close
    solution_tol = 1e-6

    diff = jnp.linalg.norm(sol_full - sol_upper) / jnp.linalg.norm(sol_full)
    assert diff < solution_tol, f"FULL and UPPER solutions differ by {diff:.2e}"


@pytest.mark.cuda
@requires_cudss
def test_cudss_options_configuration(
    linear_elasticity_problem_upper,
    internal_vars,
    material_params
):
    """Test cuDSS with explicit CUDSSOptions configuration."""
    problem = linear_elasticity_problem_upper
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Create solver with explicit cuDSS options
    from feax.solver import CUDSSOptions, CUDSSMatrixType, CUDSSMatrixView

    cudss_opts = CUDSSOptions(
        matrix_type=CUDSSMatrixType.SPD,  # Symmetric Positive Definite
        matrix_view=CUDSSMatrixView.UPPER
    )

    solver_opts = fe.SolverOptions(
        linear_solver="cudss",
        cudss_options=cudss_opts
    )

    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Solve
    solution = solver(internal_vars, initial)

    # Check solution
    solution_norm = jnp.linalg.norm(solution)
    assert solution_norm > 0, f"Solution is trivial (norm={solution_norm})"
    assert solution_norm < 1.0, f"Solution norm too large: {solution_norm}"


@pytest.mark.cuda
@requires_cudss
def test_cudss_memory_efficiency(
    linear_elasticity_problem,
    linear_elasticity_problem_upper,
    material_params
):
    """Test that UPPER matrix view uses less memory than FULL."""
    # This test verifies the memory reduction from matrix view filtering
    problem_full = linear_elasticity_problem
    problem_upper = linear_elasticity_problem_upper

    # Check nnz (non-zero entries)
    nnz_full = len(problem_full.I_filtered)
    nnz_upper = len(problem_upper.I_filtered)

    # UPPER should have approximately half the entries
    assert nnz_upper == 3600
    assert nnz_full == 6912
    assert nnz_upper < nnz_full

    # Memory reduction should be around 50%
    reduction_ratio = nnz_upper / nnz_full
    assert 0.45 < reduction_ratio < 0.55, f"Memory reduction ratio: {reduction_ratio:.2f}"


@pytest.mark.cuda
@requires_cudss
def test_cudss_vs_cg_consistency(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that cuDSS and JAX CG produce consistent solutions."""
    problem = linear_elasticity_problem
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Solve with cuDSS (default on GPU with FULL matrix)
    solver_opts_cudss = fe.SolverOptions.from_problem(problem)
    solver_cudss = fe.create_solver(problem, bc, solver_options=solver_opts_cudss, iter_num=1)
    initial_cudss = fe.zero_like_initial_guess(problem, bc)
    sol_cudss = solver_cudss(internal_vars, initial_cudss)

    # Solve with JAX CG
    solver_opts_cg = fe.SolverOptions(linear_solver="cg")
    solver_cg = fe.create_solver(problem, bc, solver_options=solver_opts_cg, iter_num=1)
    initial_cg = fe.zero_like_initial_guess(problem, bc)
    sol_cg = solver_cg(internal_vars, initial_cg)

    # Solutions should be very close
    solution_tol = 1e-6

    diff = jnp.linalg.norm(sol_cudss - sol_cg) / jnp.linalg.norm(sol_cudss)
    assert diff < solution_tol, f"cuDSS and CG solutions differ by {diff:.2e}"


