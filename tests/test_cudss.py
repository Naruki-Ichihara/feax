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

import jax
import jax.numpy as np
import pytest

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
        from feax.solvers.options import CUDSSOptions
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
    traced_params,
    material_params
):
    """Test cuDSS solver with FULL matrix view."""
    problem = linear_elasticity_problem
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: np.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Create solver with cuDSS (default on GPU)
    solver_opts = fe.DirectSolverOptions()
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, linear=True, traced_params=traced_params)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Solve
    solution = solver(traced_params, initial)

    # Check solution is non-trivial
    solution_norm = np.linalg.norm(solution)
    assert solution_norm > 0, f"Solution is trivial (norm={solution_norm})"

    # Check solution magnitude is reasonable
    assert solution_norm < 1.0, f"Solution norm too large: {solution_norm}"


@pytest.mark.cuda
@requires_cudss
def test_cudss_upper_matrix_solver(
    linear_elasticity_problem_upper,
    traced_params,
    material_params
):
    """Test cuDSS solver with UPPER triangular matrix view."""
    problem = linear_elasticity_problem_upper
    tol = material_params['tol']

    # Verify matrix view
    assert problem.matrix_view == MatrixView.UPPER

    # Create boundary conditions
    left = lambda p: np.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Create solver with cuDSS for UPPER matrix
    solver_opts = fe.DirectSolverOptions()
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, linear=True, traced_params=traced_params)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Solve
    solution = solver(traced_params, initial)

    # Check solution is non-trivial
    solution_norm = np.linalg.norm(solution)
    assert solution_norm > 0, f"Solution is trivial (norm={solution_norm})"

    # Check solution magnitude is reasonable
    assert solution_norm < 1.0, f"Solution norm too large: {solution_norm}"


@pytest.mark.cuda
@requires_cudss
def test_cudss_full_vs_upper_consistency(
    linear_elasticity_problem,
    linear_elasticity_problem_upper,
    traced_params,
    material_params
):
    """Test that cuDSS produces consistent solutions for FULL vs UPPER matrix views."""
    tol = material_params['tol']

    # Create boundary conditions (same for both)
    left = lambda p: np.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])

    # Solve with FULL matrix view
    bc_full = bc_config.create_bc(linear_elasticity_problem)
    solver_opts_full = fe.DirectSolverOptions()
    solver_full = fe.create_solver(
        linear_elasticity_problem, bc_full,
        solver_options=solver_opts_full, linear=True, traced_params=traced_params
    )
    initial_full = fe.zero_like_initial_guess(linear_elasticity_problem, bc_full)
    sol_full = solver_full(traced_params, initial_full)

    # Solve with UPPER matrix view
    bc_upper = bc_config.create_bc(linear_elasticity_problem_upper)
    solver_opts_upper = fe.DirectSolverOptions()
    solver_upper = fe.create_solver(
        linear_elasticity_problem_upper, bc_upper,
        solver_options=solver_opts_upper, linear=True, traced_params=traced_params
    )
    initial_upper = fe.zero_like_initial_guess(linear_elasticity_problem_upper, bc_upper)
    sol_upper = solver_upper(traced_params, initial_upper)

    # Solutions should be very close
    solution_tol = 1e-6

    diff = np.linalg.norm(sol_full - sol_upper) / np.linalg.norm(sol_full)
    assert diff < solution_tol, f"FULL and UPPER solutions differ by {diff:.2e}"


@pytest.mark.cuda
@requires_cudss
def test_cudss_options_configuration(
    linear_elasticity_problem_upper,
    traced_params,
    material_params
):
    """Test cuDSS with explicit CUDSSOptions configuration."""
    problem = linear_elasticity_problem_upper
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: np.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Create solver with explicit cuDSS options
    from feax.solvers.options import CUDSSMatrixType, CUDSSMatrixView, CUDSSOptions

    cudss_opts = CUDSSOptions(
        matrix_type=CUDSSMatrixType.SPD,  # Symmetric Positive Definite
        matrix_view=CUDSSMatrixView.UPPER
    )

    solver_opts = fe.DirectSolverOptions(
        solver="cudss",
        cudss_options=cudss_opts
    )

    solver = fe.create_solver(problem, bc, solver_options=solver_opts, linear=True)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Solve
    solution = solver(traced_params, initial)

    # Check solution
    solution_norm = np.linalg.norm(solution)
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
    traced_params,
    material_params
):
    """Test that cuDSS and JAX CG produce consistent solutions."""
    problem = linear_elasticity_problem
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: np.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Solve with cuDSS (default on GPU with FULL matrix)
    solver_opts_cudss = fe.DirectSolverOptions()
    solver_cudss = fe.create_solver(problem, bc, solver_options=solver_opts_cudss, linear=True, traced_params=traced_params)
    initial_cudss = fe.zero_like_initial_guess(problem, bc)
    sol_cudss = solver_cudss(traced_params, initial_cudss)

    # Solve with JAX CG
    solver_opts_cg = fe.KrylovSolverOptions(solver="cg")
    solver_cg = fe.create_solver(problem, bc, solver_options=solver_opts_cg, linear=True)
    initial_cg = fe.zero_like_initial_guess(problem, bc)
    sol_cg = solver_cg(traced_params, initial_cg)

    # Solutions should be very close
    solution_tol = 1e-6

    diff = np.linalg.norm(sol_cudss - sol_cg) / np.linalg.norm(sol_cudss)
    assert diff < solution_tol, f"cuDSS and CG solutions differ by {diff:.2e}"



@pytest.mark.cuda
@requires_cudss
def test_cudss_reuse_factorization_forward_and_adjoint(
    linear_elasticity_problem_upper,
    traced_params,
    material_params
):
    """reuse_factorization=True (factor once, reuse for the adjoint) must give
    the same solution and gradient as the default two-factorization path."""
    import jax

    problem = linear_elasticity_problem_upper  # symmetric view -> reuse eligible
    tol = material_params['tol']

    left = lambda p: np.isclose(p[0], 0., tol)
    bc = fe.DirichletBCConfig([
        fe.DirichletBCSpec(location=left, component="all", value=0.)
    ]).create_bc(problem)
    initial = fe.zero_like_initial_guess(problem, bc)

    solver_reuse = fe.create_solver(
        problem, bc,
        solver_options=fe.DirectSolverOptions(reuse_factorization=True),
        linear=True, traced_params=traced_params)
    solver_plain = fe.create_solver(
        problem, bc, solver_options=fe.DirectSolverOptions(),
        linear=True, traced_params=traced_params)

    sol_reuse = solver_reuse(traced_params, initial)
    sol_plain = solver_plain(traced_params, initial)
    ref = float(np.max(np.abs(sol_plain)))
    assert float(np.max(np.abs(sol_reuse - sol_plain))) < 1e-10 * ref

    # gradient w.r.t. the surface traction var through both adjoint paths
    def make_loss(solver):
        return lambda tp: np.sum(solver(tp, initial) ** 2)

    g_reuse = jax.grad(make_loss(solver_reuse))(traced_params)
    g_plain = jax.grad(make_loss(solver_plain))(traced_params)
    gr = g_reuse.surface_vars[0][0]
    gp = g_plain.surface_vars[0][0]
    scale = float(np.max(np.abs(gp))) + 1e-30
    assert float(np.max(np.abs(gr - gp))) / scale < 1e-8

    # and against the matrix-free Krylov adjoint as an independent reference
    solver_cg = fe.create_solver(
        problem, bc,
        solver_options=fe.KrylovSolverOptions(solver="cg", tol=1e-12, atol=1e-14),
        linear=True)
    g_cg = jax.grad(make_loss(solver_cg))(traced_params).surface_vars[0][0]
    assert float(np.max(np.abs(gr - g_cg))) / scale < 1e-6
