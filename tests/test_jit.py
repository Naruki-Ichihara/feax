"""
Tests for JIT compilation compatibility with solvers.

This module tests that solvers work correctly with JAX JIT compilation:
- JAX iterative solvers (CG, BICGSTAB, GMRES) with/without JIT
- cuDSS direct solver with/without JIT
- Solution consistency between JIT and non-JIT versions
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
# JIT Tests - JAX Iterative Solvers
# ============================================================================

@pytest.mark.cpu
def test_cg_solver_jit_compatibility(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that CG solver works with and without JIT."""
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

    # Solve without JIT
    sol_no_jit = solver(internal_vars, initial)

    # Solve with JIT
    solver_jit = jax.jit(solver)
    sol_with_jit = solver_jit(internal_vars, initial)

    # Solutions should be identical
    diff = jnp.linalg.norm(sol_no_jit - sol_with_jit)
    assert diff < 1e-10, f"JIT and non-JIT CG solutions differ by {diff:.2e}"

    # Both should be non-trivial
    assert jnp.linalg.norm(sol_no_jit) > 0
    assert jnp.linalg.norm(sol_with_jit) > 0


@pytest.mark.cpu
def test_bicgstab_solver_jit_compatibility(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that BICGSTAB solver works with and without JIT."""
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

    # Solve without JIT
    sol_no_jit = solver(internal_vars, initial)

    # Solve with JIT
    solver_jit = jax.jit(solver)
    sol_with_jit = solver_jit(internal_vars, initial)

    # Solutions should be identical
    diff = jnp.linalg.norm(sol_no_jit - sol_with_jit)
    assert diff < 1e-10, f"JIT and non-JIT BICGSTAB solutions differ by {diff:.2e}"

    # Both should be non-trivial
    assert jnp.linalg.norm(sol_no_jit) > 0
    assert jnp.linalg.norm(sol_with_jit) > 0


@pytest.mark.cpu
def test_gmres_solver_jit_compatibility(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that GMRES solver works with and without JIT."""
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

    # Solve without JIT
    sol_no_jit = solver(internal_vars, initial)

    # Solve with JIT
    solver_jit = jax.jit(solver)
    sol_with_jit = solver_jit(internal_vars, initial)

    # Solutions should be identical
    diff = jnp.linalg.norm(sol_no_jit - sol_with_jit)
    assert diff < 1e-10, f"JIT and non-JIT GMRES solutions differ by {diff:.2e}"

    # Both should be non-trivial
    assert jnp.linalg.norm(sol_no_jit) > 0
    assert jnp.linalg.norm(sol_with_jit) > 0


# ============================================================================
# JIT Tests - cuDSS Solver
# ============================================================================

@pytest.mark.cuda
@requires_cudss
def test_cudss_solver_jit_compatibility_full(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that cuDSS solver (FULL matrix) works with JIT."""
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

    # Solve with JIT (only JIT version to avoid multiple solver creation)
    solver_jit = jax.jit(solver)
    sol_with_jit = solver_jit(internal_vars, initial)

    # Solution should be non-trivial and reasonable
    solution_norm = jnp.linalg.norm(sol_with_jit)
    assert solution_norm > 0, f"Solution is trivial (norm={solution_norm})"
    assert solution_norm < 1.0, f"Solution norm too large: {solution_norm}"


@pytest.mark.cuda
@requires_cudss
def test_cudss_solver_jit_compatibility_upper(
    linear_elasticity_problem_upper,
    internal_vars,
    material_params
):
    """Test that cuDSS solver (UPPER matrix) works with JIT."""
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

    # Solve with JIT (only JIT version to avoid multiple solver creation)
    solver_jit = jax.jit(solver)
    sol_with_jit = solver_jit(internal_vars, initial)

    # Solution should be non-trivial and reasonable
    solution_norm = jnp.linalg.norm(sol_with_jit)
    assert solution_norm > 0, f"Solution is trivial (norm={solution_norm})"
    assert solution_norm < 1.0, f"Solution norm too large: {solution_norm}"


# ============================================================================
# JIT Tests - Multiple JIT Compilations
# ============================================================================

@pytest.mark.cpu
def test_multiple_jit_compilations_cg(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that CG solver can be JIT-compiled multiple times."""
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

    # First JIT compilation
    solver_jit1 = jax.jit(solver)
    sol1 = solver_jit1(internal_vars, initial)

    # Second JIT compilation (should use cached version)
    solver_jit2 = jax.jit(solver)
    sol2 = solver_jit2(internal_vars, initial)

    # Solutions should be identical
    diff = jnp.linalg.norm(sol1 - sol2)
    assert diff < 1e-12, f"Multiple JIT compilations produce different results: {diff:.2e}"


@pytest.mark.cuda
@requires_cudss
def test_multiple_jit_compilations_cudss(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that cuDSS solver can be JIT-compiled multiple times."""
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

    # First JIT compilation
    solver_jit1 = jax.jit(solver)
    sol1 = solver_jit1(internal_vars, initial)

    # Second JIT compilation (should use cached version)
    solver_jit2 = jax.jit(solver)
    sol2 = solver_jit2(internal_vars, initial)

    # Solutions should be very close
    diff = jnp.linalg.norm(sol1 - sol2) / jnp.linalg.norm(sol1)
    assert diff < 1e-10, f"Multiple JIT compilations produce different results: {diff:.2e}"
