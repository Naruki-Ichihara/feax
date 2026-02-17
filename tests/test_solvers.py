"""
Tests for solver functionality with linear elasticity problem.

This module tests JAX iterative solvers:
- CG (Conjugate Gradient)
- BICGSTAB (BiConjugate Gradient Stabilized)
- GMRES (Generalized Minimal Residual)
- Solution accuracy and consistency
- Matrix view compatibility
- Physical validity of solutions
"""

import pytest
import jax
import jax.numpy as jnp
import feax as fe
from feax.problem import MatrixView


# ============================================================================
# Solver Tests - JAX Iterative Solvers
# ============================================================================

def test_cg_solver_linear_elasticity(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test JAX CG solver on linear elasticity problem."""
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

    # Solve
    solution = solver(internal_vars, initial)

    # Check solution is non-trivial
    assert jnp.linalg.norm(solution) > 0

    # Check solution magnitude is reasonable
    assert jnp.linalg.norm(solution) < 1.0


def test_bicgstab_solver_linear_elasticity(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test JAX BICGSTAB solver on linear elasticity problem."""
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

    # Solve
    solution = solver(internal_vars, initial)

    # Check solution is non-trivial
    assert jnp.linalg.norm(solution) > 0

    # Check solution magnitude is reasonable
    assert jnp.linalg.norm(solution) < 1.0


def test_gmres_solver_linear_elasticity(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test JAX GMRES solver on linear elasticity problem."""
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

    # Solve
    solution = solver(internal_vars, initial)

    # Check solution is non-trivial
    assert jnp.linalg.norm(solution) > 0

    # Check solution magnitude is reasonable
    assert jnp.linalg.norm(solution) < 1.0


def test_solver_consistency_cg_bicgstab_gmres(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that CG, BICGSTAB, and GMRES produce consistent solutions."""
    problem = linear_elasticity_problem
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Solve with CG
    solver_opts_cg = fe.SolverOptions(linear_solver="cg")
    solver_cg = fe.create_solver(problem, bc, solver_options=solver_opts_cg, iter_num=1)
    initial_cg = fe.zero_like_initial_guess(problem, bc)
    sol_cg = solver_cg(internal_vars, initial_cg)

    # Solve with BICGSTAB
    solver_opts_bicgstab = fe.SolverOptions(linear_solver="bicgstab")
    solver_bicgstab = fe.create_solver(problem, bc, solver_options=solver_opts_bicgstab, iter_num=1)
    initial_bicgstab = fe.zero_like_initial_guess(problem, bc)
    sol_bicgstab = solver_bicgstab(internal_vars, initial_bicgstab)

    # Solve with GMRES
    solver_opts_gmres = fe.SolverOptions(linear_solver="gmres")
    solver_gmres = fe.create_solver(problem, bc, solver_options=solver_opts_gmres, iter_num=1)
    initial_gmres = fe.zero_like_initial_guess(problem, bc)
    sol_gmres = solver_gmres(internal_vars, initial_gmres)

    # All solutions should be close to each other
    solution_tol = 1e-4

    diff_cg_bicgstab = jnp.linalg.norm(sol_cg - sol_bicgstab) / jnp.linalg.norm(sol_cg)
    diff_cg_gmres = jnp.linalg.norm(sol_cg - sol_gmres) / jnp.linalg.norm(sol_cg)
    diff_bicgstab_gmres = jnp.linalg.norm(sol_bicgstab - sol_gmres) / jnp.linalg.norm(sol_bicgstab)

    assert diff_cg_bicgstab < solution_tol, f"CG and BICGSTAB solutions differ by {diff_cg_bicgstab:.2e}"
    assert diff_cg_gmres < solution_tol, f"CG and GMRES solutions differ by {diff_cg_gmres:.2e}"
    assert diff_bicgstab_gmres < solution_tol, f"BICGSTAB and GMRES solutions differ by {diff_bicgstab_gmres:.2e}"


def test_solution_physical_validity_with_cg(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that CG solution has physically valid properties."""
    problem = linear_elasticity_problem
    traction = material_params['traction']
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Create solver with CG and solve
    solver_opts = fe.SolverOptions(linear_solver="cg")
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
    initial = fe.zero_like_initial_guess(problem, bc)
    solution = solver(internal_vars, initial)

    # Check that solution is non-trivial
    solution_norm = jnp.linalg.norm(solution)
    assert solution_norm > 0, f"Solution is trivial (norm={solution_norm})"

    # Check solution magnitude is reasonable
    assert solution_norm < 1.0, f"Solution norm too large: {solution_norm}"


def test_residual_after_cg_solve(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that residual is small after solving with CG."""
    problem = linear_elasticity_problem
    tol = material_params['tol']

    # Create boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Solve with CG
    solver_opts = fe.SolverOptions(linear_solver="cg")
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
    initial = fe.zero_like_initial_guess(problem, bc)
    solution = solver(internal_vars, initial)

    # Check residual
    sol_list = problem.unflatten_fn_sol_list(solution)
    residual_list = fe.get_res(problem, sol_list, internal_vars)

    # Flatten residual list to vector by concatenating and flattening each array
    residual = jnp.concatenate([r.flatten() for r in residual_list])

    # Apply boundary conditions to residual
    residual_bc = fe.apply_boundary_to_res(bc, residual, solution)
    residual_norm = jnp.linalg.norm(residual_bc)

    # Residual should be very small after solving
    assert residual_norm < 1e-5, f"Residual too large: {residual_norm}"
