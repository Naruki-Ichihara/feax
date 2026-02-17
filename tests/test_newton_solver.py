"""
Tests for hyperelastic (non-linear) problem solving.

This module tests hyperelasticity with:
- Neo-Hookean material model
- Newton solver convergence
- Non-linear solution accuracy
- Large deformation capability
"""

import pytest
import jax
import jax.numpy as jnp
import feax as fe
from feax.problem import MatrixView


# ============================================================================
# Hyperelasticity Tests - Neo-Hookean Material
# ============================================================================

def test_neohookean_solver_convergence(
    simple_mesh,
    material_params
):
    """Test Newton solver convergence for Neo-Hookean material."""
    E = material_params['E']
    nu = material_params['nu']
    tol = material_params['tol']

    # Neo-Hookean material parameters
    mu = E / (2 * (1 + nu))
    kappa = E / (3 * (1 - 2 * nu))

    class NeoHookean(fe.Problem):
        def get_tensor_map(self):
            def first_PK_stress(u_grad, *args):
                """First Piola-Kirchhoff stress for Neo-Hookean material."""
                # Deformation gradient F = I + grad(u)
                F = jnp.eye(self.dim) + u_grad

                # Right Cauchy-Green tensor C = F^T F
                C = F.T @ F

                # Invariants
                J = jnp.linalg.det(F)
                I1 = jnp.trace(C)

                # Neo-Hookean strain energy derivatives
                # W = (mu/2)(I1 - 3) - mu*ln(J) + (kappa/2)(J-1)^2

                # First Piola-Kirchhoff stress
                # P = mu(F - F^-T) + kappa(J-1)J F^-T
                F_inv_T = jnp.linalg.inv(F).T
                P = mu * (F - F_inv_T) + kappa * (J - 1) * J * F_inv_T

                return P

            return first_PK_stress

        def get_surface_maps(self):
            return [lambda u, x, t: jnp.array([0., 0., t])]

    # Boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    right = lambda p: jnp.isclose(p[0], 10., tol)

    # Create problem
    problem = NeoHookean(
        simple_mesh, vec=3, dim=3,
        location_fns=[right],
        matrix_view=MatrixView.FULL
    )

    # Dirichlet BC
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    # Internal variables (surface traction)
    traction = material_params['traction']
    surf_var = fe.InternalVars.create_uniform_surface_var(problem, traction)
    internal_vars = fe.InternalVars((), [(surf_var,)])

    # Create Newton solver with CG for linear solve
    solver_opts = fe.SolverOptions(
        linear_solver="cg",
        max_iter=20,
        tol=1e-6
    )
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=20)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Solve
    solution = solver(internal_vars, initial)

    # Check solution is non-trivial
    solution_norm = jnp.linalg.norm(solution)
    assert solution_norm > 0, f"Solution is trivial (norm={solution_norm})"

    # Check solution magnitude is reasonable
    assert solution_norm < 10.0, f"Solution norm too large: {solution_norm}"


def test_neohookean_residual_convergence(
    simple_mesh,
    material_params
):
    """Test that Newton solver reduces residual for Neo-Hookean problem."""
    E = material_params['E']
    nu = material_params['nu']
    tol = material_params['tol']

    mu = E / (2 * (1 + nu))
    kappa = E / (3 * (1 - 2 * nu))

    class NeoHookean(fe.Problem):
        def get_tensor_map(self):
            def first_PK_stress(u_grad, *args):
                F = jnp.eye(self.dim) + u_grad
                C = F.T @ F
                J = jnp.linalg.det(F)
                I1 = jnp.trace(C)

                F_inv_T = jnp.linalg.inv(F).T
                P = mu * (F - F_inv_T) + kappa * (J - 1) * J * F_inv_T

                return P

            return first_PK_stress

        def get_surface_maps(self):
            return [lambda u, x, t: jnp.array([0., 0., t])]

    # Boundary conditions
    left = lambda p: jnp.isclose(p[0], 0., tol)
    right = lambda p: jnp.isclose(p[0], 10., tol)

    problem = NeoHookean(
        simple_mesh, vec=3, dim=3,
        location_fns=[right],
        matrix_view=MatrixView.FULL
    )

    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    traction = material_params['traction']
    surf_var = fe.InternalVars.create_uniform_surface_var(problem, traction)
    internal_vars = fe.InternalVars((), [(surf_var,)])

    solver_opts = fe.SolverOptions(
        linear_solver="cg",
        max_iter=20,
        tol=1e-6,
        rel_tol=1e-8
    )
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=20)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Solve
    solution = solver(internal_vars, initial)

    # Check final residual
    sol_list = problem.unflatten_fn_sol_list(solution)
    residual_list = fe.get_res(problem, sol_list, internal_vars)
    residual = jnp.concatenate([r.flatten() for r in residual_list])
    residual_bc = fe.apply_boundary_to_res(bc, residual, solution)
    residual_norm = jnp.linalg.norm(residual_bc)

    # Residual should be small after Newton iterations
    assert residual_norm < 1e-4, f"Residual too large: {residual_norm}"


def test_neohookean_different_solvers(
    simple_mesh,
    material_params
):
    """Test that different linear solvers work for hyperelastic problem."""
    E = material_params['E']
    nu = material_params['nu']
    tol = material_params['tol']

    mu = E / (2 * (1 + nu))
    kappa = E / (3 * (1 - 2 * nu))

    class NeoHookean(fe.Problem):
        def get_tensor_map(self):
            def first_PK_stress(u_grad, *args):
                F = jnp.eye(self.dim) + u_grad
                J = jnp.linalg.det(F)
                F_inv_T = jnp.linalg.inv(F).T
                P = mu * (F - F_inv_T) + kappa * (J - 1) * J * F_inv_T
                return P

            return first_PK_stress

        def get_surface_maps(self):
            return [lambda u, x, t: jnp.array([0., 0., t])]

    left = lambda p: jnp.isclose(p[0], 0., tol)
    right = lambda p: jnp.isclose(p[0], 10., tol)

    problem = NeoHookean(
        simple_mesh, vec=3, dim=3,
        location_fns=[right],
        matrix_view=MatrixView.FULL
    )

    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    traction = material_params['traction']
    surf_var = fe.InternalVars.create_uniform_surface_var(problem, traction)
    internal_vars = fe.InternalVars((), [(surf_var,)])

    # Test with CG, BICGSTAB, and GMRES
    solvers = ["cg", "bicgstab", "gmres"]
    solutions = []

    for solver_name in solvers:
        solver_opts = fe.SolverOptions(
            linear_solver=solver_name,
            max_iter=20,
            tol=1e-6
        )
        solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=20)
        initial = fe.zero_like_initial_guess(problem, bc)

        solution = solver(internal_vars, initial)
        solutions.append(solution)

        # Check solution is non-trivial
        assert jnp.linalg.norm(solution) > 0

    # All solvers should give similar solutions
    sol_tol = 1e-3
    diff_cg_bicgstab = jnp.linalg.norm(solutions[0] - solutions[1]) / jnp.linalg.norm(solutions[0])
    diff_cg_gmres = jnp.linalg.norm(solutions[0] - solutions[2]) / jnp.linalg.norm(solutions[0])

    assert diff_cg_bicgstab < sol_tol, f"CG and BICGSTAB solutions differ by {diff_cg_bicgstab:.2e}"
    assert diff_cg_gmres < sol_tol, f"CG and GMRES solutions differ by {diff_cg_gmres:.2e}"
