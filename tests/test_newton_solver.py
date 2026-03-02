"""
Tests for hyperelastic (non-linear) problem solving.

This module tests hyperelasticity with:
- Neo-Hookean material model
- Newton solver convergence
- Non-linear solution accuracy
- Large deformation capability
"""

import jax
import jax.numpy as np
import pytest

import feax as fe
from feax.problem import MatrixView

# ============================================================================
# Environment Checks (CUDA/cuDSS-specific regression tests)
# ============================================================================

def has_gpu():
    """Check if GPU is available."""
    try:
        devices = jax.devices("gpu")
        return len(devices) > 0
    except Exception:
        return False


def has_cudss():
    """Check if cuDSS backend is available."""
    if not has_gpu():
        return False
    try:
        from spineax.cudss.solver import CuDSSSolver  # noqa: F401
        return True
    except Exception:
        return False


requires_cudss = pytest.mark.skipif(
    not has_cudss(),
    reason="cuDSS not available (requires GPU + spineax)"
)


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
                F = np.eye(self.dim) + u_grad

                # Right Cauchy-Green tensor C = F^T F
                C = F.T @ F

                # Invariants
                J = np.linalg.det(F)
                I1 = np.trace(C)

                # Neo-Hookean strain energy derivatives
                # W = (mu/2)(I1 - 3) - mu*ln(J) + (kappa/2)(J-1)^2

                # First Piola-Kirchhoff stress
                # P = mu(F - F^-T) + kappa(J-1)J F^-T
                F_inv_T = np.linalg.inv(F).T
                P = mu * (F - F_inv_T) + kappa * (J - 1) * J * F_inv_T

                return P

            return first_PK_stress

        def get_surface_maps(self):
            return [lambda u, x, t: np.array([0., 0., t])]

    # Boundary conditions
    left = lambda p: np.isclose(p[0], 0., tol)
    right = lambda p: np.isclose(p[0], 10., tol)

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
    solver_opts = fe.IterativeSolverOptions(
        solver="cg",
        maxiter=20,
        tol=1e-6
    )
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=20)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Solve
    solution = solver(internal_vars, initial)

    # Check solution is non-trivial
    solution_norm = np.linalg.norm(solution)
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
                F = np.eye(self.dim) + u_grad
                C = F.T @ F
                J = np.linalg.det(F)
                I1 = np.trace(C)

                F_inv_T = np.linalg.inv(F).T
                P = mu * (F - F_inv_T) + kappa * (J - 1) * J * F_inv_T

                return P

            return first_PK_stress

        def get_surface_maps(self):
            return [lambda u, x, t: np.array([0., 0., t])]

    # Boundary conditions
    left = lambda p: np.isclose(p[0], 0., tol)
    right = lambda p: np.isclose(p[0], 10., tol)

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

    solver_opts = fe.IterativeSolverOptions(
        solver="cg",
        maxiter=20,
        tol=1e-6
    )
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=20)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Solve
    solution = solver(internal_vars, initial)

    # Check final residual
    sol_list = problem.unflatten_fn_sol_list(solution)
    residual_list = fe.get_res(problem, sol_list, internal_vars)
    residual = np.concatenate([r.flatten() for r in residual_list])
    residual_bc = fe.apply_boundary_to_res(bc, residual, solution)
    residual_norm = np.linalg.norm(residual_bc)

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
                F = np.eye(self.dim) + u_grad
                J = np.linalg.det(F)
                F_inv_T = np.linalg.inv(F).T
                P = mu * (F - F_inv_T) + kappa * (J - 1) * J * F_inv_T
                return P

            return first_PK_stress

        def get_surface_maps(self):
            return [lambda u, x, t: np.array([0., 0., t])]

    left = lambda p: np.isclose(p[0], 0., tol)
    right = lambda p: np.isclose(p[0], 10., tol)

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
        solver_opts = fe.IterativeSolverOptions(
            solver=solver_name,
            maxiter=20,
            tol=1e-6
        )
        solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=20)
        initial = fe.zero_like_initial_guess(problem, bc)

        solution = solver(internal_vars, initial)
        solutions.append(solution)

        # Check solution is non-trivial
        assert np.linalg.norm(solution) > 0

    # All solvers should give similar solutions
    sol_tol = 1e-3
    diff_cg_bicgstab = np.linalg.norm(solutions[0] - solutions[1]) / np.linalg.norm(solutions[0])
    diff_cg_gmres = np.linalg.norm(solutions[0] - solutions[2]) / np.linalg.norm(solutions[0])

    assert diff_cg_bicgstab < sol_tol, f"CG and BICGSTAB solutions differ by {diff_cg_bicgstab:.2e}"
    assert diff_cg_gmres < sol_tol, f"CG and GMRES solutions differ by {diff_cg_gmres:.2e}"


@pytest.mark.cuda
@requires_cudss
def test_newton_cudss_grad_prewarm_regression(
    simple_mesh,
    material_params
):
    """Regression: grad through Newton+cuDSS should not hit tracer-leak issues."""
    E = material_params["E"]
    nu = material_params["nu"]
    tol = material_params["tol"]

    mu = E / (2 * (1 + nu))
    kappa = E / (3 * (1 - 2 * nu))

    class NeoHookean(fe.Problem):
        def get_tensor_map(self):
            def first_PK_stress(u_grad, *args):
                F = np.eye(self.dim) + u_grad
                J = np.linalg.det(F)
                F_inv_T = np.linalg.inv(F).T
                return mu * (F - F_inv_T) + kappa * (J - 1) * J * F_inv_T

            return first_PK_stress

        def get_surface_maps(self):
            return [lambda u, x, t: np.array([0.0, 0.0, t])]

    left = lambda p: np.isclose(p[0], 0.0, tol)
    right = lambda p: np.isclose(p[0], 10.0, tol)

    problem = NeoHookean(
        simple_mesh, vec=3, dim=3,
        location_fns=[right],
        matrix_view=MatrixView.FULL,
    )

    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.0)
    bc = fe.DirichletBCConfig([left_fix]).create_bc(problem)

    traction0 = material_params["traction"]
    surf0 = fe.InternalVars.create_uniform_surface_var(problem, traction0)
    sample_internal_vars = fe.InternalVars((), [(surf0,)])

    solver = fe.create_solver(
        problem,
        bc,
        solver_options=fe.DirectSolverOptions(solver="cudss"),
        newton_options=fe.NewtonOptions(max_iter=5, tol=1e-6),
        internal_vars=sample_internal_vars,
    )
    initial = fe.zero_like_initial_guess(problem, bc)

    def loss(traction):
        surf = fe.InternalVars.create_uniform_surface_var(problem, traction)
        iv = fe.InternalVars((), [(surf,)])
        sol = solver(iv, initial)
        return np.sum(sol ** 2)

    grad_val = jax.grad(loss)(traction0)
    assert np.isfinite(grad_val), "Gradient should be finite for Newton+cuDSS path"
