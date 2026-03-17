"""Verify Hessian implementation through a regularized Poisson problem.

Solves  -∇²u + ε∇⁴u = f  on [0,1]² with:
    u = sin(πx)sin(πy)   (exact solution of the continuous problem)
    f = (2π² + 4επ⁴) sin(πx)sin(πy)

Weak form:
    ∫ ∇u·∇v dΩ + ε ∫ (∇²u)(∇²v) dΩ = ∫ f·v dΩ

The standard Poisson part uses get_tensor_map (gradient-based).
The Hessian regularization uses get_universal_kernel with shape_hessians.
This mirrors the TMC HuHu-LuLu regularization pattern.

Note: C⁰ elements don't consistently discretize ∇⁴u across element
boundaries, so the discrete solution won't match the continuous exact
solution. Instead we verify:
1. Jacobian is symmetric and positive definite
2. Solver converges (no NaN)
3. Hessian adds non-zero, SPD contribution to stiffness
4. As ε→0, solution converges to standard Poisson
"""

import functools

import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp
import pytest

import feax as fe
from feax.assembler import Operator


pi = np.pi


def _make_problem_and_bc(epsilon, Nx=4, Ny=4):
    """Create regularized Poisson problem and BCs."""

    class RegularizedPoissonProblem(fe.Problem):
        def get_tensor_map(self):
            def stress(u_grad, *args):
                return u_grad
            return stress

        def get_mass_map(self):
            def mass(u, x, *args):
                f = (2.0 * pi**2 + 4.0 * epsilon * pi**4) * \
                    np.sin(pi * x[0]) * np.sin(pi * x[1])
                return -f * np.ones_like(u)
            return mass

        def get_universal_kernel(self):
            op = Operator(self)

            def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads,
                       cell_JxW, cell_v_grads_JxW, cell_shape_hess):
                cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
                cell_sol = cell_sol_list[0]
                cell_JxW_1d = cell_JxW[0]

                shape_lapl = np.trace(cell_shape_hess, axis1=-2, axis2=-1)
                lapl_u = np.einsum('a,qa->q', cell_sol[:, 0], shape_lapl)
                term = epsilon * np.einsum('q,qa,q->a', lapl_u, shape_lapl, cell_JxW_1d)

                result = term[:, None]
                return jax.flatten_util.ravel_pytree(result)[0]

            return kernel

    mesh = fe.mesh.rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=1.0, domain_y=1.0,
                                  ele_type='QUAD9')
    problem = RegularizedPoissonProblem(
        mesh, vec=1, dim=2, ele_type='QUAD9', hess=True)

    all_boundary = lambda p: (
        np.isclose(p[0], 0.0, atol=1e-6) |
        np.isclose(p[0], 1.0, atol=1e-6) |
        np.isclose(p[1], 0.0, atol=1e-6) |
        np.isclose(p[1], 1.0, atol=1e-6)
    )
    bc = fe.DirichletBCConfig([
        fe.DirichletBCSpec(all_boundary, 'all', 0.0),
    ]).create_bc(problem)

    shape_hess = problem.fes[0].shape_hessians
    iv = fe.InternalVars(volume_vars=(shape_hess,))
    return problem, bc, iv, mesh


def test_hessian_jacobian_symmetric_spd():
    """Verify the Hessian contribution produces a symmetric, SPD Jacobian."""
    problem, bc, iv, mesh = _make_problem_and_bc(epsilon=0.01)

    u_zero = fe.zero_like_initial_guess(problem, bc)
    J_fn = fe.create_J_bc_function(problem, bc)
    J = J_fn(u_zero, iv)
    J_dense = onp.array(J.todense())

    # Symmetric
    asym = onp.max(onp.abs(J_dense - J_dense.T))
    print(f"Jacobian asymmetry: {asym:.2e}")
    assert asym < 1e-12, f"Jacobian not symmetric: max|J-J^T| = {asym}"

    # Positive definite (interior block)
    interior_dofs = ~onp.isin(onp.arange(len(u_zero)), onp.array(bc.bc_rows))
    int_idx = onp.where(interior_dofs)[0]
    J_int = J_dense[onp.ix_(int_idx, int_idx)]
    eigs = onp.linalg.eigvalsh(J_int)
    print(f"Interior eigenvalues: [{eigs[0]:.4e}, {eigs[-1]:.4e}]")
    assert eigs[0] > 0, f"Jacobian not positive definite: min eigenvalue = {eigs[0]}"
    print("PASSED: Jacobian is symmetric and positive definite")


def test_hessian_nonzero_contribution():
    """Verify the Hessian adds a non-zero contribution to the Jacobian."""
    # Standard Poisson (no Hessian)
    class StdPoisson(fe.Problem):
        def get_tensor_map(self):
            def stress(u_grad, *args):
                return u_grad
            return stress

    Nx, Ny = 4, 4
    mesh = fe.mesh.rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=1.0, domain_y=1.0,
                                  ele_type='QUAD9')
    std_problem = StdPoisson(mesh, vec=1, dim=2, ele_type='QUAD9')
    all_boundary = lambda p: (
        np.isclose(p[0], 0.0, atol=1e-6) | np.isclose(p[0], 1.0, atol=1e-6) |
        np.isclose(p[1], 0.0, atol=1e-6) | np.isclose(p[1], 1.0, atol=1e-6)
    )
    std_bc = fe.DirichletBCConfig([
        fe.DirichletBCSpec(all_boundary, 'all', 0.0),
    ]).create_bc(std_problem)
    std_iv = fe.InternalVars()

    # Regularized Poisson (with Hessian)
    reg_problem, reg_bc, reg_iv, _ = _make_problem_and_bc(epsilon=0.01, Nx=Nx, Ny=Ny)

    u_zero = fe.zero_like_initial_guess(std_problem, std_bc)

    J_std = onp.array(fe.create_J_bc_function(std_problem, std_bc)(u_zero, std_iv).todense())
    J_reg = onp.array(fe.create_J_bc_function(reg_problem, reg_bc)(u_zero, reg_iv).todense())

    diff = onp.max(onp.abs(J_reg - J_std))
    print(f"Max |J_reg - J_std|: {diff:.6e}")
    assert diff > 0.1, f"Hessian contribution too small: {diff}"

    # Difference should also be symmetric
    dJ = J_reg - J_std
    asym = onp.max(onp.abs(dJ - dJ.T))
    print(f"Hessian contribution asymmetry: {asym:.2e}")
    assert asym < 1e-12
    print("PASSED: Hessian adds non-zero symmetric contribution")


def test_hessian_solver_converges():
    """Verify the solver converges with Hessian regularization."""
    problem, bc, iv, mesh = _make_problem_and_bc(epsilon=0.01, Nx=8, Ny=8)

    solver = fe.create_solver(
        problem, bc,
        solver_options=fe.IterativeSolverOptions(
            solver='cg', maxiter=5000, tol=1e-10,
            use_jacobi_preconditioner=True,
        ),
        iter_num=1,
        internal_vars=iv,
    )
    sol = solver(iv, fe.zero_like_initial_guess(problem, bc))
    u = problem.unflatten_fn_sol_list(sol)[0]
    u_num = onp.array(u[:, 0])

    assert not onp.any(onp.isnan(u_num)), "Solution contains NaN!"
    assert onp.max(onp.abs(u_num)) > 0.1, "Solution is trivially zero!"
    print(f"Max |u|: {onp.max(onp.abs(u_num)):.6e}")
    print("PASSED: Solver converges with Hessian regularization")


def test_hessian_vanishes_with_zero_epsilon():
    """As ε→0, regularized solution converges to standard Poisson."""
    Nx, Ny = 8, 8

    # Standard Poisson
    class StdPoisson(fe.Problem):
        def get_tensor_map(self):
            def stress(u_grad, *args):
                return u_grad
            return stress
        def get_mass_map(self):
            def mass(u, x, *args):
                f = 2.0 * pi**2 * np.sin(pi * x[0]) * np.sin(pi * x[1])
                return -f * np.ones_like(u)
            return mass

    mesh = fe.mesh.rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=1.0, domain_y=1.0,
                                  ele_type='QUAD9')
    std_problem = StdPoisson(mesh, vec=1, dim=2, ele_type='QUAD9')
    all_boundary = lambda p: (
        np.isclose(p[0], 0.0, atol=1e-6) | np.isclose(p[0], 1.0, atol=1e-6) |
        np.isclose(p[1], 0.0, atol=1e-6) | np.isclose(p[1], 1.0, atol=1e-6)
    )
    std_bc = fe.DirichletBCConfig([
        fe.DirichletBCSpec(all_boundary, 'all', 0.0),
    ]).create_bc(std_problem)
    std_iv = fe.InternalVars()
    std_solver = fe.create_solver(
        std_problem, std_bc,
        solver_options=fe.IterativeSolverOptions(
            solver='cg', maxiter=5000, tol=1e-10,
            use_jacobi_preconditioner=True,
        ),
        iter_num=1, internal_vars=std_iv,
    )
    std_sol = std_solver(std_iv, fe.zero_like_initial_guess(std_problem, std_bc))
    u_std = onp.array(std_problem.unflatten_fn_sol_list(std_sol)[0][:, 0])

    # Regularized with tiny ε (source term uses ε=0 to match standard Poisson)
    # This tests that the Hessian stiffness with ε→0 doesn't break the solution
    problem, bc, iv, _ = _make_problem_and_bc(epsilon=1e-8, Nx=Nx, Ny=Ny)
    solver = fe.create_solver(
        problem, bc,
        solver_options=fe.IterativeSolverOptions(
            solver='cg', maxiter=5000, tol=1e-10,
            use_jacobi_preconditioner=True,
        ),
        iter_num=1, internal_vars=iv,
    )
    sol = solver(iv, fe.zero_like_initial_guess(problem, bc))
    u_reg = onp.array(problem.unflatten_fn_sol_list(sol)[0][:, 0])

    diff = onp.max(onp.abs(u_reg - u_std))
    print(f"Max |u_reg - u_std| with ε=1e-8: {diff:.6e}")
    assert diff < 1e-4, f"Solutions diverge with small ε: diff = {diff}"
    print("PASSED: Solution converges to standard Poisson as ε→0")


if __name__ == '__main__':
    print("=== Test 1: Jacobian symmetric and SPD ===")
    test_hessian_jacobian_symmetric_spd()
    print()
    print("=== Test 2: Hessian non-zero contribution ===")
    test_hessian_nonzero_contribution()
    print()
    print("=== Test 3: Solver converges ===")
    test_hessian_solver_converges()
    print()
    print("=== Test 4: ε→0 convergence ===")
    test_hessian_vanishes_with_zero_epsilon()
    print()
    print("ALL TESTS PASSED")
