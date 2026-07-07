"""
Tests for Newton solver JIT and vmap compatibility.

All solvers here are built through ``fe.create_solver`` (internally
``create_newton_solver``, a traced ``lax.while_loop`` Newton iteration).
This module verifies that path composes with JAX transformations:
- JIT compilation (single and repeated calls)
- vmap over batched BC values / traced params
- vmap + grad and jit(vmap(grad(...))) composition via the custom VJP
- Armijo line search compatibility under jit/vmap
- nonlinear (linear=False) vs linear (linear=True) solver consistency

Note: test names referencing "fori"/"while" variants are historical — the
separate newton_solve/newton_solve_py/newton_solve_fori implementations were
unified into the single traced while_loop solver.
"""

import jax
import jax.numpy as np
import pytest

import feax as fe
from feax.problem import MatrixView

# ============================================================================
# Helpers
# ============================================================================

def create_linear_problem_and_bc(simple_mesh, material_params):
    """Create a linear elasticity problem with boundary conditions."""
    E, nu = material_params['E'], material_params['nu']
    tol = material_params['tol']

    class LinearElasticity(fe.Problem):
        def get_tensor_map(self):
            def stress(u_grad, *args):
                mu = E / (2 * (1 + nu))
                lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
                eps = 0.5 * (u_grad + u_grad.T)
                return lmbda * np.trace(eps) * np.eye(3) + 2 * mu * eps
            return stress

        def get_surface_maps(self):
            return [lambda u, x, t: np.array([0., 0., t])]

    right = lambda p: np.isclose(p[0], 10., tol)
    problem = LinearElasticity(
        simple_mesh, vec=3, dim=3,
        location_fns=[right],
        matrix_view=MatrixView.FULL
    )

    left = lambda p: np.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    traction = material_params['traction']
    surf_var = fe.TracedParams.create_uniform_surface_var(problem, traction)
    tp = fe.TracedParams((), [(surf_var,)])

    return problem, bc, tp


def create_nonlinear_problem_and_bc(simple_mesh, material_params):
    """Create a Neo-Hookean hyperelastic problem with boundary conditions."""
    E, nu = material_params['E'], material_params['nu']
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

    right = lambda p: np.isclose(p[0], 10., tol)
    problem = NeoHookean(
        simple_mesh, vec=3, dim=3,
        location_fns=[right],
        matrix_view=MatrixView.FULL
    )

    left = lambda p: np.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    traction = material_params['traction']
    surf_var = fe.TracedParams.create_uniform_surface_var(problem, traction)
    tp = fe.TracedParams((), [(surf_var,)])

    return problem, bc, tp


# ============================================================================
# JIT Tests - newton_solve (while_loop)
# ============================================================================

@pytest.mark.cpu
def test_newton_solve_py_no_jit(simple_mesh, material_params):
    """Test that newton_solve_py (default) works without JIT.

    The Python-loop path uses a Python while loop and cannot be wrapped in
    jax.jit or jax.vmap.  It is the default path, optimised for fast first-call
    compilation on large problems.
    """
    problem, bc, tp = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.KrylovSolverOptions(solver="cg", maxiter=10, tol=1e-6)
    # Default:  → Python-loop path
    solver = fe.create_solver(problem, bc, solver_options=solver_opts)
    initial = fe.zero_like_initial_guess(problem, bc)

    sol = solver(tp, initial)
    assert np.linalg.norm(sol) > 0
    assert np.all(np.isfinite(sol))


@pytest.mark.cpu
def test_newton_solve_jittable_jit(simple_mesh, material_params):
    """Test that newton_solve with  works with JIT."""
    problem, bc, tp = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.KrylovSolverOptions(solver="cg", maxiter=10, tol=1e-6)
    newton_opts = fe.NewtonOptions()
    solver = fe.create_solver(
        problem, bc, solver_options=solver_opts, linear=False,
        newton_options=newton_opts,
    )
    initial = fe.zero_like_initial_guess(problem, bc)

    # Without JIT
    sol_no_jit = solver(tp, initial)

    # With JIT
    solver_jit = jax.jit(solver)
    sol_jit = solver_jit(tp, initial)

    diff = np.linalg.norm(sol_no_jit - sol_jit)
    assert diff < 1e-10, f"JIT vs non-JIT differ by {diff:.2e}"
    assert np.linalg.norm(sol_jit) > 0


@pytest.mark.cpu
def test_newton_solve_jittable_jit_multiple_calls(simple_mesh, material_params):
    """Test that JIT-compiled jittable newton gives consistent results on repeated calls."""
    problem, bc, tp = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.KrylovSolverOptions(solver="cg", maxiter=10, tol=1e-6)
    newton_opts = fe.NewtonOptions()
    solver = fe.create_solver(
        problem, bc, solver_options=solver_opts, linear=False,
        newton_options=newton_opts,
    )
    initial = fe.zero_like_initial_guess(problem, bc)

    solver_jit = jax.jit(solver)
    sol1 = solver_jit(tp, initial)
    sol2 = solver_jit(tp, initial)

    diff = np.linalg.norm(sol1 - sol2)
    assert diff < 1e-12, f"Repeated JIT calls differ by {diff:.2e}"


# ============================================================================
# JIT Tests - newton_solve_fori (fori_loop)
# ============================================================================

@pytest.mark.cpu
def test_newton_solve_fori_jit(simple_mesh, material_params):
    """Test that newton_solve_fori (iter_num>1, fori_loop) works with JIT."""
    problem, bc, tp = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.KrylovSolverOptions(solver="cg", maxiter=10, tol=1e-6)
    newton_opts = fe.NewtonOptions()
    solver = fe.create_solver(
        problem, bc, solver_options=solver_opts, linear=False,
        newton_options=newton_opts,
    )
    initial = fe.zero_like_initial_guess(problem, bc)

    # Without JIT
    sol_no_jit = solver(tp, initial)

    # With JIT
    solver_jit = jax.jit(solver)
    sol_jit = solver_jit(tp, initial)

    diff = np.linalg.norm(sol_no_jit - sol_jit)
    assert diff < 1e-10, f"fori JIT vs non-JIT differ by {diff:.2e}"
    assert np.linalg.norm(sol_jit) > 0


# ============================================================================
# JIT Tests - linear_solve (linear=True)
# ============================================================================

@pytest.mark.cpu
def test_linear_solve_jit(simple_mesh, material_params):
    """Test that linear_solve (linear=True) works with JIT."""
    problem, bc, tp = create_linear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.KrylovSolverOptions(solver="cg")
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, linear=True)
    initial = fe.zero_like_initial_guess(problem, bc)

    sol_no_jit = solver(tp, initial)

    solver_jit = jax.jit(solver)
    sol_jit = solver_jit(tp, initial)

    diff = np.linalg.norm(sol_no_jit - sol_jit)
    assert diff < 1e-10, f"linear_solve JIT vs non-JIT differ by {diff:.2e}"


# ============================================================================
# vmap Tests - newton_solve_fori (fori_loop + scan Armijo)
# ============================================================================

@pytest.mark.cpu
def test_newton_solve_fori_vmap(simple_mesh, material_params):
    """Test that newton_solve_fori with scan-based Armijo is vmappable."""
    problem, bc, tp = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.KrylovSolverOptions(solver="cg", maxiter=10, tol=1e-6)
    newton_opts = fe.NewtonOptions()
    solver = fe.create_solver(
        problem, bc, solver_options=solver_opts, linear=False,
        newton_options=newton_opts,
    )
    initial = fe.zero_like_initial_guess(problem, bc)

    # Create batch of surface variables
    batch_size = 3
    surf_var = tp.surface_vars[0][0]
    batch_surface_vars = np.stack([
        surf_var * (1.0 + 0.1 * i) for i in range(batch_size)
    ])

    def make_iv(ts):
        return fe.TracedParams((), [(ts,)])

    vmapped_solver = jax.vmap(lambda ts: solver(make_iv(ts), initial))
    batch_solutions = vmapped_solver(batch_surface_vars)

    assert batch_solutions.shape[0] == batch_size
    for i in range(batch_size):
        assert np.linalg.norm(batch_solutions[i]) > 0, f"Solution {i} is trivial"
        assert np.all(np.isfinite(batch_solutions[i])), f"Solution {i} has NaN/Inf"


@pytest.mark.cpu
def test_newton_solve_fori_vmap_jit(simple_mesh, material_params):
    """Test that vmap + JIT composition works for newton_solve_fori."""
    problem, bc, tp = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.KrylovSolverOptions(solver="cg", maxiter=10, tol=1e-6)
    newton_opts = fe.NewtonOptions()
    solver = fe.create_solver(
        problem, bc, solver_options=solver_opts, linear=False,
        newton_options=newton_opts,
    )
    initial = fe.zero_like_initial_guess(problem, bc)

    batch_size = 3
    surf_var = tp.surface_vars[0][0]
    batch_surface_vars = np.stack([
        surf_var * (1.0 + 0.1 * i) for i in range(batch_size)
    ])

    def make_iv(ts):
        return fe.TracedParams((), [(ts,)])

    # jit(vmap(...))
    solver_vmap_jit = jax.jit(jax.vmap(lambda ts: solver(make_iv(ts), initial)))
    batch_solutions = solver_vmap_jit(batch_surface_vars)

    assert batch_solutions.shape[0] == batch_size
    for i in range(batch_size):
        assert np.all(np.isfinite(batch_solutions[i])), f"Solution {i} has NaN/Inf"


# ============================================================================
# vmap Tests - linear_solve (linear=True)
# ============================================================================

@pytest.mark.cpu
def test_linear_solve_vmap(simple_mesh, material_params):
    """Test that linear_solve (linear=True) is vmappable."""
    problem, bc, tp = create_linear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.KrylovSolverOptions(solver="cg")
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, linear=True)
    initial = fe.zero_like_initial_guess(problem, bc)

    batch_size = 3
    surf_var = tp.surface_vars[0][0]
    batch_surface_vars = np.stack([
        surf_var * (1.0 + 0.01 * i) for i in range(batch_size)
    ])

    def make_iv(ts):
        return fe.TracedParams((), [(ts,)])

    vmapped_solver = jax.vmap(lambda ts: solver(make_iv(ts), initial))
    batch_solutions = vmapped_solver(batch_surface_vars)

    assert batch_solutions.shape[0] == batch_size
    for i in range(batch_size):
        assert np.linalg.norm(batch_solutions[i]) > 0


# ============================================================================
# grad Tests - Newton solver
# ============================================================================

@pytest.mark.cpu
def test_newton_solve_grad(simple_mesh, material_params):
    """Test that newton_solve (while_loop) supports grad via custom VJP."""
    problem, bc, tp = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.KrylovSolverOptions(solver="cg", maxiter=10, tol=1e-6)
    solver = fe.create_solver(problem, bc, solver_options=solver_opts)
    initial = fe.zero_like_initial_guess(problem, bc)

    surf_var = tp.surface_vars[0][0]

    def loss_fn(ts):
        iv_local = fe.TracedParams((), [(ts,)])
        sol = solver(iv_local, initial)
        return np.sum(sol**2)

    grad_fn = jax.grad(loss_fn)
    grad_val = grad_fn(surf_var)

    assert np.all(np.isfinite(grad_val)), "Gradient contains NaN/Inf"
    assert np.linalg.norm(grad_val) > 0, "Gradient is trivial"


@pytest.mark.cpu
def test_newton_solve_fori_grad(simple_mesh, material_params):
    """Test that newton_solve_fori supports grad via custom VJP."""
    problem, bc, tp = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.KrylovSolverOptions(solver="cg", maxiter=10, tol=1e-6)
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, linear=False)
    initial = fe.zero_like_initial_guess(problem, bc)

    surf_var = tp.surface_vars[0][0]

    def loss_fn(ts):
        iv_local = fe.TracedParams((), [(ts,)])
        sol = solver(iv_local, initial)
        return np.sum(sol**2)

    grad_fn = jax.grad(loss_fn)
    grad_val = grad_fn(surf_var)

    assert np.all(np.isfinite(grad_val)), "Gradient contains NaN/Inf"
    assert np.linalg.norm(grad_val) > 0, "Gradient is trivial"


# ============================================================================
# vmap + grad Composition Tests
# ============================================================================

@pytest.mark.cpu
def test_newton_solve_fori_vmap_grad(simple_mesh, material_params):
    """Test that vmap(grad(...)) works for newton_solve_fori."""
    problem, bc, tp = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.KrylovSolverOptions(solver="cg", maxiter=10, tol=1e-6)
    newton_opts = fe.NewtonOptions()
    solver = fe.create_solver(
        problem, bc, solver_options=solver_opts, linear=False,
        newton_options=newton_opts,
    )
    initial = fe.zero_like_initial_guess(problem, bc)

    batch_size = 3
    surf_var = tp.surface_vars[0][0]
    batch_surface_vars = np.stack([
        surf_var * (1.0 + 0.1 * i) for i in range(batch_size)
    ])

    def loss_fn(ts):
        iv_local = fe.TracedParams((), [(ts,)])
        sol = solver(iv_local, initial)
        return np.sum(sol**2)

    grad_fn = jax.vmap(jax.grad(loss_fn))
    batch_grads = grad_fn(batch_surface_vars)

    assert batch_grads.shape[0] == batch_size
    for i in range(batch_size):
        assert np.all(np.isfinite(batch_grads[i])), f"Gradient {i} has NaN/Inf"
        assert np.linalg.norm(batch_grads[i]) > 0, f"Gradient {i} is trivial"


@pytest.mark.cpu
def test_newton_solve_fori_jit_vmap_grad(simple_mesh, material_params):
    """Test that jit(vmap(grad(...))) works for newton_solve_fori."""
    problem, bc, tp = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.KrylovSolverOptions(solver="cg", maxiter=10, tol=1e-6)
    newton_opts = fe.NewtonOptions()
    solver = fe.create_solver(
        problem, bc, solver_options=solver_opts, linear=False,
        newton_options=newton_opts,
    )
    initial = fe.zero_like_initial_guess(problem, bc)

    batch_size = 3
    surf_var = tp.surface_vars[0][0]
    batch_surface_vars = np.stack([
        surf_var * (1.0 + 0.1 * i) for i in range(batch_size)
    ])

    def loss_fn(ts):
        iv_local = fe.TracedParams((), [(ts,)])
        sol = solver(iv_local, initial)
        return np.sum(sol**2)

    grad_fn = jax.jit(jax.vmap(jax.grad(loss_fn)))
    batch_grads = grad_fn(batch_surface_vars)

    assert batch_grads.shape[0] == batch_size
    for i in range(batch_size):
        assert np.all(np.isfinite(batch_grads[i])), f"Gradient {i} has NaN/Inf"


# ============================================================================
# Solver Consistency Tests
# ============================================================================

@pytest.mark.cpu
def test_newton_while_vs_fori_consistency(simple_mesh, material_params):
    """Test that newton_solve (while) and newton_solve_fori give similar solutions."""
    problem, bc, tp = create_nonlinear_problem_and_bc(simple_mesh, material_params)
    initial = fe.zero_like_initial_guess(problem, bc)

    solver_opts = fe.KrylovSolverOptions(solver="cg", maxiter=20, tol=1e-8)

    # while_loop version
    solver_while = fe.create_solver(problem, bc, solver_options=solver_opts)
    sol_while = solver_while(tp, initial)

    # fori_loop version with enough iterations
    solver_fori = fe.create_solver(problem, bc, solver_options=solver_opts, linear=False)
    sol_fori = solver_fori(tp, initial)

    # Should produce similar results
    diff = np.linalg.norm(sol_while - sol_fori) / (np.linalg.norm(sol_while) + 1e-30)
    assert diff < 1e-4, f"while vs fori differ by {diff:.2e}"


# ============================================================================
# create_linear_solver Tests
# ============================================================================

@pytest.mark.cpu
def test_create_linear_solver_jit(simple_mesh, material_params):
    """Test that create_linear_solver works with JIT."""
    problem, bc, tp = create_linear_problem_and_bc(simple_mesh, material_params)

    solver = fe.create_linear_solver(problem, bc)
    initial = fe.zero_like_initial_guess(problem, bc)

    sol_no_jit = solver(tp, initial)

    solver_jit = jax.jit(solver)
    sol_jit = solver_jit(tp, initial)

    diff = np.linalg.norm(sol_no_jit - sol_jit)
    assert diff < 1e-10, f"create_linear_solver JIT diff: {diff:.2e}"


@pytest.mark.cpu
def test_create_linear_solver_vmap(simple_mesh, material_params):
    """Test that create_linear_solver works with vmap."""
    problem, bc, tp = create_linear_problem_and_bc(simple_mesh, material_params)

    solver = fe.create_linear_solver(problem, bc)
    initial = fe.zero_like_initial_guess(problem, bc)

    batch_size = 3
    surf_var = tp.surface_vars[0][0]
    batch_surface_vars = np.stack([
        surf_var * (1.0 + 0.01 * i) for i in range(batch_size)
    ])

    def make_iv(ts):
        return fe.TracedParams((), [(ts,)])

    vmapped = jax.vmap(lambda ts: solver(make_iv(ts), initial))
    batch_solutions = vmapped(batch_surface_vars)

    assert batch_solutions.shape[0] == batch_size
    for i in range(batch_size):
        assert np.linalg.norm(batch_solutions[i]) > 0


@pytest.mark.cpu
def test_create_linear_solver_grad(simple_mesh, material_params):
    """Test that create_linear_solver supports grad."""
    problem, bc, tp = create_linear_problem_and_bc(simple_mesh, material_params)

    solver = fe.create_linear_solver(problem, bc)
    initial = fe.zero_like_initial_guess(problem, bc)

    surf_var = tp.surface_vars[0][0]

    def loss_fn(ts):
        iv_local = fe.TracedParams((), [(ts,)])
        sol = solver(iv_local, initial)
        return np.sum(sol**2)

    grad_val = jax.grad(loss_fn)(surf_var)
    assert np.all(np.isfinite(grad_val))
    assert np.linalg.norm(grad_val) > 0


@pytest.mark.cpu
def test_create_linear_solver_vs_create_solver_iter1(simple_mesh, material_params):
    """Test that create_linear_solver and create_solver(linear=True) give identical results."""
    problem, bc, tp = create_linear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.KrylovSolverOptions(solver="cg")
    solver1 = fe.create_linear_solver(problem, bc, solver_options=solver_opts)
    solver2 = fe.create_solver(problem, bc, solver_options=solver_opts, linear=True)
    initial = fe.zero_like_initial_guess(problem, bc)

    sol1 = solver1(tp, initial)
    sol2 = solver2(tp, initial)

    diff = np.linalg.norm(sol1 - sol2)
    assert diff < 1e-10, f"linear_solver vs create_solver(1) differ by {diff:.2e}"
