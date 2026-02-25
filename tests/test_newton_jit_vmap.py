"""
Tests for Newton solver JIT and vmap compatibility.

This module verifies that the Newton solver variants (while_loop, fori_loop)
work correctly with JAX transformations:
- JIT compilation for all solver variants
- vmap for fori_loop variant (iter_num > 1)
- vmap + grad composition for fori_loop variant
- Armijo line search compatibility (while_loop vs scan)
- newton_solve_py as reference implementation
"""

import pytest
import jax
import jax.numpy as jnp
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
                return lmbda * jnp.trace(eps) * jnp.eye(3) + 2 * mu * eps
            return stress

        def get_surface_maps(self):
            return [lambda u, x, t: jnp.array([0., 0., t])]

    right = lambda p: jnp.isclose(p[0], 10., tol)
    problem = LinearElasticity(
        simple_mesh, vec=3, dim=3,
        location_fns=[right],
        matrix_view=MatrixView.FULL
    )

    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    traction = material_params['traction']
    surf_var = fe.InternalVars.create_uniform_surface_var(problem, traction)
    iv = fe.InternalVars((), [(surf_var,)])

    return problem, bc, iv


def create_nonlinear_problem_and_bc(simple_mesh, material_params):
    """Create a Neo-Hookean hyperelastic problem with boundary conditions."""
    E, nu = material_params['E'], material_params['nu']
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

    right = lambda p: jnp.isclose(p[0], 10., tol)
    problem = NeoHookean(
        simple_mesh, vec=3, dim=3,
        location_fns=[right],
        matrix_view=MatrixView.FULL
    )

    left = lambda p: jnp.isclose(p[0], 0., tol)
    left_fix = fe.DirichletBCSpec(location=left, component="all", value=0.)
    bc_config = fe.DirichletBCConfig([left_fix])
    bc = bc_config.create_bc(problem)

    traction = material_params['traction']
    surf_var = fe.InternalVars.create_uniform_surface_var(problem, traction)
    iv = fe.InternalVars((), [(surf_var,)])

    return problem, bc, iv


# ============================================================================
# JIT Tests - newton_solve (while_loop)
# ============================================================================

@pytest.mark.cpu
def test_newton_solve_jit(simple_mesh, material_params):
    """Test that newton_solve (iter_num=None, while_loop) works with JIT."""
    problem, bc, iv = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.SolverOptions(linear_solver="cg", max_iter=10, tol=1e-6)
    solver = fe.create_solver(problem, bc, solver_options=solver_opts)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Without JIT
    sol_no_jit = solver(iv, initial)

    # With JIT
    solver_jit = jax.jit(solver)
    sol_jit = solver_jit(iv, initial)

    diff = jnp.linalg.norm(sol_no_jit - sol_jit)
    assert diff < 1e-10, f"JIT vs non-JIT differ by {diff:.2e}"
    assert jnp.linalg.norm(sol_jit) > 0


@pytest.mark.cpu
def test_newton_solve_jit_multiple_calls(simple_mesh, material_params):
    """Test that JIT-compiled newton_solve gives consistent results on repeated calls."""
    problem, bc, iv = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.SolverOptions(linear_solver="cg", max_iter=10, tol=1e-6)
    solver = fe.create_solver(problem, bc, solver_options=solver_opts)
    initial = fe.zero_like_initial_guess(problem, bc)

    solver_jit = jax.jit(solver)
    sol1 = solver_jit(iv, initial)
    sol2 = solver_jit(iv, initial)

    diff = jnp.linalg.norm(sol1 - sol2)
    assert diff < 1e-12, f"Repeated JIT calls differ by {diff:.2e}"


# ============================================================================
# JIT Tests - newton_solve_fori (fori_loop)
# ============================================================================

@pytest.mark.cpu
def test_newton_solve_fori_jit(simple_mesh, material_params):
    """Test that newton_solve_fori (iter_num>1, fori_loop) works with JIT."""
    problem, bc, iv = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.SolverOptions(linear_solver="cg", max_iter=10, tol=1e-6)
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=10)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Without JIT
    sol_no_jit = solver(iv, initial)

    # With JIT
    solver_jit = jax.jit(solver)
    sol_jit = solver_jit(iv, initial)

    diff = jnp.linalg.norm(sol_no_jit - sol_jit)
    assert diff < 1e-10, f"fori JIT vs non-JIT differ by {diff:.2e}"
    assert jnp.linalg.norm(sol_jit) > 0


# ============================================================================
# JIT Tests - linear_solve (iter_num=1)
# ============================================================================

@pytest.mark.cpu
def test_linear_solve_jit(simple_mesh, material_params):
    """Test that linear_solve (iter_num=1) works with JIT."""
    problem, bc, iv = create_linear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.SolverOptions(linear_solver="cg")
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
    initial = fe.zero_like_initial_guess(problem, bc)

    sol_no_jit = solver(iv, initial)

    solver_jit = jax.jit(solver)
    sol_jit = solver_jit(iv, initial)

    diff = jnp.linalg.norm(sol_no_jit - sol_jit)
    assert diff < 1e-10, f"linear_solve JIT vs non-JIT differ by {diff:.2e}"


# ============================================================================
# vmap Tests - newton_solve_fori (fori_loop + scan Armijo)
# ============================================================================

@pytest.mark.cpu
def test_newton_solve_fori_vmap(simple_mesh, material_params):
    """Test that newton_solve_fori with scan-based Armijo is vmappable."""
    problem, bc, iv = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.SolverOptions(linear_solver="cg", max_iter=10, tol=1e-6)
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=10)
    initial = fe.zero_like_initial_guess(problem, bc)

    # Create batch of surface variables
    batch_size = 3
    surf_var = iv.surface_vars[0][0]
    batch_surface_vars = jnp.stack([
        surf_var * (1.0 + 0.1 * i) for i in range(batch_size)
    ])

    def make_iv(sv):
        return fe.InternalVars((), [(sv,)])

    vmapped_solver = jax.vmap(lambda sv: solver(make_iv(sv), initial))
    batch_solutions = vmapped_solver(batch_surface_vars)

    assert batch_solutions.shape[0] == batch_size
    for i in range(batch_size):
        assert jnp.linalg.norm(batch_solutions[i]) > 0, f"Solution {i} is trivial"
        assert jnp.all(jnp.isfinite(batch_solutions[i])), f"Solution {i} has NaN/Inf"


@pytest.mark.cpu
def test_newton_solve_fori_vmap_jit(simple_mesh, material_params):
    """Test that vmap + JIT composition works for newton_solve_fori."""
    problem, bc, iv = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.SolverOptions(linear_solver="cg", max_iter=10, tol=1e-6)
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=10)
    initial = fe.zero_like_initial_guess(problem, bc)

    batch_size = 3
    surf_var = iv.surface_vars[0][0]
    batch_surface_vars = jnp.stack([
        surf_var * (1.0 + 0.1 * i) for i in range(batch_size)
    ])

    def make_iv(sv):
        return fe.InternalVars((), [(sv,)])

    # jit(vmap(...))
    solver_vmap_jit = jax.jit(jax.vmap(lambda sv: solver(make_iv(sv), initial)))
    batch_solutions = solver_vmap_jit(batch_surface_vars)

    assert batch_solutions.shape[0] == batch_size
    for i in range(batch_size):
        assert jnp.all(jnp.isfinite(batch_solutions[i])), f"Solution {i} has NaN/Inf"


# ============================================================================
# vmap Tests - linear_solve (iter_num=1)
# ============================================================================

@pytest.mark.cpu
def test_linear_solve_vmap(simple_mesh, material_params):
    """Test that linear_solve (iter_num=1) is vmappable."""
    problem, bc, iv = create_linear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.SolverOptions(linear_solver="cg")
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
    initial = fe.zero_like_initial_guess(problem, bc)

    batch_size = 3
    surf_var = iv.surface_vars[0][0]
    batch_surface_vars = jnp.stack([
        surf_var * (1.0 + 0.01 * i) for i in range(batch_size)
    ])

    def make_iv(sv):
        return fe.InternalVars((), [(sv,)])

    vmapped_solver = jax.vmap(lambda sv: solver(make_iv(sv), initial))
    batch_solutions = vmapped_solver(batch_surface_vars)

    assert batch_solutions.shape[0] == batch_size
    for i in range(batch_size):
        assert jnp.linalg.norm(batch_solutions[i]) > 0


# ============================================================================
# grad Tests - Newton solver
# ============================================================================

@pytest.mark.cpu
def test_newton_solve_grad(simple_mesh, material_params):
    """Test that newton_solve (while_loop) supports grad via custom VJP."""
    problem, bc, iv = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.SolverOptions(linear_solver="cg", max_iter=10, tol=1e-6)
    solver = fe.create_solver(problem, bc, solver_options=solver_opts)
    initial = fe.zero_like_initial_guess(problem, bc)

    surf_var = iv.surface_vars[0][0]

    def loss_fn(sv):
        iv_local = fe.InternalVars((), [(sv,)])
        sol = solver(iv_local, initial)
        return jnp.sum(sol**2)

    grad_fn = jax.grad(loss_fn)
    grad_val = grad_fn(surf_var)

    assert jnp.all(jnp.isfinite(grad_val)), "Gradient contains NaN/Inf"
    assert jnp.linalg.norm(grad_val) > 0, "Gradient is trivial"


@pytest.mark.cpu
def test_newton_solve_fori_grad(simple_mesh, material_params):
    """Test that newton_solve_fori supports grad via custom VJP."""
    problem, bc, iv = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.SolverOptions(linear_solver="cg", max_iter=10, tol=1e-6)
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=10)
    initial = fe.zero_like_initial_guess(problem, bc)

    surf_var = iv.surface_vars[0][0]

    def loss_fn(sv):
        iv_local = fe.InternalVars((), [(sv,)])
        sol = solver(iv_local, initial)
        return jnp.sum(sol**2)

    grad_fn = jax.grad(loss_fn)
    grad_val = grad_fn(surf_var)

    assert jnp.all(jnp.isfinite(grad_val)), "Gradient contains NaN/Inf"
    assert jnp.linalg.norm(grad_val) > 0, "Gradient is trivial"


# ============================================================================
# vmap + grad Composition Tests
# ============================================================================

@pytest.mark.cpu
def test_newton_solve_fori_vmap_grad(simple_mesh, material_params):
    """Test that vmap(grad(...)) works for newton_solve_fori."""
    problem, bc, iv = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.SolverOptions(linear_solver="cg", max_iter=10, tol=1e-6)
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=10)
    initial = fe.zero_like_initial_guess(problem, bc)

    batch_size = 3
    surf_var = iv.surface_vars[0][0]
    batch_surface_vars = jnp.stack([
        surf_var * (1.0 + 0.1 * i) for i in range(batch_size)
    ])

    def loss_fn(sv):
        iv_local = fe.InternalVars((), [(sv,)])
        sol = solver(iv_local, initial)
        return jnp.sum(sol**2)

    grad_fn = jax.vmap(jax.grad(loss_fn))
    batch_grads = grad_fn(batch_surface_vars)

    assert batch_grads.shape[0] == batch_size
    for i in range(batch_size):
        assert jnp.all(jnp.isfinite(batch_grads[i])), f"Gradient {i} has NaN/Inf"
        assert jnp.linalg.norm(batch_grads[i]) > 0, f"Gradient {i} is trivial"


@pytest.mark.cpu
def test_newton_solve_fori_jit_vmap_grad(simple_mesh, material_params):
    """Test that jit(vmap(grad(...))) works for newton_solve_fori."""
    problem, bc, iv = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.SolverOptions(linear_solver="cg", max_iter=10, tol=1e-6)
    solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=10)
    initial = fe.zero_like_initial_guess(problem, bc)

    batch_size = 3
    surf_var = iv.surface_vars[0][0]
    batch_surface_vars = jnp.stack([
        surf_var * (1.0 + 0.1 * i) for i in range(batch_size)
    ])

    def loss_fn(sv):
        iv_local = fe.InternalVars((), [(sv,)])
        sol = solver(iv_local, initial)
        return jnp.sum(sol**2)

    grad_fn = jax.jit(jax.vmap(jax.grad(loss_fn)))
    batch_grads = grad_fn(batch_surface_vars)

    assert batch_grads.shape[0] == batch_size
    for i in range(batch_size):
        assert jnp.all(jnp.isfinite(batch_grads[i])), f"Gradient {i} has NaN/Inf"


# ============================================================================
# Solver Consistency Tests
# ============================================================================

@pytest.mark.cpu
def test_newton_while_vs_fori_consistency(simple_mesh, material_params):
    """Test that newton_solve (while) and newton_solve_fori give similar solutions."""
    problem, bc, iv = create_nonlinear_problem_and_bc(simple_mesh, material_params)
    initial = fe.zero_like_initial_guess(problem, bc)

    solver_opts = fe.SolverOptions(linear_solver="cg", max_iter=20, tol=1e-8)

    # while_loop version
    solver_while = fe.create_solver(problem, bc, solver_options=solver_opts)
    sol_while = solver_while(iv, initial)

    # fori_loop version with enough iterations
    solver_fori = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=20)
    sol_fori = solver_fori(iv, initial)

    # Should produce similar results
    diff = jnp.linalg.norm(sol_while - sol_fori) / (jnp.linalg.norm(sol_while) + 1e-30)
    assert diff < 1e-4, f"while vs fori differ by {diff:.2e}"


@pytest.mark.cpu
def test_newton_py_vs_jax_consistency(simple_mesh, material_params):
    """Test that newton_solve_py and newton_solve give similar solutions."""
    problem, bc, iv = create_nonlinear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.SolverOptions(linear_solver="cg", max_iter=20, tol=1e-8)

    # JAX while_loop version via create_solver
    solver_jax = fe.create_solver(problem, bc, solver_options=solver_opts)
    initial = fe.zero_like_initial_guess(problem, bc)
    sol_jax = solver_jax(iv, initial)

    # Python version (direct call)
    from feax.assembler import create_J_bc_function, create_res_bc_function
    J_bc_func = create_J_bc_function(problem, bc)
    res_bc_func = create_res_bc_function(problem, bc)
    sol_py, _, _, _ = fe.newton_solve_py(
        J_bc_func, res_bc_func, initial, bc, solver_opts, internal_vars=iv
    )

    diff = jnp.linalg.norm(sol_jax - sol_py) / (jnp.linalg.norm(sol_jax) + 1e-30)
    assert diff < 1e-4, f"JAX vs Python Newton differ by {diff:.2e}"


# ============================================================================
# Armijo Line Search Variant Tests
# ============================================================================

@pytest.mark.cpu
def test_armijo_while_vs_scan_consistency(simple_mesh, material_params):
    """Test that while_loop and scan Armijo line searches give same results."""
    problem, bc, iv = create_nonlinear_problem_and_bc(simple_mesh, material_params)
    initial = fe.zero_like_initial_guess(problem, bc)

    from feax.assembler import create_J_bc_function, create_res_bc_function
    from feax.solver import create_armijo_line_search_jax, create_armijo_line_search_scan
    from feax.linear_solver import create_linear_solve_fn

    J_bc_func = create_J_bc_function(problem, bc)
    res_bc_func = create_res_bc_function(problem, bc)

    solver_opts = fe.SolverOptions(linear_solver="cg")
    linear_fn = create_linear_solve_fn(solver_opts)

    # Get initial residual and Jacobian
    res = res_bc_func(initial, iv)
    J = J_bc_func(initial, iv)
    res_norm = jnp.linalg.norm(res)
    x0 = jnp.zeros_like(initial)
    delta_sol = linear_fn(J, -res, x0)

    # while_loop version
    armijo_while = create_armijo_line_search_jax(res_bc_func)
    sol_w, norm_w, alpha_w, ok_w = armijo_while(initial, delta_sol, res, res_norm, iv)

    # scan version
    armijo_scan = create_armijo_line_search_scan(res_bc_func)
    sol_s, norm_s, alpha_s, ok_s = armijo_scan(initial, delta_sol, res, res_norm, iv)

    # Both should find acceptable step
    assert ok_w, "while_loop Armijo failed"
    assert ok_s, "scan Armijo failed"

    # Results should be close (both find first acceptable alpha)
    diff = jnp.linalg.norm(sol_w - sol_s)
    assert diff < 1e-10, f"while vs scan Armijo differ by {diff:.2e}"


# ============================================================================
# create_linear_solver Tests
# ============================================================================

@pytest.mark.cpu
def test_create_linear_solver_jit(simple_mesh, material_params):
    """Test that create_linear_solver works with JIT."""
    problem, bc, iv = create_linear_problem_and_bc(simple_mesh, material_params)

    solver = fe.create_linear_solver(problem, bc)
    initial = fe.zero_like_initial_guess(problem, bc)

    sol_no_jit = solver(iv, initial)

    solver_jit = jax.jit(solver)
    sol_jit = solver_jit(iv, initial)

    diff = jnp.linalg.norm(sol_no_jit - sol_jit)
    assert diff < 1e-10, f"create_linear_solver JIT diff: {diff:.2e}"


@pytest.mark.cpu
def test_create_linear_solver_vmap(simple_mesh, material_params):
    """Test that create_linear_solver works with vmap."""
    problem, bc, iv = create_linear_problem_and_bc(simple_mesh, material_params)

    solver = fe.create_linear_solver(problem, bc)
    initial = fe.zero_like_initial_guess(problem, bc)

    batch_size = 3
    surf_var = iv.surface_vars[0][0]
    batch_surface_vars = jnp.stack([
        surf_var * (1.0 + 0.01 * i) for i in range(batch_size)
    ])

    def make_iv(sv):
        return fe.InternalVars((), [(sv,)])

    vmapped = jax.vmap(lambda sv: solver(make_iv(sv), initial))
    batch_solutions = vmapped(batch_surface_vars)

    assert batch_solutions.shape[0] == batch_size
    for i in range(batch_size):
        assert jnp.linalg.norm(batch_solutions[i]) > 0


@pytest.mark.cpu
def test_create_linear_solver_grad(simple_mesh, material_params):
    """Test that create_linear_solver supports grad."""
    problem, bc, iv = create_linear_problem_and_bc(simple_mesh, material_params)

    solver = fe.create_linear_solver(problem, bc)
    initial = fe.zero_like_initial_guess(problem, bc)

    surf_var = iv.surface_vars[0][0]

    def loss_fn(sv):
        iv_local = fe.InternalVars((), [(sv,)])
        sol = solver(iv_local, initial)
        return jnp.sum(sol**2)

    grad_val = jax.grad(loss_fn)(surf_var)
    assert jnp.all(jnp.isfinite(grad_val))
    assert jnp.linalg.norm(grad_val) > 0


@pytest.mark.cpu
def test_create_linear_solver_vs_create_solver_iter1(simple_mesh, material_params):
    """Test that create_linear_solver and create_solver(iter_num=1) give identical results."""
    problem, bc, iv = create_linear_problem_and_bc(simple_mesh, material_params)

    solver_opts = fe.SolverOptions(linear_solver="cg")
    solver1 = fe.create_linear_solver(problem, bc, solver_options=solver_opts)
    solver2 = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
    initial = fe.zero_like_initial_guess(problem, bc)

    sol1 = solver1(iv, initial)
    sol2 = solver2(iv, initial)

    diff = jnp.linalg.norm(sol1 - sol2)
    assert diff < 1e-10, f"linear_solver vs create_solver(1) differ by {diff:.2e}"
