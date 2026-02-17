"""
Tests for vmap (vectorization) compatibility with solvers.

This module tests that solvers work correctly with JAX vmap transformation:
- JAX iterative solvers (CG, BICGSTAB, GMRES) with vmap
- cuDSS direct solver with vmap
- vmap composition with JIT
- vmap composition with grad
- vmap+jit+grad composition
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
# vmap Tests - JAX Iterative Solvers
# ============================================================================

@pytest.mark.cpu
def test_cg_solver_vmap_compatibility(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that CG solver works with vmap."""
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

    # Create batch of internal_vars (3 copies with small perturbations)
    batch_size = 3
    surface_var = internal_vars.surface_vars[0][0]

    # Create batch with small perturbations
    batch_surface_vars = jnp.stack([
        surface_var * (1.0 + 0.01 * i) for i in range(batch_size)
    ])

    # Reconstruct batched internal_vars
    def make_internal_vars(surf_var):
        return fe.InternalVars((), [(surf_var,)])

    # Apply vmap over batch dimension
    vmapped_solver = jax.vmap(lambda sv: solver(make_internal_vars(sv), initial))
    batch_solutions = vmapped_solver(batch_surface_vars)

    # Check solutions
    assert batch_solutions.shape[0] == batch_size
    for i in range(batch_size):
        sol_norm = jnp.linalg.norm(batch_solutions[i])
        assert sol_norm > 0, f"Solution {i} is trivial"
        assert sol_norm < 1.0, f"Solution {i} norm too large: {sol_norm}"


@pytest.mark.cpu
def test_bicgstab_solver_vmap_compatibility(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that BICGSTAB solver works with vmap."""
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

    # Create batch of internal_vars
    batch_size = 3
    surface_var = internal_vars.surface_vars[0][0]
    batch_surface_vars = jnp.stack([
        surface_var * (1.0 + 0.01 * i) for i in range(batch_size)
    ])

    def make_internal_vars(surf_var):
        return fe.InternalVars((), [(surf_var,)])
    # Apply vmap
    vmapped_solver = jax.vmap(lambda sv: solver(make_internal_vars(sv), initial))
    batch_solutions = vmapped_solver(batch_surface_vars)

    # Check solutions
    assert batch_solutions.shape[0] == batch_size
    for i in range(batch_size):
        sol_norm = jnp.linalg.norm(batch_solutions[i])
        assert sol_norm > 0, f"Solution {i} is trivial"
        assert sol_norm < 1.0, f"Solution {i} norm too large: {sol_norm}"


@pytest.mark.cpu
def test_gmres_solver_vmap_compatibility(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that GMRES solver works with vmap."""
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

    # Create batch of internal_vars
    batch_size = 3
    surface_var = internal_vars.surface_vars[0][0]
    batch_surface_vars = jnp.stack([
        surface_var * (1.0 + 0.01 * i) for i in range(batch_size)
    ])

    def make_internal_vars(surf_var):
        return fe.InternalVars((), [(surf_var,)])
    # Apply vmap
    vmapped_solver = jax.vmap(lambda sv: solver(make_internal_vars(sv), initial))
    batch_solutions = vmapped_solver(batch_surface_vars)

    # Check solutions
    assert batch_solutions.shape[0] == batch_size
    for i in range(batch_size):
        sol_norm = jnp.linalg.norm(batch_solutions[i])
        assert sol_norm > 0, f"Solution {i} is trivial"
        assert sol_norm < 1.0, f"Solution {i} norm too large: {sol_norm}"


# ============================================================================
# vmap Tests - cuDSS Solver
# ============================================================================

@pytest.mark.cuda
@requires_cudss
def test_cudss_solver_vmap_compatibility(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that cuDSS solver works with vmap."""
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

    # Create batch of internal_vars
    batch_size = 3
    surface_var = internal_vars.surface_vars[0][0]
    batch_surface_vars = jnp.stack([
        surface_var * (1.0 + 0.01 * i) for i in range(batch_size)
    ])

    def make_internal_vars(surf_var):
        return fe.InternalVars((), [(surf_var,)])
    # Apply vmap
    vmapped_solver = jax.vmap(lambda sv: solver(make_internal_vars(sv), initial))
    batch_solutions = vmapped_solver(batch_surface_vars)

    # Check solutions
    assert batch_solutions.shape[0] == batch_size
    for i in range(batch_size):
        sol_norm = jnp.linalg.norm(batch_solutions[i])
        assert sol_norm > 0, f"Solution {i} is trivial"
        assert sol_norm < 1.0, f"Solution {i} norm too large: {sol_norm}"


# ============================================================================
# vmap+JIT Composition Tests
# ============================================================================

@pytest.mark.cpu
def test_vmap_jit_composition_cg(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that vmap and JIT can be composed with CG solver."""
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

    # Create batch of internal_vars
    batch_size = 3
    surface_var = internal_vars.surface_vars[0][0]
    batch_surface_vars = jnp.stack([
        surface_var * (1.0 + 0.01 * i) for i in range(batch_size)
    ])

    def make_internal_vars(surf_var):
        return fe.InternalVars((), [(surf_var,)])
    # Test both composition orders
    # 1. jax.jit(jax.vmap(...))
    vmapped_solver_1 = jax.jit(jax.vmap(lambda sv: solver(make_internal_vars(sv), initial)))
    batch_sol_1 = vmapped_solver_1(batch_surface_vars)

    # 2. jax.vmap(jax.jit(...))
    vmapped_solver_2 = jax.vmap(jax.jit(lambda sv: solver(make_internal_vars(sv), initial)))
    batch_sol_2 = vmapped_solver_2(batch_surface_vars)

    # Solutions should be similar
    diff = jnp.linalg.norm(batch_sol_1 - batch_sol_2)
    assert diff < 1e-10, f"Different vmap+JIT compositions produce different results: {diff:.2e}"


@pytest.mark.cuda
@requires_cudss
def test_vmap_jit_composition_cudss(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that vmap and JIT can be composed with cuDSS solver."""
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

    # Create batch of internal_vars
    batch_size = 3
    surface_var = internal_vars.surface_vars[0][0]
    batch_surface_vars = jnp.stack([
        surface_var * (1.0 + 0.01 * i) for i in range(batch_size)
    ])

    def make_internal_vars(surf_var):
        return fe.InternalVars((), [(surf_var,)])
    # Test jax.jit(jax.vmap(...))
    vmapped_solver = jax.jit(jax.vmap(lambda sv: solver(make_internal_vars(sv), initial)))
    batch_solutions = vmapped_solver(batch_surface_vars)

    # Check solutions
    assert batch_solutions.shape[0] == batch_size
    for i in range(batch_size):
        sol_norm = jnp.linalg.norm(batch_solutions[i])
        assert sol_norm > 0, f"Solution {i} is trivial"


# ============================================================================
# vmap+grad Composition Tests
# ============================================================================

@pytest.mark.cpu
def test_vmap_grad_composition_cg(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that vmap and grad can be composed with CG solver."""
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

    # Create batch of internal_vars
    batch_size = 3
    surface_var = internal_vars.surface_vars[0][0]
    batch_surface_vars = jnp.stack([
        surface_var * (1.0 + 0.01 * i) for i in range(batch_size)
    ])

    def make_internal_vars(surf_var):
        return fe.InternalVars((), [(surf_var,)])
    def loss_fn(surf_var):
        sol = solver(make_internal_vars(surf_var), initial)
        return jnp.linalg.norm(sol)

    # Compute gradients for batch using vmap(grad(...))
    grad_fn = jax.vmap(jax.grad(loss_fn))
    batch_grads = grad_fn(batch_surface_vars)

    # Check gradients
    assert batch_grads.shape[0] == batch_size
    for i in range(batch_size):
        assert jnp.all(jnp.isfinite(batch_grads[i])), f"Gradient {i} contains inf or nan"
        grad_norm = jnp.linalg.norm(batch_grads[i])
        assert grad_norm > 0, f"Gradient {i} is trivial"


@pytest.mark.cuda
@requires_cudss
def test_vmap_grad_composition_cudss(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that vmap and grad can be composed with cuDSS solver."""
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

    # Create batch of internal_vars
    batch_size = 3
    surface_var = internal_vars.surface_vars[0][0]
    batch_surface_vars = jnp.stack([
        surface_var * (1.0 + 0.01 * i) for i in range(batch_size)
    ])

    def make_internal_vars(surf_var):
        return fe.InternalVars((), [(surf_var,)])
    def loss_fn(surf_var):
        sol = solver(make_internal_vars(surf_var), initial)
        return jnp.linalg.norm(sol)

    # Compute gradients for batch using vmap(grad(...))
    grad_fn = jax.vmap(jax.grad(loss_fn))
    batch_grads = grad_fn(batch_surface_vars)

    # Check gradients
    assert batch_grads.shape[0] == batch_size
    for i in range(batch_size):
        assert jnp.all(jnp.isfinite(batch_grads[i])), f"Gradient {i} contains inf or nan"
        grad_norm = jnp.linalg.norm(batch_grads[i])
        assert grad_norm > 0, f"Gradient {i} is trivial"


# ============================================================================
# vmap+jit+grad Composition Tests
# ============================================================================

@pytest.mark.cpu
def test_vmap_jit_grad_composition_cg(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that vmap, JIT, and grad can all be composed with CG solver."""
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

    # Create batch of internal_vars
    batch_size = 3
    surface_var = internal_vars.surface_vars[0][0]
    batch_surface_vars = jnp.stack([
        surface_var * (1.0 + 0.01 * i) for i in range(batch_size)
    ])

    def make_internal_vars(surf_var):
        return fe.InternalVars((), [(surf_var,)])
    def loss_fn(surf_var):
        sol = solver(make_internal_vars(surf_var), initial)
        return jnp.linalg.norm(sol)

    # Compose all three: jax.jit(jax.vmap(jax.grad(...)))
    grad_fn = jax.jit(jax.vmap(jax.grad(loss_fn)))
    batch_grads = grad_fn(batch_surface_vars)

    # Check gradients
    assert batch_grads.shape[0] == batch_size
    for i in range(batch_size):
        assert jnp.all(jnp.isfinite(batch_grads[i])), f"Gradient {i} contains inf or nan"
        grad_norm = jnp.linalg.norm(batch_grads[i])
        assert grad_norm > 0, f"Gradient {i} is trivial"


@pytest.mark.cuda
@requires_cudss
def test_vmap_jit_grad_composition_cudss(
    linear_elasticity_problem,
    internal_vars,
    material_params
):
    """Test that vmap, JIT, and grad can all be composed with cuDSS solver."""
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

    # Create batch of internal_vars
    batch_size = 3
    surface_var = internal_vars.surface_vars[0][0]
    batch_surface_vars = jnp.stack([
        surface_var * (1.0 + 0.01 * i) for i in range(batch_size)
    ])

    def make_internal_vars(surf_var):
        return fe.InternalVars((), [(surf_var,)])
    def loss_fn(surf_var):
        sol = solver(make_internal_vars(surf_var), initial)
        return jnp.linalg.norm(sol)

    # Compose all three: jax.jit(jax.vmap(jax.grad(...)))
    grad_fn = jax.jit(jax.vmap(jax.grad(loss_fn)))
    batch_grads = grad_fn(batch_surface_vars)

    # Check gradients
    assert batch_grads.shape[0] == batch_size
    for i in range(batch_size):
        assert jnp.all(jnp.isfinite(batch_grads[i])), f"Gradient {i} contains inf or nan"
        grad_norm = jnp.linalg.norm(batch_grads[i])
        assert grad_norm > 0, f"Gradient {i} is trivial"
