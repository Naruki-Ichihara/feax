"""
Pytest configuration and shared fixtures for feax tests.
"""

import pytest
import jax
import jax.numpy as jnp
import feax as fe
from feax.problem import MatrixView


@pytest.fixture(autouse=True)
def clear_jax_cache():
    """Clear JAX compilation cache between tests to avoid CUDA memory issues."""
    yield
    # Clear cache after each test
    jax.clear_caches()


@pytest.fixture
def simple_mesh():
    """Create a simple small mesh for testing."""
    L, W, H = 10, 10, 10
    mesh_size = 5  # Very coarse mesh for fast tests
    return fe.mesh.box_mesh((L, W, H), mesh_size=mesh_size)


@pytest.fixture
def material_params():
    """Material parameters for linear elasticity."""
    return {
        'E': 70e3,      # Young's modulus
        'nu': 0.3,      # Poisson's ratio
        'traction': 1e-3,
        'tol': 1e-5
    }


@pytest.fixture
def boundary_condition(material_params):
    """Boundary condition function for reuse."""
    tol = material_params['tol']
    return lambda p: jnp.isclose(p[0], 10., tol)


@pytest.fixture(scope="function")
def linear_elasticity_problem(simple_mesh, material_params, boundary_condition):
    """Create a simple linear elasticity problem (FULL matrix view)."""
    E, nu = material_params['E'], material_params['nu']

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

    return LinearElasticity(
        simple_mesh, vec=3, dim=3,
        location_fns=[boundary_condition],
        matrix_view=MatrixView.FULL
    )


@pytest.fixture(scope="function")
def linear_elasticity_problem_upper(simple_mesh, material_params, boundary_condition):
    """Create a linear elasticity problem with UPPER matrix view."""
    E, nu = material_params['E'], material_params['nu']

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

    return LinearElasticity(
        simple_mesh, vec=3, dim=3,
        location_fns=[boundary_condition],
        matrix_view=MatrixView.UPPER
    )


@pytest.fixture(scope="function")
def linear_elasticity_problem_lower(simple_mesh, material_params, boundary_condition):
    """Create a linear elasticity problem with LOWER matrix view."""
    E, nu = material_params['E'], material_params['nu']

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

    return LinearElasticity(
        simple_mesh, vec=3, dim=3,
        location_fns=[boundary_condition],
        matrix_view=MatrixView.LOWER
    )


@pytest.fixture
def internal_vars(simple_mesh, boundary_condition, material_params):
    """Create internal variables independent of matrix view (shared for all problems)."""
    traction = material_params['traction']
    # Create a temporary problem just for generating internal vars (like in the example)
    temp_problem = fe.Problem(simple_mesh, vec=3, dim=3, location_fns=[boundary_condition])
    surf_var = fe.InternalVars.create_uniform_surface_var(temp_problem, traction)
    return fe.InternalVars((), [(surf_var,)])
