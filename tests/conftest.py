"""
Pytest configuration and fixtures for feax tests.

Provides common test fixtures and configuration.
"""

import pytest
import numpy as onp
import jax.numpy as np
from feax.generate_mesh import Mesh


@pytest.fixture
def simple_tet_mesh():
    """Fixture providing a simple tetrahedral mesh."""
    points = onp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    cells = onp.array([[0, 1, 2, 3]])
    
    return Mesh(points, cells, 'TET4')


@pytest.fixture
def simple_hex_mesh():
    """Fixture providing a simple hexahedral mesh."""
    points = onp.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0]   # 7
    ])
    cells = onp.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    
    return Mesh(points, cells, 'HEX8')


@pytest.fixture
def simple_quad_mesh():
    """Fixture providing a simple quadrilateral mesh."""
    points = onp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    cells = onp.array([[0, 1, 2, 3]])
    
    return Mesh(points, cells, 'QUAD4')


@pytest.fixture
def simple_tri_mesh():
    """Fixture providing a simple triangular mesh."""
    points = onp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    cells = onp.array([[0, 1, 2]])
    
    return Mesh(points, cells, 'TRI3')


@pytest.fixture
def material_properties():
    """Fixture providing common material properties."""
    return {
        'E': 200e9,  # Young's modulus [Pa]
        'nu': 0.3,   # Poisson's ratio
        'rho': 7850  # Density [kg/m³]
    }


@pytest.fixture
def boundary_condition_functions():
    """Fixture providing common boundary condition functions."""
    return {
        'fixed_x0': lambda x: onp.isclose(x[0], 0.0),
        'fixed_y0': lambda x: onp.isclose(x[1], 0.0),
        'fixed_z0': lambda x: onp.isclose(x[2], 0.0),
        'zero_value': lambda x: 0.0,
        'unit_value': lambda x: 1.0,
        'linear_x': lambda x: x[0],
        'traction_x': lambda x: onp.array([1000.0, 0.0, 0.0])
    }


@pytest.fixture
def sample_solution_tet():
    """Fixture providing sample solution for tetrahedral mesh."""
    return onp.array([
        [0.0, 0.0, 0.0],    # node 0
        [0.1, 0.0, 0.0],    # node 1
        [0.0, 0.1, 0.0],    # node 2
        [0.0, 0.0, 0.1]     # node 3
    ])


@pytest.fixture
def sample_solution_hex():
    """Fixture providing sample solution for hexahedral mesh."""
    return onp.array([
        [0.0, 0.0, 0.0],    # node 0
        [0.1, 0.0, 0.0],    # node 1
        [0.1, 0.1, 0.0],    # node 2
        [0.0, 0.1, 0.0],    # node 3
        [0.0, 0.0, 0.1],    # node 4
        [0.1, 0.0, 0.1],    # node 5
        [0.1, 0.1, 0.1],    # node 6
        [0.0, 0.1, 0.1]     # node 7
    ])


# Configure pytest
def pytest_configure(config):
    """Configure pytest settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Set up numerical tolerances
@pytest.fixture
def tolerances():
    """Fixture providing numerical tolerances for testing."""
    return {
        'rtol': 1e-10,
        'atol': 1e-12,
        'loose_rtol': 1e-6,
        'loose_atol': 1e-8
    }


# JAX configuration for testing
@pytest.fixture(scope="session", autouse=True)
def configure_jax():
    """Configure JAX for testing."""
    import jax
    # Set JAX to use float64 for better numerical precision in tests
    jax.config.update("jax_enable_x64", True)
    # Disable JIT compilation for easier debugging
    jax.config.update("jax_disable_jit", True)
    
    # Configure JAX to allow array conversion for testing
    import os
    os.environ["JAX_ENABLE_X64"] = "True"
    os.environ["JAX_DISABLE_JIT"] = "True"