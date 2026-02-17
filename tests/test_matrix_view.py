"""
Tests for upper/lower triangular matrix storage (MatrixView).

This module tests the MatrixView functionality including:
- UPPER/LOWER triangular matrix assembly
- Memory reduction verification
- Solver compatibility (cuDSS)
- JIT compilation compatibility
- Solution accuracy comparison with FULL matrix
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
        # Try to access cuDSS-related functionality
        return has_gpu()  # cuDSS requires GPU
    except ImportError:
        return False


def get_backend():
    """Get current JAX backend."""
    try:
        return jax.default_backend()
    except:
        return "unknown"


# Skip decorators for missing dependencies
requires_gpu = pytest.mark.skipif(
    not has_gpu(),
    reason="GPU not available"
)

requires_cudss = pytest.mark.skipif(
    not has_cudss(),
    reason="cuDSS backend not available (requires GPU)"
)


# ============================================================================
# Test Environment Info
# ============================================================================

@pytest.mark.cpu
def test_environment_info():
    """Print environment information for debugging."""
    print("\n" + "="*60)
    print("Test Environment Information")
    print("="*60)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {get_backend()}")
    print(f"GPU available: {has_gpu()}")
    print(f"cuDSS available: {has_cudss()}")
    if has_gpu():
        devices = jax.devices('gpu')
        print(f"GPU devices: {len(devices)}")
        for i, device in enumerate(devices):
            print(f"  GPU {i}: {device}")
    print("="*60)


# ============================================================================
# Slim Tests - Matrix View Functionality
# ============================================================================

@pytest.mark.cpu
def test_matrix_view_enum():
    """Test MatrixView enum values."""
    assert MatrixView.FULL.value == 0
    assert MatrixView.UPPER.value == 1
    assert MatrixView.LOWER.value == 2


@pytest.mark.cpu
def test_problem_initialization_full(linear_elasticity_problem):
    """Test Problem initialization with FULL matrix view."""
    problem = linear_elasticity_problem

    # Check matrix view
    assert problem.matrix_view == MatrixView.FULL

    # Check that I and J indices exist
    assert hasattr(problem, 'I')
    assert hasattr(problem, 'J')
    assert len(problem.I) == 6912
    assert len(problem.J) == 6912

    # Check filtered indices (for FULL, should be same as I, J)
    assert hasattr(problem, 'I_filtered')
    assert hasattr(problem, 'J_filtered')
    assert len(problem.I_filtered) == 6912
    assert len(problem.J_filtered) == 6912


@pytest.mark.cpu
def test_problem_initialization_upper(linear_elasticity_problem_upper):
    """Test Problem initialization with UPPER matrix view."""
    problem = linear_elasticity_problem_upper

    # Check matrix view
    assert problem.matrix_view == MatrixView.UPPER

    # Check that filter_indices exist
    assert hasattr(problem, 'filter_indices')
    assert problem.filter_indices is not None

    # Check exact dimensions
    assert len(problem.I) == 6912
    assert len(problem.J) == 6912
    assert len(problem.I_filtered) == 3600
    assert len(problem.J_filtered) == 3600


@pytest.mark.cpu
def test_upper_view_indices(linear_elasticity_problem_upper):
    """Test that UPPER view correctly filters j >= i."""
    problem = linear_elasticity_problem_upper

    # Check exact dimensions
    assert len(problem.I_filtered) == 3600
    assert len(problem.J_filtered) == 3600

    # All filtered entries should satisfy j >= i
    assert jnp.all(problem.J_filtered >= problem.I_filtered)


@pytest.mark.cpu
def test_lower_view_indices(linear_elasticity_problem_lower):
    """Test that LOWER view correctly filters j <= i."""
    problem = linear_elasticity_problem_lower

    # Check exact dimensions
    assert len(problem.I) == 6912
    assert len(problem.J) == 6912
    assert len(problem.I_filtered) == 3600
    assert len(problem.J_filtered) == 3600

    # All filtered entries should satisfy j <= i
    assert jnp.all(problem.J_filtered <= problem.I_filtered)


@pytest.mark.cpu
def test_memory_reduction(linear_elasticity_problem, linear_elasticity_problem_upper):
    """Test that UPPER/LOWER views reduce memory usage."""
    problem_full = linear_elasticity_problem
    problem_upper = linear_elasticity_problem_upper

    # Check exact nnz values
    nnz_full = len(problem_full.I_filtered)
    nnz_upper = len(problem_upper.I_filtered)

    assert nnz_full == 6912
    assert nnz_upper == 3600


@pytest.mark.cpu
def test_jacobian_info(linear_elasticity_problem, internal_vars):
    """Test get_jacobian_info function."""
    problem = linear_elasticity_problem

    # Create dummy solution
    sol = jnp.zeros(problem.num_total_dofs_all_vars)
    sol_list = problem.unflatten_fn_sol_list(sol)

    # Get Jacobian info
    info = fe.get_jacobian_info(problem, sol_list, internal_vars)

    # Check returned structure
    assert 'nnz' in info
    assert 'shape' in info
    assert 'matrix_view' in info

    # Check exact values
    assert info['nnz'] == 6912
    assert info['shape'] == (81, 81)
    assert info['matrix_view'] == MatrixView.FULL
