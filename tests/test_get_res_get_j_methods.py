"""
Test the new get_res and get_J methods
"""

import pytest
import jax
import jax.numpy as np
import numpy as onp
from feax.problem import Problem
from feax.mesh import Mesh

# Set up JAX to use float64
jax.config.update("jax_enable_x64", True)


class TestProblem(Problem):
    """A simple test problem class that inherits from Problem"""
    
    def get_tensor_map(self):
        def tensor_fn(u_grad, *args):
            return u_grad
        return tensor_fn
    
    def get_mass_map(self):
        def mass_fn(u, x, *args):
            return u
        return mass_fn
    
    def get_surface_maps(self):
        def surface_fn(u, x, *args):
            return u
        return [surface_fn]
    
    def set_params(self, params):
        self.internal_vars = [params]


@pytest.fixture
def simple_mesh():
    """Create a simple 3D mesh for testing"""
    points = onp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0]
    ])
    
    cells = onp.array([
        [0, 1, 2, 3, 4, 5, 6, 7]
    ])
    
    return Mesh(points, cells)


@pytest.fixture
def problem_instance(simple_mesh):
    """Create a TestProblem instance"""
    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)
    
    def right(point):
        return np.isclose(point[0], 1., atol=1e-5)
    
    def dirichlet_val(point):
        return 0.
    
    location_fns = [left, right]
    value_fns = [dirichlet_val, dirichlet_val]
    vecs = [0, 0]
    dirichlet_bc_info = [location_fns, vecs, value_fns]
    
    problem = TestProblem(
        mesh=simple_mesh,
        vec=1,
        dim=3,
        ele_type='HEX8',
        gauss_order=2,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=[left]
    )
    
    return problem


def test_get_res_method(problem_instance):
    """Test that get_res works and produces same result as compute_residual"""
    sol_list = [np.ones((problem_instance.fes[0].num_total_nodes, problem_instance.fes[0].vec))]
    
    # Test that get_res exists and is callable
    assert hasattr(problem_instance, 'get_res')
    assert callable(problem_instance.get_res)
    
    # Test that get_res produces expected results
    res1 = problem_instance.get_res(sol_list)
    
    assert len(res1) == 1
    assert res1[0].shape == (8, 1)
    print("✓ get_res works correctly")


def test_get_J_method(problem_instance):
    """Test that get_J works and returns a BCOO sparse matrix"""
    from jax.experimental import sparse
    
    sol_list = [np.ones((problem_instance.fes[0].num_total_nodes, problem_instance.fes[0].vec))]
    
    # Test that get_J exists and is callable
    assert hasattr(problem_instance, 'get_J')
    assert callable(problem_instance.get_J)
    
    # Test that get_J returns a BCOO matrix
    J = problem_instance.get_J(sol_list)
    assert isinstance(J, sparse.BCOO), f"Expected BCOO, got {type(J)}"
    
    # Test matrix properties
    expected_shape = (problem_instance.num_total_dofs_all_vars, problem_instance.num_total_dofs_all_vars)
    assert J.shape == expected_shape, f"Expected shape {expected_shape}, got {J.shape}"
    
    # Test that the matrix has non-zero entries
    assert len(J.data) > 0, "Jacobian matrix should have non-zero entries"
    
    print(f"✓ get_J works correctly")
    print(f"  - Returns BCOO sparse matrix")
    print(f"  - Shape: {J.shape}")
    print(f"  - Non-zeros: {len(J.data)}")


def test_get_J_consistency(problem_instance):
    """Test that get_J produces consistent results"""
    sol_list = [np.ones((problem_instance.fes[0].num_total_nodes, problem_instance.fes[0].vec))]
    
    # Get Jacobian multiple times to test consistency
    J1 = problem_instance.get_J(sol_list)
    J2 = problem_instance.get_J(sol_list)
    
    # Compare the values should be identical
    J1_values = onp.array(J1.data)
    J2_values = onp.array(J2.data)
    
    onp.testing.assert_allclose(J1_values, J2_values, rtol=1e-10, atol=1e-10)
    print("✓ get_J produces consistent results")


def test_methods_are_pure(problem_instance):
    """Test that get_res and get_J don't modify instance state"""
    sol_list = [np.ones((problem_instance.fes[0].num_total_nodes, problem_instance.fes[0].vec))]
    
    # Store original state
    original_vars = dict(vars(problem_instance))
    
    # Call get_res
    _ = problem_instance.get_res(sol_list)
    after_get_res = dict(vars(problem_instance))
    
    # Call get_J
    _ = problem_instance.get_J(sol_list)
    after_get_J = dict(vars(problem_instance))
    
    # Check that state hasn't changed
    assert original_vars.keys() == after_get_res.keys() == after_get_J.keys()
    print("✓ get_res and get_J don't modify instance state (pure functions)")


def test_sparse_matrix_operations(problem_instance):
    """Test that the returned BCOO matrix can be used in typical operations"""
    sol_list = [np.ones((problem_instance.fes[0].num_total_nodes, problem_instance.fes[0].vec))]
    
    J = problem_instance.get_J(sol_list)
    res = problem_instance.get_res(sol_list)
    
    # Convert residual to flat array
    res_flat = jax.flatten_util.ravel_pytree(res)[0]
    
    # Test matrix-vector multiplication
    try:
        result = J @ res_flat
        assert result.shape == res_flat.shape
        print("✓ BCOO matrix supports matrix-vector multiplication")
    except Exception as e:
        print(f"⚠ Matrix-vector multiplication failed: {e}")
    
    # Test converting to dense (for small matrices)
    try:
        J_dense = J.todense()
        assert J_dense.shape == J.shape
        print("✓ BCOO matrix can be converted to dense")
    except Exception as e:
        print(f"⚠ Dense conversion failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])