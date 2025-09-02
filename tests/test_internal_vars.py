"""Test InternalVars functionality."""
import pytest
import jax.numpy as np
from feax.mesh import Mesh
from feax.problem import Problem
from feax.internal_vars import InternalVars


class DummyProblem(Problem):
    def get_tensor_map(self):
        return lambda u_grad, k: k * u_grad


@pytest.fixture
def test_mesh():
    """Simple 2D quad mesh."""
    points = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    cells = np.array([[0, 1, 2, 3]])
    return Mesh(points, cells)


@pytest.fixture  
def test_problem(test_mesh):
    """Test problem for InternalVars testing."""
    return DummyProblem(mesh=test_mesh, vec=1, dim=2, ele_type='QUAD4')


def test_uniform_volume_var(test_problem):
    """Test uniform volume variable creation."""
    var = InternalVars.create_uniform_volume_var(test_problem, 5.0)
    
    expected_shape = (test_problem.num_cells, test_problem.fes[0].num_quads)
    assert var.shape == expected_shape
    assert np.allclose(var, 5.0)


def test_spatially_varying_volume_var(test_problem):
    """Test spatially varying volume variable."""
    var = InternalVars.create_spatially_varying_volume_var(
        test_problem, lambda x: 1.0 + x[0]
    )
    
    expected_shape = (test_problem.num_cells, test_problem.fes[0].num_quads)
    assert var.shape == expected_shape
    assert not np.allclose(var, var[0, 0])  # Should vary spatially


def test_internal_vars_creation():
    """Test InternalVars object creation."""
    var1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    var2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    
    internal_vars = InternalVars(
        volume_vars=(var1, var2),
        surface_vars=[((np.array([1.0]),),)]
    )
    
    assert len(internal_vars.volume_vars) == 2
    assert len(internal_vars.surface_vars) == 1
    assert np.array_equal(internal_vars.volume_vars[0], var1)
    assert np.array_equal(internal_vars.volume_vars[1], var2)


def test_replace_volume_var(test_problem):
    """Test replacing volume variables."""
    var1 = InternalVars.create_uniform_volume_var(test_problem, 1.0)
    var2 = InternalVars.create_uniform_volume_var(test_problem, 2.0)
    var3 = InternalVars.create_uniform_volume_var(test_problem, 3.0)
    
    internal_vars = InternalVars(volume_vars=(var1, var2))
    
    # Replace second variable
    new_internal_vars = internal_vars.replace_volume_var(1, var3)
    
    assert np.allclose(new_internal_vars.volume_vars[0], 1.0)
    assert np.allclose(new_internal_vars.volume_vars[1], 3.0)
    # Original should be unchanged
    assert np.allclose(internal_vars.volume_vars[1], 2.0)


def test_empty_internal_vars():
    """Test empty InternalVars creation."""
    internal_vars = InternalVars()
    
    assert len(internal_vars.volume_vars) == 0
    assert len(internal_vars.surface_vars) == 0