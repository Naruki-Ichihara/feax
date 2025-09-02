"""Test boundary condition functionality."""
import pytest
import jax.numpy as np
from feax.mesh import Mesh
from feax.DCboundary import DirichletBCSpec, DirichletBCConfig, DirichletBC
from feax.problem import Problem


class DummyProblem(Problem):
    def get_tensor_map(self):
        return lambda u_grad: u_grad


@pytest.fixture
def test_mesh():
    """Simple 2D quad mesh."""
    points = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    cells = np.array([[0, 1, 2, 3]])
    return Mesh(points, cells)


def test_bc_spec_validation():
    """Test BC specification validation."""
    # Valid component strings
    bc1 = DirichletBCSpec(lambda pt: True, 'x', 0.0)
    assert bc1.component == 0
    
    bc2 = DirichletBCSpec(lambda pt: True, 'y', 0.0)  
    assert bc2.component == 1
    
    bc3 = DirichletBCSpec(lambda pt: True, 'all', 0.0)
    assert bc3.component == 'all'
    
    # Invalid component string
    with pytest.raises(ValueError):
        DirichletBCSpec(lambda pt: True, 'invalid', 0.0)
    
    # Negative component
    with pytest.raises(ValueError):
        DirichletBCSpec(lambda pt: True, -1, 0.0)


def test_bc_creation(test_mesh):
    """Test boundary condition creation."""
    problem = DummyProblem(mesh=test_mesh, vec=2, dim=2, ele_type='QUAD4')
    
    bc_config = DirichletBCConfig([
        DirichletBCSpec(lambda pt: np.isclose(pt[0], 0.0), 'x', 0.0),
        DirichletBCSpec(lambda pt: np.isclose(pt[0], 1.0), 'y', 1.0)
    ])
    
    bc = bc_config.create_bc(problem)
    
    assert isinstance(bc, DirichletBC)
    assert bc.total_dofs == problem.num_total_dofs_all_vars
    assert len(bc.bc_rows) > 0
    assert len(bc.bc_vals) == len(bc.bc_rows)


def test_bc_from_specs(test_mesh):
    """Test DirichletBC.from_specs factory method."""
    problem = DummyProblem(mesh=test_mesh, vec=2, dim=2, ele_type='QUAD4')
    
    specs = [
        DirichletBCSpec(lambda pt: np.isclose(pt[0], 0.0), 'all', 0.0),
        DirichletBCSpec(lambda pt: np.isclose(pt[1], 1.0), 'x', 0.5)
    ]
    
    bc = DirichletBC.from_specs(problem, specs)
    
    assert isinstance(bc, DirichletBC)
    assert bc.total_dofs == problem.num_total_dofs_all_vars


def test_empty_bc(test_mesh):
    """Test empty boundary conditions."""
    problem = DummyProblem(mesh=test_mesh, vec=2, dim=2, ele_type='QUAD4')
    
    bc_config = DirichletBCConfig([])
    bc = bc_config.create_bc(problem)
    
    assert len(bc.bc_rows) == 0
    assert len(bc.bc_vals) == 0
    assert bc.total_dofs == problem.num_total_dofs_all_vars


def test_bc_fluent_interface(test_mesh):
    """Test fluent BC configuration interface."""
    problem = DummyProblem(mesh=test_mesh, vec=2, dim=2, ele_type='QUAD4')
    
    bc_config = DirichletBCConfig()
    bc_config.add(lambda pt: np.isclose(pt[0], 0.0), 'x', 0.0)
    bc_config.add(lambda pt: np.isclose(pt[1], 0.0), 'y', 0.0)
    
    bc = bc_config.create_bc(problem)
    
    assert len(bc.bc_rows) > 0


def test_spatially_varying_bc(test_mesh):
    """Test spatially varying boundary conditions."""
    problem = DummyProblem(mesh=test_mesh, vec=2, dim=2, ele_type='QUAD4')
    
    bc_config = DirichletBCConfig([
        DirichletBCSpec(
            lambda pt: np.isclose(pt[0], 1.0),
            'x', 
            lambda pt: 0.1 * pt[1]  # Varies with y-coordinate
        )
    ])
    
    bc = bc_config.create_bc(problem)
    
    assert len(bc.bc_rows) > 0
    assert not np.allclose(bc.bc_vals, bc.bc_vals[0])  # Values should vary