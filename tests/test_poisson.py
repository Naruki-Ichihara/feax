"""Test Poisson equation problems."""
import pytest
import jax.numpy as np
import jax
from feax.mesh import Mesh
from feax.problem import Problem
from feax.internal_vars import InternalVars
from feax.DCboundary import DirichletBCSpec, DirichletBCConfig
from feax import create_solver


class PoissonProblem(Problem):
    """2D Poisson equation: -∇·(k∇u) = f"""
    
    def get_tensor_map(self):
        def tensor_map(u_grad, k):
            return k * u_grad
        return tensor_map


@pytest.fixture
def simple_mesh():
    """Create simple 2D quad mesh."""
    points = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    cells = np.array([[0, 1, 2, 3]])
    return Mesh(points, cells)


def test_poisson_solve(simple_mesh):
    """Test basic Poisson solve."""
    problem = PoissonProblem(
        mesh=simple_mesh,
        vec=1,
        dim=2,
        ele_type='QUAD4'
    )
    
    # BC: u=0 on left, u=1 on right
    bc_config = DirichletBCConfig([
        DirichletBCSpec(lambda pt: np.isclose(pt[0], 0.0), 0, 0.0),
        DirichletBCSpec(lambda pt: np.isclose(pt[0], 1.0), 0, 1.0)
    ])
    bc = bc_config.create_bc(problem)
    
    # Material: k=1
    k = InternalVars.create_uniform_volume_var(problem, 1.0)
    internal_vars = InternalVars(volume_vars=(k,))
    
    solver = create_solver(problem, bc)
    
    initial_guess = np.zeros(problem.num_total_dofs_all_vars)
    solution = solver(internal_vars, initial_guess)
    
    assert solution.shape == (problem.num_total_dofs_all_vars,)
    assert not np.any(np.isnan(solution))


def test_poisson_boundary_values(simple_mesh):
    """Test boundary condition enforcement."""
    problem = PoissonProblem(
        mesh=simple_mesh, 
        vec=1,
        dim=2,
        ele_type='QUAD4'
    )
    
    bc_config = DirichletBCConfig([
        DirichletBCSpec(lambda pt: np.isclose(pt[0], 0.0), 0, 0.0),
        DirichletBCSpec(lambda pt: np.isclose(pt[0], 1.0), 0, 1.0)
    ])
    bc = bc_config.create_bc(problem)
    
    k = InternalVars.create_uniform_volume_var(problem, 1.0)
    internal_vars = InternalVars(volume_vars=(k,))
    
    solver = create_solver(problem, bc)
    solution = solver(internal_vars, np.zeros(problem.num_total_dofs_all_vars))
    
    # Check BC enforcement
    left_nodes = np.where(np.isclose(simple_mesh.points[:, 0], 0.0))[0]
    right_nodes = np.where(np.isclose(simple_mesh.points[:, 0], 1.0))[0]
    
    assert np.allclose(solution[left_nodes], 0.0, atol=1e-10)
    assert np.allclose(solution[right_nodes], 1.0, atol=1e-10)