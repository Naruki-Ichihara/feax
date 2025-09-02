"""Test linear elasticity problems."""
import pytest
import jax.numpy as np
from feax.mesh import Mesh
from feax.problem import Problem
from feax.internal_vars import InternalVars
from feax.DCboundary import DirichletBCSpec, DirichletBCConfig
from feax import create_solver


class LinearElasticityProblem(Problem):
    """Linear elasticity problem."""
    
    def get_tensor_map(self):
        def tensor_map(u_grad, E, nu):
            eps = 0.5 * (u_grad + u_grad.T)
            trace_eps = np.trace(eps)
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
            return 2. * mu * eps + lmbda * trace_eps * np.eye(self.dim)
        return tensor_map


@pytest.fixture  
def elastic_mesh():
    """Create 2D quad mesh for elasticity."""
    points = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    cells = np.array([[0, 1, 2, 3]])
    return Mesh(points, cells)


def test_elasticity_solve(elastic_mesh):
    """Test basic elasticity solve."""
    problem = LinearElasticityProblem(
        mesh=elastic_mesh,
        vec=2,
        dim=2, 
        ele_type='QUAD4'
    )
    
    # BC: fix left edge, apply displacement on right
    bc_config = DirichletBCConfig([
        DirichletBCSpec(lambda pt: np.isclose(pt[0], 0.0), 'all', 0.0),
        DirichletBCSpec(lambda pt: np.isclose(pt[0], 1.0), 'x', 0.1)
    ])
    bc = bc_config.create_bc(problem)
    
    # Material: E=210e9, nu=0.3  
    E = InternalVars.create_uniform_volume_var(problem, 210e9)
    nu = InternalVars.create_uniform_volume_var(problem, 0.3)
    internal_vars = InternalVars(volume_vars=(E, nu))
    
    solver = create_solver(problem, bc)
    
    initial_guess = np.zeros(problem.num_total_dofs_all_vars)
    solution = solver(internal_vars, initial_guess)
    
    assert solution.shape == (problem.num_total_dofs_all_vars,)
    assert not np.any(np.isnan(solution))


def test_elasticity_boundary_conditions(elastic_mesh):
    """Test elasticity BC enforcement."""
    problem = LinearElasticityProblem(
        mesh=elastic_mesh,
        vec=2,
        dim=2,
        ele_type='QUAD4'
    )
    
    bc_config = DirichletBCConfig([
        DirichletBCSpec(lambda pt: np.isclose(pt[0], 0.0), 'all', 0.0),
        DirichletBCSpec(lambda pt: np.isclose(pt[0], 1.0), 'x', 0.1)
    ])
    bc = bc_config.create_bc(problem)
    
    E = InternalVars.create_uniform_volume_var(problem, 210e9)
    nu = InternalVars.create_uniform_volume_var(problem, 0.3)
    internal_vars = InternalVars(volume_vars=(E, nu))
    
    solver = create_solver(problem, bc)
    solution = solver(internal_vars, np.zeros(problem.num_total_dofs_all_vars))
    
    sol_reshaped = solution.reshape(-1, 2)
    
    # Check left boundary (fixed)
    left_nodes = np.where(np.isclose(elastic_mesh.points[:, 0], 0.0))[0]
    assert np.allclose(sol_reshaped[left_nodes], 0.0, atol=1e-10)
    
    # Check right boundary x-displacement
    right_nodes = np.where(np.isclose(elastic_mesh.points[:, 0], 1.0))[0] 
    assert np.allclose(sol_reshaped[right_nodes, 0], 0.1, atol=1e-10)


def test_material_parameters():
    """Test material parameter creation."""
    points = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    cells = np.array([[0, 1, 2, 3]])
    mesh = Mesh(points, cells)
    
    problem = LinearElasticityProblem(mesh=mesh, vec=2, dim=2, ele_type='QUAD4')
    
    # Uniform material
    E_uniform = InternalVars.create_uniform_volume_var(problem, 200e9)
    assert E_uniform.shape == (problem.num_cells, problem.fes[0].num_quads)
    assert np.allclose(E_uniform, 200e9)
    
    # Spatially varying material
    E_varying = InternalVars.create_spatially_varying_volume_var(
        problem, lambda x: 200e9 + 50e9 * x[0]
    )
    assert E_varying.shape == (problem.num_cells, problem.fes[0].num_quads)