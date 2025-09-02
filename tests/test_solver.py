"""Test solver functionality."""
import pytest
import jax.numpy as np
from feax.mesh import Mesh
from feax.problem import Problem
from feax.internal_vars import InternalVars
from feax.DCboundary import DirichletBCSpec, DirichletBCConfig
from feax import create_solver, SolverOptions


class SolverTestProblem(Problem):
    """Simple test problem."""
    
    def get_tensor_map(self):
        def tensor_map(u_grad, coeff):
            return coeff * u_grad
        return tensor_map


@pytest.fixture
def solver_mesh():
    """Simple mesh for solver tests."""
    points = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    cells = np.array([[0, 1, 2, 3]])
    return Mesh(points, cells)


def test_solver_creation(solver_mesh):
    """Test solver creation."""
    problem = SolverTestProblem(mesh=solver_mesh, vec=1, dim=2, ele_type='QUAD4')
    
    bc_config = DirichletBCConfig([
        DirichletBCSpec(lambda pt: np.isclose(pt[0], 0.0), 0, 0.0)
    ])
    bc = bc_config.create_bc(problem)
    
    solver = create_solver(problem, bc)
    
    assert callable(solver)


def test_solver_execution(solver_mesh):
    """Test solver execution."""
    problem = SolverTestProblem(mesh=solver_mesh, vec=1, dim=2, ele_type='QUAD4')
    
    bc_config = DirichletBCConfig([
        DirichletBCSpec(lambda pt: np.isclose(pt[0], 0.0), 0, 0.0),
        DirichletBCSpec(lambda pt: np.isclose(pt[0], 1.0), 0, 1.0)
    ])
    bc = bc_config.create_bc(problem)
    
    coeff = InternalVars.create_uniform_volume_var(problem, 1.0)
    internal_vars = InternalVars(volume_vars=(coeff,))
    
    solver = create_solver(problem, bc)
    
    initial_guess = np.zeros(problem.num_total_dofs_all_vars)
    solution = solver(internal_vars, initial_guess)
    
    assert solution.shape == (problem.num_total_dofs_all_vars,)
    assert np.all(np.isfinite(solution))


def test_solver_options(solver_mesh):
    """Test solver with options."""
    problem = SolverTestProblem(mesh=solver_mesh, vec=1, dim=2, ele_type='QUAD4')
    
    bc_config = DirichletBCConfig([
        DirichletBCSpec(lambda pt: np.isclose(pt[0], 0.0), 0, 0.0)
    ])
    bc = bc_config.create_bc(problem)
    
    options = SolverOptions(
        linear_solver='bicgstab',
        tol=1e-8,
        max_iter=100
    )
    
    solver = create_solver(problem, bc, solver_options=options)
    
    coeff = InternalVars.create_uniform_volume_var(problem, 1.0)
    internal_vars = InternalVars(volume_vars=(coeff,))
    
    solution = solver(internal_vars, np.zeros(problem.num_total_dofs_all_vars))
    
    assert np.all(np.isfinite(solution))


def test_solver_convergence(solver_mesh):
    """Test solver convergence properties."""
    problem = SolverTestProblem(mesh=solver_mesh, vec=1, dim=2, ele_type='QUAD4')
    
    bc_config = DirichletBCConfig([
        DirichletBCSpec(lambda pt: np.isclose(pt[0], 0.0), 0, 0.0),
        DirichletBCSpec(lambda pt: np.isclose(pt[0], 1.0), 0, 1.0)
    ])
    bc = bc_config.create_bc(problem)
    
    coeff = InternalVars.create_uniform_volume_var(problem, 1.0)
    internal_vars = InternalVars(volume_vars=(coeff,))
    
    solver = create_solver(problem, bc)
    
    # Test with different initial guesses
    solution1 = solver(internal_vars, np.zeros(problem.num_total_dofs_all_vars))
    solution2 = solver(internal_vars, np.ones(problem.num_total_dofs_all_vars))
    
    # Should converge to same solution regardless of initial guess
    assert np.allclose(solution1, solution2, rtol=1e-10)


def test_internal_vars_variations(solver_mesh):
    """Test solver with different internal variables."""
    problem = SolverTestProblem(mesh=solver_mesh, vec=1, dim=2, ele_type='QUAD4')
    
    bc_config = DirichletBCConfig([
        DirichletBCSpec(lambda pt: np.isclose(pt[0], 0.0), 0, 0.0),
        DirichletBCSpec(lambda pt: np.isclose(pt[0], 1.0), 0, 1.0)
    ])
    bc = bc_config.create_bc(problem)
    
    solver = create_solver(problem, bc)
    
    # Test with different coefficients
    coeff1 = InternalVars.create_uniform_volume_var(problem, 1.0)
    coeff2 = InternalVars.create_uniform_volume_var(problem, 2.0)
    
    internal_vars1 = InternalVars(volume_vars=(coeff1,))
    internal_vars2 = InternalVars(volume_vars=(coeff2,))
    
    solution1 = solver(internal_vars1, np.zeros(problem.num_total_dofs_all_vars))
    solution2 = solver(internal_vars2, np.zeros(problem.num_total_dofs_all_vars))
    
    # Solutions should be different for different coefficients (but BCs might force similarity)
    # Just ensure both solve without error
    assert np.all(np.isfinite(solution1))
    assert np.all(np.isfinite(solution2))