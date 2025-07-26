"""
Test suite for the Problem class using pytest.
Tests the modified problem.py file where onp has been replaced with np (jax.numpy).
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
    
    def custom_init(self):
        """Override custom_init for testing"""
        self.custom_init_called = True
        
    def get_tensor_map(self):
        """Define a simple tensor map for testing Laplace kernel"""
        def tensor_fn(u_grad, *args):
            # Simple identity mapping for testing
            return u_grad
        return tensor_fn
    
    def get_mass_map(self):
        """Define a simple mass map for testing mass kernel"""
        def mass_fn(u, x, *args):
            # Simple identity mapping for testing
            return u
        return mass_fn
    
    def get_universal_kernel(self):
        """Define a universal kernel for testing"""
        def universal_kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, 
                           cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
            # Simple kernel that returns zeros for testing
            return np.zeros_like(cell_sol_flat)
        return universal_kernel
    
    def get_surface_maps(self):
        """Define surface maps for testing"""
        def surface_fn(u, x, *args):
            return u
        return [surface_fn]  # Return list with one surface map
    
    def set_params(self, params):
        """Override set_params for testing parameter setting"""
        self.internal_vars = [params]
        self.params_set = True


@pytest.fixture
def simple_mesh():
    """Create a simple 3D mesh for testing"""
    # Create a simple 2x2x2 cube mesh with 1 element
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
    
    # Define hex element
    cells = onp.array([
        [0, 1, 2, 3, 4, 5, 6, 7]
    ])
    
    return Mesh(points, cells)


@pytest.fixture
def problem_instance(simple_mesh):
    """Create a TestProblem instance"""
    # Define boundary conditions
    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)
    
    def right(point):
        return np.isclose(point[0], 1., atol=1e-5)
    
    def dirichlet_val(point):
        return 0.
    
    location_fns = [left, right]
    value_fns = [dirichlet_val, dirichlet_val]
    vecs = [0, 0]  # Component 0 for both BCs
    dirichlet_bc_info = [location_fns, vecs, value_fns]
    
    # Create problem instance
    problem = TestProblem(
        mesh=simple_mesh,
        vec=1,
        dim=3,
        ele_type='HEX8',
        gauss_order=2,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=[left]  # For surface integrals
    )
    
    return problem


class TestProblemInitialization:
    """Test Problem class initialization"""
    
    def test_initialization(self, problem_instance):
        """Test that Problem initializes correctly"""
        assert problem_instance.num_vars == 1
        assert problem_instance.num_cells == 1
        assert problem_instance.num_total_dofs_all_vars == 8
        assert hasattr(problem_instance, 'custom_init_called')
        assert problem_instance.custom_init_called is True
    
    def test_arrays_are_jax_arrays(self, problem_instance):
        """Test that all arrays are JAX arrays"""
        assert isinstance(problem_instance.I, jax.Array)
        assert isinstance(problem_instance.J, jax.Array)
        assert isinstance(problem_instance.JxW, jax.Array)
        assert isinstance(problem_instance.shape_grads, jax.Array)
        assert isinstance(problem_instance.v_grads_JxW, jax.Array)


class TestKernelMethods:
    """Test kernel creation methods"""
    
    def test_laplace_kernel(self, problem_instance):
        """Test get_laplace_kernel method"""
        laplace_kernel = problem_instance.get_laplace_kernel(problem_instance.get_tensor_map())
        assert callable(laplace_kernel)
    
    def test_mass_kernel(self, problem_instance):
        """Test get_mass_kernel method"""
        mass_kernel = problem_instance.get_mass_kernel(problem_instance.get_mass_map())
        assert callable(mass_kernel)
    
    def test_surface_kernel(self, problem_instance):
        """Test get_surface_kernel method"""
        surface_kernel = problem_instance.get_surface_kernel(problem_instance.get_surface_maps()[0])
        assert callable(surface_kernel)


class TestComputationMethods:
    """Test computation methods"""
    
    @pytest.fixture
    def dummy_solution(self, problem_instance):
        """Create a dummy solution for testing"""
        return [np.zeros((problem_instance.fes[0].num_total_nodes, problem_instance.fes[0].vec))]
    
    def test_get_res(self, problem_instance, dummy_solution):
        """Test get_res method"""
        res_list = problem_instance.get_res(dummy_solution)
        assert len(res_list) == 1
        assert res_list[0].shape == (8, 1)
    
    def test_get_J(self, problem_instance, dummy_solution):
        """Test get_J method"""
        from jax.experimental import sparse
        J = problem_instance.get_J(dummy_solution)
        assert isinstance(J, sparse.BCOO)
        expected_shape = (problem_instance.num_total_dofs_all_vars, problem_instance.num_total_dofs_all_vars)
        assert J.shape == expected_shape
        assert len(J.data) > 0
    
    def test_set_params(self, problem_instance):
        """Test set_params method"""
        params = np.ones((problem_instance.num_cells, problem_instance.fes[0].num_quads))
        problem_instance.set_params(params)
        assert hasattr(problem_instance, 'params_set')
        assert problem_instance.params_set is True
        assert len(problem_instance.internal_vars) == 1


class TestHelperMethods:
    """Test helper methods"""
    
    def test_split_and_compute_cell(self, problem_instance):
        """Test split_and_compute_cell method"""
        cells_sol_list = [np.zeros((problem_instance.num_cells, 
                                   problem_instance.fes[0].num_nodes, 
                                   problem_instance.fes[0].vec))]
        cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list)
        
        weak_form = problem_instance.split_and_compute_cell(
            cells_sol_flat, np, False, problem_instance.internal_vars
        )
        assert weak_form is not None
        assert weak_form.shape == (1, 8)  # (num_cells, num_nodes*vec)
    
    def test_compute_face(self, problem_instance):
        """Test compute_face method"""
        cells_sol_list = [np.zeros((problem_instance.num_cells, 
                                   problem_instance.fes[0].num_nodes, 
                                   problem_instance.fes[0].vec))]
        cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list)
        
        face_values = problem_instance.compute_face(
            cells_sol_flat, np, False, problem_instance.internal_vars_surfaces
        )
        assert face_values is not None
        assert isinstance(face_values, list)


class TestJAXNumPyIntegration:
    """Test that JAX numpy is being used correctly"""
    
    def test_cumsum_with_jax_array(self, problem_instance):
        """Test that cumsum works with JAX arrays"""
        # This was a bug that was fixed - cumsum needs array input
        assert isinstance(problem_instance.num_nodes_cumsum, jax.Array)
    
    def test_array_operations_are_jax(self, problem_instance):
        """Test that array operations produce JAX arrays"""
        # Create some test data
        test_array = np.ones((2, 3))
        result = np.concatenate([test_array, test_array], axis=0)
        assert isinstance(result, jax.Array)
        
        stacked = np.stack([test_array, test_array])
        assert isinstance(stacked, jax.Array)
        
        transposed = np.transpose(stacked, axes=(1, 0, 2))
        assert isinstance(transposed, jax.Array)