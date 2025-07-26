"""
Test to compare outputs between feax.problem and jax_fem.problem implementations.
This ensures that replacing onp with jax.numpy produces identical results.
"""

import pytest
import jax
import jax.numpy as np
import numpy as onp
import sys
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/jax-fem')

# Import both implementations
from feax.problem import Problem as FeaxProblem
from feax.mesh import Mesh as FeaxMesh

from jax_fem.problem import Problem as JaxFemProblem
from jax_fem.generate_mesh import Mesh as JaxFemMesh

# Set up JAX to use float64
jax.config.update("jax_enable_x64", True)


class ComparisonProblemFeax(FeaxProblem):
    """Test problem for feax implementation"""
    
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


class ComparisonProblemJaxFem(JaxFemProblem):
    """Test problem for jax_fem implementation"""
    
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
def mesh_data():
    """Create identical mesh data for both implementations"""
    # Create a simple cube mesh
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
    
    return points, cells


@pytest.fixture
def bc_data():
    """Create boundary condition data"""
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
    
    return dirichlet_bc_info, location_fns


@pytest.fixture
def problems(mesh_data, bc_data):
    """Create both problem instances with identical parameters"""
    points, cells = mesh_data
    dirichlet_bc_info, location_fns = bc_data
    
    # Create meshes
    feax_mesh = FeaxMesh(points, cells)
    jaxfem_mesh = JaxFemMesh(points, cells)
    
    # Create problems
    feax_problem = ComparisonProblemFeax(
        mesh=feax_mesh,
        vec=1,
        dim=3,
        ele_type='HEX8',
        gauss_order=2,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=[location_fns[0]]
    )
    
    jaxfem_problem = ComparisonProblemJaxFem(
        mesh=jaxfem_mesh,
        vec=1,
        dim=3,
        ele_type='HEX8',
        gauss_order=2,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=[location_fns[0]]
    )
    
    return feax_problem, jaxfem_problem


class TestArrayComparison:
    """Test that arrays are identical between implementations"""
    
    def test_index_arrays(self, problems):
        """Test I and J index arrays"""
        feax_problem, jaxfem_problem = problems
        
        # Convert to numpy for comparison
        feax_I = onp.array(feax_problem.I)
        jaxfem_I = onp.array(jaxfem_problem.I)
        
        feax_J = onp.array(feax_problem.J)
        jaxfem_J = onp.array(jaxfem_problem.J)
        
        onp.testing.assert_array_equal(feax_I, jaxfem_I, 
                                     err_msg="I arrays are not equal")
        onp.testing.assert_array_equal(feax_J, jaxfem_J,
                                     err_msg="J arrays are not equal")
    
    def test_shape_arrays(self, problems):
        """Test shape gradient arrays"""
        feax_problem, jaxfem_problem = problems
        
        feax_shape_grads = onp.array(feax_problem.shape_grads)
        jaxfem_shape_grads = onp.array(jaxfem_problem.shape_grads)
        
        onp.testing.assert_allclose(feax_shape_grads, jaxfem_shape_grads,
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="shape_grads arrays are not equal")
    
    def test_jacobian_arrays(self, problems):
        """Test Jacobian weighted arrays"""
        feax_problem, jaxfem_problem = problems
        
        feax_JxW = onp.array(feax_problem.JxW)
        jaxfem_JxW = onp.array(jaxfem_problem.JxW)
        
        onp.testing.assert_allclose(feax_JxW, jaxfem_JxW,
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="JxW arrays are not equal")
    
    def test_v_grads_arrays(self, problems):
        """Test v_grads_JxW arrays"""
        feax_problem, jaxfem_problem = problems
        
        feax_v_grads = onp.array(feax_problem.v_grads_JxW)
        jaxfem_v_grads = onp.array(jaxfem_problem.v_grads_JxW)
        
        onp.testing.assert_allclose(feax_v_grads, jaxfem_v_grads,
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="v_grads_JxW arrays are not equal")


class TestComputationMethods:
    """Test computation methods produce identical results"""
    
    @pytest.fixture
    def solution(self, problems):
        """Create identical solution for both problems"""
        feax_problem, _ = problems
        # Use a non-zero solution for more interesting test
        sol = onp.random.rand(feax_problem.fes[0].num_total_nodes, 
                            feax_problem.fes[0].vec)
        return [jax.numpy.array(sol)]
    
    def test_compute_residual(self, problems, solution):
        """Test that compute_residual produces identical results"""
        feax_problem, jaxfem_problem = problems
        
        feax_res = feax_problem.get_res(solution)
        jaxfem_res = jaxfem_problem.compute_residual(solution)
        
        # Convert to numpy for comparison
        feax_res_np = onp.array(feax_res[0])
        jaxfem_res_np = onp.array(jaxfem_res[0])
        print("Feax Residual:", feax_res_np)
        print("JaxFem Residual:", jaxfem_res_np)    
        
        onp.testing.assert_allclose(feax_res_np, jaxfem_res_np,
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="Residuals are not equal")
    
    def test_newton_update(self, problems, solution):
        """Test that newton_update produces identical results"""
        feax_problem, jaxfem_problem = problems
        
        # For feax: get residual and Jacobian separately
        feax_res = feax_problem.get_res(solution)
        feax_J = feax_problem.get_J(solution)
        
        # For jaxfem: use newton_update which computes both
        jaxfem_res = jaxfem_problem.newton_update(solution)
        
        # Compare residuals
        feax_res_np = onp.array(feax_res[0])
        jaxfem_res_np = onp.array(jaxfem_res[0])
        
        onp.testing.assert_allclose(feax_res_np, jaxfem_res_np,
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="Newton residuals are not equal")
        
        # Compare Jacobian matrices data
        feax_V = onp.array(feax_J.data)
        jaxfem_V = onp.array(jaxfem_problem.V)
        
        onp.testing.assert_allclose(feax_V, jaxfem_V,
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="Jacobian matrices V are not equal")


class TestKernelOutputs:
    """Test that kernel functions produce identical outputs"""
    
    def test_laplace_kernel_output(self, problems):
        """Test Laplace kernel outputs"""
        feax_problem, jaxfem_problem = problems
        
        # Create test data
        num_nodes = feax_problem.fes[0].num_nodes
        vec = feax_problem.fes[0].vec
        cell_sol_flat = onp.random.rand(num_nodes * vec)
        cell_shape_grads = feax_problem.shape_grads[0]
        cell_v_grads_JxW = feax_problem.v_grads_JxW[0]
        
        # Get kernels
        feax_kernel = feax_problem.get_laplace_kernel(feax_problem.get_tensor_map())
        jaxfem_kernel = jaxfem_problem.get_laplace_kernel(jaxfem_problem.get_tensor_map())
        
        # Compute outputs
        feax_out = feax_kernel(cell_sol_flat, cell_shape_grads, cell_v_grads_JxW)
        jaxfem_out = jaxfem_kernel(cell_sol_flat, cell_shape_grads, cell_v_grads_JxW)
        
        onp.testing.assert_allclose(onp.array(feax_out), onp.array(jaxfem_out),
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="Laplace kernel outputs are not equal")
    
    def test_mass_kernel_output(self, problems):
        """Test mass kernel outputs"""
        feax_problem, jaxfem_problem = problems
        
        # Create test data
        num_nodes = feax_problem.fes[0].num_nodes
        vec = feax_problem.fes[0].vec
        cell_sol_flat = onp.random.rand(num_nodes * vec)
        x = feax_problem.physical_quad_points[0]
        cell_JxW = feax_problem.JxW[0]
        
        # Get kernels
        feax_kernel = feax_problem.get_mass_kernel(feax_problem.get_mass_map())
        jaxfem_kernel = jaxfem_problem.get_mass_kernel(jaxfem_problem.get_mass_map())
        
        # Compute outputs
        feax_out = feax_kernel(cell_sol_flat, x, cell_JxW)
        jaxfem_out = jaxfem_kernel(cell_sol_flat, x, cell_JxW)
        
        onp.testing.assert_allclose(onp.array(feax_out), onp.array(jaxfem_out),
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="Mass kernel outputs are not equal")


class TestParameterHandling:
    """Test parameter setting and internal variables"""
    
    def test_internal_vars(self, problems):
        """Test that internal_vars work identically"""
        feax_problem, jaxfem_problem = problems
        
        # Create test parameters
        params = onp.random.rand(feax_problem.num_cells, feax_problem.fes[0].num_quads)
        params = jax.numpy.array(params)
        
        # Create new feax problem instance with parameters
        feax_problem_with_params = ComparisonProblemFeax(
            mesh=feax_problem.mesh,
            vec=feax_problem.vec, 
            dim=feax_problem.dim,
            ele_type=feax_problem.ele_type,
            gauss_order=feax_problem.gauss_order,
            dirichlet_bc_info=feax_problem.dirichlet_bc_info,
            location_fns=feax_problem.location_fns,
            additional_info=feax_problem.additional_info,
            internal_vars=(params,)
        )
        
        # Set parameters for jaxfem
        jaxfem_problem.set_params(params)
        
        # Compare internal vars
        feax_internal = onp.array(feax_problem_with_params.internal_vars[0])
        jaxfem_internal = onp.array(jaxfem_problem.internal_vars[0])
        
        onp.testing.assert_allclose(feax_internal, jaxfem_internal,
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="Internal vars are not equal")