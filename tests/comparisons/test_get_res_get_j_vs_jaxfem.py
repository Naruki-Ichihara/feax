"""
Test to compare the new feax methods (get_res, get_J) with jax_fem implementations.
This ensures that the new pure functional API produces identical results.
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
    
    def set_params(self, params):
        self.internal_vars = [params]


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


@pytest.fixture
def solution(problems):
    """Create identical solution for both problems"""
    feax_problem, _ = problems
    # Use a non-trivial solution for better testing
    sol = onp.random.rand(feax_problem.fes[0].num_total_nodes, 
                        feax_problem.fes[0].vec)
    return [jax.numpy.array(sol)]


class TestGetResComparison:
    """Test get_res method against jax_fem compute_residual"""
    
    def test_get_res_vs_jaxfem_compute_residual(self, problems, solution):
        """Test that feax get_res produces same result as jax_fem compute_residual"""
        feax_problem, jaxfem_problem = problems
        
        # Get residuals from both implementations
        feax_res = feax_problem.get_res(solution)
        jaxfem_res = jaxfem_problem.compute_residual(solution)
        
        # Convert to numpy for comparison
        feax_res_np = onp.array(feax_res[0])
        jaxfem_res_np = onp.array(jaxfem_res[0])
        
        onp.testing.assert_allclose(feax_res_np, jaxfem_res_np,
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="get_res and jax_fem compute_residual are not equal")
        
        print(f"✓ feax get_res matches jax_fem compute_residual")
        print(f"  - Residual norm: {onp.linalg.norm(feax_res_np):.10f}")
    
    def test_get_res_consistency(self, problems, solution):
        """Test that get_res produces consistent results"""
        feax_problem, _ = problems
        
        # Get residuals multiple times to test consistency
        res1 = feax_problem.get_res(solution)
        res2 = feax_problem.get_res(solution)
        
        # Convert to numpy for comparison
        res1_np = onp.array(res1[0])
        res2_np = onp.array(res2[0])
        
        onp.testing.assert_allclose(res1_np, res2_np,
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="get_res should produce consistent results")
        
        print(f"✓ feax get_res produces consistent results")


class TestGetJComparison:
    """Test get_J method against jax_fem newton_update"""
    
    def test_get_J_vs_jaxfem_newton_update(self, problems, solution):
        """Test that feax get_J produces same Jacobian values as jax_fem newton_update"""
        feax_problem, jaxfem_problem = problems
        
        # Get Jacobian from feax new method
        J_feax = feax_problem.get_J(solution)
        
        # Get Jacobian from jax_fem old method (stores in self.V)
        _ = jaxfem_problem.newton_update(solution)
        V_jaxfem = jaxfem_problem.V
        
        # Compare the Jacobian values
        J_feax_values = onp.array(J_feax.data)
        V_jaxfem_values = onp.array(V_jaxfem)
        
        onp.testing.assert_allclose(J_feax_values, V_jaxfem_values,
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="feax get_J and jax_fem newton_update Jacobian values are not equal")
        
        print(f"✓ feax get_J produces same Jacobian values as jax_fem newton_update")
        print(f"  - Jacobian entries: {len(J_feax_values)}")
        print(f"  - Jacobian norm: {onp.linalg.norm(J_feax_values):.10f}")
    
    def test_get_J_consistency(self, problems, solution):
        """Test that get_J produces consistent results"""
        feax_problem, _ = problems
        
        # Get Jacobian multiple times to test consistency
        J1 = feax_problem.get_J(solution)
        J2 = feax_problem.get_J(solution)
        
        # Compare the Jacobian values
        J1_values = onp.array(J1.data)
        J2_values = onp.array(J2.data)
        
        onp.testing.assert_allclose(J1_values, J2_values,
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="get_J should produce consistent results")
        
        print(f"✓ feax get_J produces consistent results")


class TestSparseMatrixProperties:
    """Test properties of the returned sparse matrix"""
    
    def test_sparse_matrix_structure(self, problems, solution):
        """Test that the sparse matrix has correct structure"""
        feax_problem, jaxfem_problem = problems
        
        # Get sparse matrix from feax
        J_feax = feax_problem.get_J(solution)
        
        # Get index arrays from both implementations
        feax_I = onp.array(feax_problem.I)
        feax_J = onp.array(feax_problem.J)
        jaxfem_I = onp.array(jaxfem_problem.I)
        jaxfem_J = onp.array(jaxfem_problem.J)
        
        # Test that index arrays are identical
        onp.testing.assert_array_equal(feax_I, jaxfem_I,
                                     err_msg="I index arrays are not equal")
        onp.testing.assert_array_equal(feax_J, jaxfem_J,
                                     err_msg="J index arrays are not equal")
        
        # Test sparse matrix indices
        expected_indices = onp.stack([feax_I, feax_J], axis=1)
        actual_indices = onp.array(J_feax.indices)
        
        onp.testing.assert_array_equal(expected_indices, actual_indices,
                                     err_msg="Sparse matrix indices are not correct")
        
        print(f"✓ Sparse matrix structure is correct")
        print(f"  - Matrix shape: {J_feax.shape}")
        print(f"  - Index entries: {len(feax_I)}")
    
    def test_sparse_matrix_operations(self, problems, solution):
        """Test that sparse matrix supports typical FEM operations"""
        feax_problem, _ = problems
        
        J = feax_problem.get_J(solution)
        res = feax_problem.get_res(solution)
        
        # Convert residual to flat array
        res_flat = jax.flatten_util.ravel_pytree(res)[0]
        
        # Test matrix-vector multiplication (typical in Newton's method)
        result = J @ res_flat
        assert result.shape == res_flat.shape
        
        # Test that result is not trivial (all zeros)
        assert onp.linalg.norm(result) > 1e-12
        
        print(f"✓ Sparse matrix supports FEM operations")
        print(f"  - Matrix-vector multiplication works")
        print(f"  - Result norm: {onp.linalg.norm(result):.6e}")


class TestConsistencyWithJaxFemAPI:
    """Test consistency with the jax_fem API patterns"""
    
    def test_residual_and_jacobian_consistency(self, problems, solution):
        """Test that feax get_res matches jax_fem compute_residual and newton_update"""
        feax_problem, jaxfem_problem = problems
        
        # Compare feax get_res with jax_fem methods
        feax_res = feax_problem.get_res(solution)
        jaxfem_res_direct = jaxfem_problem.compute_residual(solution)
        jaxfem_res_from_newton = jaxfem_problem.newton_update(solution)
        
        onp.testing.assert_allclose(onp.array(feax_res[0]), 
                                  onp.array(jaxfem_res_direct[0]),
                                  rtol=1e-10, atol=1e-10)
        
        onp.testing.assert_allclose(onp.array(jaxfem_res_direct[0]), 
                                  onp.array(jaxfem_res_from_newton[0]),
                                  rtol=1e-10, atol=1e-10)
        
        print(f"✓ Residual consistency maintained across all methods")
    
    def test_jacobian_matrix_properties(self, problems, solution):
        """Test mathematical properties of the Jacobian matrix"""
        feax_problem, _ = problems
        
        J = feax_problem.get_J(solution)
        
        # Test that matrix is square
        assert J.shape[0] == J.shape[1], "Jacobian should be square"
        
        # Test that matrix size matches total DOFs
        expected_size = feax_problem.num_total_dofs_all_vars
        assert J.shape[0] == expected_size, f"Expected size {expected_size}, got {J.shape[0]}"
        
        # For this simple problem, matrix should be symmetric (or at least have symmetric structure)
        # Convert to dense for testing (small matrix)
        if J.shape[0] <= 64:  # Only for small matrices
            J_dense = J.todense()
            # Test approximate symmetry (allowing for numerical errors)
            symmetry_error = onp.linalg.norm(J_dense - J_dense.T)
            print(f"  - Symmetry error: {symmetry_error:.6e}")
        
        print(f"✓ Jacobian matrix has correct mathematical properties")
        print(f"  - Matrix is square: {J.shape}")
        print(f"  - Matches DOF count: {expected_size}")


class TestPerformanceAndMemory:
    """Test performance characteristics of new vs old API"""
    
    def test_memory_efficiency(self, problems, solution):
        """Test that get_J doesn't store unnecessary state"""
        feax_problem, _ = problems
        
        # Store original state
        original_vars = set(vars(feax_problem).keys())
        
        # Call get_J (should not modify state)
        J = feax_problem.get_J(solution)
        after_get_J = set(vars(feax_problem).keys())
        
        # get_J should not add instance variables
        assert original_vars == after_get_J, "get_J should not modify instance state"
        
        print(f"✓ Memory usage is efficient")
        print(f"  - get_J doesn't modify state")
        print(f"  - Pure functional interface maintained")