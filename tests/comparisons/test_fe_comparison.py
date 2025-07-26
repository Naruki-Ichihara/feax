"""
Test to compare outputs between feax.fe and jax_fem.fe implementations.
This ensures that replacing onp with jax.numpy in fe.py produces identical results.
"""

import pytest
import jax
import jax.numpy as np
import numpy as onp
import sys
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/jax-fem')

# Import both implementations
from feax.fe import FiniteElement as FeaxFiniteElement
from feax.mesh import Mesh as FeaxMesh

from feax.fe import FiniteElement as FeaxFiniteElement
from feax.mesh import Mesh as FeaxMesh

# Set up JAX to use float64
jax.config.update("jax_enable_x64", True)


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
def finite_elements(mesh_data, bc_data):
    """Create both FiniteElement instances with identical parameters"""
    points, cells = mesh_data
    dirichlet_bc_info, location_fns = bc_data
    
    # Create meshes
    feax_mesh = FeaxMesh(points, cells)
    jaxfem_mesh = JaxFemMesh(points, cells)
    
    # Create finite elements
    feax_fe = FeaxFiniteElement(
        mesh=feax_mesh,
        vec=1,
        dim=3,
        ele_type='HEX8',
        gauss_order=2,
        dirichlet_bc_info=dirichlet_bc_info
    )
    
    jaxfem_fe = JaxFemFiniteElement(
        mesh=jaxfem_mesh,
        vec=1,
        dim=3,
        ele_type='HEX8',
        gauss_order=2,
        dirichlet_bc_info=dirichlet_bc_info
    )
    
    return feax_fe, jaxfem_fe


class TestBasicProperties:
    """Test that basic properties are identical"""
    
    def test_dimensions(self, finite_elements):
        """Test that dimensions and counts are identical"""
        feax_fe, jaxfem_fe = finite_elements
        
        assert feax_fe.num_nodes == jaxfem_fe.num_nodes
        assert feax_fe.num_cells == jaxfem_fe.num_cells
        assert feax_fe.num_quads == jaxfem_fe.num_quads
        assert feax_fe.num_total_nodes == jaxfem_fe.num_total_nodes
        assert feax_fe.num_total_dofs == jaxfem_fe.num_total_dofs
        assert feax_fe.vec == jaxfem_fe.vec
        assert feax_fe.dim == jaxfem_fe.dim
    
    def test_mesh_data(self, finite_elements):
        """Test that mesh data is identical"""
        feax_fe, jaxfem_fe = finite_elements
        
        onp.testing.assert_array_equal(feax_fe.points, jaxfem_fe.points,
                                     err_msg="Points arrays are not equal")
        onp.testing.assert_array_equal(feax_fe.cells, jaxfem_fe.cells,
                                     err_msg="Cells arrays are not equal")


class TestShapeFunctions:
    """Test that shape function computations are identical"""
    
    def test_shape_values(self, finite_elements):
        """Test shape function values"""
        feax_fe, jaxfem_fe = finite_elements
        
        feax_vals = onp.array(feax_fe.shape_vals)
        jaxfem_vals = onp.array(jaxfem_fe.shape_vals)
        
        onp.testing.assert_allclose(feax_vals, jaxfem_vals,
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="Shape values are not equal")
    
    def test_shape_gradients_ref(self, finite_elements):
        """Test reference shape gradients"""
        feax_fe, jaxfem_fe = finite_elements
        
        feax_grads = onp.array(feax_fe.shape_grads_ref)
        jaxfem_grads = onp.array(jaxfem_fe.shape_grads_ref)
        
        onp.testing.assert_allclose(feax_grads, jaxfem_grads,
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="Reference shape gradients are not equal")
    
    def test_shape_gradients_physical(self, finite_elements):
        """Test physical shape gradients"""
        feax_fe, jaxfem_fe = finite_elements
        
        feax_grads = onp.array(feax_fe.shape_grads)
        jaxfem_grads = onp.array(jaxfem_fe.shape_grads)
        
        onp.testing.assert_allclose(feax_grads, jaxfem_grads,
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="Physical shape gradients are not equal")
    
    def test_v_grads_JxW(self, finite_elements):
        """Test v_grads_JxW arrays"""
        feax_fe, jaxfem_fe = finite_elements
        
        feax_v_grads = onp.array(feax_fe.v_grads_JxW)
        jaxfem_v_grads = onp.array(jaxfem_fe.v_grads_JxW)
        
        onp.testing.assert_allclose(feax_v_grads, jaxfem_v_grads,
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="v_grads_JxW arrays are not equal")


class TestJacobianWeights:
    """Test Jacobian and weight computations"""
    
    def test_JxW(self, finite_elements):
        """Test Jacobian times weights"""
        feax_fe, jaxfem_fe = finite_elements
        
        feax_JxW = onp.array(feax_fe.JxW)
        jaxfem_JxW = onp.array(jaxfem_fe.JxW)
        
        onp.testing.assert_allclose(feax_JxW, jaxfem_JxW,
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="JxW arrays are not equal")


class TestFaceOperations:
    """Test face-related computations"""
    
    def test_face_shape_values(self, finite_elements):
        """Test face shape function values"""
        feax_fe, jaxfem_fe = finite_elements
        
        feax_face_vals = onp.array(feax_fe.face_shape_vals)
        jaxfem_face_vals = onp.array(jaxfem_fe.face_shape_vals)
        
        onp.testing.assert_allclose(feax_face_vals, jaxfem_face_vals,
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="Face shape values are not equal")
    
    def test_face_shape_grads_ref(self, finite_elements):
        """Test face reference shape gradients"""
        feax_fe, jaxfem_fe = finite_elements
        
        feax_face_grads = onp.array(feax_fe.face_shape_grads_ref)
        jaxfem_face_grads = onp.array(jaxfem_fe.face_shape_grads_ref)
        
        onp.testing.assert_allclose(feax_face_grads, jaxfem_face_grads,
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="Face reference shape gradients are not equal")
    
    def test_face_weights(self, finite_elements):
        """Test face weights"""
        feax_fe, jaxfem_fe = finite_elements
        
        # Use the correct attribute name
        feax_face_weights = onp.array(feax_fe.quad_weights)
        jaxfem_face_weights = onp.array(jaxfem_fe.quad_weights)
        
        onp.testing.assert_allclose(feax_face_weights, jaxfem_face_weights,
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="Face weights are not equal")


class TestBoundaryConditions:
    """Test boundary condition handling"""
    
    def test_dirichlet_bc_indices(self, finite_elements):
        """Test Dirichlet boundary condition indices"""
        feax_fe, jaxfem_fe = finite_elements
        
        # Compare node indices
        for i in range(len(feax_fe.node_inds_list)):
            feax_inds = onp.array(feax_fe.node_inds_list[i])
            jaxfem_inds = onp.array(jaxfem_fe.node_inds_list[i])
            
            onp.testing.assert_array_equal(feax_inds, jaxfem_inds,
                                         err_msg=f"Node indices {i} are not equal")
        
        # Compare vector indices
        for i in range(len(feax_fe.vec_inds_list)):
            feax_vec_inds = onp.array(feax_fe.vec_inds_list[i])
            jaxfem_vec_inds = onp.array(jaxfem_fe.vec_inds_list[i])
            
            onp.testing.assert_array_equal(feax_vec_inds, jaxfem_vec_inds,
                                         err_msg=f"Vector indices {i} are not equal")


class TestPhysicalCoordinates:
    """Test physical coordinate computations"""
    
    def test_physical_quad_points(self, finite_elements):
        """Test physical quadrature points"""
        feax_fe, jaxfem_fe = finite_elements
        
        feax_quad_points = feax_fe.get_physical_quad_points()
        jaxfem_quad_points = jaxfem_fe.get_physical_quad_points()
        
        onp.testing.assert_allclose(onp.array(feax_quad_points), 
                                  onp.array(jaxfem_quad_points),
                                  rtol=1e-10, atol=1e-10,
                                  err_msg="Physical quadrature points are not equal")


class TestBoundaryIdentification:
    """Test boundary identification methods"""
    
    def test_boundary_conditions_inds(self, finite_elements):
        """Test boundary condition index identification"""
        feax_fe, jaxfem_fe = finite_elements
        
        # Define test location functions
        def left(point):
            return np.isclose(point[0], 0., atol=1e-5)
        
        location_fns = [left]
        
        # Get boundary indices from both implementations
        feax_boundary_inds = feax_fe.get_boundary_conditions_inds(location_fns)
        jaxfem_boundary_inds = jaxfem_fe.get_boundary_conditions_inds(location_fns)
        
        # Compare results
        for i in range(len(feax_boundary_inds)):
            feax_inds = onp.array(feax_boundary_inds[i])
            jaxfem_inds = onp.array(jaxfem_boundary_inds[i])
            
            onp.testing.assert_array_equal(feax_inds, jaxfem_inds,
                                         err_msg=f"Boundary indices {i} are not equal")


class TestArrayTypes:
    """Test that arrays are of correct types"""
    
    def test_jax_array_types(self, finite_elements):
        """Test that feax operations produce JAX arrays"""
        feax_fe, _ = finite_elements
        
        # Test that feax fe.py operations use JAX arrays
        # Note: shape_vals comes from basis.py which may still use numpy arrays
        # But computations in fe.py should produce JAX arrays
        assert isinstance(feax_fe.shape_grads, jax.Array), "shape_grads should be JAX array"
        assert isinstance(feax_fe.JxW, jax.Array), "JxW should be JAX array"
        assert isinstance(feax_fe.v_grads_JxW, jax.Array), "v_grads_JxW should be JAX array"
        
        # Test a method that should return JAX array
        quad_points = feax_fe.get_physical_quad_points()
        assert isinstance(quad_points, jax.Array), "Physical quad points should be JAX array"


class TestSpecialMethods:
    """Test special computation methods"""
    
    def test_face_shape_grads_computation(self, finite_elements):
        """Test get_face_shape_grads method"""
        feax_fe, jaxfem_fe = finite_elements
        
        # Get boundary conditions first
        def left(point):
            return np.isclose(point[0], 0., atol=1e-5)
        
        location_fns = [left]
        feax_boundary_inds = feax_fe.get_boundary_conditions_inds(location_fns)
        jaxfem_boundary_inds = jaxfem_fe.get_boundary_conditions_inds(location_fns)
        
        if len(feax_boundary_inds) > 0 and len(feax_boundary_inds[0]) > 0:
            # Get face shape gradients
            feax_grads, feax_nanson = feax_fe.get_face_shape_grads(feax_boundary_inds[0])
            jaxfem_grads, jaxfem_nanson = jaxfem_fe.get_face_shape_grads(jaxfem_boundary_inds[0])
            
            onp.testing.assert_allclose(onp.array(feax_grads), 
                                      onp.array(jaxfem_grads),
                                      rtol=1e-10, atol=1e-10,
                                      err_msg="Face shape gradients are not equal")
            
            onp.testing.assert_allclose(onp.array(feax_nanson), 
                                      onp.array(jaxfem_nanson),
                                      rtol=1e-10, atol=1e-10,
                                      err_msg="Nanson scale factors are not equal")
    
    def test_physical_surface_quad_points(self, finite_elements):
        """Test get_physical_surface_quad_points method"""
        feax_fe, jaxfem_fe = finite_elements
        
        # Get boundary conditions first
        def left(point):
            return np.isclose(point[0], 0., atol=1e-5)
        
        location_fns = [left]
        feax_boundary_inds = feax_fe.get_boundary_conditions_inds(location_fns)
        jaxfem_boundary_inds = jaxfem_fe.get_boundary_conditions_inds(location_fns)
        
        if len(feax_boundary_inds) > 0 and len(feax_boundary_inds[0]) > 0:
            # Get physical surface quadrature points
            feax_surf_points = feax_fe.get_physical_surface_quad_points(feax_boundary_inds[0])
            jaxfem_surf_points = jaxfem_fe.get_physical_surface_quad_points(jaxfem_boundary_inds[0])
            
            onp.testing.assert_allclose(onp.array(feax_surf_points), 
                                      onp.array(jaxfem_surf_points),
                                      rtol=1e-10, atol=1e-10,
                                      err_msg="Physical surface quad points are not equal")