"""
Test suite for feax.basis module.

Tests element type mappings, shape functions, and gradients.
"""

import pytest
import numpy as onp
import jax.numpy as np
from feax.basis import (
    get_elements, 
    get_shape_vals_and_grads, 
    get_face_shape_vals_and_grads,
    reorder_inds
)


class TestGetElements:
    """Test the get_elements function for various element types."""
    
    def test_hex8_elements(self):
        """Test HEX8 element configuration."""
        element_family, basix_ele, basix_face_ele, gauss_order, degree, re_order = get_elements('HEX8')
        
        assert degree == 1
        assert gauss_order == 2
        assert len(re_order) == 8
        assert re_order == [0, 1, 3, 2, 4, 5, 7, 6]
        
    def test_hex20_elements(self):
        """Test HEX20 element configuration."""
        element_family, basix_ele, basix_face_ele, gauss_order, degree, re_order = get_elements('HEX20')
        
        assert degree == 2
        assert gauss_order == 2
        assert len(re_order) == 20
        assert re_order == [0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 15, 14]
    
    def test_tet4_elements(self):
        """Test TET4 element configuration."""
        element_family, basix_ele, basix_face_ele, gauss_order, degree, re_order = get_elements('TET4')
        
        assert degree == 1
        assert gauss_order == 0
        assert len(re_order) == 4
        assert re_order == [0, 1, 2, 3]
    
    def test_tet10_elements(self):
        """Test TET10 element configuration."""
        element_family, basix_ele, basix_face_ele, gauss_order, degree, re_order = get_elements('TET10')
        
        assert degree == 2
        assert gauss_order == 2
        assert len(re_order) == 10
        assert re_order == [0, 1, 2, 3, 9, 6, 8, 7, 5, 4]
        
    def test_quad4_elements(self):
        """Test QUAD4 element configuration."""
        element_family, basix_ele, basix_face_ele, gauss_order, degree, re_order = get_elements('QUAD4')
        
        assert degree == 1
        assert gauss_order == 2
        assert len(re_order) == 4
        assert re_order == [0, 1, 3, 2]
    
    def test_tri3_elements(self):
        """Test TRI3 element configuration."""
        element_family, basix_ele, basix_face_ele, gauss_order, degree, re_order = get_elements('TRI3')
        
        assert degree == 1
        assert gauss_order == 0
        assert len(re_order) == 3
        assert re_order == [0, 1, 2]
    
    def test_invalid_element_type(self):
        """Test that invalid element types raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            get_elements('INVALID_TYPE')


class TestReorderInds:
    """Test the reorder_inds function."""
    
    def test_reorder_simple(self):
        """Test simple reordering of indices."""
        inds = onp.array([0, 1, 2, 3])
        re_order = onp.array([0, 1, 3, 2])  # Swap 2 and 3
        
        result = reorder_inds(inds, re_order)
        expected = onp.array([0, 1, 3, 2])
        
        onp.testing.assert_array_equal(result, expected)
    
    def test_reorder_2d_array(self):
        """Test reordering of 2D array."""
        inds = onp.array([[0, 1], [2, 3]])
        re_order = onp.array([0, 1, 3, 2])
        
        result = reorder_inds(inds, re_order)
        expected = onp.array([[0, 1], [3, 2]])
        
        onp.testing.assert_array_equal(result, expected)


class TestShapeValsAndGrads:
    """Test shape function values and gradients."""
    
    def test_hex8_shape_vals_and_grads(self):
        """Test HEX8 shape function values and gradients."""
        shape_vals, shape_grads_ref, weights = get_shape_vals_and_grads('HEX8')
        
        # Check shapes
        assert shape_vals.shape == (8, 8)  # 8 quad points, 8 nodes
        assert shape_grads_ref.shape == (8, 8, 3)  # 8 quad points, 8 nodes, 3D
        assert weights.shape == (8,)  # 8 quad points
        
        # Check that shape functions sum to 1 at each quadrature point
        shape_sum = onp.sum(shape_vals, axis=1)
        onp.testing.assert_allclose(shape_sum, onp.ones(8), rtol=1e-10)
        
        # Check weights are positive
        assert onp.all(weights > 0)
    
    def test_tet4_shape_vals_and_grads(self):
        """Test TET4 shape function values and gradients."""
        shape_vals, shape_grads_ref, weights = get_shape_vals_and_grads('TET4')
        
        # Check shapes
        assert shape_vals.shape == (1, 4)  # 1 quad point, 4 nodes
        assert shape_grads_ref.shape == (1, 4, 3)  # 1 quad point, 4 nodes, 3D
        assert weights.shape == (1,)  # 1 quad point
        
        # Check that shape functions sum to 1
        shape_sum = onp.sum(shape_vals, axis=1)
        onp.testing.assert_allclose(shape_sum, onp.ones(1), rtol=1e-10)
    
    def test_quad4_shape_vals_and_grads(self):
        """Test QUAD4 shape function values and gradients."""
        shape_vals, shape_grads_ref, weights = get_shape_vals_and_grads('QUAD4')
        
        # Check shapes
        assert shape_vals.shape == (4, 4)  # 4 quad points, 4 nodes
        assert shape_grads_ref.shape == (4, 4, 2)  # 4 quad points, 4 nodes, 2D
        assert weights.shape == (4,)  # 4 quad points
        
        # Check that shape functions sum to 1
        shape_sum = onp.sum(shape_vals, axis=1)
        onp.testing.assert_allclose(shape_sum, onp.ones(4), rtol=1e-10)
    
    def test_custom_gauss_order(self):
        """Test custom Gauss order specification."""
        # Test with different gauss orders
        shape_vals1, _, weights1 = get_shape_vals_and_grads('HEX8', gauss_order=1)
        shape_vals2, _, weights2 = get_shape_vals_and_grads('HEX8', gauss_order=3)
        
        # Different gauss orders should give different number of quadrature points
        assert shape_vals1.shape[0] != shape_vals2.shape[0]
        assert len(weights1) != len(weights2)


class TestFaceShapeValsAndGrads:
    """Test face shape function values and gradients."""
    
    def test_hex8_face_shape_vals_and_grads(self):
        """Test HEX8 face shape function values and gradients."""
        face_shape_vals, face_shape_grads_ref, face_weights, face_normals, face_inds = get_face_shape_vals_and_grads('HEX8')
        
        # Check shapes
        assert face_shape_vals.shape == (6, 4, 8)  # 6 faces, 4 quad points per face, 8 nodes
        assert face_shape_grads_ref.shape == (6, 4, 8, 3)  # 6 faces, 4 quad points, 8 nodes, 3D
        assert face_weights.shape == (6, 4)  # 6 faces, 4 quad points per face
        assert face_normals.shape == (6, 3)  # 6 faces, 3D normals
        assert face_inds.shape == (6, 4)  # 6 faces, 4 vertices per face
        
        # Check that face weights are positive
        assert onp.all(face_weights > 0)
        
        # Check that face normals have unit length
        normal_lengths = onp.linalg.norm(face_normals, axis=1)
        onp.testing.assert_allclose(normal_lengths, onp.ones(6), rtol=1e-10)
    
    def test_tet4_face_shape_vals_and_grads(self):
        """Test TET4 face shape function values and gradients."""
        face_shape_vals, face_shape_grads_ref, face_weights, face_normals, face_inds = get_face_shape_vals_and_grads('TET4')
        
        # Check shapes
        assert face_shape_vals.shape == (4, 1, 4)  # 4 faces, 1 quad point per face, 4 nodes
        assert face_shape_grads_ref.shape == (4, 1, 4, 3)  # 4 faces, 1 quad point, 4 nodes, 3D
        assert face_weights.shape == (4, 1)  # 4 faces, 1 quad point per face
        assert face_normals.shape == (4, 3)  # 4 faces, 3D normals
        assert face_inds.shape == (4, 3)  # 4 faces, 3 vertices per face
        
        # Check that face weights are positive
        assert onp.all(face_weights > 0)
    
    def test_quad4_face_shape_vals_and_grads(self):
        """Test QUAD4 face shape function values and gradients."""
        face_shape_vals, face_shape_grads_ref, face_weights, face_normals, face_inds = get_face_shape_vals_and_grads('QUAD4')
        
        # Check shapes
        assert face_shape_vals.shape == (4, 2, 4)  # 4 faces, 2 quad points per face, 4 nodes
        assert face_shape_grads_ref.shape == (4, 2, 4, 2)  # 4 faces, 2 quad points, 4 nodes, 2D
        assert face_weights.shape == (4, 2)  # 4 faces, 2 quad points per face
        assert face_normals.shape == (4, 2)  # 4 faces, 2D normals
        assert face_inds.shape == (4, 2)  # 4 faces, 2 vertices per face
        
        # Check that face weights are positive
        assert onp.all(face_weights > 0)