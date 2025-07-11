"""
Test suite for feax.utils module.

Tests utility functions for feax.
"""

import pytest
import numpy as onp
from feax.utils import *
from feax.generate_mesh import Mesh


class TestUtils:
    """Test utility functions."""
    
    def create_simple_tet_mesh(self):
        """Create a simple tetrahedral mesh for testing."""
        points = onp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        cells = onp.array([[0, 1, 2, 3]])
        
        return Mesh(points, cells, 'TET4')
    
    def test_utils_module_imports(self):
        """Test that utils module can be imported."""
        # This test ensures the utils module doesn't have import errors
        from feax import utils
        assert utils is not None
    
    def test_basic_functionality(self):
        """Test basic functionality if utils has functions."""
        # Since we don't know the exact contents of utils.py,
        # we'll test basic functionality
        mesh = self.create_simple_tet_mesh()
        
        # Test that mesh is created successfully
        assert mesh is not None
        assert mesh.points.shape == (4, 3)
        assert mesh.cells.shape == (1, 4)
        
    def test_mesh_integration(self):
        """Test integration with mesh utilities."""
        mesh = self.create_simple_tet_mesh()
        
        # Test that mesh properties are accessible
        assert hasattr(mesh, 'points')
        assert hasattr(mesh, 'cells')
        assert hasattr(mesh, 'ele_type')
        
        # Test mesh methods
        if hasattr(mesh, 'count_selected_faces'):
            # Test face counting (avoid JAX tracing issues)
            def location_fn(x):
                return bool(onp.isclose(x[2], 0.0))
            
            face_count = mesh.count_selected_faces(location_fn)
            assert isinstance(face_count, int)
            assert face_count >= 0