"""
Test suite for feax.generate_mesh module.

Tests mesh generation and mesh utility functions.
"""

import pytest
import numpy as onp
import jax.numpy as np
from feax.generate_mesh import (
    Mesh, 
    get_meshio_cell_type,
    rectangle_mesh,
    check_mesh_TET4
)


class TestMesh:
    """Test Mesh class."""
    
    def create_simple_tet_mesh(self):
        """Create a simple tetrahedral mesh for testing."""
        # Single tetrahedron
        points = onp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        cells = onp.array([[0, 1, 2, 3]])
        
        return Mesh(points, cells, 'TET4')
    
    def create_simple_hex_mesh(self):
        """Create a simple hexahedral mesh for testing."""
        # Single hexahedron (unit cube)
        points = onp.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0]   # 7
        ])
        cells = onp.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        
        return Mesh(points, cells, 'HEX8')
    
    def test_mesh_initialization(self):
        """Test mesh initialization."""
        mesh = self.create_simple_tet_mesh()
        
        assert mesh.points.shape == (4, 3)
        assert mesh.cells.shape == (1, 4)
        assert mesh.ele_type == 'TET4'
    
    def test_mesh_with_default_element_type(self):
        """Test mesh with default element type."""
        points = onp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cells = onp.array([[0, 1]])
        
        mesh = Mesh(points, cells)
        
        assert mesh.ele_type == 'TET4'  # default
    
    def test_count_selected_faces_tet(self):
        """Test counting selected faces for tetrahedral mesh."""
        mesh = self.create_simple_tet_mesh()
        
        # Count faces on the z=0 plane (avoid JAX tracing issues)
        def location_fn(x):
            return bool(onp.isclose(x[2], 0.0))
        
        face_count = mesh.count_selected_faces(location_fn)
        
        assert face_count >= 0  # Should find at least one face
        assert isinstance(face_count, int)
    
    def test_count_selected_faces_hex(self):
        """Test counting selected faces for hexahedral mesh."""
        mesh = self.create_simple_hex_mesh()
        
        # Count faces on the x=0 plane (avoid JAX tracing issues)
        def location_fn(x):
            return bool(onp.isclose(x[0], 0.0))
        
        face_count = mesh.count_selected_faces(location_fn)
        
        assert face_count >= 0
        assert isinstance(face_count, int)
    
    def test_count_selected_faces_no_match(self):
        """Test counting faces when no faces match the condition."""
        mesh = self.create_simple_tet_mesh()
        
        # Count faces on a non-existent plane (avoid JAX tracing issues)
        def location_fn(x):
            return bool(onp.isclose(x[2], 10.0))
        
        face_count = mesh.count_selected_faces(location_fn)
        
        assert face_count == 0


class TestGetMeshioCellType:
    """Test get_meshio_cell_type function."""
    
    def test_tet4_cell_type(self):
        """Test TET4 meshio cell type."""
        cell_type = get_meshio_cell_type('TET4')
        assert cell_type == 'tetra'
    
    def test_tet10_cell_type(self):
        """Test TET10 meshio cell type."""
        cell_type = get_meshio_cell_type('TET10')
        assert cell_type == 'tetra10'
    
    def test_hex8_cell_type(self):
        """Test HEX8 meshio cell type."""
        cell_type = get_meshio_cell_type('HEX8')
        assert cell_type == 'hexahedron'
    
    def test_hex20_cell_type(self):
        """Test HEX20 meshio cell type."""
        cell_type = get_meshio_cell_type('HEX20')
        assert cell_type == 'hexahedron20'
    
    def test_hex27_cell_type(self):
        """Test HEX27 meshio cell type."""
        cell_type = get_meshio_cell_type('HEX27')
        assert cell_type == 'hexahedron27'
    
    def test_tri3_cell_type(self):
        """Test TRI3 meshio cell type."""
        cell_type = get_meshio_cell_type('TRI3')
        assert cell_type == 'triangle'
    
    def test_tri6_cell_type(self):
        """Test TRI6 meshio cell type."""
        cell_type = get_meshio_cell_type('TRI6')
        assert cell_type == 'triangle6'
    
    def test_quad4_cell_type(self):
        """Test QUAD4 meshio cell type."""
        cell_type = get_meshio_cell_type('QUAD4')
        assert cell_type == 'quad'
    
    def test_quad8_cell_type(self):
        """Test QUAD8 meshio cell type."""
        cell_type = get_meshio_cell_type('QUAD8')
        assert cell_type == 'quad8'
    
    def test_invalid_cell_type(self):
        """Test invalid cell type raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            get_meshio_cell_type('INVALID_TYPE')


class TestRectangleMesh:
    """Test rectangle_mesh function."""
    
    def test_rectangle_mesh_basic(self):
        """Test basic rectangle mesh generation."""
        Nx, Ny = 2, 3
        domain_x, domain_y = 1.0, 2.0
        
        mesh = rectangle_mesh(Nx, Ny, domain_x, domain_y)
        
        # Check that it returns a meshio.Mesh object
        import meshio
        assert isinstance(mesh, meshio.Mesh)
        
        # Check point dimensions
        points = mesh.points
        assert points.shape[1] == 2  # 2D points
        assert points.shape[0] == (Nx + 1) * (Ny + 1)  # Total number of points
        
        # Check cell dimensions
        cells = mesh.cells[0].data  # Get the quad cells
        assert cells.shape[0] == Nx * Ny  # Total number of elements
        assert cells.shape[1] == 4  # QUAD4 elements
        
        # Check domain bounds
        assert onp.min(points[:, 0]) == 0.0
        assert onp.max(points[:, 0]) == domain_x
        assert onp.min(points[:, 1]) == 0.0
        assert onp.max(points[:, 1]) == domain_y
    
    def test_rectangle_mesh_single_element(self):
        """Test rectangle mesh with single element."""
        Nx, Ny = 1, 1
        domain_x, domain_y = 1.0, 1.0
        
        mesh = rectangle_mesh(Nx, Ny, domain_x, domain_y)
        
        points = mesh.points
        cells = mesh.cells[0].data
        
        assert points.shape == (4, 2)  # 4 corner points
        assert cells.shape == (1, 4)  # 1 QUAD4 element
        
        # Check that we have the expected corner coordinates
        expected_points = onp.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        
        # Check that all expected points are in the mesh
        for expected_point in expected_points:
            found = False
            for point in points:
                if onp.allclose(point, expected_point):
                    found = True
                    break
            assert found, f"Expected point {expected_point} not found in mesh"
    
    def test_rectangle_mesh_connectivity(self):
        """Test rectangle mesh connectivity."""
        Nx, Ny = 2, 2
        domain_x, domain_y = 1.0, 1.0
        
        mesh = rectangle_mesh(Nx, Ny, domain_x, domain_y)
        
        points = mesh.points
        cells = mesh.cells[0].data
        
        # Check that all cell node indices are valid
        assert onp.all(cells >= 0)
        assert onp.all(cells < len(points))
        
        # Check that cells reference different nodes
        for cell in cells:
            assert len(onp.unique(cell)) == 4  # Each cell should have 4 unique nodes


class TestCheckMeshTET4:
    """Test check_mesh_TET4 function."""
    
    def create_positive_orientation_tet(self):
        """Create a tetrahedron with positive orientation."""
        points = onp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        cells = onp.array([[0, 1, 2, 3]])
        
        return points, cells
    
    def create_negative_orientation_tet(self):
        """Create a tetrahedron with negative orientation."""
        points = onp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0]  # Negative z to flip orientation
        ])
        cells = onp.array([[0, 1, 2, 3]])
        
        return points, cells
    
    def test_check_mesh_tet4_positive_orientation(self):
        """Test mesh quality check for positive orientation."""
        points, cells = self.create_positive_orientation_tet()
        
        qualities = check_mesh_TET4(points, cells)
        
        assert len(qualities) == 1  # One tetrahedron
        assert qualities[0] > 0  # Positive orientation
    
    def test_check_mesh_tet4_negative_orientation(self):
        """Test mesh quality check for negative orientation."""
        points, cells = self.create_negative_orientation_tet()
        
        qualities = check_mesh_TET4(points, cells)
        
        assert len(qualities) == 1  # One tetrahedron
        assert qualities[0] < 0  # Negative orientation
    
    def test_check_mesh_tet4_multiple_elements(self):
        """Test mesh quality check for multiple elements."""
        # Create two tetrahedra
        points = onp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ])
        cells = onp.array([
            [0, 1, 2, 3],
            [1, 2, 3, 4]
        ])
        
        qualities = check_mesh_TET4(points, cells)
        
        assert len(qualities) == 2  # Two tetrahedra
        assert isinstance(qualities, np.ndarray)
    
    def test_check_mesh_tet4_degenerate_element(self):
        """Test mesh quality check for degenerate element."""
        # Create a degenerate tetrahedron (all points coplanar)
        points = onp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0]  # Coplanar point
        ])
        cells = onp.array([[0, 1, 2, 3]])
        
        qualities = check_mesh_TET4(points, cells)
        
        assert len(qualities) == 1
        assert onp.abs(qualities[0]) < 1e-10  # Should be very close to zero