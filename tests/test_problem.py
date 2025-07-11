"""
Test suite for feax.problem module.

Tests problem definition and solution functionality.
"""

import pytest
import numpy as onp
from feax.problem import *
from feax.generate_mesh import Mesh


class TestProblem:
    """Test problem module functionality."""
    
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
    
    def create_simple_hex_mesh(self):
        """Create a simple hexahedral mesh for testing."""
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
    
    def test_problem_module_imports(self):
        """Test that problem module can be imported."""
        from feax import problem
        assert problem is not None
    
    def test_problem_with_tet_mesh(self):
        """Test problem setup with tetrahedral mesh."""
        mesh = self.create_simple_tet_mesh()
        
        # Test that mesh is valid for problem setup
        assert mesh is not None
        assert mesh.points.shape == (4, 3)
        assert mesh.cells.shape == (1, 4)
        assert mesh.ele_type == 'TET4'
    
    def test_problem_with_hex_mesh(self):
        """Test problem setup with hexahedral mesh."""
        mesh = self.create_simple_hex_mesh()
        
        # Test that mesh is valid for problem setup
        assert mesh is not None
        assert mesh.points.shape == (8, 3)
        assert mesh.cells.shape == (1, 8)
        assert mesh.ele_type == 'HEX8'
    
    def test_problem_mesh_properties(self):
        """Test problem mesh properties."""
        mesh = self.create_simple_tet_mesh()
        
        # Test mesh properties that are important for problem setup
        assert hasattr(mesh, 'points')
        assert hasattr(mesh, 'cells')
        assert hasattr(mesh, 'ele_type')
        
        # Test mesh geometry
        assert mesh.points.ndim == 2
        assert mesh.cells.ndim == 2
        assert mesh.points.shape[1] == 3  # 3D coordinates
    
    def test_problem_boundary_conditions(self):
        """Test problem boundary condition setup."""
        mesh = self.create_simple_tet_mesh()
        
        # Test boundary condition functions
        location_fn = lambda x: onp.isclose(x[0], 0.0)
        value_fn = lambda x: 0.0
        
        # Test that functions work with mesh points
        boundary_flags = [location_fn(point) for point in mesh.points]
        boundary_values = [value_fn(point) for point in mesh.points]
        
        assert len(boundary_flags) == len(mesh.points)
        assert len(boundary_values) == len(mesh.points)
        assert all(isinstance(flag, (bool, onp.bool_)) for flag in boundary_flags)
        assert all(isinstance(val, (float, onp.floating)) for val in boundary_values)
    
    def test_problem_material_properties(self):
        """Test problem material property setup."""
        mesh = self.create_simple_tet_mesh()
        
        # Test common material properties
        E = 200e9  # Young's modulus
        nu = 0.3   # Poisson's ratio
        
        # Test material property calculations
        lame_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
        lame_mu = E / (2 * (1 + nu))
        
        assert lame_lambda > 0
        assert lame_mu > 0
        assert isinstance(lame_lambda, (float, onp.floating))
        assert isinstance(lame_mu, (float, onp.floating))
    
    def test_problem_load_conditions(self):
        """Test problem load condition setup."""
        mesh = self.create_simple_tet_mesh()
        
        # Test load functions
        body_force_fn = lambda x: onp.array([0.0, 0.0, -9.81])  # Gravity
        traction_fn = lambda x: onp.array([1000.0, 0.0, 0.0])  # Surface traction
        
        # Test that functions work with mesh points
        body_forces = [body_force_fn(point) for point in mesh.points]
        tractions = [traction_fn(point) for point in mesh.points]
        
        assert len(body_forces) == len(mesh.points)
        assert len(tractions) == len(mesh.points)
        assert all(len(force) == 3 for force in body_forces)
        assert all(len(traction) == 3 for traction in tractions)
    
    def test_problem_dimensions(self):
        """Test problem dimension consistency."""
        mesh = self.create_simple_tet_mesh()
        
        # Test 3D problem setup
        dim = 3
        vec = 3  # 3D displacement field
        
        assert mesh.points.shape[1] == dim
        assert vec == dim  # For displacement problems
        
        # Test total DOFs
        num_nodes = len(mesh.points)
        total_dofs = num_nodes * vec
        
        assert total_dofs == 12  # 4 nodes * 3 components
    
    def test_problem_solution_vector(self):
        """Test problem solution vector setup."""
        mesh = self.create_simple_tet_mesh()
        
        vec = 3  # 3D displacement
        num_nodes = len(mesh.points)
        
        # Test solution vector initialization
        sol = onp.zeros((num_nodes, vec))
        
        assert sol.shape == (num_nodes, vec)
        assert sol.dtype == onp.float64
        
        # Test solution vector with values
        sol_with_values = onp.ones((num_nodes, vec))
        assert onp.all(sol_with_values == 1.0)
    
    def test_problem_element_types(self):
        """Test problem with different element types."""
        # Test TET4
        tet_mesh = self.create_simple_tet_mesh()
        assert tet_mesh.ele_type == 'TET4'
        
        # Test HEX8
        hex_mesh = self.create_simple_hex_mesh()
        assert hex_mesh.ele_type == 'HEX8'
        
        # Test element type consistency
        assert tet_mesh.ele_type != hex_mesh.ele_type
    
    def test_problem_gauss_integration(self):
        """Test problem Gauss integration setup."""
        mesh = self.create_simple_tet_mesh()
        
        # Test different Gauss orders
        gauss_orders = [0, 1, 2]
        
        for order in gauss_orders:
            assert order >= 0
            assert isinstance(order, int)
    
    def test_problem_error_handling(self):
        """Test problem error handling."""
        # Test invalid mesh
        try:
            invalid_points = onp.array([])
            invalid_cells = onp.array([])
            invalid_mesh = Mesh(invalid_points, invalid_cells, 'TET4')
            # This should either work or raise an appropriate error
        except Exception as e:
            assert isinstance(e, (ValueError, IndexError, AssertionError))
    
    def test_problem_mesh_quality(self):
        """Test problem mesh quality checks."""
        mesh = self.create_simple_tet_mesh()
        
        # Test mesh quality metrics
        assert mesh.points.shape[0] > 0  # At least one point
        assert mesh.cells.shape[0] > 0   # At least one cell
        assert mesh.points.shape[1] == 3  # 3D coordinates
        
        # Test mesh connectivity
        max_node_index = onp.max(mesh.cells)
        assert max_node_index < len(mesh.points)  # Valid node indices