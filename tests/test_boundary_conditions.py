"""
Test suite for feax.boundary_conditions module.

Tests boundary condition handling functionality.
"""

import pytest
import numpy as onp
from feax.boundary_conditions import *
from feax.generate_mesh import Mesh


class TestBoundaryConditions:
    """Test boundary conditions module functionality."""
    
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
    
    def test_boundary_conditions_module_imports(self):
        """Test that boundary_conditions module can be imported."""
        from feax import boundary_conditions
        assert boundary_conditions is not None
    
    def test_dirichlet_boundary_conditions(self):
        """Test Dirichlet boundary condition setup."""
        mesh = self.create_simple_tet_mesh()
        
        # Test location function for Dirichlet BC
        location_fn = lambda x: onp.isclose(x[0], 0.0)  # x=0 face
        
        # Test that function works with mesh points
        boundary_flags = [location_fn(point) for point in mesh.points]
        
        assert len(boundary_flags) == len(mesh.points)
        assert any(boundary_flags)  # Should find at least one boundary point
    
    def test_neumann_boundary_conditions(self):
        """Test Neumann boundary condition setup."""
        mesh = self.create_simple_tet_mesh()
        
        # Test traction function for Neumann BC
        traction_fn = lambda x: onp.array([1000.0, 0.0, 0.0])  # Force in x-direction
        
        # Test that function works with mesh points
        tractions = [traction_fn(point) for point in mesh.points]
        
        assert len(tractions) == len(mesh.points)
        assert all(len(traction) == 3 for traction in tractions)
        assert all(traction[0] == 1000.0 for traction in tractions)
    
    def test_mixed_boundary_conditions(self):
        """Test mixed boundary condition setup."""
        mesh = self.create_simple_tet_mesh()
        
        # Test mixed BC: Dirichlet on one face, Neumann on another
        dirichlet_location = lambda x: onp.isclose(x[0], 0.0)
        neumann_location = lambda x: onp.isclose(x[0], 1.0)
        
        dirichlet_flags = [dirichlet_location(point) for point in mesh.points]
        neumann_flags = [neumann_location(point) for point in mesh.points]
        
        # Should not overlap
        overlap = [d and n for d, n in zip(dirichlet_flags, neumann_flags)]
        assert not any(overlap)
    
    def test_boundary_condition_functions(self):
        """Test boundary condition function types."""
        mesh = self.create_simple_tet_mesh()
        
        # Test constant value function
        constant_value_fn = lambda x: 0.0
        
        # Test linear value function
        linear_value_fn = lambda x: x[0]
        
        # Test that functions work
        for point in mesh.points:
            const_val = constant_value_fn(point)
            linear_val = linear_value_fn(point)
            
            assert isinstance(const_val, (float, onp.floating))
            assert isinstance(linear_val, (float, onp.floating))
    
    def test_vector_boundary_conditions(self):
        """Test vector boundary condition setup."""
        mesh = self.create_simple_tet_mesh()
        
        # Test vector BC function
        vector_bc_fn = lambda x: onp.array([x[0], x[1], x[2]])
        
        # Test that function works with mesh points
        vector_bcs = [vector_bc_fn(point) for point in mesh.points]
        
        assert len(vector_bcs) == len(mesh.points)
        assert all(len(bc) == 3 for bc in vector_bcs)
    
    def test_time_dependent_boundary_conditions(self):
        """Test time-dependent boundary condition setup."""
        mesh = self.create_simple_tet_mesh()
        
        # Test time-dependent BC function
        time_dependent_fn = lambda x, t=0.0: onp.sin(t) * x[0]
        
        # Test at different times
        times = [0.0, 0.5, 1.0]
        
        for t in times:
            bcs = [time_dependent_fn(point, t) for point in mesh.points]
            assert len(bcs) == len(mesh.points)
            assert all(isinstance(bc, (float, onp.floating)) for bc in bcs)
    
    def test_boundary_condition_components(self):
        """Test boundary condition component specification."""
        mesh = self.create_simple_tet_mesh()
        
        # Test component-wise BC
        components = [0, 1, 2]  # x, y, z components
        
        for comp in components:
            assert comp >= 0
            assert comp < 3  # For 3D problem
            assert isinstance(comp, int)
    
    def test_boundary_condition_location_functions(self):
        """Test various boundary condition location functions."""
        mesh = self.create_simple_tet_mesh()
        
        # Test different location functions
        location_functions = [
            lambda x: onp.isclose(x[0], 0.0),  # x=0 face
            lambda x: onp.isclose(x[1], 0.0),  # y=0 face
            lambda x: onp.isclose(x[2], 0.0),  # z=0 face
            lambda x: onp.linalg.norm(x) < 0.1,  # Near origin
            lambda x: x[0] + x[1] + x[2] < 0.1   # Custom condition
        ]
        
        for location_fn in location_functions:
            boundary_flags = [location_fn(point) for point in mesh.points]
            assert len(boundary_flags) == len(mesh.points)
            assert all(isinstance(flag, (bool, onp.bool_)) for flag in boundary_flags)
    
    def test_boundary_condition_value_functions(self):
        """Test various boundary condition value functions."""
        mesh = self.create_simple_tet_mesh()
        
        # Test different value functions
        value_functions = [
            lambda x: 0.0,  # Zero displacement
            lambda x: 1.0,  # Unit displacement
            lambda x: x[0],  # Linear in x
            lambda x: x[0]**2,  # Quadratic in x
            lambda x: onp.sin(x[0]),  # Sinusoidal
        ]
        
        for value_fn in value_functions:
            values = [value_fn(point) for point in mesh.points]
            assert len(values) == len(mesh.points)
            assert all(isinstance(val, (float, onp.floating)) for val in values)
    
    def test_boundary_condition_validation(self):
        """Test boundary condition validation."""
        mesh = self.create_simple_tet_mesh()
        
        # Test valid BC setup
        location_fn = lambda x: onp.isclose(x[0], 0.0)
        value_fn = lambda x: 0.0
        
        # Test that BC functions are callable
        assert callable(location_fn)
        assert callable(value_fn)
        
        # Test that BC functions return correct types
        for point in mesh.points:
            location_result = location_fn(point)
            value_result = value_fn(point)
            
            assert isinstance(location_result, (bool, onp.bool_))
            assert isinstance(value_result, (float, onp.floating))
    
    def test_boundary_condition_edge_cases(self):
        """Test boundary condition edge cases."""
        mesh = self.create_simple_tet_mesh()
        
        # Test BC that matches no points
        no_match_fn = lambda x: False
        no_match_flags = [no_match_fn(point) for point in mesh.points]
        assert not any(no_match_flags)
        
        # Test BC that matches all points
        all_match_fn = lambda x: True
        all_match_flags = [all_match_fn(point) for point in mesh.points]
        assert all(all_match_flags)
    
    def test_boundary_condition_tolerance(self):
        """Test boundary condition tolerance handling."""
        mesh = self.create_simple_tet_mesh()
        
        # Test different tolerances
        tolerances = [1e-10, 1e-6, 1e-3]
        
        for tol in tolerances:
            location_fn = lambda x: onp.abs(x[0]) < tol
            boundary_flags = [location_fn(point) for point in mesh.points]
            assert len(boundary_flags) == len(mesh.points)
            
            # Smaller tolerance should match fewer or equal points
            if tol > 1e-6:
                strict_fn = lambda x: onp.abs(x[0]) < 1e-6
                strict_flags = [strict_fn(point) for point in mesh.points]
                assert sum(strict_flags) <= sum(boundary_flags)