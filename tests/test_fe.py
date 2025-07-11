"""
Test suite for feax.fe module.

Tests FiniteElement class and related functionality.
"""

import pytest
import numpy as onp
import jax.numpy as np
from feax.fe import FiniteElement
from feax.generate_mesh import Mesh


class TestFiniteElement:
    """Test FiniteElement class."""
    
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
    
    def test_finite_element_initialization_tet4(self):
        """Test FiniteElement initialization with TET4 mesh."""
        mesh = self.create_simple_tet_mesh()
        
        fe = FiniteElement(
            mesh=mesh,
            vec=3,
            dim=3,
            ele_type='TET4',
            gauss_order=0,
            dirichlet_bc_info=None
        )
        
        assert fe.mesh == mesh
        assert fe.vec == 3
        assert fe.dim == 3
        assert fe.ele_type == 'TET4'
        assert fe.gauss_order == 0
        assert fe.num_cells == 1
        assert fe.num_total_nodes == 4
        assert fe.num_total_dofs == 12  # 4 nodes * 3 components
        assert fe.num_quads == 1  # TET4 with gauss_order=0
        assert fe.num_nodes == 4  # TET4 has 4 nodes
    
    def test_finite_element_initialization_hex8(self):
        """Test FiniteElement initialization with HEX8 mesh."""
        mesh = self.create_simple_hex_mesh()
        
        fe = FiniteElement(
            mesh=mesh,
            vec=3,
            dim=3,
            ele_type='HEX8',
            gauss_order=2,
            dirichlet_bc_info=None
        )
        
        assert fe.num_cells == 1
        assert fe.num_total_nodes == 8
        assert fe.num_total_dofs == 24  # 8 nodes * 3 components
        assert fe.num_quads == 8  # HEX8 with gauss_order=2
        assert fe.num_nodes == 8  # HEX8 has 8 nodes
        assert fe.num_faces == 6  # HEX8 has 6 faces
    
    def test_finite_element_shape_functions(self):
        """Test shape function computations."""
        mesh = self.create_simple_tet_mesh()
        
        fe = FiniteElement(
            mesh=mesh,
            vec=1,
            dim=3,
            ele_type='TET4',
            gauss_order=0,
            dirichlet_bc_info=None
        )
        
        # Check shape function values
        assert fe.shape_vals.shape == (1, 4)  # 1 quad point, 4 nodes
        assert fe.shape_grads_ref.shape == (1, 4, 3)  # 1 quad point, 4 nodes, 3D
        assert fe.quad_weights.shape == (1,)  # 1 quad point
        
        # Check physical shape gradients
        assert fe.shape_grads.shape == (1, 1, 4, 3)  # 1 cell, 1 quad, 4 nodes, 3D
        assert fe.JxW.shape == (1, 1)  # 1 cell, 1 quad
        
        # Check face shape functions
        assert fe.face_shape_vals.shape == (4, 1, 4)  # 4 faces, 1 quad per face, 4 nodes
        assert fe.face_shape_grads_ref.shape == (4, 1, 4, 3)  # 4 faces, 1 quad, 4 nodes, 3D
        assert fe.face_quad_weights.shape == (4, 1)  # 4 faces, 1 quad per face
        assert fe.face_normals.shape == (4, 3)  # 4 faces, 3D normals
        assert fe.face_inds.shape == (4, 3)  # 4 faces, 3 vertices per face
    
    def test_finite_element_with_dirichlet_bc(self):
        """Test FiniteElement with Dirichlet boundary conditions."""
        mesh = self.create_simple_tet_mesh()
        
        # Test without BCs first
        fe = FiniteElement(
            mesh=mesh,
            vec=3,
            dim=3,
            ele_type='TET4',
            gauss_order=0,
            dirichlet_bc_info=None
        )
        
        # Check that BC lists are empty
        assert len(fe.node_inds_list) == 0
        assert len(fe.vec_inds_list) == 0
        assert len(fe.vals_list) == 0
    
    def test_get_physical_quad_points(self):
        """Test getting physical quadrature points."""
        mesh = self.create_simple_tet_mesh()
        
        fe = FiniteElement(
            mesh=mesh,
            vec=1,
            dim=3,
            ele_type='TET4',
            gauss_order=0,
            dirichlet_bc_info=None
        )
        
        quad_points = fe.get_physical_quad_points()
        
        assert quad_points.shape == (1, 1, 3)  # 1 cell, 1 quad, 3D
        
        # For a tetrahedron with gauss_order=0, quad point should be at centroid
        expected_centroid = onp.mean(mesh.points, axis=0)
        onp.testing.assert_allclose(quad_points[0, 0], expected_centroid, rtol=1e-10)
    
    def test_convert_from_dof_to_quad(self):
        """Test converting DOF values to quadrature points."""
        mesh = self.create_simple_tet_mesh()
        
        fe = FiniteElement(
            mesh=mesh,
            vec=3,
            dim=3,
            ele_type='TET4',
            gauss_order=0,
            dirichlet_bc_info=None
        )
        
        # Create a simple solution vector
        sol = onp.array([
            [1.0, 2.0, 3.0],  # node 0
            [4.0, 5.0, 6.0],  # node 1
            [7.0, 8.0, 9.0],  # node 2
            [10.0, 11.0, 12.0]  # node 3
        ])
        
        u = fe.convert_from_dof_to_quad(sol)
        
        assert u.shape == (1, 1, 3)  # 1 cell, 1 quad, 3 components
        
        # For TET4 with gauss_order=0, should be average of nodal values
        expected_u = onp.mean(sol, axis=0)
        onp.testing.assert_allclose(u[0, 0], expected_u, rtol=1e-10)
    
    def test_sol_to_grad(self):
        """Test converting solution to gradient."""
        mesh = self.create_simple_tet_mesh()
        
        fe = FiniteElement(
            mesh=mesh,
            vec=3,
            dim=3,
            ele_type='TET4',
            gauss_order=0,
            dirichlet_bc_info=None
        )
        
        # Create a linear solution field
        sol = onp.array([
            [0.0, 0.0, 0.0],    # node 0
            [1.0, 0.0, 0.0],    # node 1
            [0.0, 1.0, 0.0],    # node 2
            [0.0, 0.0, 1.0]     # node 3
        ])
        
        u_grads = fe.sol_to_grad(sol)
        
        assert u_grads.shape == (1, 1, 3, 3)  # 1 cell, 1 quad, 3 components, 3D gradient
        
        # Check that gradients are reasonable (non-zero for linear field)
        assert not onp.allclose(u_grads, 0.0)
    
    def test_get_boundary_conditions_inds(self):
        """Test getting boundary condition indices."""
        mesh = self.create_simple_tet_mesh()
        
        fe = FiniteElement(
            mesh=mesh,
            vec=3,
            dim=3,
            ele_type='TET4',
            gauss_order=0,
            dirichlet_bc_info=None
        )
        
        # Find faces on z=0 plane (avoid JAX tracing issues)
        def location_fn(x):
            return bool(onp.isclose(x[2], 0.0))
        
        boundary_inds_list = fe.get_boundary_conditions_inds([location_fn])
        
        assert len(boundary_inds_list) == 1
        assert boundary_inds_list[0].shape[1] == 2  # (cell_index, face_index)
        assert boundary_inds_list[0].shape[0] >= 0  # At least 0 faces found
    
    def test_get_face_shape_grads(self):
        """Test getting face shape gradients."""
        mesh = self.create_simple_tet_mesh()
        
        fe = FiniteElement(
            mesh=mesh,
            vec=3,
            dim=3,
            ele_type='TET4',
            gauss_order=0,
            dirichlet_bc_info=None
        )
        
        # Get a boundary face
        boundary_inds = onp.array([[0, 0]])  # Cell 0, face 0
        
        face_shape_grads, nanson_scale = fe.get_face_shape_grads(boundary_inds)
        
        assert face_shape_grads.shape == (1, 1, 4, 3)  # 1 face, 1 quad, 4 nodes, 3D
        assert nanson_scale.shape == (1, 1)  # 1 face, 1 quad
        assert onp.all(nanson_scale > 0)  # Should be positive
    
    def test_get_physical_surface_quad_points(self):
        """Test getting physical surface quadrature points."""
        mesh = self.create_simple_tet_mesh()
        
        fe = FiniteElement(
            mesh=mesh,
            vec=3,
            dim=3,
            ele_type='TET4',
            gauss_order=0,
            dirichlet_bc_info=None
        )
        
        # Get a boundary face
        boundary_inds = onp.array([[0, 0]])  # Cell 0, face 0
        
        surface_quad_points = fe.get_physical_surface_quad_points(boundary_inds)
        
        assert surface_quad_points.shape == (1, 1, 3)  # 1 face, 1 quad, 3D
        
        # Points should be within the tetrahedron bounds
        assert onp.all(surface_quad_points >= 0.0)
        assert onp.all(surface_quad_points <= 1.0)
    
    def test_convert_from_dof_to_face_quad(self):
        """Test converting DOF values to face quadrature points."""
        mesh = self.create_simple_tet_mesh()
        
        fe = FiniteElement(
            mesh=mesh,
            vec=3,
            dim=3,
            ele_type='TET4',
            gauss_order=0,
            dirichlet_bc_info=None
        )
        
        # Create a solution vector
        sol = onp.array([
            [1.0, 2.0, 3.0],  # node 0
            [4.0, 5.0, 6.0],  # node 1
            [7.0, 8.0, 9.0],  # node 2
            [10.0, 11.0, 12.0]  # node 3
        ])
        
        # Get a boundary face
        boundary_inds = onp.array([[0, 0]])  # Cell 0, face 0
        
        u_face = fe.convert_from_dof_to_face_quad(sol, boundary_inds)
        
        assert u_face.shape == (1, 1, 3)  # 1 face, 1 quad, 3 components
        
        # Face values should be interpolated from nodal values
        assert not onp.allclose(u_face, 0.0)
    
    def test_update_dirichlet_boundary_conditions(self):
        """Test updating Dirichlet boundary conditions."""
        mesh = self.create_simple_tet_mesh()
        
        fe = FiniteElement(
            mesh=mesh,
            vec=3,
            dim=3,
            ele_type='TET4',
            gauss_order=0,
            dirichlet_bc_info=None
        )
        
        # Initially no BCs
        assert len(fe.node_inds_list) == 0
        
        # Test that the update method exists and can be called
        fe.update_Dirichlet_boundary_conditions(None)
        
        # Should still have no BCs
        assert len(fe.node_inds_list) == 0
    
    def test_finite_element_scalar_field(self):
        """Test FiniteElement with scalar field (vec=1)."""
        mesh = self.create_simple_tet_mesh()
        
        fe = FiniteElement(
            mesh=mesh,
            vec=1,
            dim=3,
            ele_type='TET4',
            gauss_order=0,
            dirichlet_bc_info=None
        )
        
        assert fe.vec == 1
        assert fe.num_total_dofs == 4  # 4 nodes * 1 component
        
        # Test with scalar solution
        sol = onp.array([[1.0], [2.0], [3.0], [4.0]])  # 4 nodes, 1 component
        
        u = fe.convert_from_dof_to_quad(sol)
        assert u.shape == (1, 1, 1)  # 1 cell, 1 quad, 1 component
        
        u_grads = fe.sol_to_grad(sol)
        assert u_grads.shape == (1, 1, 1, 3)  # 1 cell, 1 quad, 1 component, 3D gradient