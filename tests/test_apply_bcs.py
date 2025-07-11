"""
Test suite for feax.apply_bcs module.

Tests boundary condition application and data structures.
"""

import pytest
import numpy as onp
import jax.numpy as np
from jax.experimental.sparse import BCOO

from feax.apply_bcs import (
    DirichletBC, 
    BCInfo, 
    apply_dirichletBC
)


class TestDirichletBC:
    """Test DirichletBC dataclass."""
    
    def test_dirichlet_bc_creation(self):
        """Test DirichletBC creation and post-init conversion."""
        node_indices = [0, 1, 2]
        vec_indices = [0, 1, 2]
        values = [1.0, 2.0, 3.0]
        
        bc = DirichletBC(node_indices, vec_indices, values)
        
        # Check that arrays are converted to JAX arrays
        assert isinstance(bc.node_indices, np.ndarray)
        assert isinstance(bc.vec_indices, np.ndarray)
        assert isinstance(bc.values, np.ndarray)
        
        # Check values
        onp.testing.assert_array_equal(bc.node_indices, np.array([0, 1, 2]))
        onp.testing.assert_array_equal(bc.vec_indices, np.array([0, 1, 2]))
        onp.testing.assert_array_equal(bc.values, np.array([1.0, 2.0, 3.0]))
    
    def test_dirichlet_bc_empty_arrays(self):
        """Test DirichletBC with empty arrays."""
        bc = DirichletBC([], [], [])
        
        assert bc.node_indices.size == 0
        assert bc.vec_indices.size == 0
        assert bc.values.size == 0


class TestBCInfo:
    """Test BCInfo container."""
    
    def test_bcinfo_creation(self):
        """Test BCInfo creation."""
        bc1 = DirichletBC([0, 1], [0, 0], [1.0, 2.0])
        bc2 = DirichletBC([2, 3], [1, 1], [3.0, 4.0])
        
        bc_info = BCInfo(bcs=(bc1, bc2), vec_dim=2, num_nodes=10)
        
        assert len(bc_info.bcs) == 2
        assert bc_info.vec_dim == 2
        assert bc_info.num_nodes == 10
    
    def test_bcinfo_pytree_registration(self):
        """Test that BCInfo is properly registered as a PyTree."""
        bc1 = DirichletBC([0, 1], [0, 0], [1.0, 2.0])
        bc_info = BCInfo(bcs=(bc1,), vec_dim=2, num_nodes=10)
        
        # Test that it can be flattened and unflattened
        from jax.tree_util import tree_flatten, tree_unflatten
        
        children, aux_data = tree_flatten(bc_info)
        reconstructed = tree_unflatten(aux_data, children)
        
        assert isinstance(reconstructed, BCInfo)
        assert reconstructed.vec_dim == bc_info.vec_dim
        assert reconstructed.num_nodes == bc_info.num_nodes
        assert len(reconstructed.bcs) == len(bc_info.bcs)


class TestApplyDirichletBC:
    """Test apply_dirichletBC function."""
    
    def create_simple_system(self):
        """Create a simple test system."""
        # Create a simple 4x4 system
        n = 4
        indices = onp.array([[i, j] for i in range(n) for j in range(n)])
        data = np.ones(n * n)  # Use JAX array
        A = BCOO((data, indices), shape=(n, n))
        b = np.ones(n)  # Use JAX array
        
        return A, b
    
    def test_apply_dirichlet_bc_simple(self):
        """Test applying simple Dirichlet BC."""
        A, b = self.create_simple_system()
        
        # Apply BC: u[0] = 5.0
        bc = DirichletBC([0], [0], [5.0])
        bc_info = BCInfo(bcs=(bc,), vec_dim=1, num_nodes=4)
        
        A_mod, b_mod = apply_dirichletBC(A, b, bc_info)
        
        # Check that the system is properly modified
        assert A_mod.shape == A.shape
        assert b_mod.shape == b.shape
        
        # Check that the BC is enforced
        assert b_mod[0] == 5.0
    
    def test_apply_dirichlet_bc_multiple(self):
        """Test applying multiple Dirichlet BCs."""
        A, b = self.create_simple_system()
        
        # Apply BCs: u[0] = 1.0, u[2] = 3.0
        bc1 = DirichletBC([0], [0], [1.0])
        bc2 = DirichletBC([2], [0], [3.0])
        bc_info = BCInfo(bcs=(bc1, bc2), vec_dim=1, num_nodes=4)
        
        A_mod, b_mod = apply_dirichletBC(A, b, bc_info)
        
        # Check that both BCs are enforced
        assert b_mod[0] == 1.0
        assert b_mod[2] == 3.0
    
    def test_apply_dirichlet_bc_vector_field(self):
        """Test applying BC to vector field."""
        # Create a 6x6 system for 3 nodes with 2 components each
        n = 6
        indices = onp.array([[i, j] for i in range(n) for j in range(n)])
        data = np.ones(n * n)  # Use JAX array
        A = BCOO((data, indices), shape=(n, n))
        b = np.ones(n)  # Use JAX array
        
        # Apply BC: u[0, 0] = 1.0, u[1, 1] = 2.0 (node 0 component 0, node 1 component 1)
        bc = DirichletBC([0, 1], [0, 1], [1.0, 2.0])
        bc_info = BCInfo(bcs=(bc,), vec_dim=2, num_nodes=3)
        
        A_mod, b_mod = apply_dirichletBC(A, b, bc_info)
        
        # Check that BCs are enforced at correct DOFs
        assert b_mod[0] == 1.0  # node 0, component 0
        assert b_mod[3] == 2.0  # node 1, component 1
    
    def test_apply_dirichlet_bc_empty(self):
        """Test applying empty BC."""
        A, b = self.create_simple_system()
        
        bc_info = BCInfo(bcs=(), vec_dim=1, num_nodes=4)
        
        A_mod, b_mod = apply_dirichletBC(A, b, bc_info)
        
        # System should be unchanged
        onp.testing.assert_array_equal(A_mod.data, A.data)
        onp.testing.assert_array_equal(b_mod, b)


class TestBCInfoCreation:
    """Test manual BCInfo creation."""
    
    def test_create_bc_info_manually(self):
        """Test creating BCInfo manually with DirichletBC objects."""
        bc1 = DirichletBC([0, 1], [0, 0], [1.0, 2.0])
        bc2 = DirichletBC([2, 3], [1, 1], [3.0, 4.0])
        
        bc_info = BCInfo(bcs=(bc1, bc2), vec_dim=3, num_nodes=10)
        
        assert isinstance(bc_info, BCInfo)
        assert bc_info.vec_dim == 3
        assert bc_info.num_nodes == 10
        assert len(bc_info.bcs) == 2
    
    def test_create_fixed_bc_manually(self):
        """Test creating fixed boundary condition manually."""
        node_indices = [0, 1, 2]
        
        # Create BCs for all components
        bcs = []
        for component in range(3):
            bc = DirichletBC(
                node_indices=node_indices,
                vec_indices=[component] * len(node_indices),
                values=[0.0] * len(node_indices)
            )
            bcs.append(bc)
        
        bc_info = BCInfo(bcs=tuple(bcs), vec_dim=3, num_nodes=10)
        
        assert isinstance(bc_info, BCInfo)
        assert bc_info.vec_dim == 3
        assert bc_info.num_nodes == 10
        assert len(bc_info.bcs) == 3
        
        # Check that all components are fixed to 0
        for bc in bc_info.bcs:
            onp.testing.assert_array_equal(bc.node_indices, np.array(node_indices))
            onp.testing.assert_array_equal(bc.values, np.zeros(len(node_indices)))